use sp1_core_executor::SP1Context;
use sp1_cuda::api::{Request, Response};
use sp1_gpu_cudart::TaskScope;
use sp1_gpu_prover::cuda_worker_builder_with_machine;
use sp1_primitives::Elf;
use sp1_prover::worker::SP1LocalNodeBuilder;
use sp1_prover::SP1VerifyingKey;
use sp1_prover_types::SerializableRiscvMachine;
use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;

use std::io;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{UnixListener, UnixStream},
};

#[cfg(feature = "rocm")]
use tokio::task::JoinSet;

#[cfg(feature = "rocm")]
use sp1_gpu_cudart::{RocmAllocator, TaskHandle};

#[cfg(feature = "rocm")]
use sp1_prover_types::network_base_types::ProofMode;

#[cfg(feature = "rocm")]
use sp1_prover::worker::SP1LocalNode;

#[cfg(feature = "rocm")]
struct ConnectionTasks(JoinSet<()>);

#[cfg(not(feature = "rocm"))]
struct ConnectionTasks;

#[cfg(feature = "rocm")]
impl ConnectionTasks {
    fn new() -> Self {
        Self(JoinSet::new())
    }

    fn spawn(&mut self, future: impl Future<Output = ()> + Send + 'static) {
        self.0.spawn(future);
    }

    async fn wait_for_one(&mut self) {
        if self.0.is_empty() {
            std::future::pending::<()>().await;
        }
        if let Some(Err(error)) = self.0.join_next().await {
            tracing::error!("Connection task failed: {error}");
        }
    }

    async fn shutdown(&mut self) {
        self.0.shutdown().await;
    }
}

#[cfg(feature = "rocm")]
type PersistentNodeTask = TaskHandle<Result<SP1LocalNode, String>>;

#[cfg(feature = "rocm")]
struct SharedPersistentProver {
    node: tokio::sync::Mutex<Option<PersistentNodeTask>>,
}

#[cfg(not(feature = "rocm"))]
struct SharedPersistentProver;

#[cfg(feature = "rocm")]
impl SharedPersistentProver {
    fn new() -> Self {
        Self { node: tokio::sync::Mutex::new(None) }
    }

    async fn shutdown(&self) {
        let Some(mut task) = self.node.lock().await.take() else {
            return;
        };
        if let Ok(node) = task.take_value() {
            node.shutdown().await;
        }
        drop(task);
    }
}

#[cfg(not(feature = "rocm"))]
impl SharedPersistentProver {
    fn new() -> Self {
        Self
    }

    async fn shutdown(&self) {}
}

#[cfg(not(feature = "rocm"))]
impl ConnectionTasks {
    fn new() -> Self {
        Self
    }

    fn spawn(&mut self, future: impl Future<Output = ()> + Send + 'static) {
        let _ = tokio::spawn(future);
    }

    async fn wait_for_one(&mut self) {
        std::future::pending::<()>().await;
    }

    async fn shutdown(&mut self) {}
}

/// A cached proving key and verifying key.
#[derive(Clone)]
struct CachedProgram {
    elf: Arc<Elf>,
    vk: SP1VerifyingKey,
}

/// The server for the sp1-gpu service.
pub struct Server {
    pub device_id: u32,
    pub socket_path: PathBuf,
    pub backend_name: &'static str,
}

/// The context for a single connection to the server.
struct ConnectionCtx {
    pk_cache: HashMap<[u8; 32], CachedProgram>,
    machine: Option<SerializableRiscvMachine>,
    task_scope: TaskScope,
    persistent: Arc<SharedPersistentProver>,
}

impl Server {
    /// Run the server, indefinitely.
    pub async fn run(self, task_scope: TaskScope) {
        eprintln!(
            "Running {} {} with device {}",
            self.backend_name,
            sp1_primitives::SP1_CRATE_VERSION,
            self.device_id
        );
        let socket_path = self.socket_path;

        // Try to remove the socket file socket incase the file was never cleaned up.
        if let Err(e) = std::fs::remove_file(&socket_path) {
            tracing::warn!("Failed to remove orphaned socket: {}", e);
        }

        let listener = UnixListener::bind(&socket_path).expect("Failed to bind to socket addr");
        let mut connections = ConnectionTasks::new();
        let persistent = Arc::new(SharedPersistentProver::new());

        tracing::info!("Server listening @ {}", socket_path.display());
        loop {
            enum ServerEvent {
                Accept(io::Result<(UnixStream, tokio::net::unix::SocketAddr)>),
                ConnectionCompleted,
                CtrlC,
            }

            let event = tokio::select! {
                result = listener.accept() => ServerEvent::Accept(result),
                _ = connections.wait_for_one() => ServerEvent::ConnectionCompleted,
                _ = tokio::signal::ctrl_c() => ServerEvent::CtrlC,
            };

            match event {
                ServerEvent::Accept(result) => match result {
                    Ok((stream, _)) => {
                        tracing::info!("Connection accepted");

                        let task_scope = task_scope.clone();
                        let persistent = persistent.clone();

                        connections.spawn(async move {
                            let mut stream = stream;

                            if let Err(e) =
                                Self::handle_connection(task_scope, persistent, &mut stream).await
                            {
                                if e.kind() == io::ErrorKind::UnexpectedEof
                                    || e.kind() == io::ErrorKind::BrokenPipe
                                {
                                    tracing::info!("Connection disconnected");
                                    let _ = send_response(&mut stream, Response::ConnectionClosed)
                                        .await;
                                } else {
                                    tracing::error!("Error handling connection: {:?}", e);
                                }
                            }
                        });
                    }
                    Err(error) => {
                        tracing::error!("Failed to accept connection: {error}");
                        break;
                    }
                },
                ServerEvent::ConnectionCompleted => {}
                ServerEvent::CtrlC => {
                    tracing::info!("Ctrl-C received, shutting down");
                    break;
                }
            }
        }

        // A ROCm connection can own an active proof arena. Abort and fully
        // drain those tasks before the process releases cached device slabs.
        // The CUDA implementation keeps its original detached-task behavior.
        connections.shutdown().await;
        persistent.shutdown().await;

        if let Err(e) = std::fs::remove_file(&socket_path) {
            tracing::error!("Failed to remove orphaned socket: {}", e);
        }
    }

    async fn handle_connection(
        task_scope: TaskScope,
        persistent: Arc<SharedPersistentProver>,
        stream: &mut UnixStream,
    ) -> Result<(), io::Error> {
        let mut ctx =
            ConnectionCtx { pk_cache: Default::default(), machine: None, task_scope, persistent };

        loop {
            let mut len = [0_u8; 4];
            stream.read_exact(&mut len).await?;

            let len = u32::from_le_bytes(len);
            let mut request_buf = vec![0; len as usize];
            stream.read_exact(&mut request_buf).await?;

            let request: Request = match bincode::deserialize(&request_buf) {
                Ok(request) => request,
                Err(e) => {
                    eprintln!("Error deserializing request: {e}");
                    let response = Response::InternalError(e.to_string());
                    send_response(stream, response).await?;
                    return Ok(());
                }
            };

            let response = Self::handle_request(&mut ctx, request).await;
            send_response(stream, response).await?;
        }
    }

    #[cfg(feature = "rocm")]
    async fn persistent_node<'a>(
        ctx: &'a ConnectionCtx,
        machine: SerializableRiscvMachine,
    ) -> Result<tokio::sync::MutexGuard<'a, Option<PersistentNodeTask>>, String> {
        let mut node = ctx.persistent.node.lock().await;
        if node.is_none() {
            let task_pool = ctx.task_scope.owner();
            let task = task_pool
                .run(|proof_scope| async move {
                    let builder =
                        cuda_worker_builder_with_machine(proof_scope, machine.into()).await;
                    SP1LocalNodeBuilder::from_worker_client_builder(builder)
                        .build()
                        .await
                        .map_err(|error| error.to_string())
                })
                .await;
            if let Err(error) = task.value() {
                return Err(error.clone());
            }
            *node = Some(task);
        }
        Ok(node)
    }

    async fn handle_request(ctx: &mut ConnectionCtx, request: Request) -> Response {
        match request {
            Request::Setup { elf, machine } => {
                ctx.machine = Some(machine);
                let elf_hash = sha256(&elf);
                if let Some(pk) = ctx.pk_cache.get(&elf_hash) {
                    return Response::Setup { id: elf_hash, vk: pk.vk.clone() };
                }

                tracing::info!("Running setup");
                let setup_elf = elf.clone();
                #[cfg(feature = "rocm")]
                if RocmAllocator::selected() == RocmAllocator::Async {
                    let persistent = match Self::persistent_node(ctx, machine).await {
                        Ok(persistent) => persistent,
                        Err(error) => return Response::InternalError(error),
                    };
                    let prover = persistent
                        .as_ref()
                        .expect("persistent ROCm prover was not initialized")
                        .value()
                        .as_ref()
                        .expect("persistent ROCm prover build failed after validation");
                    let vk = match prover.setup(&setup_elf).await {
                        Ok(vk) => vk,
                        Err(error) => return Response::InternalError(error.to_string()),
                    };
                    drop(persistent);
                    let pk =
                        CachedProgram { elf: Arc::new(Elf::Dynamic(elf.into())), vk: vk.clone() };
                    ctx.pk_cache.insert(elf_hash, pk);
                    return Response::Setup { id: elf_hash, vk };
                }

                let task_pool = ctx.task_scope.owner();
                let handle = task_pool.run_proof(|proof_scope| async move {
                    let builder =
                        cuda_worker_builder_with_machine(proof_scope, machine.into()).await;
                    let prover = SP1LocalNodeBuilder::from_worker_client_builder(builder)
                        .build()
                        .await
                        .map_err(|e| format!("Failed to create prover: {e}"))?;
                    let result = prover.setup(&setup_elf).await.map_err(|e| e.to_string());
                    #[cfg(feature = "rocm")]
                    prover.shutdown().await;
                    result
                });
                let vk = match handle.await.await {
                    Ok(Ok(vk)) => vk,
                    Ok(Err(e)) => return Response::InternalError(e),
                    Err(e) => return Response::InternalError(e.to_string()),
                };
                let pk = CachedProgram { elf: Arc::new(Elf::Dynamic(elf.into())), vk: vk.clone() };
                ctx.pk_cache.insert(elf_hash, pk);
                Response::Setup { id: elf_hash, vk }
            }
            Request::Destroy { key } => {
                tracing::info!("Destroying key");
                ctx.pk_cache.remove(&key);
                Response::Ok
            }
            Request::ProveWithMode { mode, key, stdin, proof_nonce } => {
                tracing::info!("Proving with mode: {mode:?}");
                #[cfg(feature = "rocm")]
                if RocmAllocator::selected() == RocmAllocator::Arena
                    && matches!(mode, ProofMode::Plonk | ProofMode::Groth16)
                {
                    return Response::ProverError(format!(
                        "ROCm arena allocator cannot safely run {mode:?}: one request retains all \
                         recursive sub-stage allocations; restart with SP1_ROCM_ALLOCATOR=async"
                    ));
                }
                let Some(cached) = ctx.pk_cache.get(&key).cloned() else {
                    return Response::InternalError(
                        "Missing proving key, do not drop the prover while maintaing a proving key generated by it.".to_string(),
                    );
                };
                let Some(machine) = ctx.machine else {
                    return Response::InternalError(
                        "Prover not initialized, call Setup first.".to_string(),
                    );
                };
                let context = SP1Context::builder().proof_nonce(proof_nonce).build();

                #[cfg(feature = "rocm")]
                if RocmAllocator::selected() == RocmAllocator::Async {
                    let persistent = match Self::persistent_node(ctx, machine).await {
                        Ok(persistent) => persistent,
                        Err(error) => return Response::ProverError(error),
                    };
                    let prover = persistent
                        .as_ref()
                        .expect("persistent ROCm prover was not initialized")
                        .value()
                        .as_ref()
                        .expect("persistent ROCm prover build failed after validation");
                    let result = prover
                        .prove_with_mode(&cached.elf, stdin, context, mode)
                        .await
                        .map_err(|error| error.to_string());
                    return match result {
                        Ok(proof) => Response::Proof { proof },
                        Err(error) => Response::ProverError(error),
                    };
                }

                let task_pool = ctx.task_scope.owner();
                let handle = task_pool.run_proof(|proof_scope| async move {
                    let builder =
                        cuda_worker_builder_with_machine(proof_scope, machine.into()).await;
                    let prover = SP1LocalNodeBuilder::from_worker_client_builder(builder)
                        .build()
                        .await
                        .map_err(|e| e.to_string())?;
                    let result = prover
                        .prove_with_mode(&cached.elf, stdin, context, mode)
                        .await
                        .map_err(|e| e.to_string());
                    #[cfg(feature = "rocm")]
                    prover.shutdown().await;
                    result
                });
                match handle.await.await {
                    Ok(Ok(proof)) => Response::Proof { proof },
                    Ok(Err(e)) => Response::ProverError(e),
                    Err(e) => Response::ProverError(e.to_string()),
                }
            }
        }
    }
}

fn sha256(data: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

async fn send_response(stream: &mut UnixStream, response: Response) -> Result<(), io::Error> {
    let response_bytes = bincode::serialize(&response).unwrap();
    let len = response_bytes.len() as u32;
    stream.write_all(&len.to_le_bytes()).await?;
    stream.write_all(&response_bytes).await?;

    Ok(())
}
