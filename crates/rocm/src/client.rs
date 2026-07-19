use sp1_core_executor::SP1Context;
use sp1_core_machine::io::SP1Stdin;
use sp1_core_machine::riscv::RiscvAir;
use sp1_hypercube::Machine;
use sp1_primitives::{Elf, SP1Field};
use sp1_prover::worker::ProofFromNetwork;
use sp1_prover_types::network_base_types::ProofMode;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Weak},
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::UnixStream,
    process::Child,
    sync::Mutex,
};

use crate::{
    api::{Request, Response},
    pk::RocmProvingKey,
    RocmClientError,
};

/// The global client to be shared, if other clients still exist (like in a proving key.)
static CLIENT: LazyLock<Mutex<HashMap<u32, Weak<RocmClientInner>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// A client that reads and writes length delimited [`Request`] messages to the server.
#[derive(Clone)]
pub(crate) struct RocmClient {
    /// The stream to the server.
    inner: Arc<RocmClientInner>,
}

impl RocmClient {
    /// Setup a new proving key.
    pub(crate) async fn setup(
        &self,
        elf: Elf,
        machine: Machine<SP1Field, RiscvAir<SP1Field>>,
    ) -> Result<RocmProvingKey, RocmClientError> {
        let request = Request::Setup { elf: elf.as_ref().into(), machine: machine.into() };
        let response = self.send_and_recv(request).await?.into_result()?;
        match response {
            Response::Setup { id, vk } => Ok(RocmProvingKey::new(id, elf, vk, self.clone())),
            _ => Err(RocmClientError::UnexpectedResponse(response.type_of())),
        }
    }

    pub(crate) async fn prove_with_mode(
        &self,
        pk: &RocmProvingKey,
        stdin: SP1Stdin,
        context: SP1Context<'static>,
        mode: ProofMode,
    ) -> Result<ProofFromNetwork, RocmClientError> {
        let key = pk.id();
        let proof_nonce = context.proof_nonce;
        let request = Request::ProveWithMode { mode, key, stdin, proof_nonce };
        let response = self.send_and_recv(request).await?.into_result()?;
        match response {
            Response::Proof { proof } => Ok(proof),
            _ => Err(RocmClientError::UnexpectedResponse(response.type_of())),
        }
    }

    /// Remove a proving key from the server side cache.
    pub(crate) async fn destroy(&self, key: [u8; 32]) -> Result<(), RocmClientError> {
        let request = Request::Destroy { key };
        let response = self.send_and_recv(request).await?.into_result()?;
        match response {
            Response::Ok => Ok(()),
            _ => Err(RocmClientError::UnexpectedResponse(response.type_of())),
        }
    }

    async fn lock(&self) -> tokio::sync::MutexGuard<'_, UnixStream> {
        self.inner.stream.as_ref().expect("expected a valid stream").lock().await
    }
}

impl RocmClient {
    /// Connects to the server at the socket given by [`socket_path`].
    pub(crate) async fn connect(rocm_id: u32) -> Result<Self, RocmClientError> {
        RocmClientInner::connect(rocm_id).await
    }

    /// Sends a request and awaits a response, all while holding the lock on the stream.
    ///
    /// This implementation is requierd to support concurrent connections to the same device.
    pub(crate) async fn send_and_recv(
        &self,
        request: Request,
    ) -> Result<Response, RocmClientError> {
        let mut stream = self.lock().await;
        self.send(&mut stream, request).await?;
        self.recv(&mut stream).await
    }

    /// Sends a [`Request`] message to the server.
    pub(crate) async fn send(
        &self,
        stream: &mut UnixStream,
        request: Request,
    ) -> Result<(), RocmClientError> {
        self.inner.send(stream, request).await
    }

    /// Reads a [`Response`] message from the server.
    pub(crate) async fn recv(&self, stream: &mut UnixStream) -> Result<Response, RocmClientError> {
        self.inner.recv(stream).await
    }
}

struct RocmClientInner {
    stream: Option<Mutex<UnixStream>>,
    _child: Option<Child>,
}

impl RocmClientInner {
    /// Connects to the server at the socket given by [`socket_path`].
    pub(crate) async fn connect(rocm_id: u32) -> Result<RocmClient, RocmClientError> {
        // See if theres a global client still alive.
        // This may be in other instance of the client, or a proving key!
        let mut global = CLIENT.lock().await;

        // If weve already connected to this device, return that client.
        if let Some(client) = global.get(&rocm_id).and_then(|weak| weak.upgrade()) {
            tracing::debug!("Found existing client for ROCm device {}", rocm_id);
            return Ok(RocmClient { inner: client });
        }

        // A server may have been started by another process. Reuse it instead
        // of starting a second server for the same device.
        let socket_path = socket_path(rocm_id);
        let (connection, child) = match Self::connect_once(&socket_path).await {
            Ok(connection) => (connection, None),
            Err(_) => {
                let child = crate::server::start_server(rocm_id).await?;
                let connection = Self::connect_inner(rocm_id).await?;
                (connection, Some(child))
            }
        };
        let inner = RocmClientInner { stream: Some(Mutex::new(connection)), _child: child };

        let inner = Arc::new(inner);
        let _ = global.insert(rocm_id, Arc::downgrade(&inner));

        Ok(RocmClient { inner })
    }

    /// Connects to the server at [`ROCm_SOCKET`], retrying if the server is not running yet.
    async fn connect_inner(rocm_id: u32) -> Result<UnixStream, RocmClientError> {
        let socket_path = socket_path(rocm_id);

        // Retry a few times, just in case the server hasnt started yet.
        // ROCm startup includes device and NTT setup. A second client can also
        // lose the cross-process startup race and must wait for the winner.
        for _ in 0..300 {
            let Ok(this) = Self::connect_once(&socket_path).await else {
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            };

            return Ok(this);
        }

        // If we get here, the server is not running yet.
        // But we want to get the actual error, so try again.
        Self::connect_once(&socket_path).await
    }

    /// Connects to the server at the given path.
    async fn connect_once(path: &Path) -> Result<UnixStream, RocmClientError> {
        let stream = UnixStream::connect(path).await.map_err(|e| {
            RocmClientError::new_connect(e, "Could not connect to `sp1-rocm-server` socket")
        })?;

        Ok(stream)
    }

    /// Sends a [`Request`] message to the server.
    pub(crate) async fn send(
        &self,
        stream: &mut UnixStream,
        request: Request,
    ) -> Result<(), RocmClientError> {
        let request_bytes = bincode::serialize(&request).map_err(RocmClientError::Serialize)?;

        let len_le = (request_bytes.len() as u32).to_le_bytes();
        stream.write_all(&len_le).await.map_err(RocmClientError::Write)?;
        stream.write_all(&request_bytes).await.map_err(RocmClientError::Write)?;

        Ok(())
    }

    /// Reads a [`Response`] message from the server.
    pub(crate) async fn recv(&self, stream: &mut UnixStream) -> Result<Response, RocmClientError> {
        // Read the length of the response.
        let mut len_le = [0; 4];
        stream.read_exact(&mut len_le).await.map_err(RocmClientError::Read)?;

        // Allocate a buffer for the response.
        let len: usize = u32::from_le_bytes(len_le) as usize;
        let mut response_bytes = vec![0; len];
        stream.read_exact(&mut response_bytes).await.map_err(RocmClientError::Read)?;

        let response =
            bincode::deserialize(&response_bytes).map_err(RocmClientError::Deserialize)?;

        Ok(response)
    }
}

/// The socket path for the given ROCm device id.
pub fn socket_path(rocm_id: u32) -> PathBuf {
    const ROCM_SOCKET_BASE: &str = "/tmp/sp1-rocm-";

    format!("{ROCM_SOCKET_BASE}{rocm_id}.sock").into()
}

impl Drop for RocmClientInner {
    fn drop(&mut self) {
        let stream = self.stream.take().expect("stream already taken");

        tokio::spawn(async move {
            let mut stream = stream.lock().await;

            if let Err(e) = stream.shutdown().await {
                tracing::error!("Failed to shutdown the stream: {}", e);
            }
        });
    }
}
