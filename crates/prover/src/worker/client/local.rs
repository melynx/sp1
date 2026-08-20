use std::{collections::BTreeMap, sync::Arc};

use hashbrown::{HashMap, HashSet};
use mti::prelude::{MagicTypeIdExt, V7};
use sp1_prover_types::{
    Artifact, ArtifactClient, ArtifactType, ProofArtifacts, ProofRequestStatus, TaskStatus,
    TaskType,
};
use tokio::sync::{mpsc, watch, RwLock};

use crate::worker::{
    ProofId, RawTaskRequest, SubscriberBuilder, TaskId, TaskMetadata, WorkerClient,
};

struct MessageChannelState {
    tx: mpsc::UnboundedSender<Vec<u8>>,
    rx: Option<mpsc::UnboundedReceiver<Vec<u8>>>,
}

type LocalDb =
    Arc<RwLock<HashMap<TaskId, (watch::Sender<TaskStatus>, watch::Receiver<TaskStatus>)>>>;

type ProofIndex = Arc<RwLock<HashMap<ProofId, HashSet<TaskId>>>>;

pub struct LocalWorkerClientChannels {
    pub task_receivers: BTreeMap<TaskType, mpsc::Receiver<(TaskId, RawTaskRequest)>>,
}

pub struct LocalWorkerClientInner {
    db: LocalDb,
    proof_index: ProofIndex,
    /// Per-proof index of the artifacts each task references, recorded by
    /// `submit_task`. Shared with the node's artifact client, which prunes it on
    /// delete, so it only holds live artifacts; `cleanup` deletes whatever a
    /// completed proof leaked.
    artifact_index: ProofArtifacts,
    input_task_queues: HashMap<TaskType, mpsc::Sender<(TaskId, RawTaskRequest)>>,
    task_channels: RwLock<HashMap<TaskId, MessageChannelState>>,
    /// Number of times each task has been re-queued after a retryable failure.
    retry_counts: RwLock<HashMap<TaskId, u32>>,
}

/// How many times a task that failed with a retryable error is re-queued before
/// the local node fails it for good.
pub const LOCAL_TASK_MAX_RETRIES: u32 = 2;

impl LocalWorkerClientInner {
    fn create_id() -> TaskId {
        TaskId::new("local_worker".create_type_id::<V7>().to_string())
    }

    fn init(artifact_index: ProofArtifacts) -> (Self, LocalWorkerClientChannels) {
        let mut task_outputs = BTreeMap::new();
        let mut task_queues = HashMap::new();
        for task_type in [
            TaskType::UnspecifiedTaskType,
            TaskType::Controller,
            TaskType::ProveShard,
            TaskType::RecursionReduce,
            TaskType::RecursionDeferred,
            TaskType::ShrinkWrap,
            TaskType::SetupVkey,
            TaskType::MarkerDeferredRecord,
            TaskType::PlonkWrap,
            TaskType::Groth16Wrap,
            TaskType::ExecuteOnly,
            TaskType::UtilVkeyMapChunk,
            TaskType::UtilVkeyMapController,
            TaskType::CoreExecute,
        ] {
            let (tx, rx) = mpsc::channel(1);
            task_outputs.insert(task_type, rx);
            task_queues.insert(task_type, tx);
        }

        let db = Arc::new(RwLock::new(HashMap::new()));
        let proof_index = Arc::new(RwLock::new(HashMap::new()));
        let task_channels = RwLock::new(HashMap::new());
        let retry_counts = RwLock::new(HashMap::new());
        let inner = Self {
            db,
            proof_index,
            artifact_index,
            input_task_queues: task_queues,
            task_channels,
            retry_counts,
        };
        (inner, LocalWorkerClientChannels { task_receivers: task_outputs })
    }
}

pub struct LocalWorkerClient {
    inner: Arc<LocalWorkerClientInner>,
}

impl LocalWorkerClient {
    /// Creates a new local worker client.
    #[must_use]
    pub fn init() -> (Self, LocalWorkerClientChannels) {
        Self::init_with_index(ProofArtifacts::default())
    }

    /// Like [`init`](Self::init) but sharing `artifact_index` with the node's
    /// artifact client, so a proof's artifacts are pruned as they are deleted.
    #[must_use]
    pub fn init_with_index(artifact_index: ProofArtifacts) -> (Self, LocalWorkerClientChannels) {
        let (inner, channels) = LocalWorkerClientInner::init(artifact_index);
        (Self { inner: Arc::new(inner) }, channels)
    }

    pub async fn update_task_status(
        &self,
        task_id: TaskId,
        status: TaskStatus,
    ) -> anyhow::Result<()> {
        // Get the sender for this task
        let (status_tx, _) = self
            .inner
            .db
            .read()
            .await
            .get(&task_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("task does not exist"))?;

        status_tx.send(status).map_err(|_| anyhow::anyhow!("failed to send status to task"))?;

        if matches!(
            status,
            TaskStatus::Succeeded | TaskStatus::FailedFatal | TaskStatus::FailedRetryable
        ) {
            self.inner.task_channels.write().await.remove(&task_id);
            self.inner.retry_counts.write().await.remove(&task_id);
        }

        Ok(())
    }

    /// Re-queue a task that failed with a retryable error under its original id.
    ///
    /// The controller waits on the id it was given by `submit_task`, so a retry
    /// must keep that id; submitting the request again would mint a new id that
    /// nothing awaits and leave the proof waiting forever. Returns `Ok(false)`
    /// without re-queuing once the task has been retried
    /// [`LOCAL_TASK_MAX_RETRIES`] times; the caller then fails the task.
    pub async fn retry_task(
        &self,
        kind: TaskType,
        task_id: TaskId,
        task: RawTaskRequest,
    ) -> anyhow::Result<bool> {
        if !self.inner.db.read().await.contains_key(&task_id) {
            anyhow::bail!("task does not exist");
        }
        let attempt = {
            let mut counts = self.inner.retry_counts.write().await;
            let count = counts.entry(task_id.clone()).or_insert(0);
            if *count >= LOCAL_TASK_MAX_RETRIES {
                return Ok(false);
            }
            *count += 1;
            *count
        };
        tracing::warn!(%task_id, ?kind, attempt, "re-queuing task after retryable error");
        self.inner.input_task_queues[&kind]
            .send((task_id, task))
            .await
            .map_err(|e| anyhow::anyhow!("failed to re-queue task of kind {:?}: {e}", kind))?;
        Ok(true)
    }

    /// Delete the artifacts a completed proof leaked - whatever is still tracked
    /// for it (most were already deleted inline during proving and pruned from
    /// the shared index). The worker client shares the index but holds no
    /// artifact-client handle, so the owner - `SP1LocalNode` - passes one in.
    pub async fn cleanup(&self, proof_id: &ProofId, artifact_client: &impl ArtifactClient) {
        let leftovers: Vec<Artifact> = self
            .inner
            .artifact_index
            .take(&proof_id.to_string())
            .into_iter()
            .map(Artifact)
            .collect();
        if !leftovers.is_empty() {
            // One batch delete instead of N per-id calls (one tokio lock
            // acquisition instead of N). Best-effort: if it fails, the proof is
            // already untracked, so any leftover bytes are just unreferenced.
            let _ = artifact_client
                .delete_batch(&leftovers, ArtifactType::UnspecifiedArtifactType)
                .await;
        }
    }
}

impl Clone for LocalWorkerClient {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

impl WorkerClient for LocalWorkerClient {
    async fn submit_task(&self, kind: TaskType, task: RawTaskRequest) -> anyhow::Result<TaskId> {
        tracing::debug!("submitting task of kind {kind:?}");
        let task_id = LocalWorkerClientInner::create_id();
        // Add the task to the proof index.
        self.inner
            .proof_index
            .write()
            .await
            .entry(task.context.proof_id.clone())
            .or_insert_with(HashSet::new)
            .insert(task_id.clone());
        // Record the task's artifacts under this proof. The shared index is
        // pruned as artifacts are deleted, so it holds only live ones; `cleanup`
        // deletes whatever the proof leaves behind.
        let proof = task.context.proof_id.to_string();
        for artifact in task.inputs.iter().chain(task.outputs.iter()) {
            self.inner.artifact_index.track(&proof, &artifact.0);
        }
        // Create a db entry for the task.
        let (tx, rx) = watch::channel(TaskStatus::Pending);
        self.inner.db.write().await.insert(task_id.clone(), (tx, rx));
        // Send the task to the input queue.
        self.inner.input_task_queues[&kind]
            .send((task_id.clone(), task))
            .await
            .map_err(|e| anyhow::anyhow!("failed to send task of kind {:?} to queue: {e}", kind))?;
        Ok(task_id)
    }

    async fn complete_task(
        &self,
        _proof_id: ProofId,
        task_id: TaskId,
        _metadata: TaskMetadata,
    ) -> anyhow::Result<()> {
        self.update_task_status(task_id, TaskStatus::Succeeded).await
    }

    async fn complete_proof(
        &self,
        proof_id: ProofId,
        _task_id: Option<TaskId>,
        _status: ProofRequestStatus,
        _extra_data: impl Into<String> + Send,
    ) -> anyhow::Result<()> {
        // (Artifacts are released separately via `cleanup`, which the node calls
        // with its artifact client.) Remove the proof from the proof index.
        let tasks = self
            .inner
            .proof_index
            .write()
            .await
            .remove(&proof_id)
            .ok_or_else(|| anyhow::anyhow!("proof does not exist for id {proof_id}"))?;
        // Prune the db and any message channels for the proof's tasks. Two
        // short critical sections rather than one holding both writer locks
        // (avoids nesting and keeps each scope's hold time bounded by
        // hashmap removes, not the loop body). `update_task_status` removes
        // a task's channel when it reaches a terminal status; doing it here
        // too covers tasks that never did (e.g. an error mid-proof).
        {
            let mut db = self.inner.db.write().await;
            for task_id in &tasks {
                db.remove(task_id);
            }
        }
        {
            let mut channels = self.inner.task_channels.write().await;
            for task_id in &tasks {
                channels.remove(task_id);
            }
        }
        {
            let mut counts = self.inner.retry_counts.write().await;
            for task_id in &tasks {
                counts.remove(task_id);
            }
        }
        Ok(())
    }

    async fn subscriber(&self, _proof_id: ProofId) -> anyhow::Result<SubscriberBuilder<Self>> {
        let (subscriber_input_tx, mut subscriber_input_rx) = mpsc::unbounded_channel();
        let (subscriber_output_tx, subscriber_output_rx) = mpsc::unbounded_channel();

        tokio::task::spawn({
            let db = self.inner.db.clone();
            let output_tx = subscriber_output_tx.clone();
            async move {
                while let Some(id) = subscriber_input_rx.recv().await {
                    // Spawn a task to send the status to the output channel.
                    let db = db.clone();
                    let output_tx = output_tx.clone();
                    tokio::task::spawn(async move {
                        let (_, mut rx) =
                            db.read().await.get(&id).cloned().expect("task does not exist");
                        rx.mark_changed();
                        while let Ok(()) = rx.changed().await {
                            let value = *rx.borrow();
                            if matches!(
                                value,
                                TaskStatus::FailedFatal
                                    | TaskStatus::FailedRetryable
                                    | TaskStatus::Succeeded
                            ) {
                                output_tx.send((id, value)).ok();
                                return;
                            }
                        }
                    });
                }
            }
        });
        Ok(SubscriberBuilder::new(self.clone(), subscriber_input_tx, subscriber_output_rx))
    }

    async fn subscribe_task_messages(
        &self,
        task_id: &TaskId,
    ) -> anyhow::Result<mpsc::UnboundedReceiver<Vec<u8>>> {
        let mut channels = self.inner.task_channels.write().await;
        if let Some(state) = channels.get_mut(task_id) {
            let rx = state
                .rx
                .take()
                .ok_or_else(|| anyhow::anyhow!("task channel already subscribed for {task_id}"))?;
            return Ok(rx);
        }
        let (tx, rx) = mpsc::unbounded_channel();
        channels.insert(task_id.clone(), MessageChannelState { tx, rx: None });
        Ok(rx)
    }

    async fn send_task_message(&self, task_id: &TaskId, payload: Vec<u8>) -> anyhow::Result<()> {
        let mut channels = self.inner.task_channels.write().await;
        if let Some(state) = channels.get_mut(task_id) {
            state.tx.send(payload).map_err(|_| anyhow::anyhow!("task channel receiver dropped"))?;
        } else {
            let (tx, rx) = mpsc::unbounded_channel();
            tx.send(payload).expect("just-created channel cannot be closed");
            channels.insert(task_id.clone(), MessageChannelState { tx, rx: Some(rx) });
        }
        Ok(())
    }
}

#[cfg(test)]
pub mod test_utils {
    use std::{ops::Range, time::Duration};

    use rand::Rng;

    use super::*;

    pub fn mock_worker_client(
        mut random_interval: HashMap<TaskType, Range<Duration>>,
    ) -> LocalWorkerClient {
        let (worker_client, mut channels) = LocalWorkerClient::init();

        for task_type in [
            TaskType::Controller,
            TaskType::SetupVkey,
            TaskType::ProveShard,
            TaskType::MarkerDeferredRecord,
            TaskType::RecursionReduce,
            TaskType::RecursionDeferred,
            TaskType::ShrinkWrap,
            TaskType::PlonkWrap,
            TaskType::Groth16Wrap,
            TaskType::ExecuteOnly,
            TaskType::CoreExecute,
        ] {
            let mut rx = channels.task_receivers.remove(&task_type).unwrap();
            let interval = random_interval.remove(&task_type).unwrap();
            let worker_client = worker_client.clone();
            tokio::task::spawn(async move {
                while let Some((task_id, request)) = rx.recv().await {
                    let client = worker_client.clone();
                    let interval = interval.clone();
                    tokio::spawn(async move {
                        let duration = {
                            let mut rng = rand::thread_rng();
                            rng.gen_range(interval)
                        };
                        tokio::time::sleep(duration).await;
                        client
                            .complete_task(
                                request.context.proof_id,
                                task_id,
                                TaskMetadata { gpu_ms: None },
                            )
                            .await
                            .unwrap();
                    });
                }
            });
        }

        worker_client
    }
}

#[cfg(test)]
mod tests {
    use sp1_prover_types::{InMemoryArtifactClient, ProofArtifacts};

    use super::*;
    use crate::worker::{RequesterId, TaskContext};

    fn controller_request(proof_id: &ProofId, artifacts: &[&Artifact]) -> RawTaskRequest {
        RawTaskRequest {
            inputs: artifacts.iter().map(|a| (*a).clone()).collect(),
            outputs: vec![],
            context: TaskContext {
                proof_id: proof_id.clone(),
                parent_id: None,
                parent_context: None,
                requester_id: RequesterId::new("test"),
            },
        }
    }

    /// Across many proofs on a reused client, the per-proof task bookkeeping
    /// (`db`, `proof_index`) and the shared artifact index must all return to
    /// empty: artifacts deleted inline are pruned from the index (no tombstones),
    /// and `cleanup` deletes whatever a proof leaked. Regression test for the
    /// per-proof leak in the local node.
    #[tokio::test]
    async fn proof_cleanup_releases_all_state() {
        // Share one artifact index between the two clients, as SP1LocalNode does.
        let index = ProofArtifacts::default();
        let (client, mut channels) = LocalWorkerClient::init_with_index(index.clone());
        let artifacts = InMemoryArtifactClient::with_index(index.clone());
        let controller_rx = channels.task_receivers.get_mut(&TaskType::Controller).unwrap();

        for i in 0..10 {
            let proof_id = ProofId::new(format!("proof-{i}"));
            let inline = artifacts.create_artifact().unwrap();
            let leaked = artifacts.create_artifact().unwrap();
            for a in [&inline, &leaked] {
                artifacts.upload_raw(a, ArtifactType::Program, vec![0u8; 1024]).await.unwrap();
            }

            let task_id = client
                .submit_task(
                    TaskType::Controller,
                    controller_request(&proof_id, &[&inline, &leaked]),
                )
                .await
                .unwrap();
            // Drain the bounded input queue so the next iteration can submit.
            controller_rx.recv().await.unwrap();
            // Send a task message - this lazily creates a `task_channels` entry
            // that `update_task_status` would normally clean up on a terminal
            // status, but won't if the task never reaches one.
            client.send_task_message(&task_id, vec![1, 2, 3]).await.unwrap();

            // Bookkeeping, artifact index, and task channel are all populated.
            assert_eq!(index.len(), 2);
            assert_eq!(client.inner.proof_index.read().await.len(), 1);
            assert_eq!(client.inner.db.read().await.len(), 1);
            assert_eq!(client.inner.task_channels.read().await.len(), 1);

            // Deleting one inline (as the controller does mid-proof) prunes it
            // from the shared index - no tombstone left behind.
            artifacts.delete(&inline, ArtifactType::Program).await.unwrap();
            assert_eq!(index.len(), 1, "index kept a tombstone for a deleted artifact");

            // Completing the proof releases its task bookkeeping; cleanup deletes
            // whatever it leaked.
            client
                .complete_proof(proof_id.clone(), None, ProofRequestStatus::Completed, "")
                .await
                .unwrap();
            client.cleanup(&proof_id, &artifacts).await;

            assert!(index.is_empty(), "artifact index leaked after proof {i}");
            assert!(client.inner.db.read().await.is_empty(), "db leaked after proof {i}");
            assert!(client.inner.proof_index.read().await.is_empty(), "proof_index leaked");
            assert!(
                client.inner.task_channels.read().await.is_empty(),
                "task_channels leaked after proof {i}",
            );
            assert!(
                !artifacts.exists(&leaked, ArtifactType::Program).await.unwrap(),
                "leaked artifact not deleted after proof {i}"
            );
        }
    }

    /// A retryable failure re-queues the task under the id the controller is
    /// waiting on, at most [`LOCAL_TASK_MAX_RETRIES`] times; the retry count is
    /// dropped with the task's other state when the proof completes.
    #[tokio::test]
    async fn retry_task_requeues_same_id_then_gives_up() {
        let (client, mut channels) = LocalWorkerClient::init();
        let rx = channels.task_receivers.get_mut(&TaskType::ProveShard).unwrap();
        let proof_id = ProofId::new("proof-retry");
        let request = controller_request(&proof_id, &[]);

        let task_id = client.submit_task(TaskType::ProveShard, request).await.unwrap();
        let (queued_id, request) = rx.recv().await.unwrap();
        assert_eq!(queued_id, task_id);

        let mut request = Some(request);
        for _ in 0..LOCAL_TASK_MAX_RETRIES {
            let again = request.take().unwrap();
            assert!(client.retry_task(TaskType::ProveShard, task_id.clone(), again).await.unwrap());
            let (queued_id, again) = rx.recv().await.unwrap();
            assert_eq!(queued_id, task_id, "retry must keep the original task id");
            request = Some(again);
        }
        assert!(
            !client
                .retry_task(TaskType::ProveShard, task_id.clone(), request.take().unwrap())
                .await
                .unwrap(),
            "retries must be capped"
        );
        assert!(rx.try_recv().is_err(), "an exhausted retry must not re-queue");

        // An unknown task cannot be retried.
        let bogus = TaskId::new("not-a-task".to_string());
        let bogus_request = controller_request(&proof_id, &[]);
        assert!(client.retry_task(TaskType::ProveShard, bogus, bogus_request).await.is_err());

        client.update_task_status(task_id.clone(), TaskStatus::FailedFatal).await.unwrap();
        assert!(client.inner.retry_counts.read().await.is_empty());
        client.complete_proof(proof_id, None, ProofRequestStatus::Completed, "").await.unwrap();
        assert!(client.inner.db.read().await.is_empty());
    }
}
