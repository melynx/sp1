use std::{
    alloc::Layout,
    ffi::c_void,
    future::{Future, IntoFuture},
    ops::Deref,
    pin::Pin,
    ptr::{self, NonNull},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, OnceLock,
    },
    task::{Context, Poll},
    time::Duration,
};

use pin_project::pin_project;
use slop_alloc::{
    mem::{CopyDirection, CopyError, DeviceMemory},
    AllocError, Allocator, Backend, Buffer, Slice,
};
use slop_futures::queue::{AcquireWorkerError, TryAcquireWorkerError, Worker, WorkerQueue};
use sp1_gpu_sys::runtime::{
    cuda_device_get_default_mem_pool, cuda_mem_pool_set_release_threshold, CudaDevice, CudaMemPool,
    CudaStreamHandle, Dim3, KernelPtr,
};
use thiserror::Error;
use tokio::{sync::oneshot, task::JoinHandle};

use crate::{DeviceCopy, ProofArena, RocmAllocator, TaskSubArena, ToDevice};

use super::{
    stream::{StreamRef, INTERVAL_MS},
    sync::CudaSend,
    CudaError, CudaEvent, CudaStream, IntoDevice, StreamCallbackFuture,
};

const DEFAULT_NUM_TASKS: usize = 64;

static GLOBAL_TASK_POOL: OnceLock<Arc<TaskPool>> = OnceLock::new();

static POOL_ID: AtomicUsize = AtomicUsize::new(0);

pub struct TaskPoolBuilder {
    device: CudaDevice,
    mem_release_threshold: u64,
    capacity: Option<usize>,
}

pub(crate) fn global_task_pool() -> &'static Arc<TaskPool> {
    GLOBAL_TASK_POOL.get_or_init(|| Arc::new(TaskPoolBuilder::new().build().unwrap()))
}

pub struct SpawnHandle<T> {
    handle: Option<JoinHandle<Result<T, CudaError>>>,
}

impl<T> SpawnHandle<T> {
    pub fn abort(&self) {
        self.handle.as_ref().expect("spawn handle already taken").abort();
    }
}

impl<T> Drop for SpawnHandle<T> {
    fn drop(&mut self) {
        if let Some(handle) = &self.handle {
            handle.abort();
        }
    }
}

#[derive(Debug, Error)]
pub enum SpawnError {
    #[error("join handle panicked with error: {0}")]
    JoinError(#[from] tokio::task::JoinError),
    #[error("cuda error: {0}")]
    CudaError(#[from] CudaError),
    #[error("failed to acquire a task from the pool")]
    TaskSpawnError(#[from] TaskSpawnError),
}

pub struct SpawnFuture<T> {
    handle: JoinHandle<Result<T, CudaError>>,
}

impl<T> Future for SpawnFuture<T> {
    type Output = Result<T, SpawnError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.handle).poll(cx) {
            Poll::Ready(Ok(Ok(value))) => Poll::Ready(Ok(value)),
            Poll::Ready(Ok(Err(error))) => Poll::Ready(Err(SpawnError::CudaError(error))),
            Poll::Ready(Err(error)) => Poll::Ready(Err(SpawnError::JoinError(error))),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<T> Drop for SpawnFuture<T> {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

impl<T> IntoFuture for SpawnHandle<T> {
    type Output = Result<T, SpawnError>;

    type IntoFuture = SpawnFuture<T>;

    fn into_future(mut self) -> Self::IntoFuture {
        SpawnFuture { handle: self.handle.take().expect("spawn handle already taken") }
    }
}

pub fn spawn<F, Fut>(f: F) -> SpawnHandle<Fut::Output>
where
    F: FnOnce(TaskScope) -> Fut + Send + 'static,
    Fut: Future + Send + 'static,
    Fut::Output: Send + 'static,
{
    let pool = global_task_pool();
    pool.spawn(f)
}

/// Run a task on the task pool.
///
/// The future returned by this function will wait for the task to finish.
pub async fn run_in_place<F, Fut, R>(f: F) -> TaskHandle<R>
where
    F: FnOnce(TaskScope) -> Fut,
    Fut: Future<Output = R>,
{
    let pool = global_task_pool();
    pool.run(f).await
}

/// Run one proof with its own device-memory arena when selected for ROCm.
pub async fn run_proof_in_place<F, Fut, R>(f: F) -> TaskHandle<R>
where
    F: FnOnce(TaskScope) -> Fut,
    Fut: Future<Output = R>,
{
    global_task_pool().run_proof(f).await
}

/// Run a task on the task pool.
///
/// The future returned by this function will wait for the task to finish.
pub fn run_sync_in_place<F, R>(f: F) -> Result<R, CudaError>
where
    F: FnOnce(TaskScope) -> R,
{
    let pool = global_task_pool();
    pool.run_sync(f)
}

#[derive(Debug, Clone, Error)]
pub enum TaskPoolBuildError {
    #[error("failed to create CUDA stream: {0}")]
    StreamCreationFailed(CudaError),

    #[error("failed to create CUDA event: {0}")]
    EventCreationFailed(CudaError),

    #[error("failed to push task back into pool")]
    PushTaskFailed,
}

#[derive(Debug, Clone, Error)]
pub enum GlobalTaskPoolBuildError {
    #[error("failed to build global task pool")]
    BuildFailed(#[from] TaskPoolBuildError),
    #[error("global task pool already initialized")]
    AlreadyInitialized,
}

impl TaskPoolBuilder {
    pub fn new() -> Self {
        Self { capacity: None, device: CudaDevice(0), mem_release_threshold: u64::MAX }
    }

    pub fn num_tasks(mut self, num_tasks: usize) -> Self {
        self.capacity = Some(num_tasks);
        self
    }

    pub fn device(mut self, device: CudaDevice) -> Self {
        assert!(device.0 == 0, "only device 0 is supported at the moment");
        self.device = device;
        self
    }

    /// Sets the memory release threshold for the associated device.
    ///
    /// # Warning
    /// This setting will affect the memory release threshold for the entire device, not just the
    /// current task pool being built.
    pub fn mem_release_threshold(mut self, threshold: u64) -> Self {
        self.mem_release_threshold = threshold;
        self
    }

    fn allocate_new_id(&self) -> usize {
        let id = POOL_ID.fetch_add(1, Ordering::Relaxed);
        if id > usize::MAX / 2 {
            std::process::abort();
        }
        id
    }

    pub fn build(self) -> Result<TaskPool, TaskPoolBuildError> {
        let id = self.allocate_new_id();
        let num_tasks = self.capacity.unwrap_or(DEFAULT_NUM_TASKS);

        // Set the memory release threshold
        unsafe {
            let mut mem_pool = CudaMemPool(ptr::null_mut());
            CudaError::result_from_ffi(cuda_device_get_default_mem_pool(
                &mut mem_pool,
                self.device,
            ))
            .unwrap();
            CudaError::result_from_ffi(cuda_mem_pool_set_release_threshold(
                mem_pool,
                self.mem_release_threshold,
            ))
            .unwrap();
        };

        let mut tasks = Vec::with_capacity(num_tasks);
        for (i, _) in (0..num_tasks).enumerate() {
            let stream = CudaStream::create().map_err(TaskPoolBuildError::StreamCreationFailed)?;
            let end_event = CudaEvent::create().map_err(TaskPoolBuildError::EventCreationFailed)?;
            tasks.push(Task { owner_id: id, id: i, stream, end_event });
        }
        let inner = Arc::new(WorkerQueue::new(tasks));

        Ok(TaskPool { inner })
    }

    pub fn build_global(self) -> Result<(), GlobalTaskPoolBuildError> {
        let pool = self.build()?;
        GLOBAL_TASK_POOL
            .set(Arc::new(pool))
            .map_err(|_| GlobalTaskPoolBuildError::AlreadyInitialized)
    }
}

impl Default for TaskPoolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct TaskPool {
    inner: Arc<WorkerQueue<Task>>,
}

struct OwnedTask {
    inner: Worker<Task>,
}

struct TaskRunGuard {
    task: Arc<OwnedTask>,
    arena: Option<ProofArena>,
    root_arena: bool,
    armed: bool,
}

impl TaskRunGuard {
    fn new(task: Arc<OwnedTask>, arena: Option<ProofArena>, root_arena: bool) -> Self {
        Self { task, arena, root_arena, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TaskRunGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        if self.root_arena {
            if let Some(arena) = &self.arena {
                arena
                    .cancel_root(|| unsafe { self.task.stream_synchronize() })
                    .expect("cancelled proof failed while waiting for its root stream");
                return;
            }
        }
        unsafe {
            self.task
                .stream_synchronize()
                .expect("cancelled GPU task failed while waiting for its stream");
        }
    }
}

impl std::fmt::Debug for OwnedTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OwnedTask {{ inner: {:?} }}", self.inner.deref())
    }
}

#[derive(Debug, Error)]
#[error("failed to acquire a task from the pool")]
pub enum TaskSpawnError {
    AcquireError(#[from] AcquireWorkerError),
}

#[derive(Debug, Error)]
#[error("failed to acquire a task from the pool")]
pub enum TrySpawnError {
    TryAcquireError(#[from] TryAcquireWorkerError),
}

impl TaskPool {
    async fn task(inner: Arc<WorkerQueue<Task>>) -> Result<OwnedTask, TaskSpawnError> {
        let worker = inner.clone().pop().await.map_err(TaskSpawnError::AcquireError)?;
        Ok(OwnedTask { inner: worker })
    }

    fn try_task(inner: Arc<WorkerQueue<Task>>) -> Result<OwnedTask, TrySpawnError> {
        let worker = inner.clone().try_pop().map_err(TrySpawnError::TryAcquireError)?;
        Ok(OwnedTask { inner: worker })
    }

    /// Spawn a task on the task pool.
    ///
    /// This function will not block the current thread.
    pub fn spawn<F, Fut>(&self, f: F) -> SpawnHandle<Fut::Output>
    where
        F: FnOnce(TaskScope) -> Fut + Send + 'static,
        Fut: Future + Send + 'static,
        Fut::Output: Send + 'static,
    {
        let queue = self.inner.clone();
        let handle = tokio::spawn(async move {
            let task = TaskPool::task(queue).await.expect("failed to acquire a task from the pool");
            task.run(f).await.await
        });
        SpawnHandle { handle: Some(handle) }
    }

    pub fn spawn_blocking<F, R>(&self, f: F) -> SpawnHandle<R>
    where
        F: FnOnce(TaskScope) -> R + Send + 'static,
        R: Send + 'static,
    {
        let queue = self.inner.clone();
        let handle = tokio::task::spawn_blocking(move || {
            let task = TaskPool::try_task(queue).expect("failed to acquire a task from the pool");
            let task = Arc::new(task);
            task.run_sync(f)
        });
        SpawnHandle { handle: Some(handle) }
    }

    /// Run a task on the task pool.
    ///
    /// The future returned by this function will wait for the task to finish.
    pub async fn run<F, Fut, R>(&self, f: F) -> TaskHandle<R>
    where
        F: FnOnce(TaskScope) -> Fut,
        Fut: Future<Output = R>,
    {
        let queue = self.inner.clone();
        let task = TaskPool::task(queue).await.expect("failed to acquire a task from the pool");
        task.run(f).await
    }

    /// Run one proof. ROCm arena allocations share one lease for this call.
    pub async fn run_proof<F, Fut, R>(&self, f: F) -> TaskHandle<R>
    where
        F: FnOnce(TaskScope) -> Fut,
        Fut: Future<Output = R>,
    {
        let task = TaskPool::task(self.inner.clone())
            .await
            .expect("failed to acquire a task from the pool");
        let arena = if cfg!(feature = "rocm") && RocmAllocator::selected() == RocmAllocator::Arena {
            Some(ProofArena::new().expect("failed to create proof arena"))
        } else {
            None
        };
        task.run_with_arena(f, arena, true).await
    }

    pub fn run_sync<F, R>(&self, f: F) -> Result<R, CudaError>
    where
        F: FnOnce(TaskScope) -> R,
    {
        let queue = self.inner.clone();
        let task = TaskPool::try_task(queue).expect("failed to acquire a task from the pool");
        let task = Arc::new(task);
        task.run_sync(f)
    }
}

#[derive(Debug)]
pub struct TaskScope {
    task: Arc<OwnedTask>,
    sub_arena: Option<Arc<TaskSubArena>>,
}

impl Clone for TaskScope {
    fn clone(&self) -> Self {
        Self { task: self.task.clone(), sub_arena: self.sub_arena.clone() }
    }
}

impl Deref for TaskScope {
    type Target = Task;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.task.inner
    }
}

unsafe impl Backend for TaskScope {}

unsafe extern "C" fn sleep(ptr: *mut c_void) {
    let time = unsafe { Box::from_raw(ptr as *mut Duration) };
    std::thread::sleep(*time);
}

unsafe extern "C" fn sync_host(ptr: *mut c_void) {
    let tx = unsafe { Box::from_raw(ptr as *mut oneshot::Sender<bool>) };
    tx.send(true).unwrap();
}

impl TaskScope {
    fn new(task: Arc<OwnedTask>, sub_arena: Option<Arc<TaskSubArena>>) -> Self {
        Self { task, sub_arena }
    }

    pub fn arena_id(&self) -> Option<u64> {
        self.sub_arena.as_ref().map(|arena| arena.arena_id())
    }

    /// Allocates a buffer in this scope on the device.
    ///
    /// This call is not blocking. Upon successful completion, it will return a buffer with a memory
    /// that is guaranteed to be available in the scope of the task but without any absolute
    /// guarantee relative to the host or any other task.
    ///
    /// Other tasks may try to allocate memory concurrently. In order to guarantee enough memory
    /// for all expected work, the user must ensure some limit on task calls by e.g. using a
    /// semaphore.
    #[inline]
    pub fn alloc<T>(&self, capacity: usize) -> Buffer<T, Self> {
        Buffer::with_capacity_in(capacity, self.clone())
    }

    /// Tries to allocate a buffer in this scope on the device.
    #[inline]
    pub fn try_alloc<T>(
        &self,
        capacity: usize,
    ) -> Result<Buffer<T, Self>, slop_alloc::TryReserveError> {
        Buffer::try_with_capacity_in(capacity, self.clone())
    }

    /// Launches a host function in this task.
    ///
    /// # Safety
    ///
    /// The function essentially executes an extern call in `C`. The safety assumption of an extern.  
    /// The user must ensure the pointer is valid and that the data remains valid as this call will
    /// be asynchronous.
    #[inline]
    pub unsafe fn launch_host_fn(
        &self,
        host_fn: unsafe extern "C" fn(*mut c_void),
        data: *mut c_void,
    ) -> Result<(), CudaError> {
        self.launch_host_fn_uncheked(Some(host_fn), data)
    }

    /// Launches a kernel in this task.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - The kernel ptr is valid.
    /// - The arguments are passed correctly across the FFI interface.
    /// - The data lives whitin the scope of the current task.
    pub unsafe fn launch_kernel(
        &self,
        kernel: KernelPtr,
        grid_dim: impl Into<Dim3>,
        block_dim: impl Into<Dim3>,
        args: &[*mut c_void],
        shared_mem: usize,
    ) -> Result<(), CudaError> {
        self.stream().launch_kernel(kernel, grid_dim, block_dim, args, shared_mem)
    }

    /// Sends the CUDA task to sleep for **at least** the given duration.
    ///
    /// This function will not block the calling host thread. The function does a small allocation
    /// so the sleep time might be slightly longer than the given duration for very short times.
    pub fn sleep(&self, time: Duration) {
        let time_ptr = Box::into_raw(Box::new(time));
        unsafe {
            self.launch_host_fn(sleep, time_ptr as *mut c_void).unwrap();
        }
    }

    /// Copies data between slices using CudaMemCpyAsync
    ///
    /// # Safety
    /// The caller must ensure that the data is valid and that the data remains valid as this call
    pub unsafe fn copy<T: DeviceCopy>(
        &self,
        dst: &mut Slice<T, Self>,
        src: &Slice<T, Self>,
    ) -> Result<(), CopyError> {
        dst.copy_from_slice(src, self)
    }

    /// Waits for all work enqueued so far in this task to finish.
    ///
    /// This function can be useful in case there is work to be enqueued but for some reason this
    /// work cannot be done using [Self::launch_host_fn].
    pub async fn synchronize(&self) -> Result<(), CudaError> {
        let (tx, mut rx) = oneshot::channel::<bool>();
        let mut interval = tokio::time::interval(Duration::from_millis(INTERVAL_MS));

        // Launch the host function to signal the main thread that the task is done
        let tx = Box::new(tx);
        let tx_ptr = Box::into_raw(tx);
        unsafe {
            self.launch_host_fn(sync_host, tx_ptr as *mut c_void)?;
        }

        // Wait for the host function to signal the main thread that the task is done while
        // simultaneously polling the stream in the interval to catch any errors.
        loop {
            tokio::select! {
                _ = interval.tick() => {
                     match unsafe { self.stream().query() } {
                        Ok(()) => {break;}
                        Err(CudaError::NotReady) => {}
                        Err(e) => {
                            return Err(e);
                        }

                    }
                }
                _ = &mut rx => {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Joins this task into another task.
    ///
    /// The other task will wait for the current task to finish.
    #[inline]
    unsafe fn join(&self, parent: &TaskScope) -> Result<(), CudaError> {
        parent.stream.wait_unchecked(&self.end_event)
    }

    /// Copies data from the host to the device.
    #[inline]
    pub fn into_device<T: IntoDevice>(&self, data: T) -> Result<T::Output, CopyError> {
        T::into_device_in(data, self)
    }

    #[inline]
    pub fn to_device<T: ToDevice>(&self, data: &T) -> Result<T::Output, CopyError> {
        T::to_device_in(data, self)
    }

    /// Waits for all work enqueued so far in this task to finish.
    ///
    /// This function can be useful in case there is work to be enqueued but for some reason this
    /// work cannot be done using [Self::launch_host_fn].
    #[inline]
    pub fn synchronize_blocking(&self) -> Result<(), CudaError> {
        // The access to the stream is safe and therefore synchronize is safe.
        unsafe { self.stream_synchronize() }
    }

    /// # Safety
    pub unsafe fn handle(&self) -> CudaStreamHandle {
        self.stream.0
    }

    pub fn owner(&self) -> TaskPool {
        TaskPool { inner: self.task.inner.owner().clone() }
    }

    fn owner_queue(&self) -> Arc<WorkerQueue<Task>> {
        self.task.inner.owner().clone()
    }

    /// Spawns a new task from the current task pool.
    ///
    /// The task starting point will have a "happens before" relationship with the current task when
    /// the spawn is called. The handle can be used to wait for the child task to finish.
    pub fn spawn<F, Fut>(&self, f: F) -> SpawnHandle<Fut::Output>
    where
        F: FnOnce(TaskScope) -> Fut + Send + 'static,
        Fut: Future + Send + 'static,
        Fut::Output: CudaSend + 'static,
    {
        let parent = self.clone();
        let handle = tokio::spawn(async move { parent.run_in_place(f).await });
        SpawnHandle { handle: Some(handle) }
    }

    /// Runs a task in place in a new stream.
    ///
    /// Awaiting this task will peform the device calls and synchronize the end of this task to
    /// the parent, but does not do host synchronization.
    pub async fn run_in_place<F, Fut>(&self, f: F) -> Result<Fut::Output, CudaError>
    where
        F: FnOnce(TaskScope) -> Fut,
        Fut: Future,
        Fut::Output: CudaSend,
    {
        let parent = self.clone();
        let task = TaskPool::task(parent.owner_queue()).await.unwrap();
        unsafe {
            // Use the task's end event to synchronize the parent task.
            // This is safe because this is the first time this task is being run so we know
            // there are no other copies that record anything on this event at the same time.
            parent.stream.record_unchecked(&task.inner.end_event)?;
            task.inner.stream.wait_unchecked(&task.inner.end_event)?
        };
        let arena = parent.sub_arena.as_ref().map(|arena| arena.proof_arena());
        let handle = task.run_with_arena(f, arena, false).await;
        handle.join(&parent)
    }
}

impl StreamRef for TaskScope {
    #[inline]
    unsafe fn stream(&self) -> &CudaStream {
        &self.stream
    }
}

#[derive(Debug)]
pub struct Task {
    pub(crate) owner_id: usize,
    pub(crate) id: usize,
    pub(crate) stream: CudaStream,
    end_event: CudaEvent,
}

impl PartialEq for Task {
    fn eq(&self, other: &Self) -> bool {
        self.owner_id == other.owner_id && self.id == other.id
    }
}

impl Eq for Task {}

impl StreamRef for Task {
    #[inline]
    unsafe fn stream(&self) -> &CudaStream {
        &self.stream
    }
}

impl Drop for Task {
    fn drop(&mut self) {
        unsafe {
            self.end_event.query().expect("attempting to drop a task that did not finish");
            self.stream.query().expect("attempting to drop a task that did not finish");
        }
    }
}

impl IntoFuture for Task {
    type Output = Result<(), CudaError>;
    type IntoFuture = StreamCallbackFuture<Self>;

    fn into_future(self) -> Self::IntoFuture {
        StreamCallbackFuture::new(self)
    }
}

unsafe impl Allocator for TaskScope {
    #[inline]
    unsafe fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if cfg!(feature = "rocm") && RocmAllocator::selected() == RocmAllocator::Arena {
            self.sub_arena
                .as_ref()
                .ok_or(AllocError)?
                .allocate(layout)
                .map(|ptr| NonNull::slice_from_raw_parts(ptr, layout.size()))
        } else {
            self.stream.allocate(layout)
        }
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if !(cfg!(feature = "rocm") && RocmAllocator::selected() == RocmAllocator::Arena) {
            // SAFETY: the safety contract must be upheld by the caller
            self.stream.deallocate(ptr, layout)
        }
    }
}

impl DeviceMemory for TaskScope {
    #[inline]
    unsafe fn copy_nonoverlapping(
        &self,
        src: *const u8,
        dst: *mut u8,
        size: usize,
        direction: CopyDirection,
    ) -> Result<(), CopyError> {
        self.stream.copy_nonoverlapping(src, dst, size, direction)
    }

    #[inline]
    unsafe fn write_bytes(&self, dst: *mut u8, value: u8, size: usize) -> Result<(), CopyError> {
        self.stream.write_bytes(dst, value, size)
    }
}

// // Implement CanCopyFrom for TaskScope to copy from CpuBackend
// impl<T: DeviceCopy> CanCopyFrom<Buffer<T>, slop_alloc::CpuBackend> for TaskScope {
//     type Output = Buffer<T, TaskScope>;

//     fn copy_into(
//         &self,
//         value: Buffer<T>,
//     ) -> impl std::future::Future<Output = Result<Self::Output, CopyError>> + Send + Sync {
//         let result = DeviceBuffer::from_host(&value, self).map(|b| b.into_inner());
//         std::future::ready(result)
//     }
// }

// // Implement CanCopyFromRef for TaskScope to copy Point from CpuBackend
// impl<T: DeviceCopy> CanCopyFromRef<Point<T>, slop_alloc::CpuBackend> for TaskScope {
//     type Output = Point<T, TaskScope>;

//     fn copy_to(
//         &self,
//         value: &Point<T>,
//     ) -> impl std::future::Future<Output = Result<Self::Output, CopyError>> + Send + Sync {
//         let result =
//             DeviceBuffer::from_host(value.values(), self).map(|b| Point::new(b.into_inner()));
//         std::future::ready(result)
//     }
// }

// // Implement CanCopyIntoRef for TaskScope to copy Point to CpuBackend
// impl<T: DeviceCopy> CanCopyIntoRef<Point<T, TaskScope>, slop_alloc::CpuBackend> for TaskScope {
//     type Output = Point<T>;

//     fn copy_to_dst(
//         dst: &slop_alloc::CpuBackend,
//         value: &Point<T, TaskScope>,
//     ) -> impl std::future::Future<Output = Result<Self::Output, CopyError>> + Send + Sync {
//         let _ = dst;
//         let result =
//             DeviceBuffer::from_raw(value.values().clone()).to_host().map(|v| Point::new(v.into()));
//         std::future::ready(result)
//     }
// }

impl OwnedTask {
    fn is_finished(&self) -> Result<bool, CudaError> {
        self.inner.end_event.query().map(|()| true).or_else(|e| match e {
            CudaError::NotReady => Ok(false),
            e => Err(e),
        })
    }

    async fn run<F, Fut, R>(self, f: F) -> TaskHandle<R>
    where
        F: FnOnce(TaskScope) -> Fut,
        Fut: Future<Output = R>,
    {
        self.run_with_arena(f, None, false).await
    }

    async fn run_with_arena<F, Fut, R>(
        self,
        f: F,
        inherited_arena: Option<ProofArena>,
        root_arena: bool,
    ) -> TaskHandle<R>
    where
        F: FnOnce(TaskScope) -> Fut,
        Fut: Future<Output = R>,
    {
        let strong_ptr = Arc::new(self);
        let proof_arena = inherited_arena;
        let sub_arena = proof_arena.clone().map(|arena| Arc::new(TaskSubArena::new(arena)));
        let scope = TaskScope::new(strong_ptr.clone(), sub_arena);
        let mut run_guard = TaskRunGuard::new(strong_ptr.clone(), proof_arena.clone(), root_arena);
        let value = f(scope.clone()).await;
        unsafe { scope.stream.record_unchecked(&scope.end_event).unwrap() };
        if root_arena {
            if let Some(arena) = &proof_arena {
                arena
                    .close_root(|| {
                        let completion = CudaEvent::create()?;
                        unsafe { scope.stream.record_unchecked(&completion)? };
                        Ok(completion)
                    })
                    .expect("failed to record proof completion event");
            }
        }
        run_guard.disarm();
        TaskHandle { value: Some(value), scope: Some(scope), task: Some(strong_ptr) }
    }

    fn run_sync<F, R>(self: Arc<Self>, f: F) -> Result<R, CudaError>
    where
        F: FnOnce(TaskScope) -> R,
    {
        let proof_arena =
            if cfg!(feature = "rocm") && RocmAllocator::selected() == RocmAllocator::Arena {
                Some(ProofArena::new()?)
            } else {
                None
            };
        let sub_arena = proof_arena.clone().map(|arena| Arc::new(TaskSubArena::new(arena)));
        let scope = TaskScope::new(self.clone(), sub_arena);
        let mut run_guard = TaskRunGuard::new(self.clone(), proof_arena.clone(), true);
        let output = f(scope.clone());
        unsafe {
            scope.stream.record_unchecked(&scope.end_event)?;
            scope.end_event.synchronize()?;
        };
        if let Some(arena) = proof_arena {
            arena.close_root(|| {
                let completion = CudaEvent::create()?;
                unsafe { scope.stream.record_unchecked(&completion)? };
                Ok(completion)
            })?;
        }
        run_guard.disarm();
        Ok(output)
    }
}

impl StreamRef for OwnedTask {
    #[inline]
    unsafe fn stream(&self) -> &CudaStream {
        self.inner.stream()
    }
}

impl IntoFuture for TaskScope {
    type Output = Result<(), CudaError>;
    type IntoFuture = StreamCallbackFuture<Self>;

    fn into_future(self) -> Self::IntoFuture {
        StreamCallbackFuture::new(self)
    }
}

pub struct TaskHandle<T> {
    value: Option<T>,
    scope: Option<TaskScope>,
    task: Option<Arc<OwnedTask>>,
}

impl<T> TaskHandle<T> {
    /// Borrows the task output while retaining ownership of its stream worker.
    pub fn value(&self) -> &T {
        self.value.as_ref().expect("task value already taken")
    }

    /// Takes the task output while retaining ownership of its stream worker.
    pub fn take_value(&mut self) -> T {
        self.value.take().expect("task value already taken")
    }

    pub fn join(mut self, parent: &TaskScope) -> Result<T, CudaError>
    where
        T: CudaSend,
    {
        // See [TaskHandle::join] for the explanation of safety. Here this is a bit more complex,
        // but the eventual panic still applies. This is enough in most cases.
        unsafe {
            let scope = self.scope.take().expect("task scope already taken");
            if let Some(arena) = scope.sub_arena.as_ref().map(|arena| arena.proof_arena()) {
                arena.join_child(
                    || scope.join(parent),
                    || {
                        let completion = CudaEvent::create()?;
                        scope.stream.record_unchecked(&completion)?;
                        Ok(completion)
                    },
                )?;
            } else {
                scope.join(parent)?;
            }
            let value = self.value.take().expect("task value already taken").send_to_scope(parent);
            drop(self.task.take());
            // Return the value to the caller.
            Ok(value)
        }
    }

    pub fn is_finished(&self) -> Result<bool, CudaError> {
        self.task.as_ref().expect("task already taken").is_finished()
    }
}

impl<T> Drop for TaskHandle<T> {
    fn drop(&mut self) {
        // Values can own device buffers whose deallocation is queued on this task's stream. Drop
        // them before synchronizing and releasing the worker.
        drop(self.value.take());
        drop(self.scope.take());
        if let Some(task) = &self.task {
            unsafe {
                task.stream_synchronize()
                    .expect("dropped GPU task failed while waiting for its stream");
            }
        }
    }
}

#[pin_project]
pub struct StreamHandleFuture<T> {
    value: Option<T>,
    scope: Option<TaskScope>,
    #[pin]
    callback: StreamCallbackFuture<Arc<OwnedTask>>,
}

impl<T> Future for StreamHandleFuture<T> {
    type Output = Result<T, CudaError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        match this.callback.poll(cx) {
            Poll::Ready(result) => {
                // The callback has released its copy of the task. Release the scope's copy now so
                // the worker returns to the pool before the surrounding async state is dropped.
                this.scope.take();
                Poll::Ready(
                    result
                        .map(|_| this.value.take().expect("task future completed more than once")),
                )
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<T> IntoFuture for TaskHandle<T> {
    type Output = Result<T, CudaError>;
    type IntoFuture = StreamHandleFuture<T>;

    #[inline]
    fn into_future(mut self) -> Self::IntoFuture {
        StreamHandleFuture {
            value: self.value.take(),
            scope: self.scope.take(),
            callback: StreamCallbackFuture::new(self.task.take().expect("task already taken")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DEFAULT_NUM_TASKS;
    use crate::{args, sync::CudaSend, DeviceBuffer, TaskPoolBuilder};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use serial_test::serial;
    use slop_algebra::{AbstractExtensionField, AbstractField};
    use slop_alloc::mem::DeviceMemory;
    use sp1_primitives::{SP1ExtensionField, SP1Field};
    use std::time::Duration;

    #[tokio::test]
    #[serial]
    async fn test_global_task_pool() {
        crate::spawn(|_| async {}).await.unwrap();
    }

    #[tokio::test]
    #[serial]
    async fn test_local_pool() {
        let num_workers = 10;
        let num_callers = 100;
        let pool = TaskPoolBuilder::new().num_tasks(num_workers).build().unwrap();

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut handles = Vec::new();
        for _ in 0..num_callers {
            let pool = pool.clone();
            let tx = tx.clone();
            let handle = pool.spawn(|_| async move {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                tx.send(true).unwrap();
            });

            handles.push(handle);
        }
        drop(tx);

        let mut count = 0;
        while let Some(flag) = rx.recv().await {
            assert!(flag);
            count += 1;
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(count, num_callers);
    }

    #[cfg(feature = "rocm")]
    #[tokio::test]
    #[serial]
    async fn test_rocm_task_pool_reuses_idle_stream_workers() {
        let pool = TaskPoolBuilder::new().num_tasks(2).build().unwrap();
        for iteration in 0..128 {
            tokio::time::timeout(Duration::from_secs(5), async {
                pool.run_proof(|_| async {}).await.await.unwrap();
            })
            .await
            .unwrap_or_else(|_| panic!("worker was not returned after iteration {iteration}"));
        }
    }

    #[test]
    #[serial]
    fn test_partial_block_reduce_wave_counts() {
        crate::run_sync_in_place(|scope| {
            for waves in [1usize, 2, 3, 5, 7, 9] {
                let threads = waves * 32;
                let input = vec![SP1Field::one(); threads];
                let input = DeviceBuffer::from_host_slice(&input, &scope).unwrap();
                let mut output =
                    DeviceBuffer::from_host_slice(&[SP1Field::zero()], &scope).unwrap();
                let len = threads as u32;
                unsafe {
                    let args = args!(input.as_ptr(), output.as_mut_ptr(), len);
                    scope
                        .launch_kernel(
                            sp1_gpu_sys::kernels::partial_block_reduce_test_kernel_felt(),
                            (1u32, 1u32, 1u32),
                            threads,
                            &args,
                            waves * std::mem::size_of::<SP1Field>(),
                        )
                        .unwrap();
                }
                scope.synchronize_blocking().unwrap();
                assert_eq!(
                    output.to_host().unwrap()[0],
                    SP1Field::from_canonical_usize(threads),
                    "failed with {waves} waves"
                );

                let input = vec![SP1ExtensionField::one(); threads];
                let input = DeviceBuffer::from_host_slice(&input, &scope).unwrap();
                let mut output =
                    DeviceBuffer::from_host_slice(&[SP1ExtensionField::zero()], &scope).unwrap();
                unsafe {
                    let args = args!(input.as_ptr(), output.as_mut_ptr(), len);
                    scope
                        .launch_kernel(
                            sp1_gpu_sys::kernels::partial_block_reduce_test_kernel_ext(),
                            (1u32, 1u32, 1u32),
                            threads,
                            &args,
                            waves * std::mem::size_of::<SP1ExtensionField>(),
                        )
                        .unwrap();
                }
                scope.synchronize_blocking().unwrap();
                assert_eq!(
                    output.to_host().unwrap()[0],
                    SP1ExtensionField::from_canonical_usize(threads),
                    "extension reduction failed with {waves} waves"
                );
            }
        })
        .unwrap();
    }

    #[test]
    #[serial]
    fn test_koala_bear_extension_multiplication() {
        const MODULUS: u32 = 0x7f00_0001;
        let mut rng = StdRng::seed_from_u64(0x524f_434d);
        let mut left = Vec::with_capacity(4099);
        let mut right = Vec::with_capacity(4099);

        for _ in 0..4096 {
            left.push(SP1ExtensionField::from_base_fn(|_| {
                SP1Field::from_canonical_u32(rng.gen_range(0..MODULUS))
            }));
            right.push(SP1ExtensionField::from_base_fn(|_| {
                SP1Field::from_canonical_u32(rng.gen_range(0..MODULUS))
            }));
        }
        left.extend([
            SP1ExtensionField::zero(),
            SP1ExtensionField::one(),
            SP1ExtensionField::from_base_fn(|_| SP1Field::from_canonical_u32(MODULUS - 1)),
        ]);
        right.extend([
            SP1ExtensionField::from_base_fn(|_| SP1Field::from_canonical_u32(MODULUS - 1)),
            SP1ExtensionField::one(),
            SP1ExtensionField::from_base_fn(|i| {
                SP1Field::from_canonical_u32(MODULUS - 1 - i as u32)
            }),
        ]);
        let expected = left.iter().zip(&right).map(|(&a, &b)| a * b).collect::<Vec<_>>();

        crate::run_sync_in_place(|scope| {
            let left = DeviceBuffer::from_host_slice(&left, &scope).unwrap();
            let right = DeviceBuffer::from_host_slice(&right, &scope).unwrap();
            let mut output = DeviceBuffer::from_host_slice(
                &vec![SP1ExtensionField::zero(); expected.len()],
                &scope,
            )
            .unwrap();
            let len = expected.len();
            unsafe {
                let args = args!(left.as_ptr(), right.as_ptr(), output.as_mut_ptr(), len);
                scope
                    .launch_kernel(
                        sp1_gpu_sys::kernels::mul_koala_bear_ext_kernel(),
                        len.div_ceil(256),
                        256,
                        &args,
                        0,
                    )
                    .unwrap();
            }
            scope.synchronize_blocking().unwrap();
            assert_eq!(output.to_host().unwrap(), expected);
        })
        .unwrap();
    }

    #[test]
    #[serial]
    fn test_arena_preserves_device_allocation_alignment() {
        crate::run_sync_in_place(|scope| {
            let buffers = (1..=64)
                .map(|size| DeviceBuffer::<u8>::with_capacity_in(size, scope.clone()))
                .collect::<Vec<_>>();
            for buffer in buffers {
                assert_eq!(buffer.as_ptr() as usize % 256, 0);
            }
        })
        .unwrap();
    }

    #[cfg(feature = "rocm")]
    async fn assert_allocator_reuse_pattern(pattern: u8) {
        let handle = crate::run_proof_in_place(|scope| async move {
            let size = 1024 * 1024;
            let mut buffer = DeviceBuffer::<u8>::with_capacity_in(size, scope.clone());
            unsafe {
                scope.write_bytes(buffer.as_mut_ptr(), pattern, size).unwrap();
                buffer.set_len(size);
            }
            scope.synchronize_blocking().unwrap();
            buffer.to_host().unwrap()
        })
        .await;
        let host = handle.await.unwrap();
        assert!(host.iter().all(|value| *value == pattern));
    }

    #[cfg(feature = "rocm")]
    #[tokio::test]
    #[serial]
    async fn test_rocm_allocator_root_cancellation_waits_for_stream() {
        crate::run_proof_in_place(|_| async {}).await.await.unwrap();

        let start = std::time::Instant::now();
        let result = tokio::time::timeout(
            Duration::from_millis(20),
            crate::run_proof_in_place(|scope| async move {
                let size = 1024 * 1024;
                let mut buffer = DeviceBuffer::<u8>::with_capacity_in(size, scope.clone());
                unsafe {
                    scope.write_bytes(buffer.as_mut_ptr(), 0x5a, size).unwrap();
                    buffer.set_len(size);
                }
                scope.sleep(Duration::from_millis(200));
                std::future::pending::<()>().await;
                drop(buffer);
            }),
        )
        .await;
        assert!(result.is_err());
        assert!(start.elapsed() >= Duration::from_millis(150));
        assert_allocator_reuse_pattern(0xa5).await;
    }

    #[cfg(feature = "rocm")]
    #[tokio::test]
    #[serial]
    async fn test_rocm_allocator_child_cancellation_waits_for_stream() {
        crate::run_proof_in_place(|_| async {}).await.await.unwrap();

        let start = std::time::Instant::now();
        let result = tokio::time::timeout(
            Duration::from_millis(20),
            crate::run_proof_in_place(|scope| async move {
                scope
                    .run_in_place(|child| async move {
                        let size = 1024 * 1024;
                        let mut buffer = DeviceBuffer::<u8>::with_capacity_in(size, child.clone());
                        unsafe {
                            child.write_bytes(buffer.as_mut_ptr(), 0x3c, size).unwrap();
                            buffer.set_len(size);
                        }
                        child.sleep(Duration::from_millis(200));
                        std::future::pending::<()>().await;
                        drop(buffer);
                    })
                    .await
                    .unwrap();
            }),
        )
        .await;
        assert!(result.is_err());
        assert!(start.elapsed() >= Duration::from_millis(150));
        assert_allocator_reuse_pattern(0xc3).await;
    }

    #[cfg(feature = "rocm")]
    #[tokio::test]
    #[serial]
    async fn test_rocm_allocator_dropped_spawn_is_cancelled() {
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let handle = crate::run_proof_in_place(|scope| async move {
            let child = scope.spawn(|child| async move {
                let size = 1024 * 1024;
                let mut buffer = DeviceBuffer::<u8>::with_capacity_in(size, child.clone());
                unsafe {
                    child.write_bytes(buffer.as_mut_ptr(), 0x69, size).unwrap();
                    buffer.set_len(size);
                }
                child.sleep(Duration::from_millis(200));
                started_tx.send(()).unwrap();
                std::future::pending::<()>().await;
                drop(buffer);
            });
            started_rx.await.unwrap();
            drop(child);
        })
        .await;
        handle.await.unwrap();
        assert_allocator_reuse_pattern(0x96).await;
    }

    #[cfg(feature = "rocm")]
    #[tokio::test]
    #[serial]
    async fn test_rocm_allocator_dropped_spawn_future_is_cancelled() {
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let handle = crate::run_proof_in_place(|scope| async move {
            let child = scope.spawn(|child| async move {
                let size = 1024 * 1024;
                let mut buffer = DeviceBuffer::<u8>::with_capacity_in(size, child.clone());
                unsafe {
                    child.write_bytes(buffer.as_mut_ptr(), 0x6c, size).unwrap();
                    buffer.set_len(size);
                }
                child.sleep(Duration::from_millis(200));
                started_tx.send(()).unwrap();
                std::future::pending::<()>().await;
                drop(buffer);
            });
            started_rx.await.unwrap();
            assert!(tokio::time::timeout(Duration::from_millis(20), child).await.is_err());
        })
        .await;
        handle.await.unwrap();
        assert_allocator_reuse_pattern(0xc6).await;
    }

    #[cfg(feature = "rocm")]
    #[tokio::test]
    #[serial]
    async fn test_rocm_allocator_dropped_task_handle_waits_for_stream() {
        let handle = crate::run_proof_in_place(|scope| async move {
            let size = 1024 * 1024;
            let mut buffer = DeviceBuffer::<u8>::with_capacity_in(size, scope.clone());
            unsafe {
                scope.write_bytes(buffer.as_mut_ptr(), 0x78, size).unwrap();
                buffer.set_len(size);
            }
            scope.sleep(Duration::from_millis(200));
            buffer
        })
        .await;
        let start = std::time::Instant::now();
        drop(handle);
        assert!(start.elapsed() >= Duration::from_millis(150));
        assert_allocator_reuse_pattern(0x87).await;
    }

    #[cfg(feature = "rocm")]
    #[tokio::test]
    #[serial]
    async fn test_rocm_allocator_dropped_task_future_is_safe() {
        let handle = crate::run_proof_in_place(|scope| async move {
            let size = 1024 * 1024;
            let mut buffer = DeviceBuffer::<u8>::with_capacity_in(size, scope.clone());
            unsafe {
                scope.write_bytes(buffer.as_mut_ptr(), 0x4b, size).unwrap();
                buffer.set_len(size);
            }
            scope.sleep(Duration::from_millis(200));
            buffer
        })
        .await;

        let start = std::time::Instant::now();
        let result = tokio::time::timeout(Duration::from_millis(20), handle).await;
        assert!(result.is_err());
        if crate::RocmAllocator::selected() == crate::RocmAllocator::Arena {
            assert!(start.elapsed() >= Duration::from_millis(150));
        }
        assert_allocator_reuse_pattern(0xb4).await;
    }

    #[cfg(feature = "rocm")]
    #[tokio::test]
    #[serial]
    async fn test_rocm_allocator_error_path_reuses_safely() {
        let handle = crate::run_proof_in_place(|scope| async move {
            let size = 1024 * 1024;
            let mut buffer = DeviceBuffer::<u8>::with_capacity_in(size, scope.clone());
            unsafe {
                scope.write_bytes(buffer.as_mut_ptr(), 0x2d, size).unwrap();
                buffer.set_len(size);
            }
            scope.sleep(Duration::from_millis(100));
            Err::<(), &'static str>("expected prover error")
        })
        .await;
        assert_eq!(handle.await.unwrap(), Err("expected prover error"));
        assert_allocator_reuse_pattern(0xd2).await;
    }

    #[cfg(feature = "rocm")]
    #[tokio::test]
    #[serial]
    async fn test_rocm_allocator_rejects_cross_arena_transfer() {
        if crate::RocmAllocator::selected() != crate::RocmAllocator::Arena {
            return;
        }

        let first = crate::run_proof_in_place(|scope| async move {
            DeviceBuffer::<u8>::with_capacity_in(1024, scope)
        })
        .await
        .await
        .unwrap();

        let task = tokio::spawn(async move {
            let _ =
                crate::run_proof_in_place(
                    |scope| async move { unsafe { first.send_to_scope(&scope) } },
                )
                .await;
        });
        assert!(task.await.unwrap_err().is_panic());
        assert_allocator_reuse_pattern(0x1e).await;
    }

    #[cfg(feature = "rocm")]
    #[test]
    #[serial]
    fn test_rocm_allocator_sync_panic_waits_for_stream() {
        let start = std::time::Instant::now();
        let result = std::panic::catch_unwind(|| {
            let _ = crate::run_sync_in_place(|scope| {
                scope.sleep(Duration::from_millis(200));
                panic!("expected test panic");
            });
        });
        assert!(result.is_err());
        assert!(start.elapsed() >= Duration::from_millis(150));
    }

    #[tokio::test]
    #[serial]
    #[ignore = "10,000-round GPU allocator stress test"]
    async fn test_rocm_allocator_stress() {
        const ROUNDS: usize = 10_000;
        const STREAMS_PER_PROOF: usize = DEFAULT_NUM_TASKS;
        const MIB: usize = 1024 * 1024;

        let mut state = 0x8f3d_9a27_4c61_b5e2u64;
        let mut cases = Vec::with_capacity(ROUNDS);
        for index in 0..ROUNDS {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let size = match index {
                0..=20 => 1usize << (index + 8),
                21 => 258 * MIB,
                _ if index % 997 == 0 => 32 * MIB + state as usize % (224 * MIB + 1),
                _ => 256 + state as usize % (MIB - 255),
            };
            cases.push((index, size));
        }

        for batch in cases.chunks(STREAMS_PER_PROOF) {
            let batch = batch.to_vec();
            let handle = crate::run_proof_in_place(|scope| async move {
                let mut children = Vec::with_capacity(batch.len());
                for (slot, (index, size)) in batch.into_iter().enumerate() {
                    let pattern = (slot + 1) as u8;
                    children.push(scope.spawn(move |child| async move {
                        let mut buffer = DeviceBuffer::<u8>::with_capacity_in(size, child.clone());
                        unsafe {
                            child
                                .write_bytes(buffer.as_mut_ptr(), pattern, size)
                                .expect("failed to fill stress allocation");
                            buffer.set_len(size);
                        }
                        (index, pattern, buffer)
                    }));
                }

                for child in children {
                    let (index, pattern, buffer) =
                        child.await.expect("allocator stress child task failed");
                    let host = buffer.to_host().expect("failed to read stress allocation");
                    assert!(
                        host.iter().all(|value| *value == pattern),
                        "allocation pattern mismatch in round {index}"
                    );
                }
            })
            .await;
            handle.await.expect("allocator stress root task failed");
        }
    }
}
