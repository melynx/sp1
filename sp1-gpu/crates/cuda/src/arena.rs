use std::{
    alloc::Layout,
    ffi::c_void,
    ptr::{self, NonNull},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex, OnceLock,
    },
};

use slop_alloc::AllocError;
use sp1_gpu_sys::runtime::{cuda_free, cuda_malloc};

use crate::{CudaError, CudaEvent};

const MIB: usize = 1024 * 1024;
const INITIAL_SLAB: usize = 256 * MIB;
const MAX_SLAB: usize = 8 * 1024 * MIB;
const SEGMENT_SIZE: usize = 64 * MIB;
const LOCAL_LIMIT: usize = 32 * MIB;
const DEDICATED_ALIGN: usize = 2 * MIB;
// Native kernels rely on the 256-byte base alignment guaranteed by
// hipMalloc/hipMallocAsync, even when Rust types report a smaller alignment.
// Preserve that guarantee for every arena suballocation.
const DEVICE_MIN_ALIGN: usize = 256;

static NEXT_ARENA_ID: AtomicU64 = AtomicU64::new(1);
static SLAB_CACHE: OnceLock<DeviceSlabCache> = OnceLock::new();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RocmAllocator {
    Arena,
    Async,
}

impl RocmAllocator {
    pub fn selected() -> Self {
        static SELECTED: OnceLock<RocmAllocator> = OnceLock::new();
        *SELECTED.get_or_init(|| match std::env::var("SP1_ROCM_ALLOCATOR").as_deref() {
            Ok("async") => Self::Async,
            Ok("arena") | Err(_) => Self::Arena,
            Ok(value) => panic!("invalid SP1_ROCM_ALLOCATOR={value}; expected `arena` or `async`"),
        })
    }
}

#[derive(Debug)]
struct Slab {
    ptr: NonNull<u8>,
    size: usize,
    offset: AtomicUsize,
}

unsafe impl Send for Slab {}
unsafe impl Sync for Slab {}

impl Slab {
    fn allocate(size: usize) -> Result<Self, CudaError> {
        let mut ptr = ptr::null_mut::<c_void>();
        CudaError::result_from_ffi(unsafe { cuda_malloc(&mut ptr, size) })?;
        Ok(Self {
            ptr: NonNull::new(ptr.cast()).expect("GPU allocator returned a null pointer"),
            size,
            offset: AtomicUsize::new(0),
        })
    }

    fn reset(&self) {
        self.offset.store(0, Ordering::Release);
    }

    fn try_allocate(&self, layout: Layout) -> Option<NonNull<u8>> {
        let align = layout.align().max(DEVICE_MIN_ALIGN);
        let mut old = self.offset.load(Ordering::Relaxed);
        loop {
            let aligned = old.checked_add(align - 1)? & !(align - 1);
            let next = aligned.checked_add(layout.size())?;
            if next > self.size {
                return None;
            }
            match self.offset.compare_exchange_weak(old, next, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => {
                    return Some(unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(aligned)) })
                }
                Err(actual) => old = actual,
            }
        }
    }
}

#[derive(Debug, Default)]
struct DeviceSlabCache {
    slabs: Mutex<Vec<Slab>>,
}

impl DeviceSlabCache {
    fn acquire(&self, size: usize) -> Result<Slab, CudaError> {
        let mut cached = self.slabs.lock().unwrap();
        if let Some(index) = cached.iter().position(|slab| slab.size >= size) {
            let slab = cached.swap_remove(index);
            drop(cached);
            slab.reset();
            return Ok(slab);
        }
        drop(cached);

        match Slab::allocate(size) {
            Ok(slab) => Ok(slab),
            Err(CudaError::OutOfMemory) => {
                self.release_all();
                Slab::allocate(size)
            }
            Err(error) => Err(error),
        }
    }

    fn recycle(&self, slabs: Vec<Slab>) {
        for slab in &slabs {
            slab.reset();
        }
        self.slabs.lock().unwrap().extend(slabs);
    }

    fn release_all(&self) {
        for slab in self.slabs.lock().unwrap().drain(..) {
            CudaError::result_from_ffi(unsafe { cuda_free(slab.ptr.as_ptr().cast()) })
                .expect("failed to release a cached GPU slab");
        }
    }
}

#[derive(Debug)]
struct ProofArenaInner {
    id: u64,
    slabs: Mutex<Vec<Arc<Slab>>>,
    next_slab_size: AtomicUsize,
    completion: Mutex<Option<CudaEvent>>,
}

impl Drop for ProofArenaInner {
    fn drop(&mut self) {
        if let Some(event) = self.completion.get_mut().unwrap().take() {
            event.synchronize().expect("proof arena completion event failed");
        }
        let slabs = self
            .slabs
            .get_mut()
            .unwrap()
            .drain(..)
            .map(|slab| Arc::try_unwrap(slab).expect("GPU slab still has an allocation owner"))
            .collect();
        slab_cache().recycle(slabs);
    }
}

#[derive(Clone, Debug)]
pub struct ProofArena(Arc<ProofArenaInner>);

impl ProofArena {
    pub fn new() -> Result<Self, CudaError> {
        let first = Arc::new(slab_cache().acquire(INITIAL_SLAB)?);
        Ok(Self(Arc::new(ProofArenaInner {
            id: NEXT_ARENA_ID.fetch_add(1, Ordering::Relaxed),
            slabs: Mutex::new(vec![first]),
            next_slab_size: AtomicUsize::new(INITIAL_SLAB * 2),
            completion: Mutex::new(None),
        })))
    }

    pub fn id(&self) -> u64 {
        self.0.id
    }

    pub fn finish(&self, event: CudaEvent) {
        let previous = self.0.completion.lock().unwrap().replace(event);
        assert!(previous.is_none(), "proof arena finished more than once");
    }

    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        for slab in self.0.slabs.lock().unwrap().iter() {
            if let Some(ptr) = slab.try_allocate(layout) {
                return Ok(ptr);
            }
        }

        let current = self.0.next_slab_size.load(Ordering::Acquire);
        let dedicated = layout.size() > current / 2;
        let slab_size = if dedicated {
            round_up(layout.size(), DEDICATED_ALIGN).ok_or(AllocError)?
        } else {
            current
        };
        let slab = Arc::new(slab_cache().acquire(slab_size).map_err(|_| AllocError)?);
        if !dedicated {
            self.0.next_slab_size.store(current.saturating_mul(2).min(MAX_SLAB), Ordering::Release);
        }
        let ptr = slab.try_allocate(layout).ok_or(AllocError)?;
        self.0.slabs.lock().unwrap().push(slab);
        Ok(ptr)
    }
}

#[derive(Debug)]
struct Segment {
    ptr: NonNull<u8>,
    offset: AtomicUsize,
}

unsafe impl Send for Segment {}
unsafe impl Sync for Segment {}

#[derive(Debug)]
pub struct TaskSubArena {
    arena: ProofArena,
    segment: Mutex<Option<Segment>>,
}

impl TaskSubArena {
    pub fn new(arena: ProofArena) -> Self {
        Self { arena, segment: Mutex::new(None) }
    }

    pub fn arena_id(&self) -> u64 {
        self.arena.id()
    }

    pub fn proof_arena(&self) -> ProofArena {
        self.arena.clone()
    }

    pub fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        if layout.size() > LOCAL_LIMIT {
            return self.arena.allocate(layout);
        }
        let mut current = self.segment.lock().unwrap();
        loop {
            if current.is_none() {
                let segment_layout = Layout::from_size_align(SEGMENT_SIZE, DEDICATED_ALIGN)
                    .map_err(|_| AllocError)?;
                *current = Some(Segment {
                    ptr: self.arena.allocate(segment_layout)?,
                    offset: AtomicUsize::new(0),
                });
            }

            let segment = current.as_ref().unwrap();
            let align = layout.align().max(DEVICE_MIN_ALIGN);
            let mut old = segment.offset.load(Ordering::Relaxed);
            loop {
                let aligned = old.checked_add(align - 1).ok_or(AllocError)? & !(align - 1);
                let next = aligned.checked_add(layout.size()).ok_or(AllocError)?;
                if next > SEGMENT_SIZE {
                    *current = None;
                    break;
                }
                match segment.offset.compare_exchange_weak(
                    old,
                    next,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        return Ok(unsafe {
                            NonNull::new_unchecked(segment.ptr.as_ptr().add(aligned))
                        });
                    }
                    Err(actual) => old = actual,
                }
            }
        }
    }
}

fn slab_cache() -> &'static DeviceSlabCache {
    SLAB_CACHE.get_or_init(DeviceSlabCache::default)
}

pub fn shutdown_slab_cache() {
    if let Some(cache) = SLAB_CACHE.get() {
        cache.release_all();
    }
}

fn round_up(value: usize, align: usize) -> Option<usize> {
    value.checked_add(align - 1).map(|value| value & !(align - 1))
}
