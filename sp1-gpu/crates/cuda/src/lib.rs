mod arena;
mod buffer;
mod codeword;
mod device;
mod error;
mod event;
mod global;
mod mle;
mod pinned;
mod scan;
mod stream;
pub mod sync;
pub mod task;
mod tensor;
mod tracegen;

pub use error::CudaError;
pub use event::CudaEvent;
pub use stream::{CudaStream, StreamCallbackFuture};

pub use arena::{shutdown_slab_cache, ProofArena, RocmAllocator, TaskSubArena};
pub use buffer::*;
pub use device::*;
pub use mle::*;
pub use pinned::*;
pub use scan::*;
pub use task::*;
pub use tensor::*;
pub use tracegen::*;

pub mod sys {
    pub use sp1_gpu_sys::*;
}

/// Select the process device and initialize sppark NTT parameters.
///
/// Call this on the server's main thread before creating the async runtime.
pub fn initialize_device_and_ntt(device_id: i32) -> Result<(), CudaError> {
    use sp1_gpu_sys::{
        dft::sppark_init,
        runtime::{
            cuda_set_device, cuda_stream_create, cuda_stream_destroy, cuda_stream_synchronize,
            CudaDevice, CudaStreamHandle,
        },
    };
    use std::{ffi::c_void, ptr};

    unsafe {
        CudaError::result_from_ffi(cuda_set_device(CudaDevice(device_id)))?;
        let mut stream = CudaStreamHandle(ptr::null_mut::<c_void>());
        CudaError::result_from_ffi(cuda_stream_create(&mut stream))?;
        let init_result = CudaError::result_from_ffi(sppark_init(stream));
        let sync_result = CudaError::result_from_ffi(cuda_stream_synchronize(stream));
        let destroy_result = CudaError::result_from_ffi(cuda_stream_destroy(stream));
        init_result?;
        sync_result?;
        destroy_result
    }
}
