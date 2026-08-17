use crate::{CudaError, TaskScope};
use slop_alloc::{mem::CopyError, CopyIntoBackend, CopyToBackend, CpuBackend};
use sp1_gpu_sys::runtime::cuda_mem_get_info;

const BYTES_PER_GIB: usize = 1024 * 1024 * 1024;

/// Smallest ROCm device-memory size supported by the low-memory prover profile.
pub const ROCM_LOW_MEMORY_MIN_GIB: usize = 16;

/// Devices below this size use the ROCm low-memory prover profile.
pub const ROCM_STANDARD_MEMORY_MIN_GIB: usize = 20;

pub trait DeviceCopy: Copy + 'static + Sized {}

impl<T: Copy + 'static + Sized> DeviceCopy for T {}

/// Returns a pair `(free, total)` of the amount of free and total memory on the device.
pub fn cuda_memory_info() -> Result<(usize, usize), CudaError> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    CudaError::result_from_ffi(unsafe { cuda_mem_get_info(&mut free, &mut total) })?;
    Ok((free, total))
}

/// Converts a device-memory byte count to GiB, rounded up to match nominal GPU sizes.
pub fn gpu_memory_gib(total_bytes: usize) -> usize {
    total_bytes.div_ceil(BYTES_PER_GIB)
}

/// Returns whether a supported ROCm device needs the low-memory prover profile.
pub fn is_rocm_low_memory_gpu(total_bytes: usize) -> bool {
    let total_gib = gpu_memory_gib(total_bytes);
    (ROCM_LOW_MEMORY_MIN_GIB..ROCM_STANDARD_MEMORY_MIN_GIB).contains(&total_gib)
}

pub trait IntoDevice: CopyIntoBackend<TaskScope, CpuBackend> + Sized {
    fn into_device_in(self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        self.copy_into_backend(backend)
    }
}

impl<T> IntoDevice for T where T: CopyIntoBackend<TaskScope, CpuBackend> + Sized {}

pub trait ToDevice: CopyToBackend<TaskScope, CpuBackend> + Sized {
    fn to_device_in(&self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        self.copy_to_backend(backend)
    }
}

impl<T> ToDevice for T where T: CopyToBackend<TaskScope, CpuBackend> + Sized {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_device_memory_up_to_nominal_gib() {
        assert_eq!(gpu_memory_gib(16 * BYTES_PER_GIB - 1), 16);
        assert_eq!(gpu_memory_gib(16 * BYTES_PER_GIB), 16);
        assert_eq!(gpu_memory_gib(16 * BYTES_PER_GIB + 1), 17);
    }

    #[test]
    fn selects_only_supported_low_memory_rocm_devices() {
        assert!(!is_rocm_low_memory_gpu(15 * BYTES_PER_GIB));
        assert!(is_rocm_low_memory_gpu(16 * BYTES_PER_GIB));
        assert!(is_rocm_low_memory_gpu(19 * BYTES_PER_GIB));
        assert!(!is_rocm_low_memory_gpu(20 * BYTES_PER_GIB));
    }
}

#[macro_export]
macro_rules! args {
    ($($arg:expr),*) => {
        [
            $(
                &$arg as *const _ as *mut std::ffi::c_void
            ),*
        ]
    };
}
