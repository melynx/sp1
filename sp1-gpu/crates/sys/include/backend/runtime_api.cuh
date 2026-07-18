#pragma once

#if defined(SP1_GPU_BACKEND_ROCM)
#include <hip/hip_runtime.h>

using cudaError_t = hipError_t;
using cudaEvent_t = hipEvent_t;
using cudaHostFn_t = hipHostFn_t;
using cudaMemPool_t = hipMemPool_t;
using cudaDeviceProp = hipDeviceProp_t;
using cudaStream_t = hipStream_t;

#define cudaDeviceGetDefaultMemPool hipDeviceGetDefaultMemPool
#define cudaDeviceGetMemPool hipDeviceGetMemPool
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaErrorMemoryAllocation hipErrorMemoryAllocation
#define cudaErrorNotReady hipErrorNotReady
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaFree hipFree
#define cudaFreeAsync hipFreeAsync
#define cudaFreeHost hipHostFree
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGetSymbolAddress hipGetSymbolAddress
#define cudaHostRegister hipHostRegister
#define cudaHostRegisterDefault hipHostRegisterDefault
#define cudaHostUnregister hipHostUnregister
#define cudaLaunchCooperativeKernel hipLaunchCooperativeKernel
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaLaunchKernel hipLaunchKernel
#define cudaMalloc hipMalloc
#define cudaMallocAsync hipMallocAsync
#define cudaMallocHost hipHostMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaMemcpyToSymbolAsync hipMemcpyToSymbolAsync
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold
#define cudaMemPoolSetAttribute hipMemPoolSetAttribute
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize
#define cudaSetDevice hipSetDevice
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDefault hipStreamDefault
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamQuery hipStreamQuery
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaSuccess hipSuccess

#define __shfl_sync(mask, var, src, ...) __shfl((var), (src), ##__VA_ARGS__)
#define __shfl_down_sync(mask, var, delta, ...) __shfl_down((var), (delta), ##__VA_ARGS__)
#define __shfl_xor_sync(mask, var, lane_mask, ...) __shfl_xor((var), (lane_mask), ##__VA_ARGS__)
#define __trap() __builtin_trap()

static inline hipError_t cudaEventCreate(hipEvent_t* event) { return hipEventCreate(event); }

static inline hipError_t cudaEventCreate(hipEvent_t* event, unsigned int flags) {
    return hipEventCreateWithFlags(event, flags);
}

template <typename F>
static inline hipError_t cudaFuncSetAttribute(F func, hipFuncAttribute attr, int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(func), attr, value);
}

#if defined(__HIPCC__)
__device__ __forceinline__ uint32_t sp1_device_load_acquire(const uint32_t* ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void sp1_device_store_release(uint32_t* ptr, uint32_t value) {
    __hip_atomic_store(ptr, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
#endif

#else
#include <cuda_runtime.h>
#include <cuda/atomic>

__device__ __forceinline__ uint32_t sp1_device_load_acquire(const uint32_t* ptr) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> flag(*const_cast<uint32_t*>(ptr));
    return flag.load(cuda::memory_order_acquire);
}

__device__ __forceinline__ void sp1_device_store_release(uint32_t* ptr, uint32_t value) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> flag(*ptr);
    flag.store(value, cuda::memory_order_release);
}
#endif
