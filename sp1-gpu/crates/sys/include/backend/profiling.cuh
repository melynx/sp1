#pragma once

#include <cstdint>

#if defined(SP1_GPU_BACKEND_ROCM)
#include <roctracer/roctx.h>

using nvtxDomainHandle_t = void*;

inline nvtxDomainHandle_t nvtxDomainCreateA(const char*) { return nullptr; }
inline void nvtxDomainDestroy(nvtxDomainHandle_t) {}
inline uint64_t nvtxRangeStart(const char* name) { return roctxRangeStartA(name); }
inline void nvtxRangeEnd(uint64_t id) { roctxRangeStop(id); }
#else
#include <nvtx3/nvToolsExt.h>
#endif
