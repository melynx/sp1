#pragma once

#if defined(SP1_GPU_BACKEND_ROCM)
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#endif
