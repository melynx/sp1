cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED SP1_GPU_BACKEND)
    set(SP1_GPU_BACKEND "cuda")
endif()

string(TOLOWER "${SP1_GPU_BACKEND}" SP1_GPU_BACKEND)

project(sp1-gpu LANGUAGES CXX)

if(SP1_GPU_BACKEND STREQUAL "cuda")
    enable_language(CUDA)
elseif(SP1_GPU_BACKEND STREQUAL "rocm")
    enable_language(HIP)
else()
    message(FATAL_ERROR "unsupported SP1_GPU_BACKEND=${SP1_GPU_BACKEND}; expected cuda or rocm")
endif()

if(SP1_GPU_BACKEND STREQUAL "cuda")
    # Require CUDA 12.0+
    find_package(CUDAToolkit 12.0 REQUIRED)
else()
    if(DEFINED HIP_PATH AND NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "${HIP_PATH}")
    endif()
    if(DEFINED ROCM_PATH)
        list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}" "${ROCM_PATH}/lib/cmake/hip")
    endif()
    find_package(hip REQUIRED CONFIG)
endif()

if(SP1_GPU_BACKEND STREQUAL "cuda")
    # Get CUDA version for architecture selection
    set(CUDA_VERSION_MAJOR ${CUDAToolkit_VERSION_MAJOR})
    set(CUDA_VERSION_MINOR ${CUDAToolkit_VERSION_MINOR})
endif()

# C++20 for host and GPU code
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(SP1_GPU_BACKEND STREQUAL "cuda")
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
endif()

if(SP1_GPU_BACKEND STREQUAL "cuda")
    # Don't use response files for CUDA includes - this breaks clangd IDE support
    # because clangd can't parse the --options-file flag
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
endif()

# Architecture flags
if(SP1_GPU_BACKEND STREQUAL "cuda")
    # sm_100 / sm_120 (Blackwell) require CUDA 12.8+; older toolchains can't compile them.
    if(DEFINED CUDA_ARCHS)
        # Convert comma-separated string to CMake list (semicolon-separated)
        string(REPLACE "," ";" CUDA_ARCHS_LIST "${CUDA_ARCHS}")
        set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHS_LIST})
    elseif((CUDA_VERSION_MAJOR EQUAL 12 AND CUDA_VERSION_MINOR GREATER_EQUAL 8)
            OR CUDA_VERSION_MAJOR GREATER_EQUAL 13)
        set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90 100 120)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
    endif()
else()
    if(DEFINED ROCM_ARCHS)
        string(REPLACE "," ";" ROCM_ARCHS_LIST "${ROCM_ARCHS}")
        set(CMAKE_HIP_ARCHITECTURES ${ROCM_ARCHS_LIST})
    endif()
endif()

# Build type defaults to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(SP1_GPU_BACKEND STREQUAL "cuda")
    # Common CUDA flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -default-stream=per-thread")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
    # -fPIC (not -fPIE): the static archive may be linked into a cdylib downstream.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
else()
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fPIC")
endif()

# Relocatable device code (-rdc=true) for cross-file device function calls.
# Setting the variable here initializes CUDA_SEPARABLE_COMPILATION on every
# target created afterwards (including the per-module OBJECT libraries),
# so each .cu file is compiled with -dc and the final STATIC target performs
# device linking.
if(SP1_GPU_BACKEND STREQUAL "cuda")
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

# Debug build: -G implies -O0 device-side; nvcc warns when combined with -O.
if(SP1_GPU_BACKEND STREQUAL "cuda")
    if(PROFILE_DEBUG_DATA)
        set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -G")
    else()
        set(CMAKE_CUDA_FLAGS_DEBUG "-O3")
    endif()
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
else()
    set(CMAKE_HIP_FLAGS_DEBUG "-O0")
    set(CMAKE_HIP_FLAGS_RELEASE "-O3")
endif()

# Get the actual source directory (where this CMakeLists.txt lives)
set(CSL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

function(sp1_gpu_add_object_library target_name)
    if(ARGV1 STREQUAL "OBJECT")
        list(REMOVE_AT ARGN 0)
    endif()
    add_library(${target_name} OBJECT ${ARGN})
    if(SP1_GPU_BACKEND STREQUAL "rocm")
        foreach(source_file ${ARGN})
            if(source_file MATCHES "\\.cu$")
                set_source_files_properties(${source_file} PROPERTIES LANGUAGE HIP)
            endif()
        endforeach()
    endif()
endfunction()

# Create an INTERFACE library for common settings (following CUTLASS pattern)
# All CUDA modules link to this to get proper include paths and definitions
add_library(sp1_gpu_common INTERFACE)
target_include_directories(sp1_gpu_common INTERFACE
    ${CSL_SOURCE_DIR}/include
    ${CSL_SOURCE_DIR}/sppark
)
target_compile_definitions(sp1_gpu_common INTERFACE SPPARK FEATURE_KOALA_BEAR)

if(SP1_GPU_BACKEND STREQUAL "cuda")
    target_include_directories(sp1_gpu_common INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_compile_definitions(sp1_gpu_common INTERFACE SP1_GPU_BACKEND_CUDA=1)
else()
    target_compile_definitions(sp1_gpu_common INTERFACE SP1_GPU_BACKEND_ROCM=1 __HIP_PLATFORM_AMD__)
    target_link_libraries(sp1_gpu_common INTERFACE hip::host)
endif()

# Add cbindgen include directory if provided
if(DEFINED CBINDGEN_INCLUDE_DIR)
    target_include_directories(sp1_gpu_common INTERFACE ${CBINDGEN_INCLUDE_DIR})
endif()

# Output directories - use the project root's target directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CSL_SOURCE_DIR}/target/${SP1_GPU_BACKEND}-build/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CSL_SOURCE_DIR}/target/${SP1_GPU_BACKEND}-build/lib)

# Add subdirectories for each module
add_subdirectory(lib/algebra)
add_subdirectory(lib/basefold)
add_subdirectory(lib/challenger)
add_subdirectory(lib/experimental)
add_subdirectory(lib/jagged_assist)
add_subdirectory(lib/jagged_sumcheck)
add_subdirectory(lib/logup_gkr)
add_subdirectory(lib/merkle_tree)
add_subdirectory(lib/mle)
add_subdirectory(lib/ntt)
add_subdirectory(lib/runtime)
add_subdirectory(lib/scan)
add_subdirectory(lib/sum_and_reduce)
add_subdirectory(lib/tracegen)
add_subdirectory(lib/transpose)
add_subdirectory(lib/zerocheck)

# Sppark sources (external library, kept separate)
# These are C++ files that need CUDA headers
sp1_gpu_add_object_library(sppark_objs
    ${CSL_SOURCE_DIR}/sppark/lib.cpp
    ${CSL_SOURCE_DIR}/sppark/util/all_gpus.cpp
)
target_link_libraries(sppark_objs PRIVATE sp1_gpu_common)

# Collect all object libraries
set(ALL_CUDA_OBJECTS
    $<TARGET_OBJECTS:algebra_objs>
    $<TARGET_OBJECTS:basefold_objs>
    $<TARGET_OBJECTS:challenger_objs>
    $<TARGET_OBJECTS:experimental_objs>
    $<TARGET_OBJECTS:jagged_assist_objs>
    $<TARGET_OBJECTS:jagged_sumcheck_objs>
    $<TARGET_OBJECTS:logup_gkr_objs>
    $<TARGET_OBJECTS:merkle_tree_objs>
    $<TARGET_OBJECTS:mle_objs>
    $<TARGET_OBJECTS:ntt_objs>
    $<TARGET_OBJECTS:runtime_objs>
    $<TARGET_OBJECTS:scan_objs>
    $<TARGET_OBJECTS:sum_and_reduce_objs>
    $<TARGET_OBJECTS:tracegen_objs>
    $<TARGET_OBJECTS:transpose_objs>
    $<TARGET_OBJECTS:zerocheck_objs>
    $<TARGET_OBJECTS:sppark_objs>
)

# Create static library with device linking
set(SP1_GPU_TARGET "sys-${SP1_GPU_BACKEND}")
add_library(${SP1_GPU_TARGET} STATIC ${ALL_CUDA_OBJECTS})
if(SP1_GPU_BACKEND STREQUAL "cuda")
    set_target_properties(${SP1_GPU_TARGET} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

if(SP1_GPU_BACKEND STREQUAL "cuda")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
        # nothing needed
    else()
        target_link_libraries(${SP1_GPU_TARGET} PRIVATE
            CUDA::cudart
            CUDA::nvToolsExt
        )
    endif()
else()
    target_link_libraries(${SP1_GPU_TARGET} PRIVATE hip::host roctx64)
endif()

# Install rule
install(TARGETS ${SP1_GPU_TARGET}
    ARCHIVE DESTINATION lib
)
