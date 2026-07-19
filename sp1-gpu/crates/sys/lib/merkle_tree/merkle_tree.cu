#include <stdio.h>
#include "poseidon2/poseidon2_kb31_16.cuh"
#include "poseidon2/poseidon2.cuh"
#include "poseidon2/poseidon2_bn254_3.cuh"

template <typename Hasher_t, typename HashParams, typename HasherState_t>
__global__ void leafHash(
    Hasher_t hasher,
    kb31_t* input,
    typename HashParams::F_t (*digests)[HashParams::DIGEST_WIDTH],
    size_t widths,
    size_t tree_height) {
    HasherState_t state;

    size_t matrixHeight = 1 << tree_height;
    for (size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < matrixHeight;
         idx += blockDim.x * gridDim.x) {
        state.absorbRow(hasher, input, idx, widths, matrixHeight);
        size_t digestIdx = idx + (matrixHeight - 1);
        state.finalize(hasher, digests[digestIdx]);
    }
}

#if defined(SP1_GPU_BACKEND_ROCM)

// The generic state absorbs one field at a time and updates a dynamic index for
// every load. KoalaBear always has an eight-field rate. Load each full rate
// block directly while keeping the same state, permutation, and padding rules.
__global__ void leafHashKoalaBear16Rocm(
    poseidon2::KoalaBearHasher,
    kb31_t* __restrict__ input,
    kb31_t (*__restrict__ digests)[poseidon2_kb31_16::constants::DIGEST_WIDTH],
    size_t width,
    size_t tree_height) {
    constexpr size_t RATE = poseidon2_kb31_16::constants::RATE;
    constexpr size_t WIDTH = poseidon2_kb31_16::constants::WIDTH;
    size_t matrix_height = size_t(1) << tree_height;

    for (size_t row = (blockIdx.x * blockDim.x) + threadIdx.x; row < matrix_height;
         row += blockDim.x * gridDim.x) {
        __align__(16) kb31_t state[WIDTH];
#pragma unroll
        for (size_t i = 0; i < WIDTH; i++) {
            state[i].set_to_zero();
        }

        size_t column = 0;
        for (; column + RATE <= width; column += RATE) {
#pragma unroll
            for (size_t i = 0; i < RATE; i++) {
                state[i] = input[(column + i) * matrix_height + row];
            }
            poseidon2::KoalaBearHasher::permute(state, state);
        }

        if (column < width) {
            for (size_t i = 0; column + i < width; i++) {
                state[i] = input[(column + i) * matrix_height + row];
            }
            poseidon2::KoalaBearHasher::permute(state, state);
        }

        size_t digest_idx = row + (matrix_height - 1);
#pragma unroll
        for (size_t i = 0; i < poseidon2_kb31_16::constants::DIGEST_WIDTH; i++) {
            digests[digest_idx][i] = state[i];
        }
    }
}

#endif

extern "C" void* leaf_hash_merkle_tree_koala_bear_16_kernel() {
#if defined(SP1_GPU_BACKEND_ROCM)
    return (void*)leafHashKoalaBear16Rocm;
#else
    return (void*)leafHash<
        poseidon2::KoalaBearHasher,
        poseidon2_kb31_16::KoalaBear,
        poseidon2::KoalaBearHasherState>;
#endif
}

extern "C" void* leaf_hash_merkle_tree_bn254_kernel() {
    return (void*)
        leafHash<poseidon2::Bn254Hasher, poseidon2_bn254_3::Bn254, poseidon2::Bn254HasherState>;
}

template <typename Hasher_t, typename HashParams, typename HasherState_t>
__global__ void compress(
    Hasher_t hasher,
    typename HashParams::F_t (*digests)[HashParams::DIGEST_WIDTH],
    size_t layer_height) {
    size_t layerLength = 1 << layer_height;
    for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < layerLength;
         i += blockDim.x * gridDim.x) {
        size_t idx = i + (layerLength - 1);
        size_t leftIdx = (idx << 1) + 1;
        size_t rightIdx = leftIdx + 1;
        hasher.compress(digests[leftIdx], digests[rightIdx], digests[idx]);
    }
}

extern "C" void* compress_merkle_tree_koala_bear_16_kernel() {
    return (void*)compress<
        poseidon2::KoalaBearHasher,
        poseidon2_kb31_16::KoalaBear,
        poseidon2::KoalaBearHasherState>;
}

extern "C" void* compress_merkle_tree_bn254_kernel() {
    return (void*)
        compress<poseidon2::Bn254Hasher, poseidon2_bn254_3::Bn254, poseidon2::Bn254HasherState>;
}


template <typename Hasher_t, typename HashParams, typename HasherState_t>
__global__ void computePaths(
    typename HashParams::F_t (*paths)[HashParams::DIGEST_WIDTH],
    size_t* indices,
    size_t numIndices,
    typename HashParams::F_t (*digests)[HashParams::DIGEST_WIDTH],
    size_t tree_height) {
    for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < numIndices;
         i += blockDim.x * gridDim.x) {
        size_t idx = (1 << tree_height) - 1 + indices[i];
        for (int k = 0; k < tree_height; k++) {
            size_t siblingIdx = ((idx - 1) ^ 1) + 1;
            size_t parentIdx = (idx - 1) >> 1;
            typename HashParams::F_t* digest = digests[siblingIdx];
            typename HashParams::F_t* path_digest = paths[i * tree_height + k];
#pragma unroll
            for (int j = 0; j < HashParams::DIGEST_WIDTH; j++) {
                path_digest[j] = digest[j];
            }
            idx = parentIdx;
        }
    }
}


extern "C" void* compute_paths_merkle_tree_koala_bear_16_kernel() {
    return (void*)computePaths<
        poseidon2::KoalaBearHasher,
        poseidon2_kb31_16::KoalaBear,
        poseidon2::KoalaBearHasherState>;
}

extern "C" void* compute_paths_merkle_tree_bn254_kernel() {
    return (void*)
        computePaths<poseidon2::Bn254Hasher, poseidon2_bn254_3::Bn254, poseidon2::Bn254HasherState>;
}


template <typename Hasher_t, typename HashParams, typename HasherState_t>
__global__ void computeOpenings(
    kb31_t** __restrict__ inputs,
    kb31_t* __restrict__ outputs,
    size_t* indices,
    size_t numIndices,
    size_t numInputs,
    size_t* batchSizes,
    size_t* batchOffsets,
    size_t matrixHeight,
    size_t numOpeningValues) {
    for (size_t batchIdx = (blockIdx.z * blockDim.z) + threadIdx.z; batchIdx < numInputs;
         batchIdx += blockDim.z * gridDim.z) {
        kb31_t* in = inputs[batchIdx];
        size_t offset = batchOffsets[batchIdx];
        size_t batchSize = batchSizes[batchIdx];
        for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < numIndices;
             i += blockDim.x * gridDim.x) {
            size_t rowIdx = indices[i];
            for (size_t j = (blockIdx.y * blockDim.y) + threadIdx.y; j < batchSize;
                 j += blockDim.y * gridDim.y) {
                outputs[i * numOpeningValues + j + offset] = in[j * matrixHeight + rowIdx];
            }
        }
    }
}

extern "C" void* compute_openings_merkle_tree_koala_bear_16_kernel() {
    return (void*)computeOpenings<
        poseidon2::KoalaBearHasher,
        poseidon2_kb31_16::KoalaBear,
        poseidon2::KoalaBearHasherState>;
}

extern "C" void* compute_openings_merkle_tree_bn254_kernel() {
    return (void*)computeOpenings<
        poseidon2::Bn254Hasher,
        poseidon2_bn254_3::Bn254,
        poseidon2::Bn254HasherState>;
}
