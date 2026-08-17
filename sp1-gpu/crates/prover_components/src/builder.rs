use std::sync::Arc;

use sp1_core_machine::riscv::RiscvAir;
use sp1_gpu_cudart::{cuda_memory_info, gpu_memory_gib, TaskScope};
#[cfg(feature = "rocm")]
use sp1_gpu_cudart::{is_rocm_low_memory_gpu, ROCM_LOW_MEMORY_MIN_GIB};

use sp1_core_executor::{SP1CoreOpts, ELEMENT_THRESHOLD};
use sp1_gpu_shard_prover::CudaShardProver;
use sp1_hypercube::{prover::ProverSemaphore, InnerSC, Machine, MachineVerifier};
use sp1_primitives::{SP1Field, SP1GlobalContext};
use sp1_prover::{
    worker::SP1WorkerBuilder, CompressAir, ReadyWrapProverBuilder, SP1ProverComponents,
    CORE_LOG_STACKING_HEIGHT,
};

pub const RECURSION_TRACE_ALLOCATION: usize = 1 << 27;
pub const SHRINK_TRACE_ALLOCATION: usize = 1 << 25;

/// Taken from "Total number of Cells" when generating traces for wrap. Plus an extra 5%.
pub const WRAP_TRACE_ALLOCATION: usize = 85_376_340;

#[cfg(feature = "rocm")]
const LOW_MEMORY_DEFAULT_LOG2_SHARD_SIZE: usize = 23;
#[cfg(feature = "rocm")]
const LOW_MEMORY_MIN_LOG2_SHARD_SIZE: usize = 22;
#[cfg(feature = "rocm")]
const LOW_MEMORY_MAX_LOG2_SHARD_SIZE: usize = 24;

use crate::{
    new_cuda_prover, CudaProverCoreComponents, CudaProverRecursionComponents,
    SP1CudaProverComponents,
};

pub fn local_gpu_opts() -> SP1CoreOpts {
    let total_memory = cuda_memory_info().expect("failed to read GPU memory").1;
    local_gpu_opts_for_memory(total_memory)
}

#[cfg(feature = "rocm")]
fn parse_low_memory_log2_shard_size(value: &str) -> usize {
    let log2_shard_size = value.parse::<usize>().unwrap_or_else(|_| {
        panic!(
            "invalid SP1_ROCM_LOW_MEMORY_LOG2_SHARD_SIZE={value}; expected an integer from \
             {LOW_MEMORY_MIN_LOG2_SHARD_SIZE} through {LOW_MEMORY_MAX_LOG2_SHARD_SIZE}"
        )
    });

    assert!(
        (LOW_MEMORY_MIN_LOG2_SHARD_SIZE..=LOW_MEMORY_MAX_LOG2_SHARD_SIZE)
            .contains(&log2_shard_size),
        "invalid SP1_ROCM_LOW_MEMORY_LOG2_SHARD_SIZE={value}; expected an integer from \
         {LOW_MEMORY_MIN_LOG2_SHARD_SIZE} through {LOW_MEMORY_MAX_LOG2_SHARD_SIZE}"
    );

    log2_shard_size
}

#[cfg(feature = "rocm")]
fn low_memory_log2_shard_size() -> (usize, &'static str) {
    match std::env::var("SP1_ROCM_LOW_MEMORY_LOG2_SHARD_SIZE") {
        Ok(value) => (parse_low_memory_log2_shard_size(&value), "environment"),
        Err(std::env::VarError::NotPresent) => {
            (LOW_MEMORY_DEFAULT_LOG2_SHARD_SIZE, "automatic low-memory profile")
        }
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("SP1_ROCM_LOW_MEMORY_LOG2_SHARD_SIZE must contain valid Unicode")
        }
    }
}

fn local_gpu_opts_for_memory(total_memory: usize) -> SP1CoreOpts {
    let mut opts = SP1CoreOpts::default();

    let gpu_memory_total_gib = gpu_memory_gib(total_memory);

    #[cfg(feature = "rocm")]
    if gpu_memory_total_gib < ROCM_LOW_MEMORY_MIN_GIB {
        panic!(
            "Unsupported ROCm GPU memory: {gpu_memory_total_gib} GiB, must be at least \
             {ROCM_LOW_MEMORY_MIN_GIB} GiB"
        );
    }

    #[cfg(feature = "rocm")]
    if is_rocm_low_memory_gpu(total_memory) {
        let (log2_shard_size, source) = low_memory_log2_shard_size();
        opts.shard_size = 1 << log2_shard_size;
        opts.sharding_threshold.element_threshold = 1 << (log2_shard_size + 4);
        opts.global_dependencies_opt = true;
        opts.recompute_gkr_trace = true;

        tracing::info!(
            gpu_memory_total_gib,
            source,
            log2_shard_size,
            shard_size = opts.shard_size,
            shard_threshold = opts.sharding_threshold.element_threshold,
            "Using ROCm low-memory prover profile"
        );

        return opts;
    }

    let log2_shard_size = 24;
    opts.shard_size = 1 << log2_shard_size;

    // Keep the existing four-GiB allowance used by the standard GPU profile.
    let gpu_memory_gb = gpu_memory_total_gib + 4;

    if gpu_memory_gb < 24 {
        panic!("Unsupported GPU memory: {gpu_memory_gb}, must be at least 24GB");
    }

    let shard_threshold = if gpu_memory_gb <= 30 {
        ELEMENT_THRESHOLD - (1 << 26) - (1 << 25)
    } else {
        ELEMENT_THRESHOLD
    };

    tracing::debug!("Shard threshold: {shard_threshold}");
    opts.sharding_threshold.element_threshold = shard_threshold;

    opts.global_dependencies_opt = true;

    // Always recompute GKR trace
    // TODO: tune relative to GPU memory
    opts.recompute_gkr_trace = true;

    opts
}

#[cfg(all(test, feature = "rocm"))]
mod tests {
    use super::*;

    const GIB: usize = 1024 * 1024 * 1024;

    #[test]
    fn uses_low_memory_options_for_a_16_gib_rocm_gpu() {
        let opts = local_gpu_opts_for_memory(16 * GIB);

        assert_eq!(opts.shard_size, 1 << LOW_MEMORY_DEFAULT_LOG2_SHARD_SIZE);
        assert_eq!(
            opts.sharding_threshold.element_threshold,
            1 << (LOW_MEMORY_DEFAULT_LOG2_SHARD_SIZE + 4)
        );
        assert!(opts.global_dependencies_opt);
        assert!(opts.recompute_gkr_trace);
    }

    #[test]
    fn parses_supported_low_memory_shard_sizes() {
        assert_eq!(parse_low_memory_log2_shard_size("22"), 22);
        assert_eq!(parse_low_memory_log2_shard_size("23"), 23);
        assert_eq!(parse_low_memory_log2_shard_size("24"), 24);
    }

    #[test]
    #[should_panic(expected = "expected an integer from 22 through 24")]
    fn rejects_too_large_low_memory_shard_size() {
        parse_low_memory_log2_shard_size("25");
    }

    #[test]
    fn keeps_standard_options_for_a_20_gib_rocm_gpu() {
        let opts = local_gpu_opts_for_memory(20 * GIB);

        assert_eq!(opts.shard_size, 1 << 24);
        assert_eq!(
            opts.sharding_threshold.element_threshold,
            ELEMENT_THRESHOLD - (1 << 26) - (1 << 25)
        );
    }

    #[test]
    #[should_panic(expected = "must be at least 16 GiB")]
    fn rejects_a_rocm_gpu_below_16_gib() {
        local_gpu_opts_for_memory(15 * GIB);
    }
}

/// Create a [SP1CudaProverWorkerBuilder] with a default machine.
pub async fn cuda_worker_builder(scope: TaskScope) -> SP1WorkerBuilder<SP1CudaProverComponents> {
    cuda_worker_builder_with_machine(scope, RiscvAir::machine()).await
}

pub async fn core_prover_and_verifier(
    scope: TaskScope,
    machine: Machine<SP1Field, RiscvAir<SP1Field>>,
) -> (
    CudaShardProver<SP1GlobalContext, CudaProverCoreComponents>,
    MachineVerifier<SP1GlobalContext, InnerSC<RiscvAir<SP1Field>>>,
) {
    let opts = local_gpu_opts();
    let num_elts =
        opts.sharding_threshold.element_threshold as usize + (1 << CORE_LOG_STACKING_HEIGHT);
    let core_verifier = SP1CudaProverComponents::core_verifier(machine);
    (
        new_cuda_prover(&core_verifier, num_elts, 4, opts.recompute_gkr_trace, scope).await,
        core_verifier,
    )
}

pub async fn recursion_prover_and_verifier(
    scope: TaskScope,
) -> (
    CudaShardProver<SP1GlobalContext, CudaProverRecursionComponents>,
    MachineVerifier<SP1GlobalContext, InnerSC<CompressAir<SP1Field>>>,
) {
    let recursion_verifier = SP1CudaProverComponents::compress_verifier();
    (
        new_cuda_prover(&recursion_verifier, RECURSION_TRACE_ALLOCATION, 4, false, scope).await,
        recursion_verifier,
    )
}

/// Same as [`cuda_worker_builder`] but with a custom machine.
pub async fn cuda_worker_builder_with_machine(
    scope: TaskScope,
    machine: Machine<SP1Field, RiscvAir<SP1Field>>,
) -> SP1WorkerBuilder<SP1CudaProverComponents> {
    #[cfg(feature = "rocm")]
    let prover_permits = {
        let permits = std::env::var("SP1_ROCM_PROVER_PERMITS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|permits| *permits > 0)
            .unwrap_or(1);
        tracing::info!(permits, "ROCm prover work permits");
        ProverSemaphore::new(permits)
    };

    // Keep CUDA's existing single-permit behavior unchanged.
    #[cfg(not(feature = "rocm"))]
    let prover_permits = ProverSemaphore::new(1);

    // Get the core options.
    let opts = local_gpu_opts();

    let core_prover = Arc::new(core_prover_and_verifier(scope.clone(), machine.clone()).await.0);

    // TODO: tune this more precisely and make it a constant.
    let recursion_prover = Arc::new(recursion_prover_and_verifier(scope.clone()).await.0);

    let shrink_verifier = SP1CudaProverComponents::shrink_verifier();
    let shrink_prover = Arc::new(
        new_cuda_prover(&shrink_verifier, SHRINK_TRACE_ALLOCATION, 4, false, scope.clone()).await,
    );

    let wrap_verifier = SP1CudaProverComponents::wrap_verifier();
    let wrap_prover = Arc::new(
        new_cuda_prover(&wrap_verifier, WRAP_TRACE_ALLOCATION, 4, false, scope.clone()).await,
    );

    let base_builder = SP1WorkerBuilder::new_with_machine(machine)
        .with_core_opts(opts)
        .with_core_air_prover(core_prover, prover_permits.clone())
        .with_compress_air_prover(recursion_prover, prover_permits.clone())
        .with_shrink_air_prover(shrink_prover, prover_permits.clone())
        .with_wrap_air_prover(ReadyWrapProverBuilder::new(wrap_prover), prover_permits);

    #[cfg(feature = "experimental")]
    {
        if cfg!(feature = "mprotect") {
            return base_builder.without_vk_verification();
        }
        if let Ok(setting) = std::env::var("WITHOUT_VK_VERIFICATION") {
            if setting == "1" || setting == "true" {
                return base_builder.without_vk_verification();
            }
        }
    }
    base_builder
}
