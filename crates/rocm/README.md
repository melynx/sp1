# sp1-rocm

ROCm integration for SP1 GPU proving.

Provides the bridge between SP1's Rust prover and the ROCm-based GPU acceleration, enabling high-performance proof generation.

## Device memory

The ROCm prover supports GPUs with at least 16 GiB of device memory. GPUs that
report 16 to 19 GiB automatically use `1 << 23` core shards and synchronous
device allocation. They allocate buffers of 1280 MiB or larger with HIP managed
memory. HIP managed memory can move those large buffers between device memory
and host memory, so the host must have enough free memory.

Set `SP1_ROCM_MANAGED_THRESHOLD_MB` to a positive MiB value to override the managed-memory threshold. Set it to `0` to disable managed memory. GPUs that report at least 20 GiB keep the standard asynchronous allocator unless this variable overrides it.

Set `SP1_ROCM_LOW_MEMORY_LOG2_SHARD_SIZE` to `22`, `23`, or `24` to override the
automatic setting. These settings use core shard sizes of `1 << 22`, `1 << 23`,
or `1 << 24` and core sharding element thresholds of `1 << 26`, `1 << 27`, or
`1 << 28`.

---

Part of [SP1](https://github.com/succinctlabs/sp1), a performant zkVM.
