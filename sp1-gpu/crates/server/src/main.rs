#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use clap::Parser;
use server::Server;
use sp1_cuda::client as cuda_client;
use sp1_gpu_cudart::{initialize_device_and_ntt, run_in_place, shutdown_slab_cache, RocmAllocator};
use std::path::PathBuf;

mod server;

#[derive(Debug, Parser)]
struct Args {
    #[clap(long)]
    version: bool,
}

#[cfg(all(feature = "cuda", feature = "rocm"))]
compile_error!("features `cuda` and `rocm` are mutually exclusive");

#[allow(clippy::print_stdout)]
fn main() {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    if args.version {
        println!("{}", sp1_primitives::SP1_CRATE_VERSION);
        return;
    }

    let is_rocm = cfg!(feature = "rocm");
    let device_env = if is_rocm {
        std::env::var("HIP_VISIBLE_DEVICES")
            .ok()
            .or_else(|| std::env::var("ROCR_VISIBLE_DEVICES").ok())
            .expect("ROCR_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES must be set")
    } else {
        std::env::var("CUDA_VISIBLE_DEVICES").expect("CUDA_VISIBLE_DEVICES must be set")
    };

    let device_id = device_env.parse().expect("Expected only one GPU device as a u32");
    let socket_path = if is_rocm {
        PathBuf::from(format!("/tmp/sp1-rocm-{device_id}.sock"))
    } else {
        cuda_client::socket_path(device_id)
    };
    let backend_name = if is_rocm { "sp1-rocm-server" } else { "sp1-gpu-server" };
    if is_rocm {
        eprintln!("ROCm allocator: {:?}", RocmAllocator::selected());
    }

    // The visibility environment selects the physical GPU. Inside this process
    // the selected GPU is logical device zero.
    initialize_device_and_ntt(0).expect("failed to initialize GPU and sppark NTT parameters");

    let server = Server { device_id, socket_path, backend_name };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to create Tokio runtime");
    runtime.block_on(async move {
        if let Err(e) = run_in_place(|scope| server.run(scope)).await.await {
            eprintln!("Error running server: {e}");
        }
    });
    shutdown_slab_cache();
}
