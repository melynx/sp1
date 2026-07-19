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

#[cfg(feature = "rocm")]
use std::{
    fs::{File, OpenOptions},
    os::fd::AsRawFd,
    os::unix::fs::OpenOptionsExt,
};

mod server;

#[derive(Debug, Parser)]
struct Args {
    #[clap(long)]
    version: bool,
}

#[cfg(feature = "rocm")]
fn acquire_server_lock(device_id: u32) -> std::io::Result<Option<File>> {
    let path = format!("/tmp/sp1-rocm-{device_id}.lock");
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)?;
    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if result == 0 {
        return Ok(Some(file));
    }

    let error = std::io::Error::last_os_error();
    if error.kind() == std::io::ErrorKind::WouldBlock {
        Ok(None)
    } else {
        Err(error)
    }
}

#[cfg(feature = "rocm")]
fn configure_rocm_plonk_cache() -> bool {
    fn read_setting(name: &str) -> Option<String> {
        std::env::var_os(name).map(|value| {
            value.into_string().unwrap_or_else(|_| panic!("{name} must contain valid UTF-8"))
        })
    }

    let setting = read_setting("SP1_ROCM_PLONK_CACHE").or_else(|| read_setting("SP1_PLONK_CACHE"));
    let enabled = match setting.as_deref() {
        None | Some("1" | "true" | "on") => true,
        Some("0" | "false" | "off") => false,
        Some(value) => panic!(
            "invalid ROCm Plonk cache setting `{value}`; expected `1`, `0`, `true`, `false`, \
             `on`, or `off`"
        ),
    };

    // This runs at process start, before the Tokio runtime or GPU worker
    // threads exist. The Go prover reads this internal setting later.
    unsafe {
        std::env::set_var("SP1_PLONK_CACHE", if enabled { "1" } else { "0" });
    }
    enabled
}

#[cfg(all(feature = "cuda", feature = "rocm"))]
compile_error!("features `cuda` and `rocm` are mutually exclusive");

#[allow(clippy::print_stdout)]
fn main() {
    #[cfg(feature = "rocm")]
    let plonk_cache_enabled = configure_rocm_plonk_cache();

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

    #[cfg(feature = "rocm")]
    let _server_lock = match acquire_server_lock(device_id).expect("failed to lock ROCm server") {
        Some(lock) => lock,
        None => {
            eprintln!("A ROCm server is already starting or running for device {device_id}");
            return;
        }
    };

    let socket_path = if is_rocm {
        PathBuf::from(format!("/tmp/sp1-rocm-{device_id}.sock"))
    } else {
        cuda_client::socket_path(device_id)
    };
    let backend_name = if is_rocm { "sp1-rocm-server" } else { "sp1-gpu-server" };
    if is_rocm {
        eprintln!("ROCm allocator: {:?}", RocmAllocator::selected());
    }
    #[cfg(feature = "rocm")]
    eprintln!(
        "ROCm Plonk circuit cache: {}",
        if plonk_cache_enabled { "enabled" } else { "disabled" }
    );

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
