use sp1_sdk::{include_elf, Elf, ProveRequest, Prover, ProverClient, SP1Stdin};
use std::{
    env,
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

const ELF: Elf = include_elf!("fibonacci-program");

struct Args {
    implementation: String,
    warmups: usize,
    iterations: usize,
    csv: PathBuf,
}

fn parse_args() -> Args {
    let mut implementation = None;
    let mut warmups = 3;
    let mut iterations = 10;
    let mut csv = None;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("missing value for {arg}"));
        match arg.as_str() {
            "--implementation" => implementation = Some(value()),
            "--warmups" => {
                warmups = value().parse().expect("--warmups must be a positive integer")
            }
            "--iterations" => {
                iterations = value().parse().expect("--iterations must be a positive integer")
            }
            "--csv" => csv = Some(PathBuf::from(value())),
            _ => panic!("unknown argument: {arg}"),
        }
    }

    assert!(warmups > 0, "--warmups must be greater than zero");
    assert!(iterations > 0, "--iterations must be greater than zero");
    Args {
        implementation: implementation.expect("--implementation is required"),
        warmups,
        iterations,
        csv: csv.expect("--csv is required"),
    }
}

fn open_csv(path: &Path) -> BufWriter<File> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("failed to create the CSV directory");
    }
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .expect("failed to open the CSV output");
    let mut csv = BufWriter::new(file);
    writeln!(csv, "implementation,phase,iteration,prove_seconds,verified")
        .expect("failed to write the CSV header");
    csv.flush().expect("failed to flush the CSV header");
    csv
}

#[tokio::main]
async fn main() {
    sp1_sdk::utils::setup_logger();
    let args = parse_args();
    assert!(
        matches!(args.implementation.as_str(), "arena" | "async"),
        "--implementation must be `arena` or `async`"
    );
    let selected = env::var("SP1_ROCM_ALLOCATOR").unwrap_or_else(|_| "arena".to_string());
    assert_eq!(
        args.implementation, selected,
        "--implementation must match SP1_ROCM_ALLOCATOR"
    );
    let external_server = env::var_os("SP1_ROCM_EXTERNAL_SERVER").is_some();
    assert_eq!(
        Path::new("/tmp/sp1-rocm-0.sock").exists(),
        external_server,
        "the ROCm socket must be absent for auto-start and present for an external server"
    );
    let mut csv = open_csv(&args.csv);

    let client = ProverClient::builder().rocm().with_device_id(0).build().await;
    let pk = client.setup(ELF).await.expect("setup failed");

    for (phase, count) in [("warmup", args.warmups), ("measured", args.iterations)] {
        for iteration in 1..=count {
            let mut stdin = SP1Stdin::new();
            stdin.write(&500_u32);

            println!("BENCH phase={phase} iteration={iteration} start");
            std::io::stdout().flush().expect("failed to flush the start marker");
            let start = Instant::now();
            let proof = client
                .prove(&pk, stdin)
                .compressed()
                .await
                .expect("compressed proof failed");
            let prove_seconds = start.elapsed().as_secs_f64();
            println!("BENCH phase={phase} iteration={iteration} end");
            std::io::stdout().flush().expect("failed to flush the end marker");

            client
                .verify(&proof, pk.verifying_key(), None)
                .expect("proof verification failed");
            let mut public_values = proof.public_values.clone();
            let n = public_values.read::<u32>();
            let a = public_values.read::<u32>();
            let b = public_values.read::<u32>();
            assert_eq!((n, a, b), (500, 1268, 1926), "unexpected Fibonacci public values");

            writeln!(
                csv,
                "{},{phase},{iteration},{prove_seconds:.9},true",
                args.implementation
            )
            .expect("failed to write a CSV row");
            csv.flush().expect("failed to flush a CSV row");
        }
    }
}
