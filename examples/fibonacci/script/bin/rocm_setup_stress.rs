use sp1_sdk::{include_elf, Elf, Prover, ProverClient};
use std::{env, path::Path, time::Duration};

const ELF: Elf = include_elf!("fibonacci-program");

#[tokio::main]
async fn main() {
    sp1_sdk::utils::setup_logger();
    let iterations = env::args()
        .nth(1)
        .expect("usage: rocm_setup_stress ITERATIONS")
        .parse::<usize>()
        .expect("ITERATIONS must be a positive integer");
    assert!(iterations > 0, "ITERATIONS must be greater than zero");
    assert!(
        Path::new("/tmp/sp1-rocm-0.sock").exists(),
        "start the explicit fresh ROCm server before this test"
    );

    for iteration in 1..=iterations {
        let client = ProverClient::builder().rocm().with_device_id(0).build().await;
        let proving_key = client.setup(ELF).await.expect("setup failed");
        drop(proving_key);
        drop(client);
        tokio::time::sleep(Duration::from_millis(50)).await;
        println!("SETUP {iteration}/{iterations} PASSED");
    }
}
