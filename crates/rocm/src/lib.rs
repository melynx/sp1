/// The shared API between the client and server.
pub mod api;

/// The client that interacts with the ROCm server.
pub mod client;

/// The proving key type, which is a "remote" reference to a key held by the ROCm server.
pub mod pk;

/// The server startup logic.
mod server;

mod error;
pub use error::RocmClientError;

pub use pk::RocmProvingKey;
use sp1_core_executor::SP1Context;
use sp1_core_machine::io::SP1Stdin;
use sp1_core_machine::riscv::RiscvAir;
use sp1_hypercube::Machine;
use sp1_primitives::{Elf, SP1Field};
use sp1_prover::worker::ProofFromNetwork;
use sp1_prover_types::network_base_types::ProofMode;

use crate::client::RocmClient;

#[derive(Clone)]
pub struct RocmProver {
    client: RocmClient,
}

impl RocmProver {
    /// Create a new prover, using the 0th ROCm device.
    pub async fn new() -> Result<Self, RocmClientError> {
        Ok(Self { client: RocmClient::connect(0).await? })
    }

    /// Create a new prover, using the given ROCm device.
    pub async fn new_with_id(rocm_id: u32) -> Result<Self, RocmClientError> {
        Ok(Self { client: RocmClient::connect(rocm_id).await? })
    }

    /// Setup a new proving key.
    pub async fn setup(&self, elf: Elf) -> Result<RocmProvingKey, RocmClientError> {
        self.setup_with_machine(elf, RiscvAir::machine()).await
    }

    /// Same as [`Self::setup`] but with a custom machine.
    pub async fn setup_with_machine(
        &self,
        elf: Elf,
        machine: Machine<SP1Field, RiscvAir<SP1Field>>,
    ) -> Result<RocmProvingKey, RocmClientError> {
        self.client.setup(elf, machine).await
    }

    pub async fn prove_with_mode(
        &self,
        pk: &RocmProvingKey,
        stdin: SP1Stdin,
        context: SP1Context<'static>,
        mode: ProofMode,
    ) -> Result<ProofFromNetwork, RocmClientError> {
        self.client.prove_with_mode(pk, stdin, context, mode).await
    }
}
