//! # SP1 ROCm Prover
//!
//! A prover that uses the ROCm to execute and prove programs.

/// The builder for the ROCm prover.
pub mod builder;
/// The ROCm prove request type.
pub mod prove;

use crate::blocking::{prover::BaseProveRequest, Prover};

use prove::RocmProveRequest;
use sp1_core_machine::io::SP1Stdin;
use sp1_primitives::Elf;
use sp1_prover::worker::{SP1LightNode, SP1NodeCore};
use sp1_rocm::{RocmClientError, RocmProver as RocmProverImpl, RocmProvingKey};

/// A prover that uses the CPU for execution and the ROCm for proving.
#[derive(Clone)]
pub struct RocmProver {
    pub(crate) node: SP1LightNode,
    pub(crate) prover: RocmProverImpl,
}

impl Prover for RocmProver {
    type ProvingKey = RocmProvingKey;
    type Error = RocmClientError;
    type ProveRequest<'a> = RocmProveRequest<'a>;

    fn inner(&self) -> &SP1NodeCore {
        self.node.inner()
    }

    fn setup(&self, elf: Elf) -> Result<Self::ProvingKey, Self::Error> {
        let machine = self.node.inner().machine().clone();
        crate::blocking::block_on(self.prover.setup_with_machine(elf, machine))
    }

    fn prove<'a>(&'a self, pk: &'a Self::ProvingKey, stdin: SP1Stdin) -> Self::ProveRequest<'a> {
        RocmProveRequest { base: BaseProveRequest::new(self, pk, stdin) }
    }
}
