//! # SP1 ROCm Prover
//!
//! A prover that uses the ROCm to execute and prove programs.

/// The builder for the ROCm prover.
pub mod builder;
/// The ROCm prove request type.
pub mod prove;

use crate::{
    prover::{BaseProveRequest, Prover, SendFutureResult},
    ProvingKey,
};

use prove::RocmProveRequest;
use sp1_core_machine::io::SP1Stdin;
use sp1_primitives::Elf;
use sp1_prover::{
    worker::{SP1LightNode, SP1NodeCore},
    SP1VerifyingKey,
};
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

    fn setup(&self, elf: Elf) -> impl SendFutureResult<Self::ProvingKey, Self::Error> {
        let machine = self.node.inner().machine().clone();
        async move { self.prover.setup_with_machine(elf, machine).await }
    }

    fn prove<'a>(&'a self, pk: &'a Self::ProvingKey, stdin: SP1Stdin) -> Self::ProveRequest<'a> {
        RocmProveRequest { base: BaseProveRequest::new(self, pk, stdin) }
    }
}

impl ProvingKey for RocmProvingKey {
    fn elf(&self) -> &Elf {
        self.elf()
    }

    fn verifying_key(&self) -> &SP1VerifyingKey {
        self.verifying_key()
    }
}
