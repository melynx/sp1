// //! # ROCm Proving
// //!
// //! This module provides a builder for proving a program on the ROCm.
use sp1_rocm::RocmClientError;

use super::RocmProver;
use crate::{
    blocking::{
        block_on,
        prover::{BaseProveRequest, ProveRequest},
    },
    utils::proof_mode,
    SP1ProofWithPublicValues,
};

/// A builder for proving a program on the ROCm.
///
/// This builder provides a typed interface for configuring the SP1 RISC-V prover. The builder is
/// used for only the [`crate::rocm::RocmProver`] client type.
pub struct RocmProveRequest<'a> {
    pub(crate) base: BaseProveRequest<'a, RocmProver>,
}

impl<'a> ProveRequest<'a, RocmProver> for RocmProveRequest<'a> {
    fn base(&mut self) -> &mut BaseProveRequest<'a, RocmProver> {
        &mut self.base
    }

    fn run(self) -> Result<SP1ProofWithPublicValues, RocmClientError> {
        let BaseProveRequest { prover, pk, stdin, mode, mut context_builder } = self.base;
        tracing::info!(mode = ?mode, "starting proof generation");
        let context = context_builder.build();
        Ok(block_on(prover.prover.prove_with_mode(pk, stdin, context, proof_mode(mode)))?.into())
    }
}
