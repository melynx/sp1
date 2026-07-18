// //! # ROCm Proving
// //!
// //! This module provides a builder for proving a program on the ROCm.

use std::{
    future::{Future, IntoFuture},
    pin::Pin,
};

use sp1_rocm::RocmClientError;

use super::RocmProver;
use crate::{
    prover::{BaseProveRequest, ProveRequest},
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
}

impl<'a> IntoFuture for RocmProveRequest<'a> {
    type Output = Result<SP1ProofWithPublicValues, RocmClientError>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        let BaseProveRequest { prover, pk, stdin, mode, mut context_builder } = self.base;

        let context = context_builder.build();
        Box::pin(async move {
            tracing::info!(mode = ?mode, "starting proof generation");
            Ok(prover.prover.prove_with_mode(pk, stdin, context, proof_mode(mode)).await?.into())
        })
    }
}
