//! # ROCm Prover Builder
//!
//! This module provides a builder for the [`RocmProver`].

use super::RocmProver;
use sp1_core_executor::SP1CoreOpts;
use sp1_core_machine::riscv::RiscvAir;
use sp1_hypercube::Machine;
use sp1_primitives::SP1Field;
use sp1_prover::worker::SP1LightNode;
use sp1_rocm::RocmProver as RocmProverImpl;

use crate::blocking::block_on;

/// A builder for the [`RocmProver`].
///
/// The builder is used to configure the [`RocmProver`] before it is built.
#[derive(Debug)]
pub struct RocmProverBuilder {
    rocm_device_id: Option<u32>,
    /// Optional core options to configure the underlying CPU prover.
    core_opts: Option<SP1CoreOpts>,
    machine: Machine<SP1Field, RiscvAir<SP1Field>>,
}

impl Default for RocmProverBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RocmProverBuilder {
    /// Creates a new [`RocmProverBuilder`] with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::new_with_machine(RiscvAir::machine())
    }

    /// Creates a new [`RocmProverBuilder`] with a given machine.
    #[must_use]
    pub const fn new_with_machine(machine: Machine<SP1Field, RiscvAir<SP1Field>>) -> Self {
        Self { rocm_device_id: None, core_opts: None, machine }
    }

    /// Sets the ROCm device id.
    ///
    /// # Details
    /// Run the ROCm prover with the provided device id, all operations will be performed on this
    /// device index.
    ///
    /// # Example
    /// ```rust,no_run
    /// use sp1_sdk::ProverClient;
    ///
    /// let prover = ProverClient::builder().rocm().with_device_id(0).build();
    /// ```
    #[must_use]
    pub fn with_device_id(mut self, id: u32) -> Self {
        self.rocm_device_id = Some(id);
        self
    }

    /// Sets the core options for the underlying CPU prover.
    ///
    /// # Example
    /// ```rust,no_run
    /// use sp1_core_executor::SP1CoreOpts;
    /// use sp1_sdk::ProverClient;
    ///
    /// let mut opts = SP1CoreOpts::default();
    /// opts.shard_size = 500_000;
    /// let prover = ProverClient::builder().rocm().core_opts(opts).build();
    /// ```
    #[must_use]
    pub fn core_opts(mut self, opts: SP1CoreOpts) -> Self {
        self.core_opts = Some(opts);
        self
    }

    /// Sets the core options for the underlying CPU prover (alias for `core_opts`).
    ///
    /// # Example
    /// ```rust,no_run
    /// use sp1_core_executor::SP1CoreOpts;
    /// use sp1_sdk::ProverClient;
    ///
    /// let mut opts = SP1CoreOpts::default();
    /// opts.shard_size = 500_000;
    /// let prover = ProverClient::builder().rocm().with_opts(opts).build();
    /// ```
    #[must_use]
    pub fn with_opts(self, opts: SP1CoreOpts) -> Self {
        self.core_opts(opts)
    }

    /// Builds a [`RocmProver`].
    ///
    /// # Details
    /// This method will build a [`RocmProver`] with the given parameters.
    ///
    /// # Example
    /// ```rust,no_run
    /// use sp1_sdk::ProverClient;
    ///
    /// let prover = ProverClient::builder().rocm().build();
    /// ```
    #[must_use]
    pub fn build(self) -> RocmProver {
        tracing::info!("initializing rocm prover");
        let node = block_on(SP1LightNode::with_opts_and_machine(
            self.machine,
            self.core_opts.unwrap_or_default(),
        ));
        let rocm_prover = match self.rocm_device_id {
            Some(id) => crate::blocking::block_on(RocmProverImpl::new_with_id(id)),
            None => crate::blocking::block_on(RocmProverImpl::new()),
        };

        RocmProver { node, prover: rocm_prover.expect("Failed to create the ROCm prover impl") }
    }
}
