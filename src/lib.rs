//! # Pylon: Unified Spatio-Temporal Coordination Framework
//!
//! A distributed coordination infrastructure implementing unified spatio-temporal 
//! precision-by-difference calculations across temporal network synchronization, 
//! autonomous spatial navigation, and individual experience optimization domains.

pub mod core;
pub mod coordination;
pub mod cables;
pub mod algorithms;
pub mod temporal_economic;
pub mod security_assets;
pub mod types;
pub mod errors;
pub mod config;

// Re-export core types for convenience
pub use crate::{
    core::*,
    coordination::*,
    types::*,
    errors::*,
    config::*,
};

// Main Pylon coordinator
pub use coordination::PylonCoordinator;

/// Pylon framework version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration for development
pub fn default_config() -> PylonConfig {
    PylonConfig::default()
}

/// Initialize Pylon with custom configuration
pub async fn initialize(config: PylonConfig) -> Result<PylonCoordinator, PylonError> {
    PylonCoordinator::new(config).await
}

/// Initialize Pylon with default configuration
pub async fn initialize_default() -> Result<PylonCoordinator, PylonError> {
    initialize(default_config()).await
}
