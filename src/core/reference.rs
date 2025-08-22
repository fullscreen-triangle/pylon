//! Reference value management

use crate::types::{TemporalCoordinate, CoordinationDomain};
use crate::errors::PylonError;

/// Reference value manager
pub struct ReferenceManager {
    temporal_reference: TemporalCoordinate,
}

impl ReferenceManager {
    /// Create new reference manager
    pub fn new() -> Self {
        Self {
            temporal_reference: TemporalCoordinate::now(),
        }
    }

    /// Get reference value for domain
    pub fn get_reference(&self, _domain: CoordinationDomain) -> Result<TemporalCoordinate, PylonError> {
        Ok(self.temporal_reference)
    }

    /// Update reference value
    pub fn update_reference(&mut self, _domain: CoordinationDomain, reference: TemporalCoordinate) {
        self.temporal_reference = reference;
    }
}
