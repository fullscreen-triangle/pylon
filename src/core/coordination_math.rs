//! Mathematical foundations for coordination calculations

use crate::types::{PrecisionLevel, CoordinationDomain};

/// Mathematical constants for coordination
pub mod constants {
    /// Maximum theoretical precision (femtoseconds)
    pub const MAX_PRECISION: f64 = 1e-15;
    
    /// Default coordination threshold
    pub const DEFAULT_THRESHOLD: f64 = 1e-9;
    
    /// Golden ratio for precision optimization
    pub const PHI: f64 = 1.618033988749;
}

/// Mathematical utilities for coordination
pub struct CoordinationMath;

impl CoordinationMath {
    /// Calculate precision enhancement factor
    pub fn calculate_enhancement_factor(
        current_precision: f64,
        target_precision: f64,
        domain: CoordinationDomain,
    ) -> f64 {
        let domain_factor = match domain {
            CoordinationDomain::Temporal => 1.0,
            CoordinationDomain::Spatial => 0.9,
            CoordinationDomain::Individual => 0.8,
            CoordinationDomain::Economic => 0.7,
        };
        
        (target_precision / current_precision.max(1e-15)) * domain_factor
    }
    
    /// Calculate unified precision from domain precisions
    pub fn calculate_unified_precision(domain_precisions: &[f64]) -> f64 {
        if domain_precisions.is_empty() {
            return 0.0;
        }
        
        // Harmonic mean for precision combination
        let sum_reciprocals: f64 = domain_precisions.iter()
            .map(|p| 1.0 / p.max(1e-15))
            .sum();
        
        domain_precisions.len() as f64 / sum_reciprocals
    }
    
    /// Convert precision level to numeric value
    pub fn precision_level_to_value(level: PrecisionLevel) -> f64 {
        level.as_seconds()
    }
}
