//! Core types for Pylon unified coordination framework

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier type used throughout Pylon
pub type PylonId = Uuid;

/// Temporal coordinate with quantum-level precision
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TemporalCoordinate {
    /// Nanoseconds since UNIX epoch
    pub nanos_since_epoch: u128,
    /// Quantum-level precision enhancement
    pub quantum_precision: f64,
    /// Atomic clock reference offset
    pub atomic_reference_offset: i64,
}

impl TemporalCoordinate {
    /// Create new temporal coordinate from current time
    pub fn now() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        
        Self {
            nanos_since_epoch: now.as_nanos(),
            quantum_precision: 0.0,
            atomic_reference_offset: 0,
        }
    }
    
    /// Create temporal coordinate with quantum precision
    pub fn with_quantum_precision(mut self, precision: f64) -> Self {
        self.quantum_precision = precision;
        self
    }
}

/// Spatial coordinate with consciousness integration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialCoordinate {
    /// Quantum-precise position coordinates
    pub position: [f64; 3],
    /// Consciousness field measurement
    pub consciousness_metric: ConsciousnessMetric,
    /// Local gravitational field variations
    pub gravitational_field: GravitationalField,
    /// Quantum entanglement state distribution
    pub quantum_entanglement: QuantumEntanglementState,
}

/// Consciousness quantification metric
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsciousnessMetric {
    /// Integrated information theory measurement
    pub phi_value: f64,
    /// Consciousness coherence level
    pub coherence_level: f64,
    /// Biological maxwell demon activity
    pub bmd_activity: f64,
}

/// Gravitational field measurement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GravitationalField {
    /// Field strength vector
    pub field_strength: [f64; 3],
    /// Gravitational potential
    pub potential: f64,
    /// Relativistic corrections
    pub relativistic_corrections: [f64; 3],
}

/// Quantum entanglement state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantumEntanglementState {
    /// Entanglement density
    pub entanglement_density: f64,
    /// Coherence time
    pub coherence_time: Duration,
    /// Bell state correlations
    pub bell_correlations: [f64; 4],
}

/// Domain type for precision-by-difference calculations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoordinationDomain {
    /// Temporal network coordination
    Temporal,
    /// Spatial navigation coordination
    Spatial,
    /// Individual experience coordination
    Individual,
    /// Economic value coordination
    Economic,
}

/// Precision vector for unified coordination
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrecisionVector<T> 
where 
    T: Debug + Clone + PartialEq + Serialize,
{
    /// Reference value for precision calculation
    pub reference_value: T,
    /// Local measurement value
    pub local_value: T,
    /// Calculated precision enhancement
    pub precision_delta: f64,
    /// Coordination domain
    pub domain: CoordinationDomain,
    /// Temporal coordinate of measurement
    pub temporal_coordinate: TemporalCoordinate,
}

impl<T> PrecisionVector<T>
where
    T: Debug + Clone + PartialEq + Serialize,
{
    /// Create new precision vector
    pub fn new(
        reference_value: T,
        local_value: T,
        domain: CoordinationDomain,
    ) -> Self {
        Self {
            reference_value,
            local_value,
            precision_delta: 0.0,
            domain,
            temporal_coordinate: TemporalCoordinate::now(),
        }
    }
    
    /// Calculate precision enhancement (to be implemented by specific types)
    pub fn calculate_precision_delta(&mut self, calculator: impl Fn(&T, &T) -> f64) {
        self.precision_delta = calculator(&self.reference_value, &self.local_value);
    }
}

/// Unified coordination structure across multiple domains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedCoordination {
    /// Temporal precision vector
    pub temporal_precision: PrecisionVector<TemporalCoordinate>,
    /// Spatial precision vector
    pub spatial_precision: PrecisionVector<SpatialCoordinate>,
    /// Individual precision vector (will be defined in cable-individual)
    pub individual_precision: Option<PrecisionVector<IndividualState>>,
    /// Economic precision vector (will be defined in temporal-economic)
    pub economic_precision: Option<PrecisionVector<EconomicState>>,
    /// Unified precision result
    pub unified_precision: f64,
}

/// Placeholder for individual state (to be fully defined in cable-individual)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndividualState {
    pub experience_metric: f64,
    pub consciousness_state: ConsciousnessMetric,
    pub optimization_level: f64,
}

/// Placeholder for economic state (to be fully defined in temporal-economic)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EconomicState {
    pub value_reference: f64,
    pub local_credit: f64,
    pub economic_noise: f64,
}

impl UnifiedCoordination {
    /// Calculate unified precision across all domains
    pub fn calculate_unified_precision(&mut self) -> f64 {
        let mut total_precision = 0.0;
        let mut domain_count = 0;
        
        // Temporal precision
        total_precision += self.temporal_precision.precision_delta;
        domain_count += 1;
        
        // Spatial precision
        total_precision += self.spatial_precision.precision_delta;
        domain_count += 1;
        
        // Individual precision (if available)
        if let Some(ref individual) = self.individual_precision {
            total_precision += individual.precision_delta;
            domain_count += 1;
        }
        
        // Economic precision (if available)
        if let Some(ref economic) = self.economic_precision {
            total_precision += economic.precision_delta;
            domain_count += 1;
        }
        
        self.unified_precision = if domain_count > 0 {
            total_precision / domain_count as f64
        } else {
            0.0
        };
        
        self.unified_precision
    }
}

/// Fragment for distributed coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationFragment {
    /// Unique fragment identifier
    pub fragment_id: PylonId,
    /// Temporal window for fragment validity
    pub temporal_window: TemporalWindow,
    /// Spatial coordinates for fragment
    pub spatial_coordinates: SpatialCoordinate,
    /// Fragment data payload
    pub fragment_data: FragmentData,
    /// Reconstruction key for fragment assembly
    pub reconstruction_key: ReconstructionKey,
    /// Coherence validation signature
    pub coherence_signature: CoherenceSignature,
}

/// Temporal window for fragment validity
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalWindow {
    /// Start time of validity window
    pub start_time: TemporalCoordinate,
    /// End time of validity window
    pub end_time: TemporalCoordinate,
    /// Precision requirements for window
    pub precision_requirements: f64,
}

/// Fragment data payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentData {
    /// Raw fragment bytes
    pub data: Vec<u8>,
    /// Fragment type identifier
    pub fragment_type: FragmentType,
    /// Coordination domain
    pub domain: CoordinationDomain,
}

/// Type of coordination fragment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FragmentType {
    /// Network temporal coordination fragment
    TemporalCoordination,
    /// Spatial navigation fragment
    SpatialNavigation,
    /// Individual experience fragment
    IndividualOptimization,
    /// Economic transaction fragment
    EconomicTransaction,
    /// Security asset fragment
    SecurityAsset,
}

/// Reconstruction key for fragment assembly
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReconstructionKey {
    /// Key data for reconstruction
    pub key_data: Vec<u8>,
    /// Required fragments for reconstruction
    pub required_fragments: Vec<PylonId>,
    /// Reconstruction algorithm identifier
    pub algorithm_id: u32,
}

/// Coherence validation signature
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoherenceSignature {
    /// Signature data
    pub signature: Vec<u8>,
    /// Validation timestamp
    pub timestamp: TemporalCoordinate,
    /// Coherence level achieved
    pub coherence_level: f64,
}

/// Measurement precision levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionLevel {
    /// Standard precision (microsecond level)
    Standard,
    /// High precision (nanosecond level)
    High,
    /// Quantum precision (sub-nanosecond level)
    Quantum,
    /// Maximum precision (theoretical limits)
    Maximum,
}

impl PrecisionLevel {
    /// Get precision value in seconds
    pub fn as_seconds(self) -> f64 {
        match self {
            PrecisionLevel::Standard => 1e-6,     // 1 microsecond
            PrecisionLevel::High => 1e-9,         // 1 nanosecond
            PrecisionLevel::Quantum => 1e-12,     // 1 picosecond
            PrecisionLevel::Maximum => 1e-15,     // 1 femtosecond
        }
    }
}

/// Network node identifier and metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkNode {
    /// Unique node identifier
    pub node_id: PylonId,
    /// Node network address
    pub address: String,
    /// Node capabilities
    pub capabilities: NodeCapabilities,
    /// Current coordination status
    pub status: NodeStatus,
    /// Performance metrics
    pub metrics: NodeMetrics,
}

/// Node capabilities
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeCapabilities {
    /// Supported precision levels
    pub precision_levels: Vec<PrecisionLevel>,
    /// Supported coordination domains
    pub coordination_domains: Vec<CoordinationDomain>,
    /// Algorithm suite support
    pub algorithm_suites: Vec<String>,
    /// Maximum fragment throughput
    pub max_throughput: u64,
}

/// Node operational status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Node is online and operational
    Online,
    /// Node is synchronizing
    Synchronizing,
    /// Node is offline
    Offline,
    /// Node has errors
    Error,
}

/// Node performance metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeMetrics {
    /// Current latency in nanoseconds
    pub latency_ns: u64,
    /// Precision accuracy achieved
    pub precision_accuracy: f64,
    /// Throughput in fragments per second
    pub throughput_fps: f64,
    /// Error rate
    pub error_rate: f64,
    /// Last update timestamp
    pub last_update: TemporalCoordinate,
}

/// Coordination request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationRequest {
    /// Request identifier
    pub request_id: PylonId,
    /// Requesting node
    pub requesting_node: PylonId,
    /// Coordination domains required
    pub domains: Vec<CoordinationDomain>,
    /// Required precision level
    pub precision_level: PrecisionLevel,
    /// Request payload
    pub payload: CoordinationPayload,
    /// Request timestamp
    pub timestamp: TemporalCoordinate,
}

/// Coordination response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationResponse {
    /// Response identifier
    pub response_id: PylonId,
    /// Original request identifier
    pub request_id: PylonId,
    /// Responding node
    pub responding_node: PylonId,
    /// Coordination result
    pub result: CoordinationResult,
    /// Response timestamp
    pub timestamp: TemporalCoordinate,
}

/// Coordination payload data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationPayload {
    /// Temporal synchronization request
    TemporalSync {
        target_precision: f64,
        reference_time: TemporalCoordinate,
    },
    /// Spatial navigation request
    SpatialNavigation {
        destination: SpatialCoordinate,
        optimization_level: f64,
    },
    /// Individual optimization request
    IndividualOptimization {
        target_experience: f64,
        optimization_parameters: HashMap<String, f64>,
    },
    /// Economic transaction request
    EconomicTransaction {
        transaction_type: String,
        value_amount: f64,
    },
}

/// Coordination operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationResult {
    /// Successful coordination
    Success {
        unified_precision: UnifiedCoordination,
        fragments: Vec<CoordinationFragment>,
    },
    /// Partial coordination success
    Partial {
        completed_domains: Vec<CoordinationDomain>,
        failed_domains: Vec<CoordinationDomain>,
        precision_achieved: f64,
    },
    /// Coordination failure
    Failure {
        error_code: u32,
        error_message: String,
        failed_domains: Vec<CoordinationDomain>,
    },
}
