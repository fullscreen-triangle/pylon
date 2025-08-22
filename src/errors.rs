//! Error types for the Pylon coordination framework

use std::fmt;
use std::error::Error as StdError;

use serde::{Deserialize, Serialize};

use crate::types::{CoordinationDomain, PylonId};

/// Main error type for Pylon operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PylonError {
    /// Configuration errors
    Configuration(ConfigurationError),
    /// Coordination errors
    Coordination(CoordinationError),
    /// Network communication errors
    Network(NetworkError),
    /// Algorithm suite errors
    Algorithm(AlgorithmError),
    /// Security asset errors
    Security(SecurityError),
    /// Temporal coordination errors
    Temporal(TemporalError),
    /// Spatial coordination errors
    Spatial(SpatialError),
    /// Individual optimization errors
    Individual(IndividualError),
    /// Economic convergence errors
    Economic(EconomicError),
    /// Fragment processing errors
    Fragment(FragmentError),
    /// Precision calculation errors
    Precision(PrecisionError),
    /// Internal system errors
    Internal(String),
}

/// Configuration-related errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigurationError {
    /// Invalid configuration value
    InvalidValue {
        field: String,
        value: String,
        reason: String,
    },
    /// Missing required configuration
    MissingRequired {
        field: String,
    },
    /// Configuration file errors
    FileError {
        path: String,
        error: String,
    },
    /// Configuration parsing errors
    ParseError {
        error: String,
    },
}

/// Coordination operation errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationError {
    /// Precision threshold not met
    PrecisionThresholdNotMet {
        required: f64,
        achieved: f64,
        domain: CoordinationDomain,
    },
    /// Coordination timeout
    Timeout {
        duration_ms: u64,
        operation: String,
    },
    /// Node synchronization failure
    SynchronizationFailure {
        node_id: PylonId,
        reason: String,
    },
    /// Fragment reconstruction failure
    ReconstructionFailure {
        fragment_ids: Vec<PylonId>,
        reason: String,
    },
    /// Unified coordination failure
    UnifiedCoordinationFailure {
        failed_domains: Vec<CoordinationDomain>,
        error_messages: Vec<String>,
    },
}

/// Network communication errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkError {
    /// Connection failure
    ConnectionFailure {
        address: String,
        error: String,
    },
    /// Message transmission failure
    TransmissionFailure {
        message_id: PylonId,
        error: String,
    },
    /// Network timeout
    Timeout {
        operation: String,
        duration_ms: u64,
    },
    /// Protocol errors
    ProtocolError {
        protocol: String,
        error: String,
    },
    /// Node discovery failure
    NodeDiscoveryFailure {
        error: String,
    },
}

/// Algorithm suite errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlgorithmError {
    /// Buhera-East intelligence errors
    BuheraEastError {
        operation: String,
        error: String,
    },
    /// Buhera-North orchestration errors
    BuheraNorthError {
        operation: String,
        error: String,
    },
    /// Bulawayo consciousness errors
    BulawayoError {
        operation: String,
        error: String,
    },
    /// Harare statistical emergence errors
    HarareError {
        operation: String,
        error: String,
    },
    /// Kinshasa semantic computing errors
    KinshasaError {
        operation: String,
        error: String,
    },
    /// Mufakose search algorithm errors
    MufakoseError {
        operation: String,
        error: String,
    },
    /// Self-aware algorithm errors
    SelfAwareError {
        operation: String,
        error: String,
    },
    /// Algorithm suite coordination errors
    CoordinationError {
        suites_involved: Vec<String>,
        error: String,
    },
}

/// Security and cryptographic errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityError {
    /// Environmental state measurement errors
    EnvironmentalMeasurement {
        dimension: String,
        error: String,
    },
    /// MDTEC cryptographic errors
    MDTECCryptographic {
        operation: String,
        error: String,
    },
    /// Currency generation errors
    CurrencyGeneration {
        reason: String,
    },
    /// Payment verification errors
    PaymentVerification {
        reason: String,
    },
    /// Thermodynamic security validation errors
    ThermodynamicSecurity {
        security_level: f64,
        required_level: f64,
    },
    /// Authentication errors
    Authentication {
        error: String,
    },
    /// Authorization errors
    Authorization {
        operation: String,
        required_permissions: Vec<String>,
    },
}

/// Temporal coordination errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalError {
    /// Clock synchronization errors
    ClockSynchronization {
        node_id: PylonId,
        offset_ns: i64,
        max_allowed_ns: i64,
    },
    /// Temporal precision errors
    PrecisionError {
        achieved_precision: f64,
        required_precision: f64,
    },
    /// Temporal window validation errors
    WindowValidation {
        reason: String,
    },
    /// Atomic clock reference errors
    AtomicClockReference {
        error: String,
    },
    /// Temporal fragment coherence errors
    FragmentCoherence {
        fragment_id: PylonId,
        coherence_level: f64,
        required_level: f64,
    },
}

/// Spatial coordination errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpatialError {
    /// Spatial measurement errors
    Measurement {
        coordinate_type: String,
        error: String,
    },
    /// Navigation calculation errors
    NavigationCalculation {
        source: [f64; 3],
        destination: [f64; 3],
        error: String,
    },
    /// Consciousness quantification errors
    ConsciousnessQuantification {
        error: String,
    },
    /// Gravitational field calculation errors
    GravitationalField {
        error: String,
    },
    /// Quantum entanglement measurement errors
    QuantumEntanglement {
        error: String,
    },
    /// Verum integration errors
    VerumIntegration {
        error: String,
    },
}

/// Individual optimization errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndividualError {
    /// Experience measurement errors
    ExperienceMeasurement {
        metric: String,
        error: String,
    },
    /// Consciousness interface errors
    ConsciousnessInterface {
        error: String,
    },
    /// BMD integration errors
    BMDIntegration {
        error: String,
    },
    /// Reality state anchoring errors
    RealityStateAnchoring {
        error: String,
    },
    /// Individual optimization calculation errors
    OptimizationCalculation {
        target_metric: f64,
        achieved_metric: f64,
        error: String,
    },
}

/// Economic convergence errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EconomicError {
    /// Value representation errors
    ValueRepresentation {
        error: String,
    },
    /// Economic fragment processing errors
    FragmentProcessing {
        fragment_id: PylonId,
        error: String,
    },
    /// Transaction coordination errors
    TransactionCoordination {
        transaction_id: PylonId,
        error: String,
    },
    /// Reference currency errors
    ReferenceCurrency {
        error: String,
    },
    /// Temporal-economic convergence errors
    TemporalEconomicConvergence {
        temporal_precision: f64,
        economic_precision: f64,
        error: String,
    },
}

/// Fragment processing errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FragmentError {
    /// Fragment generation errors
    Generation {
        fragment_type: String,
        error: String,
    },
    /// Fragment distribution errors
    Distribution {
        fragment_id: PylonId,
        target_nodes: Vec<PylonId>,
        error: String,
    },
    /// Fragment reconstruction errors
    Reconstruction {
        fragment_ids: Vec<PylonId>,
        error: String,
    },
    /// Fragment validation errors
    Validation {
        fragment_id: PylonId,
        validation_type: String,
        error: String,
    },
    /// Fragment coherence errors
    Coherence {
        fragment_id: PylonId,
        coherence_level: f64,
        required_level: f64,
    },
}

/// Precision calculation errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrecisionError {
    /// Precision calculation failure
    CalculationFailure {
        domain: CoordinationDomain,
        error: String,
    },
    /// Precision threshold validation failure
    ThresholdValidation {
        calculated_precision: f64,
        required_precision: f64,
        domain: CoordinationDomain,
    },
    /// Reference value errors
    ReferenceValue {
        domain: CoordinationDomain,
        error: String,
    },
    /// Local measurement errors
    LocalMeasurement {
        domain: CoordinationDomain,
        error: String,
    },
    /// Precision enhancement errors
    Enhancement {
        domain: CoordinationDomain,
        current_precision: f64,
        target_precision: f64,
        error: String,
    },
}

// Implement Display trait for all error types
impl fmt::Display for PylonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PylonError::Configuration(e) => write!(f, "Configuration error: {}", e),
            PylonError::Coordination(e) => write!(f, "Coordination error: {}", e),
            PylonError::Network(e) => write!(f, "Network error: {}", e),
            PylonError::Algorithm(e) => write!(f, "Algorithm error: {}", e),
            PylonError::Security(e) => write!(f, "Security error: {}", e),
            PylonError::Temporal(e) => write!(f, "Temporal error: {}", e),
            PylonError::Spatial(e) => write!(f, "Spatial error: {}", e),
            PylonError::Individual(e) => write!(f, "Individual error: {}", e),
            PylonError::Economic(e) => write!(f, "Economic error: {}", e),
            PylonError::Fragment(e) => write!(f, "Fragment error: {}", e),
            PylonError::Precision(e) => write!(f, "Precision error: {}", e),
            PylonError::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl fmt::Display for ConfigurationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigurationError::InvalidValue { field, value, reason } => {
                write!(f, "Invalid value '{}' for field '{}': {}", value, field, reason)
            }
            ConfigurationError::MissingRequired { field } => {
                write!(f, "Missing required configuration field: {}", field)
            }
            ConfigurationError::FileError { path, error } => {
                write!(f, "Configuration file error at '{}': {}", path, error)
            }
            ConfigurationError::ParseError { error } => {
                write!(f, "Configuration parse error: {}", error)
            }
        }
    }
}

impl fmt::Display for CoordinationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoordinationError::PrecisionThresholdNotMet { required, achieved, domain } => {
                write!(f, "Precision threshold not met for {:?}: required {}, achieved {}", 
                       domain, required, achieved)
            }
            CoordinationError::Timeout { duration_ms, operation } => {
                write!(f, "Coordination timeout after {}ms for operation: {}", duration_ms, operation)
            }
            CoordinationError::SynchronizationFailure { node_id, reason } => {
                write!(f, "Synchronization failure for node {}: {}", node_id, reason)
            }
            CoordinationError::ReconstructionFailure { fragment_ids, reason } => {
                write!(f, "Fragment reconstruction failure for {} fragments: {}", 
                       fragment_ids.len(), reason)
            }
            CoordinationError::UnifiedCoordinationFailure { failed_domains, error_messages } => {
                write!(f, "Unified coordination failure in {} domains: {}", 
                       failed_domains.len(), error_messages.join(", "))
            }
        }
    }
}

// Implement Display for other error types (shortened for brevity)
impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetworkError::ConnectionFailure { address, error } => {
                write!(f, "Connection failure to '{}': {}", address, error)
            }
            NetworkError::TransmissionFailure { message_id, error } => {
                write!(f, "Transmission failure for message {}: {}", message_id, error)
            }
            NetworkError::Timeout { operation, duration_ms } => {
                write!(f, "Network timeout after {}ms for operation: {}", duration_ms, operation)
            }
            NetworkError::ProtocolError { protocol, error } => {
                write!(f, "Protocol error for {}: {}", protocol, error)
            }
            NetworkError::NodeDiscoveryFailure { error } => {
                write!(f, "Node discovery failure: {}", error)
            }
        }
    }
}

impl fmt::Display for AlgorithmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlgorithmError::BuheraEastError { operation, error } => {
                write!(f, "Buhera-East error in {}: {}", operation, error)
            }
            AlgorithmError::BuheraNorthError { operation, error } => {
                write!(f, "Buhera-North error in {}: {}", operation, error)
            }
            AlgorithmError::BulawayoError { operation, error } => {
                write!(f, "Bulawayo error in {}: {}", operation, error)
            }
            AlgorithmError::HarareError { operation, error } => {
                write!(f, "Harare error in {}: {}", operation, error)
            }
            AlgorithmError::KinshasaError { operation, error } => {
                write!(f, "Kinshasa error in {}: {}", operation, error)
            }
            AlgorithmError::MufakoseError { operation, error } => {
                write!(f, "Mufakose error in {}: {}", operation, error)
            }
            AlgorithmError::SelfAwareError { operation, error } => {
                write!(f, "Self-Aware error in {}: {}", operation, error)
            }
            AlgorithmError::CoordinationError { suites_involved, error } => {
                write!(f, "Algorithm coordination error in suites {:?}: {}", suites_involved, error)
            }
        }
    }
}

impl fmt::Display for SecurityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SecurityError::EnvironmentalMeasurement { dimension, error } => {
                write!(f, "Environmental measurement error in {}: {}", dimension, error)
            }
            SecurityError::MDTECCryptographic { operation, error } => {
                write!(f, "MDTEC cryptographic error in {}: {}", operation, error)
            }
            SecurityError::CurrencyGeneration { reason } => {
                write!(f, "Currency generation error: {}", reason)
            }
            SecurityError::PaymentVerification { reason } => {
                write!(f, "Payment verification error: {}", reason)
            }
            SecurityError::ThermodynamicSecurity { security_level, required_level } => {
                write!(f, "Thermodynamic security insufficient: {} < {}", security_level, required_level)
            }
            SecurityError::Authentication { error } => {
                write!(f, "Authentication error: {}", error)
            }
            SecurityError::Authorization { operation, required_permissions } => {
                write!(f, "Authorization error for {}: requires {:?}", operation, required_permissions)
            }
        }
    }
}

// Add remaining Display implementations for other error types
impl fmt::Display for TemporalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self) // Simplified for brevity
    }
}

impl fmt::Display for SpatialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self) // Simplified for brevity
    }
}

impl fmt::Display for IndividualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self) // Simplified for brevity
    }
}

impl fmt::Display for EconomicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self) // Simplified for brevity
    }
}

impl fmt::Display for FragmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self) // Simplified for brevity
    }
}

impl fmt::Display for PrecisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self) // Simplified for brevity
    }
}

// Implement StdError trait
impl StdError for PylonError {}
impl StdError for ConfigurationError {}
impl StdError for CoordinationError {}
impl StdError for NetworkError {}
impl StdError for AlgorithmError {}
impl StdError for SecurityError {}
impl StdError for TemporalError {}
impl StdError for SpatialError {}
impl StdError for IndividualError {}
impl StdError for EconomicError {}
impl StdError for FragmentError {}
impl StdError for PrecisionError {}

// Convenient conversion functions
impl From<ConfigurationError> for PylonError {
    fn from(error: ConfigurationError) -> Self {
        PylonError::Configuration(error)
    }
}

impl From<CoordinationError> for PylonError {
    fn from(error: CoordinationError) -> Self {
        PylonError::Coordination(error)
    }
}

impl From<NetworkError> for PylonError {
    fn from(error: NetworkError) -> Self {
        PylonError::Network(error)
    }
}

impl From<AlgorithmError> for PylonError {
    fn from(error: AlgorithmError) -> Self {
        PylonError::Algorithm(error)
    }
}

impl From<SecurityError> for PylonError {
    fn from(error: SecurityError) -> Self {
        PylonError::Security(error)
    }
}

impl From<TemporalError> for PylonError {
    fn from(error: TemporalError) -> Self {
        PylonError::Temporal(error)
    }
}

impl From<SpatialError> for PylonError {
    fn from(error: SpatialError) -> Self {
        PylonError::Spatial(error)
    }
}

impl From<IndividualError> for PylonError {
    fn from(error: IndividualError) -> Self {
        PylonError::Individual(error)
    }
}

impl From<EconomicError> for PylonError {
    fn from(error: EconomicError) -> Self {
        PylonError::Economic(error)
    }
}

impl From<FragmentError> for PylonError {
    fn from(error: FragmentError) -> Self {
        PylonError::Fragment(error)
    }
}

impl From<PrecisionError> for PylonError {
    fn from(error: PrecisionError) -> Self {
        PylonError::Precision(error)
    }
}
