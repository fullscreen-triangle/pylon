//! Pylon System Demonstration
//!
//! This demonstrates the unified spatio-temporal coordination framework

use std::collections::HashMap;
use std::time::Duration;

use tokio;
use tracing::{info, Level};
use tracing_subscriber;

use pylon::{
    PylonCoordinator, PylonConfig, CoordinationRequest, CoordinationPayload,
    CoordinationDomain, PrecisionLevel, TemporalCoordinate, PylonId,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("🔌 Starting Pylon Unified Spatio-Temporal Coordination Framework");
    info!("📡 Implementing Sango Rine Shumba Protocol");

    // Load default configuration
    let config = PylonConfig::default();
    
    // Initialize Pylon coordinator
    let coordinator = PylonCoordinator::new(config).await?;
    
    // Start the coordinator
    coordinator.start().await?;
    
    info!("✅ Pylon coordinator started successfully");
    
    // Demonstrate temporal coordination
    info!("🕐 Demonstrating temporal coordination...");
    
    let temporal_request = CoordinationRequest {
        request_id: PylonId::new_v4(),
        requesting_node: PylonId::new_v4(),
        domains: vec![CoordinationDomain::Temporal],
        precision_level: PrecisionLevel::Quantum,
        payload: CoordinationPayload::TemporalSync {
            target_precision: 1e-12, // Picosecond precision
            reference_time: TemporalCoordinate::now(),
        },
        timestamp: TemporalCoordinate::now(),
    };
    
    match coordinator.coordinate(temporal_request).await {
        Ok(response) => {
            info!("✅ Temporal coordination successful: {:?}", response.result);
        }
        Err(e) => {
            info!("❌ Temporal coordination failed: {}", e);
        }
    }
    
    // Demonstrate unified coordination across multiple domains
    info!("🌐 Demonstrating unified spatio-temporal coordination...");
    
    let unified_request = CoordinationRequest {
        request_id: PylonId::new_v4(),
        requesting_node: PylonId::new_v4(),
        domains: vec![
            CoordinationDomain::Temporal,
            CoordinationDomain::Spatial,
            CoordinationDomain::Economic,
        ],
        precision_level: PrecisionLevel::High,
        payload: CoordinationPayload::TemporalSync {
            target_precision: 1e-9, // Nanosecond precision
            reference_time: TemporalCoordinate::now(),
        },
        timestamp: TemporalCoordinate::now(),
    };
    
    match coordinator.coordinate(unified_request).await {
        Ok(response) => {
            info!("✅ Unified coordination successful: {:?}", response.result);
        }
        Err(e) => {
            info!("❌ Unified coordination failed: {}", e);
        }
    }
    
    // Display system status and metrics
    info!("📊 System Status:");
    let status = coordinator.get_status().await;
    info!("   Status: {:?}", status);
    
    let metrics = coordinator.get_metrics().await;
    info!("   Total Requests: {}", metrics.total_requests);
    info!("   Successful Coordinations: {}", metrics.successful_coordinations);
    info!("   Current Unified Precision: {:.2e}", metrics.current_unified_precision);
    info!("   Active Nodes: {}", metrics.active_nodes);
    
    let network_nodes = coordinator.get_network_nodes().await;
    info!("   Network Nodes: {}", network_nodes.len());
    
    // Run for a short time to show system operation
    info!("🔄 Running system for demonstration...");
    tokio::time::sleep(Duration::from_secs(5)).await;
    
    // Graceful shutdown
    info!("🛑 Shutting down Pylon coordinator...");
    coordinator.shutdown().await?;
    
    info!("✅ Pylon system demonstration completed successfully");
    info!("🌟 Revolutionary unified spatio-temporal coordination achieved!");
    
    Ok(())
}
