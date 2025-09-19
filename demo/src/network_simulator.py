"""
Network Simulator Module

Implements realistic global network topology simulation with physics-based latency modeling,
power grid interference effects, and environmental factors affecting network performance.

This module creates a virtual network of 10 globally distributed nodes with realistic
characteristics including geographic distances, infrastructure types, and environmental
factors that affect temporal coordination in Sango Rine Shumba.
"""

import asyncio
import json
import time
import math
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from geopy.distance import geodesic
import logging

@dataclass
class NetworkNode:
    """Represents a network node with geographic and infrastructure properties"""
    
    id: str
    name: str
    latitude: float
    longitude: float
    timezone: str
    utc_offset: str
    infrastructure_type: str
    power_grid_frequency: int
    connection_quality: str
    base_jitter_ms: float
    jitter_std_ms: float
    processing_delay_ms: float
    max_bandwidth_mbps: int
    concurrent_connections: int
    precision_level: str
    
    # Runtime state
    current_load: float = 0.0
    active_connections: int = 0
    last_measurement: float = 0.0
    local_clock_drift: float = 0.0
    environmental_factor: float = 1.0
    
    def __post_init__(self):
        """Initialize derived properties"""
        self.location = (self.latitude, self.longitude)
        self.is_active = True
        self.message_buffer = []
        self.performance_metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'total_latency': 0.0,
            'avg_latency': 0.0,
            'jitter_variance': 0.0,
            'precision_measurements': []
        }

@dataclass
class NetworkConnection:
    """Represents a connection between two network nodes"""
    
    source_node: NetworkNode
    destination_node: NetworkNode
    base_latency_ms: float
    current_latency_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    packet_loss_rate: float = 0.0
    jitter_ms: float = 0.0
    is_active: bool = True
    
    def __post_init__(self):
        """Initialize connection properties"""
        self.connection_id = f"{self.source_node.id}->{self.destination_node.id}"
        self.distance_km = geodesic(
            self.source_node.location,
            self.destination_node.location
        ).kilometers
        
        # Calculate minimum speed-of-light delay
        self.speed_of_light_delay_ms = self.distance_km * 0.0033356  # ~3.336ms per 1000km
        
        # Set bandwidth based on node capabilities
        self.bandwidth_mbps = min(
            self.source_node.max_bandwidth_mbps,
            self.destination_node.max_bandwidth_mbps
        )

class NetworkSimulator:
    """
    Comprehensive network topology simulator with realistic physics modeling
    
    This simulator creates a virtual network environment that accurately models:
    - Geographic latencies based on speed of light
    - Infrastructure-specific jitter and delays
    - Power grid frequency interference effects
    - Environmental factors (atmospheric, solar activity)
    - Node processing capabilities and limitations
    """
    
    def __init__(self, config_path: str, data_collector=None):
        """Initialize network simulator with configuration"""
        self.config_path = Path(config_path)
        self.data_collector = data_collector
        self.logger = logging.getLogger(__name__)
        
        # Network state
        self.nodes: Dict[str, NetworkNode] = {}
        self.connections: Dict[str, NetworkConnection] = {}
        self.routing_table: Dict[str, Dict[str, List[str]]] = {}
        self.message_queue = asyncio.Queue()
        
        # Physics parameters
        self.speed_of_light_ms_per_km = 0.0033356
        self.fiber_propagation_factor = 1.46
        self.atmospheric_delay_factor = 1.02
        
        # Simulation state
        self.traditional_mode = True
        self.simulation_start_time = time.time()
        self.is_running = False
        
        self.logger.info("Network simulator initialized")
    
    async def initialize(self):
        """Load configuration and initialize network topology"""
        self.logger.info("Loading network configuration...")
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        self.config = config
        
        # Create nodes
        await self._create_nodes(config['nodes'])
        
        # Create connections (full mesh)
        await self._create_connections()
        
        # Build routing table
        await self._build_routing_table()
        
        # Start background tasks
        await self._start_background_tasks()
        
        self.is_running = True
        self.logger.info(f"Network initialized with {len(self.nodes)} nodes and {len(self.connections)} connections")
    
    async def _create_nodes(self, node_configs: List[Dict]):
        """Create network nodes from configuration"""
        for node_config in node_configs:
            node = NetworkNode(
                id=node_config['id'],
                name=node_config['name'],
                latitude=node_config['location']['latitude'],
                longitude=node_config['location']['longitude'],
                timezone=node_config['location']['timezone'],
                utc_offset=node_config['location']['utc_offset'],
                infrastructure_type=node_config['infrastructure']['type'],
                power_grid_frequency=node_config['infrastructure']['power_grid_frequency'],
                connection_quality=node_config['infrastructure']['connection_quality'],
                base_jitter_ms=node_config['infrastructure']['base_jitter_ms'],
                jitter_std_ms=node_config['infrastructure']['jitter_std_ms'],
                processing_delay_ms=node_config['infrastructure']['processing_delay_ms'],
                max_bandwidth_mbps=node_config['capabilities']['max_bandwidth_mbps'],
                concurrent_connections=node_config['capabilities']['concurrent_connections'],
                precision_level=node_config['capabilities']['precision_level']
            )
            
            self.nodes[node.id] = node
            self.logger.debug(f"Created node: {node.name} ({node.id})")
    
    async def _create_connections(self):
        """Create full mesh network connections between all nodes"""
        node_list = list(self.nodes.values())
        
        for i, source_node in enumerate(node_list):
            for dest_node in node_list[i+1:]:
                # Calculate base latency
                distance_km = geodesic(source_node.location, dest_node.location).kilometers
                base_latency_ms = self._calculate_base_latency(source_node, dest_node, distance_km)
                
                # Create bidirectional connections
                conn_forward = NetworkConnection(
                    source_node=source_node,
                    destination_node=dest_node,
                    base_latency_ms=base_latency_ms
                )
                
                conn_reverse = NetworkConnection(
                    source_node=dest_node,
                    destination_node=source_node,
                    base_latency_ms=base_latency_ms
                )
                
                self.connections[conn_forward.connection_id] = conn_forward
                self.connections[conn_reverse.connection_id] = conn_reverse
                
                self.logger.debug(f"Created connection: {source_node.id} <-> {dest_node.id} ({base_latency_ms:.2f}ms)")
    
    def _calculate_base_latency(self, node1: NetworkNode, node2: NetworkNode, distance_km: float) -> float:
        """Calculate base latency between two nodes considering infrastructure"""
        
        # Speed of light delay with fiber propagation factor
        speed_of_light_delay = distance_km * self.speed_of_light_ms_per_km * self.fiber_propagation_factor
        
        # Infrastructure-specific delays
        infra_delay1 = self._get_infrastructure_delay(node1.infrastructure_type)
        infra_delay2 = self._get_infrastructure_delay(node2.infrastructure_type)
        
        # Processing delays
        processing_delay = (node1.processing_delay_ms + node2.processing_delay_ms) / 2
        
        # Distance-based routing overhead (satellite vs fiber)
        routing_overhead = self._calculate_routing_overhead(node1, node2, distance_km)
        
        total_latency = speed_of_light_delay + infra_delay1 + infra_delay2 + processing_delay + routing_overhead
        
        return max(total_latency, 0.1)  # Minimum 0.1ms latency
    
    def _get_infrastructure_delay(self, infrastructure_type: str) -> float:
        """Get additional delay based on infrastructure type"""
        infrastructure_delays = {
            'fiber_backbone': 0.5,
            'fiber_satellite_mix': 2.0,
            'mixed_infrastructure': 5.0,
            'high_altitude_variable': 8.0,
            'premium_fiber': 0.2,
            'dense_network': 1.5,
            'regional_hub': 1.0,
            'regional_connectivity': 3.0,
            'geothermal_powered': 0.8
        }
        return infrastructure_delays.get(infrastructure_type, 2.0)
    
    def _calculate_routing_overhead(self, node1: NetworkNode, node2: NetworkNode, distance_km: float) -> float:
        """Calculate routing overhead based on distance and infrastructure"""
        
        # Long distance connections may use satellite links
        if distance_km > 8000:  # Intercontinental
            satellite_probability = 0.3
            if random.random() < satellite_probability:
                return 250  # Satellite round-trip delay
        
        # Calculate hop-based overhead for very long distances
        hops = max(1, int(distance_km / 2000))  # Assume backbone links every 2000km
        hop_overhead = hops * 0.5  # 0.5ms per hop
        
        return hop_overhead
    
    async def _build_routing_table(self):
        """Build routing table for efficient path finding"""
        for source_id in self.nodes:
            self.routing_table[source_id] = {}
            for dest_id in self.nodes:
                if source_id != dest_id:
                    # For full mesh, direct connection is always best
                    self.routing_table[source_id][dest_id] = [dest_id]
    
    async def _start_background_tasks(self):
        """Start background tasks for network simulation"""
        asyncio.create_task(self._update_network_conditions())
        asyncio.create_task(self._simulate_power_grid_interference())
        asyncio.create_task(self._simulate_environmental_effects())
    
    async def _update_network_conditions(self):
        """Continuously update network conditions (jitter, load, etc.)"""
        while self.is_running:
            try:
                current_time = time.time()
                
                for connection in self.connections.values():
                    # Update current latency with jitter
                    connection.current_latency_ms = self._calculate_current_latency(connection, current_time)
                    
                    # Update packet loss based on load
                    connection.packet_loss_rate = self._calculate_packet_loss(connection)
                
                # Update node load and drift
                for node in self.nodes.values():
                    node.current_load = random.uniform(0.1, 0.9)
                    node.local_clock_drift += random.gauss(0, 1e-8)  # Realistic clock drift
                
                await asyncio.sleep(0.01)  # Update every 10ms
                
            except Exception as e:
                self.logger.error(f"Error updating network conditions: {e}")
    
    def _calculate_current_latency(self, connection: NetworkConnection, current_time: float) -> float:
        """Calculate current latency including jitter and environmental factors"""
        
        base_latency = connection.base_latency_ms
        
        # Add jitter based on node characteristics
        source_jitter = random.gauss(
            connection.source_node.base_jitter_ms,
            connection.source_node.jitter_std_ms
        )
        dest_jitter = random.gauss(
            connection.destination_node.base_jitter_ms,
            connection.destination_node.jitter_std_ms
        )
        
        total_jitter = abs(source_jitter) + abs(dest_jitter)
        
        # Apply power grid interference
        power_grid_interference = self._calculate_power_grid_interference(
            connection, current_time
        )
        
        # Apply environmental factors
        environmental_factor = (connection.source_node.environmental_factor + 
                              connection.destination_node.environmental_factor) / 2
        
        # Load-based latency increase
        avg_load = (connection.source_node.current_load + connection.destination_node.current_load) / 2
        load_multiplier = 1.0 + (avg_load * 0.5)  # Up to 50% increase under load
        
        current_latency = (base_latency + total_jitter + power_grid_interference) * environmental_factor * load_multiplier
        
        return max(current_latency, base_latency * 0.8)  # Never go below 80% of base latency
    
    def _calculate_power_grid_interference(self, connection: NetworkConnection, current_time: float) -> float:
        """Calculate power grid interference effects on timing"""
        
        # Different interference based on power grid frequencies
        source_freq = connection.source_node.power_grid_frequency
        dest_freq = connection.destination_node.power_grid_frequency
        
        interference = 0.0
        
        # 50Hz grid interference
        if source_freq == 50:
            phase = 2 * math.pi * 50 * current_time
            interference += 0.02 * math.sin(phase) + 0.01 * math.sin(2 * phase)
        
        # 60Hz grid interference  
        if source_freq == 60:
            phase = 2 * math.pi * 60 * current_time
            interference += 0.03 * math.sin(phase) + 0.015 * math.sin(2 * phase)
        
        # Cross-frequency interference when grids differ
        if source_freq != dest_freq:
            beat_frequency = abs(source_freq - dest_freq)
            beat_phase = 2 * math.pi * beat_frequency * current_time
            interference += 0.01 * math.sin(beat_phase)
        
        return abs(interference)
    
    def _calculate_packet_loss(self, connection: NetworkConnection) -> float:
        """Calculate packet loss rate based on network conditions"""
        base_loss = 0.0001  # 0.01% base packet loss
        
        # Increase loss with distance
        distance_factor = min(connection.distance_km / 20000, 1.0)
        distance_loss = distance_factor * 0.001
        
        # Increase loss with load
        avg_load = (connection.source_node.current_load + connection.destination_node.current_load) / 2
        load_loss = avg_load * 0.002
        
        # Infrastructure quality factor
        quality_factors = {
            'premium': 0.5,
            'high': 0.8,
            'moderate': 1.2,
            'variable': 1.5
        }
        
        source_quality = quality_factors.get(connection.source_node.connection_quality, 1.0)
        dest_quality = quality_factors.get(connection.destination_node.connection_quality, 1.0)
        quality_factor = (source_quality + dest_quality) / 2
        
        total_loss = (base_loss + distance_loss + load_loss) * quality_factor
        
        return min(total_loss, 0.1)  # Cap at 10% loss
    
    async def _simulate_power_grid_interference(self):
        """Simulate power grid interference effects"""
        while self.is_running:
            try:
                # Power grid interference varies with local grid conditions
                for node in self.nodes.values():
                    # Simulate grid stability variations
                    if node.power_grid_frequency == 50:
                        stability = 0.98 + 0.02 * math.sin(2 * math.pi * time.time() / 3600)  # Hourly variation
                    else:  # 60Hz
                        stability = 0.97 + 0.03 * math.sin(2 * math.pi * time.time() / 3600)
                    
                    node.environmental_factor = stability
                
                await asyncio.sleep(1.0)  # Update every second
                
            except Exception as e:
                self.logger.error(f"Error simulating power grid interference: {e}")
    
    async def _simulate_environmental_effects(self):
        """Simulate environmental effects on network performance"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Simulate atmospheric conditions
                atmospheric_variation = 1.0 + 0.05 * math.sin(2 * math.pi * current_time / 86400)  # Daily cycle
                
                # Simulate solar activity (simplified)
                solar_activity = 1.0 + 0.1 * math.sin(2 * math.pi * current_time / (11 * 365 * 86400))  # 11-year cycle
                
                for node in self.nodes.values():
                    # Apply environmental factors based on location
                    latitude_factor = 1.0 + 0.02 * abs(node.latitude) / 90  # Higher latitudes more affected
                    
                    node.environmental_factor *= atmospheric_variation * solar_activity * latitude_factor
                    node.environmental_factor = max(0.8, min(1.2, node.environmental_factor))  # Keep reasonable bounds
                
                await asyncio.sleep(60.0)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error simulating environmental effects: {e}")
    
    async def send_traditional_message(self, source_id: str, dest_id: str, message: str) -> Dict[str, Any]:
        """Send message using traditional networking (for baseline comparison)"""
        
        if source_id not in self.nodes or dest_id not in self.nodes:
            raise ValueError(f"Invalid node IDs: {source_id}, {dest_id}")
        
        send_time = time.time()
        connection_id = f"{source_id}->{dest_id}"
        
        if connection_id not in self.connections:
            raise ValueError(f"No connection found: {connection_id}")
        
        connection = self.connections[connection_id]
        
        # Simulate traditional TCP/IP transmission
        transmission_latency = connection.current_latency_ms / 1000  # Convert to seconds
        
        # Add processing delays
        processing_delay = (connection.source_node.processing_delay_ms + 
                          connection.destination_node.processing_delay_ms) / 2000  # Convert to seconds
        
        # Simulate packet loss with retransmission
        packet_loss_delay = 0.0
        if random.random() < connection.packet_loss_rate:
            packet_loss_delay = transmission_latency * 2  # Retransmission delay
        
        total_delay = transmission_latency + processing_delay + packet_loss_delay
        
        # Simulate actual transmission delay
        await asyncio.sleep(total_delay)
        
        receive_time = time.time()
        actual_latency = receive_time - send_time
        
        # Update performance metrics
        source_node = self.nodes[source_id]
        dest_node = self.nodes[dest_id]
        
        source_node.performance_metrics['messages_sent'] += 1
        dest_node.performance_metrics['messages_received'] += 1
        
        source_node.performance_metrics['total_latency'] += actual_latency
        source_node.performance_metrics['avg_latency'] = (
            source_node.performance_metrics['total_latency'] / 
            source_node.performance_metrics['messages_sent']
        )
        
        # Log performance data
        if self.data_collector:
            await self.data_collector.log_traditional_message({
                'timestamp': send_time,
                'source_node': source_id,
                'destination_node': dest_id,
                'message_size': len(message),
                'latency_seconds': actual_latency,
                'connection_latency_ms': connection.current_latency_ms,
                'packet_loss_occurred': packet_loss_delay > 0,
                'processing_delay_ms': processing_delay * 1000
            })
        
        return {
            'sent_time': send_time,
            'received_time': receive_time,
            'latency_seconds': actual_latency,
            'latency_ms': actual_latency * 1000,
            'message_size': len(message),
            'packet_loss': packet_loss_delay > 0,
            'connection_latency_ms': connection.current_latency_ms
        }
    
    async def get_precision_measurement(self, node_id: str) -> Dict[str, Any]:
        """Get precision measurement for a specific node"""
        
        if node_id not in self.nodes:
            raise ValueError(f"Node not found: {node_id}")
        
        node = self.nodes[node_id]
        current_time = time.time()
        
        # Simulate local clock measurement with realistic drift and jitter
        local_clock_time = current_time + node.local_clock_drift
        
        # Add measurement jitter based on node precision level
        if node.precision_level == 'nanosecond':
            measurement_jitter = random.gauss(0, 1e-9)
        elif node.precision_level == 'microsecond':
            measurement_jitter = random.gauss(0, 1e-6)
        else:
            measurement_jitter = random.gauss(0, 1e-3)
        
        local_measurement = local_clock_time + measurement_jitter
        node.last_measurement = local_measurement
        
        # Store precision measurement
        node.performance_metrics['precision_measurements'].append({
            'timestamp': current_time,
            'local_measurement': local_measurement,
            'drift': node.local_clock_drift,
            'jitter': measurement_jitter
        })
        
        return {
            'node_id': node_id,
            'timestamp': current_time,
            'local_measurement': local_measurement,
            'clock_drift': node.local_clock_drift,
            'measurement_jitter': measurement_jitter,
            'precision_level': node.precision_level
        }
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status"""
        
        total_connections = len(self.connections)
        active_connections = sum(1 for conn in self.connections.values() if conn.is_active)
        
        avg_latency = np.mean([conn.current_latency_ms for conn in self.connections.values()])
        min_latency = np.min([conn.current_latency_ms for conn in self.connections.values()])
        max_latency = np.max([conn.current_latency_ms for conn in self.connections.values()])
        
        avg_load = np.mean([node.current_load for node in self.nodes.values()])
        
        return {
            'total_nodes': len(self.nodes),
            'total_connections': total_connections,
            'active_connections': active_connections,
            'average_latency_ms': float(avg_latency),
            'min_latency_ms': float(min_latency),
            'max_latency_ms': float(max_latency),
            'average_load': float(avg_load),
            'simulation_uptime': time.time() - self.simulation_start_time,
            'traditional_mode': self.traditional_mode
        }
    
    async def set_traditional_mode(self, traditional: bool):
        """Set network to traditional or Sango Rine Shumba mode"""
        self.traditional_mode = traditional
        self.logger.info(f"Network mode set to: {'Traditional' if traditional else 'Sango Rine Shumba'}")
    
    def stop(self):
        """Stop the network simulator"""
        self.is_running = False
        self.logger.info("Network simulator stopped")
