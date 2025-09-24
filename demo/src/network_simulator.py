# network_simulator.py - Fixed version
import asyncio
import json
import time
import math
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PNG output
import matplotlib.pyplot as plt
import logging

try:
    from geopy.distance import geodesic
except ImportError:
    print("Warning: geopy not available, using simple distance calculation")
    def geodesic(coord1, coord2):
        # Simple haversine approximation
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in kilometers
        r = 6371
        
        class Distance:
            def __init__(self, km):
                self.kilometers = km
        
        return Distance(c * r)


@dataclass
class NetworkNode:
    """Complete network node with geographic and infrastructure properties"""

    id: str
    name: str
    latitude: float
    longitude: float
    timezone: str
    utc_offset: str
    infrastructure_type: str  # 'fiber', 'satellite', 'wireless'
    power_grid_frequency: int  # 50 or 60 Hz

    # Network characteristics
    bandwidth_mbps: float = 1000.0
    base_latency_ms: float = 1.0
    jitter_std_ms: float = 0.5
    packet_loss_rate: float = 0.001

    # Environmental factors
    weather_factor: float = 1.0
    solar_interference: float = 0.0
    atmospheric_conditions: str = "normal"

    # Runtime state
    current_load: float = 0.0
    active_connections: int = 0
    last_measurement_time: Optional[float] = None

    def __post_init__(self):
        """Initialize derived properties"""
        self.coordinates = (self.latitude, self.longitude)
        self.grid_interference = self._calculate_grid_interference()

    def _calculate_grid_interference(self) -> float:
        """Calculate power grid interference based on frequency"""
        base_interference = 0.02 if self.power_grid_frequency == 50 else 0.03
        return base_interference * (1 + random.uniform(-0.5, 0.5))


class NetworkSimulator:
    """Enhanced network simulator with realistic physics modeling"""

    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.nodes: Dict[str, NetworkNode] = {}
        self.connections: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.load_configuration(config_path)

    def load_configuration(self, config_path: Optional[str] = None):
        """Load network topology configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                self._create_nodes_from_config(config)
        else:
            self._create_default_topology()

    def _create_default_topology(self):
        """Create default 10-node global topology"""
        default_nodes = [
            {"id": "tokyo", "name": "Tokyo, Japan", "lat": 35.6762, "lon": 139.6503,
             "tz": "Asia/Tokyo", "utc": "+9", "infra": "fiber", "grid": 50},
            {"id": "sydney", "name": "Sydney, Australia", "lat": -33.8688, "lon": 151.2093,
             "tz": "Australia/Sydney", "utc": "+10", "infra": "fiber", "grid": 50},
            {"id": "harare", "name": "Harare, Zimbabwe", "lat": -17.8292, "lon": 31.0522,
             "tz": "Africa/Harare", "utc": "+2", "infra": "wireless", "grid": 50},
            {"id": "lapaz", "name": "La Paz, Bolivia", "lat": -16.5000, "lon": -68.1500,
             "tz": "America/La_Paz", "utc": "-4", "infra": "satellite", "grid": 60},
            {"id": "london", "name": "London, UK", "lat": 51.5074, "lon": -0.1278,
             "tz": "Europe/London", "utc": "+0", "infra": "fiber", "grid": 50},
        ]

        for node_config in default_nodes:
            node = NetworkNode(
                id=node_config["id"],
                name=node_config["name"],
                latitude=node_config["lat"],
                longitude=node_config["lon"],
                timezone=node_config["tz"],
                utc_offset=node_config["utc"],
                infrastructure_type=node_config["infra"],
                power_grid_frequency=node_config["grid"]
            )
            self.nodes[node.id] = node

    def calculate_realistic_latency(self, source_id: str, dest_id: str) -> float:
        """Calculate realistic latency including physics and infrastructure"""
        if source_id not in self.nodes or dest_id not in self.nodes:
            return float('inf')

        source = self.nodes[source_id]
        dest = self.nodes[dest_id]

        # Geographic distance
        distance_km = geodesic(source.coordinates, dest.coordinates).kilometers

        # Speed of light delay (fiber = ~200,000 km/s, satellite = ~300,000 km/s)
        if source.infrastructure_type == 'satellite' or dest.infrastructure_type == 'satellite':
            light_speed_delay = distance_km / 200000.0  # seconds
        else:
            light_speed_delay = distance_km / 200000.0  # fiber optic

        # Infrastructure delays
        infra_delay = self._calculate_infrastructure_delay(source, dest)

        # Environmental factors
        env_delay = self._calculate_environmental_delay(source, dest)

        total_latency = (light_speed_delay + infra_delay + env_delay) * 1000  # convert to ms

        return max(total_latency, 0.1)  # minimum 0.1ms

    def _calculate_infrastructure_delay(self, source: NetworkNode, dest: NetworkNode) -> float:
        """Calculate infrastructure-specific delays"""
        # Different infrastructure types have different processing delays
        infra_delays = {
            'fiber': 0.0001,     # 0.1ms
            'wireless': 0.002,   # 2ms  
            'satellite': 0.25    # 250ms for geostationary
        }
        
        source_delay = infra_delays.get(source.infrastructure_type, 0.001)
        dest_delay = infra_delays.get(dest.infrastructure_type, 0.001)
        
        return source_delay + dest_delay

    def _calculate_environmental_delay(self, source: NetworkNode, dest: NetworkNode) -> float:
        """Calculate environmental factor delays"""
        # Weather and atmospheric conditions
        weather_delay = (source.weather_factor + dest.weather_factor - 2.0) * 0.001
        
        # Solar interference (affects satellite communications more)
        solar_delay = 0.0
        if source.infrastructure_type == 'satellite' or dest.infrastructure_type == 'satellite':
            solar_delay = (source.solar_interference + dest.solar_interference) * 0.1
        
        # Power grid interference
        grid_delay = source.grid_interference + dest.grid_interference
        
        return max(0.0, weather_delay + solar_delay + grid_delay)

    async def initialize(self):
        """Initialize network simulator with connections"""
        self.logger.info("Initializing network simulator...")
        
        # Create connections between all nodes
        nodes = list(self.nodes.keys())
        for source_id in nodes:
            for dest_id in nodes:
                if source_id != dest_id:
                    connection = self.create_connection(source_id, dest_id)
                    self.connections[(source_id, dest_id)] = connection
        
        self.logger.info(f"Created {len(self.connections)} network connections")

    def create_connection(self, source_id: str, dest_id: str):
        """Create a network connection between two nodes"""
        if source_id not in self.nodes or dest_id not in self.nodes:
            raise ValueError(f"Invalid nodes: {source_id} -> {dest_id}")
        
        # Calculate realistic latency
        latency_ms = self.calculate_realistic_latency(source_id, dest_id)
        
        # Create connection object
        connection = {
            'source': source_id,
            'destination': dest_id,
            'current_latency_ms': latency_ms,
            'bandwidth_mbps': self.nodes[source_id].bandwidth_mbps,
            'packet_loss_rate': self.nodes[source_id].packet_loss_rate,
            'jitter_ms': self.nodes[source_id].jitter_std_ms,
            'created_time': time.time()
        }
        
        return connection

    async def get_precision_measurement(self, node_id: str) -> Dict[str, Any]:
        """Get precision measurement for a node"""
        if node_id not in self.nodes:
            raise ValueError(f"Unknown node: {node_id}")
        
        node = self.nodes[node_id]
        current_time = time.time()
        
        # Simulate measurement with realistic variations
        measurement = {
            'timestamp': current_time,
            'node_id': node_id,
            'atomic_reference': current_time,  # Simulated atomic time
            'local_measurement': current_time + random.uniform(-0.001, 0.001),  # ±1ms
            'precision_difference': random.uniform(-0.001, 0.001),  # ±1ms difference
            'measurement_quality': random.uniform(0.8, 1.0),
            'confidence': random.uniform(0.85, 0.95),
            'reference_source': 'simulated_atomic_clock'
        }
        
        node.last_measurement_time = current_time
        return measurement

    def get_network_status(self) -> Dict[str, Any]:
        """Get current network status"""
        current_time = time.time()
        
        # Calculate average latency across all connections
        if self.connections:
            latencies = [conn['current_latency_ms'] for conn in self.connections.values()]
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
        else:
            avg_latency = min_latency = max_latency = 0.0

        # Calculate network load
        total_load = sum(node.current_load for node in self.nodes.values())
        avg_load = total_load / len(self.nodes) if self.nodes else 0.0

        return {
            'total_nodes': len(self.nodes),
            'total_connections': len(self.connections),
            'average_latency_ms': avg_latency,
            'min_latency_ms': min_latency,
            'max_latency_ms': max_latency,
            'average_load': avg_load,
            'simulation_uptime': current_time - self._start_time if hasattr(self, '_start_time') else 0.0,
            'timestamp': current_time
        }

    async def send_traditional_message(self, source_id: str, dest_id: str, message: str) -> Dict[str, Any]:
        """Send a traditional message between nodes"""
        if source_id not in self.nodes or dest_id not in self.nodes:
            raise ValueError(f"Invalid nodes: {source_id} -> {dest_id}")
        
        start_time = time.time()
        connection_key = (source_id, dest_id)
        
        if connection_key not in self.connections:
            # Create connection if it doesn't exist
            self.connections[connection_key] = self.create_connection(source_id, dest_id)
        
        connection = self.connections[connection_key]
        
        # Simulate transmission
        latency_seconds = connection['current_latency_ms'] / 1000.0
        await asyncio.sleep(latency_seconds)
        
        # Simulate potential packet loss
        packet_loss = random.random() < connection['packet_loss_rate']
        
        transmission_result = {
            'timestamp': start_time,
            'source_node': source_id,
            'destination_node': dest_id,
            'message_size': len(message.encode('utf-8')),
            'latency_seconds': latency_seconds,
            'connection_latency_ms': connection['current_latency_ms'],
            'packet_loss_occurred': packet_loss,
            'processing_delay_ms': random.uniform(1.0, 5.0),
            'success': not packet_loss
        }
        
        return transmission_result

    async def set_traditional_mode(self, enabled: bool):
        """Set traditional networking mode"""
        self.traditional_mode = enabled
        mode_str = "traditional" if enabled else "sango_rine_shumba"
        self.logger.info(f"Network mode set to: {mode_str}")

    def _create_nodes_from_config(self, config: Dict[str, Any]):
        """Create nodes from configuration"""
        nodes_config = config.get('nodes', {})
        
        for node_id, node_data in nodes_config.items():
            node = NetworkNode(
                id=node_id,
                name=node_data.get('name', node_id),
                latitude=node_data['lat'],
                longitude=node_data['lon'],
                timezone=node_data.get('tz', 'UTC'),
                utc_offset=node_data.get('utc', '+0'),
                infrastructure_type=node_data.get('infra', 'fiber'),
                power_grid_frequency=node_data.get('grid', 50)
            )
            self.nodes[node.id] = node

    def __init__(self, config_path: Optional[str] = None, data_collector=None):
        """Initialize network simulator"""
        self.logger = logging.getLogger(__name__)
        self.nodes: Dict[str, NetworkNode] = {}
        self.connections: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.data_collector = data_collector
        self.traditional_mode = False
        self._start_time = time.time()
        
        self.load_configuration(config_path)

    def export_network_data(self, output_dir: Path = Path("output")) -> Dict[str, Any]:
        """Export comprehensive network data for analysis"""
        output_dir.mkdir(exist_ok=True)
        
        # Collect network topology data
        topology_data = {
            'nodes': {},
            'connections': {},
            'statistics': self.get_network_status()
        }
        
        # Export node data
        for node_id, node in self.nodes.items():
            topology_data['nodes'][node_id] = {
                'id': node.id,
                'name': node.name,
                'latitude': node.latitude,
                'longitude': node.longitude,
                'infrastructure_type': node.infrastructure_type,
                'bandwidth_mbps': node.bandwidth_mbps,
                'base_latency_ms': node.base_latency_ms,
                'current_load': node.current_load,
                'active_connections': node.active_connections
            }
        
        # Export connection data
        for conn_key, connection in self.connections.items():
            source, dest = conn_key
            conn_id = f"{source}->{dest}"
            topology_data['connections'][conn_id] = connection
        
        # Save topology data to JSON
        json_file = output_dir / "network_topology.json"
        with open(json_file, 'w') as f:
            json.dump(topology_data, f, indent=2)
        
        self.logger.info(f"Network topology exported to {json_file}")
        return topology_data

    def create_network_visualization(self, output_dir: Path = Path("output")):
        """Create network visualization plots"""
        output_dir.mkdir(exist_ok=True)
        
        if not self.nodes:
            self.logger.warning("No nodes available for visualization")
            return
        
        # Create latency heatmap
        self._create_latency_heatmap(output_dir)
        
        # Create network topology map
        self._create_topology_map(output_dir)
        
        # Create performance metrics plot
        self._create_performance_plot(output_dir)

    def _create_latency_heatmap(self, output_dir: Path):
        """Create latency heatmap between all nodes"""
        try:
            nodes = list(self.nodes.keys())
            n = len(nodes)
            
            if n < 2:
                return
                
            latency_matrix = np.zeros((n, n))
            
            for i, source in enumerate(nodes):
                for j, dest in enumerate(nodes):
                    if source != dest:
                        latency = self.calculate_realistic_latency(source, dest)
                        latency_matrix[i][j] = latency
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(latency_matrix, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Latency (ms)')
            plt.xticks(range(n), nodes, rotation=45)
            plt.yticks(range(n), nodes)
            plt.title('Network Latency Heatmap')
            plt.xlabel('Destination Node')
            plt.ylabel('Source Node')
            plt.tight_layout()
            
            heatmap_file = output_dir / "latency_heatmap.png"
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Latency heatmap saved to {heatmap_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating latency heatmap: {e}")

    def _create_topology_map(self, output_dir: Path):
        """Create network topology geographical map"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot nodes
            lats = [node.latitude for node in self.nodes.values()]
            lons = [node.longitude for node in self.nodes.values()]
            names = [node.name for node in self.nodes.values()]
            
            plt.scatter(lons, lats, c='red', s=100, alpha=0.7, zorder=5)
            
            # Add node labels
            for i, name in enumerate(names):
                plt.annotate(name, (lons[i], lats[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, alpha=0.8)
            
            # Plot connections (subset to avoid clutter)
            plotted_connections = 0
            max_connections = 20  # Limit connections to avoid clutter
            
            for (source_id, dest_id), connection in self.connections.items():
                if plotted_connections >= max_connections:
                    break
                
                source_node = self.nodes[source_id]
                dest_node = self.nodes[dest_id]
                
                # Plot line between nodes
                plt.plot([source_node.longitude, dest_node.longitude],
                        [source_node.latitude, dest_node.latitude],
                        'b-', alpha=0.3, linewidth=1)
                
                plotted_connections += 1
            
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title('Network Topology Map')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            map_file = output_dir / "network_topology_map.png"
            plt.savefig(map_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Topology map saved to {map_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating topology map: {e}")

    def _create_performance_plot(self, output_dir: Path):
        """Create network performance metrics plot"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Latency distribution
            if self.connections:
                latencies = [conn['current_latency_ms'] for conn in self.connections.values()]
                ax1.hist(latencies, bins=20, alpha=0.7, color='blue', edgecolor='black')
                ax1.set_title('Latency Distribution')
                ax1.set_xlabel('Latency (ms)')
                ax1.set_ylabel('Frequency')
                ax1.grid(True, alpha=0.3)
            
            # Bandwidth distribution
            bandwidths = [node.bandwidth_mbps for node in self.nodes.values()]
            ax2.hist(bandwidths, bins=10, alpha=0.7, color='green', edgecolor='black')
            ax2.set_title('Bandwidth Distribution')
            ax2.set_xlabel('Bandwidth (Mbps)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            # Infrastructure types
            infra_types = [node.infrastructure_type for node in self.nodes.values()]
            infra_counts = {}
            for infra in infra_types:
                infra_counts[infra] = infra_counts.get(infra, 0) + 1
            
            ax3.pie(infra_counts.values(), labels=infra_counts.keys(), autopct='%1.1f%%')
            ax3.set_title('Infrastructure Types')
            
            # Node load distribution  
            loads = [node.current_load for node in self.nodes.values()]
            node_names = [node.id for node in self.nodes.values()]
            ax4.bar(range(len(loads)), loads, color='orange', alpha=0.7)
            ax4.set_title('Node Load Distribution')
            ax4.set_xlabel('Node')
            ax4.set_ylabel('Current Load')
            ax4.set_xticks(range(len(node_names)))
            ax4.set_xticklabels(node_names, rotation=45)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            performance_file = output_dir / "network_performance.png"
            plt.savefig(performance_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Performance plot saved to {performance_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating performance plot: {e}")


# Standalone execution capability
async def main():
    """Standalone execution of network simulator"""
    print("🌐 Network Simulator - Standalone Test")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Create output directory
        output_dir = Path("network_simulator_output")
        output_dir.mkdir(exist_ok=True)
        
        logger.info("Initializing network simulator...")
        
        # Initialize network simulator with default topology
        simulator = NetworkSimulator()
        await simulator.initialize()
        
        logger.info(f"Created network with {len(simulator.nodes)} nodes")
        
        # Run network performance test
        logger.info("Running network performance tests...")
        
        # Collect measurements
        measurements = []
        nodes = list(simulator.nodes.keys())
        
        for i in range(10):  # 10 test measurements
            for node_id in nodes[:3]:  # Test first 3 nodes
                try:
                    measurement = await simulator.get_precision_measurement(node_id)
                    measurements.append(measurement)
                    await asyncio.sleep(0.1)  # 100ms between measurements
                except Exception as e:
                    logger.warning(f"Measurement failed for {node_id}: {e}")
        
        # Test traditional message transmission
        logger.info("Testing traditional message transmission...")
        
        transmission_results = []
        for i in range(5):  # 5 test messages
            if len(nodes) >= 2:
                source = nodes[0]
                dest = nodes[1]
                message = f"test_message_{i}_{int(time.time())}"
                
                try:
                    result = await simulator.send_traditional_message(source, dest, message)
                    transmission_results.append(result)
                    logger.info(f"Message {i+1} sent: {result['latency_seconds']*1000:.2f}ms latency")
                except Exception as e:
                    logger.warning(f"Message transmission failed: {e}")
        
        # Collect final statistics
        network_status = simulator.get_network_status()
        
        # Export all data to JSON
        logger.info("Exporting network data...")
        topology_data = simulator.export_network_data(output_dir)
        
        # Add measurements and results to export
        export_data = {
            'network_topology': topology_data,
            'measurements': measurements,
            'transmission_results': transmission_results,
            'network_status': network_status,
            'test_summary': {
                'total_measurements': len(measurements),
                'total_transmissions': len(transmission_results),
                'test_duration': 'approximately 10 seconds',
                'timestamp': time.time()
            }
        }
        
        # Save complete test results
        results_file = output_dir / "network_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Complete test results saved to {results_file}")
        
        # Create visualizations
        logger.info("Creating network visualizations...")
        simulator.create_network_visualization(output_dir)
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 NETWORK SIMULATOR TEST RESULTS")
        print("=" * 50)
        print(f"✅ Network nodes: {network_status['total_nodes']}")
        print(f"✅ Network connections: {network_status['total_connections']}")
        print(f"✅ Average latency: {network_status['average_latency_ms']:.2f}ms")
        print(f"✅ Measurements collected: {len(measurements)}")
        print(f"✅ Message transmissions: {len(transmission_results)}")
        print(f"\n📁 All outputs saved to: {output_dir.absolute()}")
        print(f"📄 JSON results: {results_file.name}")
        print(f"📈 Visualizations: latency_heatmap.png, network_topology_map.png, network_performance.png")
        
        return True
        
    except Exception as e:
        logger.error(f"Network simulator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        exit(1)