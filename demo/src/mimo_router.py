"""
MIMO Router Module

Implements MIMO-like multi-path routing where temporal fragments are transmitted
through different network paths and coordinated to arrive simultaneously at
destination nodes. This creates bandwidth efficiency improvements and enhanced
reliability through path diversity.

The router coordinates fragment transmission timing to achieve temporal
convergence at the destination, demonstrating the practical benefits of
Sango Rine Shumba's temporal coordination framework.
"""

import asyncio
import time
import random
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import logging
import numpy as np
from collections import defaultdict, deque

@dataclass
class RoutingPath:
    """Represents a network routing path between two nodes"""
    
    path_id: str
    source_node: str
    destination_node: str
    intermediate_nodes: List[str]
    estimated_latency: float
    bandwidth_mbps: float
    reliability: float
    current_load: float = 0.0
    
    # Path quality metrics
    jitter_ms: float = 0.0
    packet_loss_rate: float = 0.0
    congestion_level: float = 0.0
    
    # Usage statistics
    fragments_sent: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    total_latency: float = 0.0
    
    def __post_init__(self):
        """Initialize derived properties"""
        self.path_length = len(self.intermediate_nodes) + 1  # +1 for final hop
        self.is_active = True
        self.last_used_time = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate delivery success rate for this path"""
        total_attempts = self.successful_deliveries + self.failed_deliveries
        if total_attempts == 0:
            return 1.0  # Assume perfect until proven otherwise
        return self.successful_deliveries / total_attempts
    
    @property
    def average_latency(self) -> float:
        """Calculate average latency for this path"""
        if self.successful_deliveries == 0:
            return self.estimated_latency
        return self.total_latency / self.successful_deliveries
    
    @property
    def quality_score(self) -> float:
        """Calculate overall path quality score"""
        latency_score = max(0, 1 - (self.average_latency / 1000))  # Normalize to 1s
        reliability_score = self.success_rate
        congestion_score = max(0, 1 - self.congestion_level)
        load_score = max(0, 1 - self.current_load)
        
        return (latency_score + reliability_score + congestion_score + load_score) / 4

@dataclass
class FragmentDelivery:
    """Tracks delivery of a single fragment through a routing path"""
    
    fragment_id: str
    path_id: str
    send_time: float
    expected_arrival_time: float
    actual_arrival_time: Optional[float] = None
    delivery_success: bool = False
    
    # Delivery metrics
    transmission_latency: float = 0.0
    path_latency: float = 0.0
    jitter: float = 0.0
    
    def __post_init__(self):
        """Initialize delivery tracking"""
        self.is_delivered = False
        self.timeout_time = self.send_time + 10.0  # 10 second timeout
    
    @property
    def is_overdue(self) -> bool:
        """Check if fragment delivery is overdue"""
        return time.time() > self.expected_arrival_time + 1.0  # 1 second tolerance
    
    @property
    def is_timed_out(self) -> bool:
        """Check if fragment delivery has timed out"""
        return time.time() > self.timeout_time

@dataclass
class MIMOTransmission:
    """Represents a MIMO transmission across multiple paths"""
    
    transmission_id: str
    message_id: str
    source_node: str
    destination_node: str
    fragments: List[str]  # fragment_ids
    paths: List[str]  # path_ids
    target_arrival_time: float
    
    # Transmission tracking
    start_time: float = 0.0
    deliveries: List[FragmentDelivery] = field(default_factory=list)
    completed_fragments: Set[str] = field(default_factory=set)
    failed_fragments: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Initialize transmission tracking"""
        if self.start_time == 0.0:
            self.start_time = time.time()
        
        self.is_complete = False
        self.completion_time: Optional[float] = None
        self.success = False
    
    @property
    def completion_ratio(self) -> float:
        """Get completion ratio (0.0 to 1.0)"""
        if not self.fragments:
            return 1.0
        return len(self.completed_fragments) / len(self.fragments)
    
    @property
    def is_successful(self) -> bool:
        """Check if transmission was successful (all fragments delivered)"""
        return len(self.completed_fragments) == len(self.fragments)
    
    @property
    def convergence_quality(self) -> float:
        """Calculate temporal convergence quality"""
        if not self.deliveries or len(self.completed_fragments) < 2:
            return 0.0
        
        completed_deliveries = [d for d in self.deliveries if d.delivery_success]
        if len(completed_deliveries) < 2:
            return 0.0
        
        arrival_times = [d.actual_arrival_time for d in completed_deliveries if d.actual_arrival_time]
        if len(arrival_times) < 2:
            return 0.0
        
        # Calculate temporal convergence (lower variance = better convergence)
        arrival_variance = np.var(arrival_times)
        max_variance = 1.0  # 1 second maximum variance
        convergence = max(0, 1 - (arrival_variance / max_variance))
        
        return convergence

class MIMORouter:
    """
    MIMO-like multi-path routing system
    
    Implements sophisticated routing strategies that:
    1. Distribute fragments across multiple network paths
    2. Coordinate transmission timing for simultaneous arrival
    3. Optimize bandwidth utilization through path diversity
    4. Provide enhanced reliability through redundancy
    5. Demonstrate temporal coordination benefits
    """
    
    def __init__(self, network_simulator, temporal_fragmenter, data_collector=None):
        """Initialize MIMO router"""
        self.network_simulator = network_simulator
        self.temporal_fragmenter = temporal_fragmenter
        self.data_collector = data_collector
        self.logger = logging.getLogger(__name__)
        
        # Routing infrastructure
        self.routing_paths: Dict[str, RoutingPath] = {}
        self.path_matrix: Dict[str, Dict[str, List[str]]] = {}  # source -> dest -> path_ids
        self.active_transmissions: Dict[str, MIMOTransmission] = {}
        
        # Routing parameters
        self.max_paths_per_transmission = 4
        self.convergence_tolerance_ms = 50  # 50ms arrival tolerance
        self.path_selection_strategy = "latency_optimized"
        self.load_balancing_enabled = True
        
        # Performance metrics
        self.performance_metrics = {
            'transmissions_started': 0,
            'transmissions_completed': 0,
            'fragments_routed': 0,
            'fragments_delivered': 0,
            'average_convergence_quality': 0.0,
            'bandwidth_efficiency': 0.0,
            'path_utilization': defaultdict(float),
            'latency_improvements': deque(maxlen=1000)
        }
        
        # Service state
        self.is_running = False
        
        self.logger.info("MIMO router initialized")
    
    async def start_routing_service(self):
        """Start MIMO routing background service"""
        self.logger.info("Starting MIMO routing service...")
        self.is_running = True
        
        # Initialize routing paths
        await self._initialize_routing_paths()
        
        # Start background tasks
        monitoring_task = asyncio.create_task(self._monitor_transmissions())
        optimization_task = asyncio.create_task(self._optimize_path_selection())
        
        try:
            await asyncio.gather(monitoring_task, optimization_task)
        except asyncio.CancelledError:
            self.logger.info("MIMO routing service stopped")
    
    async def _initialize_routing_paths(self):
        """Initialize routing paths between all node pairs"""
        
        nodes = list(self.network_simulator.nodes.keys())
        path_counter = 0
        
        for source_node in nodes:
            self.path_matrix[source_node] = {}
            
            for dest_node in nodes:
                if source_node != dest_node:
                    # Create multiple paths between each node pair
                    paths = await self._create_paths_between_nodes(source_node, dest_node)
                    self.path_matrix[source_node][dest_node] = [p.path_id for p in paths]
                    
                    for path in paths:
                        self.routing_paths[path.path_id] = path
                        path_counter += 1
        
        self.logger.info(f"Initialized {path_counter} routing paths between {len(nodes)} nodes")
    
    async def _create_paths_between_nodes(self, source: str, dest: str) -> List[RoutingPath]:
        """Create multiple routing paths between two nodes"""
        
        paths = []
        connection_id = f"{source}->{dest}"
        
        # Get base connection info
        if connection_id in self.network_simulator.connections:
            base_connection = self.network_simulator.connections[connection_id]
            base_latency = base_connection.current_latency_ms
            base_bandwidth = base_connection.bandwidth_mbps
            
            # Create primary direct path
            primary_path = RoutingPath(
                path_id=f"path_direct_{source}_{dest}",
                source_node=source,
                destination_node=dest,
                intermediate_nodes=[],
                estimated_latency=base_latency,
                bandwidth_mbps=base_bandwidth,
                reliability=0.95
            )
            paths.append(primary_path)
            
            # Create alternative paths through different intermediate nodes
            other_nodes = [n for n in self.network_simulator.nodes.keys() 
                          if n not in [source, dest]]
            
            for i, intermediate in enumerate(other_nodes[:2]):  # Max 2 alternative paths
                # Calculate path through intermediate node
                hop1_id = f"{source}->{intermediate}"
                hop2_id = f"{intermediate}->{dest}"
                
                if (hop1_id in self.network_simulator.connections and 
                    hop2_id in self.network_simulator.connections):
                    
                    hop1_conn = self.network_simulator.connections[hop1_id]
                    hop2_conn = self.network_simulator.connections[hop2_id]
                    
                    path_latency = hop1_conn.current_latency_ms + hop2_conn.current_latency_ms
                    path_bandwidth = min(hop1_conn.bandwidth_mbps, hop2_conn.bandwidth_mbps)
                    
                    alt_path = RoutingPath(
                        path_id=f"path_alt{i+1}_{source}_{intermediate}_{dest}",
                        source_node=source,
                        destination_node=dest,
                        intermediate_nodes=[intermediate],
                        estimated_latency=path_latency,
                        bandwidth_mbps=path_bandwidth,
                        reliability=0.85  # Lower reliability for longer paths
                    )
                    paths.append(alt_path)
        
        return paths
    
    async def route_fragments(self, fragments: List) -> str:
        """Route temporal fragments using MIMO strategy"""
        
        if not fragments:
            return ""
        
        # Extract routing information
        first_fragment = fragments[0]
        message_id = first_fragment.message_id
        source_node = first_fragment.source_node
        destination_node = first_fragment.destination_node
        
        transmission_id = f"mimo_tx_{int(time.time() * 1000000)}"
        
        # Select optimal paths for this transmission
        selected_paths = await self._select_optimal_paths(
            source_node, destination_node, len(fragments)
        )
        
        if not selected_paths:
            raise RuntimeError(f"No routing paths available from {source_node} to {destination_node}")
        
        # Calculate target arrival time based on slowest path
        slowest_latency = max(path.estimated_latency for path in selected_paths)
        target_arrival_time = time.time() + (slowest_latency / 1000) + 0.1  # +100ms buffer
        
        # Create MIMO transmission
        mimo_tx = MIMOTransmission(
            transmission_id=transmission_id,
            message_id=message_id,
            source_node=source_node,
            destination_node=destination_node,
            fragments=[f.fragment_id for f in fragments],
            paths=[p.path_id for p in selected_paths],
            target_arrival_time=target_arrival_time
        )
        
        # Distribute fragments across selected paths
        await self._distribute_fragments_across_paths(fragments, selected_paths, mimo_tx)
        
        # Store transmission
        self.active_transmissions[transmission_id] = mimo_tx
        
        # Update performance metrics
        self.performance_metrics['transmissions_started'] += 1
        self.performance_metrics['fragments_routed'] += len(fragments)
        
        # Log routing data
        if self.data_collector:
            await self.data_collector.log_mimo_routing({
                'timestamp': time.time(),
                'transmission_id': transmission_id,
                'message_id': message_id,
                'source_node': source_node,
                'destination_node': destination_node,
                'fragment_count': len(fragments),
                'path_count': len(selected_paths),
                'target_arrival_time': target_arrival_time,
                'estimated_convergence_ms': self.convergence_tolerance_ms
            })
        
        self.logger.debug(f"Started MIMO transmission {transmission_id} with {len(fragments)} fragments across {len(selected_paths)} paths")
        
        return transmission_id
    
    async def _select_optimal_paths(self, source: str, dest: str, fragment_count: int) -> List[RoutingPath]:
        """Select optimal paths for fragment distribution"""
        
        available_paths = []
        if source in self.path_matrix and dest in self.path_matrix[source]:
            path_ids = self.path_matrix[source][dest]
            available_paths = [self.routing_paths[pid] for pid in path_ids if self.routing_paths[pid].is_active]
        
        if not available_paths:
            return []
        
        # Apply path selection strategy
        if self.path_selection_strategy == "latency_optimized":
            selected = self._select_low_latency_paths(available_paths, fragment_count)
        elif self.path_selection_strategy == "reliability_optimized":
            selected = self._select_high_reliability_paths(available_paths, fragment_count)
        elif self.path_selection_strategy == "load_balanced":
            selected = self._select_load_balanced_paths(available_paths, fragment_count)
        else:
            selected = self._select_quality_optimized_paths(available_paths, fragment_count)
        
        return selected
    
    def _select_low_latency_paths(self, paths: List[RoutingPath], fragment_count: int) -> List[RoutingPath]:
        """Select paths optimized for low latency"""
        # Sort by average latency
        paths.sort(key=lambda p: p.average_latency)
        
        # Select up to max_paths_per_transmission or fragment_count, whichever is smaller
        max_paths = min(self.max_paths_per_transmission, fragment_count, len(paths))
        return paths[:max_paths]
    
    def _select_high_reliability_paths(self, paths: List[RoutingPath], fragment_count: int) -> List[RoutingPath]:
        """Select paths optimized for reliability"""
        # Sort by success rate and reliability
        paths.sort(key=lambda p: p.success_rate * p.reliability, reverse=True)
        
        max_paths = min(self.max_paths_per_transmission, fragment_count, len(paths))
        return paths[:max_paths]
    
    def _select_load_balanced_paths(self, paths: List[RoutingPath], fragment_count: int) -> List[RoutingPath]:
        """Select paths with load balancing consideration"""
        # Sort by current load (ascending) and quality
        paths.sort(key=lambda p: p.current_load - p.quality_score)
        
        max_paths = min(self.max_paths_per_transmission, fragment_count, len(paths))
        return paths[:max_paths]
    
    def _select_quality_optimized_paths(self, paths: List[RoutingPath], fragment_count: int) -> List[RoutingPath]:
        """Select paths optimized for overall quality"""
        # Sort by quality score
        paths.sort(key=lambda p: p.quality_score, reverse=True)
        
        max_paths = min(self.max_paths_per_transmission, fragment_count, len(paths))
        return paths[:max_paths]
    
    async def _distribute_fragments_across_paths(self, fragments: List, paths: List[RoutingPath], mimo_tx: MIMOTransmission):
        """Distribute fragments across selected paths with timing coordination"""
        
        # Assign fragments to paths (round-robin distribution)
        path_assignments = {}
        for i, fragment in enumerate(fragments):
            path = paths[i % len(paths)]
            if path.path_id not in path_assignments:
                path_assignments[path.path_id] = []
            path_assignments[path.path_id].append(fragment)
        
        # Calculate transmission timing for each path to achieve convergence
        target_time = mimo_tx.target_arrival_time
        
        for path_id, path_fragments in path_assignments.items():
            path = self.routing_paths[path_id]
            
            # Calculate send time to achieve target arrival
            transmission_delay = path.estimated_latency / 1000  # Convert to seconds
            send_time = target_time - transmission_delay
            
            # Ensure send time is in the future
            current_time = time.time()
            if send_time <= current_time:
                send_time = current_time + 0.001  # 1ms minimum delay
            
            # Schedule fragment transmission
            for fragment in path_fragments:
                delivery = FragmentDelivery(
                    fragment_id=fragment.fragment_id,
                    path_id=path_id,
                    send_time=send_time,
                    expected_arrival_time=target_time
                )
                mimo_tx.deliveries.append(delivery)
                
                # Simulate fragment transmission
                asyncio.create_task(self._transmit_fragment(fragment, path, delivery))
                
                # Update path utilization
                path.current_load += 0.1  # Simulate load increase
                path.fragments_sent += 1
    
    async def _transmit_fragment(self, fragment, path: RoutingPath, delivery: FragmentDelivery):
        """Simulate fragment transmission through a routing path"""
        
        # Wait until send time
        current_time = time.time()
        if delivery.send_time > current_time:
            await asyncio.sleep(delivery.send_time - current_time)
        
        try:
            # Simulate transmission with realistic delays and potential failures
            actual_latency = self._simulate_path_transmission(path)
            
            # Calculate actual arrival time
            actual_send_time = time.time()
            actual_arrival_time = actual_send_time + (actual_latency / 1000)
            
            # Simulate transmission delay
            await asyncio.sleep(actual_latency / 1000)
            
            # Check for transmission success
            success_probability = path.success_rate * path.reliability
            transmission_success = random.random() < success_probability
            
            if transmission_success:
                # Successful delivery
                delivery.actual_arrival_time = time.time()
                delivery.delivery_success = True
                delivery.transmission_latency = actual_latency
                delivery.path_latency = path.estimated_latency
                delivery.jitter = actual_latency - path.estimated_latency
                
                # Update path statistics
                path.successful_deliveries += 1
                path.total_latency += actual_latency
                
                # Update transmission tracking
                transmission = None
                for tx in self.active_transmissions.values():
                    if delivery.fragment_id in tx.fragments:
                        tx.completed_fragments.add(delivery.fragment_id)
                        transmission = tx
                        break
                
                self.performance_metrics['fragments_delivered'] += 1
                
                self.logger.debug(f"Fragment {delivery.fragment_id} delivered via {path.path_id} in {actual_latency:.1f}ms")
                
            else:
                # Failed delivery
                delivery.delivery_success = False
                path.failed_deliveries += 1
                
                # Update transmission tracking
                for tx in self.active_transmissions.values():
                    if delivery.fragment_id in tx.fragments:
                        tx.failed_fragments.add(delivery.fragment_id)
                        break
                
                self.logger.warning(f"Fragment {delivery.fragment_id} transmission failed via {path.path_id}")
            
        except Exception as e:
            self.logger.error(f"Error transmitting fragment {delivery.fragment_id}: {e}")
            delivery.delivery_success = False
            path.failed_deliveries += 1
        
        finally:
            # Reduce path load
            path.current_load = max(0, path.current_load - 0.1)
            path.last_used_time = time.time()
    
    def _simulate_path_transmission(self, path: RoutingPath) -> float:
        """Simulate realistic path transmission with jitter and variations"""
        
        base_latency = path.estimated_latency
        
        # Add realistic jitter based on path characteristics
        jitter_std = base_latency * 0.1  # 10% jitter
        jitter = random.gauss(0, jitter_std)
        
        # Add congestion effects
        congestion_factor = 1.0 + (path.congestion_level * 0.5)  # Up to 50% increase
        
        # Add load effects
        load_factor = 1.0 + (path.current_load * 0.3)  # Up to 30% increase
        
        # Calculate final latency
        actual_latency = base_latency * congestion_factor * load_factor + jitter
        
        # Ensure minimum latency
        return max(actual_latency, base_latency * 0.5)
    
    async def _monitor_transmissions(self):
        """Monitor active transmissions and update completion status"""
        while self.is_running:
            try:
                current_time = time.time()
                completed_transmissions = []
                
                for tx_id, transmission in self.active_transmissions.items():
                    # Check if transmission is complete
                    if not transmission.is_complete:
                        total_fragments = len(transmission.fragments)
                        completed_fragments = len(transmission.completed_fragments)
                        
                        # Check for completion or timeout
                        is_complete = (completed_fragments == total_fragments or
                                     current_time > transmission.target_arrival_time + 5.0)  # 5 second timeout
                        
                        if is_complete:
                            transmission.is_complete = True
                            transmission.completion_time = current_time
                            transmission.success = transmission.is_successful
                            
                            # Update performance metrics
                            self.performance_metrics['transmissions_completed'] += 1
                            
                            if transmission.success:
                                # Calculate convergence quality
                                convergence = transmission.convergence_quality
                                self.performance_metrics['average_convergence_quality'] = (
                                    0.9 * self.performance_metrics['average_convergence_quality'] +
                                    0.1 * convergence
                                )
                                
                                # Calculate latency improvement vs single path
                                baseline_latency = max(self.routing_paths[pid].estimated_latency 
                                                     for pid in transmission.paths)
                                actual_completion_time = transmission.completion_time - transmission.start_time
                                improvement = baseline_latency / 1000 - actual_completion_time
                                
                                if improvement > 0:
                                    self.performance_metrics['latency_improvements'].append(improvement)
                            
                            completed_transmissions.append(tx_id)
                            
                            # Log completion
                            if self.data_collector:
                                await self.data_collector.log_mimo_completion({
                                    'timestamp': current_time,
                                    'transmission_id': tx_id,
                                    'success': transmission.success,
                                    'completion_ratio': transmission.completion_ratio,
                                    'convergence_quality': transmission.convergence_quality,
                                    'total_time': current_time - transmission.start_time
                                })
                
                # Clean up completed transmissions
                for tx_id in completed_transmissions:
                    del self.active_transmissions[tx_id]
                
                await asyncio.sleep(0.1)  # Monitor every 100ms
                
            except Exception as e:
                self.logger.error(f"Error monitoring transmissions: {e}")
                await asyncio.sleep(1.0)
    
    async def _optimize_path_selection(self):
        """Continuously optimize path selection based on performance"""
        while self.is_running:
            try:
                # Update path quality metrics
                for path in self.routing_paths.values():
                    # Update congestion estimation
                    if path.fragments_sent > 0:
                        load_factor = min(path.current_load, 1.0)
                        path.congestion_level = 0.8 * path.congestion_level + 0.2 * load_factor
                    
                    # Update reliability estimates
                    if path.fragments_sent > 10:  # Minimum sample size
                        recent_success_rate = path.success_rate
                        path.reliability = 0.9 * path.reliability + 0.1 * recent_success_rate
                
                # Calculate bandwidth efficiency
                total_bandwidth_used = sum(p.current_load * p.bandwidth_mbps for p in self.routing_paths.values())
                total_bandwidth_available = sum(p.bandwidth_mbps for p in self.routing_paths.values())
                
                if total_bandwidth_available > 0:
                    self.performance_metrics['bandwidth_efficiency'] = total_bandwidth_used / total_bandwidth_available
                
                await asyncio.sleep(5.0)  # Optimize every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in path optimization: {e}")
                await asyncio.sleep(10.0)
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        
        # Path utilization statistics
        path_utilizations = {}
        for path_id, path in self.routing_paths.items():
            path_utilizations[path_id] = {
                'current_load': path.current_load,
                'fragments_sent': path.fragments_sent,
                'success_rate': path.success_rate,
                'average_latency': path.average_latency,
                'quality_score': path.quality_score
            }
        
        # Overall performance
        total_transmissions = max(1, self.performance_metrics['transmissions_completed'])
        success_rate = (self.performance_metrics['transmissions_completed'] / 
                       max(1, self.performance_metrics['transmissions_started']))
        
        # Latency improvements
        latency_improvements = list(self.performance_metrics['latency_improvements'])
        avg_improvement = np.mean(latency_improvements) if latency_improvements else 0.0
        
        return {
            'active_transmissions': len(self.active_transmissions),
            'transmissions_started': self.performance_metrics['transmissions_started'],
            'transmissions_completed': self.performance_metrics['transmissions_completed'],
            'transmission_success_rate': success_rate,
            'fragments_routed': self.performance_metrics['fragments_routed'],
            'fragments_delivered': self.performance_metrics['fragments_delivered'],
            'fragment_delivery_rate': (self.performance_metrics['fragments_delivered'] / 
                                     max(1, self.performance_metrics['fragments_routed'])),
            'average_convergence_quality': self.performance_metrics['average_convergence_quality'],
            'bandwidth_efficiency': self.performance_metrics['bandwidth_efficiency'],
            'average_latency_improvement': avg_improvement,
            'total_paths': len(self.routing_paths),
            'active_paths': len([p for p in self.routing_paths.values() if p.is_active]),
            'path_utilization': path_utilizations
        }
    
    def stop(self):
        """Stop MIMO routing service"""
        self.is_running = False
        self.logger.info("MIMO router stopped")
