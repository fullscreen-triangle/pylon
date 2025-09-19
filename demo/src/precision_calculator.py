"""
Precision Calculator Module

Implements the core precision-by-difference calculations that form the foundation
of the Sango Rine Shumba temporal coordination framework.

The precision-by-difference calculation ΔP_i(k) = T_ref(k) - t_i(k) transforms
temporal variations from errors into coordination resources, enabling enhanced
network synchronization beyond individual component capabilities.
"""

import asyncio
import time
import math
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import numpy as np
from collections import deque

@dataclass
class PrecisionMeasurement:
    """Represents a single precision-by-difference measurement"""
    
    node_id: str
    measurement_time: float
    atomic_reference: float
    local_measurement: float
    precision_difference: float
    measurement_quality: float
    
    # Metadata
    reference_source: str = "unknown"
    local_precision_level: str = "microsecond"
    environmental_factors: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived properties"""
        self.age_seconds = 0.0
        self.is_valid = abs(self.precision_difference) < 1.0  # 1 second threshold
        
        # Calculate measurement confidence based on various factors
        self.confidence = self._calculate_confidence()
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence in this measurement"""
        confidence = 1.0
        
        # Reduce confidence for large differences
        if abs(self.precision_difference) > 0.1:  # 100ms
            confidence *= 0.8
        if abs(self.precision_difference) > 0.01:  # 10ms
            confidence *= 0.9
            
        # Factor in measurement quality
        confidence *= self.measurement_quality
        
        # Consider environmental factors
        if 'power_grid_interference' in self.environmental_factors:
            confidence *= (1.0 - self.environmental_factors['power_grid_interference'])
        
        return max(0.1, min(1.0, confidence))
    
    def update_age(self):
        """Update the age of this measurement"""
        self.age_seconds = time.time() - self.measurement_time
    
    @property
    def is_fresh(self, max_age_seconds: float = 5.0) -> bool:
        """Check if measurement is still fresh"""
        self.update_age()
        return self.age_seconds <= max_age_seconds

@dataclass
class CoordinationMatrix:
    """Represents the temporal coordination matrix for network synchronization"""
    
    matrix_id: str
    generation_time: float
    measurements: List[PrecisionMeasurement]
    
    # Calculated properties
    temporal_window_start: float = 0.0
    temporal_window_end: float = 0.0
    temporal_window_duration: float = 0.0
    coordination_accuracy: float = 0.0
    synchronization_quality: float = 0.0
    
    def __post_init__(self):
        """Calculate matrix properties"""
        if self.measurements:
            self._calculate_temporal_window()
            self._calculate_coordination_metrics()
    
    def _calculate_temporal_window(self):
        """Calculate temporal coherence window boundaries"""
        if not self.measurements:
            return
        
        precision_differences = [m.precision_difference for m in self.measurements]
        
        # Window boundaries: [T_ref + min(ΔP), T_ref + max(ΔP)]
        min_precision = min(precision_differences)
        max_precision = max(precision_differences)
        
        reference_time = self.measurements[0].atomic_reference
        
        self.temporal_window_start = reference_time + min_precision
        self.temporal_window_end = reference_time + max_precision
        self.temporal_window_duration = self.temporal_window_end - self.temporal_window_start
    
    def _calculate_coordination_metrics(self):
        """Calculate coordination accuracy and quality metrics"""
        if not self.measurements:
            return
        
        precision_differences = [m.precision_difference for m in self.measurements]
        confidences = [m.confidence for m in self.measurements]
        
        # Coordination accuracy based on precision difference variance
        if len(precision_differences) > 1:
            precision_std = statistics.stdev(precision_differences)
            self.coordination_accuracy = 1.0 / (1.0 + precision_std)
        else:
            self.coordination_accuracy = 1.0
        
        # Synchronization quality based on measurement confidence
        self.synchronization_quality = statistics.mean(confidences) if confidences else 0.0
    
    def get_temporal_coordinate(self, progress: float) -> float:
        """Get temporal coordinate within the window (progress 0.0 to 1.0)"""
        if self.temporal_window_duration == 0:
            return self.temporal_window_start
        
        return self.temporal_window_start + (progress * self.temporal_window_duration)
    
    def is_coordinate_within_window(self, coordinate: float) -> bool:
        """Check if a temporal coordinate falls within this matrix's window"""
        return self.temporal_window_start <= coordinate <= self.temporal_window_end

class PrecisionCalculator:
    """
    Precision-by-difference calculation engine
    
    This class implements the core mathematical framework of Sango Rine Shumba:
    - Continuous precision-by-difference calculations ΔP_i(k) = T_ref(k) - t_i(k)
    - Temporal coordination matrix generation
    - Enhanced precision through difference calculations
    - Network synchronization quality monitoring
    """
    
    def __init__(self, atomic_clock, network_simulator, data_collector=None):
        """Initialize precision calculator"""
        self.atomic_clock = atomic_clock
        self.network_simulator = network_simulator
        self.data_collector = data_collector
        self.logger = logging.getLogger(__name__)
        
        # Measurement storage
        self.measurements: Dict[str, deque] = {}  # Per-node measurement history
        self.coordination_matrices: List[CoordinationMatrix] = []
        self.current_matrix: Optional[CoordinationMatrix] = None
        
        # Calculation parameters
        self.measurement_interval = 0.05  # 50ms default
        self.matrix_generation_interval = 0.1  # 100ms
        self.measurement_history_size = 1000
        self.outlier_threshold = 3.0  # Standard deviations
        
        # Performance metrics
        self.performance_metrics = {
            'total_measurements': 0,
            'valid_measurements': 0,
            'outlier_measurements': 0,
            'coordination_matrices_generated': 0,
            'average_precision_enhancement': 0.0,
            'sync_quality_history': deque(maxlen=1000)
        }
        
        # Calculation state
        self.is_running = False
        self.last_matrix_time = 0.0
        
        self.logger.info("Precision calculator initialized")
    
    async def start_continuous_calculation(self):
        """Start continuous precision-by-difference calculations"""
        self.logger.info("Starting continuous precision-by-difference calculations...")
        self.is_running = True
        
        # Initialize measurement queues for each node
        for node_id in self.network_simulator.nodes:
            self.measurements[node_id] = deque(maxlen=self.measurement_history_size)
        
        # Start background calculation tasks
        calculation_task = asyncio.create_task(self._calculation_loop())
        matrix_task = asyncio.create_task(self._matrix_generation_loop())
        analysis_task = asyncio.create_task(self._analysis_loop())
        
        try:
            await asyncio.gather(calculation_task, matrix_task, analysis_task)
        except asyncio.CancelledError:
            self.logger.info("Precision calculation tasks cancelled")
    
    async def _calculation_loop(self):
        """Main calculation loop for precision-by-difference measurements"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Get atomic clock reference
                reference_data = await self.atomic_clock.get_reference_time()
                if not reference_data:
                    await asyncio.sleep(self.measurement_interval)
                    continue
                
                atomic_reference = reference_data['timestamp']
                reference_source = reference_data['source']
                
                # Calculate precision differences for all nodes
                for node_id, node in self.network_simulator.nodes.items():
                    try:
                        # Get local measurement from node
                        local_data = await self.network_simulator.get_precision_measurement(node_id)
                        local_measurement = local_data['local_measurement']
                        
                        # Calculate precision difference: ΔP_i(k) = T_ref(k) - t_i(k)
                        precision_difference = atomic_reference - local_measurement
                        
                        # Assess measurement quality
                        measurement_quality = self._assess_measurement_quality(
                            node, local_data, reference_data
                        )
                        
                        # Create measurement record
                        measurement = PrecisionMeasurement(
                            node_id=node_id,
                            measurement_time=current_time,
                            atomic_reference=atomic_reference,
                            local_measurement=local_measurement,
                            precision_difference=precision_difference,
                            measurement_quality=measurement_quality,
                            reference_source=reference_source,
                            local_precision_level=node.precision_level,
                            environmental_factors=self._get_environmental_factors(node)
                        )
                        
                        # Validate and store measurement
                        if self._validate_measurement(measurement, node_id):
                            self.measurements[node_id].append(measurement)
                            self.performance_metrics['valid_measurements'] += 1
                            
                            # Log measurement data
                            if self.data_collector:
                                await self.data_collector.log_precision_measurement({
                                    'timestamp': current_time,
                                    'node_id': node_id,
                                    'atomic_reference': atomic_reference,
                                    'local_measurement': local_measurement,
                                    'precision_difference': precision_difference,
                                    'measurement_quality': measurement_quality,
                                    'confidence': measurement.confidence,
                                    'reference_source': reference_source
                                })
                        else:
                            self.performance_metrics['outlier_measurements'] += 1
                        
                        self.performance_metrics['total_measurements'] += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to calculate precision for node {node_id}: {e}")
                
                await asyncio.sleep(self.measurement_interval)
                
            except Exception as e:
                self.logger.error(f"Error in calculation loop: {e}")
                await asyncio.sleep(1.0)
    
    def _assess_measurement_quality(self, node, local_data, reference_data) -> float:
        """Assess the quality of a measurement"""
        quality = 1.0
        
        # Factor in atomic clock quality
        if 'uncertainty' in reference_data:
            reference_uncertainty = reference_data['uncertainty']
            if reference_uncertainty > 1e-6:  # More than 1μs uncertainty
                quality *= 0.9
            if reference_uncertainty > 1e-3:  # More than 1ms uncertainty
                quality *= 0.7
        
        # Factor in local measurement quality
        if 'measurement_jitter' in local_data:
            jitter = abs(local_data['measurement_jitter'])
            if jitter > 1e-6:
                quality *= 0.95
            if jitter > 1e-3:
                quality *= 0.8
        
        # Factor in node load
        if node.current_load > 0.8:
            quality *= 0.9
        if node.current_load > 0.95:
            quality *= 0.7
        
        # Factor in environmental conditions
        quality *= node.environmental_factor
        
        return max(0.1, min(1.0, quality))
    
    def _get_environmental_factors(self, node) -> Dict[str, float]:
        """Get environmental factors affecting measurement"""
        factors = {}
        
        # Power grid interference
        if node.power_grid_frequency == 50:
            interference = 0.02 * abs(math.sin(2 * math.pi * 50 * time.time()))
        else:  # 60Hz
            interference = 0.03 * abs(math.sin(2 * math.pi * 60 * time.time()))
        
        factors['power_grid_interference'] = min(interference, 0.1)
        
        # Environmental factor from node
        factors['atmospheric_conditions'] = 1.0 - node.environmental_factor
        
        # Infrastructure stability
        quality_factors = {
            'premium': 0.05,
            'high': 0.1,
            'moderate': 0.2,
            'variable': 0.3
        }
        factors['infrastructure_instability'] = quality_factors.get(node.connection_quality, 0.15)
        
        return factors
    
    def _validate_measurement(self, measurement: PrecisionMeasurement, node_id: str) -> bool:
        """Validate a precision measurement for outliers and errors"""
        
        # Basic sanity checks
        if abs(measurement.precision_difference) > 10.0:  # 10 seconds is clearly wrong
            self.logger.warning(f"Extreme precision difference rejected: {measurement.precision_difference}s")
            return False
        
        if measurement.measurement_quality < 0.1:
            return False
        
        # Check against recent measurements for outlier detection
        if len(self.measurements[node_id]) >= 10:
            recent_measurements = list(self.measurements[node_id])[-10:]
            recent_differences = [m.precision_difference for m in recent_measurements]
            
            mean_diff = statistics.mean(recent_differences)
            if len(recent_differences) > 1:
                std_diff = statistics.stdev(recent_differences)
                
                # Z-score outlier detection
                z_score = abs(measurement.precision_difference - mean_diff) / std_diff if std_diff > 0 else 0
                if z_score > self.outlier_threshold:
                    self.logger.debug(f"Outlier measurement rejected for {node_id}: z-score={z_score:.2f}")
                    return False
        
        return True
    
    async def _matrix_generation_loop(self):
        """Generate coordination matrices from precision measurements"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check if it's time to generate a new matrix
                if current_time - self.last_matrix_time >= self.matrix_generation_interval:
                    matrix = await self._generate_coordination_matrix()
                    if matrix:
                        self.coordination_matrices.append(matrix)
                        self.current_matrix = matrix
                        self.last_matrix_time = current_time
                        
                        # Keep only recent matrices
                        if len(self.coordination_matrices) > 100:
                            self.coordination_matrices = self.coordination_matrices[-50:]
                        
                        self.performance_metrics['coordination_matrices_generated'] += 1
                        
                        # Log matrix data
                        if self.data_collector:
                            await self.data_collector.log_coordination_matrix({
                                'timestamp': current_time,
                                'matrix_id': matrix.matrix_id,
                                'num_measurements': len(matrix.measurements),
                                'temporal_window_duration': matrix.temporal_window_duration,
                                'coordination_accuracy': matrix.coordination_accuracy,
                                'synchronization_quality': matrix.synchronization_quality
                            })
                
                await asyncio.sleep(0.01)  # Check every 10ms
                
            except Exception as e:
                self.logger.error(f"Error in matrix generation loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _generate_coordination_matrix(self) -> Optional[CoordinationMatrix]:
        """Generate a coordination matrix from current measurements"""
        
        current_measurements = []
        current_time = time.time()
        
        # Collect most recent valid measurements from each node
        for node_id, measurement_queue in self.measurements.items():
            if measurement_queue:
                # Get the most recent fresh measurement
                for measurement in reversed(measurement_queue):
                    if measurement.is_fresh(max_age_seconds=1.0):  # Within last second
                        current_measurements.append(measurement)
                        break
        
        # Need at least 3 nodes for meaningful coordination
        if len(current_measurements) < 3:
            return None
        
        # Create coordination matrix
        matrix_id = f"matrix_{int(current_time * 1000000)}"  # Microsecond precision ID
        matrix = CoordinationMatrix(
            matrix_id=matrix_id,
            generation_time=current_time,
            measurements=current_measurements
        )
        
        self.logger.debug(f"Generated coordination matrix {matrix_id} with {len(current_measurements)} measurements")
        return matrix
    
    async def _analysis_loop(self):
        """Analyze precision enhancement and synchronization quality"""
        while self.is_running:
            try:
                if self.current_matrix and len(self.coordination_matrices) > 5:
                    # Calculate precision enhancement
                    enhancement = self._calculate_precision_enhancement()
                    if enhancement > 0:
                        self.performance_metrics['average_precision_enhancement'] = (
                            0.9 * self.performance_metrics['average_precision_enhancement'] + 
                            0.1 * enhancement
                        )
                    
                    # Track synchronization quality
                    sync_quality = self.current_matrix.synchronization_quality
                    self.performance_metrics['sync_quality_history'].append(sync_quality)
                
                await asyncio.sleep(5.0)  # Analyze every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
    
    def _calculate_precision_enhancement(self) -> float:
        """Calculate precision enhancement achieved through difference calculations"""
        
        if not self.current_matrix or len(self.current_matrix.measurements) < 2:
            return 0.0
        
        measurements = self.current_matrix.measurements
        
        # Calculate individual node precision (based on jitter)
        individual_precisions = []
        for measurement in measurements:
            node = self.network_simulator.nodes[measurement.node_id]
            # Estimate individual precision from jitter characteristics
            individual_precision = node.base_jitter_ms / 1000  # Convert to seconds
            individual_precisions.append(individual_precision)
        
        # Calculate coordination precision (from precision differences)
        precision_differences = [m.precision_difference for m in measurements]
        if len(precision_differences) > 1:
            coordination_precision = statistics.stdev(precision_differences)
        else:
            coordination_precision = abs(precision_differences[0])
        
        # Calculate enhancement as ratio
        avg_individual_precision = statistics.mean(individual_precisions)
        if coordination_precision > 0 and avg_individual_precision > 0:
            enhancement = avg_individual_precision / coordination_precision
            return max(1.0, enhancement)  # Enhancement of at least 1.0 (no degradation)
        
        return 1.0
    
    def get_current_coordination_matrix(self) -> Optional[CoordinationMatrix]:
        """Get the current coordination matrix"""
        return self.current_matrix
    
    def get_node_precision_history(self, node_id: str, max_points: int = 100) -> List[PrecisionMeasurement]:
        """Get precision measurement history for a specific node"""
        if node_id not in self.measurements:
            return []
        
        measurements = list(self.measurements[node_id])
        return measurements[-max_points:] if len(measurements) > max_points else measurements
    
    def get_precision_statistics(self) -> Dict[str, Any]:
        """Get comprehensive precision calculation statistics"""
        
        current_time = time.time()
        
        # Overall statistics
        total_measurements = self.performance_metrics['total_measurements']
        valid_rate = (self.performance_metrics['valid_measurements'] / total_measurements 
                     if total_measurements > 0 else 0.0)
        outlier_rate = (self.performance_metrics['outlier_measurements'] / total_measurements 
                       if total_measurements > 0 else 0.0)
        
        # Current precision differences
        current_differences = []
        current_confidences = []
        if self.current_matrix:
            for measurement in self.current_matrix.measurements:
                current_differences.append(measurement.precision_difference)
                current_confidences.append(measurement.confidence)
        
        # Synchronization quality trend
        sync_quality_history = list(self.performance_metrics['sync_quality_history'])
        avg_sync_quality = statistics.mean(sync_quality_history) if sync_quality_history else 0.0
        
        return {
            'total_measurements': total_measurements,
            'valid_measurement_rate': valid_rate,
            'outlier_rate': outlier_rate,
            'coordination_matrices_generated': self.performance_metrics['coordination_matrices_generated'],
            'average_precision_enhancement': self.performance_metrics['average_precision_enhancement'],
            'average_sync_quality': avg_sync_quality,
            'current_matrix_id': self.current_matrix.matrix_id if self.current_matrix else None,
            'current_precision_differences': current_differences,
            'current_measurement_confidences': current_confidences,
            'temporal_window_duration': self.current_matrix.temporal_window_duration if self.current_matrix else 0.0,
            'coordination_accuracy': self.current_matrix.coordination_accuracy if self.current_matrix else 0.0,
            'active_nodes': len([q for q in self.measurements.values() if len(q) > 0]),
            'calculation_uptime': current_time - (current_time if not hasattr(self, '_start_time') else self._start_time)
        }
    
    def stop(self):
        """Stop precision calculations"""
        self.is_running = False
        self.logger.info("Precision calculator stopped")
