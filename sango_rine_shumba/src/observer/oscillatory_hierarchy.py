import time
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from .finite_observer import FiniteObserver

class ScaleSpecificObserver(FiniteObserver):
    """Observer for specific scale in 8-level oscillatory hierarchy"""
    
    def __init__(self, scale_frequency: float, scale_id: int, scale_name: str, reference_frequency: float = 1.0):
        super().__init__(scale_frequency, max_observation_space=1024, observer_id=f"scale_{scale_id}_{scale_name}")
        self.scale_id = scale_id
        self.scale_name = scale_name
        self.reference_frequency = reference_frequency
        self.scale_specific_observations = []
        
        self.logger.info(f"Scale {scale_id} ({scale_name}) observer initialized at {scale_frequency:.2e} Hz")
        
    def extract_gear_ratio_signature(self) -> float:
        """Extract gear ratio for this scale relative to reference"""
        return self.frequency / self.reference_frequency
    
    def observe_scale_specific_signal(self, signal: Any, signal_metadata: Optional[Dict] = None) -> bool:
        """Observe signal with scale-specific processing"""
        # First use base observation
        base_observed = self.observe_signal(signal)
        
        if base_observed:
            # Add scale-specific processing
            scale_specific_record = {
                'timestamp': time.time(),
                'scale_id': self.scale_id,
                'scale_name': self.scale_name,
                'gear_ratio_signature': self.extract_gear_ratio_signature(),
                'signal_processed': True,
                'metadata': signal_metadata or {}
            }
            
            self.scale_specific_observations.append(scale_specific_record)
            
            # Keep bounded
            if len(self.scale_specific_observations) > 500:
                self.scale_specific_observations = self.scale_specific_observations[-500:]
        
        return base_observed
    
    def get_scale_statistics(self) -> Dict[str, Any]:
        """Get scale-specific statistics"""
        base_stats = self.get_observation_statistics()
        
        scale_stats = {
            'scale_id': self.scale_id,
            'scale_name': self.scale_name,
            'scale_frequency': self.frequency,
            'gear_ratio_signature': self.extract_gear_ratio_signature(),
            'scale_specific_observations': len(self.scale_specific_observations),
            'frequency_separation_orders_of_magnitude': abs(np.log10(self.frequency) - np.log10(self.reference_frequency))
        }
        
        # Merge with base statistics
        base_stats.update(scale_stats)
        return base_stats


class QuantumNetworkObserver(ScaleSpecificObserver):
    """Scale 1: Quantum Network Coherence (10^12-10^15 Hz)"""
    
    def __init__(self):
        super().__init__(1e13, 1, "QuantumNetworkCoherence", 1.0)
        self.quantum_coherence_events = []
    
    def observe_quantum_coherence(self, coherence_data: Dict) -> bool:
        """Observe quantum network coherence events"""
        if self.observe_scale_specific_signal(coherence_data, {'signal_type': 'quantum_coherence'}):
            self.quantum_coherence_events.append({
                'timestamp': time.time(),
                'coherence_strength': coherence_data.get('coherence_strength', 0.0),
                'entanglement_pairs': coherence_data.get('entanglement_pairs', 0),
                'decoherence_time': coherence_data.get('decoherence_time', 0.0)
            })
            return True
        return False


class AtomicSyncObserver(ScaleSpecificObserver):
    """Scale 2: Atomic Clock Synchronization (10^6-10^9 Hz)"""
    
    def __init__(self):
        super().__init__(1e7, 2, "AtomicClockSync", 1.0)
        self.sync_measurements = []
    
    def observe_atomic_sync(self, sync_data: Dict) -> bool:
        """Observe atomic clock synchronization events"""
        if self.observe_scale_specific_signal(sync_data, {'signal_type': 'atomic_sync'}):
            self.sync_measurements.append({
                'timestamp': time.time(),
                'clock_drift': sync_data.get('clock_drift', 0.0),
                'sync_precision': sync_data.get('sync_precision', 0.0),
                'reference_source': sync_data.get('reference_source', 'unknown')
            })
            return True
        return False


class PrecisionDiffObserver(ScaleSpecificObserver):
    """Scale 3: Precision-by-Difference Calculations (10^1-10^4 Hz)"""
    
    def __init__(self):
        super().__init__(1e2, 3, "PrecisionByDifference", 1.0)
        self.precision_calculations = []
    
    def observe_precision_calculation(self, precision_data: Dict) -> bool:
        """Observe precision-by-difference calculations"""
        if self.observe_scale_specific_signal(precision_data, {'signal_type': 'precision_calculation'}):
            self.precision_calculations.append({
                'timestamp': time.time(),
                'precision_difference': precision_data.get('precision_difference', 0.0),
                'measurement_quality': precision_data.get('measurement_quality', 0.0),
                'node_id': precision_data.get('node_id', 'unknown')
            })
            return True
        return False


class FragmentCoordObserver(ScaleSpecificObserver):
    """Scale 4: Network Fragment Coordination (10^-1-10^1 Hz)"""
    
    def __init__(self):
        super().__init__(1e0, 4, "NetworkFragmentCoord", 1.0)
        self.fragment_events = []
    
    def observe_fragment_coordination(self, fragment_data: Dict) -> bool:
        """Observe network fragment coordination events"""
        if self.observe_scale_specific_signal(fragment_data, {'signal_type': 'fragment_coordination'}):
            self.fragment_events.append({
                'timestamp': time.time(),
                'fragment_count': fragment_data.get('fragment_count', 0),
                'coordination_success': fragment_data.get('coordination_success', False),
                'temporal_window': fragment_data.get('temporal_window', 0.0)
            })
            return True
        return False


class SpatioTemporalObserver(ScaleSpecificObserver):
    """Scale 5: Spatio-Temporal Integration (10^-2-10^-1 Hz)"""
    
    def __init__(self):
        super().__init__(1e-1, 5, "SpatioTemporalIntegration", 1.0)
        self.integration_events = []
    
    def observe_spatiotemporal_integration(self, integration_data: Dict) -> bool:
        """Observe spatio-temporal integration events"""
        if self.observe_scale_specific_signal(integration_data, {'signal_type': 'spatiotemporal_integration'}):
            self.integration_events.append({
                'timestamp': time.time(),
                'spatial_coordinates': integration_data.get('spatial_coordinates', []),
                'temporal_coordinates': integration_data.get('temporal_coordinates', []),
                'integration_quality': integration_data.get('integration_quality', 0.0)
            })
            return True
        return False


class DistributedCoordObserver(ScaleSpecificObserver):
    """Scale 6: Distributed System Coordination (10^-3-10^-2 Hz)"""
    
    def __init__(self):
        super().__init__(1e-2, 6, "DistributedSystemCoord", 1.0)
        self.coordination_events = []
    
    def observe_distributed_coordination(self, coord_data: Dict) -> bool:
        """Observe distributed system coordination events"""
        if self.observe_scale_specific_signal(coord_data, {'signal_type': 'distributed_coordination'}):
            self.coordination_events.append({
                'timestamp': time.time(),
                'node_count': coord_data.get('node_count', 0),
                'coordination_latency': coord_data.get('coordination_latency', 0.0),
                'consensus_achieved': coord_data.get('consensus_achieved', False)
            })
            return True
        return False


class EcosystemIntegrationObserver(ScaleSpecificObserver):
    """Scale 7: Network Ecosystem Integration (10^-4-10^-3 Hz)"""
    
    def __init__(self):
        super().__init__(1e-3, 7, "NetworkEcosystemIntegration", 1.0)
        self.ecosystem_events = []
    
    def observe_ecosystem_integration(self, ecosystem_data: Dict) -> bool:
        """Observe network ecosystem integration events"""
        if self.observe_scale_specific_signal(ecosystem_data, {'signal_type': 'ecosystem_integration'}):
            self.ecosystem_events.append({
                'timestamp': time.time(),
                'ecosystem_health': ecosystem_data.get('ecosystem_health', 0.0),
                'integration_points': ecosystem_data.get('integration_points', []),
                'cross_system_communication': ecosystem_data.get('cross_system_communication', False)
            })
            return True
        return False


class CulturalNetworkObserver(ScaleSpecificObserver):
    """Scale 8: Cultural Network Dynamics (10^-6-10^-4 Hz)"""
    
    def __init__(self):
        super().__init__(1e-5, 8, "CulturalNetworkDynamics", 1.0)
        self.cultural_events = []
    
    def observe_cultural_dynamics(self, cultural_data: Dict) -> bool:
        """Observe cultural network dynamics events"""
        if self.observe_scale_specific_signal(cultural_data, {'signal_type': 'cultural_dynamics'}):
            self.cultural_events.append({
                'timestamp': time.time(),
                'cultural_pattern': cultural_data.get('cultural_pattern', ''),
                'network_influence': cultural_data.get('network_influence', 0.0),
                'adaptation_rate': cultural_data.get('adaptation_rate', 0.0)
            })
            return True
        return False


class OscillatoryHierarchy:
    """Manages the complete 8-scale oscillatory hierarchy"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize all 8 scale observers
        self.scale_observers = {
            1: QuantumNetworkObserver(),
            2: AtomicSyncObserver(),
            3: PrecisionDiffObserver(),
            4: FragmentCoordObserver(),
            5: SpatioTemporalObserver(),
            6: DistributedCoordObserver(),
            7: EcosystemIntegrationObserver(),
            8: CulturalNetworkObserver()
        }
        
        self.start_time = time.time()
        self.total_observations = 0
        
        self.logger.info("Oscillatory hierarchy initialized with 8 scale observers")
    
    def get_observer(self, scale_id: int) -> Optional[ScaleSpecificObserver]:
        """Get observer for specific scale"""
        return self.scale_observers.get(scale_id)
    
    def observe_at_scale(self, scale_id: int, signal: Any, signal_metadata: Optional[Dict] = None) -> bool:
        """Observe signal at specific scale"""
        observer = self.get_observer(scale_id)
        if observer:
            self.total_observations += 1
            return observer.observe_scale_specific_signal(signal, signal_metadata)
        return False
    
    def observe_across_all_scales(self, signal: Any, signal_metadata: Optional[Dict] = None) -> Dict[int, bool]:
        """Observe signal across all scales simultaneously"""
        results = {}
        for scale_id, observer in self.scale_observers.items():
            results[scale_id] = observer.observe_scale_specific_signal(signal, signal_metadata)
            self.total_observations += 1
        return results
    
    def get_hierarchy_statistics(self) -> Dict[str, Any]:
        """Get comprehensive hierarchy statistics"""
        current_time = time.time()
        runtime = current_time - self.start_time
        
        scale_stats = {}
        total_scale_observations = 0
        total_successful_observations = 0
        
        for scale_id, observer in self.scale_observers.items():
            stats = observer.get_scale_statistics()
            scale_stats[scale_id] = stats
            total_scale_observations += stats['total_observations']
            total_successful_observations += stats['successful_observations']
        
        hierarchy_stats = {
            'hierarchy_metadata': {
                'total_scales': len(self.scale_observers),
                'runtime_seconds': runtime,
                'total_observations': self.total_observations,
                'observations_per_second': self.total_observations / max(1, runtime)
            },
            'scale_statistics': scale_stats,
            'aggregate_statistics': {
                'total_scale_observations': total_scale_observations,
                'total_successful_observations': total_successful_observations,
                'overall_success_rate': total_successful_observations / max(1, total_scale_observations),
                'frequency_range_orders_of_magnitude': 18,  # 10^-5 to 10^13 = 18 orders
                'average_observations_per_scale': total_scale_observations / len(self.scale_observers)
            }
        }
        
        return hierarchy_stats
    
    def export_hierarchy_data(self, filepath: str):
        """Export complete hierarchy data to JSON"""
        export_data = {
            'export_metadata': {
                'export_timestamp': time.time(),
                'hierarchy_type': '8-scale oscillatory',
                'frequency_range': '10^-5 to 10^13 Hz'
            },
            'hierarchy_statistics': self.get_hierarchy_statistics(),
            'scale_configurations': {
                scale_id: {
                    'scale_name': observer.scale_name,
                    'frequency': observer.frequency,
                    'gear_ratio_signature': observer.extract_gear_ratio_signature()
                }
                for scale_id, observer in self.scale_observers.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Hierarchy data exported to {filepath}")
    
    def validate_frequency_hierarchy(self) -> Dict[str, Any]:
        """Validate that frequencies follow hierarchical ordering"""
        frequencies = [(scale_id, observer.frequency) for scale_id, observer in self.scale_observers.items()]
        frequencies.sort(key=lambda x: x[0])  # Sort by scale_id
        
        validation_results = {
            'hierarchy_valid': True,
            'violations': [],
            'frequency_separations': []
        }
        
        for i in range(len(frequencies) - 1):
            scale_id_1, freq_1 = frequencies[i]
            scale_id_2, freq_2 = frequencies[i + 1]
            
            # Higher scale should have higher frequency
            if freq_2 <= freq_1:
                validation_results['hierarchy_valid'] = False
                validation_results['violations'].append({
                    'scale_1': scale_id_1,
                    'scale_2': scale_id_2,
                    'frequency_1': freq_1,
                    'frequency_2': freq_2
                })
            
            # Record frequency separation
            separation = abs(np.log10(freq_2) - np.log10(freq_1))
            validation_results['frequency_separations'].append({
                'from_scale': scale_id_1,
                'to_scale': scale_id_2,
                'separation_orders_of_magnitude': separation
            })
        
        self.logger.info(f"Frequency hierarchy validation: {'PASSED' if validation_results['hierarchy_valid'] else 'FAILED'}")
        return validation_results
    
    def __repr__(self):
        return f"OscillatoryHierarchy(scales=8, observations={self.total_observations})"