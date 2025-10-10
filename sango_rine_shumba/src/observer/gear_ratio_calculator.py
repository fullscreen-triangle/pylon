import numpy as np
import time
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

class GearRatioCalculator:
    """Calculates gear ratios between hierarchical levels"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Eight-scale oscillatory hierarchy frequencies (Hz)
        self.scale_frequencies = {
            1: 1e13,   # Quantum Network Coherence (10^12-10^15 Hz)
            2: 1e7,    # Atomic Clock Synchronization (10^6-10^9 Hz)  
            3: 1e2,    # Precision-by-Difference Calculations (10^1-10^4 Hz)
            4: 1e0,    # Network Fragment Coordination (10^-1-10^1 Hz)
            5: 1e-1,   # Spatio-Temporal Integration (10^-2-10^-1 Hz)
            6: 1e-2,   # Distributed System Coordination (10^-3-10^-2 Hz)
            7: 1e-3,   # Network Ecosystem Integration (10^-4-10^-3 Hz)
            8: 1e-5    # Cultural Network Dynamics (10^-6-10^-4 Hz)
        }
        
        # Pre-compute compound ratio table for O(1) lookup
        self.compound_ratio_table = self._precompute_compound_ratios()
        
        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0.0
        self.start_time = time.time()
        
        self.logger.info("GearRatioCalculator initialized with 8-scale oscillatory hierarchy")
    
    def calculate_gear_ratio(self, source_freq: float, target_freq: float) -> float:
        """R_ij = ω_i / ω_j - Direct frequency ratio"""
        start_time = time.time()
        
        if target_freq == 0:
            raise ValueError("Target frequency cannot be zero")
        
        ratio = source_freq / target_freq
        
        # Track performance
        calculation_time = time.time() - start_time
        self.calculation_count += 1
        self.total_calculation_time += calculation_time
        
        self.logger.debug(f"Calculated gear ratio: {source_freq:.2e}/{target_freq:.2e} = {ratio:.6f}")
        
        return ratio
    
    def get_compound_ratio(self, source_scale: int, target_scale: int) -> float:
        """R_i→j = ∏ R_k,k+1 for direct navigation - O(1) lookup"""
        if source_scale < 1 or source_scale > 8 or target_scale < 1 or target_scale > 8:
            raise ValueError("Scale indices must be between 1 and 8")
        
        return self.compound_ratio_table[source_scale][target_scale]
    
    def _precompute_compound_ratios(self) -> Dict[int, Dict[int, float]]:
        """Pre-compute all compound ratios for O(1) lookup"""
        self.logger.info("Pre-computing compound ratio table...")
        
        compound_ratios = {}
        
        for source in range(1, 9):  # Scales 1-8
            compound_ratios[source] = {}
            for target in range(1, 9):
                if source == target:
                    compound_ratios[source][target] = 1.0
                else:
                    # Calculate compound ratio through intermediate levels
                    source_freq = self.scale_frequencies[source]
                    target_freq = self.scale_frequencies[target]
                    compound_ratios[source][target] = source_freq / target_freq
        
        self.logger.info("Compound ratio table pre-computation complete")
        return compound_ratios
    
    def validate_gear_ratio_transitivity(self, scale_i: int, scale_j: int, scale_k: int) -> bool:
        """Validate R_ik = R_ij * R_jk (transitivity property)"""
        try:
            r_ij = self.get_compound_ratio(scale_i, scale_j)
            r_jk = self.get_compound_ratio(scale_j, scale_k)
            r_ik = self.get_compound_ratio(scale_i, scale_k)
            
            calculated_r_ik = r_ij * r_jk
            
            # Allow small numerical errors
            tolerance = 1e-10
            is_valid = abs(r_ik - calculated_r_ik) < tolerance
            
            if not is_valid:
                self.logger.warning(f"Transitivity validation failed: R_{scale_i}{scale_k}={r_ik:.10f}, "
                                  f"R_{scale_i}{scale_j}*R_{scale_j}{scale_k}={calculated_r_ik:.10f}")
            
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Error validating transitivity: {e}")
            return False
    
    def calculate_frequency_separation_factor(self, scale_i: int, scale_j: int) -> float:
        """Calculate how many orders of magnitude separate two scales"""
        freq_i = self.scale_frequencies[scale_i]
        freq_j = self.scale_frequencies[scale_j]
        
        return abs(np.log10(freq_i) - np.log10(freq_j))
    
    def get_optimal_navigation_path(self, source_scale: int, target_scale: int) -> List[int]:
        """Get optimal path for hierarchical navigation (though direct O(1) is always optimal)"""
        # In gear ratio systems, direct navigation is always O(1) and optimal
        # But this method can be useful for understanding hierarchical relationships
        
        if source_scale == target_scale:
            return [source_scale]
        
        # Direct path is always optimal
        return [source_scale, target_scale]
    
    def calculate_amplification_factor(self, encoding_layers: List[float]) -> float:
        """Calculate semantic distance amplification factor Γ = ∏ γ_i"""
        if not encoding_layers:
            return 1.0
        
        amplification = 1.0
        for layer_factor in encoding_layers:
            amplification *= layer_factor
        
        self.logger.debug(f"Calculated amplification factor: {amplification:.2f} from {len(encoding_layers)} layers")
        return amplification
    
    def extract_gear_ratio_from_ambiguous_segment(self, segment_data: bytes, reference_frequency: float) -> float:
        """Extract gear ratio from compression-resistant ambiguous data segment"""
        try:
            # Use segment entropy and hash to derive gear ratio
            import hashlib
            
            # Calculate segment hash for deterministic ratio extraction
            segment_hash = hashlib.sha256(segment_data).hexdigest()
            hash_int = int(segment_hash[:16], 16)  # Use first 16 hex chars
            
            # Normalize to range [0.1, 10.0] to get reasonable gear ratios
            ratio_factor = 0.1 + (hash_int % 10000) / 10000.0 * 9.9
            
            # Calculate gear ratio relative to reference frequency
            extracted_ratio = ratio_factor * reference_frequency / (reference_frequency + 1.0)
            
            self.logger.debug(f"Extracted gear ratio {extracted_ratio:.6f} from {len(segment_data)}-byte segment")
            return extracted_ratio
            
        except Exception as e:
            self.logger.error(f"Error extracting gear ratio from segment: {e}")
            return 1.0  # Default ratio
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get calculator performance statistics"""
        current_time = time.time()
        runtime = current_time - self.start_time
        
        avg_calculation_time = self.total_calculation_time / max(1, self.calculation_count)
        calculations_per_second = self.calculation_count / max(1, runtime)
        
        return {
            'total_calculations': self.calculation_count,
            'total_calculation_time': self.total_calculation_time,
            'average_calculation_time': avg_calculation_time,
            'calculations_per_second': calculations_per_second,
            'runtime_seconds': runtime,
            'scale_frequencies': self.scale_frequencies,
            'compound_ratio_table_size': len(self.compound_ratio_table) * len(self.compound_ratio_table[1])
        }
    
    def export_compound_ratio_table(self, filepath: str):
        """Export compound ratio table to JSON"""
        export_data = {
            'metadata': {
                'export_timestamp': time.time(),
                'scale_count': 8,
                'total_ratios': 64
            },
            'scale_frequencies': self.scale_frequencies,
            'compound_ratio_table': self.compound_ratio_table,
            'performance_statistics': self.get_performance_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Compound ratio table exported to {filepath}")
    
    def validate_all_transitivity_properties(self) -> Dict[str, Any]:
        """Validate transitivity for all scale combinations"""
        validation_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'failures': []
        }
        
        # Test all combinations of three scales
        for i in range(1, 9):
            for j in range(1, 9):
                for k in range(1, 9):
                    if i != j and j != k:  # Skip degenerate cases
                        validation_results['total_tests'] += 1
                        
                        if self.validate_gear_ratio_transitivity(i, j, k):
                            validation_results['passed_tests'] += 1
                        else:
                            validation_results['failed_tests'] += 1
                            validation_results['failures'].append((i, j, k))
        
        validation_results['success_rate'] = validation_results['passed_tests'] / max(1, validation_results['total_tests'])
        
        self.logger.info(f"Transitivity validation: {validation_results['passed_tests']}/{validation_results['total_tests']} passed "
                        f"({validation_results['success_rate']:.3f} success rate)")
        
        return validation_results
    
    def __repr__(self):
        return f"GearRatioCalculator(scales=8, calculations={self.calculation_count})"