import time
import hashlib
import numpy as np
from typing import Dict, Any, Optional, List
import json
import logging

class FiniteObserver:
    """Base class for observers with finite observation space"""
    
    def __init__(self, observation_frequency: float, max_observation_space: int, observer_id: str = None):
        self.frequency = observation_frequency
        self.max_space = max_observation_space  # Finite constraint
        self.current_observations: Dict[str, Any] = {}
        self.observer_id = observer_id or f"observer_{int(time.time())}"
        self.observation_history: List[Dict] = []
        self.total_observations = 0
        self.successful_observations = 0
        self.failed_observations = 0
        self.start_time = time.time()
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.observer_id}")
        
        self.logger.info(f"Finite observer initialized: freq={observation_frequency}Hz, max_space={max_observation_space}")
    
    def observe_signal(self, signal: Any) -> bool:
        """Binary utility function - either observes or doesn't"""
        self.total_observations += 1
        
        # Finite space constraint
        if len(self.current_observations) >= self.max_space:
            self.failed_observations += 1
            self.logger.debug(f"Observation failed: space limit reached ({self.max_space})")
            return False
        
        # Check if can observe at frequency
        if self._can_observe_at_frequency(signal):
            self.successful_observations += 1
            self._store_observation(signal)
            self.logger.debug(f"Signal observed successfully: {type(signal).__name__}")
            return True
        else:
            self.failed_observations += 1
            self.logger.debug(f"Signal observation failed: frequency mismatch")
            return False
    
    def _can_observe_at_frequency(self, signal: Any) -> bool:
        """Check if signal can be observed at this observer's frequency"""
        try:
            # Extract frequency characteristics from signal
            signal_freq = self._extract_signal_frequency(signal)
            
            # Allow observation if signal frequency is within reasonable range of observer frequency
            frequency_tolerance = 0.1  # 10% tolerance
            freq_diff = abs(signal_freq - self.frequency) / self.frequency
            
            return freq_diff <= frequency_tolerance
            
        except Exception as e:
            self.logger.warning(f"Error checking signal frequency: {e}")
            return False
    
    def _extract_signal_frequency(self, signal: Any) -> float:
        """Extract characteristic frequency from signal"""
        if isinstance(signal, (int, float)):
            # For numeric signals, use modulation based on value
            return self.frequency * (1.0 + 0.1 * np.sin(signal))
        
        elif isinstance(signal, str):
            # For string signals, use hash-based frequency
            signal_hash = int(hashlib.md5(signal.encode()).hexdigest()[:8], 16)
            return self.frequency * (1.0 + 0.1 * (signal_hash % 1000) / 1000.0)
        
        elif isinstance(signal, bytes):
            # For byte signals, use length and content based frequency
            content_hash = int(hashlib.sha256(signal).hexdigest()[:8], 16)
            return self.frequency * (1.0 + 0.1 * (content_hash % 1000) / 1000.0)
        
        elif isinstance(signal, dict):
            # For dict signals, use key count and content hash
            key_factor = len(signal.keys()) % 10
            content_str = json.dumps(signal, sort_keys=True)
            content_hash = int(hashlib.md5(content_str.encode()).hexdigest()[:8], 16)
            return self.frequency * (1.0 + 0.1 * key_factor / 10.0 + 0.05 * (content_hash % 100) / 100.0)
        
        else:
            # Default: use string representation
            signal_str = str(signal)
            signal_hash = int(hashlib.md5(signal_str.encode()).hexdigest()[:8], 16)
            return self.frequency * (1.0 + 0.1 * (signal_hash % 1000) / 1000.0)
    
    def _store_observation(self, signal: Any):
        """Store observation in finite space"""
        observation_id = f"obs_{len(self.current_observations)}_{int(time.time() * 1000000) % 1000000}"
        
        observation_record = {
            'id': observation_id,
            'timestamp': time.time(),
            'signal_type': type(signal).__name__,
            'signal_size': len(str(signal)),
            'extracted_frequency': self._extract_signal_frequency(signal),
            'observation_space_usage': len(self.current_observations) / self.max_space
        }
        
        # Store in current observations
        self.current_observations[observation_id] = {
            'signal': signal,
            'metadata': observation_record
        }
        
        # Add to history (for metrics)
        self.observation_history.append(observation_record)
        
        # Keep history bounded
        if len(self.observation_history) > 1000:
            self.observation_history = self.observation_history[-1000:]
    
    def clear_observations(self):
        """Clear current observations to free space"""
        cleared_count = len(self.current_observations)
        self.current_observations.clear()
        self.logger.info(f"Cleared {cleared_count} observations")
    
    def get_observation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive observation statistics"""
        current_time = time.time()
        runtime = current_time - self.start_time
        
        # Calculate success rate
        success_rate = self.successful_observations / max(1, self.total_observations)
        
        # Calculate space utilization
        space_utilization = len(self.current_observations) / self.max_space
        
        # Calculate observation rate
        observation_rate = self.total_observations / max(1, runtime)
        
        # Frequency analysis of stored observations
        if self.observation_history:
            extracted_frequencies = [obs['extracted_frequency'] for obs in self.observation_history]
            freq_mean = np.mean(extracted_frequencies)
            freq_std = np.std(extracted_frequencies)
            freq_variance = np.var(extracted_frequencies)
        else:
            freq_mean = freq_std = freq_variance = 0.0
        
        return {
            'observer_id': self.observer_id,
            'configured_frequency': self.frequency,
            'max_observation_space': self.max_space,
            'current_observations': len(self.current_observations),
            'total_observations': self.total_observations,
            'successful_observations': self.successful_observations,
            'failed_observations': self.failed_observations,
            'success_rate': success_rate,
            'space_utilization': space_utilization,
            'observation_rate_per_second': observation_rate,
            'runtime_seconds': runtime,
            'frequency_statistics': {
                'mean_extracted_frequency': freq_mean,
                'frequency_std_deviation': freq_std,
                'frequency_variance': freq_variance,
                'frequency_deviation_from_configured': abs(freq_mean - self.frequency) if freq_mean > 0 else 0
            }
        }
    
    def export_observations(self, filepath: str):
        """Export observations to JSON file"""
        export_data = {
            'observer_metadata': {
                'observer_id': self.observer_id,
                'frequency': self.frequency,
                'max_space': self.max_space,
                'export_timestamp': time.time()
            },
            'statistics': self.get_observation_statistics(),
            'observation_history': self.observation_history[-100:],  # Last 100 observations
            'current_observations_summary': [
                {
                    'id': obs_id,
                    'metadata': obs_data['metadata']
                }
                for obs_id, obs_data in list(self.current_observations.items())[:10]  # First 10 current
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Observations exported to {filepath}")
    
    def __repr__(self):
        return f"FiniteObserver(id={self.observer_id}, freq={self.frequency}Hz, space={len(self.current_observations)}/{self.max_space})"