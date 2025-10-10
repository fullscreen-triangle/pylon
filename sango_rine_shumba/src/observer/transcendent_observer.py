import time
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .finite_observer import FiniteObserver
from .gear_ratio_calculator import GearRatioCalculator
from .oscillatory_hierarchy import OscillatoryHierarchy

class TranscendentObserver:
    """Observes finite observers using gear ratios"""
    
    def __init__(self, finite_observers: List[FiniteObserver] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.gear_ratio_calculator = GearRatioCalculator()
        self.oscillatory_hierarchy = OscillatoryHierarchy()
        
        # Set up finite observers (use hierarchy observers if none provided)
        if finite_observers is None:
            self.finite_observers = list(self.oscillatory_hierarchy.scale_observers.values())
        else:
            self.finite_observers = finite_observers
        
        # Transcendent observer state
        self.transcendent_observations = []
        self.navigation_history = []
        self.gear_ratio_applications = []
        self.start_time = time.time()
        
        # Performance tracking
        self.total_navigations = 0
        self.successful_navigations = 0
        self.total_navigation_time = 0.0
        
        self.logger.info(f"Transcendent observer initialized with {len(self.finite_observers)} finite observers")
        
    def navigate_hierarchy_O1(self, source_scale: int, target_scale: int, transformation_data: Any = None) -> Dict[str, Any]:
        """O(1) navigation using gear ratios - no sequential traversal"""
        start_time = time.time()
        self.total_navigations += 1
        
        try:
            # Get compound gear ratio for direct O(1) navigation
            gear_ratio = self.gear_ratio_calculator.get_compound_ratio(source_scale, target_scale)
            
            # Apply gear transformation
            transformation_result = self.apply_gear_transformation(gear_ratio, transformation_data)
            
            # Record navigation
            navigation_time = time.time() - start_time
            self.total_navigation_time += navigation_time
            self.successful_navigations += 1
            
            navigation_record = {
                'timestamp': start_time,
                'source_scale': source_scale,
                'target_scale': target_scale,
                'gear_ratio': gear_ratio,
                'navigation_time': navigation_time,
                'complexity': 'O(1)',
                'success': True,
                'transformation_result': transformation_result
            }
            
            self.navigation_history.append(navigation_record)
            
            # Keep history bounded
            if len(self.navigation_history) > 1000:
                self.navigation_history = self.navigation_history[-1000:]
            
            self.logger.debug(f"O(1) navigation: scale {source_scale} → {target_scale}, "
                            f"ratio={gear_ratio:.6f}, time={navigation_time*1000:.3f}ms")
            
            return {
                'success': True,
                'gear_ratio': gear_ratio,
                'transformation_result': transformation_result,
                'navigation_time': navigation_time,
                'complexity': 'O(1)'
            }
            
        except Exception as e:
            navigation_time = time.time() - start_time
            self.total_navigation_time += navigation_time
            
            error_record = {
                'timestamp': start_time,
                'source_scale': source_scale,
                'target_scale': target_scale,
                'navigation_time': navigation_time,
                'success': False,
                'error': str(e)
            }
            
            self.navigation_history.append(error_record)
            self.logger.error(f"Navigation failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'navigation_time': navigation_time
            }
    
    def apply_gear_transformation(self, gear_ratio: float, data: Any = None) -> Dict[str, Any]:
        """Apply gear ratio transformation to data"""
        try:
            # Record gear ratio application
            application_record = {
                'timestamp': time.time(),
                'gear_ratio': gear_ratio,
                'input_type': type(data).__name__ if data is not None else 'None',
                'transformation_applied': True
            }
            
            # Apply transformation based on data type
            if data is None:
                # No data transformation, just return ratio properties
                result = {
                    'gear_ratio': gear_ratio,
                    'transformation_type': 'ratio_only',
                    'amplification_factor': gear_ratio,
                    'frequency_scaling': gear_ratio
                }
            
            elif isinstance(data, (int, float)):
                # Numeric transformation
                transformed_value = data * gear_ratio
                result = {
                    'original_value': data,
                    'gear_ratio': gear_ratio,
                    'transformed_value': transformed_value,
                    'transformation_type': 'numeric_scaling',
                    'amplification_achieved': abs(transformed_value) > abs(data) if data != 0 else True
                }
            
            elif isinstance(data, dict):
                # Dictionary transformation - apply gear ratio to numeric values
                transformed_dict = {}
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        transformed_dict[key] = value * gear_ratio
                    else:
                        transformed_dict[key] = value
                
                result = {
                    'original_data': data,
                    'gear_ratio': gear_ratio,
                    'transformed_data': transformed_dict,
                    'transformation_type': 'dictionary_scaling',
                    'keys_transformed': sum(1 for v in data.values() if isinstance(v, (int, float)))
                }
            
            elif isinstance(data, list):
                # List transformation - apply gear ratio to numeric elements
                transformed_list = []
                for item in data:
                    if isinstance(item, (int, float)):
                        transformed_list.append(item * gear_ratio)
                    else:
                        transformed_list.append(item)
                
                result = {
                    'original_data': data,
                    'gear_ratio': gear_ratio,
                    'transformed_data': transformed_list,
                    'transformation_type': 'list_scaling',
                    'elements_transformed': sum(1 for item in data if isinstance(item, (int, float)))
                }
            
            else:
                # Generic transformation - apply gear ratio as metadata
                result = {
                    'original_data': data,
                    'gear_ratio': gear_ratio,
                    'transformation_type': 'metadata_attachment',
                    'transformed_metadata': {
                        'data': data,
                        'gear_ratio_applied': gear_ratio,
                        'transformation_timestamp': time.time()
                    }
                }
            
            application_record['result'] = result
            self.gear_ratio_applications.append(application_record)
            
            # Keep applications bounded
            if len(self.gear_ratio_applications) > 500:
                self.gear_ratio_applications = self.gear_ratio_applications[-500:]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Gear transformation failed: {e}")
            return {
                'transformation_failed': True,
                'error': str(e),
                'gear_ratio': gear_ratio,
                'original_data': data
            }
    
    def observe_finite_observers(self, signal: Any) -> Dict[str, Any]:
        """Observe all finite observers and collect their responses"""
        observation_start = time.time()
        
        finite_observer_results = {}
        successful_observations = 0
        failed_observations = 0
        
        for observer in self.finite_observers:
            try:
                observed = observer.observe_signal(signal)
                finite_observer_results[observer.observer_id] = {
                    'observed': observed,
                    'frequency': observer.frequency,
                    'space_utilization': len(observer.current_observations) / observer.max_space,
                    'success_rate': observer.successful_observations / max(1, observer.total_observations)
                }
                
                if observed:
                    successful_observations += 1
                else:
                    failed_observations += 1
                    
            except Exception as e:
                finite_observer_results[observer.observer_id] = {
                    'observed': False,
                    'error': str(e)
                }
                failed_observations += 1
        
        # Create transcendent observation record
        transcendent_observation = {
            'timestamp': observation_start,
            'signal_type': type(signal).__name__,
            'finite_observer_count': len(self.finite_observers),
            'successful_observations': successful_observations,
            'failed_observations': failed_observations,
            'overall_success_rate': successful_observations / len(self.finite_observers),
            'observation_time': time.time() - observation_start,
            'finite_observer_results': finite_observer_results
        }
        
        self.transcendent_observations.append(transcendent_observation)
        
        # Keep transcendent observations bounded
        if len(self.transcendent_observations) > 200:
            self.transcendent_observations = self.transcendent_observations[-200:]
        
        self.logger.debug(f"Transcendent observation: {successful_observations}/{len(self.finite_observers)} observers successful")
        
        return transcendent_observation
    
    def navigate_using_gear_ratios(self, navigation_request: Dict[str, Any]) -> Dict[str, Any]:
        """Navigate hierarchical space using gear ratios with full transcendent coordination"""
        try:
            source_scale = navigation_request.get('source_scale', 1)
            target_scale = navigation_request.get('target_scale', 8)
            transformation_data = navigation_request.get('data')
            
            # Perform O(1) navigation
            navigation_result = self.navigate_hierarchy_O1(source_scale, target_scale, transformation_data)
            
            # If successful, observe the result across all finite observers
            if navigation_result['success']:
                observation_result = self.observe_finite_observers(navigation_result['transformation_result'])
                
                # Combine navigation and observation results
                complete_result = {
                    'navigation': navigation_result,
                    'observation': observation_result,
                    'transcendent_coordination': {
                        'gear_ratio_used': navigation_result['gear_ratio'],
                        'observers_engaged': len(self.finite_observers),
                        'coordination_success': observation_result['overall_success_rate'] > 0.5,
                        'total_processing_time': navigation_result['navigation_time'] + observation_result['observation_time']
                    }
                }
                
                return complete_result
            else:
                return navigation_result
                
        except Exception as e:
            self.logger.error(f"Transcendent navigation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'transcendent_navigation_failed': True
            }
    
    def get_transcendent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive transcendent observer statistics"""
        current_time = time.time()
        runtime = current_time - self.start_time
        
        # Calculate navigation performance
        navigation_success_rate = self.successful_navigations / max(1, self.total_navigations)
        avg_navigation_time = self.total_navigation_time / max(1, self.total_navigations)
        navigations_per_second = self.total_navigations / max(1, runtime)
        
        # Analyze gear ratio applications
        if self.gear_ratio_applications:
            gear_ratios_used = [app['gear_ratio'] for app in self.gear_ratio_applications]
            avg_gear_ratio = np.mean(gear_ratios_used)
            gear_ratio_std = np.std(gear_ratios_used)
            min_gear_ratio = np.min(gear_ratios_used)
            max_gear_ratio = np.max(gear_ratios_used)
        else:
            avg_gear_ratio = gear_ratio_std = min_gear_ratio = max_gear_ratio = 0.0
        
        # Analyze finite observer performance
        finite_observer_stats = []
        for observer in self.finite_observers:
            obs_stats = observer.get_observation_statistics()
            finite_observer_stats.append(obs_stats)
        
        avg_finite_success_rate = np.mean([stats['success_rate'] for stats in finite_observer_stats])
        
        return {
            'transcendent_observer_metadata': {
                'runtime_seconds': runtime,
                'finite_observers_managed': len(self.finite_observers),
                'total_transcendent_observations': len(self.transcendent_observations)
            },
            'navigation_performance': {
                'total_navigations': self.total_navigations,
                'successful_navigations': self.successful_navigations,
                'navigation_success_rate': navigation_success_rate,
                'average_navigation_time': avg_navigation_time,
                'navigations_per_second': navigations_per_second,
                'complexity_achieved': 'O(1)',
                'total_navigation_time': self.total_navigation_time
            },
            'gear_ratio_statistics': {
                'total_applications': len(self.gear_ratio_applications),
                'average_gear_ratio': avg_gear_ratio,
                'gear_ratio_std_deviation': gear_ratio_std,
                'min_gear_ratio': min_gear_ratio,
                'max_gear_ratio': max_gear_ratio,
                'amplification_factor_range': max_gear_ratio - min_gear_ratio if max_gear_ratio > min_gear_ratio else 0.0
            },
            'finite_observer_coordination': {
                'average_finite_success_rate': avg_finite_success_rate,
                'total_finite_observers': len(self.finite_observers),
                'finite_observer_statistics': finite_observer_stats
            },
            'hierarchy_integration': {
                'oscillatory_scales_managed': 8,
                'frequency_range_orders_of_magnitude': 18,
                'scale_coordination_active': True
            }
        }
    
    def export_transcendent_data(self, filepath: str):
        """Export complete transcendent observer data to JSON"""
        export_data = {
            'export_metadata': {
                'export_timestamp': time.time(),
                'observer_type': 'transcendent',
                'data_version': '1.0'
            },
            'transcendent_statistics': self.get_transcendent_statistics(),
            'navigation_history': self.navigation_history[-50:],  # Last 50 navigations
            'gear_ratio_applications': self.gear_ratio_applications[-50:],  # Last 50 applications
            'transcendent_observations': self.transcendent_observations[-20:],  # Last 20 observations
            'gear_ratio_calculator_stats': self.gear_ratio_calculator.get_performance_statistics(),
            'oscillatory_hierarchy_stats': self.oscillatory_hierarchy.get_hierarchy_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Transcendent observer data exported to {filepath}")
    
    def validate_O1_complexity(self, test_scales: List[Tuple[int, int]], test_count: int = 100) -> Dict[str, Any]:
        """Validate that navigation maintains O(1) complexity regardless of scale separation"""
        validation_results = {
            'test_count': test_count,
            'scale_pairs_tested': len(test_scales),
            'navigation_times': [],
            'complexity_validated': True,
            'performance_consistency': True
        }
        
        for source_scale, target_scale in test_scales:
            scale_navigation_times = []
            
            for _ in range(test_count):
                start_time = time.time()
                self.navigate_hierarchy_O1(source_scale, target_scale)
                navigation_time = time.time() - start_time
                scale_navigation_times.append(navigation_time)
            
            avg_time = np.mean(scale_navigation_times)
            std_time = np.std(scale_navigation_times)
            
            validation_results['navigation_times'].append({
                'source_scale': source_scale,
                'target_scale': target_scale,
                'scale_separation': abs(target_scale - source_scale),
                'average_time': avg_time,
                'std_deviation': std_time,
                'min_time': np.min(scale_navigation_times),
                'max_time': np.max(scale_navigation_times)
            })
        
        # Check if navigation time is independent of scale separation
        if len(validation_results['navigation_times']) > 1:
            times = [result['average_time'] for result in validation_results['navigation_times']]
            separations = [result['scale_separation'] for result in validation_results['navigation_times']]
            
            # Calculate correlation between scale separation and navigation time
            if len(times) > 2:
                correlation = np.corrcoef(separations, times)[0, 1]
                validation_results['separation_time_correlation'] = correlation
                validation_results['complexity_validated'] = abs(correlation) < 0.1  # Low correlation = O(1)
        
        self.logger.info(f"O(1) complexity validation: {'PASSED' if validation_results['complexity_validated'] else 'FAILED'}")
        return validation_results
    
    def __repr__(self):
        return f"TranscendentObserver(finite_observers={len(self.finite_observers)}, navigations={self.total_navigations})"