import time
import json
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from pathlib import Path
from .finite_observer import FiniteObserver
from .transcendent_observer import TranscendentObserver
from .gear_ratio_calculator import GearRatioCalculator
from .oscillatory_hierarchy import OscillatoryHierarchy

class ObserverMetrics:
    """Statistical and metric validation/analysis of all observers including transcendent"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analysis_start_time = time.time()
        self.metrics_history = []
        
        self.logger.info("Observer metrics analyzer initialized")
    
    def analyze_finite_observer(self, observer: FiniteObserver) -> Dict[str, Any]:
        """Comprehensive analysis of finite observer"""
        stats = observer.get_observation_statistics()
        
        # Calculate information content metrics
        information_metrics = self._calculate_information_content(observer)
        
        # Analyze observation patterns
        pattern_metrics = self._analyze_observation_patterns(observer)
        
        # Performance analysis
        performance_metrics = self._analyze_observer_performance(observer)
        
        analysis = {
            'observer_id': observer.observer_id,
            'analysis_timestamp': time.time(),
            'basic_statistics': stats,
            'information_content': information_metrics,
            'observation_patterns': pattern_metrics,
            'performance_analysis': performance_metrics
        }
        
        return analysis
    
    def analyze_transcendent_observer(self, transcendent: TranscendentObserver) -> Dict[str, Any]:
        """Comprehensive analysis of transcendent observer"""
        stats = transcendent.get_transcendent_statistics()
        
        # Analyze coordination effectiveness
        coordination_metrics = self._analyze_coordination_effectiveness(transcendent)
        
        # Analyze gear ratio usage patterns
        gear_ratio_metrics = self._analyze_gear_ratio_patterns(transcendent)
        
        # Compare with finite observers
        comparison_metrics = self._compare_transcendent_vs_finite(transcendent)
        
        analysis = {
            'observer_type': 'transcendent',
            'analysis_timestamp': time.time(),
            'transcendent_statistics': stats,
            'coordination_effectiveness': coordination_metrics,
            'gear_ratio_analysis': gear_ratio_metrics,
            'finite_observer_comparison': comparison_metrics
        }
        
        return analysis
    
    def compare_observer_information_content(self, observers: List[FiniteObserver]) -> Dict[str, Any]:
        """Compare information content between different observers"""
        comparison_results = {
            'total_observers': len(observers),
            'comparison_timestamp': time.time(),
            'observer_comparisons': [],
            'aggregate_metrics': {}
        }
        
        observer_info_contents = []
        observer_entropies = []
        observer_success_rates = []
        
        for observer in observers:
            info_content = self._calculate_information_content(observer)
            
            comparison_results['observer_comparisons'].append({
                'observer_id': observer.observer_id,
                'frequency': observer.frequency,
                'information_content': info_content,
                'success_rate': observer.successful_observations / max(1, observer.total_observations),
                'space_utilization': len(observer.current_observations) / observer.max_space
            })
            
            observer_info_contents.append(info_content['total_information_bits'])
            observer_entropies.append(info_content['average_entropy'])
            observer_success_rates.append(observer.successful_observations / max(1, observer.total_observations))
        
        # Calculate aggregate metrics
        if observer_info_contents:
            comparison_results['aggregate_metrics'] = {
                'mean_information_content': np.mean(observer_info_contents),
                'std_information_content': np.std(observer_info_contents),
                'max_information_content': np.max(observer_info_contents),
                'min_information_content': np.min(observer_info_contents),
                'information_content_range': np.max(observer_info_contents) - np.min(observer_info_contents),
                'mean_entropy': np.mean(observer_entropies),
                'entropy_variance': np.var(observer_entropies),
                'success_rate_correlation_with_info': np.corrcoef(observer_info_contents, observer_success_rates)[0, 1] if len(observer_info_contents) > 1 else 0.0
            }
        
        return comparison_results
    
    def _calculate_information_content(self, observer: FiniteObserver) -> Dict[str, Any]:
        """Calculate information content metrics for observer"""
        if not observer.observation_history:
            return {
                'total_information_bits': 0.0,
                'average_entropy': 0.0,
                'information_density': 0.0,
                'temporal_information_rate': 0.0
            }
        
        # Calculate entropy of observation frequencies
        observation_types = [obs.get('signal_type', 'unknown') for obs in observer.observation_history]
        type_counts = {}
        for obs_type in observation_types:
            type_counts[obs_type] = type_counts.get(obs_type, 0) + 1
        
        total_observations = len(observation_types)
        entropy = 0.0
        for count in type_counts.values():
            probability = count / total_observations
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        # Calculate information density
        total_signal_sizes = sum(obs.get('signal_size', 0) for obs in observer.observation_history)
        information_density = total_signal_sizes / max(1, len(observer.observation_history))
        
        # Calculate temporal information rate
        if len(observer.observation_history) > 1:
            time_span = observer.observation_history[-1]['timestamp'] - observer.observation_history[0]['timestamp']
            temporal_rate = total_signal_sizes / max(1, time_span)
        else:
            temporal_rate = 0.0
        
        return {
            'total_information_bits': total_signal_sizes * 8,  # Convert bytes to bits
            'average_entropy': entropy,
            'information_density': information_density,
            'temporal_information_rate': temporal_rate,
            'unique_signal_types': len(type_counts),
            'observation_type_distribution': type_counts
        }
    
    def _analyze_observation_patterns(self, observer: FiniteObserver) -> Dict[str, Any]:
        """Analyze observation patterns for finite observer"""
        if not observer.observation_history:
            return {'pattern_analysis_available': False}
        
        # Analyze temporal patterns
        timestamps = [obs['timestamp'] for obs in observer.observation_history]
        if len(timestamps) > 1:
            time_intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_interval = np.mean(time_intervals)
            interval_std = np.std(time_intervals)
            interval_regularity = 1.0 / (1.0 + interval_std)  # Higher = more regular
        else:
            avg_interval = interval_std = interval_regularity = 0.0
        
        # Analyze frequency patterns
        extracted_frequencies = [obs.get('extracted_frequency', observer.frequency) for obs in observer.observation_history]
        freq_mean = np.mean(extracted_frequencies)
        freq_deviation = abs(freq_mean - observer.frequency)
        freq_stability = 1.0 / (1.0 + np.std(extracted_frequencies))
        
        return {
            'pattern_analysis_available': True,
            'temporal_patterns': {
                'average_observation_interval': avg_interval,
                'interval_standard_deviation': interval_std,
                'interval_regularity_score': interval_regularity
            },
            'frequency_patterns': {
                'mean_extracted_frequency': freq_mean,
                'frequency_deviation_from_configured': freq_deviation,
                'frequency_stability_score': freq_stability
            },
            'observation_consistency': {
                'total_observations': len(observer.observation_history),
                'success_rate': observer.successful_observations / max(1, observer.total_observations),
                'space_utilization_trend': len(observer.current_observations) / observer.max_space
            }
        }
    
    def _analyze_observer_performance(self, observer: FiniteObserver) -> Dict[str, Any]:
        """Analyze performance characteristics of observer"""
        current_time = time.time()
        runtime = current_time - observer.start_time
        
        # Calculate throughput metrics
        observations_per_second = observer.total_observations / max(1, runtime)
        successful_per_second = observer.successful_observations / max(1, runtime)
        
        # Analyze efficiency
        space_efficiency = len(observer.current_observations) / observer.max_space
        success_rate = observer.successful_observations / max(1, observer.total_observations)
        
        # Performance scoring
        throughput_score = min(1.0, observations_per_second / 10.0)  # Normalize to [0,1]
        efficiency_score = success_rate * space_efficiency
        overall_performance = (throughput_score + efficiency_score) / 2.0
        
        return {
            'throughput_metrics': {
                'observations_per_second': observations_per_second,
                'successful_observations_per_second': successful_per_second,
                'throughput_score': throughput_score
            },
            'efficiency_metrics': {
                'success_rate': success_rate,
                'space_efficiency': space_efficiency,
                'efficiency_score': efficiency_score
            },
            'overall_performance': {
                'performance_score': overall_performance,
                'runtime_seconds': runtime,
                'performance_classification': self._classify_performance(overall_performance)
            }
        }
    
    def _analyze_coordination_effectiveness(self, transcendent: TranscendentObserver) -> Dict[str, Any]:
        """Analyze how effectively transcendent observer coordinates finite observers"""
        if not transcendent.transcendent_observations:
            return {'coordination_analysis_available': False}
        
        # Analyze coordination success rates
        success_rates = [obs['overall_success_rate'] for obs in transcendent.transcendent_observations]
        avg_success_rate = np.mean(success_rates)
        success_rate_std = np.std(success_rates)
        coordination_stability = 1.0 / (1.0 + success_rate_std)
        
        # Analyze observer engagement
        observer_counts = [obs['successful_observations'] for obs in transcendent.transcendent_observations]
        avg_engaged_observers = np.mean(observer_counts)
        engagement_consistency = 1.0 / (1.0 + np.std(observer_counts))
        
        # Calculate coordination efficiency
        total_possible_observations = len(transcendent.finite_observers) * len(transcendent.transcendent_observations)
        total_successful_observations = sum(observer_counts)
        coordination_efficiency = total_successful_observations / max(1, total_possible_observations)
        
        return {
            'coordination_analysis_available': True,
            'success_rate_analysis': {
                'average_coordination_success_rate': avg_success_rate,
                'success_rate_standard_deviation': success_rate_std,
                'coordination_stability_score': coordination_stability
            },
            'observer_engagement': {
                'average_engaged_observers': avg_engaged_observers,
                'total_finite_observers': len(transcendent.finite_observers),
                'engagement_consistency_score': engagement_consistency
            },
            'coordination_efficiency': {
                'overall_efficiency': coordination_efficiency,
                'total_coordination_attempts': len(transcendent.transcendent_observations),
                'efficiency_classification': self._classify_efficiency(coordination_efficiency)
            }
        }
    
    def _analyze_gear_ratio_patterns(self, transcendent: TranscendentObserver) -> Dict[str, Any]:
        """Analyze gear ratio usage patterns in transcendent observer"""
        if not transcendent.gear_ratio_applications:
            return {'gear_ratio_analysis_available': False}
        
        # Extract gear ratios
        gear_ratios = [app['gear_ratio'] for app in transcendent.gear_ratio_applications]
        
        # Statistical analysis
        ratio_mean = np.mean(gear_ratios)
        ratio_std = np.std(gear_ratios)
        ratio_min = np.min(gear_ratios)
        ratio_max = np.max(gear_ratios)
        ratio_range = ratio_max - ratio_min
        
        # Analyze usage patterns
        ratio_usage_counts = {}
        for ratio in gear_ratios:
            ratio_bucket = round(ratio, 2)  # Group by 2 decimal places
            ratio_usage_counts[ratio_bucket] = ratio_usage_counts.get(ratio_bucket, 0) + 1
        
        most_used_ratio = max(ratio_usage_counts.items(), key=lambda x: x[1])
        
        return {
            'gear_ratio_analysis_available': True,
            'statistical_analysis': {
                'mean_gear_ratio': ratio_mean,
                'gear_ratio_std_deviation': ratio_std,
                'min_gear_ratio': ratio_min,
                'max_gear_ratio': ratio_max,
                'gear_ratio_range': ratio_range
            },
            'usage_patterns': {
                'total_applications': len(gear_ratios),
                'unique_ratios_used': len(ratio_usage_counts),
                'most_frequently_used_ratio': most_used_ratio[0],
                'most_frequent_usage_count': most_used_ratio[1],
                'ratio_diversity_score': len(ratio_usage_counts) / len(gear_ratios)
            }
        }
    
    def _compare_transcendent_vs_finite(self, transcendent: TranscendentObserver) -> Dict[str, Any]:
        """Compare transcendent observer performance vs finite observers"""
        finite_performances = []
        finite_success_rates = []
        
        for observer in transcendent.finite_observers:
            performance = self._analyze_observer_performance(observer)
            finite_performances.append(performance['overall_performance']['performance_score'])
            finite_success_rates.append(observer.successful_observations / max(1, observer.total_observations))
        
        transcendent_stats = transcendent.get_transcendent_statistics()
        transcendent_success_rate = transcendent_stats['navigation_performance']['navigation_success_rate']
        
        comparison = {
            'transcendent_performance': {
                'navigation_success_rate': transcendent_success_rate,
                'total_navigations': transcendent_stats['navigation_performance']['total_navigations'],
                'average_navigation_time': transcendent_stats['navigation_performance']['average_navigation_time']
            },
            'finite_observer_performance': {
                'average_finite_success_rate': np.mean(finite_success_rates) if finite_success_rates else 0.0,
                'finite_success_rate_std': np.std(finite_success_rates) if finite_success_rates else 0.0,
                'best_finite_success_rate': np.max(finite_success_rates) if finite_success_rates else 0.0,
                'worst_finite_success_rate': np.min(finite_success_rates) if finite_success_rates else 0.0
            },
            'comparative_analysis': {
                'transcendent_vs_average_finite': transcendent_success_rate - (np.mean(finite_success_rates) if finite_success_rates else 0.0),
                'transcendent_vs_best_finite': transcendent_success_rate - (np.max(finite_success_rates) if finite_success_rates else 0.0),
                'coordination_advantage': transcendent_success_rate > (np.mean(finite_success_rates) if finite_success_rates else 0.0)
            }
        }
        
        return comparison
    
    def _classify_performance(self, performance_score: float) -> str:
        """Classify performance based on score"""
        if performance_score >= 0.8:
            return "Excellent"
        elif performance_score >= 0.6:
            return "Good"
        elif performance_score >= 0.4:
            return "Fair"
        elif performance_score >= 0.2:
            return "Poor"
        else:
            return "Critical"
    
    def _classify_efficiency(self, efficiency_score: float) -> str:
        """Classify efficiency based on score"""
        if efficiency_score >= 0.9:
            return "Highly Efficient"
        elif efficiency_score >= 0.7:
            return "Efficient"
        elif efficiency_score >= 0.5:
            return "Moderately Efficient"
        elif efficiency_score >= 0.3:
            return "Inefficient"
        else:
            return "Highly Inefficient"
    
    def generate_comprehensive_report(self, transcendent: TranscendentObserver, output_dir: str) -> str:
        """Generate comprehensive observer analysis report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Analyze transcendent observer
        transcendent_analysis = self.analyze_transcendent_observer(transcendent)
        
        # Analyze all finite observers
        finite_analyses = []
        for observer in transcendent.finite_observers:
            analysis = self.analyze_finite_observer(observer)
            finite_analyses.append(analysis)
        
        # Compare observers
        observer_comparison = self.compare_observer_information_content(transcendent.finite_observers)
        
        # Compile comprehensive report
        report = {
            'report_metadata': {
                'generation_timestamp': time.time(),
                'report_type': 'comprehensive_observer_analysis',
                'total_observers_analyzed': len(transcendent.finite_observers) + 1,
                'analysis_duration': time.time() - self.analysis_start_time
            },
            'transcendent_observer_analysis': transcendent_analysis,
            'finite_observer_analyses': finite_analyses,
            'inter_observer_comparison': observer_comparison,
            'summary_recommendations': self._generate_recommendations(transcendent_analysis, finite_analyses, observer_comparison)
        }
        
        # Save report
        report_file = output_path / "comprehensive_observer_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate visualizations
        self._generate_analysis_visualizations(report, output_path)
        
        self.logger.info(f"Comprehensive observer report generated: {report_file}")
        return str(report_file)
    
    def _generate_recommendations(self, transcendent_analysis: Dict, finite_analyses: List[Dict], comparison: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Transcendent performance recommendations
        transcendent_success_rate = transcendent_analysis['transcendent_statistics']['navigation_performance']['navigation_success_rate']
        if transcendent_success_rate < 0.8:
            recommendations.append("Consider optimizing transcendent observer navigation algorithms - success rate below 80%")
        
        # Finite observer recommendations
        avg_finite_success = comparison['aggregate_metrics'].get('mean_information_content', 0)
        if avg_finite_success < 1000:  # bits
            recommendations.append("Finite observers showing low information content - consider increasing observation frequency or expanding observation space")
        
        # Coordination recommendations
        if 'coordination_effectiveness' in transcendent_analysis:
            coord_efficiency = transcendent_analysis['coordination_effectiveness'].get('coordination_efficiency', {}).get('overall_efficiency', 0)
            if coord_efficiency < 0.7:
                recommendations.append("Coordination efficiency below 70% - consider adjusting gear ratio parameters or observer selection")
        
        # Performance recommendations
        poor_performers = [analysis for analysis in finite_analyses 
                         if analysis['performance_analysis']['overall_performance']['performance_score'] < 0.4]
        if poor_performers:
            recommendations.append(f"{len(poor_performers)} observers showing poor performance - consider parameter tuning or replacement")
        
        if not recommendations:
            recommendations.append("All observers performing within acceptable parameters")
        
        return recommendations
    
    def _generate_analysis_visualizations(self, report: Dict, output_path: Path):
        """Generate visualization plots for analysis report"""
        try:
            # Plot 1: Observer performance comparison
            finite_analyses = report['finite_observer_analyses']
            observer_ids = [analysis['observer_id'] for analysis in finite_analyses]
            performance_scores = [analysis['performance_analysis']['overall_performance']['performance_score'] 
                                for analysis in finite_analyses]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(observer_ids)), performance_scores, 
                          color=['green' if score >= 0.7 else 'orange' if score >= 0.4 else 'red' for score in performance_scores])
            plt.xlabel('Observer ID')
            plt.ylabel('Performance Score')
            plt.title('Finite Observer Performance Comparison')
            plt.xticks(range(len(observer_ids)), [f"Obs_{i}" for i in range(len(observer_ids))], rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / "observer_performance_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot 2: Information content distribution
            info_contents = [analysis['information_content']['total_information_bits'] 
                           for analysis in finite_analyses]
            
            plt.figure(figsize=(10, 6))
            plt.hist(info_contents, bins=min(10, len(info_contents)), alpha=0.7, color='blue', edgecolor='black')
            plt.xlabel('Information Content (bits)')
            plt.ylabel('Number of Observers')
            plt.title('Distribution of Observer Information Content')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / "information_content_distribution.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Analysis visualizations generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
    
    def export_metrics_data(self, filepath: str):
        """Export all metrics data to JSON"""
        export_data = {
            'export_metadata': {
                'export_timestamp': time.time(),
                'metrics_analyzer_version': '1.0',
                'analysis_duration': time.time() - self.analysis_start_time
            },
            'metrics_history': self.metrics_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Observer metrics data exported to {filepath}")
    
    def __repr__(self):
        return f"ObserverMetrics(analyses_completed={len(self.metrics_history)})"