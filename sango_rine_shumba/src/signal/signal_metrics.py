#!/usr/bin/env python3
"""
Signal Metrics and Statistical Analysis System

Performs comprehensive statistical and metric validation/analysis of all signals:
- Hardware signal analysis (CPU, memory, disk, network, audio, timing)
- Network signal analysis (connectivity, latency, traffic, DNS, WiFi)
- Cross-signal correlation analysis
- Temporal pattern detection
- Frequency domain analysis
- Statistical validation and outlier detection
- Signal quality metrics
- Predictive analysis and trend detection
"""

import time
import json
import logging
import numpy as np
import scipy.stats as stats
from scipy import signal as scipy_signal
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import signal collectors
try:
    from .hardware_signals import HardwareSignalCollector
    from .network_signals import NetworkSignalCollector
    SIGNAL_COLLECTORS_AVAILABLE = True
except ImportError:
    SIGNAL_COLLECTORS_AVAILABLE = False

class SignalMetricsAnalyzer:
    """
    Comprehensive signal metrics and statistical analysis system.
    
    Analyzes both hardware and network signals to provide:
    - Statistical summaries and distributions
    - Correlation analysis between different signal types
    - Temporal pattern detection
    - Frequency domain analysis
    - Signal quality assessment
    - Anomaly detection
    - Predictive modeling
    """
    
    def __init__(self, analysis_window: float = 300.0):
        self.logger = logging.getLogger(__name__)
        self.analysis_window = analysis_window  # Analysis window in seconds
        
        # Analysis results storage
        self.metrics = {
            'hardware_metrics': {},
            'network_metrics': {},
            'cross_correlation_metrics': {},
            'temporal_metrics': {},
            'frequency_metrics': {},
            'quality_metrics': {},
            'anomaly_metrics': {},
            'statistical_metrics': {}
        }
        
        # Initialize collectors if available
        self.hardware_collector = None
        self.network_collector = None
        
        if SIGNAL_COLLECTORS_AVAILABLE:
            try:
                self.hardware_collector = HardwareSignalCollector(
                    sample_duration=analysis_window,
                    sample_rate=2.0
                )
                self.network_collector = NetworkSignalCollector(
                    sample_duration=analysis_window,
                    sample_rate=0.5
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize signal collectors: {e}")
        
        self.logger.info(f"Signal metrics analyzer initialized - window: {analysis_window}s")
    
    def analyze_all_signals(self, 
                          hardware_data: Optional[Dict] = None,
                          network_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of all available signals.
        
        Args:
            hardware_data: Pre-collected hardware signal data (optional)
            network_data: Pre-collected network signal data (optional)
        
        Returns:
            Complete analysis results with all metrics
        """
        self.logger.info("Starting comprehensive signal analysis...")
        
        analysis_start_time = time.time()
        
        # Collect signals if not provided
        if hardware_data is None and self.hardware_collector:
            self.logger.info("Collecting hardware signals...")
            hardware_data = self.hardware_collector.collect_all_hardware_signals()
        
        if network_data is None and self.network_collector:
            self.logger.info("Collecting network signals...")
            network_data = self.network_collector.collect_all_network_signals()
        
        # Analyze hardware signals
        if hardware_data:
            self.logger.info("Analyzing hardware signals...")
            self.metrics['hardware_metrics'] = self._analyze_hardware_signals(hardware_data)
        
        # Analyze network signals  
        if network_data:
            self.logger.info("Analyzing network signals...")
            self.metrics['network_metrics'] = self._analyze_network_signals(network_data)
        
        # Cross-correlation analysis
        if hardware_data and network_data:
            self.logger.info("Computing cross-signal correlations...")
            self.metrics['cross_correlation_metrics'] = self._analyze_cross_correlations(
                hardware_data, network_data
            )
        
        # Temporal pattern analysis
        self.logger.info("Analyzing temporal patterns...")
        self.metrics['temporal_metrics'] = self._analyze_temporal_patterns(
            hardware_data, network_data
        )
        
        # Frequency domain analysis
        self.logger.info("Performing frequency domain analysis...")
        self.metrics['frequency_metrics'] = self._analyze_frequency_domain(
            hardware_data, network_data
        )
        
        # Signal quality assessment
        self.logger.info("Assessing signal quality...")
        self.metrics['quality_metrics'] = self._assess_signal_quality(
            hardware_data, network_data
        )
        
        # Anomaly detection
        self.logger.info("Detecting anomalies...")
        self.metrics['anomaly_metrics'] = self._detect_anomalies(
            hardware_data, network_data
        )
        
        # Statistical validation
        self.logger.info("Performing statistical validation...")
        self.metrics['statistical_metrics'] = self._perform_statistical_validation(
            hardware_data, network_data
        )
        
        analysis_time = time.time() - analysis_start_time
        
        return {
            'analysis_metadata': {
                'analysis_start_time': analysis_start_time,
                'analysis_duration': analysis_time,
                'analysis_window': self.analysis_window,
                'hardware_signals_analyzed': hardware_data is not None,
                'network_signals_analyzed': network_data is not None,
                'total_metrics_computed': sum(len(metrics) for metrics in self.metrics.values())
            },
            'signal_metrics': self.metrics.copy(),
            'raw_hardware_data': hardware_data,
            'raw_network_data': network_data
        }
    
    def _analyze_hardware_signals(self, hardware_data: Dict) -> Dict[str, Any]:
        """Analyze hardware signal patterns and statistics"""
        hardware_signals = hardware_data.get('hardware_signals', {})
        analysis = {
            'signal_statistics': {},
            'temporal_analysis': {},
            'resource_utilization': {},
            'performance_metrics': {}
        }
        
        # CPU signal analysis
        if hardware_signals.get('cpu_signals'):
            cpu_data = hardware_signals['cpu_signals']
            cpu_analysis = self._analyze_cpu_signals(cpu_data)
            analysis['signal_statistics']['cpu'] = cpu_analysis
            
            # CPU utilization trends
            timestamps = [d['timestamp'] for d in cpu_data]
            cpu_usage = []
            for d in cpu_data:
                if d.get('cpu_percent'):
                    avg_usage = np.mean(d['cpu_percent'])
                    cpu_usage.append(avg_usage)
            
            if cpu_usage:
                analysis['resource_utilization']['cpu'] = {
                    'mean_utilization': float(np.mean(cpu_usage)),
                    'max_utilization': float(np.max(cpu_usage)),
                    'utilization_variance': float(np.var(cpu_usage)),
                    'high_load_periods': len([u for u in cpu_usage if u > 80]),
                    'utilization_trend': self._calculate_trend(timestamps, cpu_usage) if len(cpu_usage) > 2 else 0
                }
        
        # Memory signal analysis
        if hardware_signals.get('memory_signals'):
            mem_data = hardware_signals['memory_signals']
            mem_analysis = self._analyze_memory_signals(mem_data)
            analysis['signal_statistics']['memory'] = mem_analysis
            
            # Memory utilization trends
            memory_usage = [d['virtual_memory']['percent'] for d in mem_data]
            timestamps = [d['timestamp'] for d in mem_data]
            
            analysis['resource_utilization']['memory'] = {
                'mean_utilization': float(np.mean(memory_usage)),
                'max_utilization': float(np.max(memory_usage)),
                'utilization_variance': float(np.var(memory_usage)),
                'memory_pressure_periods': len([u for u in memory_usage if u > 85]),
                'utilization_trend': self._calculate_trend(timestamps, memory_usage) if len(memory_usage) > 2 else 0
            }
        
        # Audio signal analysis (if available)
        if hardware_signals.get('audio_signals'):
            audio_data = hardware_signals['audio_signals']
            audio_analysis = self._analyze_audio_signals(audio_data)
            analysis['signal_statistics']['audio'] = audio_analysis
        
        # Timing signal analysis
        if hardware_signals.get('timing_signals'):
            timing_data = hardware_signals['timing_signals']
            timing_analysis = self._analyze_timing_signals(timing_data)
            analysis['signal_statistics']['timing'] = timing_analysis
            
            # Clock drift analysis
            if len(timing_data) > 1:
                system_times = [d['system_time'] for d in timing_data]
                perf_counters = [d['perf_counter'] for d in timing_data]
                
                analysis['performance_metrics']['timing'] = {
                    'clock_stability': self._calculate_clock_stability(system_times, perf_counters),
                    'timing_jitter': self._calculate_timing_jitter(timing_data)
                }
        
        return analysis
    
    def _analyze_network_signals(self, network_data: Dict) -> Dict[str, Any]:
        """Analyze network signal patterns and statistics"""
        network_signals = network_data.get('network_signals', {})
        analysis = {
            'connectivity_analysis': {},
            'performance_analysis': {},
            'reliability_analysis': {},
            'traffic_analysis': {}
        }
        
        # Connectivity analysis
        if network_signals.get('connectivity_signals'):
            conn_data = network_signals['connectivity_signals']
            conn_analysis = self._analyze_connectivity_patterns(conn_data)
            analysis['connectivity_analysis'] = conn_analysis
        
        # Latency analysis
        if network_signals.get('latency_signals'):
            latency_data = network_signals['latency_signals']
            latency_analysis = self._analyze_latency_patterns(latency_data)
            analysis['performance_analysis']['latency'] = latency_analysis
        
        # DNS analysis
        if network_signals.get('dns_signals'):
            dns_data = network_signals['dns_signals']
            dns_analysis = self._analyze_dns_patterns(dns_data)
            analysis['performance_analysis']['dns'] = dns_analysis
        
        # Traffic analysis
        if network_signals.get('traffic_signals'):
            traffic_data = network_signals['traffic_signals']
            traffic_analysis = self._analyze_traffic_patterns(traffic_data)
            analysis['traffic_analysis'] = traffic_analysis
        
        # Interface analysis
        if network_signals.get('interface_signals'):
            interface_data = network_signals['interface_signals']
            interface_analysis = self._analyze_interface_patterns(interface_data)
            analysis['reliability_analysis']['interfaces'] = interface_analysis
        
        return analysis
    
    def _analyze_cpu_signals(self, cpu_data: List[Dict]) -> Dict[str, Any]:
        """Detailed analysis of CPU signals"""
        if not cpu_data:
            return {}
        
        # Extract CPU usage data
        all_cpu_usage = []
        per_core_usage = []
        cpu_frequencies = []
        
        for sample in cpu_data:
            if sample.get('cpu_percent'):
                all_cpu_usage.extend(sample['cpu_percent'])
                per_core_usage.append(sample['cpu_percent'])
            
            if sample.get('cpu_frequencies'):
                cpu_frequencies.extend(sample['cpu_frequencies'])
        
        analysis = {}
        
        if all_cpu_usage:
            analysis['usage_statistics'] = {
                'mean': float(np.mean(all_cpu_usage)),
                'median': float(np.median(all_cpu_usage)),
                'std': float(np.std(all_cpu_usage)),
                'min': float(np.min(all_cpu_usage)),
                'max': float(np.max(all_cpu_usage)),
                'percentiles': {
                    '25th': float(np.percentile(all_cpu_usage, 25)),
                    '75th': float(np.percentile(all_cpu_usage, 75)),
                    '95th': float(np.percentile(all_cpu_usage, 95)),
                    '99th': float(np.percentile(all_cpu_usage, 99))
                }
            }
            
            # Core balance analysis
            if per_core_usage and len(per_core_usage) > 1:
                core_variances = [np.var(cores) for cores in per_core_usage if cores]
                if core_variances:
                    analysis['core_balance'] = {
                        'average_core_variance': float(np.mean(core_variances)),
                        'load_imbalance_score': float(np.std(core_variances)),
                        'max_core_variance': float(np.max(core_variances))
                    }
        
        if cpu_frequencies:
            clean_frequencies = [f for f in cpu_frequencies if f > 0]
            if clean_frequencies:
                analysis['frequency_statistics'] = {
                    'mean_frequency': float(np.mean(clean_frequencies)),
                    'frequency_range': float(np.max(clean_frequencies) - np.min(clean_frequencies)),
                    'frequency_stability': 1.0 / (1.0 + np.std(clean_frequencies))
                }
        
        return analysis
    
    def _analyze_memory_signals(self, memory_data: List[Dict]) -> Dict[str, Any]:
        """Detailed analysis of memory signals"""
        if not memory_data:
            return {}
        
        # Extract memory usage data
        virtual_memory_usage = [d['virtual_memory']['percent'] for d in memory_data]
        available_memory = [d['virtual_memory']['available'] for d in memory_data]
        swap_usage = [d['swap_memory']['percent'] for d in memory_data if d['swap_memory']['total'] > 0]
        
        analysis = {
            'virtual_memory': {
                'mean_usage': float(np.mean(virtual_memory_usage)),
                'max_usage': float(np.max(virtual_memory_usage)),
                'usage_variance': float(np.var(virtual_memory_usage)),
                'memory_pressure_events': len([u for u in virtual_memory_usage if u > 90])
            },
            'available_memory': {
                'mean_available': float(np.mean(available_memory)),
                'min_available': float(np.min(available_memory)),
                'availability_trend': self._calculate_trend(
                    list(range(len(available_memory))), available_memory
                ) if len(available_memory) > 2 else 0
            }
        }
        
        if swap_usage:
            analysis['swap_memory'] = {
                'mean_swap_usage': float(np.mean(swap_usage)),
                'max_swap_usage': float(np.max(swap_usage)),
                'swap_events': len([u for u in swap_usage if u > 1])
            }
        
        return analysis
    
    def _analyze_audio_signals(self, audio_data: List[Dict]) -> Dict[str, Any]:
        """Detailed analysis of audio signals"""
        if not audio_data:
            return {}
        
        rms_levels = [d['rms_level'] for d in audio_data]
        peak_levels = [d['peak_level'] for d in audio_data]
        zero_crossings = [d['zero_crossings'] for d in audio_data]
        dominant_freqs = [d['dominant_frequency'] for d in audio_data if 'dominant_frequency' in d]
        
        analysis = {
            'amplitude_statistics': {
                'mean_rms': float(np.mean(rms_levels)),
                'max_peak': float(np.max(peak_levels)),
                'dynamic_range': float(np.max(peak_levels) - np.min(peak_levels)),
                'rms_stability': 1.0 / (1.0 + np.std(rms_levels))
            },
            'signal_characteristics': {
                'mean_zero_crossings': float(np.mean(zero_crossings)),
                'zero_crossing_variance': float(np.var(zero_crossings)),
                'signal_activity_periods': len([level for level in rms_levels if level > np.mean(rms_levels) * 2])
            }
        }
        
        if dominant_freqs:
            analysis['frequency_characteristics'] = {
                'mean_dominant_frequency': float(np.mean(dominant_freqs)),
                'frequency_range': float(np.max(dominant_freqs) - np.min(dominant_freqs)),
                'frequency_stability': 1.0 / (1.0 + np.std(dominant_freqs))
            }
        
        return analysis
    
    def _analyze_timing_signals(self, timing_data: List[Dict]) -> Dict[str, Any]:
        """Detailed analysis of timing signals"""
        if not timing_data or len(timing_data) < 2:
            return {}
        
        # Extract timing measurements
        timestamps = [d['timestamp'] for d in timing_data]
        perf_counters = [d['perf_counter'] for d in timing_data]
        monotonic_times = [d['monotonic'] for d in timing_data]
        
        # Calculate timing intervals
        timestamp_intervals = np.diff(timestamps)
        perf_intervals = np.diff(perf_counters)
        monotonic_intervals = np.diff(monotonic_times)
        
        analysis = {
            'timing_stability': {
                'timestamp_jitter': float(np.std(timestamp_intervals)),
                'perf_counter_jitter': float(np.std(perf_intervals)),
                'monotonic_jitter': float(np.std(monotonic_intervals)),
                'timing_regularity': 1.0 / (1.0 + np.std(timestamp_intervals))
            },
            'clock_analysis': {
                'mean_interval': float(np.mean(timestamp_intervals)),
                'interval_variance': float(np.var(timestamp_intervals)),
                'max_interval_deviation': float(np.max(np.abs(timestamp_intervals - np.mean(timestamp_intervals))))
            }
        }
        
        # Clock drift analysis
        if len(perf_counters) > 2:
            # Linear regression to detect drift
            x = np.arange(len(perf_counters))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, perf_counters)
            
            analysis['clock_drift'] = {
                'drift_rate': float(slope),
                'drift_correlation': float(r_value),
                'drift_significance': float(p_value),
                'drift_stability': 1.0 / (1.0 + std_err)
            }
        
        return analysis
    
    def _analyze_connectivity_patterns(self, connectivity_data: List[Dict]) -> Dict[str, Any]:
        """Analyze connectivity patterns and reliability"""
        if not connectivity_data:
            return {}
        
        # Extract connectivity metrics
        success_rates = []
        response_times = []
        host_performance = {}
        
        for sample in connectivity_data:
            total_tests = len(sample.get('connectivity_tests', []))
            successful_tests = sample.get('total_successful_connections', 0)
            
            if total_tests > 0:
                success_rate = successful_tests / total_tests
                success_rates.append(success_rate)
            
            if sample.get('average_response_time', 0) > 0:
                response_times.append(sample['average_response_time'])
            
            # Per-host analysis
            for test in sample.get('connectivity_tests', []):
                host = test['host']
                if host not in host_performance:
                    host_performance[host] = {'successes': 0, 'attempts': 0, 'response_times': []}
                
                host_performance[host]['attempts'] += 1
                if test['http_success']:
                    host_performance[host]['successes'] += 1
                    if test['http_time'] > 0:
                        host_performance[host]['response_times'].append(test['http_time'])
        
        analysis = {
            'overall_reliability': {
                'mean_success_rate': float(np.mean(success_rates)) if success_rates else 0,
                'min_success_rate': float(np.min(success_rates)) if success_rates else 0,
                'success_rate_variance': float(np.var(success_rates)) if success_rates else 0,
                'reliability_score': float(np.mean(success_rates)) if success_rates else 0
            }
        }
        
        if response_times:
            analysis['response_performance'] = {
                'mean_response_time': float(np.mean(response_times)),
                'median_response_time': float(np.median(response_times)),
                'response_time_p95': float(np.percentile(response_times, 95)),
                'response_time_variance': float(np.var(response_times))
            }
        
        # Per-host analysis
        host_stats = {}
        for host, perf in host_performance.items():
            if perf['attempts'] > 0:
                host_stats[host] = {
                    'success_rate': perf['successes'] / perf['attempts'],
                    'total_attempts': perf['attempts'],
                    'mean_response_time': np.mean(perf['response_times']) if perf['response_times'] else 0
                }
        
        if host_stats:
            analysis['host_performance'] = host_stats
        
        return analysis
    
    def _analyze_latency_patterns(self, latency_data: List[Dict]) -> Dict[str, Any]:
        """Analyze network latency patterns"""
        if not latency_data:
            return {}
        
        all_latencies = []
        host_latencies = {}
        
        for sample in latency_data:
            for test in sample.get('latency_tests', []):
                if test['success'] and test['latency_ms'] > 0:
                    latency_ms = test['latency_ms']
                    all_latencies.append(latency_ms)
                    
                    host = test['host']
                    if host not in host_latencies:
                        host_latencies[host] = []
                    host_latencies[host].append(latency_ms)
        
        analysis = {}
        
        if all_latencies:
            analysis['overall_latency'] = {
                'mean_latency': float(np.mean(all_latencies)),
                'median_latency': float(np.median(all_latencies)),
                'min_latency': float(np.min(all_latencies)),
                'max_latency': float(np.max(all_latencies)),
                'latency_jitter': float(np.std(all_latencies)),
                'percentiles': {
                    '50th': float(np.percentile(all_latencies, 50)),
                    '95th': float(np.percentile(all_latencies, 95)),
                    '99th': float(np.percentile(all_latencies, 99))
                }
            }
            
            # Latency quality classification
            excellent_count = len([l for l in all_latencies if l < 20])
            good_count = len([l for l in all_latencies if 20 <= l < 50])
            acceptable_count = len([l for l in all_latencies if 50 <= l < 100])
            poor_count = len([l for l in all_latencies if l >= 100])
            
            total_count = len(all_latencies)
            analysis['latency_quality'] = {
                'excellent_percentage': (excellent_count / total_count) * 100,
                'good_percentage': (good_count / total_count) * 100,
                'acceptable_percentage': (acceptable_count / total_count) * 100,
                'poor_percentage': (poor_count / total_count) * 100
            }
        
        # Per-host latency analysis
        host_stats = {}
        for host, latencies in host_latencies.items():
            if latencies:
                host_stats[host] = {
                    'mean_latency': float(np.mean(latencies)),
                    'latency_variance': float(np.var(latencies)),
                    'latency_stability': 1.0 / (1.0 + np.std(latencies))
                }
        
        if host_stats:
            analysis['host_latency_performance'] = host_stats
        
        return analysis
    
    def _analyze_dns_patterns(self, dns_data: List[Dict]) -> Dict[str, Any]:
        """Analyze DNS resolution patterns"""
        if not dns_data:
            return {}
        
        all_resolution_times = []
        domain_performance = {}
        
        for sample in dns_data:
            for test in sample.get('dns_tests', []):
                if test['success'] and test['resolution_time'] > 0:
                    resolution_time = test['resolution_time']
                    all_resolution_times.append(resolution_time)
                    
                    domain = test['domain']
                    if domain not in domain_performance:
                        domain_performance[domain] = []
                    domain_performance[domain].append(resolution_time)
        
        analysis = {}
        
        if all_resolution_times:
            analysis['overall_dns_performance'] = {
                'mean_resolution_time': float(np.mean(all_resolution_times)),
                'median_resolution_time': float(np.median(all_resolution_times)),
                'resolution_time_p95': float(np.percentile(all_resolution_times, 95)),
                'dns_stability': 1.0 / (1.0 + np.std(all_resolution_times))
            }
        
        # Per-domain analysis
        domain_stats = {}
        for domain, times in domain_performance.items():
            if times:
                domain_stats[domain] = {
                    'mean_resolution_time': float(np.mean(times)),
                    'resolution_variance': float(np.var(times)),
                    'fastest_resolution': float(np.min(times)),
                    'slowest_resolution': float(np.max(times))
                }
        
        if domain_stats:
            analysis['domain_performance'] = domain_stats
        
        return analysis
    
    def _analyze_traffic_patterns(self, traffic_data: List[Dict]) -> Dict[str, Any]:
        """Analyze network traffic patterns"""
        if not traffic_data or len(traffic_data) < 2:
            return {}
        
        # Extract traffic rates
        send_rates = [d.get('bytes_sent_rate', 0) for d in traffic_data[1:]]  # Skip first sample
        recv_rates = [d.get('bytes_recv_rate', 0) for d in traffic_data[1:]]
        
        # Remove negative rates (can happen due to counter resets)
        send_rates = [rate for rate in send_rates if rate >= 0]
        recv_rates = [rate for rate in recv_rates if rate >= 0]
        
        analysis = {}
        
        if send_rates:
            analysis['outbound_traffic'] = {
                'mean_send_rate': float(np.mean(send_rates)),
                'max_send_rate': float(np.max(send_rates)),
                'send_rate_variance': float(np.var(send_rates)),
                'traffic_bursts': len([rate for rate in send_rates if rate > np.mean(send_rates) * 3])
            }
        
        if recv_rates:
            analysis['inbound_traffic'] = {
                'mean_recv_rate': float(np.mean(recv_rates)),
                'max_recv_rate': float(np.max(recv_rates)),
                'recv_rate_variance': float(np.var(recv_rates)),
                'download_bursts': len([rate for rate in recv_rates if rate > np.mean(recv_rates) * 3])
            }
        
        if send_rates and recv_rates:
            # Traffic balance analysis
            total_send = sum(send_rates)
            total_recv = sum(recv_rates)
            
            analysis['traffic_balance'] = {
                'upload_download_ratio': total_send / total_recv if total_recv > 0 else 0,
                'traffic_symmetry': min(total_send, total_recv) / max(total_send, total_recv) if max(total_send, total_recv) > 0 else 0
            }
        
        return analysis
    
    def _analyze_interface_patterns(self, interface_data: List[Dict]) -> Dict[str, Any]:
        """Analyze network interface reliability patterns"""
        if not interface_data:
            return {}
        
        # Track interface stability and errors
        interface_stats = {}
        
        for sample in interface_data:
            for interface_name, interface_info in sample.get('interfaces', {}).items():
                if interface_name not in interface_stats:
                    interface_stats[interface_name] = {
                        'samples': 0,
                        'up_time': 0,
                        'errors': [],
                        'drops': []
                    }
                
                stats = interface_stats[interface_name]
                stats['samples'] += 1
                
                if interface_info.get('is_up'):
                    stats['up_time'] += 1
                
                # Track errors and drops
                errors = interface_info.get('errin', 0) + interface_info.get('errout', 0)
                drops = interface_info.get('dropin', 0) + interface_info.get('dropout', 0)
                
                stats['errors'].append(errors)
                stats['drops'].append(drops)
        
        analysis = {}
        
        for interface_name, stats in interface_stats.items():
            if stats['samples'] > 0:
                # Calculate error rates
                error_increases = 0
                drop_increases = 0
                
                if len(stats['errors']) > 1:
                    error_increases = sum(1 for i in range(1, len(stats['errors'])) 
                                        if stats['errors'][i] > stats['errors'][i-1])
                
                if len(stats['drops']) > 1:
                    drop_increases = sum(1 for i in range(1, len(stats['drops'])) 
                                       if stats['drops'][i] > stats['drops'][i-1])
                
                analysis[interface_name] = {
                    'uptime_percentage': (stats['up_time'] / stats['samples']) * 100,
                    'error_events': error_increases,
                    'drop_events': drop_increases,
                    'reliability_score': (stats['up_time'] / stats['samples']) * (1.0 / (1.0 + error_increases + drop_increases))
                }
        
        return analysis
    
    def _analyze_cross_correlations(self, hardware_data: Dict, network_data: Dict) -> Dict[str, Any]:
        """Analyze correlations between hardware and network signals"""
        correlations = {}
        
        try:
            # CPU usage vs network activity correlation
            cpu_signals = hardware_data.get('hardware_signals', {}).get('cpu_signals', [])
            network_traffic = network_data.get('network_signals', {}).get('traffic_signals', [])
            
            if cpu_signals and network_traffic:
                cpu_usage_series = []
                network_activity_series = []
                
                # Align timestamps (approximate matching)
                for cpu_sample in cpu_signals:
                    cpu_timestamp = cpu_sample['timestamp']
                    
                    if cpu_sample.get('cpu_percent'):
                        avg_cpu = np.mean(cpu_sample['cpu_percent'])
                        cpu_usage_series.append((cpu_timestamp, avg_cpu))
                
                for net_sample in network_traffic:
                    net_timestamp = net_sample['timestamp']
                    net_activity = (net_sample.get('bytes_sent_rate', 0) + 
                                  net_sample.get('bytes_recv_rate', 0))
                    network_activity_series.append((net_timestamp, net_activity))
                
                # Find overlapping time periods and calculate correlation
                if len(cpu_usage_series) > 3 and len(network_activity_series) > 3:
                    correlation = self._calculate_time_series_correlation(
                        cpu_usage_series, network_activity_series
                    )
                    correlations['cpu_network_correlation'] = correlation
            
            # Memory usage vs network latency correlation  
            memory_signals = hardware_data.get('hardware_signals', {}).get('memory_signals', [])
            latency_signals = network_data.get('network_signals', {}).get('latency_signals', [])
            
            if memory_signals and latency_signals:
                memory_series = [(d['timestamp'], d['virtual_memory']['percent']) for d in memory_signals]
                
                latency_series = []
                for latency_sample in latency_signals:
                    timestamp = latency_sample['timestamp']
                    avg_latency = latency_sample.get('average_latency', 0)
                    if avg_latency > 0:
                        latency_series.append((timestamp, avg_latency))
                
                if len(memory_series) > 3 and len(latency_series) > 3:
                    correlation = self._calculate_time_series_correlation(
                        memory_series, latency_series
                    )
                    correlations['memory_latency_correlation'] = correlation
            
        except Exception as e:
            self.logger.warning(f"Cross-correlation analysis error: {e}")
        
        return correlations
    
    def _analyze_temporal_patterns(self, hardware_data: Optional[Dict], 
                                 network_data: Optional[Dict]) -> Dict[str, Any]:
        """Analyze temporal patterns across all signals"""
        patterns = {
            'periodicity_analysis': {},
            'trend_analysis': {},
            'seasonal_patterns': {}
        }
        
        # Analyze hardware signal patterns
        if hardware_data:
            hw_signals = hardware_data.get('hardware_signals', {})
            
            # CPU usage periodicity
            if hw_signals.get('cpu_signals'):
                cpu_data = hw_signals['cpu_signals']
                cpu_usage = []
                timestamps = []
                
                for sample in cpu_data:
                    if sample.get('cpu_percent'):
                        cpu_usage.append(np.mean(sample['cpu_percent']))
                        timestamps.append(sample['timestamp'])
                
                if len(cpu_usage) > 10:
                    periodicity = self._detect_periodicity(cpu_usage)
                    patterns['periodicity_analysis']['cpu_usage'] = periodicity
        
        # Analyze network signal patterns
        if network_data:
            net_signals = network_data.get('network_signals', {})
            
            # Latency periodicity
            if net_signals.get('latency_signals'):
                latency_data = net_signals['latency_signals']
                latencies = []
                
                for sample in latency_data:
                    avg_latency = sample.get('average_latency', 0)
                    if avg_latency > 0:
                        latencies.append(avg_latency)
                
                if len(latencies) > 10:
                    periodicity = self._detect_periodicity(latencies)
                    patterns['periodicity_analysis']['network_latency'] = periodicity
        
        return patterns
    
    def _analyze_frequency_domain(self, hardware_data: Optional[Dict], 
                                network_data: Optional[Dict]) -> Dict[str, Any]:
        """Perform frequency domain analysis of signals"""
        frequency_analysis = {}
        
        try:
            # Hardware signal frequency analysis
            if hardware_data:
                hw_signals = hardware_data.get('hardware_signals', {})
                
                # CPU usage frequency spectrum
                if hw_signals.get('cpu_signals'):
                    cpu_data = hw_signals['cpu_signals']
                    cpu_usage = []
                    
                    for sample in cpu_data:
                        if sample.get('cpu_percent'):
                            cpu_usage.append(np.mean(sample['cpu_percent']))
                    
                    if len(cpu_usage) > 16:  # Minimum for meaningful FFT
                        fft_analysis = self._perform_fft_analysis(cpu_usage)
                        frequency_analysis['cpu_usage_spectrum'] = fft_analysis
                
                # Audio frequency analysis (if available)
                if hw_signals.get('audio_signals'):
                    audio_data = hw_signals['audio_signals']
                    rms_levels = [d['rms_level'] for d in audio_data]
                    
                    if len(rms_levels) > 16:
                        fft_analysis = self._perform_fft_analysis(rms_levels)
                        frequency_analysis['audio_rms_spectrum'] = fft_analysis
            
            # Network signal frequency analysis
            if network_data:
                net_signals = network_data.get('network_signals', {})
                
                # Latency frequency spectrum
                if net_signals.get('latency_signals'):
                    latency_data = net_signals['latency_signals']
                    latencies = [sample.get('average_latency', 0) for sample in latency_data 
                               if sample.get('average_latency', 0) > 0]
                    
                    if len(latencies) > 16:
                        fft_analysis = self._perform_fft_analysis(latencies)
                        frequency_analysis['latency_spectrum'] = fft_analysis
        
        except Exception as e:
            self.logger.warning(f"Frequency domain analysis error: {e}")
        
        return frequency_analysis
    
    def _assess_signal_quality(self, hardware_data: Optional[Dict], 
                             network_data: Optional[Dict]) -> Dict[str, Any]:
        """Assess overall signal quality and reliability"""
        quality_assessment = {
            'hardware_quality': {},
            'network_quality': {},
            'overall_quality_score': 0
        }
        
        quality_scores = []
        
        # Hardware signal quality
        if hardware_data:
            hw_analysis = self.metrics.get('hardware_metrics', {})
            
            # CPU signal quality
            if 'cpu' in hw_analysis.get('signal_statistics', {}):
                cpu_stats = hw_analysis['signal_statistics']['cpu']
                cpu_quality = self._calculate_cpu_quality_score(cpu_stats)
                quality_assessment['hardware_quality']['cpu'] = cpu_quality
                quality_scores.append(cpu_quality)
            
            # Memory signal quality
            if 'memory' in hw_analysis.get('signal_statistics', {}):
                memory_stats = hw_analysis['signal_statistics']['memory']
                memory_quality = self._calculate_memory_quality_score(memory_stats)
                quality_assessment['hardware_quality']['memory'] = memory_quality
                quality_scores.append(memory_quality)
        
        # Network signal quality
        if network_data:
            net_analysis = self.metrics.get('network_metrics', {})
            
            # Connectivity quality
            conn_analysis = net_analysis.get('connectivity_analysis', {})
            if conn_analysis:
                conn_quality = self._calculate_connectivity_quality_score(conn_analysis)
                quality_assessment['network_quality']['connectivity'] = conn_quality
                quality_scores.append(conn_quality)
            
            # Latency quality  
            perf_analysis = net_analysis.get('performance_analysis', {})
            if 'latency' in perf_analysis:
                latency_quality = self._calculate_latency_quality_score(perf_analysis['latency'])
                quality_assessment['network_quality']['latency'] = latency_quality
                quality_scores.append(latency_quality)
        
        # Overall quality score
        if quality_scores:
            quality_assessment['overall_quality_score'] = float(np.mean(quality_scores))
        
        return quality_assessment
    
    def _detect_anomalies(self, hardware_data: Optional[Dict], 
                         network_data: Optional[Dict]) -> Dict[str, Any]:
        """Detect anomalies in signal patterns"""
        anomalies = {
            'hardware_anomalies': {},
            'network_anomalies': {},
            'cross_signal_anomalies': {}
        }
        
        # Hardware anomaly detection
        if hardware_data:
            hw_signals = hardware_data.get('hardware_signals', {})
            
            # CPU usage anomalies
            if hw_signals.get('cpu_signals'):
                cpu_anomalies = self._detect_cpu_anomalies(hw_signals['cpu_signals'])
                anomalies['hardware_anomalies']['cpu'] = cpu_anomalies
            
            # Memory usage anomalies
            if hw_signals.get('memory_signals'):
                memory_anomalies = self._detect_memory_anomalies(hw_signals['memory_signals'])
                anomalies['hardware_anomalies']['memory'] = memory_anomalies
        
        # Network anomaly detection
        if network_data:
            net_signals = network_data.get('network_signals', {})
            
            # Latency anomalies
            if net_signals.get('latency_signals'):
                latency_anomalies = self._detect_latency_anomalies(net_signals['latency_signals'])
                anomalies['network_anomalies']['latency'] = latency_anomalies
            
            # Connectivity anomalies
            if net_signals.get('connectivity_signals'):
                connectivity_anomalies = self._detect_connectivity_anomalies(net_signals['connectivity_signals'])
                anomalies['network_anomalies']['connectivity'] = connectivity_anomalies
        
        return anomalies
    
    def _perform_statistical_validation(self, hardware_data: Optional[Dict], 
                                      network_data: Optional[Dict]) -> Dict[str, Any]:
        """Perform statistical validation of signal data"""
        validation = {
            'normality_tests': {},
            'stationarity_tests': {},
            'distribution_analysis': {},
            'statistical_significance': {}
        }
        
        # Hardware signal validation
        if hardware_data:
            hw_signals = hardware_data.get('hardware_signals', {})
            
            # CPU usage distribution analysis
            if hw_signals.get('cpu_signals'):
                cpu_data = hw_signals['cpu_signals']
                cpu_usage = []
                
                for sample in cpu_data:
                    if sample.get('cpu_percent'):
                        cpu_usage.extend(sample['cpu_percent'])
                
                if len(cpu_usage) > 8:  # Minimum for statistical tests
                    validation['normality_tests']['cpu_usage'] = self._test_normality(cpu_usage)
                    validation['distribution_analysis']['cpu_usage'] = self._analyze_distribution(cpu_usage)
        
        # Network signal validation
        if network_data:
            net_signals = network_data.get('network_signals', {})
            
            # Latency distribution analysis
            if net_signals.get('latency_signals'):
                all_latencies = []
                for sample in net_signals['latency_signals']:
                    for test in sample.get('latency_tests', []):
                        if test['success'] and test['latency_ms'] > 0:
                            all_latencies.append(test['latency_ms'])
                
                if len(all_latencies) > 8:
                    validation['normality_tests']['network_latency'] = self._test_normality(all_latencies)
                    validation['distribution_analysis']['network_latency'] = self._analyze_distribution(all_latencies)
        
        return validation
    
    # Helper methods for calculations
    
    def _calculate_trend(self, x: List[float], y: List[float]) -> float:
        """Calculate linear trend (slope) of time series data"""
        try:
            slope, _, r_value, _, _ = stats.linregress(x, y)
            return float(slope * r_value)  # Weight by correlation
        except:
            return 0.0
    
    def _calculate_clock_stability(self, system_times: List[float], 
                                 perf_counters: List[float]) -> float:
        """Calculate clock stability metric"""
        try:
            if len(system_times) < 3:
                return 1.0
            
            # Calculate relative drift between clocks
            sys_diffs = np.diff(system_times)
            perf_diffs = np.diff(perf_counters)
            
            if len(sys_diffs) > 0 and len(perf_diffs) > 0:
                drift_ratios = sys_diffs / perf_diffs
                stability = 1.0 / (1.0 + np.std(drift_ratios))
                return float(stability)
            
            return 1.0
        except:
            return 1.0
    
    def _calculate_timing_jitter(self, timing_data: List[Dict]) -> float:
        """Calculate timing jitter from timing signals"""
        try:
            timestamps = [d['timestamp'] for d in timing_data]
            if len(timestamps) < 3:
                return 0.0
            
            intervals = np.diff(timestamps)
            expected_interval = np.mean(intervals)
            jitter = np.std(intervals) / expected_interval if expected_interval > 0 else 0
            return float(jitter)
        except:
            return 0.0
    
    def _calculate_time_series_correlation(self, series1: List[Tuple[float, float]], 
                                         series2: List[Tuple[float, float]]) -> float:
        """Calculate correlation between two time series with potentially different timestamps"""
        try:
            # Simple approach: interpolate to common time grid
            if len(series1) < 3 or len(series2) < 3:
                return 0.0
            
            # Extract values at closest timestamps
            values1 = []
            values2 = []
            
            for t1, v1 in series1:
                # Find closest timestamp in series2
                closest_idx = min(range(len(series2)), 
                                key=lambda i: abs(series2[i][0] - t1))
                
                if abs(series2[closest_idx][0] - t1) < 5.0:  # Within 5 seconds
                    values1.append(v1)
                    values2.append(series2[closest_idx][1])
            
            if len(values1) > 2:
                correlation, _ = stats.pearsonr(values1, values2)
                return float(correlation)
            
            return 0.0
        except:
            return 0.0
    
    def _detect_periodicity(self, signal: List[float]) -> Dict[str, float]:
        """Detect periodic patterns in signal using autocorrelation"""
        try:
            if len(signal) < 10:
                return {'dominant_period': 0, 'periodicity_strength': 0}
            
            # Normalize signal
            signal_norm = (signal - np.mean(signal)) / np.std(signal)
            
            # Calculate autocorrelation
            autocorr = np.correlate(signal_norm, signal_norm, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks in autocorrelation (excluding zero lag)
            peaks = []
            for i in range(2, min(len(autocorr), len(signal)//2)):
                if (autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and 
                    autocorr[i] > 0.1):  # Minimum correlation threshold
                    peaks.append((i, autocorr[i]))
            
            if peaks:
                # Find strongest period
                dominant_period, strength = max(peaks, key=lambda x: x[1])
                return {
                    'dominant_period': float(dominant_period),
                    'periodicity_strength': float(strength)
                }
            
            return {'dominant_period': 0, 'periodicity_strength': 0}
        except:
            return {'dominant_period': 0, 'periodicity_strength': 0}
    
    def _perform_fft_analysis(self, signal: List[float]) -> Dict[str, Any]:
        """Perform FFT analysis on signal"""
        try:
            if len(signal) < 8:
                return {}
            
            # Window the signal to reduce spectral leakage
            windowed_signal = signal * np.hanning(len(signal))
            
            # Compute FFT
            fft_result = np.fft.fft(windowed_signal)
            freqs = np.fft.fftfreq(len(signal))
            
            # Power spectrum
            power_spectrum = np.abs(fft_result)**2
            
            # Find dominant frequency (excluding DC component)
            dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            dominant_freq = freqs[dominant_freq_idx]
            
            return {
                'dominant_frequency': float(dominant_freq),
                'spectral_centroid': float(np.sum(freqs[:len(freqs)//2] * power_spectrum[:len(power_spectrum)//2]) / 
                                         np.sum(power_spectrum[:len(power_spectrum)//2])),
                'spectral_bandwidth': float(np.std(freqs[:len(freqs)//2], 
                                                  weights=power_spectrum[:len(power_spectrum)//2])),
                'total_power': float(np.sum(power_spectrum))
            }
        except:
            return {}
    
    def _calculate_cpu_quality_score(self, cpu_stats: Dict) -> float:
        """Calculate CPU signal quality score (0-1)"""
        try:
            score = 1.0
            
            # Penalize high variance (unstable signals)
            if 'usage_statistics' in cpu_stats:
                usage_std = cpu_stats['usage_statistics']['std']
                score *= 1.0 / (1.0 + usage_std / 50.0)  # Normalize by reasonable std
            
            # Reward good core balance
            if 'core_balance' in cpu_stats:
                imbalance = cpu_stats['core_balance']['load_imbalance_score']
                score *= 1.0 / (1.0 + imbalance / 10.0)
            
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def _calculate_memory_quality_score(self, memory_stats: Dict) -> float:
        """Calculate memory signal quality score (0-1)"""
        try:
            score = 1.0
            
            # Penalize high memory usage
            if 'virtual_memory' in memory_stats:
                mean_usage = memory_stats['virtual_memory']['mean_usage']
                score *= max(0.1, 1.0 - mean_usage / 100.0)
                
                # Penalize memory pressure events
                pressure_events = memory_stats['virtual_memory']['memory_pressure_events']
                score *= 1.0 / (1.0 + pressure_events)
            
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def _calculate_connectivity_quality_score(self, conn_stats: Dict) -> float:
        """Calculate connectivity quality score (0-1)"""
        try:
            if 'overall_reliability' in conn_stats:
                success_rate = conn_stats['overall_reliability']['mean_success_rate']
                return float(success_rate)
            return 0.5
        except:
            return 0.5
    
    def _calculate_latency_quality_score(self, latency_stats: Dict) -> float:
        """Calculate latency quality score (0-1)"""
        try:
            score = 1.0
            
            if 'overall_latency' in latency_stats:
                mean_latency = latency_stats['overall_latency']['mean_latency']
                # Good: <50ms, Acceptable: <100ms, Poor: >100ms
                if mean_latency < 50:
                    score = 1.0
                elif mean_latency < 100:
                    score = 0.7
                else:
                    score = max(0.1, 1.0 / (1.0 + mean_latency / 100.0))
                
                # Penalize high jitter
                jitter = latency_stats['overall_latency']['latency_jitter']
                score *= 1.0 / (1.0 + jitter / 50.0)
            
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def _detect_cpu_anomalies(self, cpu_data: List[Dict]) -> Dict[str, Any]:
        """Detect anomalies in CPU usage patterns"""
        anomalies = {'spike_events': 0, 'sustained_high_load': 0, 'unusual_patterns': []}
        
        try:
            cpu_usage = []
            for sample in cpu_data:
                if sample.get('cpu_percent'):
                    avg_usage = np.mean(sample['cpu_percent'])
                    cpu_usage.append(avg_usage)
            
            if len(cpu_usage) < 5:
                return anomalies
            
            mean_usage = np.mean(cpu_usage)
            std_usage = np.std(cpu_usage)
            
            # Detect spikes (usage > mean + 2*std)
            spike_threshold = mean_usage + 2 * std_usage
            anomalies['spike_events'] = len([u for u in cpu_usage if u > spike_threshold])
            
            # Detect sustained high load (>80% for extended periods)
            high_load_count = 0
            for usage in cpu_usage:
                if usage > 80:
                    high_load_count += 1
                else:
                    if high_load_count > len(cpu_usage) * 0.3:  # >30% of samples
                        anomalies['sustained_high_load'] += 1
                    high_load_count = 0
            
        except Exception as e:
            self.logger.warning(f"CPU anomaly detection error: {e}")
        
        return anomalies
    
    def _detect_memory_anomalies(self, memory_data: List[Dict]) -> Dict[str, Any]:
        """Detect anomalies in memory usage patterns"""
        anomalies = {'memory_leaks': 0, 'sudden_spikes': 0, 'low_memory_events': 0}
        
        try:
            memory_usage = [d['virtual_memory']['percent'] for d in memory_data]
            
            if len(memory_usage) < 5:
                return anomalies
            
            # Detect potential memory leaks (consistent upward trend)
            if len(memory_usage) > 10:
                slope = self._calculate_trend(list(range(len(memory_usage))), memory_usage)
                if slope > 0.5:  # Increasing at >0.5% per sample
                    anomalies['memory_leaks'] = 1
            
            # Detect sudden spikes
            if len(memory_usage) > 2:
                for i in range(1, len(memory_usage)):
                    change = memory_usage[i] - memory_usage[i-1]
                    if change > 15:  # >15% increase in one sample
                        anomalies['sudden_spikes'] += 1
            
            # Low memory events
            anomalies['low_memory_events'] = len([u for u in memory_usage if u > 90])
            
        except Exception as e:
            self.logger.warning(f"Memory anomaly detection error: {e}")
        
        return anomalies
    
    def _detect_latency_anomalies(self, latency_data: List[Dict]) -> Dict[str, Any]:
        """Detect anomalies in network latency patterns"""
        anomalies = {'timeout_events': 0, 'latency_spikes': 0, 'connection_failures': 0}
        
        try:
            all_latencies = []
            total_tests = 0
            failed_tests = 0
            
            for sample in latency_data:
                for test in sample.get('latency_tests', []):
                    total_tests += 1
                    if test['success']:
                        if test['latency_ms'] > 0:
                            all_latencies.append(test['latency_ms'])
                    else:
                        failed_tests += 1
            
            if all_latencies:
                mean_latency = np.mean(all_latencies)
                std_latency = np.std(all_latencies)
                
                # Detect latency spikes
                spike_threshold = mean_latency + 3 * std_latency
                anomalies['latency_spikes'] = len([l for l in all_latencies if l > spike_threshold])
                
                # Detect timeouts (very high latencies)
                anomalies['timeout_events'] = len([l for l in all_latencies if l > 3000])  # >3 seconds
            
            # Connection failure rate
            if total_tests > 0:
                failure_rate = failed_tests / total_tests
                if failure_rate > 0.1:  # >10% failure rate
                    anomalies['connection_failures'] = failed_tests
            
        except Exception as e:
            self.logger.warning(f"Latency anomaly detection error: {e}")
        
        return anomalies
    
    def _detect_connectivity_anomalies(self, connectivity_data: List[Dict]) -> Dict[str, Any]:
        """Detect anomalies in connectivity patterns"""
        anomalies = {'outages': 0, 'intermittent_failures': 0, 'degraded_performance': 0}
        
        try:
            success_rates = []
            response_times = []
            
            for sample in connectivity_data:
                total_tests = len(sample.get('connectivity_tests', []))
                successful_tests = sample.get('total_successful_connections', 0)
                
                if total_tests > 0:
                    success_rate = successful_tests / total_tests
                    success_rates.append(success_rate)
                
                avg_response = sample.get('average_response_time', 0)
                if avg_response > 0:
                    response_times.append(avg_response)
            
            # Detect outages (success rate = 0)
            anomalies['outages'] = len([rate for rate in success_rates if rate == 0])
            
            # Detect intermittent failures (success rate < 50%)
            anomalies['intermittent_failures'] = len([rate for rate in success_rates if 0 < rate < 0.5])
            
            # Detect degraded performance (response time > 2x average)
            if response_times:
                avg_response_time = np.mean(response_times)
                degraded_threshold = avg_response_time * 2
                anomalies['degraded_performance'] = len([rt for rt in response_times if rt > degraded_threshold])
            
        except Exception as e:
            self.logger.warning(f"Connectivity anomaly detection error: {e}")
        
        return anomalies
    
    def _test_normality(self, data: List[float]) -> Dict[str, float]:
        """Test if data follows normal distribution"""
        try:
            if len(data) < 8:
                return {'shapiro_stat': 0, 'shapiro_p': 1, 'is_normal': False}
            
            # Shapiro-Wilk test
            stat, p_value = stats.shapiro(data[:5000])  # Limit sample size for performance
            
            return {
                'shapiro_stat': float(stat),
                'shapiro_p': float(p_value),
                'is_normal': bool(p_value > 0.05)
            }
        except:
            return {'shapiro_stat': 0, 'shapiro_p': 1, 'is_normal': False}
    
    def _analyze_distribution(self, data: List[float]) -> Dict[str, float]:
        """Analyze data distribution characteristics"""
        try:
            data_array = np.array(data)
            
            return {
                'skewness': float(stats.skew(data_array)),
                'kurtosis': float(stats.kurtosis(data_array)),
                'mean': float(np.mean(data_array)),
                'median': float(np.median(data_array)),
                'std': float(np.std(data_array)),
                'range': float(np.max(data_array) - np.min(data_array))
            }
        except:
            return {}
    
    def save_results(self, filepath: str, results: Optional[Dict] = None):
        """Save signal metrics analysis results to JSON file"""
        if results is None:
            results = self.analyze_all_signals()
        
        # Convert numpy types to JSON serializable
        def convert_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, bool):
                return bool(obj)
            return obj
        
        json_results = json.loads(json.dumps(results, default=convert_for_json))
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Signal metrics analysis saved to {filepath}")
        return json_results
    
    def create_visualizations(self, output_dir: str, results: Optional[Dict] = None):
        """Create comprehensive visualizations of signal analysis"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if results is None:
            results = self.analyze_all_signals()
        
        try:
            # Create comprehensive analysis dashboard
            fig, axes = plt.subplots(3, 3, figsize=(20, 15))
            fig.suptitle('Comprehensive Signal Metrics Analysis', fontsize=16, fontweight='bold')
            
            metrics = results['signal_metrics']
            
            # Plot 1: Quality scores overview
            ax = axes[0, 0]
            quality_metrics = metrics.get('quality_metrics', {})
            
            quality_scores = []
            quality_labels = []
            
            if 'hardware_quality' in quality_metrics:
                for component, score in quality_metrics['hardware_quality'].items():
                    quality_labels.append(f'HW {component}')
                    quality_scores.append(score)
            
            if 'network_quality' in quality_metrics:
                for component, score in quality_metrics['network_quality'].items():
                    quality_labels.append(f'NET {component}')
                    quality_scores.append(score)
            
            if quality_scores:
                colors = ['green' if score > 0.7 else 'orange' if score > 0.4 else 'red' for score in quality_scores]
                bars = ax.bar(range(len(quality_labels)), quality_scores, color=colors, alpha=0.7)
                ax.set_xlabel('Component')
                ax.set_ylabel('Quality Score')
                ax.set_title('Signal Quality Assessment')
                ax.set_xticks(range(len(quality_labels)))
                ax.set_xticklabels(quality_labels, rotation=45, ha='right')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                
                # Add score labels on bars
                for bar, score in zip(bars, quality_scores):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 2: Hardware resource utilization
            ax = axes[0, 1]
            hw_metrics = metrics.get('hardware_metrics', {})
            resource_util = hw_metrics.get('resource_utilization', {})
            
            if resource_util:
                resources = list(resource_util.keys())
                utilizations = [resource_util[res]['mean_utilization'] for res in resources]
                
                colors = ['red' if util > 80 else 'orange' if util > 60 else 'green' for util in utilizations]
                bars = ax.bar(resources, utilizations, color=colors, alpha=0.7)
                ax.set_ylabel('Utilization (%)')
                ax.set_title('Average Resource Utilization')
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3)
                
                for bar, util in zip(bars, utilizations):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{util:.1f}%', ha='center', va='bottom')
            
            # Plot 3: Network performance metrics
            ax = axes[0, 2]
            net_metrics = metrics.get('network_metrics', {})
            perf_analysis = net_metrics.get('performance_analysis', {})
            
            if 'latency' in perf_analysis and 'overall_latency' in perf_analysis['latency']:
                latency_data = perf_analysis['latency']['overall_latency']
                
                latency_labels = ['Mean', 'Median', '95th %ile', '99th %ile']
                latency_values = [
                    latency_data['mean_latency'],
                    latency_data['median_latency'],
                    latency_data['percentiles']['95th'],
                    latency_data['percentiles']['99th']
                ]
                
                ax.bar(latency_labels, latency_values, color='skyblue', alpha=0.7, edgecolor='navy')
                ax.set_ylabel('Latency (ms)')
                ax.set_title('Network Latency Distribution')
                ax.grid(True, alpha=0.3)
            
            # Plot 4: Anomaly detection summary
            ax = axes[1, 0]
            anomaly_metrics = metrics.get('anomaly_metrics', {})
            
            anomaly_counts = []
            anomaly_labels = []
            
            # Hardware anomalies
            hw_anomalies = anomaly_metrics.get('hardware_anomalies', {})
            for component, anomalies in hw_anomalies.items():
                if isinstance(anomalies, dict):
                    total_anomalies = sum(v for v in anomalies.values() if isinstance(v, (int, float)))
                    anomaly_counts.append(total_anomalies)
                    anomaly_labels.append(f'HW {component}')
            
            # Network anomalies
            net_anomalies = anomaly_metrics.get('network_anomalies', {})
            for component, anomalies in net_anomalies.items():
                if isinstance(anomalies, dict):
                    total_anomalies = sum(v for v in anomalies.values() if isinstance(v, (int, float)))
                    anomaly_counts.append(total_anomalies)
                    anomaly_labels.append(f'NET {component}')
            
            if anomaly_counts:
                colors = ['red' if count > 5 else 'orange' if count > 2 else 'green' for count in anomaly_counts]
                ax.bar(range(len(anomaly_labels)), anomaly_counts, color=colors, alpha=0.7)
                ax.set_xlabel('Component')
                ax.set_ylabel('Anomaly Count')
                ax.set_title('Detected Anomalies')
                ax.set_xticks(range(len(anomaly_labels)))
                ax.set_xticklabels(anomaly_labels, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
            
            # Plot 5: Cross-correlation heatmap
            ax = axes[1, 1]
            cross_corr = metrics.get('cross_correlation_metrics', {})
            
            if cross_corr:
                corr_labels = list(cross_corr.keys())
                corr_values = list(cross_corr.values())
                
                # Create simple correlation visualization
                y_pos = range(len(corr_labels))
                colors = ['red' if abs(val) > 0.7 else 'orange' if abs(val) > 0.4 else 'green' 
                         for val in corr_values]
                
                bars = ax.barh(y_pos, corr_values, color=colors, alpha=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([label.replace('_', ' ').title() for label in corr_labels])
                ax.set_xlabel('Correlation Coefficient')
                ax.set_title('Cross-Signal Correlations')
                ax.set_xlim(-1, 1)
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                ax.grid(True, alpha=0.3)
            
            # Plot 6: Frequency domain analysis
            ax = axes[1, 2]
            freq_metrics = metrics.get('frequency_metrics', {})
            
            if freq_metrics:
                freq_components = list(freq_metrics.keys())
                dominant_freqs = []
                
                for component in freq_components:
                    if 'dominant_frequency' in freq_metrics[component]:
                        dominant_freqs.append(freq_metrics[component]['dominant_frequency'])
                
                if dominant_freqs:
                    ax.bar(range(len(freq_components)), dominant_freqs, color='purple', alpha=0.7)
                    ax.set_xlabel('Signal Component')
                    ax.set_ylabel('Dominant Frequency')
                    ax.set_title('Frequency Domain Analysis')
                    ax.set_xticks(range(len(freq_components)))
                    ax.set_xticklabels([comp.replace('_', ' ').title() for comp in freq_components], 
                                      rotation=45, ha='right')
                    ax.grid(True, alpha=0.3)
            
            # Plot 7: Statistical validation summary
            ax = axes[2, 0]
            stat_metrics = metrics.get('statistical_metrics', {})
            normality_tests = stat_metrics.get('normality_tests', {})
            
            if normality_tests:
                test_labels = list(normality_tests.keys())
                p_values = [test['shapiro_p'] for test in normality_tests.values()]
                
                colors = ['green' if p > 0.05 else 'red' for p in p_values]
                ax.bar(range(len(test_labels)), p_values, color=colors, alpha=0.7)
                ax.set_xlabel('Signal Type')
                ax.set_ylabel('Shapiro-Wilk p-value')
                ax.set_title('Normality Test Results')
                ax.set_xticks(range(len(test_labels)))
                ax.set_xticklabels([label.replace('_', ' ').title() for label in test_labels], 
                                  rotation=45, ha='right')
                ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Plot 8: Temporal patterns
            ax = axes[2, 1]
            temporal_metrics = metrics.get('temporal_metrics', {})
            periodicity = temporal_metrics.get('periodicity_analysis', {})
            
            if periodicity:
                pattern_labels = list(periodicity.keys())
                periods = [pattern['dominant_period'] for pattern in periodicity.values()]
                strengths = [pattern['periodicity_strength'] for pattern in periodicity.values()]
                
                # Scatter plot of period vs strength
                ax.scatter(periods, strengths, c=range(len(periods)), cmap='viridis', s=100, alpha=0.7)
                ax.set_xlabel('Dominant Period')
                ax.set_ylabel('Periodicity Strength')
                ax.set_title('Temporal Pattern Analysis')
                ax.grid(True, alpha=0.3)
                
                # Add labels
                for i, label in enumerate(pattern_labels):
                    ax.annotate(label.replace('_', ' ').title(), (periods[i], strengths[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Plot 9: Overall summary
            ax = axes[2, 2]
            
            # Create a summary metrics visualization
            overall_quality = quality_metrics.get('overall_quality_score', 0)
            
            # Create a gauge-like visualization
            colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
            angles = np.linspace(0, np.pi, len(colors))
            
            for i, (color, angle) in enumerate(zip(colors, angles)):
                start_angle = angle - np.pi/len(colors)/2
                end_angle = angle + np.pi/len(colors)/2
                
                wedge = plt.matplotlib.patches.Wedge((0, 0), 1, np.degrees(start_angle), 
                                                   np.degrees(end_angle), facecolor=color, alpha=0.7)
                ax.add_patch(wedge)
            
            # Add needle for current score
            needle_angle = np.pi * (1 - overall_quality)
            ax.arrow(0, 0, 0.8 * np.cos(needle_angle), 0.8 * np.sin(needle_angle),
                    head_width=0.05, head_length=0.1, fc='black', ec='black', linewidth=2)
            
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.2, 1.2)
            ax.set_aspect('equal')
            ax.set_title(f'Overall Quality Score: {overall_quality:.2f}')
            ax.axis('off')
            
            # Add score labels
            score_labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
            for i, (label, angle) in enumerate(zip(score_labels, angles)):
                x = 1.15 * np.cos(angle)
                y = 1.15 * np.sin(angle)
                ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path / "signal_metrics_dashboard.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Signal metrics visualizations saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")


def main():
    """Standalone execution of signal metrics analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Signal Metrics and Statistical Analysis System")
    parser.add_argument('--duration', type=float, default=120.0,
                       help='Analysis duration in seconds (default: 120)')
    parser.add_argument('--output-dir', default='signal_metrics_results',
                       help='Output directory for results')
    parser.add_argument('--hardware-data', type=str,
                       help='Path to pre-collected hardware signal data JSON file')
    parser.add_argument('--network-data', type=str,
                       help='Path to pre-collected network signal data JSON file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    print("📊 Signal Metrics and Statistical Analysis System")
    print("=" * 60)
    print(f"Analysis Duration: {args.duration} seconds")
    print(f"Output Directory: {args.output_dir}")
    print()
    
    # Initialize analyzer
    analyzer = SignalMetricsAnalyzer(analysis_window=args.duration)
    
    # Load pre-collected data if provided
    hardware_data = None
    network_data = None
    
    if args.hardware_data:
        try:
            with open(args.hardware_data, 'r') as f:
                hardware_data = json.load(f)
            print(f"📁 Loaded hardware data from {args.hardware_data}")
        except Exception as e:
            print(f"❌ Error loading hardware data: {e}")
    
    if args.network_data:
        try:
            with open(args.network_data, 'r') as f:
                network_data = json.load(f)
            print(f"📁 Loaded network data from {args.network_data}")
        except Exception as e:
            print(f"❌ Error loading network data: {e}")
    
    print("🔄 Starting comprehensive signal analysis...")
    
    # Perform analysis
    results = analyzer.analyze_all_signals(hardware_data, network_data)
    
    # Save results
    results_file = Path(args.output_dir) / "signal_metrics_analysis.json"
    analyzer.save_results(str(results_file), results)
    
    # Create visualizations
    print("📊 Creating analysis visualizations...")
    analyzer.create_visualizations(args.output_dir, results)
    
    # Print summary
    print("\n📋 Analysis Summary:")
    print(f"Analysis Duration: {results['analysis_metadata']['analysis_duration']:.2f}s")
    print(f"Total Metrics Computed: {results['analysis_metadata']['total_metrics_computed']}")
    
    metrics = results['signal_metrics']
    
    # Quality assessment summary
    quality_metrics = metrics.get('quality_metrics', {})
    if 'overall_quality_score' in quality_metrics:
        overall_score = quality_metrics['overall_quality_score']
        print(f"\n🎯 Overall Quality Score: {overall_score:.3f}")
        
        if overall_score > 0.8:
            print("   ✅ Excellent signal quality")
        elif overall_score > 0.6:
            print("   🟡 Good signal quality")
        elif overall_score > 0.4:
            print("   🟠 Fair signal quality")
        else:
            print("   ❌ Poor signal quality")
    
    # Hardware analysis summary
    if 'hardware_metrics' in metrics:
        hw_metrics = metrics['hardware_metrics']
        print(f"\n🖥️  Hardware Analysis:")
        
        resource_util = hw_metrics.get('resource_utilization', {})
        for resource, stats in resource_util.items():
            util = stats['mean_utilization']
            status = "🔴" if util > 80 else "🟡" if util > 60 else "🟢"
            print(f"   {status} {resource.upper()}: {util:.1f}% average utilization")
    
    # Network analysis summary
    if 'network_metrics' in metrics:
        net_metrics = metrics['network_metrics']
        print(f"\n🌐 Network Analysis:")
        
        conn_analysis = net_metrics.get('connectivity_analysis', {})
        if 'overall_reliability' in conn_analysis:
            success_rate = conn_analysis['overall_reliability']['mean_success_rate']
            status = "🟢" if success_rate > 0.95 else "🟡" if success_rate > 0.85 else "🔴"
            print(f"   {status} Connectivity: {success_rate:.1%} success rate")
        
        perf_analysis = net_metrics.get('performance_analysis', {})
        if 'latency' in perf_analysis and 'overall_latency' in perf_analysis['latency']:
            avg_latency = perf_analysis['latency']['overall_latency']['mean_latency']
            status = "🟢" if avg_latency < 50 else "🟡" if avg_latency < 100 else "🔴"
            print(f"   {status} Latency: {avg_latency:.1f}ms average")
    
    # Anomaly detection summary
    anomaly_metrics = metrics.get('anomaly_metrics', {})
    total_anomalies = 0
    
    for category in ['hardware_anomalies', 'network_anomalies']:
        if category in anomaly_metrics:
            for component, anomalies in anomaly_metrics[category].items():
                if isinstance(anomalies, dict):
                    total_anomalies += sum(v for v in anomalies.values() if isinstance(v, (int, float)))
    
    if total_anomalies > 0:
        status = "🔴" if total_anomalies > 10 else "🟡" if total_anomalies > 5 else "🟠"
        print(f"\n{status} Anomalies Detected: {total_anomalies} total anomalies found")
    else:
        print(f"\n🟢 No significant anomalies detected")
    
    print(f"\n💾 Results saved to {args.output_dir}/")
    print(f"📊 Dashboard: signal_metrics_dashboard.png")
    print(f"📄 Full Analysis: signal_metrics_analysis.json")
    
    print("\n🎉 Signal metrics analysis complete!")


if __name__ == "__main__":
    main()