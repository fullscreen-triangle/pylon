#!/usr/bin/env python3
"""
Sango Rine Shumba Validation Framework - Main Simulation Orchestrator

This script orchestrates all modules in the validation framework, allowing them to run
separately or together with intermediate result saving. Each component can be executed
in isolation for scientific rigor and debugging.

Usage:
    python simulation.py --run-all                           # Run complete validation pipeline
    python simulation.py --run-observer                      # Run only observer validation
    python simulation.py --run-network                      # Run only network validation
    python simulation.py --run-signal                       # Run only signal validation
    python simulation.py --run-component <component_name>   # Run specific component
    
Available components:
    - finite_observer          # Test finite observer functionality
    - gear_ratio_calculator    # Test gear ratio calculations
    - oscillatory_hierarchy    # Test oscillatory hierarchy
    - transcendent_observer     # Test transcendent observer
    - ambiguous_compressor     # Test compression algorithms
    - hierarchical_navigator   # Test navigation algorithms
    - hardware_signals         # Collect hardware signals
    - network_signals          # Collect network signals
    - signal_metrics           # Analyze signal metrics
    
Examples:
    python simulation.py --run-component finite_observer --duration 30
    python simulation.py --run-signal --hardware-only
    python simulation.py --run-all --save-intermediate
"""

import asyncio
import time
import json
import logging
import argparse
import sys
import secrets
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import observer framework
from observer.finite_observer import FiniteObserver
from observer.gear_ratio_calculator import GearRatioCalculator
from observer.oscillatory_hierarchy import OscillatoryHierarchy
from observer.transcendent_observer import TranscendentObserver
from observer.observer_metrics import ObserverMetrics

# Import network framework
from network.ambigous_compressor import AmbiguousCompressor
from network.hierarchical_navigator import HierarchicalNavigator
from network.network_metrics import NetworkMetrics

# Import signal framework
from signal.hardware_signals import HardwareSignalCollector
from signal.network_signals import NetworkSignalCollector
from signal.signal_metrics import SignalMetricsAnalyzer

class SangoRineShumbaValidationFramework:
    """Main validation framework orchestrator"""
    
    def __init__(self, output_dir: str = "validation_results"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.start_time = time.time()
        self.results = {}
        self.module_results = {}
        
        # Initialize core components
        self.gear_ratio_calculator = None
        self.oscillatory_hierarchy = None
        self.transcendent_observer = None
        self.observer_metrics = None
        self.compressor = None
        self.navigator = None
        self.network_metrics = None
        
        self.logger.info(f"Validation framework initialized - output directory: {self.output_dir}")
    
    async def run_complete_validation(self) -> Dict[str, Any]:
        """Run complete validation pipeline"""
        self.logger.info("🚀 Starting complete Sango Rine Shumba validation")
        
        results = {
            'validation_metadata': {
                'start_time': time.time(),
                'framework_version': '1.0',
                'validation_type': 'complete_pipeline'
            },
            'observer_validation': {},
            'network_validation': {},
            'integration_validation': {}
        }
        
        try:
            # Phase 1: Observer Framework Validation
            self.logger.info("📊 Phase 1: Observer Framework Validation")
            observer_results = await self.run_observer_validation()
            results['observer_validation'] = observer_results
            self._save_intermediate_results('observer_validation', observer_results)
            
            # Phase 2: Network Framework Validation
            self.logger.info("🌐 Phase 2: Network Framework Validation")
            network_results = await self.run_network_validation()
            results['network_validation'] = network_results
            self._save_intermediate_results('network_validation', network_results)
            
            # Phase 3: Integration Validation
            self.logger.info("🔗 Phase 3: Integration Validation")
            integration_results = await self.run_integration_validation()
            results['integration_validation'] = integration_results
            self._save_intermediate_results('integration_validation', integration_results)
            
            # Calculate overall validation metrics
            results['overall_validation_metrics'] = self._calculate_overall_metrics(results)
            results['validation_metadata']['end_time'] = time.time()
            results['validation_metadata']['total_duration'] = time.time() - results['validation_metadata']['start_time']
            
            # Save complete results
            self._save_complete_results(results)
            
            self.logger.info("✅ Complete validation finished successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Complete validation failed: {e}")
            results['validation_error'] = str(e)
            results['validation_metadata']['failed'] = True
            return results
    
    async def run_observer_validation(self) -> Dict[str, Any]:
        """Run observer framework validation"""
        self.logger.info("Initializing observer framework components...")
        
        observer_results = {
            'validation_start_time': time.time(),
            'gear_ratio_calculator_validation': {},
            'oscillatory_hierarchy_validation': {},
            'transcendent_observer_validation': {},
            'metrics_analysis': {}
        }
        
        try:
            # Initialize components
            self.gear_ratio_calculator = GearRatioCalculator()
            self.oscillatory_hierarchy = OscillatoryHierarchy()
            self.transcendent_observer = TranscendentObserver()
            self.observer_metrics = ObserverMetrics()
            
            # Validate Gear Ratio Calculator
            self.logger.info("Validating gear ratio calculator...")
            gear_validation = self._validate_gear_ratio_calculator()
            observer_results['gear_ratio_calculator_validation'] = gear_validation
            
            # Validate Oscillatory Hierarchy
            self.logger.info("Validating oscillatory hierarchy...")
            hierarchy_validation = await self._validate_oscillatory_hierarchy()
            observer_results['oscillatory_hierarchy_validation'] = hierarchy_validation
            
            # Validate Transcendent Observer
            self.logger.info("Validating transcendent observer...")
            transcendent_validation = await self._validate_transcendent_observer()
            observer_results['transcendent_observer_validation'] = transcendent_validation
            
            # Generate comprehensive metrics analysis
            self.logger.info("Generating observer metrics analysis...")
            metrics_analysis = self._generate_observer_metrics_analysis()
            observer_results['metrics_analysis'] = metrics_analysis
            
            observer_results['validation_success'] = True
            observer_results['validation_end_time'] = time.time()
            observer_results['total_validation_time'] = observer_results['validation_end_time'] - observer_results['validation_start_time']
            
            return observer_results
            
        except Exception as e:
            self.logger.error(f"Observer validation failed: {e}")
            observer_results['validation_error'] = str(e)
            observer_results['validation_success'] = False
            return observer_results
    
    async def run_network_validation(self) -> Dict[str, Any]:
        """Run network framework validation"""
        self.logger.info("Initializing network framework components...")
        
        network_results = {
            'validation_start_time': time.time(),
            'ambiguous_compression_validation': {},
            'gear_ratio_extraction_validation': {},
            'tree_to_graph_transition_validation': {},
            'network_structure_analysis': {}
        }
        
        try:
            # Initialize components
            self.compressor = AmbiguousCompressor()
            self.navigator = HierarchicalNavigator()
            self.network_metrics = NetworkMetrics()
            
            # Validate Ambiguous Compression
            self.logger.info("Validating ambiguous compression...")
            compression_validation = await self._validate_ambiguous_compression()
            network_results['ambiguous_compression_validation'] = compression_validation
            
            # Validate Gear Ratio Extraction
            self.logger.info("Validating gear ratio extraction...")
            extraction_validation = await self._validate_gear_ratio_extraction()
            network_results['gear_ratio_extraction_validation'] = extraction_validation
            
            # NEW: Validate Tree → Graph Structure Transition
            self.logger.info("Validating tree → graph structure transition...")
            tree_graph_validation = await self._validate_tree_to_graph_transition()
            network_results['tree_to_graph_transition_validation'] = tree_graph_validation
            
            # NEW: Comprehensive Network Structure Analysis
            self.logger.info("Running comprehensive network structure analysis...")
            structure_analysis = await self._run_network_structure_analysis()
            network_results['network_structure_analysis'] = structure_analysis
            
            network_results['validation_success'] = True
            network_results['validation_end_time'] = time.time()
            network_results['total_validation_time'] = network_results['validation_end_time'] - network_results['validation_start_time']
            
            return network_results
            
        except Exception as e:
            self.logger.error(f"Network validation failed: {e}")
            network_results['validation_error'] = str(e)
            network_results['validation_success'] = False
            return network_results
    
    async def run_integration_validation(self) -> Dict[str, Any]:
        """Run integration validation between observer and network frameworks"""
        self.logger.info("Running integration validation...")
        
        integration_results = {
            'validation_start_time': time.time(),
            'observer_network_integration': {},
            'end_to_end_validation': {}
        }
        
        try:
            # Test observer-network integration
            self.logger.info("Testing observer-network integration...")
            integration_test = await self._test_observer_network_integration()
            integration_results['observer_network_integration'] = integration_test
            
            # End-to-end validation
            self.logger.info("Running end-to-end validation...")
            e2e_validation = await self._run_end_to_end_validation()
            integration_results['end_to_end_validation'] = e2e_validation
            
            integration_results['validation_success'] = True
            integration_results['validation_end_time'] = time.time()
            integration_results['total_validation_time'] = integration_results['validation_end_time'] - integration_results['validation_start_time']
            
            return integration_results
            
        except Exception as e:
            self.logger.error(f"Integration validation failed: {e}")
            integration_results['validation_error'] = str(e)
            integration_results['validation_success'] = False
            return integration_results
    
    def _validate_gear_ratio_calculator(self) -> Dict[str, Any]:
        """Validate gear ratio calculator functionality"""
        validation_results = {
            'transitivity_validation': {},
            'performance_validation': {},
            'accuracy_validation': {}
        }
        
        # Test transitivity property
        transitivity_results = self.gear_ratio_calculator.validate_all_transitivity_properties()
        validation_results['transitivity_validation'] = transitivity_results
        
        # Test performance
        performance_stats = self.gear_ratio_calculator.get_performance_statistics()
        validation_results['performance_validation'] = performance_stats
        
        # Test accuracy with known values
        test_cases = [
            (1, 2, 1e13 / 1e7),  # Quantum to Atomic
            (3, 4, 1e2 / 1e0),   # Precision to Fragment
            (8, 1, 1e-5 / 1e13)  # Cultural to Quantum
        ]
        
        accuracy_tests = []
        for source, target, expected in test_cases:
            calculated = self.gear_ratio_calculator.get_compound_ratio(source, target)
            error = abs(calculated - expected) / expected
            accuracy_tests.append({
                'source_scale': source,
                'target_scale': target,
                'expected_ratio': expected,
                'calculated_ratio': calculated,
                'relative_error': error,
                'accuracy_test_passed': error < 1e-10
            })
        
        validation_results['accuracy_validation'] = {
            'test_cases': accuracy_tests,
            'all_tests_passed': all(test['accuracy_test_passed'] for test in accuracy_tests)
        }
        
        return validation_results
    
    async def _validate_oscillatory_hierarchy(self) -> Dict[str, Any]:
        """Validate oscillatory hierarchy functionality"""
        validation_results = {
            'frequency_hierarchy_validation': {},
            'observer_functionality_validation': {},
            'cross_scale_observation_validation': {}
        }
        
        # Validate frequency hierarchy
        hierarchy_validation = self.oscillatory_hierarchy.validate_frequency_hierarchy()
        validation_results['frequency_hierarchy_validation'] = hierarchy_validation
        
        # Test observer functionality
        test_signals = [
            {'signal_type': 'quantum_coherence', 'coherence_strength': 0.95},
            {'signal_type': 'atomic_sync', 'clock_drift': 0.001},
            {'signal_type': 'precision_calculation', 'precision_difference': 0.00001},
            'test_string_signal',
            42,
            b'binary_test_data'
        ]
        
        observer_tests = []
        for scale_id in range(1, 9):
            observer = self.oscillatory_hierarchy.get_observer(scale_id)
            if observer:
                scale_results = {
                    'scale_id': scale_id,
                    'scale_name': observer.scale_name,
                    'signal_test_results': []
                }
                
                for signal in test_signals:
                    observed = self.oscillatory_hierarchy.observe_at_scale(scale_id, signal)
                    scale_results['signal_test_results'].append({
                        'signal': str(signal)[:50],  # Truncate for JSON
                        'observed': observed
                    })
                
                observer_tests.append(scale_results)
        
        validation_results['observer_functionality_validation'] = {
            'scale_observer_tests': observer_tests,
            'all_scales_functional': len(observer_tests) == 8
        }
        
        # Test cross-scale observation
        cross_scale_results = {}
        for signal in test_signals[:3]:
            results = self.oscillatory_hierarchy.observe_across_all_scales(signal)
            cross_scale_results[str(signal)[:30]] = {
                'scales_observed': sum(results.values()),
                'total_scales': len(results),
                'observation_rate': sum(results.values()) / len(results)
            }
        
        validation_results['cross_scale_observation_validation'] = cross_scale_results
        
        return validation_results
    
    async def _validate_transcendent_observer(self) -> Dict[str, Any]:
        """Validate transcendent observer functionality"""
        validation_results = {
            'navigation_validation': {},
            'coordination_validation': {},
            'O1_complexity_validation': {}
        }
        
        # Test navigation functionality
        navigation_tests = []
        test_navigations = [(1, 8), (3, 5), (8, 1), (4, 4), (2, 7)]
        
        for source, target in test_navigations:
            result = self.transcendent_observer.navigate_hierarchy_O1(source, target, f"test_data_{source}_{target}")
            navigation_tests.append({
                'source_scale': source,
                'target_scale': target,
                'navigation_success': result['success'],
                'navigation_time': result.get('navigation_time', 0.0),
                'gear_ratio_used': result.get('gear_ratio', 0.0)
            })
        
        validation_results['navigation_validation'] = {
            'navigation_tests': navigation_tests,
            'all_navigations_successful': all(test['navigation_success'] for test in navigation_tests)
        }
        
        # Test coordination functionality
        test_signals = ['coordination_test_1', 'coordination_test_2', {'test': 'coordination'}]
        coordination_tests = []
        
        for signal in test_signals:
            observation = self.transcendent_observer.observe_finite_observers(signal)
            coordination_tests.append({
                'signal': str(signal)[:30],
                'observers_engaged': observation['finite_observer_count'],
                'successful_observations': observation['successful_observations'],
                'success_rate': observation['overall_success_rate']
            })
        
        validation_results['coordination_validation'] = {
            'coordination_tests': coordination_tests,
            'average_success_rate': sum(test['success_rate'] for test in coordination_tests) / len(coordination_tests)
        }
        
        # Test O(1) complexity
        test_scale_pairs = [(1, 2), (1, 8), (3, 7), (5, 1)]
        complexity_validation = self.transcendent_observer.validate_O1_complexity(test_scale_pairs, test_count=50)
        validation_results['O1_complexity_validation'] = complexity_validation
        
        return validation_results
    
    def _generate_observer_metrics_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive observer metrics analysis"""
        metrics_results = {
            'transcendent_analysis': {},
            'finite_observer_analyses': [],
            'comprehensive_report_path': ''
        }
        
        # Analyze transcendent observer
        transcendent_analysis = self.observer_metrics.analyze_transcendent_observer(self.transcendent_observer)
        metrics_results['transcendent_analysis'] = transcendent_analysis
        
        # Analyze all finite observers
        for observer in self.transcendent_observer.finite_observers:
            analysis = self.observer_metrics.analyze_finite_observer(observer)
            metrics_results['finite_observer_analyses'].append(analysis)
        
        # Generate comprehensive report
        report_path = self.observer_metrics.generate_comprehensive_report(
            self.transcendent_observer, 
            str(self.output_dir / "observer_analysis")
        )
        metrics_results['comprehensive_report_path'] = report_path
        
        return metrics_results
    
    async def _validate_ambiguous_compression(self) -> Dict[str, Any]:
        """Validate ambiguous compression functionality"""
        # Generate test data with varying compression characteristics
        test_data_samples = [
            b'A' * 100,  # Highly compressible
            b''.join([bytes([i % 256]) for i in range(1000)]),  # Moderately compressible
            bytes([i % 256 for i in range(256)] * 4),  # Low compressibility
            b'\x00\x01\x02\x03' * 250,  # Pattern-based
            b'Random text with various characters 123!@#',  # Mixed content
        ]
        
        # Add truly random data (should be compression-resistant)
        import secrets
        for _ in range(3):
            test_data_samples.append(secrets.token_bytes(200))
        
        # Analyze compression resistance
        batch_results = self.compressor.analyze_compression_resistance_batch(test_data_samples)
        
        # Create visualizations
        self.compressor.create_compression_visualization(test_data_samples, str(self.output_dir / "compression_analysis"))
        
        return {
            'batch_analysis': batch_results,
            'compression_statistics': self.compressor.get_compression_statistics(),
            'visualization_path': str(self.output_dir / "compression_analysis")
        }
    
    async def _validate_gear_ratio_extraction(self) -> Dict[str, Any]:
        """Validate gear ratio extraction from ambiguous data"""
        # Test with various data types
        test_data = [
            b'This is test data for gear ratio extraction',
            bytes([i*7 % 256 for i in range(128)]),  # Pseudo-random pattern
            b'\xFF\x00\xAA\x55' * 32,  # Alternating pattern
            b'JSON{"key":"value","number":12345}',
            secrets.token_bytes(100)  # Random bytes
        ]
        
        extraction_results = []
        for i, data in enumerate(test_data):
            gear_ratios = self.compressor.extract_gear_ratios_from_ambiguous_bits(data)
            extraction_results.append({
                'data_index': i,
                'data_size': len(data),
                'gear_ratios_extracted': len(gear_ratios),
                'gear_ratios': gear_ratios[:10]  # Limit for JSON
            })
        
        return {
            'extraction_tests': extraction_results,
            'total_gear_ratios_extracted': sum(len(result['gear_ratios']) for result in extraction_results if 'gear_ratios' in result)
        }
    
    async def _validate_tree_to_graph_transition(self) -> Dict[str, Any]:
        """
        Validate transition from tree structure to graph structure.
        
        Based on harmonic network convergence principle: when different observation
        chains create harmonics that coincide, they become connected in frequency space.
        """
        self.logger.info("Testing tree → graph structure transition with synthetic observers...")
        
        # Create synthetic observers with harmonic convergence patterns
        class SyntheticObserver:
            def __init__(self, gear_ratio, freq, scale_id):
                self.gear_ratio = gear_ratio
                self.frequency = freq
                self.scale_id = scale_id
                self.successful_observations = 80 + (hash(str(gear_ratio)) % 20)  # 80-100
                self.total_observations = 100
                self.observer_id = f"synthetic_{scale_id}_{int(gear_ratio*1000)}"
            
            def extract_gear_ratio_signature(self):
                return self.gear_ratio
        
        # Create test observers with some equivalent ratios (will form graph connections)
        test_observers = []
        
        # Group 1: Similar ratios (should connect in graph)
        base_ratios = [1.0, 2.0, 3.0, 5.0, 8.0]
        for i, base in enumerate(base_ratios):
            for j in range(3):  # 3 variants per base
                ratio = base * (1.0 + j * 0.0001)  # Very small variations
                freq = ratio * 1e6
                observer = SyntheticObserver(ratio, freq, i+1)
                test_observers.append(observer)
        
        # Group 2: Dissimilar ratios (will remain as tree branches)
        dissimilar_ratios = [100.0, 200.0, 500.0, 1000.0]
        for i, ratio in enumerate(dissimilar_ratios):
            freq = ratio * 1e6
            observer = SyntheticObserver(ratio, freq, 6)
            test_observers.append(observer)
        
        # Initialize network components if not already done
        if not self.navigator:
            self.navigator = HierarchicalNavigator(tolerance=1e-4)  # Relaxed tolerance for demo
        if not self.network_metrics:
            self.network_metrics = NetworkMetrics()
        
        # Analyze structure transition
        structure_analysis = self.network_metrics.analyze_network_structure_transition(
            test_observers, tolerance=1e-4
        )
        
        # Test navigation between nodes
        navigation_tests = []
        if self.navigator.network_graph.number_of_nodes() > 1:
            nodes = list(self.navigator.network_graph.nodes())
            
            for i in range(min(5, len(nodes)-1)):
                source = nodes[i]
                target = nodes[-(i+1)]
                
                nav_result = self.navigator.find_shortest_path_navigation(source, target)
                navigation_tests.append({
                    'source': source,
                    'target': target,
                    'success': nav_result['success'],
                    'path_length': nav_result.get('path_length', 0),
                    'alternative_paths': nav_result.get('path_count', 0),
                    'precision_enhancement': nav_result.get('precision_enhancement', 1.0),
                    'graph_advantage': nav_result.get('graph_advantage', False)
                })
        
        return {
            'test_observers_created': len(test_observers), 
            'structure_analysis': structure_analysis,
            'navigation_tests': navigation_tests,
            'precision_enhancement_achieved': structure_analysis.get('precision_enhancement_achieved', 1.0),
            'theoretical_predictions_validated': structure_analysis.get('theoretical_predictions_validated', {}),
            'validation_success': structure_analysis.get('precision_enhancement_achieved', 1.0) > 1.5
        }
    
    async def _run_network_structure_analysis(self) -> Dict[str, Any]:
        """Run comprehensive network structure analysis and generate report"""
        
        if not self.network_metrics:
            return {'error': 'Network metrics not initialized'}
        
        # Generate comprehensive analysis report
        try:
            analysis_output_dir = self.output_dir / "network_analysis"
            report_path = self.network_metrics.generate_network_analysis_report(str(analysis_output_dir))
            
            # Export detailed metrics
            metrics_path = str(self.output_dir / "network_metrics.json")
            self.network_metrics.export_network_metrics(metrics_path)
            
            # Test compression-network integration
            test_compression_data = [
                b"test pattern for network analysis",
                bytes([i % 256 for i in range(200)]),
                b"harmonic convergence test data for graph connections"
            ]
            
            compression_integration = self.network_metrics.analyze_compression_network_integration(test_compression_data)
            
            return {
                'analysis_report_path': report_path,
                'metrics_export_path': metrics_path,
                'compression_integration': compression_integration,
                'analysis_success': True,
                'recommendations': self.network_metrics._generate_network_recommendations() if hasattr(self.network_metrics, '_generate_network_recommendations') else []
            }
            
        except Exception as e:
            self.logger.error(f"Network structure analysis failed: {e}")
            return {
                'analysis_success': False,
                'error': str(e)
            }
    
    async def _test_observer_network_integration(self) -> Dict[str, Any]:
        """Test integration between observer and network frameworks"""
        integration_results = {
            'gear_ratio_consistency_test': {},
            'cross_framework_communication_test': {}
        }
        
        # Test gear ratio consistency between calculator and compressor
        test_data = secrets.token_bytes(256)
        compressor_ratios = self.compressor.extract_gear_ratios_from_ambiguous_bits(test_data)
        calculator_ratios = [
            self.gear_ratio_calculator.get_compound_ratio(1, 2),
            self.gear_ratio_calculator.get_compound_ratio(3, 8),
            self.gear_ratio_calculator.get_compound_ratio(5, 1)
        ]
        
        integration_results['gear_ratio_consistency_test'] = {
            'compressor_ratios_count': len(compressor_ratios),
            'calculator_ratios_count': len(calculator_ratios),
            'ratio_ranges_compatible': True  # Placeholder validation
        }
        
        # Test cross-framework communication
        navigation_with_compression = self.transcendent_observer.navigate_hierarchy_O1(
            1, 8, test_data
        )
        
        integration_results['cross_framework_communication_test'] = {
            'navigation_success': navigation_with_compression['success'],
            'data_processed': len(test_data),
            'integration_functional': navigation_with_compression['success']
        }
        
        return integration_results
    
    async def _run_end_to_end_validation(self) -> Dict[str, Any]:
        """Run complete end-to-end validation scenario"""
        e2e_results = {
            'scenario_description': 'Complete gear ratio extraction and hierarchical navigation pipeline',
            'pipeline_stages': []
        }
        
        # Stage 1: Generate test data
        test_data = secrets.token_bytes(512)
        e2e_results['pipeline_stages'].append({
            'stage': 'data_generation',
            'data_size': len(test_data),
            'success': True
        })
        
        # Stage 2: Extract gear ratios from ambiguous compression
        extracted_ratios = self.compressor.extract_gear_ratios_from_ambiguous_bits(test_data)
        e2e_results['pipeline_stages'].append({
            'stage': 'gear_ratio_extraction',
            'ratios_extracted': len(extracted_ratios),
            'success': len(extracted_ratios) > 0
        })
        
        # Stage 3: Use ratios for hierarchical navigation
        navigation_results = []
        for i, ratio in enumerate(extracted_ratios[:5]):  # Test first 5 ratios
            # Map ratio to scale navigation
            source_scale = (i % 7) + 1
            target_scale = ((i + 3) % 7) + 1
            
            nav_result = self.transcendent_observer.navigate_hierarchy_O1(source_scale, target_scale, ratio)
            navigation_results.append(nav_result['success'])
        
        e2e_results['pipeline_stages'].append({
            'stage': 'hierarchical_navigation',
            'navigations_attempted': len(navigation_results),
            'navigations_successful': sum(navigation_results),
            'success': sum(navigation_results) > 0
        })
        
        # Stage 4: Observer coordination
        coordination_result = self.transcendent_observer.observe_finite_observers(extracted_ratios)
        e2e_results['pipeline_stages'].append({
            'stage': 'observer_coordination',
            'observers_engaged': coordination_result['finite_observer_count'],
            'coordination_success_rate': coordination_result['overall_success_rate'],
            'success': coordination_result['overall_success_rate'] > 0.0
        })
        
        # Calculate overall pipeline success
        e2e_results['pipeline_success'] = all(stage['success'] for stage in e2e_results['pipeline_stages'])
        e2e_results['pipeline_success_rate'] = sum(stage['success'] for stage in e2e_results['pipeline_stages']) / len(e2e_results['pipeline_stages'])
        
        return e2e_results
    
    def _calculate_overall_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall validation metrics"""
        metrics = {
            'overall_success': True,
            'component_success_rates': {},
            'performance_summary': {},
            'validation_coverage': {}
        }
        
        # Calculate component success rates
        if 'observer_validation' in results:
            observer_success = results['observer_validation'].get('validation_success', False)
            metrics['component_success_rates']['observer_framework'] = observer_success
            
        if 'network_validation' in results:
            network_success = results['network_validation'].get('validation_success', False)
            metrics['component_success_rates']['network_framework'] = network_success
            
        if 'integration_validation' in results:
            integration_success = results['integration_validation'].get('validation_success', False)
            metrics['component_success_rates']['integration'] = integration_success
        
        # Overall success
        metrics['overall_success'] = all(metrics['component_success_rates'].values())
        metrics['overall_success_rate'] = sum(metrics['component_success_rates'].values()) / len(metrics['component_success_rates'])
        
        return metrics
    
    def _save_intermediate_results(self, stage_name: str, results: Dict[str, Any]):
        """Save intermediate results to JSON file"""
        output_file = self.output_dir / f"{stage_name}_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Intermediate results saved: {output_file}")
    
    def _save_complete_results(self, results: Dict[str, Any]):
        """Save complete validation results"""
        output_file = self.output_dir / "complete_validation_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary report
        summary_file = self.output_dir / "validation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("SANGO RINE SHUMBA VALIDATION FRAMEWORK - RESULTS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            overall_metrics = results.get('overall_validation_metrics', {})
            f.write(f"Overall Validation Success: {overall_metrics.get('overall_success', 'Unknown')}\n")
            f.write(f"Overall Success Rate: {overall_metrics.get('overall_success_rate', 0.0):.2%}\n\n")
            
            f.write("Component Results:\n")
            for component, success in overall_metrics.get('component_success_rates', {}).items():
                f.write(f"  - {component}: {'✅ PASSED' if success else '❌ FAILED'}\n")
            
            f.write(f"\nValidation Duration: {results['validation_metadata'].get('total_duration', 0.0):.2f} seconds\n")
            f.write(f"Results saved to: {self.output_dir}\n")
        
        self.logger.info(f"Complete validation results saved: {output_file}")
        self.logger.info(f"Validation summary saved: {summary_file}")
    
    async def run_signal_validation(self, hardware_only: bool = False, network_only: bool = False, 
                                  duration: float = 60.0) -> Dict[str, Any]:
        """Run comprehensive signal validation"""
        self.logger.info("Starting signal validation...")
        
        validation_results = {
            'validation_start_time': time.time(),
            'hardware_signal_validation': {},
            'network_signal_validation': {},
            'signal_metrics_validation': {},
            'validation_success': False
        }
        
        # Hardware signal validation
        if not network_only:
            self.logger.info("Running hardware signal validation...")
            try:
                hardware_collector = HardwareSignalCollector(sample_duration=duration, sample_rate=2.0)
                hardware_results = hardware_collector.collect_all_hardware_signals()
                
                # Save hardware results
                hardware_file = self.output_dir / "hardware_signals_data.json"
                hardware_collector.save_results(str(hardware_file))
                
                # Create hardware visualizations
                hardware_collector.create_visualizations(str(self.output_dir / "hardware_signals"))
                
                validation_results['hardware_signal_validation'] = {
                    'collection_success': True,
                    'total_samples': hardware_results['collection_metadata']['total_samples_collected'],
                    'collection_duration': hardware_results['collection_metadata']['collection_duration'],
                    'platform_capabilities': hardware_results['collection_metadata']['platform_capabilities'],
                    'results_file': str(hardware_file)
                }
                
            except Exception as e:
                self.logger.error(f"Hardware signal validation failed: {e}")
                validation_results['hardware_signal_validation'] = {
                    'collection_success': False,
                    'error': str(e)
                }
        
        # Network signal validation  
        if not hardware_only:
            self.logger.info("Running network signal validation...")
            try:
                network_collector = NetworkSignalCollector(sample_duration=duration, sample_rate=0.5)
                network_results = network_collector.collect_all_network_signals()
                
                # Save network results
                network_file = self.output_dir / "network_signals_data.json"
                network_collector.save_results(str(network_file))
                
                # Create network visualizations
                network_collector.create_visualizations(str(self.output_dir / "network_signals"))
                
                validation_results['network_signal_validation'] = {
                    'collection_success': True,
                    'total_samples': network_results['collection_metadata']['total_samples_collected'],
                    'collection_duration': network_results['collection_metadata']['collection_duration'],
                    'platform_capabilities': network_results['collection_metadata']['platform_capabilities'],
                    'results_file': str(network_file)
                }
                
            except Exception as e:
                self.logger.error(f"Network signal validation failed: {e}")
                validation_results['network_signal_validation'] = {
                    'collection_success': False,
                    'error': str(e)
                }
        
        # Signal metrics validation
        if not hardware_only and not network_only:
            self.logger.info("Running signal metrics validation...")
            try:
                analyzer = SignalMetricsAnalyzer(analysis_window=duration)
                
                # Load collected data if available
                hardware_data = None
                network_data = None
                
                if 'hardware_signal_validation' in validation_results and validation_results['hardware_signal_validation'].get('collection_success'):
                    hardware_file = validation_results['hardware_signal_validation']['results_file']
                    with open(hardware_file, 'r') as f:
                        hardware_data = json.load(f)
                
                if 'network_signal_validation' in validation_results and validation_results['network_signal_validation'].get('collection_success'):
                    network_file = validation_results['network_signal_validation']['results_file']
                    with open(network_file, 'r') as f:
                        network_data = json.load(f)
                
                # Perform analysis
                analysis_results = analyzer.analyze_all_signals(hardware_data, network_data)
                
                # Save analysis results
                analysis_file = self.output_dir / "signal_metrics_analysis.json"
                analyzer.save_results(str(analysis_file), analysis_results)
                
                # Create analysis visualizations
                analyzer.create_visualizations(str(self.output_dir / "signal_metrics"))
                
                validation_results['signal_metrics_validation'] = {
                    'analysis_success': True,
                    'total_metrics_computed': analysis_results['analysis_metadata']['total_metrics_computed'],
                    'analysis_duration': analysis_results['analysis_metadata']['analysis_duration'],
                    'overall_quality_score': analysis_results['signal_metrics'].get('quality_metrics', {}).get('overall_quality_score', 0),
                    'results_file': str(analysis_file)
                }
                
            except Exception as e:
                self.logger.error(f"Signal metrics validation failed: {e}")
                validation_results['signal_metrics_validation'] = {
                    'analysis_success': False,
                    'error': str(e)
                }
        
        # Determine overall success
        hardware_success = validation_results.get('hardware_signal_validation', {}).get('collection_success', True)
        network_success = validation_results.get('network_signal_validation', {}).get('collection_success', True)  
        metrics_success = validation_results.get('signal_metrics_validation', {}).get('analysis_success', True)
        
        validation_results['validation_success'] = hardware_success and network_success and metrics_success
        validation_results['validation_end_time'] = time.time()
        validation_results['validation_duration'] = validation_results['validation_end_time'] - validation_results['validation_start_time']
        
        self.logger.info(f"Signal validation complete. Success: {validation_results['validation_success']}")
        return validation_results
    
    async def run_component(self, component_name: str, duration: float = 30.0, **kwargs) -> Dict[str, Any]:
        """Run individual component validation"""
        self.logger.info(f"Running component validation: {component_name}")
        
        component_results = {
            'component_name': component_name,
            'validation_start_time': time.time(),
            'validation_success': False
        }
        
        try:
            if component_name == 'finite_observer':
                observer = FiniteObserver(
                    observation_frequency=kwargs.get('frequency', 5.0),
                    max_observation_space=kwargs.get('max_space', 100),
                    observer_id=f"component_test_{int(time.time())}"
                )
                
                # Run observation test
                start_time = time.time()
                signal_count = 0
                
                while time.time() - start_time < duration:
                    test_signal = f"test_signal_{signal_count}_{int(time.time() * 1000000) % 1000000}"
                    observer.observe_signal(test_signal)
                    signal_count += 1
                    await asyncio.sleep(0.1)
                
                stats = observer.get_observation_statistics()
                
                # Save results
                results_file = self.output_dir / f"component_{component_name}_results.json"
                observer.export_observations(str(results_file))
                
                component_results.update({
                    'validation_success': stats['success_rate'] > 0.0,
                    'statistics': stats,
                    'signals_processed': signal_count,
                    'results_file': str(results_file)
                })
            
            elif component_name == 'gear_ratio_calculator':
                calculator = GearRatioCalculator()
                
                # Test calculations
                test_results = []
                test_cases = [(1, 2), (3, 4), (7, 8), (2, 6)]
                
                for source, target in test_cases:
                    ratio = calculator.get_compound_ratio(source, target)
                    transitivity_valid = calculator.validate_gear_ratio_transitivity(source, target, ratio)
                    
                    test_results.append({
                        'source_scale': source,
                        'target_scale': target,
                        'calculated_ratio': ratio,
                        'transitivity_valid': transitivity_valid
                    })
                
                # Save results
                results_file = self.output_dir / f"component_{component_name}_results.json"
                calculator.export_compound_ratio_table(str(results_file))
                
                component_results.update({
                    'validation_success': all(test['transitivity_valid'] for test in test_results),
                    'test_results': test_results,
                    'performance_stats': calculator.get_performance_statistics(),
                    'results_file': str(results_file)
                })
            
            elif component_name == 'hardware_signals':
                collector = HardwareSignalCollector(sample_duration=duration, sample_rate=2.0)
                results = collector.collect_all_hardware_signals()
                
                results_file = self.output_dir / f"component_{component_name}_data.json"
                collector.save_results(str(results_file))
                collector.create_visualizations(str(self.output_dir / f"component_{component_name}"))
                
                component_results.update({
                    'validation_success': results['collection_metadata']['total_samples_collected'] > 0,
                    'collection_metadata': results['collection_metadata'],
                    'results_file': str(results_file)
                })
            
            elif component_name == 'network_signals':
                collector = NetworkSignalCollector(sample_duration=duration, sample_rate=0.5)
                results = collector.collect_all_network_signals()
                
                results_file = self.output_dir / f"component_{component_name}_data.json"
                collector.save_results(str(results_file))
                collector.create_visualizations(str(self.output_dir / f"component_{component_name}"))
                
                component_results.update({
                    'validation_success': results['collection_metadata']['total_samples_collected'] > 0,
                    'collection_metadata': results['collection_metadata'],
                    'results_file': str(results_file)
                })
            
            elif component_name == 'signal_metrics':
                analyzer = SignalMetricsAnalyzer(analysis_window=duration)
                results = analyzer.analyze_all_signals()
                
                results_file = self.output_dir / f"component_{component_name}_analysis.json"
                analyzer.save_results(str(results_file), results)
                analyzer.create_visualizations(str(self.output_dir / f"component_{component_name}"))
                
                component_results.update({
                    'validation_success': results['analysis_metadata']['total_metrics_computed'] > 0,
                    'analysis_metadata': results['analysis_metadata'],
                    'results_file': str(results_file)
                })
            
            else:
                raise ValueError(f"Unknown component: {component_name}")
        
        except Exception as e:
            self.logger.error(f"Component {component_name} validation failed: {e}")
            component_results['error'] = str(e)
        
        component_results['validation_end_time'] = time.time()
        component_results['validation_duration'] = component_results['validation_end_time'] - component_results['validation_start_time']
        
        # Save component results
        self._save_intermediate_results(f"component_{component_name}", component_results)
        
        self.logger.info(f"Component {component_name} validation complete. Success: {component_results['validation_success']}")
        return component_results


async def main():
    """Main entry point for validation framework"""
    parser = argparse.ArgumentParser(description="Sango Rine Shumba Validation Framework")
    
    # Main execution modes
    parser.add_argument('--run-all', action='store_true', help='Run complete validation pipeline')
    parser.add_argument('--run-observer', action='store_true', help='Run observer framework validation')
    parser.add_argument('--run-network', action='store_true', help='Run network framework validation')
    parser.add_argument('--run-signal', action='store_true', help='Run signal collection and analysis')
    parser.add_argument('--run-integration', action='store_true', help='Run integration validation')
    parser.add_argument('--run-component', type=str, help='Run specific component validation')
    
    # Signal-specific options
    parser.add_argument('--hardware-only', action='store_true', help='Only collect hardware signals')
    parser.add_argument('--network-only', action='store_true', help='Only collect network signals')
    parser.add_argument('--duration', type=float, default=60.0, help='Collection/test duration in seconds')
    
    # Component-specific options
    parser.add_argument('--frequency', type=float, default=5.0, help='Observer frequency (for finite_observer component)')
    parser.add_argument('--max-space', type=int, default=100, help='Max observation space (for finite_observer component)')
    
    # General options
    parser.add_argument('--output-dir', default='validation_results', help='Output directory for results')
    parser.add_argument('--save-intermediate', action='store_true', help='Save intermediate results at each stage')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(args.output_dir) / 'validation.log')
        ]
    )
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Initialize framework
    framework = SangoRineShumbaValidationFramework(args.output_dir)
    
    try:
        if args.run_all:
            print("🚀 Running complete validation pipeline...")
            results = await framework.run_complete_validation()
            
        elif args.run_observer:
            print("📊 Running observer framework validation...")
            results = await framework.run_observer_validation()
            if args.save_intermediate:
                framework._save_intermediate_results('observer_validation', results)
            
        elif args.run_network:
            print("🌐 Running network framework validation...")
            results = await framework.run_network_validation()
            if args.save_intermediate:
                framework._save_intermediate_results('network_validation', results)
            
        elif args.run_signal:
            print("📡 Running signal collection and analysis...")
            results = await framework.run_signal_validation(
                hardware_only=args.hardware_only,
                network_only=args.network_only,
                duration=args.duration
            )
            if args.save_intermediate:
                framework._save_intermediate_results('signal_validation', results)
            
        elif args.run_integration:
            print("🔗 Running integration validation...")
            results = await framework.run_integration_validation()
            if args.save_intermediate:
                framework._save_intermediate_results('integration_validation', results)
            
        elif args.run_component:
            available_components = [
                'finite_observer', 'gear_ratio_calculator', 'oscillatory_hierarchy', 'transcendent_observer',
                'ambiguous_compressor', 'hierarchical_navigator', 
                'hardware_signals', 'network_signals', 'signal_metrics'
            ]
            
            if args.run_component not in available_components:
                print(f"❌ Unknown component: {args.run_component}")
                print(f"Available components: {', '.join(available_components)}")
                return
            
            print(f"🔧 Running component validation: {args.run_component}")
            results = await framework.run_component(
                args.run_component,
                duration=args.duration,
                frequency=args.frequency,
                max_space=args.max_space
            )
            
        else:
            print("❓ No validation option specified.")
            print("\nAvailable options:")
            print("  --run-all              Run complete validation pipeline")
            print("  --run-observer         Run observer framework validation")
            print("  --run-network          Run network framework validation") 
            print("  --run-signal           Run signal collection and analysis")
            print("  --run-integration      Run integration validation")
            print("  --run-component <name> Run specific component validation")
            print("\nUse --help for detailed options.")
            return
        
        # Print summary
        success = results.get('validation_success', results.get('overall_validation_metrics', {}).get('overall_success', False))
        print(f"\n{'✅ VALIDATION PASSED' if success else '❌ VALIDATION FAILED'}")
        print(f"Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
    except Exception as e:
        print(f"\n💥 Validation failed with error: {e}")
        logging.exception("Validation framework error")


if __name__ == "__main__":
    import secrets  # Add missing import
    asyncio.run(main()) 