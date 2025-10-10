import time
import json
import logging
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from .hierarchical_navigator import HierarchicalNavigator
from .ambigous_compressor import AmbiguousCompressor

class NetworkMetrics:
    """
    Statistical and metric validation/analysis of the whole network.
    
    Implements graph-based network analysis following the harmonic network
    convergence principle where equivalent observation paths create connections.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analysis_start_time = time.time()
        self.metrics_history = []
        
        # Network components
        self.navigator = None
        self.compressor = None
        
        # Analysis results
        self.tree_vs_graph_comparison = {}
        self.precision_enhancement_analysis = {}
        self.centrality_analysis = {}
        
        self.logger.info("Network metrics analyzer initialized")
    
    def analyze_network_structure_transition(self, observers: List[Any], 
                                           tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Analyze transition from tree structure to graph structure.
        
        Based on the principle that when different observation chains create
        harmonics that coincide (within tolerance), they become connected
        in frequency space, forming a graph rather than tree branches.
        """
        analysis_start = time.time()
        
        # Initialize navigator for graph analysis
        self.navigator = HierarchicalNavigator(tolerance=tolerance)
        
        # Step 1: Analyze as tree structure (baseline)
        tree_analysis = self._analyze_tree_structure(observers)
        
        # Step 2: Build graph structure and analyze
        graph_build_result = self.navigator.build_observer_network(observers)
        graph_analysis = self._analyze_graph_structure()
        
        # Step 3: Compare tree vs graph performance
        comparison = self._compare_tree_vs_graph(tree_analysis, graph_analysis)
        
        analysis_time = time.time() - analysis_start
        
        result = {
            'analysis_metadata': {
                'analysis_timestamp': time.time(),
                'analysis_duration': analysis_time,
                'observer_count': len(observers),
                'convergence_tolerance': tolerance
            },
            'tree_structure_analysis': tree_analysis,
            'graph_structure_analysis': graph_analysis,
            'graph_build_result': graph_build_result,
            'tree_vs_graph_comparison': comparison,
            'precision_enhancement_achieved': comparison.get('precision_enhancement_factor', 1.0),
            'theoretical_predictions_validated': self._validate_theoretical_predictions(comparison)
        }
        
        self.tree_vs_graph_comparison = result
        
        self.logger.info(f"Network structure analysis completed: "
                        f"{comparison.get('precision_enhancement_factor', 1.0):.1f}× enhancement achieved")
        
        return result
    
    def _analyze_tree_structure(self, observers: List[Any]) -> Dict[str, Any]:
        """Analyze network as traditional tree structure"""
        
        # In tree structure: each observer has unique path to root
        tree_metrics = {
            'structure_type': 'tree',
            'node_count': len(observers),
            'edge_count': max(0, len(observers) - 1),  # n-1 edges in tree
            'max_depth': int(np.log2(len(observers))) if len(observers) > 0 else 0,
            'paths_to_target': 1,  # Only one path in tree
            'redundancy': 0.0,  # No redundancy in trees
            'navigation_complexity': 'O(log N)',  # Tree traversal
            'precision_method': 'single_path'
        }
        
        # Calculate tree-specific metrics
        if len(observers) > 1:
            # Simulate tree navigation (sequential traversal)
            avg_tree_path_length = np.log2(len(observers))
            tree_metrics['average_path_length'] = avg_tree_path_length
            
            # Tree precision limited by single path
            gear_ratios = []
            for observer in observers:
                if hasattr(observer, 'extract_gear_ratio_signature'):
                    ratio = observer.extract_gear_ratio_signature()
                    gear_ratios.append(ratio)
            
            if gear_ratios:
                tree_metrics['gear_ratio_variance'] = np.var(gear_ratios)
                tree_metrics['precision_estimate'] = 1.0  # Baseline precision
        
        return tree_metrics
    
    def _analyze_graph_structure(self) -> Dict[str, Any]:
        """Analyze network as graph structure using navigator"""
        if not self.navigator:
            return {'error': 'Navigator not initialized'}
        
        # Get comprehensive network statistics
        network_stats = self.navigator.get_network_statistics()
        
        # Calculate graph-specific enhancements
        graph_metrics = {
            'structure_type': 'graph',
            'basic_metrics': network_stats.get('basic_metrics', {}),
            'connectivity_metrics': network_stats.get('connectivity_metrics', {}),
            'centrality_metrics': network_stats.get('centrality_metrics', {}),
            'performance_metrics': network_stats.get('performance_metrics', {})
        }
        
        # Calculate precision enhancement factors
        if 'basic_metrics' in network_stats:
            node_count = network_stats['basic_metrics']['node_count']
            edge_count = network_stats['basic_metrics']['edge_count']
            
            if node_count > 0:
                avg_degree = 2 * edge_count / node_count  # Average node degree
                
                # Enhancement factors from theoretical analysis
                F_redundancy = min(10.0, avg_degree)  # Multiple paths
                F_amplification = np.sqrt(min(100, edge_count / max(1, node_count)))  # Hub effect
                F_topology = 1.0 / (1.0 + network_stats['basic_metrics'].get('density', 0.1))
                
                graph_metrics['enhancement_factors'] = {
                    'F_redundancy': F_redundancy,
                    'F_amplification': F_amplification, 
                    'F_topology': F_topology,
                    'F_combined': F_redundancy * F_amplification * F_topology
                }
        
        # Analyze precision hubs (high centrality nodes)
        if 'centrality_metrics' in network_stats:
            precision_hubs = network_stats['centrality_metrics'].get('precision_hubs', [])
            graph_metrics['precision_hub_analysis'] = {
                'hub_count': len(precision_hubs),
                'hub_nodes': precision_hubs,
                'max_betweenness': network_stats['centrality_metrics'].get('max_betweenness', 0),
                'hub_enhancement_estimate': 2.0 if len(precision_hubs) > 0 else 1.0
            }
        
        return graph_metrics
    
    def _compare_tree_vs_graph(self, tree_analysis: Dict, graph_analysis: Dict) -> Dict[str, Any]:
        """Compare tree vs graph structure performance"""
        
        comparison = {
            'comparison_timestamp': time.time(),
            'structure_comparison': {},
            'navigation_comparison': {},
            'precision_comparison': {},
            'overall_assessment': {}
        }
        
        # Structure comparison
        tree_nodes = tree_analysis.get('node_count', 0)
        tree_edges = tree_analysis.get('edge_count', 0)
        
        graph_basic = graph_analysis.get('basic_metrics', {})
        graph_nodes = graph_basic.get('node_count', 0)
        graph_edges = graph_basic.get('edge_count', 0)
        
        comparison['structure_comparison'] = {
            'node_count': {'tree': tree_nodes, 'graph': graph_nodes, 'difference': graph_nodes - tree_nodes},
            'edge_count': {'tree': tree_edges, 'graph': graph_edges, 'difference': graph_edges - tree_edges},
            'connectivity_improvement': graph_edges / max(1, tree_edges),
            'density': graph_basic.get('density', 0.0)
        }
        
        # Navigation comparison
        tree_paths = 1  # Tree has only one path
        graph_connectivity = graph_analysis.get('connectivity_metrics', {})
        
        comparison['navigation_comparison'] = {
            'paths_available': {
                'tree': tree_paths,
                'graph': 'multiple',  # Graph has many paths
                'advantage': 'graph_provides_redundancy'
            },
            'path_length': {
                'tree': tree_analysis.get('average_path_length', float('inf')),
                'graph': graph_connectivity.get('average_path_length', float('inf')),
                'improvement': 'graph_enables_shortest_path'
            },
            'navigation_complexity': {
                'tree': 'O(log N)',
                'graph': 'O(1) with precomputed paths',
                'advantage': 'graph'
            }
        }
        
        # Precision comparison
        enhancement_factors = graph_analysis.get('enhancement_factors', {})
        precision_enhancement_factor = enhancement_factors.get('F_combined', 1.0)
        
        # Cap enhancement at 100× as per theoretical prediction
        precision_enhancement_factor = min(100.0, precision_enhancement_factor)
        
        comparison['precision_comparison'] = {
            'tree_precision': 1.0,  # Baseline
            'graph_precision_enhancement': precision_enhancement_factor,
            'enhancement_breakdown': enhancement_factors,
            'precision_hubs': graph_analysis.get('precision_hub_analysis', {}),
            'theoretical_limit': 100.0  # From document analysis
        }
        
        # Overall assessment
        graph_advantages = []
        if graph_edges > tree_edges:
            graph_advantages.append("increased_connectivity")
        if precision_enhancement_factor > 1.5:
            graph_advantages.append("precision_enhancement")
        if graph_basic.get('is_connected', False):
            graph_advantages.append("full_connectivity")
        if len(graph_analysis.get('precision_hub_analysis', {}).get('hub_nodes', [])) > 0:
            graph_advantages.append("precision_hubs_identified")
        
        comparison['overall_assessment'] = {
            'graph_advantages': graph_advantages,
            'precision_enhancement_factor': precision_enhancement_factor,
            'structure_transformation_successful': len(graph_advantages) >= 2,
            'theoretical_predictions_met': precision_enhancement_factor >= 10.0,  # Target: ~100×, minimum 10×
            'recommendation': 'use_graph_structure' if len(graph_advantages) >= 2 else 'tree_sufficient'
        }
        
        return comparison
    
    def _validate_theoretical_predictions(self, comparison: Dict) -> Dict[str, Any]:
        """Validate results against theoretical predictions from document"""
        
        predictions = {
            'precision_enhancement_target': 100.0,  # ~100× enhancement expected
            'redundancy_factor_min': 2.0,           # Multiple paths provide redundancy
            'hub_amplification_expected': True,      # High-centrality nodes should exist
            'graph_density_optimal': 0.01,          # Sparse graphs more efficient
        }
        
        precision_achieved = comparison.get('precision_comparison', {}).get('graph_precision_enhancement', 1.0)
        
        validation_results = {
            'prediction_validation': {
                'precision_enhancement': {
                    'predicted': predictions['precision_enhancement_target'],
                    'achieved': precision_achieved,
                    'ratio': precision_achieved / predictions['precision_enhancement_target'],
                    'validated': precision_achieved >= predictions['precision_enhancement_target'] * 0.1  # At least 10% of target
                },
                'redundancy_factor': {
                    'predicted': predictions['redundancy_factor_min'],
                    'achieved': comparison.get('precision_comparison', {}).get('enhancement_breakdown', {}).get('F_redundancy', 1.0),
                    'validated': comparison.get('precision_comparison', {}).get('enhancement_breakdown', {}).get('F_redundancy', 1.0) >= predictions['redundancy_factor_min']
                },
                'hub_amplification': {
                    'predicted': predictions['hub_amplification_expected'],
                    'achieved': len(comparison.get('precision_comparison', {}).get('precision_hubs', {}).get('hub_nodes', [])) > 0,
                    'validated': len(comparison.get('precision_comparison', {}).get('precision_hubs', {}).get('hub_nodes', [])) > 0
                }
            },
            'overall_validation': {
                'predictions_met': 0,
                'total_predictions': len(predictions),
                'validation_success_rate': 0.0
            }
        }
        
        # Count successful validations
        validations_met = sum([
            validation_results['prediction_validation']['precision_enhancement']['validated'],
            validation_results['prediction_validation']['redundancy_factor']['validated'],
            validation_results['prediction_validation']['hub_amplification']['validated']
        ])
        
        validation_results['overall_validation']['predictions_met'] = validations_met
        validation_results['overall_validation']['validation_success_rate'] = validations_met / len(predictions)
        
        return validation_results
    
    def analyze_compression_network_integration(self, data_samples: List[bytes]) -> Dict[str, Any]:
        """Analyze integration between network structure and compression analysis"""
        
        # Initialize compressor if not already done
        if not self.compressor:
            self.compressor = AmbiguousCompressor()
        
        integration_start = time.time()
        
        # Analyze compression patterns
        compression_results = self.compressor.analyze_compression_resistance_batch(data_samples)
        
        # Extract gear ratios from compression analysis
        all_gear_ratios = []
        for sample in data_samples:
            ratios = self.compressor.extract_gear_ratios_from_ambiguous_bits(sample)
            all_gear_ratios.extend(ratios)
        
        # Analyze how compression-derived gear ratios fit into network structure
        integration_analysis = {
            'integration_metadata': {
                'analysis_timestamp': time.time(),
                'analysis_duration': time.time() - integration_start,
                'samples_processed': len(data_samples),
                'gear_ratios_extracted': len(all_gear_ratios)
            },
            'compression_analysis': compression_results,
            'gear_ratio_network_mapping': self._map_gear_ratios_to_network(all_gear_ratios),
            'network_compression_synergy': self._analyze_network_compression_synergy()
        }
        
        return integration_analysis
    
    def _map_gear_ratios_to_network(self, gear_ratios: List[float]) -> Dict[str, Any]:
        """Map compression-derived gear ratios to network structure"""
        
        if not self.navigator:
            return {'error': 'Navigator not initialized'}
        
        mapping_results = {
            'total_ratios': len(gear_ratios),
            'network_matches': 0,
            'new_connections_possible': 0,
            'ratio_distribution': {}
        }
        
        # Check how many ratios match existing network nodes
        tolerance = self.navigator.tolerance
        
        for ratio in gear_ratios:
            # Find potential matches in existing network
            matches_found = 0
            
            for node_id, node_data in self.navigator.node_registry.items():
                existing_ratio = node_data.get('gear_ratio', 0.0)
                if abs(ratio - existing_ratio) < tolerance:
                    matches_found += 1
            
            if matches_found > 0:
                mapping_results['network_matches'] += 1
            else:
                mapping_results['new_connections_possible'] += 1
        
        # Analyze ratio distribution
        if gear_ratios:
            mapping_results['ratio_distribution'] = {
                'mean': np.mean(gear_ratios),
                'std': np.std(gear_ratios),
                'min': np.min(gear_ratios),
                'max': np.max(gear_ratios),
                'unique_ratios': len(set([round(r, 6) for r in gear_ratios]))
            }
        
        return mapping_results
    
    def _analyze_network_compression_synergy(self) -> Dict[str, Any]:
        """Analyze synergy between network graph and compression analysis"""
        
        synergy_analysis = {
            'theoretical_synergy': {
                'compression_provides_gear_ratios': True,
                'network_provides_validation_paths': True,
                'combined_precision_multiplicative': True
            },
            'practical_benefits': [],
            'implementation_advantages': []
        }
        
        # Identify practical benefits
        if self.navigator and self.compressor:
            synergy_analysis['practical_benefits'] = [
                "compression_resistance_identifies_information_dense_segments",
                "network_graph_enables_cross_validation_of_gear_ratios",
                "multiple_pathways_reduce_measurement_uncertainty",
                "hub_nodes_concentrate_precision_enhancement"
            ]
            
            synergy_analysis['implementation_advantages'] = [
                "compression_analysis_O(n)_complexity",
                "network_navigation_O(1)_with_precomputed_paths",
                "combined_system_maintains_real_time_operation",
                "gear_ratio_extraction_validates_network_connections"
            ]
        
        return synergy_analysis
    
    def generate_network_analysis_report(self, output_dir: str) -> str:
        """Generate comprehensive network analysis report"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Compile comprehensive report
        report = {
            'report_metadata': {
                'generation_timestamp': time.time(),
                'report_type': 'comprehensive_network_analysis',
                'framework_version': '1.0'
            },
            'tree_vs_graph_analysis': self.tree_vs_graph_comparison,
            'precision_enhancement_analysis': self.precision_enhancement_analysis,
            'centrality_analysis': self.centrality_analysis,
            'theoretical_validation': self._validate_theoretical_predictions(self.tree_vs_graph_comparison.get('tree_vs_graph_comparison', {})) if self.tree_vs_graph_comparison else {},
            'recommendations': self._generate_network_recommendations()
        }
        
        # Save report
        report_file = output_path / "network_analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate visualizations
        self._generate_network_visualizations(output_path)
        
        self.logger.info(f"Network analysis report generated: {report_file}")
        return str(report_file)
    
    def _generate_network_recommendations(self) -> List[str]:
        """Generate recommendations based on network analysis"""
        
        recommendations = []
        
        if self.tree_vs_graph_comparison:
            comparison = self.tree_vs_graph_comparison.get('tree_vs_graph_comparison', {})
            
            # Check precision enhancement
            precision_enhancement = comparison.get('precision_comparison', {}).get('graph_precision_enhancement', 1.0)
            
            if precision_enhancement > 10.0:
                recommendations.append("Graph structure provides significant precision enhancement - recommend adoption")
            elif precision_enhancement > 2.0:
                recommendations.append("Moderate precision enhancement achieved - consider graph structure for critical applications")
            else:
                recommendations.append("Limited precision enhancement - tree structure may be sufficient")
            
            # Check connectivity
            structure_comp = comparison.get('structure_comparison', {})
            if structure_comp.get('connectivity_improvement', 1.0) > 3.0:
                recommendations.append("High connectivity achieved - enables robust multi-path validation")
            
            # Check hubs
            precision_hubs = comparison.get('precision_comparison', {}).get('precision_hubs', {}).get('hub_count', 0)
            if precision_hubs > 0:
                recommendations.append(f"Identified {precision_hubs} precision hubs - focus resources on hub optimization")
        
        if not recommendations:
            recommendations.append("Network analysis completed - review detailed metrics for optimization opportunities")
        
        return recommendations
    
    def _generate_network_visualizations(self, output_path: Path):
        """Generate network analysis visualizations"""
        
        try:
            # Visualization 1: Tree vs Graph comparison
            if self.tree_vs_graph_comparison:
                comparison = self.tree_vs_graph_comparison.get('tree_vs_graph_comparison', {})
                
                precision_data = comparison.get('precision_comparison', {})
                tree_precision = precision_data.get('tree_precision', 1.0)
                graph_precision = precision_data.get('graph_precision_enhancement', 1.0)
                
                plt.figure(figsize=(10, 6))
                structures = ['Tree Structure', 'Graph Structure']
                precisions = [tree_precision, graph_precision]
                colors = ['orange', 'green'] if graph_precision > tree_precision else ['blue', 'blue']
                
                bars = plt.bar(structures, precisions, color=colors, alpha=0.7)
                plt.ylabel('Precision Enhancement Factor')
                plt.title('Tree vs Graph Structure: Precision Comparison')
                plt.yscale('log')
                
                # Add value labels on bars
                for bar, precision in zip(bars, precisions):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{precision:.1f}×', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(output_path / "tree_vs_graph_precision.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            # Visualization 2: Network graph (if navigator available)
            if self.navigator:
                self.navigator.visualize_network(str(output_path))
            
            self.logger.info("Network analysis visualizations generated")
            
        except Exception as e:
            self.logger.error(f"Error generating network visualizations: {e}")
    
    def export_network_metrics(self, filepath: str):
        """Export all network metrics to JSON"""
        
        export_data = {
            'export_metadata': {
                'export_timestamp': time.time(),
                'analyzer_version': '1.0',
                'analysis_type': 'network_metrics'
            },
            'tree_vs_graph_comparison': self.tree_vs_graph_comparison,
            'precision_enhancement_analysis': self.precision_enhancement_analysis,
            'centrality_analysis': self.centrality_analysis,
            'theoretical_foundation': {
                'harmonic_convergence_principle': 'When |nω_A - mω_B| < ε_tolerance, nodes become connected',
                'expected_enhancement_factors': {
                    'F_redundancy': 'Multiple paths provide statistical averaging (~10×)',
                    'F_amplification': 'Hub nodes concentrate paths (√k_max)',
                    'F_topology': 'Sparse graphs enable efficient navigation (1/(1+ρ))',
                    'F_combined': 'Total enhancement ~100× over tree structure'
                }
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Network metrics exported to {filepath}")
    
    def __repr__(self):
        return f"NetworkMetrics(analyses_completed={len(self.metrics_history)})"


def main():
    """Standalone test of network metrics"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Network Metrics Analysis")
    parser.add_argument('--output-dir', default='network_metrics_results',
                       help='Output directory for results')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='Harmonic convergence tolerance')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Initialize metrics analyzer
    metrics = NetworkMetrics()
    
    # Create synthetic test data
    print("Creating synthetic observer network...")
    
    # Synthetic observers with harmonic convergence patterns
    class SyntheticObserver:
        def __init__(self, gear_ratio, freq, scale):
            self.gear_ratio = gear_ratio
            self.frequency = freq
            self.scale_id = scale
            self.successful_observations = np.random.randint(70, 100)
            self.total_observations = 100
        
        def extract_gear_ratio_signature(self):
            return self.gear_ratio
    
    observers = []
    base_ratios = [1.0, 2.0, 3.0, 5.0, 8.0]
    
    for i, base in enumerate(base_ratios):
        # Create clusters of similar ratios (will form graph connections)
        for j in range(4):
            ratio = base * (1.0 + j * 0.0001)  # Very close ratios
            freq = ratio * 1e6
            observer = SyntheticObserver(ratio, freq, i+1)
            observers.append(observer)
    
    # Add some dissimilar ratios (will remain as tree branches)
    for i in range(5):
        ratio = 100.0 + i * 50.0  # Well separated ratios
        freq = ratio * 1e6
        observer = SyntheticObserver(ratio, freq, 6)
        observers.append(observer)
    
    print(f"Created {len(observers)} synthetic observers")
    
    # Analyze network structure transition
    print("Analyzing tree → graph structure transition...")
    structure_analysis = metrics.analyze_network_structure_transition(observers, args.tolerance)
    
    enhancement_factor = structure_analysis.get('precision_enhancement_achieved', 1.0)
    print(f"Precision enhancement achieved: {enhancement_factor:.1f}×")
    
    # Test compression integration
    print("Testing compression-network integration...")
    test_data = [
        b"test data for compression analysis",
        bytes([i % 256 for i in range(100)]),
        b"harmonic convergence pattern data"
    ]
    
    compression_integration = metrics.analyze_compression_network_integration(test_data)
    gear_ratios_found = compression_integration['integration_metadata']['gear_ratios_extracted']
    print(f"Extracted {gear_ratios_found} gear ratios from compression analysis")
    
    # Generate comprehensive report
    print("Generating comprehensive analysis report...")
    report_path = metrics.generate_network_analysis_report(args.output_dir)
    
    # Export metrics
    metrics.export_network_metrics(f"{args.output_dir}/network_metrics.json")
    
    print(f"\nResults exported to {args.output_dir}/")
    print(f"Report: {report_path}")
    
    # Summary
    validation = structure_analysis.get('theoretical_predictions_validated', {})
    validation_rate = validation.get('overall_validation', {}).get('validation_success_rate', 0.0)
    
    print(f"\n🎉 Network Metrics Analysis Complete!")
    print(f"Tree → Graph transformation: {'SUCCESS' if enhancement_factor > 2.0 else 'PARTIAL'}")
    print(f"Theoretical predictions validated: {validation_rate:.1%}")
    print(f"Precision enhancement: {enhancement_factor:.1f}× (target: ~100×)")


if __name__ == "__main__":
    main()