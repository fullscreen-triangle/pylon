#!/usr/bin/env python3
"""
Hierarchical Navigator: Graph-Based Network Navigation
Implements transition from tree to graph structure for enhanced precision.

Based on Harmonic Network Convergence Principle:
When different gear ratio paths produce similar results (within tolerance),
they become connected in the network graph, enabling multi-path validation.
"""

import time
import json
import logging
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from collections import defaultdict
import heapq

class HierarchicalNavigator:
    """
    Graph-based navigation system for Sango Rine Shumba network.
    
    Transforms traditional tree-based observer hierarchies into random graphs
    by identifying equivalent nodes and creating cross-connections.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        self.logger = logging.getLogger(__name__)
        self.tolerance = tolerance  # Harmonic convergence tolerance
        
        # Graph structure
        self.network_graph = nx.Graph()
        self.node_registry = {}  # node_id -> node_data
        self.equivalence_groups = defaultdict(set)  # Similar nodes grouped
        
        # Navigation statistics
        self.navigation_count = 0
        self.path_cache = {}  # (source, target) -> shortest_path
        self.precision_enhancements = []
        self.start_time = time.time()
        
        self.logger.info(f"Hierarchical navigator initialized with tolerance {tolerance}")
    
    def add_observer_node(self, node_id: str, observer_data: Dict[str, Any]) -> str:
        """Add an observer node to the network graph"""
        # Extract key properties for equivalence checking
        gear_ratio = observer_data.get('gear_ratio', 0.0)
        frequency = observer_data.get('frequency', 0.0)
        scale_id = observer_data.get('scale_id', 0)
        
        node_data = {
            'node_id': node_id,
            'gear_ratio': gear_ratio,
            'frequency': frequency,
            'scale_id': scale_id,
            'observer_type': observer_data.get('observer_type', 'finite'),
            'precision': observer_data.get('precision', 1e-12),
            'creation_time': time.time()
        }
        
        # Add to graph
        self.network_graph.add_node(node_id, **node_data)
        self.node_registry[node_id] = node_data
        
        # Check for equivalent nodes and create connections
        self._create_equivalence_connections(node_id, node_data)
        
        self.logger.debug(f"Added observer node {node_id} with gear_ratio={gear_ratio:.6f}")
        return node_id
    
    def _create_equivalence_connections(self, new_node_id: str, new_node_data: Dict):
        """Create graph connections based on harmonic convergence principle"""
        new_gear_ratio = new_node_data['gear_ratio']
        new_frequency = new_node_data['frequency']
        
        connections_created = 0
        
        for existing_id, existing_data in self.node_registry.items():
            if existing_id == new_node_id:
                continue
                
            # Check gear ratio equivalence
            gear_diff = abs(new_gear_ratio - existing_data['gear_ratio'])
            freq_diff = abs(new_frequency - existing_data['frequency'])
            
            # Harmonic convergence condition: |nω_A - mω_B| < ε_tolerance
            if gear_diff < self.tolerance or freq_diff < self.tolerance:
                # Calculate connection weight (inverse of difference)
                weight = 1.0 / (gear_diff + freq_diff + 1e-12)
                
                # Add edge
                self.network_graph.add_edge(new_node_id, existing_id, weight=weight)
                
                # Group equivalent nodes
                equiv_key = round(new_gear_ratio / self.tolerance) * self.tolerance
                self.equivalence_groups[equiv_key].add(new_node_id)
                self.equivalence_groups[equiv_key].add(existing_id)
                
                connections_created += 1
                
        self.logger.debug(f"Created {connections_created} equivalence connections for {new_node_id}")
    
    def find_shortest_path_navigation(self, source_node: str, target_node: str) -> Dict[str, Any]:
        """Find shortest path between nodes using graph structure"""
        cache_key = (source_node, target_node)
        
        # Check cache first
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        start_time = time.time()
        
        try:
            # Use NetworkX shortest path with weights
            path = nx.shortest_path(
                self.network_graph, 
                source=source_node, 
                target=target_node,
                weight='weight'
            )
            
            # Calculate path metrics
            path_length = len(path) - 1
            total_weight = 0.0
            
            for i in range(len(path) - 1):
                edge_data = self.network_graph[path[i]][path[i+1]]
                total_weight += edge_data.get('weight', 1.0)
            
            # Multi-path validation (find alternative paths)
            try:
                all_paths = list(nx.all_simple_paths(
                    self.network_graph, 
                    source=source_node, 
                    target=target_node,
                    cutoff=path_length + 2  # Allow slightly longer paths
                ))[:10]  # Limit to 10 paths for performance
            except:
                all_paths = [path]
            
            navigation_time = time.time() - start_time
            
            result = {
                'success': True,
                'shortest_path': path,
                'path_length': path_length,
                'total_weight': total_weight,
                'alternative_paths': all_paths,
                'path_count': len(all_paths),
                'navigation_time': navigation_time,
                'precision_enhancement': self._calculate_path_precision_enhancement(all_paths),
                'graph_advantage': len(all_paths) > 1  # Multiple paths = graph advantage
            }
            
            # Cache result
            self.path_cache[cache_key] = result
            self.navigation_count += 1
            
            self.logger.debug(f"Navigation {source_node} → {target_node}: "
                            f"{len(all_paths)} paths, enhancement {result['precision_enhancement']:.2f}×")
            
            return result
            
        except nx.NetworkXNoPath:
            self.logger.warning(f"No path found between {source_node} and {target_node}")
            return {
                'success': False,
                'error': 'No path found',
                'navigation_time': time.time() - start_time
            }
        except Exception as e:
            self.logger.error(f"Navigation error: {e}")
            return {
                'success': False,
                'error': str(e),
                'navigation_time': time.time() - start_time
            }
    
    def _calculate_path_precision_enhancement(self, paths: List[List[str]]) -> float:
        """Calculate precision enhancement from multi-path validation"""
        if len(paths) <= 1:
            return 1.0
        
        # Enhancement factors from document:
        # F_redundancy = average_node_degree (multiple paths)
        # F_amplification = sqrt(k_max) (hub amplification)  
        # F_topology = 1/(1+ρ) (graph density)
        
        path_count = len(paths)
        
        # Redundancy factor (statistical averaging)
        F_redundancy = min(10.0, path_count)  # Cap at 10× for realistic enhancement
        
        # Hub amplification (nodes with high connectivity)
        node_degrees = []
        for path in paths:
            for node in path:
                degree = self.network_graph.degree(node)
                node_degrees.append(degree)
        
        if node_degrees:
            max_degree = max(node_degrees)
            F_amplification = min(10.0, np.sqrt(max_degree))  # Cap at 10×
        else:
            F_amplification = 1.0
        
        # Topology factor (sparse graphs are more efficient)
        graph_density = nx.density(self.network_graph)
        F_topology = 1.0 / (1.0 + graph_density)
        
        # Combined enhancement (from document: ~100× typical)
        total_enhancement = F_redundancy * F_amplification * F_topology
        
        # Cap total enhancement at 100× for realistic bounds
        return min(100.0, total_enhancement)
    
    def build_observer_network(self, observers: List[Any]) -> Dict[str, Any]:
        """Build complete network graph from observer hierarchy"""
        build_start = time.time()
        
        # Add all observer nodes
        for i, observer in enumerate(observers):
            observer_data = {
                'gear_ratio': getattr(observer, 'extract_gear_ratio_signature', lambda: 1.0)(),
                'frequency': getattr(observer, 'frequency', 1.0),
                'scale_id': getattr(observer, 'scale_id', i),
                'observer_type': type(observer).__name__,
                'precision': 1e-12,  # Default precision
                'success_rate': getattr(observer, 'successful_observations', 0) / max(1, getattr(observer, 'total_observations', 1))
            }
            
            node_id = f"observer_{i}_{type(observer).__name__}"
            self.add_observer_node(node_id, observer_data)
        
        # Calculate network statistics
        network_stats = self.get_network_statistics()
        
        build_time = time.time() - build_start
        
        self.logger.info(f"Built observer network: {len(observers)} nodes, "
                        f"{self.network_graph.number_of_edges()} edges, "
                        f"build time {build_time*1000:.2f}ms")
        
        return {
            'build_success': True,
            'node_count': self.network_graph.number_of_nodes(),
            'edge_count': self.network_graph.number_of_edges(),
            'build_time': build_time,
            'network_statistics': network_stats,
            'equivalence_groups': len(self.equivalence_groups),
            'avg_group_size': np.mean([len(group) for group in self.equivalence_groups.values()]) if self.equivalence_groups else 0
        }
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network analysis statistics"""
        if self.network_graph.number_of_nodes() == 0:
            return {'empty_graph': True}
        
        stats = {
            'basic_metrics': {
                'node_count': self.network_graph.number_of_nodes(),
                'edge_count': self.network_graph.number_of_edges(),
                'density': nx.density(self.network_graph),
                'is_connected': nx.is_connected(self.network_graph)
            }
        }
        
        if nx.is_connected(self.network_graph):
            stats['connectivity_metrics'] = {
                'average_path_length': nx.average_shortest_path_length(self.network_graph),
                'diameter': nx.diameter(self.network_graph),
                'radius': nx.radius(self.network_graph)
            }
        
        # Centrality measures
        if self.network_graph.number_of_nodes() > 1:
            betweenness = nx.betweenness_centrality(self.network_graph)
            closeness = nx.closeness_centrality(self.network_graph)
            degree = nx.degree_centrality(self.network_graph)
            
            stats['centrality_metrics'] = {
                'max_betweenness': max(betweenness.values()) if betweenness else 0,
                'avg_betweenness': np.mean(list(betweenness.values())) if betweenness else 0,
                'max_closeness': max(closeness.values()) if closeness else 0,
                'max_degree_centrality': max(degree.values()) if degree else 0,
                'precision_hubs': [node for node, centrality in betweenness.items() if centrality > 0.1]
            }
        
        # Performance metrics
        stats['performance_metrics'] = {
            'total_navigations': self.navigation_count,
            'cache_hit_rate': len(self.path_cache) / max(1, self.navigation_count),
            'average_precision_enhancement': np.mean(self.precision_enhancements) if self.precision_enhancements else 1.0
        }
        
        return stats
    
    def visualize_network(self, output_dir: str) -> str:
        """Create network visualization"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            plt.figure(figsize=(15, 10))
            
            # Create layout
            if self.network_graph.number_of_nodes() > 0:
                pos = nx.spring_layout(self.network_graph, k=3, iterations=50)
                
                # Draw nodes colored by scale_id
                node_colors = []
                for node in self.network_graph.nodes():
                    scale_id = self.network_graph.nodes[node].get('scale_id', 0)
                    node_colors.append(scale_id)
                
                # Draw network
                nx.draw_networkx_nodes(self.network_graph, pos, 
                                     node_color=node_colors, 
                                     node_size=300, 
                                     cmap=plt.cm.viridis,
                                     alpha=0.7)
                
                nx.draw_networkx_edges(self.network_graph, pos, 
                                     alpha=0.5, 
                                     width=0.5)
                
                # Draw labels for high-centrality nodes only
                if hasattr(self, '_high_centrality_nodes'):
                    labels = {node: node.split('_')[1] for node in self._high_centrality_nodes}
                    nx.draw_networkx_labels(self.network_graph, pos, 
                                          labels, font_size=8)
                
                plt.title("Sango Rine Shumba Network Graph Structure\n"
                         f"Nodes: {self.network_graph.number_of_nodes()}, "
                         f"Edges: {self.network_graph.number_of_edges()}")
                plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), 
                           label='Scale ID')
            else:
                plt.text(0.5, 0.5, 'Empty Network Graph', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title("Network Graph Visualization")
            
            plt.axis('off')
            plt.tight_layout()
            
            viz_file = output_path / "network_graph_structure.png"
            plt.savefig(viz_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Network visualization saved: {viz_file}")
            return str(viz_file)
            
        except Exception as e:
            self.logger.error(f"Network visualization failed: {e}")
            return ""
    
    def export_network_data(self, filepath: str):
        """Export network graph and statistics to JSON"""
        export_data = {
            'export_metadata': {
                'export_timestamp': time.time(),
                'navigator_version': '1.0',
                'network_type': 'hierarchical_graph'
            },
            'network_statistics': self.get_network_statistics(),
            'graph_structure': {
                'nodes': dict(self.network_graph.nodes(data=True)),
                'edges': list(self.network_graph.edges(data=True))
            },
            'equivalence_groups': {str(k): list(v) for k, v in self.equivalence_groups.items()},
            'navigation_cache_stats': {
                'cached_paths': len(self.path_cache),
                'total_navigations': self.navigation_count
            },
            'theoretical_foundation': {
                'harmonic_convergence_principle': 'When |nω_A - mω_B| < ε_tolerance, create graph edge',
                'precision_enhancement_factors': {
                    'F_redundancy': 'Multiple paths provide statistical averaging',
                    'F_amplification': 'Hub nodes concentrate observation paths',
                    'F_topology': 'Sparse graphs enable efficient navigation'
                },
                'expected_enhancement': '~100× over tree structure'
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Network data exported to {filepath}")
    
    def __repr__(self):
        return (f"HierarchicalNavigator(nodes={self.network_graph.number_of_nodes()}, "
                f"edges={self.network_graph.number_of_edges()}, "
                f"navigations={self.navigation_count})")


def main():
    """Standalone test of hierarchical navigator"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Hierarchical Navigator Test")
    parser.add_argument('--output-dir', default='navigator_results', 
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
    
    # Initialize navigator
    navigator = HierarchicalNavigator(tolerance=args.tolerance)
    
    # Create test network with synthetic observers
    print("Creating test observer network...")
    test_observers = []
    
    # Create synthetic observers with some equivalent gear ratios
    base_ratios = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0]  # Fibonacci-like
    for i, base_ratio in enumerate(base_ratios):
        for j in range(3):  # 3 variants per base ratio
            class TestObserver:
                def __init__(self, gear_ratio, freq, scale):
                    self.gear_ratio = gear_ratio
                    self.frequency = freq
                    self.scale_id = scale
                    self.successful_observations = 80 + np.random.randint(-10, 10)
                    self.total_observations = 100
                
                def extract_gear_ratio_signature(self):
                    return self.gear_ratio
            
            # Add small variations to create near-equivalent nodes
            variant_ratio = base_ratio * (1.0 + (j-1) * 0.0001)  # Very small variations
            variant_freq = base_ratio * 1e6 * (1.0 + (j-1) * 0.0001)
            
            observer = TestObserver(variant_ratio, variant_freq, i+1)
            test_observers.append(observer)
    
    # Build network
    build_result = navigator.build_observer_network(test_observers)
    print(f"Network built: {build_result['node_count']} nodes, {build_result['edge_count']} edges")
    
    # Test navigation between nodes
    print("\nTesting navigation paths...")
    nodes = list(navigator.network_graph.nodes())
    
    if len(nodes) >= 2:
        # Test several navigation examples
        for i in range(min(5, len(nodes)-1)):
            source = nodes[i]
            target = nodes[-(i+1)]  # Navigate to opposite end
            
            result = navigator.find_shortest_path_navigation(source, target)
            if result['success']:
                print(f"Navigation {source} → {target}:")
                print(f"  Path length: {result['path_length']}")
                print(f"  Alternative paths: {result['path_count']}")
                print(f"  Precision enhancement: {result['precision_enhancement']:.2f}×")
                print(f"  Graph advantage: {'Yes' if result['graph_advantage'] else 'No'}")
            else:
                print(f"Navigation {source} → {target}: FAILED - {result.get('error', 'Unknown')}")
    
    # Generate statistics and visualization
    print(f"\nGenerating network analysis...")
    stats = navigator.get_network_statistics()
    print(f"Network density: {stats['basic_metrics']['density']:.3f}")
    print(f"Connected: {stats['basic_metrics']['is_connected']}")
    
    if 'centrality_metrics' in stats:
        print(f"Max betweenness centrality: {stats['centrality_metrics']['max_betweenness']:.3f}")
        print(f"Precision hubs: {len(stats['centrality_metrics']['precision_hubs'])}")
    
    # Export results
    navigator.export_network_data(f"{args.output_dir}/network_analysis.json")
    viz_file = navigator.visualize_network(args.output_dir)
    
    print(f"\nResults exported to {args.output_dir}/")
    if viz_file:
        print(f"Visualization: {viz_file}")
    
    print(f"\n🎉 Hierarchical Navigator test completed!")
    print(f"Demonstrated transition from tree → graph structure")
    print(f"Graph enhancement: {stats.get('performance_metrics', {}).get('average_precision_enhancement', 1.0):.1f}×")


if __name__ == "__main__":
    main()