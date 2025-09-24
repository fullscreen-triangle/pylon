# json_visualization.py - Lightweight visualization using JSON data
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any
import folium


class SangoJSONVisualization:
    """Lightweight visualization system reading from JSON experiment data"""

    def __init__(self, json_storage):
        self.storage = json_storage
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path("experiment_visualizations")
        self.output_dir.mkdir(exist_ok=True)

    def create_precision_timeline(self) -> go.Figure:
        """Create precision difference timeline from JSON data"""
        precision_data = self.storage.experiment_data['precision_calculations']

        if not precision_data:
            self.logger.warning("No precision data available for visualization")
            return go.Figure()

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(precision_data)
        df['precision_difference_ms'] = df['precision_difference'] * 1000  # Convert to ms
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

        fig = go.Figure()

        # Group by node_id if available
        if 'node_id' in df.columns:
            for node_id in df['node_id'].unique():
                node_data = df[df['node_id'] == node_id]
                fig.add_trace(go.Scatter(
                    x=node_data['datetime'],
                    y=node_data['precision_difference_ms'],
                    mode='lines+markers',
                    name=f'Node {node_id}',
                    line=dict(width=2)
                ))
        else:
            fig.add_trace(go.Scatter(
                x=df['datetime'],
                y=df['precision_difference_ms'],
                mode='lines+markers',
                name='Precision Difference',
                line=dict(width=2, color='blue')
            ))

        fig.update_layout(
            title="Sango Rine Shumba Precision Analysis - Validation Experiment",
            xaxis_title="Time",
            yaxis_title="Precision Difference (milliseconds)",
            height=500,
            template="plotly_white"
        )

        # Save visualization
        output_file = self.output_dir / "precision_timeline.html"
        fig.write_html(str(output_file))
        self.logger.info(f"Precision timeline saved to {output_file}")

        return fig[[1]](  # __1)

    def create_performance_comparison(self) -> go.Figure:
        """Create web performance comparison chart"""
        web_data = self.storage.experiment_data['web_performance']

        if not web_data:
            self.logger.warning("No web performance data available")
            return go.Figure()

        df = pd.DataFrame(web_data)

        # Separate traditional vs sango data
        traditional = df[df['loading_method'] == 'traditional'] if 'loading_method' in df.columns else pd.DataFrame()
        sango = df[df['loading_method'] == 'sango_streaming'] if 'loading_method' in df.columns else pd.DataFrame()

        fig = go.Figure()

        if not traditional.empty:
            fig.add_trace(go.Box(
                y=traditional['total_load_time_ms'],
                name='Traditional Loading',
                marker_color='red',
                boxpoints='all',
                jitter=0.3
            ))

        if not sango.empty:
            fig.add_trace(go.Box(
                y=sango['total_load_time_ms'],
                name='Sango Streaming',
                marker_color='green',
                boxpoints='all',
                jitter=0.3
            ))

        fig.update_layout(
            title="Web Performance Comparison - Validation Results",
            yaxis_title="Load Time (milliseconds)",
            height=500,
            template="plotly_white"
        )

        # Save visualization
        output_file = self.output_dir / "performance_comparison.html"
        fig.write_html(str(output_file))
        self.logger.info(f"Performance comparison saved to {output_file}")

        return fig

    def create_network_latency_heatmap(self) -> go.Figure:
        """Create network latency heatmap from JSON data"""
        network_data = self.storage.experiment_data['network_measurements']

        if not network_data:
            self.logger.warning("No network data available")
            return go.Figure()

        df = pd.DataFrame(network_data)

        # Create latency matrix
        if 'source_node_id' in df.columns and 'destination_node_id' in df.columns:
            # Pivot to create matrix
            latency_matrix = df.pivot_table(
                values='measured_latency_ms',
                index='source_node_id',
                columns='destination_node_id',
                aggfunc='mean'
            )

            fig = go.Figure(data=go.Heatmap(
                z=latency_matrix.values,
                x=latency_matrix.columns,
                y=latency_matrix.index,
                colorscale='RdYlBu_r',
                colorbar=dict(title="Latency (ms)")
            ))

            fig.update_layout(
                title="Network Latency Matrix - Validation Experiment",
                xaxis_title="Destination Node",
                yaxis_title="Source Node",
                height=500
            )
        else:
            # Simple latency distribution
            fig = go.Figure(data=go.Histogram(
                x=df['measured_latency_ms'],
                nbinsx=20,
                name='Latency Distribution'
            ))

            fig.update_layout(
                title="Network Latency Distribution",
                xaxis_title="Latency (ms)",
                yaxis_title="Frequency"
            )

        # Save visualization
        output_file = self.output_dir / "network_latency.html"
        fig.write_html(str(output_file))

        return fig[[2]](  # __2)

    def create_experiment_dashboard(self) -> go.Figure:
        """Create comprehensive experiment dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Precision Over Time',
                'Performance Improvement',
                'Network Latency Distribution',
                'Data Collection Summary'
            ),
            specs=[[{"secondary_y": False}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )

        # Plot 1: Precision over time
        precision_data = self.storage.experiment_data['precision_calculations']
        if precision_data:
            df = pd.DataFrame(precision_data)
            df['precision_ms'] = df['precision_difference'] * 1000
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

            fig.add_trace(
                go.Scatter(
                    x=df['datetime'],
                    y=df['precision_ms'],
                    mode='lines',
                    name='Precision',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )

        # Plot 2: Performance improvement
        web_data = self.storage.experiment_data['web_performance']
        if web_data:
            improvements = [w.get('improvement_percentage', 0) for w in web_data
                            if w.get('improvement_percentage', 0) > 0]
            if improvements:
                fig.add_trace(
                    go.Bar(
                        x=list(range(len(improvements))),
                        y=improvements,
                        name='Improvement %',
                        marker_color='green'
                    ),
                    row=1, col=2
                )

        # Plot 3: Network latency distribution
        network_data = self.storage.experiment_data['network_measurements']
        if network_data:
            latencies = [n.get('measured_latency_ms', 0) for n in network_data]
            fig.add_trace(
                go.Histogram(
                    x=latencies,
                    name='Latency Distribution',
                    marker_color='orange'
                ),
                row=2, col=1
            )

        # Plot 4: Data collection summary
        data_counts = {
            'Atomic': len(self.storage.experiment_data['atomic_measurements']),
            'Precision': len(self.storage.experiment_data['precision_calculations']),
            'Network': len(self.storage.experiment_data['network_measurements']),
            'Web': len(self.storage.experiment_data['web_performance'])
        }

        fig.add_trace(
            go.Bar(
                x=list(data_counts.keys()),
                y=list(data_counts.values()),
                name='Data Points',
                marker_color='purple'
            ),
            row=2, col=2
        )

        fig.update_layout(
            title="Sango Rine Shumba Validation Experiment Dashboard",
            height=800,
            showlegend=False
        )

        # Save dashboard
        output_file = self.output_dir / "experiment_dashboard.html"
        fig.write_html(str(output_file))
        self.logger.info(f"Experiment dashboard saved to {output_file}")

        return fig[[3]](3)

    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report with all visualizations"""

        # Create all visualizations
        precision_fig = self.create_precision_timeline()
        performance_fig = self.create_performance_comparison()
        network_fig = self.create_network_latency_heatmap()
        dashboard_fig = self.create_experiment_dashboard()

        # Get summary statistics
        summary = self.storage.export_summary_report()

        # Create validation report
        validation_report = {
            'experiment_metadata': self.storage.experiment_data['metadata'],
            'summary_statistics': summary,
            'validation_conclusions': self._generate_validation_conclusions(summary),
            'visualizations_generated': [
                str(self.output_dir / "precision_timeline.html"),
                str(self.output_dir / "performance_comparison.html"),
                str(self.output_dir / "network_latency.html"),
                str(self.output_dir / "experiment_dashboard.html")
            ],
            'data_export': str(self.storage.save_experiment_data(f"final_experiment_data_{int(time.time())}.json"))
        }

        # Save validation report
        report_file = self.output_dir / f"validation_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(validation_report, f, indent=2, default=self.storage._json_serializer)

        return validation_report

    def _generate_validation_conclusions(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation conclusions based on experimental data"""
        conclusions = {
            'precision_validation': 'insufficient_data',
            'performance_validation': 'insufficient_data',
            'network_validation': 'insufficient_data',
            'overall_assessment': 'incomplete'
        }

        # Analyze precision results
        if 'key_findings' in summary and 'precision_analysis' in summary['key_findings']:
            precision = summary['key_findings']['precision_analysis']
            avg_precision = abs(precision.get('average_precision_difference_ms', 0))

            if avg_precision < 1.0:  # Less than 1ms average difference
                conclusions['precision_validation'] = 'excellent'
            elif avg_precision < 5.0:  # Less than 5ms
                conclusions['precision_validation'] = 'good'
            elif avg_precision < 20.0:  # Less than 20ms
                conclusions['precision_validation'] = 'acceptable'
            else:
                conclusions['precision_validation'] = 'needs_improvement'

        # Analyze performance results
        if 'key_findings' in summary and 'web_performance_comparison' in summary['key_findings']:
            perf = summary['key_findings']['web_performance_comparison']
            improvement = perf.get('average_improvement_percentage', 0)

            if improvement > 50:  # More than 50% improvement
                conclusions['performance_validation'] = 'excellent'
            elif improvement > 25:  # More than 25% improvement
                conclusions['performance_validation'] = 'good'
            elif improvement > 10:  # More than 10% improvement
                conclusions['performance_validation'] = 'acceptable'
            else:
                conclusions['performance_validation'] = 'marginal'

        # Overall assessment
        validations = [conclusions['precision_validation'], conclusions['performance_validation']]
        if all(v in ['excellent', 'good'] for v in validations if v != 'insufficient_data'):
            conclusions['overall_assessment'] = 'validation_successful'
        elif any(v in ['excellent', 'good', 'acceptable'] for v in validations if v != 'insufficient_data'):
            conclusions['overall_assessment'] = 'partial_validation'
        else:
            conclusions['overall_assessment'] = 'validation_inconclusive'

        return conclusions
