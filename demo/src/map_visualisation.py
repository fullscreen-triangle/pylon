# visualization.py - Comprehensive visualization with geographic maps
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import sqlite3
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
import logging


class SangoVisualization:
    """Advanced visualization system for Sango Rine Shumba framework"""

    def __init__(self, data_storage):
        self.data_storage = data_storage
        self.logger = logging.getLogger(__name__)
        self.output_path = Path("visualizations")
        self.output_path.mkdir(exist_ok=True)

    def create_global_network_map(self, save_html: bool = True) -> folium.Map:
        """Create interactive global network topology map with real-time data"""

        # Get latest node states
        with sqlite3.connect(self.data_storage.db_path) as conn:
            nodes_df = pd.read_sql_query('''
                                         SELECT DISTINCT node_id,
                                                         latitude,
                                                         longitude,
                                                         timezone,
                                                         infrastructure_type,
                                                         current_load,
                                                         active_connections,
                                                         operational_status
                                         FROM node_states
                                         WHERE timestamp > ?
                                         ORDER BY timestamp DESC
                                         ''', conn, params=[time.time() - 3600])  # Last hour

        # Create base map centered on global view
        m = folium.Map(
            location=[20, 0],  # Center of world
            zoom_start=2,
            tiles='OpenStreetMap'
        )

        # Add network nodes
        for _, node in nodes_df.iterrows():
            # Color based on infrastructure type
            color_map = {
                'fiber': 'blue',
                'satellite': 'red',
                'wireless': 'green',
                'hybrid': 'purple'
            }
            color = color_map.get(node['infrastructure_type'], 'gray')

            # Size based on load
            radius = 5 + (node['current_load'] * 15)  # 5-20 pixel radius

            # Create popup with detailed info
            popup_html = f"""
            <b>{node['node_id']}</b><br>
            Infrastructure: {node['infrastructure_type']}<br>
            Current Load: {node['current_load']:.2%}<br>
            Active Connections: {node['active_connections']}<br>
            Status: {node['operational_status']}<br>
            Timezone: {node['timezone']}
            """

            folium.CircleMarker(
                location=[node['latitude'], node['longitude']],
                radius=radius,
                popup=folium.Popup(popup_html, max_width=300),
                color=color,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)

        # Add network connections with latency visualization
        self._add_network_connections_to_map(m, nodes_df)

        if save_html:
            map_file = self.output_path / f"global_network_map_{int(time.time())}.html"
            m.save(str(map_file))
            self.logger.info(f"Global network map saved to {map_file}")

        return m[[1]](  # __1)

    def _add_network_connections_to_map(self, map_obj: folium.Map, nodes_df: pd.DataFrame):
        """Add network connection lines with latency color coding"""

        with sqlite3.connect(self.data_storage.db_path) as conn:
            connections_df = pd.read_sql_query('''
                                               SELECT source_node_id,
                                                      destination_node_id,
                                                      AVG(calculated_latency_ms) as avg_latency,
                                                      AVG(packet_loss_rate)      as avg_packet_loss
                                               FROM network_measurements
                                               WHERE timestamp > ?
                                               GROUP BY source_node_id, destination_node_id
                                               ''', conn, params=[time.time() - 3600])

        # Create node location lookup
        node_locations = {
            row['node_id']: [row['latitude'], row['longitude']]
            for _, row in nodes_df.iterrows()
        }

        for _, conn in connections_df.iterrows():
            source_loc = node_locations.get(conn['source_node_id'])
            dest_loc = node_locations.get(conn['destination_node_id'])

            if source_loc and dest_loc:
                # Color based on latency (green=fast, red=slow)
                latency = conn['avg_latency']
                if latency < 50:
                    color = 'green'
                elif latency < 150:
                    color = 'orange'
                else:
                    color = 'red'

                # Line weight based on packet loss (thinner = more loss)
                weight = max(1, 5 - (conn['avg_packet_loss'] * 100))

                folium.PolyLine(
                    locations=[source_loc, dest_loc],
                    color=color,
                    weight=weight,
                    opacity=0.6,
                    popup=f"Latency: {latency:.1f}ms, Loss: {conn['avg_packet_loss']:.3%}"
                ).add_to(map_obj)

    def create_precision_dashboard(self) -> go.Figure:
        """Create comprehensive precision analysis dashboard"""

        # Get precision data
        with sqlite3.connect(self.data_storage.db_path) as conn:
            precision_df = pd.read_sql_query('''
                                             SELECT timestamp, node_id, precision_difference, measurement_quality, standard_deviation, confidence_interval_lower, confidence_interval_upper
                                             FROM precision_calculations
                                             WHERE timestamp > ?
                                             ORDER BY timestamp
                                             ''', conn, params=[time.time() - 7200])  # Last 2 hours

        # Convert timestamp to datetime
        precision_df['datetime'] = pd.to_datetime(precision_df['timestamp'], unit='s')

        # Create subplot dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Precision Difference Over Time',
                'Measurement Quality Distribution',
                'Standard Deviation by Node',
                'Confidence Intervals'
            ),
            specs=[[{"secondary_y": True}, {"type": "histogram"}],
                   [{"type": "box"}, {"secondary_y": True}]]
        )

        # Plot 1: Precision difference time series
        for node_id in precision_df['node_id'].unique():
            node_data = precision_df[precision_df['node_id'] == node_id]
            fig.add_trace(
                go.Scatter(
                    x=node_data['datetime'],
                    y=node_data['precision_difference'] * 1000,  # Convert to ms
                    mode='lines+markers',
                    name=f'{node_id} Precision',
                    line=dict(width=2)
                ),
                row=1, col=1
            )

        # Plot 2: Quality distribution
        fig.add_trace(
            go.Histogram(
                x=precision_df['measurement_quality'],
                nbinsx=20,
                name='Quality Distribution',
                marker_color='lightblue'
            ),
            row=1, col=2
        )

        # Plot 3: Standard deviation by node
        fig.add_trace(
            go.Box(
                y=precision_df['standard_deviation'] * 1000,  # Convert to ms
                x=precision_df['node_id'],
                name='Std Dev by Node',
                marker_color='lightgreen'
            ),
            row=2, col=1
        )

        # Plot 4: Confidence intervals
        for node_id in precision_df['node_id'].unique()[:3]:  # Limit to 3 nodes for clarity
            node_data = precision_df[precision_df['node_id'] == node_id].tail(50)  # Last 50 measurements

            fig.add_trace(
                go.Scatter(
                    x=node_data['datetime'],
                    y=node_data['confidence_interval_upper'] * 1000,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=2, col=2
            )

            fig.add_trace(
                go.Scatter(
                    x=node_data['datetime'],
                    y=node_data['confidence_interval_lower'] * 1000,
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=f'rgba({hash(node_id) % 255}, {(hash(node_id) * 2) % 255}, {(hash(node_id) * 3) % 255}, 0.3)',
                    name=f'{node_id} CI',
                ),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title="Sango Rine Shumba Precision Analysis Dashboard",
            height=800,
            showlegend=True
        )

        # Update axes labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Precision Difference (ms)", row=1, col=1)
        fig.update_xaxes(title_text="Measurement Quality", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Node ID", row=2, col=1)
        fig.update_yaxes(title_text="Standard Deviation (ms)", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=2)
        fig.update_yaxes(title_text="Confidence Interval (ms)", row=2, col=2)

        # Save dashboard
        dashboard_file = self.output_path / f"precision_dashboard_{int(time.time())}.html"
        fig.write_html(str(dashboard_file))
        self.logger.info(f"Precision dashboard saved to {dashboard_file}")

        return fig[[2]](  # __2)

    def create_web_performance_comparison(self) -> go.Figure:
        """Create web performance comparison: Traditional vs Sango Streaming"""

        with sqlite3.connect(self.data_storage.db_path) as conn:
            perf_df = pd.read_sql_query('''
                                        SELECT timestamp, page_id, url, loading_method, total_load_time_ms, dns_time_ms, tcp_time_ms, html_time_ms, css_time_ms, js_time_ms, image_time_ms, improvement_percentage, page_size_kb, complexity_score
                                        FROM web_performance
                                        WHERE timestamp > ?
                                        ORDER BY timestamp
                                        ''', conn, params=[time.time() - 3600])

        # Create comparison visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Load Time Comparison by Method',
                'Performance Improvement Distribution',
                'Load Time Breakdown (Traditional)',
                'Page Complexity vs Load Time'
            ),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )

        # Group by loading method
        traditional_data = perf_df[perf_df['loading_method'] == 'traditional']
        sango_data = perf_df[perf_df['loading_method'] == 'sango_streaming']

        # Plot 1: Load time comparison
        if not traditional_data.empty:
            fig.add_trace(
                go.Bar(
                    x=traditional_data['page_id'],
                    y=traditional_data['total_load_time_ms'],
                    name='Traditional Loading',
                    marker_color='red',
                    opacity=0.7
                ),
                row=1, col=1
            )

        if not sango_data.empty:
            fig.add_trace(
                go.Bar(
                    x=sango_data['page_id'],
                    y=sango_data['total_load_time_ms'],
                    name='Sango Streaming',
                    marker_color='green',
                    opacity=0.7
                ),
                row=1, col=1
            )

        # Plot 2: Improvement distribution
        improvements = perf_df[perf_df['improvement_percentage'] > 0]['improvement_percentage']
        if not improvements.empty:
            fig.add_trace(
                go.Histogram(
                    x=improvements,
                    nbinsx=15,
                    name='Improvement %',
                    marker_color='blue'
                ),
                row=1, col=2
            )

        # Plot 3: Load time breakdown for traditional method
        if not traditional_data.empty:
            sample_page = traditional_data.iloc[0]  # Take first page as example

            breakdown_data = {
                'DNS': sample_page['dns_time_ms'],
                'TCP': sample_page['tcp_time_ms'],
                'HTML': sample_page['html_time_ms'],
                'CSS': sample_page['css_time_ms'],
                'JS': sample_page['js_time_ms'],
                'Images': sample_page['image_time_ms']
            }

            fig.add_trace(
                go.Bar(
                    x=list(breakdown_data.keys()),
                    y=list(breakdown_data.values()),
                    name='Time Breakdown',
                    marker_color='orange'
                ),
                row=2, col=1
            )

        # Plot 4: Complexity vs performance
        fig.add_trace(
            go.Scatter(
                x=perf_df['complexity_score'],
                y=perf_df['total_load_time_ms'],
                mode='markers',
                marker=dict(
                    size=perf_df['page_size_kb'] / 10,  # Size based on page size
                    color=perf_df['loading_method'].map({'traditional': 'red', 'sango_streaming': 'green'}),
                    opacity=0.6
                ),
                text=perf_df['page_id'],
                name='Complexity vs Load Time'
            ),
            row=2, col=2
        )

        fig.update_layout(
            title="Web Performance Analysis: Traditional vs Sango Rine Shumba",
            height=800,
            showlegend=True
        )

        # Save performance comparison
        perf_file = self.output_path / f"web_performance_comparison_{int(time.time())}.html"
        fig.write_html(str(perf_file))
        self.logger.info(f"Web performance comparison saved to {perf_file}")

        return fig[[3]](  # __3)

    def create_real_time_monitoring_dashboard(self) -> go.Figure:
        """Create real-time monitoring dashboard with live updates"""

        # This would typically be updated via WebSocket or similar for true real-time
        fig = go.Figure()

        # Get recent data from memory cache
        recent_precision = self.data_storage.memory_cache['precision_calculations'][-100:]
        recent_network = self.data_storage.memory_cache['network_latencies'][-100:]

        if recent_precision:
            timestamps = [p['timestamp'] for p in recent_precision]
            precision_values = [p.get('precision_difference', 0) * 1000 for p in recent_precision]  # ms

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=precision_values,
                    mode='lines+markers',
                    name='Precision Difference (ms)',
                    line=dict(color='blue', width=2)
                )
            )

        fig.update_layout(
            title="Real-Time Sango Rine Shumba Monitoring",
            xaxis_title="Time",
            yaxis_title="Precision Difference (ms)",
            template="plotly_dark",
            height=400
        )

        return fig

    def generate_comprehensive_report(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive analysis report with all visualizations"""

        start_time = time.time() - (time_range_hours * 3600)

        report = {
            'generation_time': datetime.now().isoformat(),
            'time_range_hours': time_range_hours,
            'visualizations': {},
            'statistics': {},
            'files_generated': []
        }

        try:
            # Generate all visualizations
            network_map = self.create_global_network_map()
            precision_dashboard = self.create_precision_dashboard()
            performance_comparison = self.create_web_performance_comparison()

            # Calculate summary statistics
            with sqlite3.connect(self.data_storage.db_path) as conn:
                # Precision statistics
                precision_stats = pd.read_sql_query('''
                                                    SELECT COUNT(*)                     as total_measurements,
                                                           AVG(precision_difference)    as avg_precision_diff,
                                                           STDDEV(precision_difference) as std_precision_diff,
                                                           MIN(precision_difference)    as min_precision_diff,
                                                           MAX(precision_difference)    as max_precision_diff,
                                                           AVG(measurement_quality)     as avg_quality
                                                    FROM precision_calculations
                                                    WHERE timestamp > ?
                                                    ''', conn, params=[start_time])

                # Network statistics
                network_stats = pd.read_sql_query('''
                                                  SELECT COUNT(*)                       as total_connections,
                                                         AVG(calculated_latency_ms)     as avg_latency,
                                                         AVG(packet_loss_rate)          as avg_packet_loss,
                                                         COUNT(DISTINCT source_node_id) as active_nodes
                                                  FROM network_measurements
                                                  WHERE timestamp > ?
                                                  ''', conn, params=[start_time])

                # Web performance statistics
                web_stats = pd.read_sql_query('''
                                              SELECT loading_method,
                                                     COUNT(*)                    as page_loads,
                                                     AVG(total_load_time_ms)     as avg_load_time,
                                                     AVG(improvement_percentage) as avg_improvement
                                              FROM web_performance
                                              WHERE timestamp > ?
                                              GROUP BY loading_method
                                              ''', conn, params=[start_time])

            report['statistics'] = {
                'precision': precision_stats.to_dict('records')[0] if not precision_stats.empty else {},
                'network': network_stats.to_dict('records')[0] if not network_stats.empty else {},
                'web_performance': web_stats.to_dict('records') if not web_stats.empty else []
            }

            # Export data to CSV
            for table in ['precision_calculations', 'network_measurements', 'web_performance']:
                csv_file = self.data_storage.export_to_csv(table, (start_time, time.time()))
                report['files_generated'].append(str(csv_file))

            # Save report as JSON
            report_file = self.output_path / f"comprehensive_report_{int(time.time())}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            report['files_generated'].append(str(report_file))

            self.logger.info(f"Comprehensive report generated with {len(report['files_generated'])} files")

        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            report['error'] = str(e)

        return report
