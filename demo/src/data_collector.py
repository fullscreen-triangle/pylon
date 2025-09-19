"""
Data Collector Module

Comprehensive data collection and persistence system for the Sango Rine Shumba
demonstration. Captures all experimental measurements, intermediate results,
and performance metrics for analysis and publication.

This module ensures complete data transparency and reproducibility for research
validation and peer review.
"""

import asyncio
import sqlite3
import json
import csv
import time
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import pandas as pd
import numpy as np

class DataCollector:
    """
    Comprehensive data collection system
    
    Provides structured data collection for all aspects of the Sango Rine Shumba
    demonstration including:
    - Network performance measurements
    - Precision-by-difference calculations
    - Temporal fragmentation metrics
    - MIMO routing statistics
    - Comparative analysis data
    """
    
    def __init__(self, experiment_id: str, data_dir: str = "data"):
        """Initialize data collector"""
        self.experiment_id = experiment_id
        self.data_dir = Path(data_dir)
        self.experiment_dir = self.data_dir / "experiments" / experiment_id
        self.logger = logging.getLogger(__name__)
        
        # Database connection
        self.db_connection: Optional[sqlite3.Connection] = None
        self.db_path = self.experiment_dir / "experiment_data.db"
        
        # Data storage
        self.csv_writers = {}
        self.csv_files = {}
        
        # Performance tracking
        self.data_points_collected = 0
        self.last_flush_time = time.time()
        self.flush_interval = 5.0  # 5 seconds
        
        self.logger.info(f"Data collector initialized for experiment {experiment_id}")
    
    async def initialize(self):
        """Initialize data collection infrastructure"""
        self.logger.info("Initializing data collection infrastructure...")
        
        # Create directory structure
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "intermediate").mkdir(exist_ok=True)
        (self.experiment_dir / "visualizations").mkdir(exist_ok=True)
        (self.experiment_dir / "exports").mkdir(exist_ok=True)
        
        # Initialize database
        await self._initialize_database()
        
        # Initialize CSV files
        await self._initialize_csv_files()
        
        # Start background tasks
        asyncio.create_task(self._periodic_flush())
        
        self.logger.info("Data collection infrastructure ready")
    
    async def _initialize_database(self):
        """Initialize SQLite database for structured data storage"""
        self.db_connection = sqlite3.connect(str(self.db_path))
        
        # Create tables
        tables = {
            'experiment_metadata': '''
                CREATE TABLE IF NOT EXISTS experiment_metadata (
                    id TEXT PRIMARY KEY,
                    start_time REAL,
                    parameters TEXT,
                    git_commit TEXT,
                    system_info TEXT
                )
            ''',
            'network_measurements': '''
                CREATE TABLE IF NOT EXISTS network_measurements (
                    timestamp REAL,
                    measurement_type TEXT,
                    source_node TEXT,
                    destination_node TEXT,
                    value REAL,
                    unit TEXT,
                    metadata TEXT
                )
            ''',
            'precision_calculations': '''
                CREATE TABLE IF NOT EXISTS precision_calculations (
                    timestamp REAL,
                    node_id TEXT,
                    atomic_reference REAL,
                    local_measurement REAL,
                    precision_difference REAL,
                    measurement_quality REAL,
                    confidence REAL,
                    reference_source TEXT
                )
            ''',
            'coordination_matrices': '''
                CREATE TABLE IF NOT EXISTS coordination_matrices (
                    timestamp REAL,
                    matrix_id TEXT,
                    num_measurements INTEGER,
                    temporal_window_duration REAL,
                    coordination_accuracy REAL,
                    synchronization_quality REAL
                )
            ''',
            'message_fragmentation': '''
                CREATE TABLE IF NOT EXISTS message_fragmentation (
                    timestamp REAL,
                    message_id TEXT,
                    source_node TEXT,
                    destination_node TEXT,
                    original_size INTEGER,
                    fragment_count INTEGER,
                    total_fragment_size INTEGER,
                    overhead_ratio REAL,
                    average_entropy REAL,
                    fragmentation_time REAL,
                    temporal_window_duration REAL
                )
            ''',
            'message_reconstruction': '''
                CREATE TABLE IF NOT EXISTS message_reconstruction (
                    timestamp REAL,
                    message_id TEXT,
                    fragment_count INTEGER,
                    reconstructed_size INTEGER,
                    reconstruction_time REAL,
                    success INTEGER,
                    error TEXT
                )
            ''',
            'mimo_routing': '''
                CREATE TABLE IF NOT EXISTS mimo_routing (
                    timestamp REAL,
                    transmission_id TEXT,
                    message_id TEXT,
                    source_node TEXT,
                    destination_node TEXT,
                    fragment_count INTEGER,
                    path_count INTEGER,
                    target_arrival_time REAL,
                    estimated_convergence_ms REAL
                )
            ''',
            'mimo_completion': '''
                CREATE TABLE IF NOT EXISTS mimo_completion (
                    timestamp REAL,
                    transmission_id TEXT,
                    success INTEGER,
                    completion_ratio REAL,
                    convergence_quality REAL,
                    total_time REAL
                )
            ''',
            'traditional_messages': '''
                CREATE TABLE IF NOT EXISTS traditional_messages (
                    timestamp REAL,
                    source_node TEXT,
                    destination_node TEXT,
                    message_size INTEGER,
                    latency_seconds REAL,
                    connection_latency_ms REAL,
                    packet_loss_occurred INTEGER,
                    processing_delay_ms REAL
                )
            ''',
            'browser_page_loads': '''
                CREATE TABLE IF NOT EXISTS browser_page_loads (
                    timestamp REAL,
                    session_id TEXT,
                    page_id TEXT,
                    node_id TEXT,
                    method TEXT,
                    total_time REAL,
                    page_size_kb REAL,
                    complexity_score REAL,
                    render_time REAL,
                    download_time REAL
                )
            ''',
            'user_interactions': '''
                CREATE TABLE IF NOT EXISTS user_interactions (
                    timestamp REAL,
                    session_id TEXT,
                    interaction_id TEXT,
                    interaction_type TEXT,
                    reaction_time_ms REAL,
                    execution_time_ms REAL,
                    biometric_signature TEXT
                )
            ''',
            'biometric_verifications': '''
                CREATE TABLE IF NOT EXISTS biometric_verifications (
                    timestamp REAL,
                    session_id TEXT,
                    user_id TEXT,
                    interaction_id TEXT,
                    verification_method TEXT,
                    verified INTEGER,
                    confidence REAL,
                    verification_time_ms REAL,
                    biometric_features TEXT
                )
            ''',
            'zero_latency_events': '''
                CREATE TABLE IF NOT EXISTS zero_latency_events (
                    timestamp REAL,
                    session_id TEXT,
                    interaction_id TEXT,
                    interaction_type TEXT,
                    predicted INTEGER,
                    response_time_ms REAL,
                    response_data TEXT,
                    user_satisfaction_boost REAL
                )
            ''',
            'browser_comparison_reports': '''
                CREATE TABLE IF NOT EXISTS browser_comparison_reports (
                    timestamp REAL,
                    traditional_avg_load_time REAL,
                    sango_avg_load_time REAL,
                    load_time_improvement REAL,
                    user_experience_improvement REAL,
                    zero_latency_events INTEGER,
                    biometric_verifications INTEGER
                )
            '''
        }
        
        for table_name, create_sql in tables.items():
            self.db_connection.execute(create_sql)
        
        self.db_connection.commit()
        self.logger.debug(f"Created {len(tables)} database tables")
    
    async def _initialize_csv_files(self):
        """Initialize CSV files for real-time data export"""
        csv_files = [
            'network_performance',
            'precision_metrics',
            'fragmentation_stats',
            'routing_performance',
            'comparative_analysis'
        ]
        
        for file_name in csv_files:
            file_path = self.experiment_dir / f"{file_name}.csv"
            csv_file = open(file_path, 'w', newline='', encoding='utf-8')
            self.csv_files[file_name] = csv_file
            
            # Initialize with headers based on file type
            if file_name == 'network_performance':
                fieldnames = ['timestamp', 'measurement_type', 'source_node', 'destination_node', 
                            'latency_ms', 'bandwidth_mbps', 'packet_loss', 'jitter_ms']
            elif file_name == 'precision_metrics':
                fieldnames = ['timestamp', 'node_id', 'precision_difference', 'measurement_quality',
                            'confidence', 'reference_source', 'coordination_accuracy']
            elif file_name == 'fragmentation_stats':
                fieldnames = ['timestamp', 'message_id', 'fragment_count', 'average_entropy',
                            'fragmentation_time', 'reconstruction_success', 'reconstruction_time']
            elif file_name == 'routing_performance':
                fieldnames = ['timestamp', 'transmission_id', 'path_count', 'convergence_quality',
                            'completion_ratio', 'bandwidth_efficiency', 'latency_improvement']
            else:  # comparative_analysis
                fieldnames = ['timestamp', 'metric_name', 'traditional_value', 'sango_value',
                            'improvement_ratio', 'statistical_significance']
            
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            self.csv_writers[file_name] = writer
    
    async def log_precision_measurement(self, data: Dict[str, Any]):
        """Log precision-by-difference measurement"""
        try:
            # Store in database
            self.db_connection.execute('''
                INSERT INTO precision_calculations 
                (timestamp, node_id, atomic_reference, local_measurement, precision_difference, 
                 measurement_quality, confidence, reference_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['timestamp'],
                data['node_id'],
                data['atomic_reference'],
                data['local_measurement'],
                data['precision_difference'],
                data['measurement_quality'],
                data['confidence'],
                data['reference_source']
            ))
            
            # Write to CSV
            self.csv_writers['precision_metrics'].writerow({
                'timestamp': data['timestamp'],
                'node_id': data['node_id'],
                'precision_difference': data['precision_difference'],
                'measurement_quality': data['measurement_quality'],
                'confidence': data['confidence'],
                'reference_source': data['reference_source'],
                'coordination_accuracy': data.get('coordination_accuracy', 0.0)
            })
            
            self.data_points_collected += 1
            
        except Exception as e:
            self.logger.error(f"Error logging precision measurement: {e}")
    
    async def log_coordination_matrix(self, data: Dict[str, Any]):
        """Log coordination matrix generation"""
        try:
            self.db_connection.execute('''
                INSERT INTO coordination_matrices
                (timestamp, matrix_id, num_measurements, temporal_window_duration, 
                 coordination_accuracy, synchronization_quality)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                data['timestamp'],
                data['matrix_id'],
                data['num_measurements'],
                data['temporal_window_duration'],
                data['coordination_accuracy'],
                data['synchronization_quality']
            ))
            
            self.data_points_collected += 1
            
        except Exception as e:
            self.logger.error(f"Error logging coordination matrix: {e}")
    
    async def log_message_fragmentation(self, data: Dict[str, Any]):
        """Log message fragmentation event"""
        try:
            self.db_connection.execute('''
                INSERT INTO message_fragmentation
                (timestamp, message_id, source_node, destination_node, original_size,
                 fragment_count, total_fragment_size, overhead_ratio, average_entropy,
                 fragmentation_time, temporal_window_duration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['timestamp'],
                data['message_id'],
                data['source_node'],
                data['destination_node'],
                data['original_size'],
                data['fragment_count'],
                data['total_fragment_size'],
                data['overhead_ratio'],
                data['average_entropy'],
                data['fragmentation_time'],
                data['temporal_window_duration']
            ))
            
            # Write to CSV
            self.csv_writers['fragmentation_stats'].writerow({
                'timestamp': data['timestamp'],
                'message_id': data['message_id'],
                'fragment_count': data['fragment_count'],
                'average_entropy': data['average_entropy'],
                'fragmentation_time': data['fragmentation_time'],
                'reconstruction_success': '',  # Will be filled later
                'reconstruction_time': ''
            })
            
            self.data_points_collected += 1
            
        except Exception as e:
            self.logger.error(f"Error logging message fragmentation: {e}")
    
    async def log_message_reconstruction(self, data: Dict[str, Any]):
        """Log message reconstruction event"""
        try:
            self.db_connection.execute('''
                INSERT INTO message_reconstruction
                (timestamp, message_id, fragment_count, reconstructed_size,
                 reconstruction_time, success, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['timestamp'],
                data['message_id'],
                data['fragment_count'],
                data.get('reconstructed_size', 0),
                data['reconstruction_time'],
                1 if data['success'] else 0,
                data.get('error', '')
            ))
            
            self.data_points_collected += 1
            
        except Exception as e:
            self.logger.error(f"Error logging message reconstruction: {e}")
    
    async def log_mimo_routing(self, data: Dict[str, Any]):
        """Log MIMO routing event"""
        try:
            self.db_connection.execute('''
                INSERT INTO mimo_routing
                (timestamp, transmission_id, message_id, source_node, destination_node,
                 fragment_count, path_count, target_arrival_time, estimated_convergence_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['timestamp'],
                data['transmission_id'],
                data['message_id'],
                data['source_node'],
                data['destination_node'],
                data['fragment_count'],
                data['path_count'],
                data['target_arrival_time'],
                data['estimated_convergence_ms']
            ))
            
            self.data_points_collected += 1
            
        except Exception as e:
            self.logger.error(f"Error logging MIMO routing: {e}")
    
    async def log_mimo_completion(self, data: Dict[str, Any]):
        """Log MIMO transmission completion"""
        try:
            self.db_connection.execute('''
                INSERT INTO mimo_completion
                (timestamp, transmission_id, success, completion_ratio,
                 convergence_quality, total_time)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                data['timestamp'],
                data['transmission_id'],
                1 if data['success'] else 0,
                data['completion_ratio'],
                data['convergence_quality'],
                data['total_time']
            ))
            
            # Write to CSV
            self.csv_writers['routing_performance'].writerow({
                'timestamp': data['timestamp'],
                'transmission_id': data['transmission_id'],
                'path_count': '',  # Will need to join with routing table
                'convergence_quality': data['convergence_quality'],
                'completion_ratio': data['completion_ratio'],
                'bandwidth_efficiency': '',  # Will be calculated
                'latency_improvement': ''
            })
            
            self.data_points_collected += 1
            
        except Exception as e:
            self.logger.error(f"Error logging MIMO completion: {e}")
    
    async def log_traditional_message(self, data: Dict[str, Any]):
        """Log traditional message transmission"""
        try:
            self.db_connection.execute('''
                INSERT INTO traditional_messages
                (timestamp, source_node, destination_node, message_size,
                 latency_seconds, connection_latency_ms, packet_loss_occurred,
                 processing_delay_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['timestamp'],
                data['source_node'],
                data['destination_node'],
                data['message_size'],
                data['latency_seconds'],
                data['connection_latency_ms'],
                1 if data['packet_loss_occurred'] else 0,
                data['processing_delay_ms']
            ))
            
            # Write to CSV
            self.csv_writers['network_performance'].writerow({
                'timestamp': data['timestamp'],
                'measurement_type': 'traditional',
                'source_node': data['source_node'],
                'destination_node': data['destination_node'],
                'latency_ms': data['latency_seconds'] * 1000,
                'bandwidth_mbps': '',
                'packet_loss': data['packet_loss_occurred'],
                'jitter_ms': ''
            })
            
            self.data_points_collected += 1
            
        except Exception as e:
            self.logger.error(f"Error logging traditional message: {e}")
    
    async def _periodic_flush(self):
        """Periodically flush data to disk"""
        while True:
            try:
                current_time = time.time()
                if current_time - self.last_flush_time >= self.flush_interval:
                    # Flush database
                    if self.db_connection:
                        self.db_connection.commit()
                    
                    # Flush CSV files
                    for csv_file in self.csv_files.values():
                        csv_file.flush()
                    
                    self.last_flush_time = current_time
                    
                    if self.data_points_collected % 1000 == 0:
                        self.logger.debug(f"Flushed data - {self.data_points_collected} points collected")
                
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in periodic flush: {e}")
                await asyncio.sleep(5.0)
    
    async def generate_performance_comparison(self):
        """Generate comprehensive performance comparison between traditional and Sango Rine Shumba"""
        self.logger.info("Generating performance comparison analysis...")
        
        try:
            # Query traditional message performance
            traditional_results = self.db_connection.execute('''
                SELECT AVG(latency_seconds * 1000) as avg_latency_ms,
                       MIN(latency_seconds * 1000) as min_latency_ms,
                       MAX(latency_seconds * 1000) as max_latency_ms,
                       COUNT(*) as message_count,
                       SUM(packet_loss_occurred) as packet_losses
                FROM traditional_messages
            ''').fetchone()
            
            # Query Sango Rine Shumba performance
            sango_results = self.db_connection.execute('''
                SELECT AVG(mc.total_time * 1000) as avg_completion_ms,
                       MIN(mc.total_time * 1000) as min_completion_ms,
                       MAX(mc.total_time * 1000) as max_completion_ms,
                       COUNT(*) as transmission_count,
                       AVG(mc.convergence_quality) as avg_convergence_quality,
                       COUNT(CASE WHEN mc.success = 1 THEN 1 END) as successful_transmissions
                FROM mimo_completion mc
            ''').fetchone()
            
            # Calculate improvements
            comparison_data = {
                'experiment_id': self.experiment_id,
                'timestamp': time.time(),
                'traditional': {
                    'avg_latency_ms': traditional_results[0] or 0,
                    'min_latency_ms': traditional_results[1] or 0,
                    'max_latency_ms': traditional_results[2] or 0,
                    'message_count': traditional_results[3] or 0,
                    'packet_losses': traditional_results[4] or 0
                },
                'sango_rine_shumba': {
                    'avg_completion_ms': sango_results[0] or 0,
                    'min_completion_ms': sango_results[1] or 0,
                    'max_completion_ms': sango_results[2] or 0,
                    'transmission_count': sango_results[3] or 0,
                    'avg_convergence_quality': sango_results[4] or 0,
                    'success_rate': (sango_results[5] or 0) / max(1, sango_results[3] or 1)
                }
            }
            
            # Calculate improvement ratios
            if traditional_results[0] and sango_results[0]:
                comparison_data['improvements'] = {
                    'latency_reduction': 1 - (sango_results[0] / traditional_results[0]),
                    'min_latency_improvement': 1 - (sango_results[1] / traditional_results[1]) if traditional_results[1] else 0,
                    'max_latency_improvement': 1 - (sango_results[2] / traditional_results[2]) if traditional_results[2] else 0
                }
            
            # Save comparison data
            comparison_file = self.experiment_dir / "performance_comparison.json"
            with open(comparison_file, 'w') as f:
                json.dump(comparison_data, f, indent=2)
            
            self.logger.info(f"Performance comparison saved to {comparison_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating performance comparison: {e}")
    
    async def generate_statistical_summary(self):
        """Generate statistical summary of all collected data"""
        self.logger.info("Generating statistical summary...")
        
        try:
            summary = {
                'experiment_id': self.experiment_id,
                'generation_time': time.time(),
                'data_collection': {
                    'total_data_points': self.data_points_collected,
                    'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024)
                }
            }
            
            # Precision calculation statistics
            precision_stats = self.db_connection.execute('''
                SELECT COUNT(*) as measurement_count,
                       AVG(precision_difference) as avg_precision_diff,
                       STDEV(precision_difference) as std_precision_diff,
                       MIN(precision_difference) as min_precision_diff,
                       MAX(precision_difference) as max_precision_diff,
                       AVG(measurement_quality) as avg_quality,
                       AVG(confidence) as avg_confidence
                FROM precision_calculations
                WHERE measurement_quality > 0
            ''').fetchone()
            
            if precision_stats[0]:
                summary['precision_calculations'] = {
                    'measurement_count': precision_stats[0],
                    'avg_precision_difference_ms': (precision_stats[1] or 0) * 1000,
                    'std_precision_difference_ms': (precision_stats[2] or 0) * 1000,
                    'min_precision_difference_ms': (precision_stats[3] or 0) * 1000,
                    'max_precision_difference_ms': (precision_stats[4] or 0) * 1000,
                    'avg_quality': precision_stats[5] or 0,
                    'avg_confidence': precision_stats[6] or 0
                }
            
            # Fragmentation statistics
            fragmentation_stats = self.db_connection.execute('''
                SELECT COUNT(*) as message_count,
                       AVG(fragment_count) as avg_fragments,
                       AVG(average_entropy) as avg_entropy,
                       AVG(fragmentation_time) as avg_fragmentation_time,
                       AVG(overhead_ratio) as avg_overhead
                FROM message_fragmentation
            ''').fetchone()
            
            if fragmentation_stats[0]:
                summary['fragmentation'] = {
                    'messages_fragmented': fragmentation_stats[0],
                    'avg_fragment_count': fragmentation_stats[1] or 0,
                    'avg_entropy': fragmentation_stats[2] or 0,
                    'avg_fragmentation_time_ms': (fragmentation_stats[3] or 0) * 1000,
                    'avg_overhead_ratio': fragmentation_stats[4] or 0
                }
            
            # MIMO routing statistics
            mimo_stats = self.db_connection.execute('''
                SELECT COUNT(*) as transmission_count,
                       AVG(path_count) as avg_paths,
                       AVG(convergence_quality) as avg_convergence,
                       AVG(completion_ratio) as avg_completion,
                       COUNT(CASE WHEN success = 1 THEN 1 END) as successful_count
                FROM mimo_completion mc
                JOIN mimo_routing mr ON mc.transmission_id = mr.transmission_id
            ''').fetchone()
            
            if mimo_stats[0]:
                summary['mimo_routing'] = {
                    'transmission_count': mimo_stats[0],
                    'avg_path_count': mimo_stats[1] or 0,
                    'avg_convergence_quality': mimo_stats[2] or 0,
                    'avg_completion_ratio': mimo_stats[3] or 0,
                    'success_rate': (mimo_stats[4] or 0) / mimo_stats[0]
                }
            
            # Save statistical summary
            summary_file = self.experiment_dir / "statistical_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Statistical summary saved to {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating statistical summary: {e}")
    
    async def generate_publication_figures(self):
        """Generate publication-ready figures and data exports"""
        self.logger.info("Generating publication figures...")
        
        try:
            # Export data for external analysis tools
            tables = [
                'precision_calculations',
                'coordination_matrices',
                'message_fragmentation',
                'mimo_completion',
                'traditional_messages'
            ]
            
            for table in tables:
                # Export to CSV
                df = pd.read_sql_query(f"SELECT * FROM {table}", self.db_connection)
                csv_path = self.experiment_dir / "exports" / f"{table}.csv"
                df.to_csv(csv_path, index=False)
                
                # Export to JSON for web visualization
                json_path = self.experiment_dir / "exports" / f"{table}.json"
                df.to_json(json_path, orient='records', indent=2)
            
            self.logger.info("Publication data exports completed")
            
        except Exception as e:
            self.logger.error(f"Error generating publication figures: {e}")
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get data collection statistics"""
        return {
            'experiment_id': self.experiment_id,
            'data_points_collected': self.data_points_collected,
            'database_path': str(self.db_path),
            'experiment_directory': str(self.experiment_dir),
            'collection_uptime': time.time() - (getattr(self, '_start_time', time.time())),
            'last_flush_time': self.last_flush_time,
            'csv_files_count': len(self.csv_files)
        }
    
    async def log_browser_page_load(self, data: Dict[str, Any]):
        """Log browser page load event"""
        try:
            self.db_connection.execute('''
                INSERT INTO browser_page_loads
                (timestamp, session_id, page_id, node_id, method, total_time, 
                 page_size_kb, complexity_score, render_time, download_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['timestamp'],
                data['session_id'],
                data['page_id'],
                data['node_id'],
                data['method'],
                data['total_time'],
                data['page_size_kb'],
                data['complexity_score'],
                data.get('render_time', 0),
                data.get('download_time', 0)
            ))
            
            self.data_points_collected += 1
            
        except Exception as e:
            self.logger.error(f"Error logging browser page load: {e}")
    
    async def log_user_interaction(self, data: Dict[str, Any]):
        """Log user interaction event"""
        try:
            self.db_connection.execute('''
                INSERT INTO user_interactions
                (timestamp, session_id, interaction_id, interaction_type,
                 reaction_time_ms, execution_time_ms, biometric_signature)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['timestamp'],
                data['session_id'],
                data['interaction_id'],
                data['interaction_type'],
                data['reaction_time_ms'],
                data['execution_time_ms'],
                data['biometric_signature']
            ))
            
            self.data_points_collected += 1
            
        except Exception as e:
            self.logger.error(f"Error logging user interaction: {e}")
    
    async def log_biometric_verification(self, data: Dict[str, Any]):
        """Log biometric verification event"""
        try:
            biometric_features_json = json.dumps(data.get('biometric_features', {}))
            
            self.db_connection.execute('''
                INSERT INTO biometric_verifications
                (timestamp, session_id, user_id, interaction_id, verification_method,
                 verified, confidence, verification_time_ms, biometric_features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['timestamp'],
                data.get('session_id', ''),
                data['user_id'],
                data['interaction_id'],
                data['verification_method'],
                1 if data['verified'] else 0,
                data['confidence'],
                data['verification_time_ms'],
                biometric_features_json
            ))
            
            self.data_points_collected += 1
            
        except Exception as e:
            self.logger.error(f"Error logging biometric verification: {e}")
    
    async def log_zero_latency_event(self, data: Dict[str, Any]):
        """Log zero-latency response event"""
        try:
            response_data_json = json.dumps(data.get('response_data', {}))
            
            self.db_connection.execute('''
                INSERT INTO zero_latency_events
                (timestamp, session_id, interaction_id, interaction_type,
                 predicted, response_time_ms, response_data, user_satisfaction_boost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['timestamp'],
                data['session_id'],
                data['interaction_id'],
                data['interaction_type'],
                1 if data['predicted'] else 0,
                data['response_time_ms'],
                response_data_json,
                data['user_satisfaction_boost']
            ))
            
            self.data_points_collected += 1
            
        except Exception as e:
            self.logger.error(f"Error logging zero-latency event: {e}")
    
    async def log_browser_comparison_report(self, data: Dict[str, Any]):
        """Log browser comparison report"""
        try:
            perf_summary = data['performance_summary']
            improvements = data['improvements']
            
            self.db_connection.execute('''
                INSERT INTO browser_comparison_reports
                (timestamp, traditional_avg_load_time, sango_avg_load_time,
                 load_time_improvement, user_experience_improvement,
                 zero_latency_events, biometric_verifications)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['timestamp'],
                perf_summary['traditional']['avg_load_time'],
                perf_summary['sango_rine_shumba']['avg_load_time'],
                improvements['load_time_reduction'],
                improvements['user_experience_improvement'],
                improvements['zero_latency_events'],
                improvements['biometric_verifications']
            ))
            
            self.data_points_collected += 1
            
        except Exception as e:
            self.logger.error(f"Error logging browser comparison report: {e}")
    
    async def log_predictive_response(self, data: Dict[str, Any]):
        """Log predictive response event (alias for zero_latency_event)"""
        await self.log_zero_latency_event(data)
    
    async def log_interaction_analysis(self, data: Dict[str, Any]):
        """Log interaction analysis results"""
        try:
            # Store as JSON for complex analysis data
            analysis_file = self.experiment_dir / "interaction_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.debug(f"Interaction analysis saved to {analysis_file}")
            
        except Exception as e:
            self.logger.error(f"Error logging interaction analysis: {e}")

    async def close(self):
        """Close data collector and flush remaining data"""
        self.logger.info("Closing data collector...")
        
        # Final flush
        if self.db_connection:
            self.db_connection.commit()
            self.db_connection.close()
        
        # Close CSV files
        for csv_file in self.csv_files.values():
            csv_file.close()
        
        self.logger.info(f"Data collection completed - {self.data_points_collected} total data points")
