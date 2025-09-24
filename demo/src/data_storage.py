# data_storage.py - Comprehensive data persistence
import asyncio
import sqlite3
import json
import pickle
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone


@dataclass
class DataStorageConfig:
    """Configuration for data storage system"""
    database_path: str = "sango_rine_shumba.db"
    json_backup_path: str = "data_backups"
    csv_export_path: str = "csv_exports"
    enable_real_time_backup: bool = True
    backup_interval_seconds: int = 300  # 5 minutes
    max_memory_cache_size: int = 10000


class SangoDataStorage:
    """Comprehensive data storage for all Sango Rine Shumba measurements"""

    def __init__(self, config: DataStorageConfig = None):
        self.config = config or DataStorageConfig()
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(self.config.database_path)
        self.backup_path = Path(self.config.json_backup_path)
        self.csv_path = Path(self.config.csv_export_path)

        # Create directories
        self.backup_path.mkdir(exist_ok=True)
        self.csv_path.mkdir(exist_ok=True)

        # Memory cache for real-time access
        self.memory_cache: Dict[str, List[Dict]] = {
            'atomic_measurements': [],
            'precision_calculations': [],
            'network_latencies': [],
            'web_performance': [],
            'temporal_fragments': [],
            'node_states': []
        }

        self._initialize_database()

    def _initialize_database(self):
        """Initialize SQLite database with all required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Atomic clock measurements
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS atomic_measurements
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               timestamp
                               REAL
                               NOT
                               NULL,
                               source_name
                               TEXT
                               NOT
                               NULL,
                               atomic_time
                               REAL
                               NOT
                               NULL,
                               precision_level
                               TEXT,
                               accuracy_seconds
                               REAL,
                               sync_status
                               TEXT,
                               network_latency_ms
                               REAL,
                               measurement_quality
                               REAL,
                               metadata
                               TEXT
                           )
                           ''')

            # Precision by difference calculations
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS precision_calculations
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               timestamp
                               REAL
                               NOT
                               NULL,
                               node_id
                               TEXT
                               NOT
                               NULL,
                               atomic_reference
                               REAL
                               NOT
                               NULL,
                               local_measurement
                               REAL
                               NOT
                               NULL,
                               precision_difference
                               REAL
                               NOT
                               NULL,
                               measurement_quality
                               REAL,
                               confidence_interval_lower
                               REAL,
                               confidence_interval_upper
                               REAL,
                               standard_deviation
                               REAL,
                               environmental_factors
                               TEXT
                           )
                           ''')[[0]](  # __0)

                # Network topology and latencies
                cursor.execute('''
                               CREATE TABLE IF NOT EXISTS network_measurements
                               (
                                   id
                                   INTEGER
                                   PRIMARY
                                   KEY
                                   AUTOINCREMENT,
                                   timestamp
                                   REAL
                                   NOT
                                   NULL,
                                   source_node_id
                                   TEXT
                                   NOT
                                   NULL,
                                   destination_node_id
                                   TEXT
                                   NOT
                                   NULL,
                                   geographic_distance_km
                                   REAL,
                                   calculated_latency_ms
                                   REAL,
                                   measured_latency_ms
                                   REAL,
                                   packet_loss_rate
                                   REAL,
                                   bandwidth_mbps
                                   REAL,
                                   infrastructure_type
                                   TEXT,
                                   weather_factor
                                   REAL,
                                   grid_interference
                                   REAL
                               )
                               ''')

            # Web performance measurements
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS web_performance
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               timestamp
                               REAL
                               NOT
                               NULL,
                               page_id
                               TEXT
                               NOT
                               NULL,
                               url
                               TEXT
                               NOT
                               NULL,
                               loading_method
                               TEXT
                               NOT
                               NULL,
                               total_load_time_ms
                               REAL,
                               dns_time_ms
                               REAL,
                               tcp_time_ms
                               REAL,
                               html_time_ms
                               REAL,
                               css_time_ms
                               REAL,
                               js_time_ms
                               REAL,
                               image_time_ms
                               REAL,
                               paint_time_ms
                               REAL,
                               improvement_percentage
                               REAL,
                               page_size_kb
                               REAL,
                               complexity_score
                               REAL
                           )
                           ''')

            # Temporal fragment coordination
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS temporal_fragments
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               timestamp
                               REAL
                               NOT
                               NULL,
                               fragment_id
                               TEXT
                               NOT
                               NULL,
                               source_node
                               TEXT
                               NOT
                               NULL,
                               destination_node
                               TEXT
                               NOT
                               NULL,
                               fragment_type
                               TEXT,
                               temporal_offset_ms
                               REAL,
                               coordination_success
                               BOOLEAN,
                               delivery_time_ms
                               REAL,
                               preemptive_advantage_ms
                               REAL,
                               synchronization_error_ms
                               REAL
                           )
                           ''')

            # Node geographical and state information
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS node_states
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               timestamp
                               REAL
                               NOT
                               NULL,
                               node_id
                               TEXT
                               NOT
                               NULL,
                               latitude
                               REAL,
                               longitude
                               REAL,
                               timezone
                               TEXT,
                               infrastructure_type
                               TEXT,
                               current_load
                               REAL,
                               active_connections
                               INTEGER,
                               power_grid_frequency
                               INTEGER,
                               weather_conditions
                               TEXT,
                               operational_status
                               TEXT
                           )
                           ''')

            conn.commit()
            self.logger.info("Database initialized successfully")

            async

            def store_atomic_measurement(self, measurement_data: Dict[str, Any]):

                """Store atomic clock measurement with full metadata"""
            timestamp = time.time()

            # Add to memory cache
            cache_entry = {
                'timestamp': timestamp,
                **measurement_data
            }
            self.memory_cache['atomic_measurements'].append(cache_entry)

            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                               INSERT INTO atomic_measurements
                               (timestamp, source_name, atomic_time, precision_level, accuracy_seconds,
                                sync_status, network_latency_ms, measurement_quality, metadata)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                               ''', (
                                   timestamp,
                                   measurement_data.get('source_name', ''),
                                   measurement_data.get('atomic_time', 0.0),
                                   measurement_data.get('precision_level', ''),
                                   measurement_data.get('accuracy_seconds', 0.0),
                                   measurement_data.get('sync_status', ''),
                                   measurement_data.get('network_latency_ms', 0.0),
                                   measurement_data.get('measurement_quality', 0.0),
                                   json.dumps(measurement_data.get('metadata', {}))
                               ))
                conn.commit()

        async def store_precision_calculation(self, calculation_data: Dict[str, Any]):
            """Store precision-by-difference calculation"""
            timestamp = time.time()

            cache_entry = {
                'timestamp': timestamp,
                **calculation_data
            }
            self.memory_cache['precision_calculations'].append(cache_entry)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                               INSERT INTO precision_calculations
                               (timestamp, node_id, atomic_reference, local_measurement, precision_difference,
                                measurement_quality, confidence_interval_lower, confidence_interval_upper,
                                standard_deviation, environmental_factors)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                               ''', (
                                   timestamp,
                                   calculation_data.get('node_id', ''),
                                   calculation_data.get('atomic_reference', 0.0),
                                   calculation_data.get('local_measurement', 0.0),
                                   calculation_data.get('precision_difference', 0.0),
                                   calculation_data.get('measurement_quality', 0.0),
                                   calculation_data.get('confidence_interval', (0.0, 0.0))[0],
                                   calculation_data.get('confidence_interval', (0.0, 0.0))[1],
                                   calculation_data.get('standard_deviation', 0.0),
                                   json.dumps(calculation_data.get('environmental_factors', {}))
                               ))
                conn.commit()

        def export_to_csv(self, table_name: str, time_range: Optional[Tuple[float, float]] = None) -> Path:
            """Export data to CSV for analysis"""
            with sqlite3.connect(self.db_path) as conn:
                if time_range:
                    query = f"SELECT * FROM {table_name} WHERE timestamp BETWEEN ? AND ?"
                    df = pd.read_sql_query(query, conn, params=time_range)
                else:
                    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = self.csv_path / f"{table_name}_{timestamp_str}.csv"
            df.to_csv(csv_file, index=False)

            self.logger.info(f"Exported {len(df)} records to {csv_file}")
            return csv_file
