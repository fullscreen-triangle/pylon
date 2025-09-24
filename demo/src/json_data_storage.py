# json_data_storage.py - Lightweight JSON-based storage for validation experiments
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class ExperimentConfig:
    """Configuration for validation experiment"""
    data_directory: str = "experiment_data"
    enable_real_time_cache: bool = True
    max_cache_size: int = 1000
    auto_save_interval: int = 60  # seconds


class SangoJSONStorage:
    """Lightweight JSON storage for Sango Rine Shumba validation experiment"""

    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(self.config.data_directory)
        self.data_dir.mkdir(exist_ok=True)

        # In-memory data structure for fast access during experiment
        self.experiment_data = {
            'metadata': {
                'experiment_start': time.time(),
                'framework_version': '1.0.0-validation',
                'description': 'Sango Rine Shumba temporal coordination validation'
            },
            'atomic_measurements': [],
            'precision_calculations': [],
            'network_measurements': [],
            'web_performance': [],
            'temporal_fragments': [],
            'node_states': []
        }

        # Load existing data if available
        self._load_existing_data()

        # Start auto-save task
        if self.config.enable_real_time_cache:
            asyncio.create_task(self._auto_save_loop())

    def _load_existing_data(self):
        """Load existing experiment data if available"""
        data_file = self.data_dir / "experiment_data.json"
        if data_file.exists():
            try:
                with open(data_file, 'r') as f:
                    existing_data = json.load(f)
                    # Merge with current structure, keeping existing data
                    for key in self.experiment_data.keys():
                        if key in existing_data:
                            self.experiment_data[key] = existing_data[key]
                self.logger.info(f"Loaded existing experiment data from {data_file}")
            except Exception as e:
                self.logger.warning(f"Could not load existing data: {e}")

    async def _auto_save_loop(self):
        """Periodically save data to prevent loss"""
        while True:
            try:
                await asyncio.sleep(self.config.auto_save_interval)
                self.save_experiment_data()
            except Exception as e:
                self.logger.error(f"Auto-save error: {e}")

    def add_atomic_measurement(self, measurement_data: Dict[str, Any]):
        """Add atomic clock measurement to experiment data"""
        entry = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            **measurement_data
        }

        self.experiment_data['atomic_measurements'].append(entry)

        # Keep cache size manageable
        if len(self.experiment_data['atomic_measurements']) > self.config.max_cache_size:
            self.experiment_data['atomic_measurements'] = \
                self.experiment_data['atomic_measurements'][-self.config.max_cache_size:]

    def add_precision_calculation(self, calculation_data: Dict[str, Any]):
        """Add precision calculation to experiment data"""
        entry = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            **calculation_data
        }

        self.experiment_data['precision_calculations'].append(entry)

        if len(self.experiment_data['precision_calculations']) > self.config.max_cache_size:
            self.experiment_data['precision_calculations'] = \
                self.experiment_data['precision_calculations'][-self.config.max_cache_size:]

    def add_network_measurement(self, network_data: Dict[str, Any]):
        """Add network measurement to experiment data"""
        entry = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            **network_data
        }

        self.experiment_data['network_measurements'].append(entry)

        if len(self.experiment_data['network_measurements']) > self.config.max_cache_size:
            self.experiment_data['network_measurements'] = \
                self.experiment_data['network_measurements'][-self.config.max_cache_size:]

    def add_web_performance(self, performance_data: Dict[str, Any]):
        """Add web performance measurement to experiment data"""
        entry = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            **performance_data
        }

        self.experiment_data['web_performance'].append(entry)

        if len(self.experiment_data['web_performance']) > self.config.max_cache_size:
            self.experiment_data['web_performance'] = \
                self.experiment_data['web_performance'][-self.config.max_cache_size:]

    def save_experiment_data(self, filename: Optional[str] = None):
        """Save all experiment data to JSON file"""
        if filename is None:
            filename = f"experiment_data.json"

        file_path = self.data_dir / filename

        # Add current timestamp to metadata
        self.experiment_data['metadata']['last_saved'] = time.time()
        self.experiment_data['metadata']['total_measurements'] = sum([
            len(self.experiment_data['atomic_measurements']),
            len(self.experiment_data['precision_calculations']),
            len(self.experiment_data['network_measurements']),
            len(self.experiment_data['web_performance'])
        ])

        try:
            with open(file_path, 'w') as f:
                json.dump(self.experiment_data, f, indent=2, default=self._json_serializer)

            self.logger.info(f"Experiment data saved to {file_path}")
            return file_path

        except Exception as e:
            self.logger.error(f"Failed to save experiment data: {e}")
            raise

    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types and other objects"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return str(obj)

    def export_summary_report(self) -> Dict[str, Any]:
        """Generate and save summary report of experiment"""
        summary = {
            'experiment_summary': {
                'start_time': self.experiment_data['metadata']['experiment_start'],
                'duration_hours': (time.time() - self.experiment_data['metadata']['experiment_start']) / 3600,
                'total_atomic_measurements': len(self.experiment_data['atomic_measurements']),
                'total_precision_calculations': len(self.experiment_data['precision_calculations']),
                'total_network_measurements': len(self.experiment_data['network_measurements']),
                'total_web_performance_tests': len(self.experiment_data['web_performance'])
            },
            'key_findings': self._calculate_key_findings(),
            'data_quality': self._assess_data_quality()
        }

        # Save summary report
        summary_file = self.data_dir / f"experiment_summary_{int(time.time())}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=self._json_serializer)

        return summary[[0]](0)

    def _calculate_key_findings(self) -> Dict[str, Any]:
        """Calculate key experimental findings from collected data"""
        findings = {}

        # Precision analysis
        if self.experiment_data['precision_calculations']:
            precision_diffs = [p.get('precision_difference', 0) for p in self.experiment_data['precision_calculations']]
            findings['precision_analysis'] = {
                'average_precision_difference_ms': np.mean(precision_diffs) * 1000,
                'std_deviation_ms': np.std(precision_diffs) * 1000,
                'min_precision_ms': np.min(precision_diffs) * 1000,
                'max_precision_ms': np.max(precision_diffs) * 1000,
                'measurements_within_1ms': sum(1 for p in precision_diffs if abs(p) < 0.001)
            }

        # Network performance analysis
        if self.experiment_data['network_measurements']:
            latencies = [n.get('measured_latency_ms', 0) for n in self.experiment_data['network_measurements']]
            findings['network_analysis'] = {
                'average_latency_ms': np.mean(latencies),
                'latency_std_deviation': np.std(latencies),
                'min_latency_ms': np.min(latencies),
                'max_latency_ms': np.max(latencies)
            }

        # Web performance comparison
        if self.experiment_data['web_performance']:
            traditional_loads = [w for w in self.experiment_data['web_performance']
                                 if w.get('loading_method') == 'traditional']
            sango_loads = [w for w in self.experiment_data['web_performance']
                           if w.get('loading_method') == 'sango_streaming']

            if traditional_loads and sango_loads:
                trad_times = [t.get('total_load_time_ms', 0) for t in traditional_loads]
                sango_times = [s.get('total_load_time_ms', 0) for s in sango_loads]

                findings['web_performance_comparison'] = {
                    'traditional_avg_ms': np.mean(trad_times),
                    'sango_avg_ms': np.mean(sango_times),
                    'average_improvement_percentage': ((np.mean(trad_times) - np.mean(sango_times)) / np.mean(
                        trad_times)) * 100 if np.mean(trad_times) > 0 else 0,
                    'traditional_tests': len(traditional_loads),
                    'sango_tests': len(sango_loads)
                }

        return findings

    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess the quality and completeness of collected data"""
        quality = {
            'data_completeness': {},
            'measurement_consistency': {},
            'temporal_coverage': {}
        }

        # Check data completeness
        for data_type in ['atomic_measurements', 'precision_calculations', 'network_measurements', 'web_performance']:
            data_count = len(self.experiment_data[data_type])
            quality['data_completeness'][data_type] = {
                'count': data_count,
                'sufficient': data_count >= 10  # Minimum threshold for validation
            }

        # Check temporal coverage
        if self.experiment_data['precision_calculations']:
            timestamps = [p['timestamp'] for p in self.experiment_data['precision_calculations']]
            quality['temporal_coverage'] = {
                'start_time': min(timestamps),
                'end_time': max(timestamps),
                'duration_minutes': (max(timestamps) - min(timestamps)) / 60,
                'measurement_frequency_per_minute': len(timestamps) / ((max(timestamps) - min(timestamps)) / 60) if len(
                    timestamps) > 1 else 0
            }

        return quality
