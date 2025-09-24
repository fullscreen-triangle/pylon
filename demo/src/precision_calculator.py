# precision_calculator.py - Fixed version
import asyncio
import time
import math
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PNG output
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path
import json


@dataclass
class PrecisionMeasurement:
    """Complete precision-by-difference measurement"""

    node_id: str
    measurement_time: float
    atomic_reference: float
    local_measurement: float
    precision_difference: float
    measurement_quality: float

    # Statistical metadata
    confidence_interval: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    standard_deviation: float = 0.0
    measurement_count: int = 1

    # Environmental factors
    network_conditions: str = "normal"
    temperature_factor: float = 1.0

    def __post_init__(self):
        """Validate measurement data"""
        if self.measurement_quality < 0.0 or self.measurement_quality > 1.0:
            raise ValueError("Measurement quality must be between 0.0 and 1.0")
        if abs(self.precision_difference) > 1.0:  # More than 1 second difference is suspicious
            logging.warning(f"Large precision difference detected: {self.precision_difference}")


class PrecisionCalculator:
    """Enhanced precision-by-difference calculator with statistical validation"""

    def __init__(self, window_size: int = 100):
        self.logger = logging.getLogger(__name__)
        self.window_size = window_size
        self.measurements: Dict[str, deque] = {}
        self.precision_history: Dict[str, List[float]] = {}
        self.atomic_clock_service = None

    async def calculate_precision_by_difference(
            self,
            node_id: str,
            local_time: float,
            quality_threshold: float = 0.8
    ) -> PrecisionMeasurement:
        """Calculate ΔP_i(k) = T_ref(k) - t_i(k) with validation"""

        try:
            # Get atomic reference
            if not self.atomic_clock_service:
                raise RuntimeError("Atomic clock service not initialized")

            atomic_ref = await self.atomic_clock_service.get_atomic_reference()

            # Calculate precision difference
            precision_diff = atomic_ref - local_time

            # Assess measurement quality
            quality = self._assess_measurement_quality(node_id, precision_diff)

            # Create measurement
            measurement = PrecisionMeasurement(
                node_id=node_id,
                measurement_time=time.time(),
                atomic_reference=atomic_ref,
                local_measurement=local_time,
                precision_difference=precision_diff,
                measurement_quality=quality
            )

            # Store measurement
            self._store_measurement(measurement)

            # Calculate statistical properties
            self._update_statistics(measurement)

            return measurement

        except Exception as e:
            self.logger.error(f"Precision calculation failed for {node_id}: {e}")
            # Return fallback measurement
            return PrecisionMeasurement(
                node_id=node_id,
                measurement_time=time.time(),
                atomic_reference=local_time,
                local_measurement=local_time,
                precision_difference=0.0,
                measurement_quality=0.0
            )

    def _assess_measurement_quality(self, node_id: str, precision_diff: float) -> float:
        """Assess measurement quality based on historical data"""
        if node_id not in self.precision_history:
            return 0.5  # Default quality for first measurement

        history = self.precision_history[node_id]
        if len(history) < 5:
            return 0.6

        # Calculate quality based on consistency with recent measurements
        recent_mean = statistics.mean(history[-10:])
        recent_std = statistics.stdev(history[-10:]) if len(history) > 1 else 0.1

        # Quality decreases with distance from recent mean
        deviation = abs(precision_diff - recent_mean)
        quality = max(0.0, 1.0 - (deviation / (3 * recent_std + 0.001)))

        return min(1.0, quality)

    def _store_measurement(self, measurement: PrecisionMeasurement):
        """Store measurement in history"""
        node_id = measurement.node_id
        
        # Initialize deque for node if not exists
        if node_id not in self.measurements:
            self.measurements[node_id] = deque(maxlen=self.window_size)
        
        if node_id not in self.precision_history:
            self.precision_history[node_id] = []
        
        # Store measurement
        self.measurements[node_id].append(measurement)
        self.precision_history[node_id].append(measurement.precision_difference)
        
        # Keep only recent history
        if len(self.precision_history[node_id]) > self.window_size:
            self.precision_history[node_id] = self.precision_history[node_id][-self.window_size:]

    def _update_statistics(self, measurement: PrecisionMeasurement):
        """Update statistical properties of measurement"""
        node_id = measurement.node_id
        
        if node_id in self.precision_history and len(self.precision_history[node_id]) > 1:
            history = self.precision_history[node_id]
            
            # Calculate confidence interval (95%)
            mean = statistics.mean(history)
            std = statistics.stdev(history) if len(history) > 1 else 0.0
            margin = 1.96 * std / math.sqrt(len(history))  # 95% CI
            
            measurement.confidence_interval = (mean - margin, mean + margin)
            measurement.standard_deviation = std
            measurement.measurement_count = len(history)

    def set_atomic_clock_service(self, atomic_clock_service):
        """Set atomic clock service reference"""
        self.atomic_clock_service = atomic_clock_service

    def get_current_coordination_matrix(self):
        """Get current coordination matrix for temporal operations"""
        current_time = time.time()
        
        # Create coordination matrix based on recent measurements
        coordination_data = {
            'temporal_window_start': current_time - 0.1,  # 100ms window
            'temporal_window_end': current_time + 0.1,
            'temporal_window_duration': 0.2,  # 200ms total
            'measurement_timestamp': current_time,
            'active_nodes': list(self.measurements.keys()),
            'coordination_quality': self._calculate_coordination_quality()
        }
        
        # Return as simple object with attributes
        class CoordinationMatrix:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        return CoordinationMatrix(coordination_data)

    def _calculate_coordination_quality(self) -> float:
        """Calculate overall coordination quality across all nodes"""
        if not self.measurements:
            return 0.0
        
        quality_scores = []
        for node_measurements in self.measurements.values():
            if node_measurements:
                recent_measurement = list(node_measurements)[-1]
                quality_scores.append(recent_measurement.measurement_quality)
        
        return statistics.mean(quality_scores) if quality_scores else 0.0

    async def start_continuous_calculation(self):
        """Start continuous precision calculation service"""
        self.logger.info("Starting continuous precision calculation service...")
        
        while True:
            try:
                # Calculate precision for all known nodes
                for node_id in list(self.measurements.keys()):
                    try:
                        local_time = time.time()
                        measurement = await self.calculate_precision_by_difference(node_id, local_time)
                        
                        self.logger.debug(f"Continuous calculation for {node_id}: "
                                        f"{measurement.precision_difference * 1000:.3f}ms")
                        
                    except Exception as e:
                        self.logger.warning(f"Continuous calculation failed for {node_id}: {e}")
                
                # Sleep between calculations
                await asyncio.sleep(1.0)  # 1 second interval
                
            except asyncio.CancelledError:
                self.logger.info("Continuous precision calculation stopped")
                break
            except Exception as e:
                self.logger.error(f"Error in continuous precision calculation: {e}")
                await asyncio.sleep(5.0)  # Longer sleep on error

    def get_precision_statistics(self) -> Dict[str, Any]:
        """Get comprehensive precision statistics"""
        stats = {
            'total_nodes': len(self.measurements),
            'total_measurements': sum(len(measurements) for measurements in self.measurements.values()),
            'node_statistics': {},
            'overall_statistics': {}
        }
        
        all_precisions = []
        all_qualities = []
        
        # Per-node statistics
        for node_id, node_measurements in self.measurements.items():
            if not node_measurements:
                continue
                
            measurements_list = list(node_measurements)
            precisions = [m.precision_difference for m in measurements_list]
            qualities = [m.measurement_quality for m in measurements_list]
            
            all_precisions.extend(precisions)
            all_qualities.extend(qualities)
            
            node_stats = {
                'measurement_count': len(measurements_list),
                'mean_precision_ms': statistics.mean(precisions) * 1000,
                'std_precision_ms': statistics.stdev(precisions) * 1000 if len(precisions) > 1 else 0.0,
                'min_precision_ms': min(precisions) * 1000,
                'max_precision_ms': max(precisions) * 1000,
                'mean_quality': statistics.mean(qualities),
                'last_measurement_time': measurements_list[-1].measurement_time
            }
            
            stats['node_statistics'][node_id] = node_stats
        
        # Overall statistics
        if all_precisions:
            stats['overall_statistics'] = {
                'mean_precision_ms': statistics.mean(all_precisions) * 1000,
                'std_precision_ms': statistics.stdev(all_precisions) * 1000 if len(all_precisions) > 1 else 0.0,
                'min_precision_ms': min(all_precisions) * 1000,
                'max_precision_ms': max(all_precisions) * 1000,
                'mean_quality': statistics.mean(all_qualities),
                'coordination_quality': self._calculate_coordination_quality()
            }
        
        return stats

    def export_precision_data(self, output_dir: Path = Path("output")) -> Dict[str, Any]:
        """Export precision calculation data"""
        output_dir.mkdir(exist_ok=True)
        
        # Collect all measurement data
        export_data = {
            'precision_statistics': self.get_precision_statistics(),
            'detailed_measurements': {},
            'export_timestamp': time.time()
        }
        
        # Export detailed measurements for each node
        for node_id, node_measurements in self.measurements.items():
            measurements_data = []
            for measurement in node_measurements:
                measurement_data = {
                    'timestamp': measurement.measurement_time,
                    'atomic_reference': measurement.atomic_reference,
                    'local_measurement': measurement.local_measurement,
                    'precision_difference': measurement.precision_difference,
                    'measurement_quality': measurement.measurement_quality,
                    'confidence_interval': measurement.confidence_interval,
                    'standard_deviation': measurement.standard_deviation
                }
                measurements_data.append(measurement_data)
            
            export_data['detailed_measurements'][node_id] = measurements_data
        
        # Save to JSON
        json_file = output_dir / "precision_calculation_data.json"
        with open(json_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Precision calculation data exported to {json_file}")
        return export_data

    def create_precision_visualization(self, output_dir: Path = Path("output")):
        """Create precision calculation visualizations"""
        output_dir.mkdir(exist_ok=True)
        
        if not self.measurements:
            self.logger.warning("No measurements available for visualization")
            return
        
        self._create_precision_time_series(output_dir)
        self._create_precision_distribution(output_dir)
        self._create_quality_analysis(output_dir)
        self._create_coordination_matrix_plot(output_dir)

    def _create_precision_time_series(self, output_dir: Path):
        """Create precision difference time series plot"""
        try:
            plt.figure(figsize=(15, 10))
            
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            for i, (node_id, node_measurements) in enumerate(self.measurements.items()):
                if not node_measurements:
                    continue
                
                measurements_list = list(node_measurements)
                timestamps = [m.measurement_time for m in measurements_list]
                precisions = [m.precision_difference * 1000 for m in measurements_list]  # Convert to ms
                
                # Normalize timestamps to start from 0
                if timestamps:
                    start_time = min(timestamps)
                    relative_times = [(t - start_time) for t in timestamps]
                    
                    color = colors[i % len(colors)]
                    plt.plot(relative_times, precisions, 'o-', color=color, label=node_id, 
                           markersize=4, linewidth=1, alpha=0.7)
            
            plt.xlabel('Time (seconds)')
            plt.ylabel('Precision Difference (ms)')
            plt.title('Precision-by-Difference Time Series')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            timeseries_file = output_dir / "precision_time_series.png"
            plt.savefig(timeseries_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Precision time series plot saved to {timeseries_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating precision time series plot: {e}")

    def _create_precision_distribution(self, output_dir: Path):
        """Create precision difference distribution plot"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            all_precisions = []
            for node_measurements in self.measurements.values():
                precisions = [m.precision_difference * 1000 for m in node_measurements]
                all_precisions.extend(precisions)
            
            if not all_precisions:
                return
            
            # Overall distribution histogram
            axes[0, 0].hist(all_precisions, bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].set_xlabel('Precision Difference (ms)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Overall Precision Distribution')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Box plot by node
            node_data = []
            node_labels = []
            for node_id, node_measurements in self.measurements.items():
                if node_measurements:
                    precisions = [m.precision_difference * 1000 for m in node_measurements]
                    node_data.append(precisions)
                    node_labels.append(node_id)
            
            if node_data:
                axes[0, 1].boxplot(node_data, labels=node_labels)
                axes[0, 1].set_ylabel('Precision Difference (ms)')
                axes[0, 1].set_title('Precision by Node')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Statistical summary
            axes[1, 0].text(0.1, 0.8, f"Mean: {np.mean(all_precisions):.3f} ms", 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].text(0.1, 0.7, f"Std Dev: {np.std(all_precisions):.3f} ms", 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].text(0.1, 0.6, f"Min: {np.min(all_precisions):.3f} ms", 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].text(0.1, 0.5, f"Max: {np.max(all_precisions):.3f} ms", 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].text(0.1, 0.4, f"Total Measurements: {len(all_precisions)}", 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('Statistical Summary')
            axes[1, 0].axis('off')
            
            # Quality scores over time
            all_qualities = []
            all_times = []
            for node_measurements in self.measurements.values():
                for measurement in node_measurements:
                    all_qualities.append(measurement.measurement_quality)
                    all_times.append(measurement.measurement_time)
            
            if all_times:
                start_time = min(all_times)
                relative_times = [(t - start_time) for t in all_times]
                axes[1, 1].scatter(relative_times, all_qualities, alpha=0.6, color='green')
                axes[1, 1].set_xlabel('Time (seconds)')
                axes[1, 1].set_ylabel('Measurement Quality')
                axes[1, 1].set_title('Measurement Quality Over Time')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            distribution_file = output_dir / "precision_distribution.png"
            plt.savefig(distribution_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Precision distribution plot saved to {distribution_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating precision distribution plot: {e}")

    def _create_quality_analysis(self, output_dir: Path):
        """Create measurement quality analysis plot"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Quality by node
            node_qualities = {}
            for node_id, node_measurements in self.measurements.items():
                if node_measurements:
                    qualities = [m.measurement_quality for m in node_measurements]
                    node_qualities[node_id] = np.mean(qualities)
            
            if node_qualities:
                nodes = list(node_qualities.keys())
                qualities = list(node_qualities.values())
                
                colors = ['green' if q >= 0.8 else 'orange' if q >= 0.6 else 'red' for q in qualities]
                bars = ax1.bar(range(len(nodes)), qualities, color=colors, alpha=0.7)
                ax1.set_xlabel('Nodes')
                ax1.set_ylabel('Average Quality Score')
                ax1.set_title('Average Measurement Quality by Node')
                ax1.set_xticks(range(len(nodes)))
                ax1.set_xticklabels(nodes, rotation=45)
                ax1.set_ylim(0, 1.1)
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.2f}', ha='center', va='bottom')
            
            # Quality distribution
            all_qualities = []
            for node_measurements in self.measurements.values():
                qualities = [m.measurement_quality for m in node_measurements]
                all_qualities.extend(qualities)
            
            if all_qualities:
                ax2.hist(all_qualities, bins=20, alpha=0.7, color='purple', edgecolor='black')
                ax2.axvline(x=np.mean(all_qualities), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(all_qualities):.3f}')
                ax2.set_xlabel('Quality Score')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Quality Score Distribution')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
            
            plt.tight_layout()
            
            quality_file = output_dir / "quality_analysis.png"
            plt.savefig(quality_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Quality analysis plot saved to {quality_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating quality analysis plot: {e}")

    def _create_coordination_matrix_plot(self, output_dir: Path):
        """Create coordination matrix visualization"""
        try:
            coordination_matrix = self.get_current_coordination_matrix()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Temporal window visualization
            current_time = time.time()
            window_start = coordination_matrix.temporal_window_start
            window_end = coordination_matrix.temporal_window_end
            
            # Show temporal window
            ax1.axvspan(window_start - current_time, window_end - current_time, 
                       alpha=0.3, color='blue', label='Coordination Window')
            ax1.axvline(x=0, color='red', linestyle='--', label='Current Time')
            ax1.set_xlabel('Time Offset (seconds)')
            ax1.set_ylabel('Coordination Activity')
            ax1.set_title('Temporal Coordination Window')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Node coordination status
            if self.measurements:
                nodes = list(self.measurements.keys())
                coordination_scores = []
                
                for node_id in nodes:
                    if self.measurements[node_id]:
                        recent_measurement = list(self.measurements[node_id])[-1]
                        coordination_scores.append(recent_measurement.measurement_quality)
                    else:
                        coordination_scores.append(0.0)
                
                colors = ['green' if s >= 0.8 else 'orange' if s >= 0.6 else 'red' for s in coordination_scores]
                bars = ax2.bar(range(len(nodes)), coordination_scores, color=colors, alpha=0.7)
                ax2.set_xlabel('Nodes')
                ax2.set_ylabel('Coordination Score')
                ax2.set_title('Node Coordination Status')
                ax2.set_xticks(range(len(nodes)))
                ax2.set_xticklabels(nodes, rotation=45)
                ax2.set_ylim(0, 1.1)
                ax2.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            coordination_file = output_dir / "coordination_matrix.png"
            plt.savefig(coordination_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Coordination matrix plot saved to {coordination_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating coordination matrix plot: {e}")


# Standalone execution capability
async def main():
    """Standalone execution of precision calculator"""
    print("📊 Precision Calculator - Standalone Test")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Create output directory
        output_dir = Path("precision_calculator_output")
        output_dir.mkdir(exist_ok=True)
        
        logger.info("Initializing precision calculator...")
        
        # Create precision calculator
        precision_calc = PrecisionCalculator(window_size=50)
        
        # Import atomic clock for testing
        try:
            from atomic_clock import AtomicClockService
            async with AtomicClockService() as atomic_clock:
                await atomic_clock.initialize()
                precision_calc.set_atomic_clock_service(atomic_clock)
                
                logger.info("Running precision calculation tests...")
                
                # Simulate multiple nodes
                test_nodes = ['tokyo', 'london', 'harare', 'sydney', 'lapaz']
                
                # Collect measurements over time
                for measurement_round in range(20):  # 20 rounds of measurements
                    for node_id in test_nodes:
                        try:
                            # Simulate slightly different local times
                            local_time = time.time() + random.uniform(-0.002, 0.002)  # ±2ms variation
                            
                            measurement = await precision_calc.calculate_precision_by_difference(
                                node_id, local_time
                            )
                            
                            if measurement_round % 5 == 0:  # Log every 5th round
                                precision_ms = measurement.precision_difference * 1000
                                logger.info(f"Round {measurement_round+1}, {node_id}: "
                                          f"{precision_ms:.3f}ms, quality: {measurement.measurement_quality:.3f}")
                            
                        except Exception as e:
                            logger.warning(f"Measurement failed for {node_id} in round {measurement_round+1}: {e}")
                    
                    await asyncio.sleep(0.2)  # 200ms between rounds
                
                # Get coordination matrix
                coordination_matrix = precision_calc.get_current_coordination_matrix()
                logger.info(f"Coordination window: {coordination_matrix.temporal_window_duration:.3f}s")
                
        except ImportError:
            logger.warning("AtomicClockService not available, using simulated measurements")
            
            # Simulate measurements without atomic clock
            test_nodes = ['tokyo', 'london', 'harare', 'sydney', 'lapaz']
            
            for measurement_round in range(20):
                for node_id in test_nodes:
                    # Create simulated measurement
                    current_time = time.time()
                    simulated_atomic = current_time + random.uniform(-0.001, 0.001)
                    local_time = current_time + random.uniform(-0.002, 0.002)
                    
                    measurement = PrecisionMeasurement(
                        node_id=node_id,
                        measurement_time=current_time,
                        atomic_reference=simulated_atomic,
                        local_measurement=local_time,
                        precision_difference=simulated_atomic - local_time,
                        measurement_quality=random.uniform(0.8, 1.0)
                    )
                    
                    precision_calc._store_measurement(measurement)
                    precision_calc._update_statistics(measurement)
                    
                    if measurement_round % 5 == 0:
                        precision_ms = measurement.precision_difference * 1000
                        logger.info(f"Round {measurement_round+1}, {node_id}: "
                                  f"{precision_ms:.3f}ms (simulated)")
                
                await asyncio.sleep(0.2)
        
        # Get final statistics
        stats = precision_calc.get_precision_statistics()
        
        # Export all data to JSON
        logger.info("Exporting precision calculation data...")
        precision_data = precision_calc.export_precision_data(output_dir)
        
        # Add test summary
        export_data = {
            'precision_calculation_results': precision_data,
            'test_summary': {
                'total_nodes_tested': len(test_nodes),
                'measurements_per_node': 20,
                'total_measurements': stats['total_measurements'],
                'overall_precision_ms': stats['overall_statistics'].get('mean_precision_ms', 0),
                'precision_std_ms': stats['overall_statistics'].get('std_precision_ms', 0),
                'mean_quality': stats['overall_statistics'].get('mean_quality', 0),
                'test_duration': 'approximately 4 seconds',
                'timestamp': time.time()
            }
        }
        
        # Save complete test results
        results_file = output_dir / "precision_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Complete test results saved to {results_file}")
        
        # Create visualizations
        logger.info("Creating precision calculation visualizations...")
        precision_calc.create_precision_visualization(output_dir)
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 PRECISION CALCULATOR TEST RESULTS")
        print("=" * 50)
        print(f"✅ Nodes tested: {len(test_nodes)}")
        print(f"✅ Total measurements: {stats['total_measurements']}")
        
        if stats['overall_statistics']:
            print(f"✅ Average precision: {stats['overall_statistics']['mean_precision_ms']:.3f} ± "
                  f"{stats['overall_statistics']['std_precision_ms']:.3f}ms")
            print(f"✅ Mean quality score: {stats['overall_statistics']['mean_quality']:.3f}")
            print(f"✅ Coordination quality: {stats['overall_statistics']['coordination_quality']:.3f}")
        
        print(f"\n📁 All outputs saved to: {output_dir.absolute()}")
        print(f"📄 JSON results: {results_file.name}")
        print(f"📈 Visualizations: precision_time_series.png, precision_distribution.png, "
              f"quality_analysis.png, coordination_matrix.png")
        
        return True
        
    except Exception as e:
        logger.error(f"Precision calculator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        import random
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        exit(1)
