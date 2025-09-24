# atomic_clock.py - Fixed version
import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PNG output
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class AtomicTimeSource:
    """Represents an atomic clock time source"""
    name: str
    description: str
    api_endpoint: str
    precision_level: str
    accuracy: float  # Accuracy in seconds
    backup_endpoints: List[str] = field(default_factory=list)
    timeout_seconds: float = 5.0
    retry_attempts: int = 3
    last_sync_time: Optional[float] = None
    sync_status: str = "unknown"


class AtomicClockService:
    """Enhanced atomic clock service with proper error handling"""

    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.sources = self._load_time_sources(config_path)
        self.session: Optional[aiohttp.ClientSession] = None
        self._reference_time: Optional[float] = None
        self._last_update: Optional[float] = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10.0)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def get_atomic_reference(self) -> float:
        """Get atomic clock reference with comprehensive error handling"""
        if not self.session:
            raise RuntimeError("AtomicClockService not properly initialized")

        for source in self.sources:
            try:
                async with self.session.get(
                        source.api_endpoint,
                        timeout=aiohttp.ClientTimeout(total=source.timeout_seconds)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        timestamp = self._parse_timestamp(data, source)
                        self._reference_time = timestamp
                        self._last_update = time.time()
                        source.last_sync_time = timestamp
                        source.sync_status = "success"
                        return timestamp

            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout accessing {source.name}")
                source.sync_status = "timeout"
            except aiohttp.ClientError as e:
                self.logger.warning(f"Network error with {source.name}: {e}")
                source.sync_status = "network_error"
            except Exception as e:
                self.logger.error(f"Unexpected error with {source.name}: {e}")
                source.sync_status = "error"

        # Fallback to system time if all sources fail
        self.logger.warning("All atomic clock sources failed, using system time")
        return time.time()

    def _load_time_sources(self, config_path: Optional[str] = None) -> List[AtomicTimeSource]:
        """Load atomic time sources configuration"""
        
        # Default time sources (publicly available NTP and time APIs)
        default_sources = [
            AtomicTimeSource(
                name="WorldTimeAPI",
                description="World Time API",
                api_endpoint="http://worldtimeapi.org/api/timezone/Etc/UTC",
                precision_level="millisecond",
                accuracy=0.001,  # 1ms accuracy
                timeout_seconds=5.0
            ),
            AtomicTimeSource(
                name="TimeAPI",
                description="TimeAPI.io service",
                api_endpoint="http://timeapi.io/api/Time/current/zone?timeZone=UTC",
                precision_level="millisecond", 
                accuracy=0.001,
                timeout_seconds=3.0
            ),
            # Fallback simulated atomic source (for offline testing)
            AtomicTimeSource(
                name="SimulatedAtomic",
                description="Simulated high-precision atomic clock",
                api_endpoint="simulated://atomic",
                precision_level="microsecond",
                accuracy=0.000001,  # 1μs accuracy
                timeout_seconds=1.0
            )
        ]

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    sources_config = config.get('atomic_time_sources', [])
                    
                    sources = []
                    for source_config in sources_config:
                        source = AtomicTimeSource(
                            name=source_config['name'],
                            description=source_config.get('description', ''),
                            api_endpoint=source_config['api_endpoint'],
                            precision_level=source_config.get('precision_level', 'millisecond'),
                            accuracy=source_config.get('accuracy', 0.001),
                            backup_endpoints=source_config.get('backup_endpoints', []),
                            timeout_seconds=source_config.get('timeout_seconds', 5.0)
                        )
                        sources.append(source)
                    
                    if sources:
                        return sources
            except Exception as e:
                self.logger.warning(f"Failed to load time sources config: {e}")

        return default_sources

    def _parse_timestamp(self, data: Dict[str, Any], source: AtomicTimeSource) -> float:
        """Parse timestamp from API response data"""
        
        try:
            # Handle different API response formats
            if source.api_endpoint.startswith("simulated://"):
                # Simulated atomic clock - return high-precision time
                return time.time()
                
            elif "worldtimeapi.org" in source.api_endpoint:
                # WorldTimeAPI format
                if 'unixtime' in data:
                    return float(data['unixtime'])
                elif 'datetime' in data:
                    dt = datetime.fromisoformat(data['datetime'].replace('Z', '+00:00'))
                    return dt.timestamp()
                    
            elif "timeapi.io" in source.api_endpoint:
                # TimeAPI.io format  
                if 'dateTime' in data:
                    dt = datetime.fromisoformat(data['dateTime'].replace('Z', '+00:00'))
                    return dt.timestamp()
                    
            # Generic parsing attempts
            for time_field in ['unixtime', 'timestamp', 'time', 'unix', 'epoch']:
                if time_field in data:
                    return float(data[time_field])
                    
            for datetime_field in ['datetime', 'dateTime', 'date_time', 'iso']:
                if datetime_field in data:
                    dt_str = data[datetime_field]
                    if isinstance(dt_str, str):
                        # Handle various ISO format variations
                        dt_str = dt_str.replace('Z', '+00:00')
                        dt = datetime.fromisoformat(dt_str)
                        return dt.timestamp()

            # If no recognized format, try to parse as direct timestamp
            if isinstance(data, (int, float)):
                return float(data)

            self.logger.warning(f"Unknown timestamp format from {source.name}: {data}")
            return time.time()
            
        except Exception as e:
            self.logger.error(f"Error parsing timestamp from {source.name}: {e}")
            return time.time()

    async def initialize(self):
        """Initialize atomic clock service"""
        self.logger.info("Initializing atomic clock service...")
        
        # Create session if not exists
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10.0)
            )
        
        # Test initial connection to time sources
        initial_reference = await self.get_atomic_reference()
        
        self.logger.info(f"Atomic clock service initialized with reference time: {initial_reference}")
        return initial_reference

    async def stop(self):
        """Stop atomic clock service and cleanup resources"""
        self.logger.info("Stopping atomic clock service...")
        
        if self.session:
            await self.session.close()
            self.session = None
        
        self.logger.info("Atomic clock service stopped")

    def get_source_statistics(self) -> Dict[str, Any]:
        """Get statistics about time source performance"""
        
        stats = {
            'total_sources': len(self.sources),
            'source_details': [],
            'last_reference_time': self._reference_time,
            'last_update_time': self._last_update,
            'reference_age_seconds': time.time() - self._last_update if self._last_update else None
        }
        
        for source in self.sources:
            source_stats = {
                'name': source.name,
                'description': source.description,
                'precision_level': source.precision_level,
                'accuracy': source.accuracy,
                'sync_status': source.sync_status,
                'last_sync_time': source.last_sync_time,
                'endpoint': source.api_endpoint
            }
            stats['source_details'].append(source_stats)
        
        return stats

    async def get_precision_measurement(self) -> Dict[str, Any]:
        """Get a precision measurement from atomic reference"""
        
        current_time = time.time()
        atomic_ref = await self.get_atomic_reference()
        
        # Calculate precision difference
        precision_diff = atomic_ref - current_time
        
        # Assess measurement quality based on source reliability
        active_sources = [s for s in self.sources if s.sync_status == "success"]
        quality = len(active_sources) / len(self.sources) if self.sources else 0.0
        
        measurement = {
            'timestamp': current_time,
            'local_time': current_time,
            'atomic_reference': atomic_ref,
            'precision_difference': precision_diff,
            'measurement_quality': quality,
            'confidence': 0.95 if quality > 0.5 else 0.75,
            'active_sources': len(active_sources),
            'total_sources': len(self.sources)
        }
        
        return measurement

    def export_timing_data(self, output_dir: Path = Path("output")) -> Dict[str, Any]:
        """Export atomic clock timing data"""
        output_dir.mkdir(exist_ok=True)
        
        # Collect timing statistics
        timing_data = {
            'service_statistics': self.get_source_statistics(),
            'measurement_timestamp': time.time(),
            'source_performance': {}
        }
        
        # Add individual source performance
        for source in self.sources:
            timing_data['source_performance'][source.name] = {
                'accuracy': source.accuracy,
                'precision_level': source.precision_level,
                'status': source.sync_status,
                'last_sync': source.last_sync_time,
                'reliability': 1.0 if source.sync_status == "success" else 0.0
            }
        
        # Save to JSON
        json_file = output_dir / "atomic_clock_data.json"
        with open(json_file, 'w') as f:
            json.dump(timing_data, f, indent=2)
        
        self.logger.info(f"Atomic clock data exported to {json_file}")
        return timing_data

    def create_timing_visualization(self, measurements: List[Dict[str, Any]], output_dir: Path = Path("output")):
        """Create timing visualization plots"""
        output_dir.mkdir(exist_ok=True)
        
        if not measurements:
            self.logger.warning("No measurements available for visualization")
            return
        
        self._create_precision_plot(measurements, output_dir)
        self._create_source_reliability_plot(output_dir)
        self._create_timing_accuracy_plot(measurements, output_dir)

    def _create_precision_plot(self, measurements: List[Dict[str, Any]], output_dir: Path):
        """Create precision difference over time plot"""
        try:
            if not measurements:
                return
                
            timestamps = [m['timestamp'] for m in measurements]
            precision_diffs = [m['precision_difference'] * 1000 for m in measurements]  # Convert to ms
            
            # Normalize timestamps to start from 0
            start_time = min(timestamps)
            relative_times = [(t - start_time) for t in timestamps]
            
            plt.figure(figsize=(12, 8))
            plt.plot(relative_times, precision_diffs, 'b-o', markersize=4, linewidth=1, alpha=0.7)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Precision Difference (ms)')
            plt.title('Atomic Clock Precision Over Time')
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            mean_precision = np.mean(precision_diffs)
            std_precision = np.std(precision_diffs)
            plt.axhline(y=mean_precision, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_precision:.3f}ms')
            plt.axhline(y=mean_precision + std_precision, color='orange', linestyle=':', alpha=0.7, label=f'±1σ: {std_precision:.3f}ms')
            plt.axhline(y=mean_precision - std_precision, color='orange', linestyle=':', alpha=0.7)
            
            plt.legend()
            plt.tight_layout()
            
            precision_file = output_dir / "atomic_precision_plot.png"
            plt.savefig(precision_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Precision plot saved to {precision_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating precision plot: {e}")

    def _create_source_reliability_plot(self, output_dir: Path):
        """Create source reliability visualization"""
        try:
            source_names = [s.name for s in self.sources]
            reliability_scores = []
            
            for source in self.sources:
                if source.sync_status == "success":
                    score = 1.0
                elif source.sync_status in ["timeout", "network_error"]:
                    score = 0.5
                else:
                    score = 0.0
                reliability_scores.append(score)
            
            plt.figure(figsize=(10, 6))
            colors = ['green' if s >= 0.8 else 'orange' if s >= 0.5 else 'red' for s in reliability_scores]
            bars = plt.bar(range(len(source_names)), reliability_scores, color=colors, alpha=0.7)
            
            plt.xlabel('Time Sources')
            plt.ylabel('Reliability Score')
            plt.title('Atomic Time Source Reliability')
            plt.xticks(range(len(source_names)), source_names, rotation=45)
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            reliability_file = output_dir / "source_reliability.png"
            plt.savefig(reliability_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Source reliability plot saved to {reliability_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating reliability plot: {e}")

    def _create_timing_accuracy_plot(self, measurements: List[Dict[str, Any]], output_dir: Path):
        """Create timing accuracy distribution plot"""
        try:
            if not measurements:
                return
                
            precision_diffs = [m['precision_difference'] * 1000 for m in measurements]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram of precision differences
            ax1.hist(precision_diffs, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax1.set_xlabel('Precision Difference (ms)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Precision Difference Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Box plot for statistical summary
            ax2.boxplot(precision_diffs, labels=['Precision Diff'])
            ax2.set_ylabel('Precision Difference (ms)')
            ax2.set_title('Precision Statistical Summary')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            accuracy_file = output_dir / "timing_accuracy.png"
            plt.savefig(accuracy_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Timing accuracy plot saved to {accuracy_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating accuracy plot: {e}")


# Standalone execution capability
async def main():
    """Standalone execution of atomic clock service"""
    print("⏰ Atomic Clock Service - Standalone Test")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Create output directory
        output_dir = Path("atomic_clock_output")
        output_dir.mkdir(exist_ok=True)
        
        logger.info("Initializing atomic clock service...")
        
        # Initialize atomic clock service
        async with AtomicClockService() as clock_service:
            await clock_service.initialize()
            
            logger.info("Running atomic clock precision tests...")
            
            # Collect precision measurements
            measurements = []
            for i in range(20):  # 20 measurements over 10 seconds
                try:
                    measurement = await clock_service.get_precision_measurement()
                    measurements.append(measurement)
                    
                    # Log measurement
                    precision_ms = measurement['precision_difference'] * 1000
                    logger.info(f"Measurement {i+1}: {precision_ms:.3f}ms precision difference")
                    
                    await asyncio.sleep(0.5)  # 500ms between measurements
                    
                except Exception as e:
                    logger.warning(f"Measurement {i+1} failed: {e}")
            
            # Get final statistics
            source_stats = clock_service.get_source_statistics()
            
            # Export all data to JSON
            logger.info("Exporting atomic clock data...")
            timing_data = clock_service.export_timing_data(output_dir)
            
            # Add measurements to export
            export_data = {
                'atomic_clock_service': timing_data,
                'measurements': measurements,
                'test_summary': {
                    'total_measurements': len(measurements),
                    'successful_measurements': len([m for m in measurements if m['measurement_quality'] > 0]),
                    'average_precision_ms': np.mean([m['precision_difference'] * 1000 for m in measurements]) if measurements else 0,
                    'precision_std_ms': np.std([m['precision_difference'] * 1000 for m in measurements]) if measurements else 0,
                    'test_duration': 'approximately 10 seconds',
                    'timestamp': time.time()
                }
            }
            
            # Save complete test results
            results_file = output_dir / "atomic_clock_test_results.json"
            with open(results_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Complete test results saved to {results_file}")
            
            # Create visualizations
            logger.info("Creating atomic clock visualizations...")
            clock_service.create_timing_visualization(measurements, output_dir)
            
            # Print summary
            print("\n" + "=" * 50)
            print("📊 ATOMIC CLOCK SERVICE TEST RESULTS")
            print("=" * 50)
            print(f"✅ Time sources configured: {source_stats['total_sources']}")
            print(f"✅ Active sources: {len([s for s in source_stats['source_details'] if s['sync_status'] == 'success'])}")
            print(f"✅ Measurements collected: {len(measurements)}")
            
            if measurements:
                avg_precision = np.mean([m['precision_difference'] * 1000 for m in measurements])
                std_precision = np.std([m['precision_difference'] * 1000 for m in measurements])
                print(f"✅ Average precision: {avg_precision:.3f} ± {std_precision:.3f}ms")
            
            print(f"\n📁 All outputs saved to: {output_dir.absolute()}")
            print(f"📄 JSON results: {results_file.name}")
            print(f"📈 Visualizations: atomic_precision_plot.png, source_reliability.png, timing_accuracy.png")
            
            return True
            
    except Exception as e:
        logger.error(f"Atomic clock service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        exit(1)
