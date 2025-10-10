import time
import hashlib
import numpy as np
from typing import Dict, Any, Optional, List
import json
import logging

class FiniteObserver:
    """Base class for observers with finite observation space"""
    
    def __init__(self, observation_frequency: float, max_observation_space: int, observer_id: str = None):
        self.frequency = observation_frequency
        self.max_space = max_observation_space  # Finite constraint
        self.current_observations: Dict[str, Any] = {}
        self.observer_id = observer_id or f"observer_{int(time.time())}"
        self.observation_history: List[Dict] = []
        self.total_observations = 0
        self.successful_observations = 0
        self.failed_observations = 0
        self.start_time = time.time()
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.observer_id}")
        
        self.logger.info(f"Finite observer initialized: freq={observation_frequency}Hz, max_space={max_observation_space}")
    
    def observe_signal(self, signal: Any) -> bool:
        """Binary utility function - either observes or doesn't"""
        self.total_observations += 1
        
        # Finite space constraint
        if len(self.current_observations) >= self.max_space:
            self.failed_observations += 1
            self.logger.debug(f"Observation failed: space limit reached ({self.max_space})")
            return False
        
        # Check if can observe at frequency
        if self._can_observe_at_frequency(signal):
            self.successful_observations += 1
            self._store_observation(signal)
            self.logger.debug(f"Signal observed successfully: {type(signal).__name__}")
            return True
        else:
            self.failed_observations += 1
            self.logger.debug(f"Signal observation failed: frequency mismatch")
            return False
    
    def _can_observe_at_frequency(self, signal: Any) -> bool:
        """Check if signal can be observed at this observer's frequency"""
        try:
            # Extract frequency characteristics from signal
            signal_freq = self._extract_signal_frequency(signal)
            
            # Allow observation if signal frequency is within reasonable range of observer frequency
            frequency_tolerance = 0.1  # 10% tolerance
            freq_diff = abs(signal_freq - self.frequency) / self.frequency
            
            return freq_diff <= frequency_tolerance
            
        except Exception as e:
            self.logger.warning(f"Error checking signal frequency: {e}")
            return False
    
    def _extract_signal_frequency(self, signal: Any) -> float:
        """Extract characteristic frequency from signal"""
        if isinstance(signal, (int, float)):
            # For numeric signals, use modulation based on value
            return self.frequency * (1.0 + 0.1 * np.sin(signal))
        
        elif isinstance(signal, str):
            # For string signals, use hash-based frequency
            signal_hash = int(hashlib.md5(signal.encode()).hexdigest()[:8], 16)
            return self.frequency * (1.0 + 0.1 * (signal_hash % 1000) / 1000.0)
        
        elif isinstance(signal, bytes):
            # For byte signals, use length and content based frequency
            content_hash = int(hashlib.sha256(signal).hexdigest()[:8], 16)
            return self.frequency * (1.0 + 0.1 * (content_hash % 1000) / 1000.0)
        
        elif isinstance(signal, dict):
            # For dict signals, use key count and content hash
            key_factor = len(signal.keys()) % 10
            content_str = json.dumps(signal, sort_keys=True)
            content_hash = int(hashlib.md5(content_str.encode()).hexdigest()[:8], 16)
            return self.frequency * (1.0 + 0.1 * key_factor / 10.0 + 0.05 * (content_hash % 100) / 100.0)
        
        else:
            # Default: use string representation
            signal_str = str(signal)
            signal_hash = int(hashlib.md5(signal_str.encode()).hexdigest()[:8], 16)
            return self.frequency * (1.0 + 0.1 * (signal_hash % 1000) / 1000.0)
    
    def _store_observation(self, signal: Any):
        """Store observation in finite space"""
        observation_id = f"obs_{len(self.current_observations)}_{int(time.time() * 1000000) % 1000000}"
        
        observation_record = {
            'id': observation_id,
            'timestamp': time.time(),
            'signal_type': type(signal).__name__,
            'signal_size': len(str(signal)),
            'extracted_frequency': self._extract_signal_frequency(signal),
            'observation_space_usage': len(self.current_observations) / self.max_space
        }
        
        # Store in current observations
        self.current_observations[observation_id] = {
            'signal': signal,
            'metadata': observation_record
        }
        
        # Add to history (for metrics)
        self.observation_history.append(observation_record)
        
        # Keep history bounded
        if len(self.observation_history) > 1000:
            self.observation_history = self.observation_history[-1000:]
    
    def clear_observations(self):
        """Clear current observations to free space"""
        cleared_count = len(self.current_observations)
        self.current_observations.clear()
        self.logger.info(f"Cleared {cleared_count} observations")
    
    def get_observation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive observation statistics"""
        current_time = time.time()
        runtime = current_time - self.start_time
        
        # Calculate success rate
        success_rate = self.successful_observations / max(1, self.total_observations)
        
        # Calculate space utilization
        space_utilization = len(self.current_observations) / self.max_space
        
        # Calculate observation rate
        observation_rate = self.total_observations / max(1, runtime)
        
        # Frequency analysis of stored observations
        if self.observation_history:
            extracted_frequencies = [obs['extracted_frequency'] for obs in self.observation_history]
            freq_mean = np.mean(extracted_frequencies)
            freq_std = np.std(extracted_frequencies)
            freq_variance = np.var(extracted_frequencies)
        else:
            freq_mean = freq_std = freq_variance = 0.0
        
        return {
            'observer_id': self.observer_id,
            'configured_frequency': self.frequency,
            'max_observation_space': self.max_space,
            'current_observations': len(self.current_observations),
            'total_observations': self.total_observations,
            'successful_observations': self.successful_observations,
            'failed_observations': self.failed_observations,
            'success_rate': success_rate,
            'space_utilization': space_utilization,
            'observation_rate_per_second': observation_rate,
            'runtime_seconds': runtime,
            'frequency_statistics': {
                'mean_extracted_frequency': freq_mean,
                'frequency_std_deviation': freq_std,
                'frequency_variance': freq_variance,
                'frequency_deviation_from_configured': abs(freq_mean - self.frequency) if freq_mean > 0 else 0
            }
        }
    
    def export_observations(self, filepath: str):
        """Export observations to JSON file"""
        export_data = {
            'observer_metadata': {
                'observer_id': self.observer_id,
                'frequency': self.frequency,
                'max_space': self.max_space,
                'export_timestamp': time.time()
            },
            'statistics': self.get_observation_statistics(),
            'observation_history': self.observation_history[-100:],  # Last 100 observations
            'current_observations_summary': [
                {
                    'id': obs_id,
                    'metadata': obs_data['metadata']
                }
                for obs_id, obs_data in list(self.current_observations.items())[:10]  # First 10 current
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Observations exported to {filepath}")
    
    def __repr__(self):
        return f"FiniteObserver(id={self.observer_id}, freq={self.frequency}Hz, space={len(self.current_observations)}/{self.max_space})"


def main():
    """Standalone execution of finite observer demonstration"""
    import argparse
    import random
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Finite Observer Demonstration")
    parser.add_argument('--frequency', type=float, default=5.0,
                       help='Observer frequency in Hz (default: 5.0)')
    parser.add_argument('--max-space', type=int, default=100,
                       help='Maximum observation space (default: 100)')
    parser.add_argument('--duration', type=float, default=30.0,
                       help='Test duration in seconds (default: 30)')
    parser.add_argument('--output-dir', default='finite_observer_results',
                       help='Output directory for results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    print("🔍 Finite Observer Demonstration")
    print("=" * 40)
    print(f"Observer Frequency: {args.frequency} Hz")
    print(f"Maximum Observation Space: {args.max_space}")
    print(f"Test Duration: {args.duration} seconds")
    print(f"Output Directory: {args.output_dir}")
    print()
    
    # Initialize finite observer
    observer = FiniteObserver(
        observation_frequency=args.frequency,
        max_observation_space=args.max_space,
        observer_id="demo_observer"
    )
    
    print(f"📡 Initialized: {observer}")
    print()
    
    # Generate and observe various signal types
    print("🔄 Starting signal observation...")
    
    start_time = time.time()
    signal_count = 0
    successful_observations = 0
    
    # Test different signal types
    signal_types = ['numeric', 'string', 'dict', 'bytes']
    
    while time.time() - start_time < args.duration:
        signal_type = random.choice(signal_types)
        
        if signal_type == 'numeric':
            signal = random.uniform(-100, 100)
        elif signal_type == 'string':
            signal = f"signal_{signal_count}_{random.randint(1000, 9999)}"
        elif signal_type == 'dict':
            signal = {
                'value': random.uniform(0, 1),
                'timestamp': time.time(),
                'id': signal_count,
                'metadata': {'type': 'test', 'source': 'demo'}
            }
        else:  # bytes
            signal = f"binary_signal_{signal_count}".encode()
        
        # Attempt observation
        success = observer.observe_signal(signal)
        if success:
            successful_observations += 1
        
        signal_count += 1
        
        # Control rate approximately
        time.sleep(0.1)  # 10 Hz signal generation rate
    
    actual_duration = time.time() - start_time
    
    print(f"✅ Signal observation complete!")
    print(f"   Duration: {actual_duration:.2f}s")
    print(f"   Signals generated: {signal_count}")
    print(f"   Successful observations: {successful_observations}")
    print(f"   Success rate: {successful_observations/signal_count:.1%}")
    print()
    
    # Get statistics
    stats = observer.get_observation_statistics()
    
    print("📊 Observer Statistics:")
    print(f"   Observer success rate: {stats['success_rate']:.1%}")
    print(f"   Space utilization: {stats['space_utilization']:.1%}")
    print(f"   Observation rate: {stats['observation_rate_per_second']:.2f} obs/sec")
    print(f"   Configured frequency: {stats['configured_frequency']:.2f} Hz")
    print(f"   Mean extracted frequency: {stats['frequency_statistics']['mean_extracted_frequency']:.2f} Hz")
    print(f"   Frequency deviation: {stats['frequency_statistics']['frequency_deviation_from_configured']:.2f} Hz")
    print()
    
    # Save results
    results_file = Path(args.output_dir) / "finite_observer_data.json"
    observer.export_observations(str(results_file))
    
    # Create visualizations
    print("📊 Creating visualizations...")
    
    # Plot 1: Observation success over time
    if observer.observation_history:
        plt.figure(figsize=(15, 10))
        
        # Success rate over time
        plt.subplot(2, 3, 1)
        timestamps = [obs['timestamp'] - observer.start_time for obs in observer.observation_history]
        success_flags = [1 if obs['success'] else 0 for obs in observer.observation_history]
        
        # Running average success rate
        window_size = max(10, len(success_flags) // 20)
        running_success = []
        for i in range(len(success_flags)):
            start_idx = max(0, i - window_size + 1)
            running_success.append(np.mean(success_flags[start_idx:i+1]))
        
        plt.plot(timestamps, running_success, 'b-', alpha=0.7, linewidth=2)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Running Success Rate')
        plt.title('Observation Success Rate Over Time')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        
        # Frequency analysis
        plt.subplot(2, 3, 2)
        extracted_freqs = [obs['extracted_frequency'] for obs in observer.observation_history]
        
        plt.hist(extracted_freqs, bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(x=args.frequency, color='red', linestyle='--', linewidth=2, 
                   label=f'Configured: {args.frequency:.2f} Hz')
        plt.xlabel('Extracted Frequency (Hz)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Extracted Signal Frequencies')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Space utilization over time
        plt.subplot(2, 3, 3)
        space_utilization = []
        for i, obs in enumerate(observer.observation_history):
            # Estimate space usage at this point in time
            successful_so_far = sum(1 for j in range(i+1) if observer.observation_history[j]['success'])
            space_util = min(successful_so_far / args.max_space, 1.0)
            space_utilization.append(space_util * 100)
        
        plt.plot(timestamps, space_utilization, 'orange', alpha=0.7, linewidth=2)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Space Utilization (%)')
        plt.title('Observation Space Utilization Over Time')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)
        
        # Signal type distribution
        plt.subplot(2, 3, 4)
        signal_types = [obs['signal_type'] for obs in observer.observation_history]
        type_counts = {}
        for sig_type in signal_types:
            type_counts[sig_type] = type_counts.get(sig_type, 0) + 1
        
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
        
        plt.bar(types, counts, color=colors, alpha=0.7, edgecolor='black')
        plt.xlabel('Signal Type')
        plt.ylabel('Observation Count')
        plt.title('Signal Type Distribution')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Frequency deviation analysis
        plt.subplot(2, 3, 5)
        freq_deviations = [abs(obs['extracted_frequency'] - args.frequency) for obs in observer.observation_history]
        
        plt.hist(freq_deviations, bins=15, alpha=0.7, color='purple', edgecolor='black')
        plt.xlabel('Frequency Deviation (Hz)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Frequency Deviations')
        plt.grid(True, alpha=0.3)
        
        # Performance summary
        plt.subplot(2, 3, 6)
        
        # Create performance metrics pie chart
        performance_data = [
            stats['successful_observations'],
            stats['failed_observations']
        ]
        performance_labels = ['Successful', 'Failed']
        colors = ['green', 'red']
        
        plt.pie(performance_data, labels=performance_labels, colors=colors, autopct='%1.1f%%', 
               startangle=90, alpha=0.7)
        plt.title('Overall Observation Performance')
        
        plt.tight_layout()
        plt.savefig(Path(args.output_dir) / "finite_observer_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Analysis plot: finite_observer_analysis.png")
    
    # Summary statistics
    summary_stats = {
        'test_configuration': {
            'observer_frequency': args.frequency,
            'max_observation_space': args.max_space,
            'test_duration_planned': args.duration,
            'test_duration_actual': actual_duration
        },
        'signal_generation': {
            'total_signals_generated': signal_count,
            'signal_generation_rate': signal_count / actual_duration,
            'successful_observations': successful_observations,
            'generation_success_rate': successful_observations / signal_count if signal_count > 0 else 0
        },
        'observer_performance': stats
    }
    
    # Save summary
    summary_file = Path(args.output_dir) / "finite_observer_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"💾 Results saved to {args.output_dir}/")
    print(f"   📄 Data: finite_observer_data.json")
    print(f"   📊 Summary: finite_observer_summary.json")
    print(f"   🖼️  Plots: finite_observer_analysis.png")
    
    # Performance evaluation
    if stats['success_rate'] > 0.8:
        print(f"\n✅ Observer performed excellently (success rate: {stats['success_rate']:.1%})")
    elif stats['success_rate'] > 0.6:
        print(f"\n🟡 Observer performed well (success rate: {stats['success_rate']:.1%})")
    elif stats['success_rate'] > 0.4:
        print(f"\n🟠 Observer performance was moderate (success rate: {stats['success_rate']:.1%})")
    else:
        print(f"\n🔴 Observer performance was poor (success rate: {stats['success_rate']:.1%})")
    
    if stats['space_utilization'] > 0.9:
        print(f"⚠️  Space utilization is high ({stats['space_utilization']:.1%}) - consider increasing max_space")
    elif stats['space_utilization'] < 0.1:
        print(f"ℹ️  Space utilization is low ({stats['space_utilization']:.1%}) - could reduce max_space")
    
    print("\n🎉 Finite observer demonstration complete!")


if __name__ == "__main__":
    main()