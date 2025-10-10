#!/usr/bin/env python3
"""
Hardware Signals Measurement System

Measures all hardware signals from the computer system:
- CPU and other clocks
- Screen pixel changes
- Keyboard input
- Cursor/mouse movements 
- Ambient noise using microphone
- Any other hardware part that produces periodic signals

The goal is to capture ALL periodic signals, regardless of usefulness or practicality.
"""

import time
import json
import logging
import numpy as np
import psutil
import threading
import queue
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Platform-specific imports
import platform
system_os = platform.system()

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    
try:
    if system_os == "Windows":
        import win32gui
        import win32api
        import win32con
        WINDOWS_API_AVAILABLE = True
    else:
        WINDOWS_API_AVAILABLE = False
except ImportError:
    WINDOWS_API_AVAILABLE = False

try:
    if system_os == "Linux":
        import Xlib
        import Xlib.display
        XLIB_AVAILABLE = True
    else:
        XLIB_AVAILABLE = False
except ImportError:
    XLIB_AVAILABLE = False

class HardwareSignalCollector:
    """
    Collects all available hardware signals from the computer system.
    
    Captures periodic signals from:
    - CPU performance counters
    - System clocks and timers
    - Screen/display changes
    - Input devices (keyboard, mouse)
    - Audio devices (microphone ambient noise)
    - Memory access patterns
    - Disk I/O patterns
    - Network interface activity
    - Temperature sensors
    - Fan speeds
    - Power consumption
    """
    
    def __init__(self, sample_duration: float = 10.0, sample_rate: float = 100.0):
        self.logger = logging.getLogger(__name__)
        self.sample_duration = sample_duration
        self.sample_rate = sample_rate  # Samples per second
        self.sample_interval = 1.0 / sample_rate
        
        # Signal storage
        self.signals = {
            'cpu_signals': [],
            'memory_signals': [],
            'disk_signals': [],
            'network_signals': [],
            'audio_signals': [],
            'input_signals': [],
            'display_signals': [],
            'thermal_signals': [],
            'power_signals': [],
            'timing_signals': []
        }
        
        # Threading controls
        self.collection_active = False
        self.collection_threads = []
        self.signal_queues = {}
        
        # Initialize platform-specific components
        self._initialize_platform_components()
        
        # Start time for synchronization
        self.start_time = None
        
        self.logger.info(f"Hardware signal collector initialized - duration: {sample_duration}s, rate: {sample_rate}Hz")
    
    def _initialize_platform_components(self):
        """Initialize platform-specific signal collection components"""
        self.platform_capabilities = {
            'audio_capture': AUDIO_AVAILABLE,
            'windows_api': WINDOWS_API_AVAILABLE and system_os == "Windows",
            'linux_xlib': XLIB_AVAILABLE and system_os == "Linux",
            'cpu_counters': True,
            'system_stats': True
        }
        
        self.logger.info(f"Platform capabilities: {self.platform_capabilities}")
    
    def collect_all_hardware_signals(self) -> Dict[str, Any]:
        """
        Collect all available hardware signals for specified duration.
        
        Returns comprehensive signal data from all hardware sources.
        """
        self.logger.info(f"Starting hardware signal collection for {self.sample_duration}s...")
        
        self.collection_active = True
        self.start_time = time.time()
        
        # Initialize signal queues
        for signal_type in self.signals.keys():
            self.signal_queues[signal_type] = queue.Queue()
        
        # Start collection threads for different signal types
        collection_threads = [
            threading.Thread(target=self._collect_cpu_signals, daemon=True),
            threading.Thread(target=self._collect_memory_signals, daemon=True),
            threading.Thread(target=self._collect_disk_signals, daemon=True),
            threading.Thread(target=self._collect_network_signals, daemon=True),
            threading.Thread(target=self._collect_timing_signals, daemon=True),
        ]
        
        # Add optional signal collectors based on platform capabilities
        if self.platform_capabilities['audio_capture']:
            collection_threads.append(threading.Thread(target=self._collect_audio_signals, daemon=True))
        
        if self.platform_capabilities['windows_api']:
            collection_threads.append(threading.Thread(target=self._collect_windows_input_signals, daemon=True))
            collection_threads.append(threading.Thread(target=self._collect_windows_display_signals, daemon=True))
        
        # Start all threads
        for thread in collection_threads:
            thread.start()
            self.collection_threads.append(thread)
        
        # Collection duration
        time.sleep(self.sample_duration)
        
        # Stop collection
        self.collection_active = False
        
        # Wait for threads to finish (with timeout)
        for thread in self.collection_threads:
            thread.join(timeout=2.0)
        
        # Collect results from queues
        self._collect_queue_results()
        
        collection_time = time.time() - self.start_time
        
        # Generate analysis
        analysis_results = self._analyze_collected_signals()
        
        return {
            'collection_metadata': {
                'collection_start_time': self.start_time,
                'collection_duration': collection_time,
                'planned_duration': self.sample_duration,
                'sample_rate': self.sample_rate,
                'platform_capabilities': self.platform_capabilities,
                'total_samples_collected': sum(len(signals) for signals in self.signals.values())
            },
            'hardware_signals': self.signals.copy(),
            'signal_analysis': analysis_results,
            'platform_info': self._get_platform_info()
        }
    
    def _collect_cpu_signals(self):
        """Collect CPU-related periodic signals"""
        while self.collection_active:
            try:
                timestamp = time.time() - self.start_time
                
                # CPU usage per core
                cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
                
                # CPU frequency per core (if available)
                try:
                    cpu_freq = psutil.cpu_freq(percpu=True)
                    cpu_frequencies = [freq.current if freq else 0 for freq in cpu_freq] if cpu_freq else []
                except:
                    cpu_frequencies = []
                
                # CPU times
                cpu_times = psutil.cpu_times()
                
                # Load average (Unix systems)
                try:
                    load_avg = psutil.getloadavg()
                except:
                    load_avg = [0, 0, 0]
                
                signal_data = {
                    'timestamp': timestamp,
                    'cpu_percent': cpu_percent,
                    'cpu_frequencies': cpu_frequencies,
                    'cpu_times': {
                        'user': cpu_times.user,
                        'system': cpu_times.system,
                        'idle': cpu_times.idle
                    },
                    'load_average': load_avg,
                    'cpu_count': psutil.cpu_count(),
                    'cpu_count_logical': psutil.cpu_count(logical=True)
                }
                
                self.signal_queues['cpu_signals'].put(signal_data)
                
            except Exception as e:
                self.logger.warning(f"CPU signal collection error: {e}")
            
            time.sleep(self.sample_interval)
    
    def _collect_memory_signals(self):
        """Collect memory-related periodic signals"""
        while self.collection_active:
            try:
                timestamp = time.time() - self.start_time
                
                # Virtual memory
                vmem = psutil.virtual_memory()
                
                # Swap memory
                swap = psutil.swap_memory()
                
                signal_data = {
                    'timestamp': timestamp,
                    'virtual_memory': {
                        'total': vmem.total,
                        'available': vmem.available,
                        'percent': vmem.percent,
                        'used': vmem.used,
                        'free': vmem.free
                    },
                    'swap_memory': {
                        'total': swap.total,
                        'used': swap.used,
                        'free': swap.free,
                        'percent': swap.percent
                    }
                }
                
                self.signal_queues['memory_signals'].put(signal_data)
                
            except Exception as e:
                self.logger.warning(f"Memory signal collection error: {e}")
            
            time.sleep(self.sample_interval)
    
    def _collect_disk_signals(self):
        """Collect disk I/O periodic signals"""
        while self.collection_active:
            try:
                timestamp = time.time() - self.start_time
                
                # Disk I/O counters
                disk_io = psutil.disk_io_counters()
                
                # Disk usage for main partitions
                disk_usage = []
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        disk_usage.append({
                            'device': partition.device,
                            'mountpoint': partition.mountpoint,
                            'fstype': partition.fstype,
                            'total': usage.total,
                            'used': usage.used,
                            'free': usage.free,
                            'percent': (usage.used / usage.total) * 100 if usage.total > 0 else 0
                        })
                    except:
                        continue
                
                signal_data = {
                    'timestamp': timestamp,
                    'disk_io': {
                        'read_count': disk_io.read_count if disk_io else 0,
                        'write_count': disk_io.write_count if disk_io else 0,
                        'read_bytes': disk_io.read_bytes if disk_io else 0,
                        'write_bytes': disk_io.write_bytes if disk_io else 0,
                        'read_time': disk_io.read_time if disk_io else 0,
                        'write_time': disk_io.write_time if disk_io else 0
                    },
                    'disk_usage': disk_usage
                }
                
                self.signal_queues['disk_signals'].put(signal_data)
                
            except Exception as e:
                self.logger.warning(f"Disk signal collection error: {e}")
            
            time.sleep(self.sample_interval)
    
    def _collect_network_signals(self):
        """Collect network interface periodic signals"""
        while self.collection_active:
            try:
                timestamp = time.time() - self.start_time
                
                # Network I/O counters
                net_io = psutil.net_io_counters(pernic=True)
                
                # Network connections
                try:
                    connections = len(psutil.net_connections())
                except:
                    connections = 0
                
                signal_data = {
                    'timestamp': timestamp,
                    'network_io': {},
                    'total_connections': connections
                }
                
                # Per-interface statistics
                for interface, stats in net_io.items():
                    signal_data['network_io'][interface] = {
                        'bytes_sent': stats.bytes_sent,
                        'bytes_recv': stats.bytes_recv,
                        'packets_sent': stats.packets_sent,
                        'packets_recv': stats.packets_recv,
                        'errin': stats.errin,
                        'errout': stats.errout,
                        'dropin': stats.dropin,
                        'dropout': stats.dropout
                    }
                
                self.signal_queues['network_signals'].put(signal_data)
                
            except Exception as e:
                self.logger.warning(f"Network signal collection error: {e}")
            
            time.sleep(self.sample_interval)
    
    def _collect_timing_signals(self):
        """Collect timing-related signals from system clocks"""
        while self.collection_active:
            try:
                timestamp = time.time() - self.start_time
                
                # High-resolution timing measurements
                perf_counter = time.perf_counter()
                monotonic = time.monotonic()
                process_time = time.process_time()
                thread_time = time.thread_time()
                
                signal_data = {
                    'timestamp': timestamp,
                    'system_time': time.time(),
                    'perf_counter': perf_counter,
                    'monotonic': monotonic,
                    'process_time': process_time,
                    'thread_time': thread_time,
                    'clock_resolution': {
                        'perf_counter': time.get_clock_info('perf_counter').resolution,
                        'monotonic': time.get_clock_info('monotonic').resolution,
                        'process_time': time.get_clock_info('process_time').resolution,
                        'thread_time': time.get_clock_info('thread_time').resolution
                    }
                }
                
                self.signal_queues['timing_signals'].put(signal_data)
                
            except Exception as e:
                self.logger.warning(f"Timing signal collection error: {e}")
            
            time.sleep(self.sample_interval)
    
    def _collect_audio_signals(self):
        """Collect ambient noise from microphone"""
        if not AUDIO_AVAILABLE:
            return
            
        try:
            # Audio parameters
            sample_rate = 44100  # Standard audio sample rate
            channels = 1
            duration_per_sample = 0.1  # 100ms per audio sample
            
            while self.collection_active:
                timestamp = time.time() - self.start_time
                
                try:
                    # Record short audio sample
                    audio_data = sd.rec(
                        int(sample_rate * duration_per_sample),
                        samplerate=sample_rate,
                        channels=channels,
                        dtype='float64'
                    )
                    sd.wait()  # Wait for recording to complete
                    
                    # Calculate audio metrics
                    audio_flat = audio_data.flatten()
                    rms = np.sqrt(np.mean(audio_flat**2))
                    peak = np.max(np.abs(audio_flat))
                    zero_crossings = np.sum(np.diff(np.signbit(audio_flat)))
                    
                    # Frequency analysis (simplified)
                    fft = np.fft.fft(audio_flat)
                    freqs = np.fft.fftfreq(len(audio_flat), 1/sample_rate)
                    dominant_freq = freqs[np.argmax(np.abs(fft))]
                    
                    signal_data = {
                        'timestamp': timestamp,
                        'rms_level': float(rms),
                        'peak_level': float(peak),
                        'zero_crossings': int(zero_crossings),
                        'dominant_frequency': float(dominant_freq),
                        'sample_rate': sample_rate,
                        'duration': duration_per_sample,
                        'samples': len(audio_flat)
                    }
                    
                    self.signal_queues['audio_signals'].put(signal_data)
                    
                except Exception as e:
                    self.logger.warning(f"Audio capture error: {e}")
                    time.sleep(0.5)  # Wait before retry
                
        except Exception as e:
            self.logger.error(f"Audio signal collection failed: {e}")
    
    def _collect_windows_input_signals(self):
        """Collect Windows-specific input signals (mouse, keyboard)"""
        if not WINDOWS_API_AVAILABLE:
            return
            
        last_cursor_pos = None
        
        while self.collection_active:
            try:
                timestamp = time.time() - self.start_time
                
                # Get cursor position
                cursor_pos = win32gui.GetCursorPos()
                
                # Calculate cursor movement
                cursor_movement = 0
                if last_cursor_pos:
                    cursor_movement = ((cursor_pos[0] - last_cursor_pos[0])**2 + 
                                     (cursor_pos[1] - last_cursor_pos[1])**2)**0.5
                
                # Get keyboard state (check some common keys)
                key_states = {}
                common_keys = [win32con.VK_SPACE, win32con.VK_RETURN, win32con.VK_SHIFT, 
                              win32con.VK_CONTROL, win32con.VK_MENU]  # Alt key
                
                for key in common_keys:
                    key_states[f'vk_{key}'] = bool(win32api.GetAsyncKeyState(key) & 0x8000)
                
                signal_data = {
                    'timestamp': timestamp,
                    'cursor_position': cursor_pos,
                    'cursor_movement': cursor_movement,
                    'key_states': key_states
                }
                
                self.signal_queues['input_signals'].put(signal_data)
                last_cursor_pos = cursor_pos
                
            except Exception as e:
                self.logger.warning(f"Windows input signal collection error: {e}")
            
            time.sleep(self.sample_interval)
    
    def _collect_windows_display_signals(self):
        """Collect Windows-specific display signals"""
        if not WINDOWS_API_AVAILABLE:
            return
            
        while self.collection_active:
            try:
                timestamp = time.time() - self.start_time
                
                # Get active window information
                try:
                    active_window = win32gui.GetForegroundWindow()
                    window_rect = win32gui.GetWindowRect(active_window)
                    window_title = win32gui.GetWindowText(active_window)
                    
                    signal_data = {
                        'timestamp': timestamp,
                        'active_window_handle': active_window,
                        'window_rect': window_rect,
                        'window_title': window_title,
                        'window_area': (window_rect[2] - window_rect[0]) * (window_rect[3] - window_rect[1])
                    }
                    
                    self.signal_queues['display_signals'].put(signal_data)
                    
                except Exception as e:
                    self.logger.warning(f"Display signal error: {e}")
                
            except Exception as e:
                self.logger.warning(f"Windows display signal collection error: {e}")
            
            time.sleep(self.sample_interval)
    
    def _collect_queue_results(self):
        """Collect all results from signal queues"""
        for signal_type, signal_queue in self.signal_queues.items():
            signals = []
            while not signal_queue.empty():
                try:
                    signals.append(signal_queue.get_nowait())
                except queue.Empty:
                    break
            self.signals[signal_type] = signals
    
    def _analyze_collected_signals(self) -> Dict[str, Any]:
        """Analyze all collected signals for patterns and characteristics"""
        analysis = {
            'signal_summary': {},
            'temporal_patterns': {},
            'frequency_analysis': {},
            'correlation_analysis': {}
        }
        
        # Signal summary statistics
        for signal_type, signals in self.signals.items():
            if not signals:
                analysis['signal_summary'][signal_type] = {'count': 0, 'active': False}
                continue
                
            analysis['signal_summary'][signal_type] = {
                'count': len(signals),
                'active': True,
                'time_span': signals[-1]['timestamp'] - signals[0]['timestamp'] if len(signals) > 1 else 0,
                'sample_rate': len(signals) / (signals[-1]['timestamp'] - signals[0]['timestamp']) if len(signals) > 1 else 0
            }
        
        # Temporal pattern analysis for numeric signals
        self._analyze_temporal_patterns(analysis)
        
        return analysis
    
    def _analyze_temporal_patterns(self, analysis: Dict):
        """Analyze temporal patterns in collected signals"""
        temporal_patterns = {}
        
        for signal_type, signals in self.signals.items():
            if not signals or len(signals) < 3:
                continue
                
            # Extract timestamps
            timestamps = [s['timestamp'] for s in signals]
            
            if len(timestamps) > 1:
                # Calculate sampling intervals
                intervals = np.diff(timestamps)
                
                temporal_patterns[signal_type] = {
                    'mean_interval': float(np.mean(intervals)),
                    'std_interval': float(np.std(intervals)),
                    'min_interval': float(np.min(intervals)),
                    'max_interval': float(np.max(intervals)),
                    'interval_regularity': float(1.0 / (1.0 + np.std(intervals)))  # Regularity score
                }
        
        analysis['temporal_patterns'] = temporal_patterns
    
    def _get_platform_info(self) -> Dict[str, Any]:
        """Get comprehensive platform information"""
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'total_memory': psutil.virtual_memory().total,
            'boot_time': psutil.boot_time()
        }
    
    def save_results(self, filepath: str):
        """Save collected signals to JSON file"""
        results = self.collect_all_hardware_signals()
        
        # Convert numpy types to JSON serializable
        def convert_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Deep convert all results
        json_results = json.loads(json.dumps(results, default=convert_for_json))
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Hardware signals saved to {filepath}")
        
        return json_results
    
    def create_visualizations(self, output_dir: str, results: Optional[Dict] = None):
        """Create visualizations of collected hardware signals"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if results is None:
            results = self.collect_all_hardware_signals()
        
        signals = results['hardware_signals']
        
        try:
            # Plot 1: Signal collection overview
            plt.figure(figsize=(15, 10))
            
            signal_counts = []
            signal_names = []
            
            for signal_type, signal_data in signals.items():
                signal_names.append(signal_type.replace('_', ' ').title())
                signal_counts.append(len(signal_data))
            
            plt.subplot(2, 2, 1)
            bars = plt.bar(range(len(signal_names)), signal_counts)
            plt.xlabel('Signal Type')
            plt.ylabel('Number of Samples')
            plt.title('Hardware Signal Collection Overview')
            plt.xticks(range(len(signal_names)), signal_names, rotation=45, ha='right')
            
            # Add count labels on bars
            for bar, count in zip(bars, signal_counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        str(count), ha='center', va='bottom')
            
            # Plot 2: CPU usage over time
            if signals['cpu_signals']:
                plt.subplot(2, 2, 2)
                cpu_data = signals['cpu_signals']
                timestamps = [d['timestamp'] for d in cpu_data]
                
                if cpu_data[0]['cpu_percent']:  # Check if we have CPU data
                    avg_cpu = [np.mean(d['cpu_percent']) for d in cpu_data]
                    plt.plot(timestamps, avg_cpu, 'b-', alpha=0.7)
                    plt.xlabel('Time (seconds)')
                    plt.ylabel('CPU Usage (%)')
                    plt.title('Average CPU Usage Over Time')
                    plt.grid(True, alpha=0.3)
            
            # Plot 3: Memory usage over time
            if signals['memory_signals']:
                plt.subplot(2, 2, 3)
                mem_data = signals['memory_signals']
                timestamps = [d['timestamp'] for d in mem_data]
                memory_percent = [d['virtual_memory']['percent'] for d in mem_data]
                
                plt.plot(timestamps, memory_percent, 'g-', alpha=0.7)
                plt.xlabel('Time (seconds)')
                plt.ylabel('Memory Usage (%)')
                plt.title('Memory Usage Over Time')
                plt.grid(True, alpha=0.3)
            
            # Plot 4: Network activity
            if signals['network_signals']:
                plt.subplot(2, 2, 4)
                net_data = signals['network_signals']
                timestamps = [d['timestamp'] for d in net_data]
                
                # Sum all network bytes for overview
                total_bytes = []
                for d in net_data:
                    total = 0
                    for interface_data in d['network_io'].values():
                        total += interface_data['bytes_sent'] + interface_data['bytes_recv']
                    total_bytes.append(total)
                
                if len(total_bytes) > 1:
                    # Plot rate of change
                    byte_rates = np.diff(total_bytes)
                    plt.plot(timestamps[1:], byte_rates, 'r-', alpha=0.7)
                    plt.xlabel('Time (seconds)')
                    plt.ylabel('Network Bytes/interval')
                    plt.title('Network Activity Rate')
                    plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / "hardware_signals_overview.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot 5: Audio signal analysis (if available)
            if signals['audio_signals'] and len(signals['audio_signals']) > 0:
                plt.figure(figsize=(12, 8))
                
                audio_data = signals['audio_signals']
                timestamps = [d['timestamp'] for d in audio_data]
                rms_levels = [d['rms_level'] for d in audio_data]
                peak_levels = [d['peak_level'] for d in audio_data]
                
                plt.subplot(2, 1, 1)
                plt.plot(timestamps, rms_levels, 'b-', alpha=0.7, label='RMS Level')
                plt.plot(timestamps, peak_levels, 'r-', alpha=0.7, label='Peak Level')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Audio Level')
                plt.title('Ambient Audio Signal Analysis')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Dominant frequencies
                if 'dominant_frequency' in audio_data[0]:
                    plt.subplot(2, 1, 2)
                    dom_freqs = [d['dominant_frequency'] for d in audio_data]
                    plt.plot(timestamps, dom_freqs, 'g-', alpha=0.7)
                    plt.xlabel('Time (seconds)')
                    plt.ylabel('Dominant Frequency (Hz)')
                    plt.title('Dominant Frequency Over Time')
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_path / "audio_signals_analysis.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            self.logger.info(f"Hardware signal visualizations saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")


def main():
    """Standalone execution of hardware signals measurement"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hardware Signals Measurement System")
    parser.add_argument('--duration', type=float, default=30.0,
                       help='Collection duration in seconds (default: 30)')
    parser.add_argument('--sample-rate', type=float, default=10.0,
                       help='Sample rate in Hz (default: 10)')
    parser.add_argument('--output-dir', default='hardware_signals_results',
                       help='Output directory for results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    print("🖥️  Hardware Signals Measurement System")
    print("=" * 50)
    print(f"Collection Duration: {args.duration} seconds")
    print(f"Sample Rate: {args.sample_rate} Hz")
    print(f"Output Directory: {args.output_dir}")
    print()
    
    # Initialize signal collector
    collector = HardwareSignalCollector(
        sample_duration=args.duration,
        sample_rate=args.sample_rate
    )
    
    print("Available signal types:")
    for capability, available in collector.platform_capabilities.items():
        status = "✅" if available else "❌"
        print(f"  {status} {capability}")
    print()
    
    # Collect signals
    print("🔄 Starting hardware signal collection...")
    results = collector.collect_all_hardware_signals()
    
    # Save results
    results_file = Path(args.output_dir) / "hardware_signals_data.json"
    collector.save_results(str(results_file))
    
    # Create visualizations
    print("📊 Creating visualizations...")
    collector.create_visualizations(args.output_dir, results)
    
    # Print summary
    print("\n📋 Collection Summary:")
    print(f"Total collection time: {results['collection_metadata']['collection_duration']:.2f}s")
    print(f"Total samples collected: {results['collection_metadata']['total_samples_collected']}")
    print()
    
    signal_summary = results['signal_analysis']['signal_summary']
    for signal_type, summary in signal_summary.items():
        if summary['active']:
            print(f"📡 {signal_type}: {summary['count']} samples "
                 f"({summary['sample_rate']:.1f} Hz)")
        else:
            print(f"❌ {signal_type}: No samples collected")
    
    print(f"\n💾 Results saved to {args.output_dir}/")
    print(f"📊 Visualizations: hardware_signals_overview.png")
    if results['hardware_signals']['audio_signals']:
        print(f"🔊 Audio analysis: audio_signals_analysis.png")
    
    print("\n🎉 Hardware signal collection complete!")


if __name__ == "__main__":
    main() 