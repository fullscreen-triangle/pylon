#!/usr/bin/env python3
"""
Network Signals Capture System

Captures all signals that come from network sources:
- Internet connectivity patterns
- WiFi signal strength and activity
- Bluetooth device discovery and connections
- Cellular network information
- Network traffic patterns
- DNS resolution timing
- Connection latency measurements
- Network interface state changes
"""

import time
import json
import logging
import socket
import subprocess
import threading
import queue
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import requests
from urllib.parse import urlparse

# Platform-specific imports
system_os = platform.system()

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import ping3
    PING_AVAILABLE = True
except ImportError:
    PING_AVAILABLE = False

try:
    if system_os == "Windows":
        import wmi
        WMI_AVAILABLE = True
    else:
        WMI_AVAILABLE = False
except ImportError:
    WMI_AVAILABLE = False

class NetworkSignalCollector:
    """
    Collects network signals from various sources.
    
    Captures signals from:
    - Internet connectivity (latency, throughput)
    - WiFi networks (signal strength, available networks)
    - Bluetooth devices (discovery, connections)
    - Cellular networks (signal strength, carrier info)
    - Network interfaces (status, statistics)
    - DNS resolution times
    - TCP/UDP connection patterns
    - Network traffic analysis
    """
    
    def __init__(self, sample_duration: float = 60.0, sample_rate: float = 1.0):
        self.logger = logging.getLogger(__name__)
        self.sample_duration = sample_duration
        self.sample_rate = sample_rate  # Samples per second
        self.sample_interval = 1.0 / sample_rate
        
        # Network signal storage
        self.signals = {
            'connectivity_signals': [],
            'wifi_signals': [],
            'bluetooth_signals': [],
            'cellular_signals': [],
            'interface_signals': [],
            'dns_signals': [],
            'traffic_signals': [],
            'latency_signals': []
        }
        
        # Threading controls
        self.collection_active = False
        self.collection_threads = []
        self.signal_queues = {}
        
        # Network test targets
        self.test_hosts = [
            'google.com',
            'cloudflare.com', 
            '8.8.8.8',
            '1.1.1.1',
            'github.com'
        ]
        
        # Initialize platform capabilities
        self._initialize_platform_components()
        
        self.start_time = None
        
        self.logger.info(f"Network signal collector initialized - duration: {sample_duration}s, rate: {sample_rate}Hz")
    
    def _initialize_platform_components(self):
        """Initialize platform-specific network collection components"""
        self.platform_capabilities = {
            'psutil_available': PSUTIL_AVAILABLE,
            'ping_available': PING_AVAILABLE, 
            'wmi_available': WMI_AVAILABLE and system_os == "Windows",
            'subprocess_available': True,
            'socket_available': True,
            'requests_available': True
        }
        
        # Test internet connectivity
        try:
            response = requests.get('http://google.com', timeout=5)
            self.platform_capabilities['internet_available'] = response.status_code == 200
        except:
            self.platform_capabilities['internet_available'] = False
        
        self.logger.info(f"Network capabilities: {self.platform_capabilities}")
    
    def collect_all_network_signals(self) -> Dict[str, Any]:
        """
        Collect all available network signals for specified duration.
        
        Returns comprehensive network signal data from all sources.
        """
        self.logger.info(f"Starting network signal collection for {self.sample_duration}s...")
        
        self.collection_active = True
        self.start_time = time.time()
        
        # Initialize signal queues
        for signal_type in self.signals.keys():
            self.signal_queues[signal_type] = queue.Queue()
        
        # Start collection threads for different signal types
        collection_threads = [
            threading.Thread(target=self._collect_connectivity_signals, daemon=True),
            threading.Thread(target=self._collect_interface_signals, daemon=True),
            threading.Thread(target=self._collect_dns_signals, daemon=True),
            threading.Thread(target=self._collect_latency_signals, daemon=True),
        ]
        
        # Add conditional signal collectors
        if self.platform_capabilities['internet_available']:
            collection_threads.append(threading.Thread(target=self._collect_traffic_signals, daemon=True))
        
        if system_os == "Windows" and self.platform_capabilities['wmi_available']:
            collection_threads.append(threading.Thread(target=self._collect_windows_wifi_signals, daemon=True))
        elif system_os == "Linux":
            collection_threads.append(threading.Thread(target=self._collect_linux_wifi_signals, daemon=True))
        
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
            thread.join(timeout=3.0)
        
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
                'total_samples_collected': sum(len(signals) for signals in self.signals.values()),
                'test_hosts': self.test_hosts
            },
            'network_signals': self.signals.copy(),
            'signal_analysis': analysis_results,
            'platform_info': self._get_platform_info()
        }
    
    def _collect_connectivity_signals(self):
        """Collect internet connectivity signals"""
        while self.collection_active:
            try:
                timestamp = time.time() - self.start_time
                
                connectivity_tests = []
                
                # Test multiple hosts for connectivity
                for host in self.test_hosts[:3]:  # Test first 3 hosts
                    try:
                        start_time = time.time()
                        
                        # HTTP connectivity test
                        if host.replace('.', '').isdigit():  # IP address
                            # For IP addresses, use socket connection
                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            sock.settimeout(5)
                            result = sock.connect_ex((host, 80))
                            sock.close()
                            
                            http_success = result == 0
                            http_time = time.time() - start_time if http_success else -1
                        else:
                            # For hostnames, use HTTP request
                            try:
                                response = requests.get(f'http://{host}', timeout=5)
                                http_success = response.status_code == 200
                                http_time = time.time() - start_time
                            except:
                                http_success = False
                                http_time = -1
                        
                        # DNS resolution test
                        dns_start = time.time()
                        try:
                            socket.gethostbyname(host)
                            dns_success = True
                            dns_time = time.time() - dns_start
                        except:
                            dns_success = False
                            dns_time = -1
                        
                        connectivity_tests.append({
                            'host': host,
                            'http_success': http_success,
                            'http_time': http_time,
                            'dns_success': dns_success,
                            'dns_time': dns_time
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Connectivity test error for {host}: {e}")
                        connectivity_tests.append({
                            'host': host,
                            'http_success': False,
                            'http_time': -1,
                            'dns_success': False,
                            'dns_time': -1,
                            'error': str(e)
                        })
                
                signal_data = {
                    'timestamp': timestamp,
                    'connectivity_tests': connectivity_tests,
                    'total_successful_connections': sum(1 for test in connectivity_tests if test['http_success']),
                    'average_response_time': np.mean([test['http_time'] for test in connectivity_tests if test['http_time'] > 0]) if any(test['http_time'] > 0 for test in connectivity_tests) else 0
                }
                
                self.signal_queues['connectivity_signals'].put(signal_data)
                
            except Exception as e:
                self.logger.warning(f"Connectivity signal collection error: {e}")
            
            time.sleep(self.sample_interval)
    
    def _collect_interface_signals(self):
        """Collect network interface signals"""
        if not PSUTIL_AVAILABLE:
            return
            
        while self.collection_active:
            try:
                timestamp = time.time() - self.start_time
                
                # Network interface statistics
                net_io = psutil.net_io_counters(pernic=True)
                
                # Network interface addresses
                net_addrs = psutil.net_if_addrs()
                
                # Network interface stats (status, speed, etc.)
                net_stats = psutil.net_if_stats()
                
                interface_data = {}
                
                for interface_name in net_io.keys():
                    io_stats = net_io[interface_name]
                    
                    interface_info = {
                        'bytes_sent': io_stats.bytes_sent,
                        'bytes_recv': io_stats.bytes_recv,
                        'packets_sent': io_stats.packets_sent,
                        'packets_recv': io_stats.packets_recv,
                        'errin': io_stats.errin,
                        'errout': io_stats.errout,
                        'dropin': io_stats.dropin,
                        'dropout': io_stats.dropout
                    }
                    
                    # Add address info if available
                    if interface_name in net_addrs:
                        addresses = []
                        for addr in net_addrs[interface_name]:
                            addresses.append({
                                'family': str(addr.family),
                                'address': addr.address,
                                'netmask': getattr(addr, 'netmask', None),
                                'broadcast': getattr(addr, 'broadcast', None)
                            })
                        interface_info['addresses'] = addresses
                    
                    # Add interface stats if available
                    if interface_name in net_stats:
                        stats = net_stats[interface_name]
                        interface_info.update({
                            'is_up': stats.isup,
                            'duplex': str(stats.duplex),
                            'speed': stats.speed,
                            'mtu': stats.mtu
                        })
                    
                    interface_data[interface_name] = interface_info
                
                # Network connections count
                try:
                    connections = psutil.net_connections()
                    connection_count = len(connections)
                    
                    # Count by status
                    connection_status = {}
                    for conn in connections:
                        status = conn.status
                        connection_status[status] = connection_status.get(status, 0) + 1
                        
                except:
                    connection_count = 0
                    connection_status = {}
                
                signal_data = {
                    'timestamp': timestamp,
                    'interfaces': interface_data,
                    'total_connections': connection_count,
                    'connection_status_counts': connection_status,
                    'active_interfaces': len([name for name, stats in net_stats.items() if stats.isup]) if net_stats else 0
                }
                
                self.signal_queues['interface_signals'].put(signal_data)
                
            except Exception as e:
                self.logger.warning(f"Interface signal collection error: {e}")
            
            time.sleep(self.sample_interval)
    
    def _collect_dns_signals(self):
        """Collect DNS resolution timing signals"""
        test_domains = ['google.com', 'github.com', 'stackoverflow.com', 'wikipedia.org']
        
        while self.collection_active:
            try:
                timestamp = time.time() - self.start_time
                
                dns_tests = []
                
                for domain in test_domains:
                    try:
                        start_time = time.time()
                        
                        # DNS resolution
                        ip_address = socket.gethostbyname(domain)
                        resolution_time = time.time() - start_time
                        
                        dns_tests.append({
                            'domain': domain,
                            'ip_address': ip_address,
                            'resolution_time': resolution_time,
                            'success': True
                        })
                        
                    except Exception as e:
                        dns_tests.append({
                            'domain': domain,
                            'resolution_time': -1,
                            'success': False,
                            'error': str(e)
                        })
                
                signal_data = {
                    'timestamp': timestamp,
                    'dns_tests': dns_tests,
                    'successful_resolutions': sum(1 for test in dns_tests if test['success']),
                    'average_resolution_time': np.mean([test['resolution_time'] for test in dns_tests if test['success']]) if any(test['success'] for test in dns_tests) else 0
                }
                
                self.signal_queues['dns_signals'].put(signal_data)
                
            except Exception as e:
                self.logger.warning(f"DNS signal collection error: {e}")
            
            time.sleep(self.sample_interval * 2)  # DNS tests less frequently
    
    def _collect_latency_signals(self):
        """Collect network latency signals using ping"""
        ping_hosts = ['8.8.8.8', '1.1.1.1', 'google.com']
        
        while self.collection_active:
            try:
                timestamp = time.time() - self.start_time
                
                latency_tests = []
                
                for host in ping_hosts:
                    try:
                        if PING_AVAILABLE:
                            # Use ping3 library if available
                            latency = ping3.ping(host, timeout=3)
                            if latency is not None:
                                latency_ms = latency * 1000  # Convert to milliseconds
                                success = True
                            else:
                                latency_ms = -1
                                success = False
                        else:
                            # Fallback: use subprocess to call system ping
                            if system_os == "Windows":
                                cmd = ['ping', '-n', '1', '-w', '3000', host]
                            else:
                                cmd = ['ping', '-c', '1', '-W', '3', host]
                            
                            start_time = time.time()
                            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                            
                            if result.returncode == 0:
                                # Parse ping output for latency
                                if system_os == "Windows":
                                    # Windows ping output parsing
                                    match = re.search(r'Average = (\d+)ms', result.stdout)
                                    if match:
                                        latency_ms = float(match.group(1))
                                    else:
                                        match = re.search(r'time=(\d+)ms', result.stdout)
                                        latency_ms = float(match.group(1)) if match else -1
                                else:
                                    # Linux/Unix ping output parsing
                                    match = re.search(r'time=([0-9.]+) ms', result.stdout)
                                    latency_ms = float(match.group(1)) if match else -1
                                
                                success = latency_ms > 0
                            else:
                                latency_ms = -1
                                success = False
                        
                        latency_tests.append({
                            'host': host,
                            'latency_ms': latency_ms,
                            'success': success
                        })
                        
                    except Exception as e:
                        latency_tests.append({
                            'host': host,
                            'latency_ms': -1,
                            'success': False,
                            'error': str(e)
                        })
                
                signal_data = {
                    'timestamp': timestamp,
                    'latency_tests': latency_tests,
                    'successful_pings': sum(1 for test in latency_tests if test['success']),
                    'average_latency': np.mean([test['latency_ms'] for test in latency_tests if test['success']]) if any(test['success'] for test in latency_tests) else 0
                }
                
                self.signal_queues['latency_signals'].put(signal_data)
                
            except Exception as e:
                self.logger.warning(f"Latency signal collection error: {e}")
            
            time.sleep(self.sample_interval * 3)  # Ping tests less frequently
    
    def _collect_traffic_signals(self):
        """Collect network traffic pattern signals"""
        if not PSUTIL_AVAILABLE:
            return
            
        last_io_counters = None
        
        while self.collection_active:
            try:
                timestamp = time.time() - self.start_time
                
                # Get current network I/O counters
                current_io = psutil.net_io_counters()
                
                traffic_data = {
                    'timestamp': timestamp,
                    'bytes_sent': current_io.bytes_sent,
                    'bytes_recv': current_io.bytes_recv,
                    'packets_sent': current_io.packets_sent,
                    'packets_recv': current_io.packets_recv
                }
                
                # Calculate rates if we have previous data
                if last_io_counters:
                    time_diff = self.sample_interval
                    
                    traffic_data.update({
                        'bytes_sent_rate': (current_io.bytes_sent - last_io_counters.bytes_sent) / time_diff,
                        'bytes_recv_rate': (current_io.bytes_recv - last_io_counters.bytes_recv) / time_diff,
                        'packets_sent_rate': (current_io.packets_sent - last_io_counters.packets_sent) / time_diff,
                        'packets_recv_rate': (current_io.packets_recv - last_io_counters.packets_recv) / time_diff
                    })
                
                last_io_counters = current_io
                
                self.signal_queues['traffic_signals'].put(traffic_data)
                
            except Exception as e:
                self.logger.warning(f"Traffic signal collection error: {e}")
            
            time.sleep(self.sample_interval)
    
    def _collect_windows_wifi_signals(self):
        """Collect WiFi signals on Windows using WMI"""
        if not WMI_AVAILABLE:
            return
            
        try:
            c = wmi.WMI()
        except:
            return
        
        while self.collection_active:
            try:
                timestamp = time.time() - self.start_time
                
                wifi_data = {
                    'timestamp': timestamp,
                    'wifi_adapters': [],
                    'available_networks': []
                }
                
                # Get WiFi adapters
                try:
                    for interface in c.Win32_NetworkAdapter():
                        if interface.NetConnectionID and 'Wi-Fi' in interface.NetConnectionID:
                            adapter_info = {
                                'name': interface.Name,
                                'connection_id': interface.NetConnectionID,
                                'status': interface.NetConnectionStatus,
                                'mac_address': interface.MACAddress,
                                'enabled': interface.NetEnabled
                            }
                            wifi_data['wifi_adapters'].append(adapter_info)
                except Exception as e:
                    self.logger.warning(f"WiFi adapter enumeration error: {e}")
                
                # Try to get WiFi profiles using netsh (command line)
                try:
                    result = subprocess.run(['netsh', 'wlan', 'show', 'profiles'], 
                                          capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        profiles = []
                        for line in result.stdout.split('\n'):
                            if 'All User Profile' in line:
                                profile_name = line.split(':')[1].strip()
                                profiles.append(profile_name)
                        
                        wifi_data['saved_profiles'] = profiles
                        wifi_data['profile_count'] = len(profiles)
                    
                except Exception as e:
                    self.logger.warning(f"WiFi profile enumeration error: {e}")
                
                self.signal_queues['wifi_signals'].put(wifi_data)
                
            except Exception as e:
                self.logger.warning(f"Windows WiFi signal collection error: {e}")
            
            time.sleep(self.sample_interval * 10)  # WiFi scans less frequently
    
    def _collect_linux_wifi_signals(self):
        """Collect WiFi signals on Linux using iwconfig/nmcli"""
        while self.collection_active:
            try:
                timestamp = time.time() - self.start_time
                
                wifi_data = {
                    'timestamp': timestamp,
                    'wifi_interfaces': [],
                    'available_networks': []
                }
                
                # Try to get WiFi interfaces using iwconfig
                try:
                    result = subprocess.run(['iwconfig'], capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        interfaces = []
                        current_interface = None
                        
                        for line in result.stdout.split('\n'):
                            line = line.strip()
                            if line and not line.startswith(' '):
                                # New interface
                                if 'IEEE 802.11' in line:
                                    interface_name = line.split()[0]
                                    current_interface = {
                                        'name': interface_name,
                                        'type': 'wifi'
                                    }
                                    interfaces.append(current_interface)
                            elif current_interface and line:
                                # Parse interface details
                                if 'ESSID:' in line:
                                    essid_match = re.search(r'ESSID:"([^"]*)"', line)
                                    if essid_match:
                                        current_interface['essid'] = essid_match.group(1)
                                
                                if 'Signal level=' in line:
                                    signal_match = re.search(r'Signal level=(-?\d+)', line)
                                    if signal_match:
                                        current_interface['signal_level'] = int(signal_match.group(1))
                        
                        wifi_data['wifi_interfaces'] = interfaces
                        
                except Exception as e:
                    self.logger.warning(f"iwconfig error: {e}")
                
                # Try nmcli for more detailed info
                try:
                    result = subprocess.run(['nmcli', 'dev', 'wifi', 'list'], 
                                          capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        networks = []
                        lines = result.stdout.split('\n')[1:]  # Skip header
                        
                        for line in lines:
                            if line.strip():
                                parts = line.split()
                                if len(parts) >= 3:
                                    network = {
                                        'ssid': parts[0] if parts[0] != '--' else 'Hidden',
                                        'mode': parts[1],
                                        'channel': parts[2],
                                        'signal': parts[7] if len(parts) > 7 else 'Unknown'
                                    }
                                    networks.append(network)
                        
                        wifi_data['available_networks'] = networks
                        
                except Exception as e:
                    self.logger.warning(f"nmcli error: {e}")
                
                self.signal_queues['wifi_signals'].put(wifi_data)
                
            except Exception as e:
                self.logger.warning(f"Linux WiFi signal collection error: {e}")
            
            time.sleep(self.sample_interval * 10)  # WiFi scans less frequently
    
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
        """Analyze all collected network signals for patterns"""
        analysis = {
            'signal_summary': {},
            'connectivity_analysis': {},
            'latency_analysis': {},
            'traffic_analysis': {}
        }
        
        # Signal summary statistics
        for signal_type, signals in self.signals.items():
            if not signals:
                analysis['signal_summary'][signal_type] = {'count': 0, 'active': False}
                continue
                
            analysis['signal_summary'][signal_type] = {
                'count': len(signals),
                'active': True,
                'time_span': signals[-1]['timestamp'] - signals[0]['timestamp'] if len(signals) > 1 else 0
            }
        
        # Connectivity analysis
        if self.signals['connectivity_signals']:
            conn_signals = self.signals['connectivity_signals']
            success_rates = []
            avg_times = []
            
            for signal in conn_signals:
                total_tests = len(signal['connectivity_tests'])
                successful_tests = signal['total_successful_connections']
                success_rate = successful_tests / total_tests if total_tests > 0 else 0
                success_rates.append(success_rate)
                
                if signal['average_response_time'] > 0:
                    avg_times.append(signal['average_response_time'])
            
            analysis['connectivity_analysis'] = {
                'overall_success_rate': np.mean(success_rates) if success_rates else 0,
                'success_rate_std': np.std(success_rates) if success_rates else 0,
                'average_response_time': np.mean(avg_times) if avg_times else 0,
                'response_time_std': np.std(avg_times) if avg_times else 0
            }
        
        # Latency analysis
        if self.signals['latency_signals']:
            latency_signals = self.signals['latency_signals']
            all_latencies = []
            
            for signal in latency_signals:
                for test in signal['latency_tests']:
                    if test['success'] and test['latency_ms'] > 0:
                        all_latencies.append(test['latency_ms'])
            
            if all_latencies:
                analysis['latency_analysis'] = {
                    'min_latency': float(np.min(all_latencies)),
                    'max_latency': float(np.max(all_latencies)),
                    'average_latency': float(np.mean(all_latencies)),
                    'latency_std': float(np.std(all_latencies)),
                    'latency_percentiles': {
                        '50th': float(np.percentile(all_latencies, 50)),
                        '95th': float(np.percentile(all_latencies, 95)),
                        '99th': float(np.percentile(all_latencies, 99))
                    }
                }
        
        return analysis
    
    def _get_platform_info(self) -> Dict[str, Any]:
        """Get network-related platform information"""
        platform_info = {
            'system': platform.system(),
            'network_capabilities': self.platform_capabilities,
            'hostname': socket.gethostname()
        }
        
        try:
            platform_info['fqdn'] = socket.getfqdn()
        except:
            pass
        
        if PSUTIL_AVAILABLE:
            try:
                # Network interface info
                interfaces = list(psutil.net_if_stats().keys())
                platform_info['network_interfaces'] = interfaces
                platform_info['interface_count'] = len(interfaces)
            except:
                pass
        
        return platform_info
    
    def save_results(self, filepath: str):
        """Save collected network signals to JSON file"""
        results = self.collect_all_network_signals()
        
        # Convert numpy types to JSON serializable
        def convert_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json_results = json.loads(json.dumps(results, default=convert_for_json))
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Network signals saved to {filepath}")
        return json_results
    
    def create_visualizations(self, output_dir: str, results: Optional[Dict] = None):
        """Create visualizations of collected network signals"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if results is None:
            results = self.collect_all_network_signals()
        
        signals = results['network_signals']
        
        try:
            # Plot 1: Network signal collection overview
            plt.figure(figsize=(15, 10))
            
            # Signal count overview
            plt.subplot(2, 3, 1)
            signal_names = []
            signal_counts = []
            
            for signal_type, signal_data in signals.items():
                signal_names.append(signal_type.replace('_', ' ').title())
                signal_counts.append(len(signal_data))
            
            bars = plt.bar(range(len(signal_names)), signal_counts)
            plt.xlabel('Signal Type')
            plt.ylabel('Number of Samples')
            plt.title('Network Signal Collection Overview')
            plt.xticks(range(len(signal_names)), signal_names, rotation=45, ha='right')
            
            for bar, count in zip(bars, signal_counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        str(count), ha='center', va='bottom')
            
            # Plot 2: Connectivity success rate over time
            if signals['connectivity_signals']:
                plt.subplot(2, 3, 2)
                conn_data = signals['connectivity_signals']
                timestamps = [d['timestamp'] for d in conn_data]
                success_rates = []
                
                for d in conn_data:
                    total = len(d['connectivity_tests'])
                    successful = d['total_successful_connections']
                    rate = successful / total if total > 0 else 0
                    success_rates.append(rate * 100)
                
                plt.plot(timestamps, success_rates, 'b-', alpha=0.7, marker='o', markersize=3)
                plt.xlabel('Time (seconds)')
                plt.ylabel('Success Rate (%)')
                plt.title('Connectivity Success Rate Over Time')
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 105)
            
            # Plot 3: Response times
            if signals['connectivity_signals']:
                plt.subplot(2, 3, 3)
                conn_data = signals['connectivity_signals']
                timestamps = [d['timestamp'] for d in conn_data]
                response_times = [d['average_response_time'] * 1000 for d in conn_data if d['average_response_time'] > 0]
                response_timestamps = [timestamps[i] for i, d in enumerate(conn_data) if d['average_response_time'] > 0]
                
                if response_times:
                    plt.plot(response_timestamps, response_times, 'g-', alpha=0.7, marker='o', markersize=3)
                    plt.xlabel('Time (seconds)')
                    plt.ylabel('Response Time (ms)')
                    plt.title('Average Response Time Over Time')
                    plt.grid(True, alpha=0.3)
            
            # Plot 4: Latency measurements
            if signals['latency_signals']:
                plt.subplot(2, 3, 4)
                latency_data = signals['latency_signals']
                
                # Collect all latency measurements
                all_latencies = []
                hosts = set()
                
                for d in latency_data:
                    for test in d['latency_tests']:
                        if test['success'] and test['latency_ms'] > 0:
                            all_latencies.append(test['latency_ms'])
                            hosts.add(test['host'])
                
                if all_latencies:
                    plt.hist(all_latencies, bins=20, alpha=0.7, color='orange', edgecolor='black')
                    plt.xlabel('Latency (ms)')
                    plt.ylabel('Frequency')
                    plt.title(f'Latency Distribution ({len(hosts)} hosts)')
                    plt.grid(True, alpha=0.3)
            
            # Plot 5: Network traffic (if available)
            if signals['traffic_signals'] and len(signals['traffic_signals']) > 1:
                plt.subplot(2, 3, 5)
                traffic_data = signals['traffic_signals']
                timestamps = [d['timestamp'] for d in traffic_data]
                
                # Plot traffic rates if available
                recv_rates = [d.get('bytes_recv_rate', 0) / 1024 for d in traffic_data]  # KB/s
                send_rates = [d.get('bytes_sent_rate', 0) / 1024 for d in traffic_data]  # KB/s
                
                if any(rate > 0 for rate in recv_rates + send_rates):
                    plt.plot(timestamps, recv_rates, 'b-', alpha=0.7, label='Received', linewidth=2)
                    plt.plot(timestamps, send_rates, 'r-', alpha=0.7, label='Sent', linewidth=2)
                    plt.xlabel('Time (seconds)')
                    plt.ylabel('Traffic Rate (KB/s)')
                    plt.title('Network Traffic Rate')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
            
            # Plot 6: DNS resolution times
            if signals['dns_signals']:
                plt.subplot(2, 3, 6)
                dns_data = signals['dns_signals']
                
                all_dns_times = []
                for d in dns_data:
                    for test in d['dns_tests']:
                        if test['success'] and test['resolution_time'] > 0:
                            all_dns_times.append(test['resolution_time'] * 1000)  # ms
                
                if all_dns_times:
                    plt.hist(all_dns_times, bins=15, alpha=0.7, color='purple', edgecolor='black')
                    plt.xlabel('DNS Resolution Time (ms)')
                    plt.ylabel('Frequency')
                    plt.title('DNS Resolution Time Distribution')
                    plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / "network_signals_overview.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Network signal visualizations saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")


def main():
    """Standalone execution of network signals measurement"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Network Signals Capture System")
    parser.add_argument('--duration', type=float, default=60.0,
                       help='Collection duration in seconds (default: 60)')
    parser.add_argument('--sample-rate', type=float, default=0.5,
                       help='Sample rate in Hz (default: 0.5)')
    parser.add_argument('--output-dir', default='network_signals_results',
                       help='Output directory for results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    print("🌐 Network Signals Capture System")
    print("=" * 50)
    print(f"Collection Duration: {args.duration} seconds")
    print(f"Sample Rate: {args.sample_rate} Hz")
    print(f"Output Directory: {args.output_dir}")
    print()
    
    # Initialize signal collector
    collector = NetworkSignalCollector(
        sample_duration=args.duration,
        sample_rate=args.sample_rate
    )
    
    print("Available network capabilities:")
    for capability, available in collector.platform_capabilities.items():
        status = "✅" if available else "❌"
        print(f"  {status} {capability}")
    print()
    
    # Collect signals
    print("🔄 Starting network signal collection...")
    results = collector.collect_all_network_signals()
    
    # Save results
    results_file = Path(args.output_dir) / "network_signals_data.json"
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
            print(f"📡 {signal_type}: {summary['count']} samples")
        else:
            print(f"❌ {signal_type}: No samples collected")
    
    # Analysis summary
    analysis = results['signal_analysis']
    if 'connectivity_analysis' in analysis:
        conn_analysis = analysis['connectivity_analysis']
        print(f"\n🔗 Connectivity Analysis:")
        print(f"   Success Rate: {conn_analysis['overall_success_rate']:.1%}")
        print(f"   Avg Response Time: {conn_analysis['average_response_time']*1000:.1f} ms")
    
    if 'latency_analysis' in analysis:
        latency_analysis = analysis['latency_analysis']
        print(f"\n⏱️  Latency Analysis:")
        print(f"   Average Latency: {latency_analysis['average_latency']:.1f} ms")
        print(f"   95th Percentile: {latency_analysis['latency_percentiles']['95th']:.1f} ms")
    
    print(f"\n💾 Results saved to {args.output_dir}/")
    print(f"📊 Visualizations: network_signals_overview.png")
    
    print("\n🎉 Network signal collection complete!")


if __name__ == "__main__":
    main()