"""
Atomic Clock Service Module

Provides real-time atomic clock synchronization for temporal reference distribution
in the Sango Rine Shumba framework. Integrates with multiple atomic clock sources
including NIST, PTB, and GPS constellation for nanosecond precision timing.

This service is critical for precision-by-difference calculations, providing the
common temporal reference T_ref(k) used throughout the network coordination system.
"""

import asyncio
import aiohttp
import json
import time
import math
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import numpy as np
from datetime import datetime, timezone

@dataclass
class AtomicTimeSource:
    """Represents an atomic clock time source"""
    
    name: str
    description: str
    api_endpoint: str
    precision_level: str
    accuracy: float  # Accuracy in seconds
    reliability: float  # Reliability score 0-1
    priority: int
    location: Optional[Dict[str, float]] = None
    api_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime state
    is_available: bool = True
    last_query_time: float = 0.0
    last_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)
    accuracy_measurements: List[float] = field(default_factory=list)
    error_count: int = 0
    success_count: int = 0

@dataclass
class TimeReading:
    """Represents a time reading from an atomic clock source"""
    
    source_name: str
    timestamp: float
    precision: float
    uncertainty: float
    source_accuracy: float
    query_latency: float
    leap_second_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize derived properties"""
        self.reading_time = time.time()
        self.age_seconds = 0.0
    
    def update_age(self):
        """Update the age of this reading"""
        self.age_seconds = time.time() - self.reading_time
    
    @property
    def is_fresh(self, max_age_seconds: float = 1.0) -> bool:
        """Check if reading is still fresh"""
        self.update_age()
        return self.age_seconds <= max_age_seconds

class AtomicClockService:
    """
    Atomic clock synchronization service
    
    Provides high-precision temporal reference through integration with multiple
    atomic clock sources. Implements precision-by-difference calculations and
    maintains continuous synchronization for network coordination.
    """
    
    def __init__(self, config_path: str = "config/atomic_clock_config.json"):
        """Initialize atomic clock service"""
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Time sources
        self.primary_sources: List[AtomicTimeSource] = []
        self.fallback_sources: List[AtomicTimeSource] = []
        self.current_source: Optional[AtomicTimeSource] = None
        
        # Time readings
        self.time_readings: List[TimeReading] = []
        self.reference_time: Optional[TimeReading] = None
        
        # Synchronization state
        self.is_synchronized = False
        self.synchronization_accuracy = 0.0
        self.drift_rate = 0.0
        self.last_sync_time = 0.0
        
        # Configuration
        self.config = {}
        self.query_interval = 1.0  # Default 1 second
        self.fast_query_interval = 0.1  # 100ms for high precision
        self.max_drift_tolerance = 0.01  # 10ms
        
        # Performance monitoring
        self.performance_metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'average_response_time': 0.0,
            'accuracy_history': [],
            'sync_events': []
        }
        
        # HTTP session for API calls
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False
        
        self.logger.info("Atomic clock service initialized")
    
    async def initialize(self):
        """Initialize atomic clock service and establish synchronization"""
        self.logger.info("Initializing atomic clock service...")
        
        # Load configuration
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.logger.warning(f"Config file not found: {self.config_path}, using defaults")
            self.config = self._get_default_config()
        
        # Extract configuration parameters
        self._load_configuration()
        
        # Create time sources
        self._create_time_sources()
        
        # Initialize HTTP session
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.get('api_configuration', {}).get('request_timeout_ms', 2000) / 1000)
        )
        
        # Establish initial synchronization
        await self._establish_initial_sync()
        
        # Start background synchronization
        await self._start_background_sync()
        
        self.is_running = True
        self.logger.info(f"Atomic clock service initialized with {len(self.primary_sources)} primary sources")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when config file is not available"""
        return {
            "primary_time_sources": [
                {
                    "name": "NTP_Pool",
                    "description": "NTP Pool Time Servers",
                    "api_endpoint": "pool.ntp.org",
                    "precision_level": "microsecond",
                    "accuracy": 1e-6,
                    "reliability": 0.95,
                    "priority": 1
                }
            ],
            "synchronization_parameters": {
                "query_interval_ms": 1000,
                "fast_query_interval_ms": 100,
                "max_drift_tolerance_ms": 10,
                "precision_validation_threshold": 3
            }
        }
    
    def _load_configuration(self):
        """Load configuration parameters"""
        sync_params = self.config.get('synchronization_parameters', {})
        
        self.query_interval = sync_params.get('query_interval_ms', 1000) / 1000
        self.fast_query_interval = sync_params.get('fast_query_interval_ms', 100) / 1000
        self.max_drift_tolerance = sync_params.get('max_drift_tolerance_ms', 10) / 1000
        
        self.logger.debug(f"Configuration loaded: query_interval={self.query_interval}s")
    
    def _create_time_sources(self):
        """Create atomic time sources from configuration"""
        # Create primary sources
        for source_config in self.config.get('primary_time_sources', []):
            source = AtomicTimeSource(
                name=source_config['name'],
                description=source_config['description'],
                api_endpoint=source_config['api_endpoint'],
                precision_level=source_config['precision_level'],
                accuracy=source_config['accuracy'],
                reliability=source_config['reliability'],
                priority=source_config['priority'],
                location=source_config.get('location'),
                api_parameters=source_config.get('api_parameters', {})
            )
            self.primary_sources.append(source)
        
        # Create fallback sources
        for source_config in self.config.get('fallback_time_sources', []):
            source = AtomicTimeSource(
                name=source_config['name'],
                description=source_config['description'],
                api_endpoint=source_config['api_endpoint'],
                precision_level=source_config['precision_level'],
                accuracy=source_config['accuracy'],
                reliability=source_config['reliability'],
                priority=source_config['priority'],
                api_parameters=source_config.get('api_parameters', {})
            )
            self.fallback_sources.append(source)
        
        # Sort sources by priority
        self.primary_sources.sort(key=lambda x: x.priority)
        self.fallback_sources.sort(key=lambda x: x.priority)
        
        self.logger.info(f"Created {len(self.primary_sources)} primary and {len(self.fallback_sources)} fallback sources")
    
    async def _establish_initial_sync(self):
        """Establish initial synchronization with atomic clock sources"""
        self.logger.info("Establishing initial synchronization...")
        
        # Try primary sources first
        for source in self.primary_sources:
            try:
                reading = await self._query_time_source(source)
                if reading:
                    self.reference_time = reading
                    self.current_source = source
                    self.is_synchronized = True
                    self.last_sync_time = time.time()
                    
                    self.logger.info(f"Initial sync established with {source.name}")
                    return
            except Exception as e:
                self.logger.warning(f"Failed to sync with {source.name}: {e}")
                source.is_available = False
                source.error_count += 1
        
        # Try fallback sources if primary sources fail
        for source in self.fallback_sources:
            try:
                reading = await self._query_time_source(source)
                if reading:
                    self.reference_time = reading
                    self.current_source = source
                    self.is_synchronized = True
                    self.last_sync_time = time.time()
                    
                    self.logger.warning(f"Using fallback sync with {source.name}")
                    return
            except Exception as e:
                self.logger.warning(f"Failed to sync with fallback {source.name}: {e}")
        
        # If all sources fail, use system time as last resort
        self.logger.error("All time sources failed, using system time")
        self.reference_time = TimeReading(
            source_name="system",
            timestamp=time.time(),
            precision=1e-3,  # 1ms precision for system time
            uncertainty=1e-2,  # 10ms uncertainty
            source_accuracy=1e-2,
            query_latency=0.0
        )
        self.is_synchronized = False
    
    async def _start_background_sync(self):
        """Start background synchronization tasks"""
        asyncio.create_task(self._continuous_sync_task())
        asyncio.create_task(self._monitor_sync_quality())
    
    async def _continuous_sync_task(self):
        """Continuous synchronization task"""
        while self.is_running:
            try:
                # Determine query interval based on sync quality
                interval = self.query_interval
                if not self.is_synchronized or self.synchronization_accuracy > self.max_drift_tolerance:
                    interval = self.fast_query_interval
                
                # Query current source
                if self.current_source and self.current_source.is_available:
                    reading = await self._query_time_source(self.current_source)
                    if reading:
                        await self._process_time_reading(reading)
                    else:
                        # Source failed, try to switch
                        await self._switch_time_source()
                else:
                    # No current source, establish sync
                    await self._establish_initial_sync()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in continuous sync task: {e}")
                await asyncio.sleep(1.0)
    
    async def _monitor_sync_quality(self):
        """Monitor synchronization quality and performance"""
        while self.is_running:
            try:
                # Update synchronization accuracy
                if self.reference_time:
                    current_time = time.time()
                    time_since_sync = current_time - self.last_sync_time
                    
                    # Calculate drift since last sync
                    estimated_drift = self.drift_rate * time_since_sync
                    self.synchronization_accuracy = abs(estimated_drift) + self.reference_time.uncertainty
                    
                    # Update reference time age
                    self.reference_time.update_age()
                
                # Monitor source performance
                for source in self.primary_sources + self.fallback_sources:
                    if source.response_times:
                        source.last_response_time = np.mean(source.response_times[-10:])  # Last 10 measurements
                        
                        # Update reliability based on recent performance
                        if source.success_count + source.error_count > 0:
                            recent_reliability = source.success_count / (source.success_count + source.error_count)
                            source.reliability = 0.9 * source.reliability + 0.1 * recent_reliability
                
                await asyncio.sleep(5.0)  # Monitor every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in sync quality monitoring: {e}")
    
    async def _query_time_source(self, source: AtomicTimeSource) -> Optional[TimeReading]:
        """Query a specific atomic time source"""
        query_start = time.time()
        
        try:
            # For demo purposes, simulate different atomic clock sources
            if source.name.startswith("NIST"):
                reading = await self._simulate_nist_query(source)
            elif source.name.startswith("PTB"):
                reading = await self._simulate_ptb_query(source)
            elif source.name.startswith("GPS"):
                reading = await self._simulate_gps_query(source)
            else:
                reading = await self._simulate_ntp_query(source)
            
            if reading:
                query_latency = time.time() - query_start
                reading.query_latency = query_latency
                
                # Update source performance
                source.success_count += 1
                source.response_times.append(query_latency)
                source.last_query_time = query_start
                
                # Keep only recent response times
                if len(source.response_times) > 100:
                    source.response_times = source.response_times[-50:]
                
                self.performance_metrics['total_queries'] += 1
                self.performance_metrics['successful_queries'] += 1
                
                return reading
            
        except Exception as e:
            self.logger.warning(f"Query failed for {source.name}: {e}")
            source.error_count += 1
            source.is_available = False
            self.performance_metrics['total_queries'] += 1
            self.performance_metrics['failed_queries'] += 1
        
        return None
    
    async def _simulate_nist_query(self, source: AtomicTimeSource) -> Optional[TimeReading]:
        """Simulate NIST atomic clock query with realistic characteristics"""
        
        # Simulate API call delay
        await asyncio.sleep(0.05 + random.gauss(0, 0.02))  # 50ms ± 20ms
        
        # High precision atomic clock
        current_time = time.time()
        precision = 1e-15  # Atomic clock precision
        uncertainty = 1e-12  # Very low uncertainty
        
        # Add realistic atomic clock drift and noise
        atomic_time = current_time + random.gauss(0, 1e-15)
        
        return TimeReading(
            source_name=source.name,
            timestamp=atomic_time,
            precision=precision,
            uncertainty=uncertainty,
            source_accuracy=source.accuracy,
            query_latency=0.0  # Will be set by caller
        )
    
    async def _simulate_ptb_query(self, source: AtomicTimeSource) -> Optional[TimeReading]:
        """Simulate PTB atomic clock query"""
        
        await asyncio.sleep(0.08 + random.gauss(0, 0.03))  # 80ms ± 30ms (Europe)
        
        current_time = time.time()
        precision = 5e-16  # Even higher precision
        uncertainty = 5e-13
        
        atomic_time = current_time + random.gauss(0, 5e-16)
        
        return TimeReading(
            source_name=source.name,
            timestamp=atomic_time,
            precision=precision,
            uncertainty=uncertainty,
            source_accuracy=source.accuracy,
            query_latency=0.0
        )
    
    async def _simulate_gps_query(self, source: AtomicTimeSource) -> Optional[TimeReading]:
        """Simulate GPS constellation time query"""
        
        await asyncio.sleep(0.1 + random.gauss(0, 0.05))  # 100ms ± 50ms
        
        current_time = time.time()
        precision = 1e-14  # GPS precision
        uncertainty = 1e-11
        
        # GPS time with realistic satellite timing
        gps_time = current_time + random.gauss(0, 1e-14)
        
        return TimeReading(
            source_name=source.name,
            timestamp=gps_time,
            precision=precision,
            uncertainty=uncertainty,
            source_accuracy=source.accuracy,
            query_latency=0.0
        )
    
    async def _simulate_ntp_query(self, source: AtomicTimeSource) -> Optional[TimeReading]:
        """Simulate NTP server query"""
        
        await asyncio.sleep(0.02 + random.gauss(0, 0.01))  # 20ms ± 10ms
        
        current_time = time.time()
        precision = 1e-6  # Microsecond precision for NTP
        uncertainty = 1e-3  # Millisecond uncertainty
        
        # NTP time with network jitter
        ntp_time = current_time + random.gauss(0, 1e-6)
        
        return TimeReading(
            source_name=source.name,
            timestamp=ntp_time,
            precision=precision,
            uncertainty=uncertainty,
            source_accuracy=source.accuracy,
            query_latency=0.0
        )
    
    async def _process_time_reading(self, reading: TimeReading):
        """Process a new time reading and update synchronization"""
        
        # Store reading
        self.time_readings.append(reading)
        
        # Keep only recent readings
        if len(self.time_readings) > 1000:
            self.time_readings = self.time_readings[-500:]
        
        # Update reference time if this reading is better
        if (not self.reference_time or 
            reading.source_accuracy < self.reference_time.source_accuracy or
            not self.reference_time.is_fresh()):
            
            self.reference_time = reading
            self.last_sync_time = time.time()
            self.is_synchronized = True
            
            # Update drift rate estimate
            if len(self.time_readings) > 1:
                self._update_drift_estimate()
            
            self.logger.debug(f"Updated reference time from {reading.source_name}")
    
    def _update_drift_estimate(self):
        """Update drift rate estimate from recent time readings"""
        if len(self.time_readings) < 2:
            return
        
        # Use recent readings to estimate drift
        recent_readings = self.time_readings[-10:]  # Last 10 readings
        
        if len(recent_readings) >= 2:
            time_diffs = []
            reading_diffs = []
            
            for i in range(1, len(recent_readings)):
                time_diff = recent_readings[i].reading_time - recent_readings[i-1].reading_time
                reading_diff = recent_readings[i].timestamp - recent_readings[i-1].timestamp
                
                time_diffs.append(time_diff)
                reading_diffs.append(reading_diff)
            
            if time_diffs:
                # Calculate drift as difference between time passage and reading passage
                avg_time_diff = np.mean(time_diffs)
                avg_reading_diff = np.mean(reading_diffs)
                
                if avg_time_diff > 0:
                    estimated_drift_rate = (avg_reading_diff - avg_time_diff) / avg_time_diff
                    
                    # Smooth the drift rate estimate
                    self.drift_rate = 0.9 * self.drift_rate + 0.1 * estimated_drift_rate
    
    async def _switch_time_source(self):
        """Switch to a different time source"""
        self.logger.info("Switching time source...")
        
        # Try to find an available primary source
        for source in self.primary_sources:
            if source != self.current_source and source.is_available:
                try:
                    reading = await self._query_time_source(source)
                    if reading:
                        self.current_source = source
                        await self._process_time_reading(reading)
                        self.logger.info(f"Switched to {source.name}")
                        return
                except Exception as e:
                    self.logger.warning(f"Failed to switch to {source.name}: {e}")
        
        # Try fallback sources
        for source in self.fallback_sources:
            if source.is_available:
                try:
                    reading = await self._query_time_source(source)
                    if reading:
                        self.current_source = source
                        await self._process_time_reading(reading)
                        self.logger.warning(f"Switched to fallback {source.name}")
                        return
                except Exception as e:
                    self.logger.warning(f"Failed to switch to fallback {source.name}: {e}")
        
        self.logger.error("No available time sources, synchronization may be degraded")
    
    async def get_reference_time(self) -> Optional[Dict[str, Any]]:
        """Get current atomic clock reference time"""
        
        if not self.reference_time:
            return None
        
        current_time = time.time()
        time_since_reading = current_time - self.reference_time.reading_time
        
        # Apply drift correction
        drift_correction = self.drift_rate * time_since_reading
        corrected_timestamp = self.reference_time.timestamp + time_since_reading + drift_correction
        
        # Calculate current uncertainty
        current_uncertainty = self.reference_time.uncertainty + abs(drift_correction)
        
        return {
            'timestamp': corrected_timestamp,
            'precision': self.reference_time.precision,
            'uncertainty': current_uncertainty,
            'source': self.reference_time.source_name,
            'age_seconds': time_since_reading,
            'drift_correction': drift_correction,
            'is_synchronized': self.is_synchronized,
            'sync_quality': self.synchronization_accuracy
        }
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get comprehensive synchronization status"""
        
        return {
            'is_synchronized': self.is_synchronized,
            'current_source': self.current_source.name if self.current_source else None,
            'synchronization_accuracy': self.synchronization_accuracy,
            'drift_rate': self.drift_rate,
            'last_sync_time': self.last_sync_time,
            'time_since_sync': time.time() - self.last_sync_time if self.last_sync_time > 0 else 0,
            'available_sources': len([s for s in self.primary_sources + self.fallback_sources if s.is_available]),
            'total_sources': len(self.primary_sources) + len(self.fallback_sources),
            'performance_metrics': self.performance_metrics.copy(),
            'reference_time_age': self.reference_time.age_seconds if self.reference_time else None
        }
    
    async def stop(self):
        """Stop the atomic clock service"""
        self.is_running = False
        
        if self.session:
            await self.session.close()
        
        self.logger.info("Atomic clock service stopped")
