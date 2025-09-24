# web_browser_simulator.py - Fixed version
import asyncio
import time
import random
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import numpy as np
from collections import deque
import uuid


@dataclass
class WebPage:
    """Complete web page with realistic loading characteristics"""

    page_id: str
    url: str
    html_size_kb: float
    css_size_kb: float
    js_size_kb: float
    image_size_kb: float

    # Rendering characteristics
    dom_elements: int
    js_execution_time_ms: float
    css_layout_time_ms: float
    paint_time_ms: float

    # Interactive elements
    interactive_elements: List[str] = field(default_factory=list)
    form_fields: List[str] = field(default_factory=list)

    # Performance metadata
    critical_rendering_path_ms: float = 0.0
    time_to_interactive_ms: float = 0.0
    largest_contentful_paint_ms: float = 0.0

    def __post_init__(self):
        """Calculate derived properties"""
        self.total_size_kb = (self.html_size_kb + self.css_size_kb +
                              self.js_size_kb + self.image_size_kb)
        self.complexity_score = self._calculate_complexity()
        self._calculate_performance_metrics()

    def _calculate_complexity(self) -> float:
        """Calculate page complexity score"""
        size_factor = min(self.total_size_kb / 1000.0, 2.0)  # Cap at 2x
        dom_factor = min(self.dom_elements / 1000.0, 3.0)  # Cap at 3x
        js_factor = min(self.js_execution_time_ms / 1000.0, 2.0)  # Cap at 2x

        return (size_factor + dom_factor + js_factor) / 3.0

    def _calculate_performance_metrics(self):
        """Calculate realistic performance timing"""
        # Critical rendering path
        self.critical_rendering_path_ms = (
                self.css_layout_time_ms +
                self.paint_time_ms +
                (self.html_size_kb * 0.1)  # HTML parsing time
        )

        # Time to interactive
        self.time_to_interactive_ms = (
                self.critical_rendering_path_ms +
                self.js_execution_time_ms
        )

        # Largest contentful paint
        self.largest_contentful_paint_ms = (
                self.critical_rendering_path_ms +
                (self.image_size_kb * 0.05)  # Image loading time
        )


class WebBrowserSimulator:
    """Enhanced web browser simulator with realistic performance modeling"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pages: Dict[str, WebPage] = {}
        self.loading_history: List[Dict[str, Any]] = []
        self.network_simulator = None
        self.temporal_coordinator = None

    async def simulate_traditional_loading(self, page: WebPage, bandwidth_mbps: float = 100.0) -> Dict[str, Any]:
        """Simulate traditional web page loading with realistic timing"""
        start_time = time.perf_counter()

        # DNS resolution (20-200ms)
        dns_time = random.uniform(0.02, 0.2)
        await asyncio.sleep(dns_time)

        # TCP connection establishment (RTT dependent)
        tcp_time = random.uniform(0.01, 0.1)
        await asyncio.sleep(tcp_time)

        # HTML download
        html_download_time = (page.html_size_kb * 8) / (bandwidth_mbps * 1000)  # Convert to seconds
        await asyncio.sleep(html_download_time)

        # HTML parsing (parallel with resource discovery)
        html_parse_time = page.html_size_kb * 0.0001  # 0.1ms per KB
        await asyncio.sleep(html_parse_time)

        # CSS download and parsing
        css_download_time = (page.css_size_kb * 8) / (bandwidth_mbps * 1000)
        await asyncio.sleep(css_download_time + page.css_layout_time_ms / 1000)

        # JavaScript download and execution
        js_download_time = (page.js_size_kb * 8) / (bandwidth_mbps * 1000)
        await asyncio.sleep(js_download_time + page.js_execution_time_ms / 1000)

        # Image loading (can be parallel, but affects LCP)
        image_download_time = (page.image_size_kb * 8) / (bandwidth_mbps * 1000)
        await asyncio.sleep(image_download_time)

        # Final paint
        await asyncio.sleep(page.paint_time_ms / 1000)

        total_time = time.perf_counter() - start_time

        return {
            'page_id': page.page_id,
            'url': page.url,
            'total_load_time_ms': total_time * 1000,
            'dns_time_ms': dns_time * 1000,
            'tcp_time_ms': tcp_time * 1000,
            'html_time_ms': (html_download_time + html_parse_time) * 1000,
            'css_time_ms': (css_download_time + page.css_layout_time_ms),
            'js_time_ms': (js_download_time + page.js_execution_time_ms),
            'image_time_ms': image_download_time * 1000,
            'paint_time_ms': page.paint_time_ms,
            'method': 'traditional'
        }

    async def simulate_sango_streaming(self, page: WebPage) -> Dict[str, Any]:
        """Simulate Sango Rine Shumba temporal streaming"""
        start_time = time.perf_counter()

        # Preemptive state already delivered (negative latency)
        preemptive_advantage = random.uniform(0.05, 0.15)  # 50-150ms advantage

        # Temporal fragmentation allows parallel delivery
        fragment_coordination_time = 0.002  # 2ms for coordination
        await asyncio.sleep(fragment_coordination_time)

        # All resources arrive simultaneously through temporal coordination
        simultaneous_delivery_time = max(
            page.critical_rendering_path_ms / 1000,
            0.01  # Minimum 10ms for coordination
        ) * 0.2  # 80% reduction through temporal streaming

        await asyncio.sleep(simultaneous_delivery_time)

        total_time = (time.perf_counter() - start_time) - preemptive_advantage

        return {
            'page_id': page.page_id,
            'url': page.url,
            'total_load_time_ms': max(total_time * 1000, 1.0),  # Minimum 1ms
            'preemptive_advantage_ms': preemptive_advantage * 1000,
            'coordination_time_ms': fragment_coordination_time * 1000,
            'streaming_time_ms': simultaneous_delivery_time * 1000,
            'improvement_percentage': 0.0,  # Will be calculated later
            'method': 'sango_streaming'
        }
