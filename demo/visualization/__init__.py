"""
Sango Rine Shumba Network Simulation - Visualization Package

This package contains comprehensive visualization components for real-time
display and analysis of Sango Rine Shumba temporal coordination demonstration.

Modules:
- performance_dashboard: Real-time performance dashboard with live metrics
- network_visualizer: Interactive network topology with temporal coordination
- precision_plotter: Precision-by-difference calculation visualization
- fragment_tracker: Message fragmentation and reconstruction tracking
- comparative_analyzer: Traditional vs. Sango Rine Shumba comparison
"""

__version__ = "1.0.0"
__author__ = "Kundai Farai Sachikonye"
__description__ = "Sango Rine Shumba Visualization Suite"

# Visualization component imports
from .performance_dashboard import PerformanceDashboard
from .network_visualizer import NetworkVisualizer

__all__ = [
    "PerformanceDashboard",
    "NetworkVisualizer"
]
