"""
Sango Rine Shumba Network Simulation - Core Source Package

This package contains the core implementation modules for the Sango Rine Shumba
temporal coordination framework demonstration.

Modules:
- network_simulator: Global network topology simulation with realistic physics
- atomic_clock: Real atomic clock API integration for temporal reference
- precision_calculator: Precision-by-difference calculation engine
- temporal_fragmenter: Message fragmentation across temporal coordinates
- mimo_router: MIMO-like multi-path routing system
- state_predictor: Preemptive interface state prediction
- data_collector: Comprehensive data collection and analysis
"""

__version__ = "1.0.0"
__author__ = "Kundai Farai Sachikonye"
__description__ = "Sango Rine Shumba Temporal Coordination Framework"

# Core component imports
from .network_simulator import NetworkSimulator, NetworkNode
from .atomic_clock import AtomicClockService
from .precision_calculator import PrecisionCalculator
from .temporal_fragmenter import TemporalFragmenter, MessageFragment
from .mimo_router import MIMORouter
from .state_predictor import StatePredictor
from .data_collector import DataCollector
from .web_browser_simulator import WebBrowserSimulator
from .computer_interaction_simulator import ComputerInteractionSimulator

__all__ = [
    "NetworkSimulator",
    "NetworkNode", 
    "AtomicClockService",
    "PrecisionCalculator",
    "TemporalFragmenter",
    "MessageFragment",
    "MIMORouter",
    "StatePredictor",
    "DataCollector",
    "WebBrowserSimulator",
    "ComputerInteractionSimulator"
]
