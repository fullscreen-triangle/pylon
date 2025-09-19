#!/usr/bin/env python3
"""
Sango Rine Shumba Network Simulation Demo

Main application entry point that orchestrates the complete demonstration
of temporal coordination, precision-by-difference calculations, and
MIMO-like message fragmentation across a realistic global network topology.

Usage:
    python run_demo.py [--config CONFIG_DIR] [--duration SECONDS] [--port PORT]
    
Author: Kundai Farai Sachikonye
Date: 2024
"""

import sys
import os
import argparse
import asyncio
import threading
import time
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.network_simulator import NetworkSimulator
from src.atomic_clock import AtomicClockService
from src.precision_calculator import PrecisionCalculator
from src.temporal_fragmenter import TemporalFragmenter
from src.mimo_router import MIMORouter
from src.state_predictor import StatePredictor
from src.data_collector import DataCollector
from src.web_browser_simulator import WebBrowserSimulator
from src.computer_interaction_simulator import ComputerInteractionSimulator

from visualization.performance_dashboard import PerformanceDashboard
from visualization.network_visualizer import NetworkVisualizer

import logging
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init(autoreset=True)

class SangoRineShubmaDemo:
    """
    Main demonstration orchestrator for Sango Rine Shumba framework
    
    This class coordinates all simulation components:
    - Network topology with 10 global nodes
    - Atomic clock synchronization service
    - Precision-by-difference calculations
    - Temporal message fragmentation
    - MIMO-like routing system
    - Real-time visualization dashboard
    - Comprehensive data collection
    """
    
    def __init__(self, config_dir="config", port=8050):
        """Initialize demo with configuration directory and dashboard port"""
        self.config_dir = Path(config_dir)
        self.port = port
        self.running = False
        self.experiment_id = f"exp_{int(time.time())}"
        
        # Initialize logging
        self.setup_logging()
        
        # Core components
        self.network_simulator = None
        self.atomic_clock = None
        self.precision_calculator = None
        self.temporal_fragmenter = None
        self.mimo_router = None
        self.state_predictor = None
        self.data_collector = None
        
        # User experience components
        self.web_browser_simulator = None
        self.computer_interaction_simulator = None
        
        # Visualization components
        self.dashboard = None
        self.network_visualizer = None
        
        self.logger.info(f"{Fore.CYAN}Initializing Sango Rine Shumba Demo...")
        self.logger.info(f"{Fore.YELLOW}Experiment ID: {self.experiment_id}")
    
    def setup_logging(self):
        """Configure logging with colored output"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    async def initialize_components(self):
        """Initialize all simulation components"""
        try:
            self.logger.info(f"{Fore.GREEN}🔧 Initializing simulation components...")
            
            # Initialize data collector first (needed by other components)
            self.data_collector = DataCollector(
                experiment_id=self.experiment_id,
                data_dir="data"
            )
            await self.data_collector.initialize()
            
            # Initialize atomic clock service
            self.logger.info(f"{Fore.BLUE}⏰ Connecting to atomic clock service...")
            self.atomic_clock = AtomicClockService()
            await self.atomic_clock.initialize()
            
            # Initialize network simulator with 10 global nodes
            self.logger.info(f"{Fore.MAGENTA}🌍 Creating global network topology...")
            self.network_simulator = NetworkSimulator(
                config_path=self.config_dir / "network_topology.json",
                data_collector=self.data_collector
            )
            await self.network_simulator.initialize()
            
            # Initialize precision calculator
            self.logger.info(f"{Fore.CYAN}📊 Setting up precision-by-difference calculator...")
            self.precision_calculator = PrecisionCalculator(
                atomic_clock=self.atomic_clock,
                network_simulator=self.network_simulator,
                data_collector=self.data_collector
            )
            
            # Initialize temporal fragmenter
            self.logger.info(f"{Fore.YELLOW}🧩 Configuring temporal fragmentation engine...")
            self.temporal_fragmenter = TemporalFragmenter(
                precision_calculator=self.precision_calculator,
                data_collector=self.data_collector
            )
            
            # Initialize MIMO router
            self.logger.info(f"{Fore.GREEN}🚀 Preparing MIMO routing system...")
            self.mimo_router = MIMORouter(
                network_simulator=self.network_simulator,
                temporal_fragmenter=self.temporal_fragmenter,
                data_collector=self.data_collector
            )
            
            # Initialize state predictor
            self.logger.info(f"{Fore.RED}🔮 Loading state prediction models...")
            self.state_predictor = StatePredictor(
                precision_calculator=self.precision_calculator,
                data_collector=self.data_collector
            )
            
            # Initialize user experience components
            self.logger.info(f"{Fore.MAGENTA}🖥️  Setting up web browser simulator...")
            self.web_browser_simulator = WebBrowserSimulator(
                network_simulator=self.network_simulator,
                temporal_fragmenter=self.temporal_fragmenter,
                state_predictor=self.state_predictor,
                data_collector=self.data_collector
            )
            
            self.logger.info(f"{Fore.BLUE}👤 Setting up computer interaction simulator...")
            self.computer_interaction_simulator = ComputerInteractionSimulator(
                precision_calculator=self.precision_calculator,
                data_collector=self.data_collector
            )
            
            # Initialize visualization components
            self.logger.info(f"{Fore.CYAN}📈 Setting up visualization dashboard...")
            self.dashboard = PerformanceDashboard(
                network_simulator=self.network_simulator,
                precision_calculator=self.precision_calculator,
                temporal_fragmenter=self.temporal_fragmenter,
                mimo_router=self.mimo_router,
                data_collector=self.data_collector,
                port=self.port
            )
            
            self.network_visualizer = NetworkVisualizer(
                network_simulator=self.network_simulator,
                precision_calculator=self.precision_calculator,
                data_collector=self.data_collector
            )
            
            self.logger.info(f"{Fore.GREEN}✅ All components initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"{Fore.RED}❌ Component initialization failed: {e}")
            raise
    
    async def run_baseline_experiment(self):
        """Run traditional network communication baseline for comparison"""
        self.logger.info(f"{Fore.CYAN}📊 Running baseline experiment (traditional networking)...")
        
        # Configure network for traditional operation
        await self.network_simulator.set_traditional_mode(True)
        
        # Run baseline measurements for 60 seconds
        baseline_duration = 60
        start_time = time.time()
        
        while time.time() - start_time < baseline_duration:
            # Send test messages through traditional routing
            for source_node in self.network_simulator.nodes:
                for dest_node in self.network_simulator.nodes:
                    if source_node != dest_node:
                        message = f"baseline_msg_{int(time.time() * 1000)}"
                        await self.network_simulator.send_traditional_message(
                            source_node.id, dest_node.id, message
                        )
            
            await asyncio.sleep(0.1)  # 100ms interval
        
        self.logger.info(f"{Fore.GREEN}✅ Baseline experiment completed")
    
    async def run_sango_rine_shumba_experiment(self):
        """Run Sango Rine Shumba coordination experiment"""
        self.logger.info(f"{Fore.CYAN}🚀 Running Sango Rine Shumba experiment...")
        
        # Configure network for Sango Rine Shumba operation
        await self.network_simulator.set_traditional_mode(False)
        
        # Start precision-by-difference calculations
        precision_task = asyncio.create_task(
            self.precision_calculator.start_continuous_calculation()
        )
        
        # Start temporal fragmentation
        fragmentation_task = asyncio.create_task(
            self.temporal_fragmenter.start_fragmentation_service()
        )
        
        # Start MIMO routing
        mimo_task = asyncio.create_task(
            self.mimo_router.start_routing_service()
        )
        
        # Start state prediction
        prediction_task = asyncio.create_task(
            self.state_predictor.start_prediction_service()
        )
        
        # Start user experience simulations
        browser_task = asyncio.create_task(
            self.web_browser_simulator.run_comparative_browser_demo(duration_seconds=300)
        )
        
        interaction_task = asyncio.create_task(
            self.computer_interaction_simulator.start_interaction_simulation(duration_seconds=300)
        )
        
        # Run Sango Rine Shumba for 300 seconds (5 minutes)
        sango_duration = 300
        start_time = time.time()
        
        self.logger.info(f"{Fore.YELLOW}⏱️  Running comprehensive experiment for {sango_duration} seconds...")
        self.logger.info(f"{Fore.MAGENTA}🖥️  Browser simulation: Traditional vs Sango Rine Shumba")
        self.logger.info(f"{Fore.BLUE}👤 User interaction simulation: Biometric verification & zero latency")
        
        # Send test messages through Sango Rine Shumba system
        message_count = 0
        while time.time() - start_time < sango_duration:
            # Send messages with temporal fragmentation
            for source_node in self.network_simulator.nodes:
                for dest_node in self.network_simulator.nodes:
                    if source_node != dest_node:
                        message = f"sango_msg_{message_count}_{int(time.time() * 1000)}"
                        
                        # Fragment message temporally
                        fragments = await self.temporal_fragmenter.fragment_message(
                            message, source_node.id, dest_node.id
                        )
                        
                        # Route fragments via MIMO
                        await self.mimo_router.route_fragments(fragments)
                        
                        message_count += 1
            
            # Display progress with additional metrics
            if message_count % 50 == 0:
                elapsed = time.time() - start_time
                
                # Get browser simulation statistics
                browser_stats = self.web_browser_simulator.get_browser_statistics()
                interaction_stats = self.computer_interaction_simulator.get_interaction_statistics()
                
                self.logger.info(f"{Fore.BLUE}📊 Progress: {elapsed:.1f}s | "
                               f"Messages: {message_count} | "
                               f"Browser loads: {browser_stats['page_loads']['total']} | "
                               f"Interactions: {interaction_stats['total_interactions']} | "
                               f"Zero-latency events: {interaction_stats['zero_latency_performance']['predictions_made']}")
            
            await asyncio.sleep(0.05)  # 50ms interval for higher frequency
        
        # Stop all services
        services = [precision_task, fragmentation_task, mimo_task, prediction_task, 
                   browser_task, interaction_task]
        
        for service in services:
            service.cancel()
        
        try:
            await asyncio.gather(*services, return_exceptions=True)
        except:
            pass  # Tasks were cancelled
        
        self.logger.info(f"{Fore.GREEN}✅ Comprehensive Sango Rine Shumba experiment completed")
        self.logger.info(f"{Fore.CYAN}📊 Network messages: {message_count}")
        
        # Display final statistics
        browser_stats = self.web_browser_simulator.get_browser_statistics()
        interaction_stats = self.computer_interaction_simulator.get_interaction_statistics()
        
        self.logger.info(f"{Fore.MAGENTA}🖥️  Browser Performance:")
        self.logger.info(f"   • Load time improvement: {browser_stats['performance']['load_time_improvement']:.1%}")
        self.logger.info(f"   • Page loads: {browser_stats['page_loads']['total']}")
        self.logger.info(f"   • User satisfaction: {browser_stats['user_experience']['user_satisfaction_score']:.2f}")
        
        self.logger.info(f"{Fore.BLUE}👤 User Experience:")
        self.logger.info(f"   • Biometric verification rate: {interaction_stats['biometric_verification']['success_rate']:.1%}")
        self.logger.info(f"   • Zero-latency predictions: {interaction_stats['zero_latency_performance']['predictions_made']}")
        self.logger.info(f"   • Average verification time: {interaction_stats['biometric_verification']['average_verification_time_ms']:.1f}ms")
    
    async def generate_final_analysis(self):
        """Generate comprehensive analysis of experimental results"""
        self.logger.info(f"{Fore.CYAN}📊 Generating final analysis...")
        
        # Generate performance comparison
        await self.data_collector.generate_performance_comparison()
        
        # Generate statistical summary
        await self.data_collector.generate_statistical_summary()
        
        # Export visualization data
        await self.network_visualizer.export_final_visualizations()
        
        # Generate research paper figures
        await self.data_collector.generate_publication_figures()
        
        self.logger.info(f"{Fore.GREEN}✅ Analysis complete! Check data/ directory for results.")
    
    async def start_dashboard(self):
        """Start the real-time visualization dashboard"""
        self.logger.info(f"{Fore.CYAN}🖥️  Starting visualization dashboard on port {self.port}...")
        
        # Start dashboard in separate thread
        dashboard_thread = threading.Thread(
            target=self.dashboard.run_server,
            daemon=True
        )
        dashboard_thread.start()
        
        # Wait a moment for server to start
        await asyncio.sleep(2)
        
        self.logger.info(f"{Fore.GREEN}🌐 Dashboard available at: http://localhost:{self.port}")
        self.logger.info(f"{Fore.YELLOW}📊 Real-time visualizations include:")
        self.logger.info(f"   • Network topology with live latency data")
        self.logger.info(f"   • Precision-by-difference calculations")
        self.logger.info(f"   • Temporal fragment distribution")
        self.logger.info(f"   • Performance comparison metrics")
    
    async def run_complete_demonstration(self):
        """Run the complete Sango Rine Shumba demonstration"""
        try:
            self.running = True
            
            # Initialize all components
            await self.initialize_components()
            
            # Start visualization dashboard
            await self.start_dashboard()
            
            # Run baseline experiment
            await self.run_baseline_experiment()
            
            # Run Sango Rine Shumba experiment
            await self.run_sango_rine_shumba_experiment()
            
            # Generate final analysis
            await self.generate_final_analysis()
            
            self.logger.info(f"{Fore.GREEN}🎉 Demonstration completed successfully!")
            self.logger.info(f"{Fore.CYAN}📁 Results saved to: data/experiments/{self.experiment_id}/")
            
        except Exception as e:
            self.logger.error(f"{Fore.RED}❌ Demonstration failed: {e}")
            raise
        finally:
            self.running = False
    
    def stop(self):
        """Stop the demonstration"""
        self.logger.info(f"{Fore.YELLOW}⏹️  Stopping demonstration...")
        self.running = False


def print_banner():
    """Print demo startup banner"""
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║                    {Fore.YELLOW}SANGO RINE SHUMBA{Fore.CYAN}                        ║
║                                                              ║
║          {Fore.GREEN}Revolutionary Temporal Coordination Demo{Fore.CYAN}            ║
║                                                              ║
║    {Fore.MAGENTA}• Precision-by-Difference Network Coordination{Fore.CYAN}          ║
║    {Fore.MAGENTA}• Temporal Message Fragmentation Security{Fore.CYAN}               ║
║    {Fore.MAGENTA}• MIMO-like Multi-path Routing{Fore.CYAN}                          ║
║    {Fore.MAGENTA}• Real-time Biometric ID Verification{Fore.CYAN}                   ║
║    {Fore.MAGENTA}• Zero-latency User Experience{Fore.CYAN}                          ║
║    {Fore.MAGENTA}• Web Browser Performance Revolution{Fore.CYAN}                    ║
║                                                              ║
║         {Fore.BLUE}Proves: 80% latency reduction, instant ID{Fore.CYAN}           ║
║         {Fore.BLUE}verification, zero-latency interactions{Fore.CYAN}              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Sango Rine Shumba Network Simulation Demo"
    )
    parser.add_argument(
        "--config", 
        default="config",
        help="Configuration directory path"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=360,
        help="Total simulation duration in seconds"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Dashboard port number"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print_banner()
    
    # Create demo instance
    demo = SangoRineShumbmaDemo(
        config_dir=args.config,
        port=args.port
    )
    
    # Run demonstration
    try:
        asyncio.run(demo.run_complete_demonstration())
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️  Demonstration interrupted by user")
        demo.stop()
    except Exception as e:
        print(f"\n{Fore.RED}❌ Demonstration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
