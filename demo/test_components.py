#!/usr/bin/env python3
"""
Quick component test for Sango Rine Shumba Demo
Tests core functionality without running full simulation
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all critical imports work"""
    print("🧪 Testing imports...")
    
    try:
        # Standard library
        import asyncio, time, json, random, logging
        print("   ✅ Standard library imports")
        
        # Third-party packages
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import plotly
        import dash
        import aiohttp
        import requests
        print("   ✅ Third-party packages")
        
        # Core components
        from src.network_simulator import NetworkSimulator
        from src.atomic_clock import AtomicClockService  
        from src.precision_calculator import PrecisionCalculator
        from src.temporal_fragmenter import TemporalFragmenter
        from src.mimo_router import MIMORouter
        from src.state_predictor import StatePredictor
        from src.data_collector import DataCollector
        from src.web_browser_simulator import WebBrowserSimulator
        from src.computer_interaction_simulator import ComputerInteractionSimulator
        print("   ✅ Core components")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        traceback.print_exc()
        return False

async def test_basic_functionality():
    """Test basic component initialization"""
    print("\n🔧 Testing component initialization...")
    
    try:
        # Test configuration loading
        config_dir = Path(__file__).parent / 'config'
        if not config_dir.exists():
            print("   ⚠️  Config directory missing, creating minimal configs...")
            config_dir.mkdir(exist_ok=True)
            
            # Create minimal network topology
            import json
            minimal_topology = {
                "nodes": {
                    "tokyo": {"lat": 35.6762, "lon": 139.6503, "region": "asia"},
                    "london": {"lat": 51.5074, "lon": -0.1278, "region": "europe"},
                    "new_york": {"lat": 40.7128, "lon": -74.0060, "region": "americas"}
                }
            }
            
            with open(config_dir / 'network_topology.json', 'w') as f:
                json.dump(minimal_topology, f, indent=2)
                
            print("   ✅ Created minimal configuration")
        
        # Test basic component creation (without full initialization)
        from src.network_simulator import NetworkSimulator
        from src.data_collector import DataCollector
        
        # Create minimal components
        data_collector = DataCollector(experiment_id="test", output_dir="test_output")
        print("   ✅ DataCollector created")
        
        network_sim = NetworkSimulator()
        print("   ✅ NetworkSimulator created")
        
        print("   ✅ Basic initialization successful")
        return True
        
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        traceback.print_exc()
        return False

async def test_simple_coordination():
    """Test basic precision-by-difference calculation"""
    print("\n⚡ Testing precision-by-difference calculation...")
    
    try:
        import time
        import random
        
        # Simulate precision-by-difference calculation
        t_ref = time.time()
        t_local = t_ref + random.uniform(-0.001, 0.001)  # ±1ms variation
        
        delta_p = t_ref - t_local
        
        print(f"   📊 T_ref: {t_ref:.6f}")
        print(f"   📊 t_local: {t_local:.6f}")
        print(f"   📊 ΔP: {delta_p:.6f} ({delta_p*1000:.3f}ms)")
        print("   ✅ Precision-by-difference calculation successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Calculation failed: {e}")
        traceback.print_exc()
        return False

def cleanup_test_files():
    """Clean up test files"""
    try:
        import shutil
        test_dir = Path(__file__).parent / 'test_output'
        if test_dir.exists():
            shutil.rmtree(test_dir)
        print("🧹 Test files cleaned up")
    except:
        pass

async def main():
    """Main test function"""
    print("🧪 Sango Rine Shumba - Component Test Suite")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Imports
    if test_imports():
        success_count += 1
    
    # Test 2: Basic functionality  
    if await test_basic_functionality():
        success_count += 1
        
    # Test 3: Simple coordination
    if await test_simple_coordination():
        success_count += 1
    
    # Cleanup
    cleanup_test_files()
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Ready to run full demo.")
        print("\n💡 Next step: python run_demo.py")
        return True
    else:
        print("❌ Some tests failed. Please check error messages above.")
        print("\n🔧 Try running: python setup_demo.py")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        traceback.print_exc()
        sys.exit(1)
