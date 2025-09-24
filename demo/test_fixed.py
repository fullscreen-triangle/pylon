#!/usr/bin/env python3
"""
Test script for fixed dataclass issues
Verifies that the computer interaction simulator can be imported without errors
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test critical imports"""
    print("🧪 Testing fixed imports...")
    
    try:
        # Test the fixed dataclass
        from src.computer_interaction_simulator import BiometricProfile, ComputerInteractionSimulator
        print("   ✅ BiometricProfile dataclass fixed")
        print("   ✅ ComputerInteractionSimulator imports successfully")
        
        # Test other core components
        from src.network_simulator import NetworkSimulator
        from src.data_collector import DataCollector
        from src.precision_calculator import PrecisionCalculator
        print("   ✅ Core components import successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_biometric_profile_creation():
    """Test creating a BiometricProfile instance"""
    print("\n🔧 Testing BiometricProfile creation...")
    
    try:
        from src.computer_interaction_simulator import BiometricProfile
        
        # Create a test profile with all required fields
        profile = BiometricProfile(
            user_id="test_user",
            profile_name="Test User",
            average_typing_speed_wpm=75.0,
            keystroke_dwell_time_ms=120.0,
            keystroke_flight_time_ms=150.0,
            mouse_velocity_profile=(600.0, 100.0),
            mouse_acceleration_signature=0.8,
            click_pressure_profile=(0.6, 0.1),
            drag_smoothness_factor=0.75,
            saccade_velocity_deg_per_sec=400.0,
            fixation_duration_ms=250.0,
            blink_rate_per_minute=20.0,
            reading_pattern="left_to_right",
            scroll_speed_preference=100.0,
            application_switching_pattern=["Browser", "Editor", "Terminal"],
            window_management_style="tiled",
            multitasking_frequency=5.0,
            stress_response_factor=0.4,
            fatigue_degradation_rate=0.2,
            consistency_score=0.85
            # typing_rhythm_signature will be auto-generated
        )
        
        print(f"   ✅ Profile created: {profile.user_id}")
        print(f"   ✅ Typing speed: {profile.average_typing_speed_wpm} WPM")
        print(f"   ✅ Mouse velocity: {profile.mouse_velocity_profile}")
        print(f"   ✅ Biometric hash: {profile.biometric_hash}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Profile creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simplified_demo_import():
    """Test that simplified demo can be imported"""
    print("\n📊 Testing simplified demo import...")
    
    try:
        # This should work without dashboard dependencies
        from run_simple_demo import SimplifiedSangoRineShumbaDemo
        print("   ✅ SimplifiedSangoRineShumbaDemo imports successfully")
        
        # Test instantiation (without running)
        demo = SimplifiedSangoRineShumbaDemo()
        print(f"   ✅ Demo instance created with ID: {demo.experiment_id}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Simplified demo import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🔧 Testing Fixed Sango Rine Shumba Components")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Basic imports
    if test_imports():
        success_count += 1
    
    # Test 2: BiometricProfile creation
    if test_biometric_profile_creation():
        success_count += 1
    
    # Test 3: Simplified demo import  
    if test_simplified_demo_import():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All fixes successful! Ready to run demo.")
        print(f"\n💡 Run the demo with:")
        print(f"   python run_simple_demo.py")
        return True
    else:
        print("❌ Some issues remain. Check error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
