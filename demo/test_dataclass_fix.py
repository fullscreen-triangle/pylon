#!/usr/bin/env python3
"""
Test script to verify BiometricProfile dataclass is fixed
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_biometric_profile():
    """Test BiometricProfile creation"""
    print("Testing BiometricProfile import...")
    
    try:
        from src.computer_interaction_simulator import BiometricProfile
        print("✅ BiometricProfile imported successfully")
        
        # Test creating an instance
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
            application_switching_pattern=["Browser", "Editor"],
            window_management_style="tiled",
            multitasking_frequency=5.0,
            stress_response_factor=0.4,
            fatigue_degradation_rate=0.2,
            consistency_score=0.85
        )
        
        print(f"✅ BiometricProfile created: {profile.user_id}")
        print(f"✅ Biometric hash: {profile.biometric_hash}")
        print(f"✅ Typing signature length: {len(profile.typing_rhythm_signature)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_biometric_profile():
        print("\n🎉 BiometricProfile dataclass is fixed!")
    else:
        print("\n💥 BiometricProfile still has issues")
