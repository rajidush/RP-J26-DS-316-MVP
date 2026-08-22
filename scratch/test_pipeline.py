import sys
import os
import time

# Add the backend path to sys.path so we can import guard
backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "offline_backend")
sys.path.append(backend_path)

from guard import ZeroTrustGuard

def test_pipeline():
    print("=== STARTING ZERO-TRUST VIDEO GUARD PIPELINE TESTS ===")
    guard = ZeroTrustGuard()
    
    # 1. Test simulation state loading
    print("\n--- 1. Testing Simulation Mode State Initialization ---")
    guard.set_simulation_mode(True)
    state = guard.get_state()
    print(f"Simulation Mode Active: {state['simulation_mode']}")
    print(f"Initial Threat Score: {state['threat_score']}")
    print(f"Initial Threat Type: {state['threat_type']}")
    print(f"Initial Processes Tracked: {len(state['processes'])}")
    print(f"Initial Objects Tracked: {len(state['objects'])}")
    
    # 2. Test simulation step execution
    print("\n--- 2. Testing Simulation Step Progression ---")
    guard.start()
    time.sleep(1.2) # Allow thread to run at least one step
    state_after = guard.get_state()
    print(f"Threat Score after step: {state_after['threat_score']}")
    print(f"Objects detected: {[o['label'] for o in state_after['objects']]}")
    print(f"Active playback: {state_after['playback']['smtc_title']}")
    
    # 3. Test Live Mode compatibility (checks psutil, window tracking, screen grab fallbacks)
    print("\n--- 3. Testing Live Mode System Hooks ---")
    guard.set_simulation_mode(False)
    time.sleep(1.2) # Allow live scanner loop to run
    live_state = guard.get_state()
    print(f"Live Mode Active (Sim Mode = False): {not live_state['simulation_mode']}")
    print(f"Real-Time Processes Found: {len(live_state['processes'])}")
    print(f"Top 5 processes:")
    for proc in live_state['processes'][:5]:
        print(f"  - PID {proc['pid']}: {proc['name']} (Mimics Browser: {proc['is_browser']}, Anomaly: {proc['is_anomaly']})")
        
    print(f"\nReal-Time Playback Status:")
    print(f"  - SMTC Playing: {live_state['playback']['smtc_active']}")
    print(f"  - Active Session Title: {live_state['playback']['smtc_title']}")
    print(f"  - GPU Decode Load: {live_state['playback']['gpu_decode_load']}%")
    print(f"  - Fullscreen Mode: {live_state['playback']['fullscreen_active']}")
    
    print(f"\nScene Change Calculator:")
    print(f"  - Frame change ratio: {live_state['scene_change']}%")
    print(f"  - Grabbed Image size: {len(live_state['last_frame'])} bytes (Base64)")
    
    print(f"\nLocal Inference Engine:")
    print(f"  - Caffe Net Loaded: {live_state['model_loaded']}")
    print(f"  - Live Detections: {[o['label'] for o in live_state['objects']]}")
    print(f"  - Threat Score: {live_state['threat_score']}")
    
    guard.stop()
    print("\n=== PIPELINE VERIFICATION COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_pipeline()
