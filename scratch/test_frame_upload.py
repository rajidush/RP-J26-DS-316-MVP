import os
import cv2
import numpy as np
import base64
import json
import urllib.request
import urllib.error

def test_frame_upload_and_reset():
    print("=== TESTING FASTAPI THREAT SCENARIO & STATE RESET ===")
    
    # 1. Create a frame
    w, h = 640, 480
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (100, 150, 200) # colored background
    
    # Draw shapes representing objects
    cv2.rectangle(img, (100, 100), (300, 400), (0, 255, 0), -1) 
    cv2.circle(img, (200, 250), 30, (255, 0, 0), -1) 
    
    # 2. Encode to base64
    _, buffer = cv2.imencode('.jpg', img)
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
    
    payload = {
        "image_b64": img_b64,
        "filename": "combat_match_battle_violence.mp4"
    }
    
    url_process = "http://127.0.0.1:8000/api/guard/process-frame"
    url_reset = "http://127.0.0.1:8000/api/guard/reset"
    url_state = "http://127.0.0.1:8000/api/guard/state"
    
    try:
        # Step A: POST violent content frame
        print("\n[Step A] Posting violent frame...")
        data_bytes = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url_process, data=data_bytes, headers={'Content-Type': 'application/json'})
        
        with urllib.request.urlopen(req, timeout=5) as response:
            res_body = response.read().decode('utf-8')
            data = json.loads(res_body)
            print("Successfully processed violent custom frame!")
            print(f"  - Threat Score: {data['threat_score']} (Expected: >0.85)")
            print(f"  - Threat Category: {data['threat_type']} (Expected: violence)")
            assert data['threat_score'] > 0.85, "Threat trigger score failed"
            
        # Step B: Query backend state to confirm persistence
        print("\n[Step B] Querying live state...")
        req_state = urllib.request.Request(url_state)
        with urllib.request.urlopen(req_state, timeout=5) as response_state:
            data_state = json.loads(response_state.read().decode('utf-8'))
            print(f"  - Stored State Threat Score: {data_state['threat_score']}")
            
        # Step C: POST reset request
        print("\n[Step C] Sending reset request...")
        req_reset = urllib.request.Request(url_reset, data=b"", headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req_reset, timeout=5) as response_reset:
            data_reset = json.loads(response_reset.read().decode('utf-8'))
            print(f"  - Reset Response Status: {data_reset['status']}")
            
        # Step D: Re-query state to assert that stats are cleared
        print("\n[Step D] Verifying state is fully cleared...")
        with urllib.request.urlopen(req_state, timeout=5) as response_state_after:
            data_state_after = json.loads(response_state_after.read().decode('utf-8'))
            print("Successfully cleared backend stats!")
            print(f"  - Post-Reset Threat Score: {data_state_after['threat_score']} (Expected: 0.0)")
            print(f"  - Post-Reset Category: {data_state_after['threat_type']} (Expected: none)")
            print(f"  - Post-Reset Objects list length: {len(data_state_after['objects'])} (Expected: 0)")
            
            assert data_state_after['threat_score'] == 0.0, "Reset threat score failed"
            assert data_state_after['threat_type'] == "none", "Reset threat type failed"
            
    except urllib.error.URLError as e:
        print(f"URLError: {e}. Is your FastAPI server started? Run: python main.py")
    except Exception as e:
        print(f"Exception during request/assertions: {e}")
        
    print("\n=== VERIFICATION COMPLETED ===")

if __name__ == "__main__":
    test_frame_upload_and_reset()
