from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid
from socratic_agent import SocraticAgentManager

app = FastAPI(
    title="Socratic Digital Child Safety Engine",
    description="Offline Cognitive State-Machine Prototype safeguarding minors through pedagogical dialog."
)

# Enable CORS for frontend web and Electron integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our Socratic agent core
# Configured for default LM Studio local server on http://localhost:1234/v1
agent_manager = SocraticAgentManager(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Initialize Zero-Trust Guard
from guard import ZeroTrustGuard
guard_system = ZeroTrustGuard()
guard_system.start()

# --- Schemas ---

class GuardToggleRequest(BaseModel):
    simulation_mode: bool

class GuardMonitorRequest(BaseModel):
    active: bool

class FrameAnalysisRequest(BaseModel):
    image_b64: str
    filename: str = ""

class ThreatTriggerRequest(BaseModel):
    violence_score: float = Field(0.0, description="Confidence score for violence (0.0 to 1.0)")
    hate_speech_score: float = Field(0.0, description="Confidence score for hate speech (0.0 to 1.0)")
    adult_content_score: float = Field(0.0, description="Confidence score for adult content (0.0 to 1.0)")
    child_age: int = Field(10, description="The age of the child using the computer (e.g. 8 or 14)")

class ThreatTriggerResponse(BaseModel):
    threat_detected: bool = Field(..., description="Flag indicating if safety threshold was breached.")
    threat_type: str = Field("none", description="The classified category of content threat.")
    session_id: str = Field(..., description="Unique UUID to track Socratic Dialogue session.")
    initial_response: str = Field(..., description="First age-specific Socratic intervention question.")
    child_age: int
    current_phase: str

class DialogueTurnRequest(BaseModel):
    session_id: str = Field(..., description="UUID representing the active dialogue session.")
    child_response: str = Field(..., description="The typed response from the minor.")

class DialogueTurnResponse(BaseModel):
    socratic_response_to_child: str
    child_emotion: str
    agreed_to_boundary: bool
    current_phase: str
    completed: bool

# --- Endpoints ---

@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "Socratic Buddy Child Safety Interceptor",
        "engine_target": "LM Studio (http://localhost:1234/v1)"
    }

@app.post("/api/perception/trigger", response_model=ThreatTriggerResponse)
def trigger_threat(payload: ThreatTriggerRequest):
    """
    Simulates the 'Perception Layer' interceptor.
    Analyzes content threat scores. If a threshold is breached (score > 0.85),
    it flags threat_detected = True, extracts threat_type, and triggers Socratic Buddy.
    """
    # 1. Determine maximum threat score and check if threshold is breached (> 0.85)
    threats = {
        "violence": payload.violence_score,
        "hate_speech": payload.hate_speech_score,
        "adult_content": payload.adult_content_score
    }
    
    max_threat_type = "none"
    max_score = 0.0
    for t_type, score in threats.items():
        if score > max_score:
            max_score = score
            max_threat_type = t_type

    # Safety Threshold Gate (> 0.85)
    if max_score > 0.85:
        threat_detected = True
        threat_type = max_threat_type
    else:
        # No breach detected, no interception needed
        return ThreatTriggerResponse(
            threat_detected=False,
            threat_type="none",
            session_id="",
            initial_response="Safe content. No interception triggered.",
            child_age=payload.child_age,
            current_phase="none"
        )

    # 2. Threshold breached - Generate Socratic Session ID
    session_id = str(uuid.uuid4())
    
    # Initialize state machine at Acknowledge state
    session_state = agent_manager.init_session(
        session_id=session_id,
        child_age=payload.child_age,
        threat_type=threat_type
    )

    # 3. Create the initial intervention message matching Socratic State
    # Acknowledge: First contact.
    if payload.child_age <= 10:
        initial_question = "Hi there! I noticed some scary things on your screen, so I've covered it to keep you safe. Can you tell me what you were looking at, okay?"
    else:
        initial_question = f"Hey. I've temporarily intercepted the screen because I detected material that looks like {threat_type.replace('_', ' ')}. Socratic Buddy is here to chat. How did you end up on this page?"

    # Pre-populate history with the first assistant greeting to keep context intact
    import json
    initial_payload = {
        "socratic_response_to_child": initial_question,
        "child_emotion": "neutral",
        "agreed_to_boundary": False
    }
    session_state["history"].append({"role": "assistant", "content": json.dumps(initial_payload)})

    return ThreatTriggerResponse(
        threat_detected=True,
        threat_type=threat_type,
        session_id=session_id,
        initial_response=initial_question,
        child_age=payload.child_age,
        current_phase="Acknowledge"
    )

@app.post("/api/dialogue/turn", response_model=DialogueTurnResponse)
def execute_dialogue_turn(payload: DialogueTurnRequest):
    """
    Accepts the child's text, feeds it into the Socratic State Machine loop,
    returns structured Socratic Buddy outputs, and updates states.
    """
    session = agent_manager.get_session(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Socratic safety session not found or expired.")

    if session["completed"]:
        return DialogueTurnResponse(
            socratic_response_to_child="The safety boundary is active. This screen is securely closed.",
            child_emotion="resolved",
            agreed_to_boundary=True,
            current_phase="Contract",
            completed=True
        )

    try:
        updated_session, result = agent_manager.execute_turn(
            session_id=payload.session_id,
            child_response=payload.child_response
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Socratic loop failure: {str(e)}")

    return DialogueTurnResponse(
        socratic_response_to_child=result["socratic_response_to_child"],
        child_emotion=result["child_emotion"],
        agreed_to_boundary=result["agreed_to_boundary"],
        current_phase=result["state_info"]["next_phase"],
        completed=result["state_info"]["completed"]
    )

@app.get("/api/guard/state")
def get_guard_state():
    """
    Returns the complete real-time state of the Zero-Trust Guard pipeline.
    """
    return guard_system.get_state()

@app.post("/api/guard/toggle")
def toggle_guard_simulation(payload: GuardToggleRequest):
    """
    Toggles between Simulation Mode and Live Mode.
    """
    guard_system.set_simulation_mode(payload.simulation_mode)
    return {"status": "ok", "simulation_mode": payload.simulation_mode}

@app.post("/api/guard/toggle-monitor")
def toggle_guard_monitor(payload: GuardMonitorRequest):
    """
    Toggles active background scanning.
    """
    if payload.active:
        guard_system.start()
    else:
        guard_system.stop()
    return {"status": "ok", "active": payload.active}

@app.post("/api/guard/process-frame")
def process_custom_frame(payload: FrameAnalysisRequest):
    """
    Accepts a base64 encoded frame from an uploaded/playing video in the browser,
    runs process scanners, scene-change differencing, and object detections.
    """
    return guard_system.process_custom_frame(payload.image_b64, payload.filename)

@app.post("/api/guard/reset")
def reset_guard_state():
    """
    Resets the state of the Zero-Trust Guard.
    """
    guard_system.reset()
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # In production, run on uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
