import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from socratic_agent import SocraticAgentManager
import dashboard_store


app = FastAPI(
    title="Socratic Digital Child Safety Engine",
    description="Offline Cognitive State-Machine Prototype safeguarding minors through pedagogical dialog."
)

# Enable CORS for frontend web and Electron integration
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r".*",
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our Socratic agent core
# Configured for default LM Studio local server on http://localhost:1234/v1
agent_manager = SocraticAgentManager(
    base_url=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
    api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
    model_name=os.getenv("LM_STUDIO_MODEL", "google/gemma-3-1b")
)

# --- Schemas ---

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
        "engine_target": f"LM Studio ({agent_manager.base_url})",
        "model": agent_manager.model_name,
        "note": "C2 Analyst lives in repo-root /analyst (python -m analyst). This service is C3/C4 demo API.",
    }


@app.post("/api/perception/trigger", response_model=ThreatTriggerResponse)
def trigger_threat(payload: ThreatTriggerRequest):
    """
    Simulates the 'Perception Layer' interceptor.
    Analyzes content threat scores. If a threshold is breached (score > 0.85),
    it flags threat_detected = True, extracts threat_type, and triggers Socratic Buddy.
    """
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

    if max_score > 0.85:
        threat_detected = True
        threat_type = max_threat_type
    else:
        return ThreatTriggerResponse(
            threat_detected=False,
            threat_type="none",
            session_id="",
            initial_response="Safe content. No interception triggered.",
            child_age=payload.child_age,
            current_phase="none"
        )

    session_id = str(uuid.uuid4())

    session_state = agent_manager.init_session(
        session_id=session_id,
        child_age=payload.child_age,
        threat_type=threat_type
    )

    dashboard_store.add_socratic_session_start(
        session_id=session_id,
        child_age=payload.child_age,
        threat_type=threat_type
    )

    if payload.child_age <= 10:
        initial_question = "Hi there! I noticed some scary things on your screen, so I've covered it to keep you safe. Can you tell me what you were looking at, okay?"
    else:
        initial_question = f"Hey. I've temporarily intercepted the screen because I detected material that looks like {threat_type.replace('_', ' ')}. Socratic Buddy is here to chat. How did you end up on this page?"

    import json
    initial_payload = {
        "socratic_response_to_child": initial_question,
        "child_emotion": "neutral",
        "agreed_to_boundary": False
    }
    session_state["history"].append({
        "role": "user",
        "content": f"[Safety Interceptor Alert]: Potential {threat_type.replace('_', ' ')} threat detected on screen."
    })
    session_state["history"].append({"role": "assistant", "content": json.dumps(initial_payload)})

    dashboard_store.add_socratic_turn(
        session_id=session_id,
        child_response="",
        socratic_response_to_child=initial_question,
        child_emotion="neutral",
        agreed_to_boundary=False,
        current_phase="Acknowledge",
        completed=False
    )

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
        dashboard_store.add_socratic_turn(
            session_id=payload.session_id,
            child_response=payload.child_response,
            socratic_response_to_child=result["socratic_response_to_child"],
            child_emotion=result["child_emotion"],
            agreed_to_boundary=result["agreed_to_boundary"],
            current_phase=result["state_info"]["next_phase"],
            completed=result["state_info"]["completed"]
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

@app.get("/api/parent/dashboard-data")
def get_parent_dashboard_data():
    """Returns all logged parent dashboard metrics, logs and history."""
    try:
        return dashboard_store.get_dashboard_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dashboard history: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
