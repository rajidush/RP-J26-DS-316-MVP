import os
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from socratic_agent import SocraticAgentManager
import dashboard_store
import analyst_bridge


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
    nsfw_score: float = Field(0.0, description="NSFW classifier score")
    violence_score: float = Field(0.0, description="MoViNet-A0 violence score / Violence score")
    weapons_score: float = Field(0.0, description="YOLOv8-nano weapons score")
    hate_speech_score: float = Field(0.0, description="Hate speech score")
    adult_content_score: float = Field(0.0, description="Adult content score")
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
    Analyzes content threat scores against individual safety thresholds.
    - nsfw / adult content score > 0.80
    - violence_score > 0.80
    - hate_speech_score > 0.80 (also filled from live C2 Analyst if client sends 0)
    - weapons_score > 0.75
    """
    threat_detected = False
    threat_type = "none"

    # Prefer client score; if unset, pull live C2 Analyst hate reading (safe no-op if offline)
    hate_score = float(payload.hate_speech_score or 0.0)
    if hate_score <= 0.0:
        try:
            hate_score = float(analyst_bridge.hate_speech_score_from_latest() or 0.0)
        except Exception:
            hate_score = 0.0

    nsfw_val = max(payload.nsfw_score, payload.adult_content_score)
    if nsfw_val > 0.80:
        threat_detected = True
        threat_type = "adult_content"
    elif payload.violence_score > 0.80:
        threat_detected = True
        threat_type = "violence"
    elif hate_score > 0.80:
        threat_detected = True
        threat_type = "hate_speech"
    elif payload.weapons_score > 0.75:
        threat_detected = True
        threat_type = "weapons"

    # Safety Threshold Gate
    if not threat_detected:
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
        "child_emotion": "Neutral",
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
        child_emotion="Neutral",
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

def map_emotion(raw_emotion: str) -> str:
    if not raw_emotion:
        return "Neutral"
    emo = raw_emotion.lower().strip()
    if emo in ["resolved", "compliant", "happy", "reflective", "curious", "positive"]:
        return "Positive"
    elif emo in ["scared", "defensive", "frustrated", "angry", "negative"]:
        return "Negative"
    else:
        return "Neutral"

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
        last_emotion = "Neutral"
        data = dashboard_store.get_dashboard_data()
        for s in data.get("socratic_sessions", []):
            if s.get("session_id") == payload.session_id:
                turns = s.get("turns", [])
                for turn in reversed(turns):
                    if turn.get("child_emotion"):
                        last_emotion = map_emotion(turn["child_emotion"])
                        break
                break
        return DialogueTurnResponse(
            socratic_response_to_child="The safety boundary is active. This screen is securely closed.",
            child_emotion=last_emotion,
            agreed_to_boundary=True,
            current_phase="Contract",
            completed=True
        )

    try:
        updated_session, result = agent_manager.execute_turn(
            session_id=payload.session_id,
            child_response=payload.child_response
        )
        mapped_emotion = map_emotion(result["child_emotion"])
        dashboard_store.add_socratic_turn(
            session_id=payload.session_id,
            child_response=payload.child_response,
            socratic_response_to_child=result["socratic_response_to_child"],
            child_emotion=mapped_emotion,
            agreed_to_boundary=result["agreed_to_boundary"],
            current_phase=result["state_info"]["next_phase"],
            completed=result["state_info"]["completed"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Socratic loop failure: {str(e)}")

    return DialogueTurnResponse(
        socratic_response_to_child=result["socratic_response_to_child"],
        child_emotion=mapped_emotion,
        agreed_to_boundary=result["agreed_to_boundary"],
        current_phase=result["state_info"]["next_phase"],
        completed=result["state_info"]["completed"]
    )

@app.get("/api/analyst/status")
def get_analyst_status():
    """Live C2 Analyst panel + latest reading for main app / parent dashboard."""
    try:
        merged = analyst_bridge.get_merged_analyst_runs(limit=1)
        return {
            "ok": True,
            "analyst_status": merged.get("analyst_status") or {},
            "analyst_db_available": bool(merged.get("analyst_db_available")),
            "analyst_stats": merged.get("analyst_stats") or {"total": 0, "hate": 0, "not_hate": 0},
            "latest_run": merged.get("latest_run"),
            "hate_speech_score": float(merged.get("hate_speech_score") or 0.0),
            "panel_url": analyst_bridge.ANALYST_PANEL_URL,
        }
    except Exception as e:
        return {
            "ok": False,
            "analyst_status": {"online": False, "panel_url": analyst_bridge.ANALYST_PANEL_URL},
            "analyst_db_available": False,
            "analyst_stats": {"total": 0, "hate": 0, "not_hate": 0},
            "latest_run": None,
            "hate_speech_score": 0.0,
            "panel_url": analyst_bridge.ANALYST_PANEL_URL,
            "error": str(e)[:160],
        }


@app.get("/api/parent/dashboard-data")
def get_parent_dashboard_data(child_age: Optional[int] = None):
    """Returns all logged parent dashboard metrics, logs and history, plus calculated summary card metrics."""
    try:
        data = dashboard_store.get_dashboard_data()
        bridge = analyst_bridge.get_merged_analyst_runs(limit=100)

        # C2 Analyst DB is source of truth when available; else legacy JSON history
        if bridge["analyst_db_available"]:
            analyst_runs = bridge["analyst_runs"]
        else:
            analyst_runs = data.get("analyst_runs", [])

        socratic_sessions = data.get("socratic_sessions", [])

        hate_by_age = bridge.get("hate_speech_by_age") or {}
        legacy_hate = data.get("hate_speech_detected", {})

        if child_age is not None:
            runs_filtered = [r for r in analyst_runs if r.get("child_age") == child_age]
            sessions_filtered = [s for s in socratic_sessions if s.get("child_age") == child_age]
            hate_speech_detected = hate_by_age.get(str(child_age))
            if hate_speech_detected is None:
                hate_speech_detected = legacy_hate.get(str(child_age), 0)
            screen_time_dict = data.get("screen_time_minutes", {})
            screen_time_minutes = screen_time_dict.get(str(child_age), 0)
        else:
            runs_filtered = analyst_runs
            sessions_filtered = socratic_sessions
            hate_speech_detected = sum(hate_by_age.values()) or sum(legacy_hate.values())
            screen_time_minutes = sum(data.get("screen_time_minutes", {}).values())

        content_intercepted = len(sessions_filtered)
        high_severity_alerts = len([r for r in runs_filtered if r.get("decision") == "hate"])

        return {
            "analyst_runs": analyst_runs,
            "socratic_sessions": socratic_sessions,
            "content_intercepted": content_intercepted,
            "high_severity_alerts": high_severity_alerts,
            "hate_speech_detected": hate_speech_detected,
            "screen_time_minutes": screen_time_minutes,
            "analyst_status": bridge["analyst_status"],
            "analyst_stats": bridge["analyst_stats"],
            "analyst_panel_url": analyst_bridge.ANALYST_PANEL_URL,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dashboard history: {e}")

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
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
