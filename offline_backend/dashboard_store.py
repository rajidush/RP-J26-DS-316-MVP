import json
import os
import threading
from datetime import datetime
from typing import Dict, List, Optional

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_history.json")
lock = threading.Lock()

def _load_history() -> dict:
    if not os.path.exists(HISTORY_FILE):
        return {"analyst_runs": [], "socratic_sessions": []}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"analyst_runs": [], "socratic_sessions": []}

def _save_history(data: dict) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving dashboard history: {e}")

def add_analyst_run(
    child_age: int,
    decision: str,
    risk_score: float,
    category: str,
    ocr_text: str,
    transcript: str,
    overlay_text: str,
    source: Dict[str, bool],
    session_hint: str
) -> None:
    with lock:
        data = _load_history()
        run_entry = {
            "id": f"analyst-{int(datetime.now().timestamp() * 1000)}",
            "timestamp": datetime.now().isoformat(),
            "time_str": datetime.now().strftime("%I:%M %p"),
            "child_age": child_age,
            "decision": decision,
            "risk_score": risk_score,
            "category": category,
            "ocr_text": ocr_text,
            "transcript": transcript,
            "overlay_text": overlay_text,
            "source": source,
            "session_hint": session_hint
        }
        data["analyst_runs"].append(run_entry)
        # Limit history to 100 runs
        if len(data["analyst_runs"]) > 100:
            data["analyst_runs"] = data["analyst_runs"][-100:]
        _save_history(data)

def add_socratic_session_start(session_id: str, child_age: int, threat_type: str) -> None:
    with lock:
        data = _load_history()
        session_entry = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "time_str": datetime.now().strftime("%I:%M %p"),
            "child_age": child_age,
            "threat_type": threat_type,
            "turns": [],
            "completed": False,
            "agreed_to_boundary": False
        }
        # Avoid duplicate sessions
        existing_idx = next((i for i, s in enumerate(data["socratic_sessions"]) if s["session_id"] == session_id), None)
        if existing_idx is not None:
            data["socratic_sessions"][existing_idx] = session_entry
        else:
            data["socratic_sessions"].append(session_entry)
            
        # Limit history to 50 sessions
        if len(data["socratic_sessions"]) > 50:
            data["socratic_sessions"] = data["socratic_sessions"][-50:]
        _save_history(data)

def add_socratic_turn(
    session_id: str,
    child_response: str,
    socratic_response_to_child: str,
    child_emotion: str,
    agreed_to_boundary: bool,
    current_phase: str,
    completed: bool
) -> None:
    with lock:
        data = _load_history()
        for session in data["socratic_sessions"]:
            if session["session_id"] == session_id:
                turn_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "time_str": datetime.now().strftime("%I:%M %p"),
                    "child_response": child_response,
                    "socratic_response_to_child": socratic_response_to_child,
                    "child_emotion": child_emotion,
                    "agreed_to_boundary": agreed_to_boundary,
                    "current_phase": current_phase
                }
                session["turns"].append(turn_entry)
                session["completed"] = completed
                session["agreed_to_boundary"] = agreed_to_boundary
                break
        _save_history(data)

def get_dashboard_data() -> dict:
    with lock:
        history = _load_history()
        # Default keys for future compatibility
        if "hate_speech_detected" not in history:
            history["hate_speech_detected"] = {}
        if "screen_time_minutes" not in history:
            history["screen_time_minutes"] = {}
        return history
