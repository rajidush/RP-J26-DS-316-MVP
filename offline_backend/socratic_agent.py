import json
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from pydantic import BaseModel, Field

# Define structured output schemas using Pydantic
class SocraticResponse(BaseModel):
    socratic_response_to_child: str = Field(
        ..., 
        description="The safe, age-appropriate pedagogical response directly spoken to the child."
    )
    child_emotion: str = Field(
        ..., 
        description="The inferred emotional state of the child based on their response (e.g., curious, defensive, scared, compliant, frustrated)."
    )
    agreed_to_boundary: bool = Field(
        ..., 
        description="True if the child has explicitly agreed to close the content, stop, or pivot. False otherwise."
    )

class SocraticAgentManager:
    def __init__(self, base_url: str = "http://localhost:1234/v1", api_key: str = "lm-studio"):
        """
        Initializes the offline Socratic agent using the OpenAI library pointing to LM Studio.
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        # Dictionary to store active child sessions: session_id -> session_data
        self.sessions: Dict[str, dict] = {}

    def init_session(self, session_id: str, child_age: int, threat_type: str) -> dict:
        """
        Starts a brand new state-machine session for an intercepted threat.
        """
        self.sessions[session_id] = {
            "child_age": child_age,
            "threat_type": threat_type,
            "current_phase": "Acknowledge",  # Initial state
            "history": [],                   # Chat history for sliding window
            "threat_detected": True,
            "completed": False
        }
        return self.sessions[session_id]

    def get_session(self, session_id: str) -> Optional[dict]:
        return self.sessions.get(session_id)

    def _get_system_prompt(self, child_age: int, threat_type: str, current_phase: str) -> str:
        """
        Implements the Socratic Buddy safety guide system prompt.
        """
        system_prompt = (
            f"You are an educational safety guide. A video containing {threat_type.replace('_', ' ')} was blocked. "
            "Do not lecture. Ask one open-ended question to help the user critically reflect on this content.\n\n"
            "OUTPUT COMPLIANCE:\n"
            "You MUST output raw JSON matching this schema:\n"
            "{\n"
            '  "socratic_response_to_child": "string (the single open-ended question spoken to the child)",\n'
            '  "child_emotion": "string (one-word categorization of child\'s emotional state)",\n'
            '  "agreed_to_boundary": true/false (set to true if they agree to close/pivot, default false)\n'
            "}\n"
            "Do NOT include any preamble, commentary, markdown wrapping, or code blocks. Output ONLY valid JSON."
        )
        return system_prompt

    def _slide_window_memory(self, history: List[Dict], max_turns: int = 4) -> List[Dict]:
        """
        Sliding Window Memory Manager: Keeps only the last N rounds of user-assistant turns.
        This prevents context amnesia / hallucination in small 1B-3B parameters models.
        """
        # A turn consists of 1 user message and 1 assistant message.
        # So we keep the last (max_turns * 2) messages.
        return history[-(max_turns * 2):] if len(history) > (max_turns * 2) else history

    def execute_turn(self, session_id: str, child_response: str) -> Tuple[dict, dict]:
        """
        Executes a deterministic state machine turn.
        Updates state transitions, trims memory, calls the local SLM, and handles schema output.
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found.")

        # 1. Update/Append Child Response to History
        session["history"].append({"role": "user", "content": child_response})

        # 2. Memory Trim (Sliding Window) to protect 1B model's attention
        trimmed_history = self._slide_window_memory(session["history"], max_turns=4)

        # 3. Dynamic System Prompt Generation based on Age, Threat, and active State Phase
        system_prompt = self._get_system_prompt(
            child_age=session["child_age"],
            threat_type=session["threat_type"],
            current_phase=session["current_phase"]
        )

        # Combine System Prompt with sliding window history
        messages = [{"role": "system", "content": system_prompt}] + trimmed_history

        # 4. Call Local SLM with low temperature for deterministic behavior
        try:
            response = self.client.chat.completions.create(
                model="local-model",  # In LM Studio, 'local-model' maps to the loaded SLM
                messages=messages,
                temperature=0.2,      # Fixed low temperature to enforce state constraints and prevent drift
                response_format={"type": "json_object"}  # Forces JSON mode if SLM supports it
            )
            raw_content = response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback for offline resilience if LM Studio connection fails
            raw_content = self._get_fallback_response(session, child_response, str(e))

        # 5. Safe Parse JSON Structure
        try:
            # Strip any markdown code blocks if the SLM accidentally emitted them
            clean_json = raw_content
            if clean_json.startswith("```json"):
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif clean_json.startswith("```"):
                clean_json = clean_json.split("```")[1].split("```")[0].strip()
            
            parsed_output = json.loads(clean_json)
            socratic_text = parsed_output.get("socratic_response_to_child", "")
            child_emotion = parsed_output.get("child_emotion", "unknown")
            agreed = parsed_output.get("agreed_to_boundary", False)
        except Exception:
            # Regex or basic string fallback if JSON parsing failed completely
            socratic_text = raw_content
            child_emotion = "unclear"
            agreed = False
            if "agree" in child_response.lower() or "ok" in child_response.lower() or "yes" in child_response.lower():
                agreed = True

        # 6. Update Socratic State Machine Phase Progression (Deterministic Transition)
        previous_phase = session["current_phase"]
        if session["current_phase"] == "Acknowledge":
            # Transition to 'Reason' after child acknowledges
            session["current_phase"] = "Reason"
        elif session["current_phase"] == "Reason":
            # Transition to 'Contract' once the child has engaged in reasoning
            session["current_phase"] = "Contract"
        elif session["current_phase"] == "Contract" and agreed:
            # Terminate and mark completed if contract is sealed
            session["completed"] = True

        # Append assistant's response to the real session history
        assistant_json_str = json.dumps({
            "socratic_response_to_child": socratic_text,
            "child_emotion": child_emotion,
            "agreed_to_boundary": agreed
        })
        session["history"].append({"role": "assistant", "content": assistant_json_str})

        result_payload = {
            "socratic_response_to_child": socratic_text,
            "child_emotion": child_emotion,
            "agreed_to_boundary": agreed,
            "state_info": {
                "previous_phase": previous_phase,
                "next_phase": session["current_phase"],
                "completed": session["completed"],
                "child_age": session["child_age"],
                "threat_type": session["threat_type"]
            }
        }

        return session, result_payload

    def _get_fallback_response(self, session: dict, child_response: str, error_msg: str) -> str:
        """
        Local fallback model in case of API failure (offline resilience).
        """
        age = session["child_age"]
        phase = session["current_phase"]
        threat = session["threat_type"]
        
        # Simple heuristics for offline safety fallback
        if age <= 10:
            if phase == "Acknowledge":
                response = "I noticed something a bit scary here on your screen. I want to help you stay safe. Let's close this together, okay?"
            elif phase == "Reason":
                response = "Some pictures or words can make us feel confused or scared inside. Why do you think it's a good idea to play with something else right now?"
            else:
                response = "Thank you for being so helpful! Let's click home and find a fun safe game to play now, deal?"
        else:
            if phase == "Acknowledge":
                response = f"I've stepped in because a content threat was detected ({threat}). I want to respect your space, but this content carries risks. How do you feel about this interruption?"
            elif phase == "Reason":
                response = f"We encounter a lot of intense material online. In your view, why is accessing this specific type of ({threat}) media problematic or unsafe?"
            else:
                response = "I appreciate you talking this through with me. Let's agree to close this app and pivot to a safer space. Can we lock that in?"

        return json.dumps({
            "socratic_response_to_child": response,
            "child_emotion": "reflective",
            "agreed_to_boundary": "agree" in child_response.lower() or "ok" in child_response.lower() or "yes" in child_response.lower()
        })
