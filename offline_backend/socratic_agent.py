import os
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
    def __init__(
        self, 
        base_url: str = "http://localhost:1234/v1", 
        api_key: str = "lm-studio",
        model_name: Optional[str] = None
    ):
        """
        Initializes the offline Socratic agent using the OpenAI library pointing to LM Studio.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name or os.getenv("LM_STUDIO_MODEL", "google/gemma-3-1b")
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
        Builds a fully phase-aware, age-differentiated system prompt.

        The 3-Step Socratic State Machine:
          - Acknowledge: Empathise and gently open dialogue without revealing raw harmful content.
          - Reason:      Probe critical thinking — why is this content problematic?
          - Contract:    Negotiate a concrete, child-led safety commitment.

        Age Routing:
          - Age <= 10 (Protective):             Simple vocabulary, reassuring tone, one easy safe question.
          - Age >= 11 (Autonomy & Negotiation): Peer-level framing, respectful critical inquiry.
        """
        threat_label = threat_type.replace("_", " ")
        protective_mode = child_age <= 10

        # --- Phase-specific instruction blocks ---
        if current_phase == "Acknowledge":
            if protective_mode:
                phase_instruction = (
                    f"A safety system just covered the screen because it detected {threat_label}. "
                    "Your job in this turn is ONLY to: (1) Gently reassure the child that they are safe, "
                    "(2) Acknowledge that something was on the screen without describing or repeating it, "
                    "(3) Ask ONE simple, warm, open-ended question to understand how they feel right now. "
                    "Do NOT ask why they were watching it. Do NOT lecture. Keep vocabulary elementary and kind."
                )
            else:
                phase_instruction = (
                    f"A safety system intercepted the screen because it detected {threat_label}. "
                    "Your job in this turn is ONLY to: (1) Briefly acknowledge the interception without judgment, "
                    "(2) Signal that you respect their space while explaining that this content poses real risks, "
                    "(3) Ask ONE open-ended question to let them share their perspective on what happened. "
                    "Do NOT repeat or describe the harmful content. Do NOT moralize. Speak as a thoughtful peer."
                )

        elif current_phase == "Reason":
            if protective_mode:
                phase_instruction = (
                    "The child has acknowledged what happened. Your job in this turn is ONLY to: "
                    "(1) Validate any emotion they expressed, "
                    "(2) Ask ONE simple question that helps them think about WHY some content can feel scary or be harmful. "
                    "Use simple analogies (e.g., 'like how some foods are not good for us'). "
                    "Do NOT answer the question for them. Do NOT lecture. One question only."
                )
            else:
                phase_instruction = (
                    "The child has acknowledged the interception. Your job in this turn is ONLY to: "
                    "(1) Acknowledge their response with genuine respect, "
                    "(2) Ask ONE probing, critical-thinking question that encourages them to reason about "
                    f"why {threat_label} content can be harmful — to themselves or to others. "
                    "Invite reflection, not recitation of rules. Challenge them intellectually as a peer. "
                    "One question only. No lecturing."
                )

        elif current_phase == "Contract":
            if protective_mode:
                phase_instruction = (
                    "The child has engaged in reasoning. Your job in this turn is ONLY to: "
                    "(1) Warmly praise their thoughtfulness, "
                    "(2) Propose a clear, friendly safety agreement — asking them to close or move away from the content now. "
                    "Frame it as a team decision ('Can we close this together?'). "
                    "Set agreed_to_boundary=true ONLY if the child's last message contains explicit verbal agreement "
                    "(e.g., 'ok', 'yes', 'sure', 'deal', 'I agree', 'let's do it'). "
                    "If they have not agreed yet, set agreed_to_boundary=false and gently re-invite agreement."
                )
            else:
                phase_instruction = (
                    "The child has reflected on the reasoning. Your job in this turn is ONLY to: "
                    "(1) Acknowledge their reasoning maturely, "
                    "(2) Propose a concrete, negotiated safety boundary — invite them to commit to closing or "
                    "stepping away from this content category. "
                    "Frame it as a mutual agreement between equals, not a command. "
                    "Set agreed_to_boundary=true ONLY if the child's last message contains an explicit, "
                    "clear verbal commitment (e.g., 'agree', 'yes', 'I'll close it', 'deal', 'fine', 'ok'). "
                    "If they have not clearly agreed, set agreed_to_boundary=false and invite once more."
                )
        else:
            # Defensive catch-all — should never reach here in a correctly-running session
            phase_instruction = (
                f"You are a safety guide. {threat_label} content was detected. "
                "Speak calmly to the child and guide them toward closing this content safely."
            )

        # --- JSON output schema instruction (always appended) ---
        schema_instruction = (
            "\n\nOUTPUT COMPLIANCE — CRITICAL:\n"
            "You MUST output ONLY raw JSON. No preamble, no commentary, no markdown, no code blocks.\n"
            "Match this exact schema:\n"
            "{\n"
            '  "socratic_response_to_child": "<your single spoken response to the child>",\n'
            '  "child_emotion": "<one-word inferred emotion: curious|defensive|scared|compliant|frustrated|neutral|reflective>",\n'
            '  "agreed_to_boundary": <true only if child explicitly agreed, false otherwise>\n'
            "}\n"
            "Output ONLY the JSON object. Nothing before or after it."
        )

        return phase_instruction + schema_instruction

    def _format_messages_for_llm(self, system_prompt: str, history: List[Dict], threat_type: str, max_turns: int = 4) -> List[Dict]:
        """
        Formats and sanitizes conversation history for strict Jinja template engines (e.g. Gemma 3, Llama 3).
        Guarantees:
        1. Conversation starts with a 'user' message after system prompt.
        2. Strict alternation of roles: user -> assistant -> user -> assistant ...
        3. Avoids consecutive duplicate roles by merging contents.
        """
        # 1. Slide window memory (last max_turns rounds = max_turns * 2 messages)
        raw_slice = history[-(max_turns * 2):] if len(history) > (max_turns * 2) else list(history)

        # 2. Ensure the slice starts with a 'user' message
        while raw_slice and raw_slice[0].get("role") != "user":
            raw_slice.pop(0)

        # If empty, seed with the initial intercept context
        if not raw_slice:
            raw_slice = [{"role": "user", "content": f"[Safety Interceptor Alert]: Potential {threat_type} threat detected on screen."}]

        # 3. Deduplicate consecutive identical roles
        clean_history = []
        for msg in raw_slice:
            role = msg.get("role", "user")
            content = str(msg.get("content", "")).strip()
            if not content:
                continue

            if clean_history and clean_history[-1]["role"] == role:
                clean_history[-1]["content"] += f"\n{content}"
            else:
                clean_history.append({"role": role, "content": content})

        # Ensure first message is user
        if not clean_history or clean_history[0]["role"] != "user":
            clean_history.insert(0, {"role": "user", "content": f"[Safety Interceptor Alert]: Potential {threat_type} threat detected on screen."})

        # Prepend system prompt
        return [{"role": "system", "content": system_prompt}] + clean_history

    def execute_turn(self, session_id: str, child_response: str) -> Tuple[dict, dict]:
        """
        Main inference loop — executes one deterministic state-machine turn.

        State Machine Contract:
          Acknowledge (turn 1) -> Reason (turn 2) -> Contract (turn 3+, loops until agreed_to_boundary=True)

        Steps:
          1.  Append child's response to rolling history.
          2.  Generate the phase-aware, age-differentiated system prompt.
          3.  Format messages with strict Jinja role alternation (user/assistant).
          4.  Call local SLM with 3-tier fallback (json_object -> plain -> merged system).
          5.  Parse and validate structured JSON output.
          6.  Apply deterministic phase transitions:
                Acknowledge -> Reason  (always, after one turn)
                Reason      -> Contract (always, after one turn)
                Contract    -> completed=True  (only when agreed_to_boundary=True)
          7.  Append assistant response to history and return result payload.
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found.")

        # --- Step 1: Append child's message to history ---
        session["history"].append({"role": "user", "content": child_response})

        # --- Step 2: Phase-aware, age-differentiated system prompt ---
        system_prompt = self._get_system_prompt(
            child_age=session["child_age"],
            threat_type=session["threat_type"],
            current_phase=session["current_phase"]
        )

        # --- Step 3: Format messages for Jinja-compatible LLM ---
        messages = self._format_messages_for_llm(
            system_prompt=system_prompt,
            history=session["history"],
            threat_type=session["threat_type"],
            max_turns=4
        )

        # --- Step 4: Call local SLM with 3-tier fallback ---
        raw_content = ""
        try:
            try:
                # Tier 1: Primary — json_object constraint + low temperature
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
            except Exception:
                try:
                    # Tier 2: Retry without response_format (for models lacking grammar sampling)
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.2
                    )
                except Exception:
                    # Tier 3: Merge system prompt into first user turn (for models rejecting system role)
                    merged_messages = []
                    sys_text = system_prompt
                    for msg in messages:
                        if msg["role"] == "system":
                            continue
                        if msg["role"] == "user" and sys_text:
                            merged_messages.append({
                                "role": "user",
                                "content": f"{sys_text}\n\n{msg['content']}"
                            })
                            sys_text = ""
                        else:
                            merged_messages.append(msg)

                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=merged_messages,
                        temperature=0.2
                    )

            raw_content = response.choices[0].message.content.strip()

        except Exception as e:
            # Offline fallback: deterministic rule-based scaffolding (no LLM dependency)
            raw_content = self._get_fallback_response(session, child_response, str(e))

        # --- Step 5: Parse and validate JSON output ---
        socratic_text = ""
        child_emotion = "reflective"
        agreed = False

        try:
            clean_json = raw_content.strip()

            # Strip markdown code fences if the LLM wrapped output
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0].strip()

            # Extract the JSON object if preamble text leaked through
            if "{" in clean_json and "}" in clean_json:
                first_brace = clean_json.find("{")
                last_brace = clean_json.rfind("}")
                clean_json = clean_json[first_brace:last_brace + 1]

            parsed_output = json.loads(clean_json)
            socratic_text = parsed_output.get("socratic_response_to_child", "").strip()
            child_emotion = parsed_output.get("child_emotion", "reflective").strip()

            # agreed_to_boundary is only meaningful and trusted in the Contract phase.
            # In earlier phases, we force it to False regardless of LLM output to prevent
            # premature session termination.
            raw_agreed = bool(parsed_output.get("agreed_to_boundary", False))
            agreed = raw_agreed if session["current_phase"] == "Contract" else False

        except Exception:
            # JSON parse failure: use raw content as spoken text and apply keyword heuristic
            # for agreement — but only in the Contract phase.
            socratic_text = raw_content.strip()
            child_emotion = "reflective"
            if session["current_phase"] == "Contract":
                agreed = any(
                    w in child_response.lower()
                    for w in ["agree", "ok", "yes", "deal", "sure", "close", "i'll", "let's", "fine"]
                )
            else:
                agreed = False

        # Ensure we always have a spoken response
        if not socratic_text:
            socratic_text = raw_content or "Let's talk through what happened on your screen."

        # --- Step 6: Deterministic phase transition (state machine) ---
        previous_phase = session["current_phase"]

        if session["current_phase"] == "Acknowledge":
            # One turn in Acknowledge -> always advance to Reason
            session["current_phase"] = "Reason"

        elif session["current_phase"] == "Reason":
            # One turn in Reason -> always advance to Contract
            session["current_phase"] = "Contract"

        elif session["current_phase"] == "Contract":
            # Stay in Contract until the child explicitly agrees; mark complete on agreement
            if agreed:
                session["completed"] = True
            # current_phase stays "Contract" whether agreed or not —
            # the loop re-runs with the same Contract prompt until boundary is sealed.

        # --- Step 7: Append assistant response to rolling history ---
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
        Deterministic offline fallback — runs when the local SLM server is unreachable.

        Mirrors the 3-phase Socratic scaffolding with age-differentiated language so the
        pedagogical flow remains intact even without the LLM. The fallback checks for
        explicit agreement keywords only in the Contract phase.
        """
        age = session["child_age"]
        phase = session["current_phase"]
        threat = session["threat_type"].replace("_", " ")
        protective_mode = age <= 10

        # Explicit agreement detection — gated to Contract phase only
        child_agreed = False
        if phase == "Contract":
            child_agreed = any(
                w in child_response.lower()
                for w in ["agree", "ok", "yes", "deal", "sure", "close", "i'll", "let's", "fine"]
            )

        # Phase + age matrix of fallback responses
        if protective_mode:
            if phase == "Acknowledge":
                response = (
                    "Hey, I just covered the screen to keep you safe — something a bit scary showed up. "
                    "You're totally safe right now! How are you feeling?"
                )
            elif phase == "Reason":
                response = (
                    "Some things we see on screens can feel confusing or scary inside, a bit like eating "
                    "something that doesn't agree with your tummy. Why do you think it might be a good idea "
                    "to stay away from things like that?"
                )
            else:  # Contract
                if child_agreed:
                    response = "You're so thoughtful! Let's close this together and find something fun and safe instead, deal?"
                else:
                    response = "You're doing really well talking about this! Can we make a little promise together to close this and pick something safer? What do you think?"
        else:
            if phase == "Acknowledge":
                response = (
                    f"I've stepped in because the system flagged {threat} content. "
                    "I'm not here to judge you — I just want to check in. "
                    "How are you feeling about this interruption?"
                )
            elif phase == "Reason":
                response = (
                    f"Thanks for sharing that. A lot of people come across {threat} material online. "
                    "In your own view, why do you think this type of content might be harmful — "
                    "to you or to the people involved in it?"
                )
            else:  # Contract
                if child_agreed:
                    response = (
                        "I really respect how you've thought this through. "
                        "Let's lock in our agreement — you'll close this and step away from this content. Deal?"
                    )
                else:
                    response = (
                        "I appreciate you engaging with me on this. "
                        "Can we agree, as equals, to close this content and move on? "
                        "Your call — what do you think?"
                    )

        return json.dumps({
            "socratic_response_to_child": response,
            "child_emotion": "reflective",
            "agreed_to_boundary": child_agreed
        })
