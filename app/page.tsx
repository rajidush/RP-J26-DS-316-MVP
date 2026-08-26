"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { BACKEND_URL } from "./lib/backend";
import { 
  Shield, 
  Terminal, 
  Sliders, 
  BookOpen, 
  Code, 
  Cpu, 
  AlertTriangle, 
  Smartphone, 
  Send, 
  Layers, 
  RefreshCw, 
  CheckCircle, 
  Copy, 
  ChevronRight, 
  User, 
  FileText, 
  Heart, 
  Gamepad2, 
  Chrome, 
  Lock,
  Compass,
  ArrowRight,
  Activity,
  Clock
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

// Code contents for the Offline Code Hub
const codeSocraticAgent = `import json
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
        description="The inferred emotional state of the child based on their response."
    )
    agreed_to_boundary: bool = Field(
        ..., 
        description="True if the child has explicitly agreed to close the content, stop, or pivot. False otherwise."
    )

class SocraticAgentManager:
    def __init__(self, base_url: str = "http://localhost:1234/v1", api_key: str = "lm-studio", model_name: str = "google/gemma-3-1b"):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.sessions: Dict[str, dict] = {}

    def init_session(self, session_id: str, child_age: int, threat_type: str) -> dict:
        self.sessions[session_id] = {
            "child_age": child_age,
            "threat_type": threat_type,
            "current_phase": "Acknowledge",  # State Machine starts here
            "history": [],                   # Conversations stored for sliding window
            "threat_detected": True,
            "completed": False
        }
        return self.sessions[session_id]

    def get_session(self, session_id: str) -> Optional[dict]:
        return self.sessions.get(session_id)

    def _get_system_prompt(self, child_age: int, threat_type: str, current_phase: str) -> str:
        # Dynamic Age-Based Routing
        if child_age <= 10:
            age_routing = (
                "ROLE & TONE:\\n"
                "You are 'Socratic Buddy', a warm, protective, and gentle digital guardian for a young child (under 10).\\n"
                "Use simple words, short sentences, and a comforting, safe voice.\\n"
                "Never be scolding. Keep your interaction firmly supportive.\\n\\n"
                "CRITICAL INSTRUCTION:\\n"
                "You MUST keep your vocabulary elementary and write very short responses. "
                "End your response with EXACTLY ONE simple, short, supportive question like: 'This has scary things in it. Let's close it, okay?'"
            )
        else:
            age_routing = (
                "ROLE & TONE:\\n"
                "You are 'Socratic Buddy', a respectful, critical-thinking dialogue partner for a pre-teen/teenager (11+).\\n"
                "Treat them with high respect as an autonomous young adult. Avoid talking down or lecturing.\\n"
                "Ask high-level, open-ended critical thinking questions about the content's risks.\\n\\n"
                "CRITICAL INSTRUCTION:\\n"
                "Guide them to analyze why this content could be harmful. Encourage self-reflection and negotiation rather than pure compliance."
            )

        # Pedagogical state constraints
        if current_phase == "Acknowledge":
            phase_instruction = (
                "CURRENT PHASE: ACKNOWLEDGE\\n"
                f"Goal: Gently alert the user that a screen intercept occurred due to '{threat_type}'. "
                "Acknowledge and validate whatever they might be feeling (e.g. curiosity, surprise) and ensure they feel safe. "
                "Ask a single question to check on how they are feeling right now."
            )
        elif current_phase == "Reason":
            phase_instruction = (
                "CURRENT PHASE: REASON\\n"
                "Goal: Guide the user into critical inquiry about the risks. "
                f"Help them think about why '{threat_type}' might not be safe. "
                "Ask guidance questions to let them deduce the boundary themselves, reinforcing their cognitive autonomy."
            )
        else:  # Contract
            phase_instruction = (
                "CURRENT PHASE: CONTRACT & PIVOT\\n"
                "Goal: Solidify a collaborative safety agreement/boundary. "
                "Mutually agree to close the window or pivot to a safer, cooler alternative activity. "
                "Confirm their explicit agreement to close/pivot."
            )

        return (
            "You are the secure, offline core of a pedagogical Child Safety System.\\n"
            "You must strictly follow the age-appropriate vocabulary and phase directives specified below.\\n\\n"
            f"{age_routing}\\n\\n"
            f"{phase_instruction}\\n\\n"
            "OUTPUT COMPLIANCE:\\n"
            "You MUST output raw JSON matching this schema:\\n"
            "{\\n"
            '  "socratic_response_to_child": "string (the voice of Socratic Buddy spoken to the child)",\\n'
            '  "child_emotion": "string (one-word categorization)",\\n'
            '  "agreed_to_boundary": true/false\\n'
            "}\\n"
            "Do NOT include code blocks or commentary. Output ONLY valid JSON."
        )

    def _slide_window_memory(self, history: List[Dict], max_turns: int = 4) -> List[Dict]:
        # Trims context memory to prevent hallucination in 1B parameters model
        return history[-(max_turns * 2):] if len(history) > (max_turns * 2) else history

    def execute_turn(self, session_id: str, child_response: str) -> Tuple[dict, dict]:
        session = self.get_session(session_id)
        if not session:
            raise ValueError("Session not found")

        session["history"].append({"role": "user", "content": child_response})
        trimmed_history = self._slide_window_memory(session["history"], max_turns=4)

        system_prompt = self._get_system_prompt(
            child_age=session["child_age"],
            threat_type=session["threat_type"],
            current_phase=session["current_phase"]
        )

        messages = [{"role": "system", "content": system_prompt}] + trimmed_history

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            raw_content = response.choices[0].message.content.strip()
        except Exception as e:
            raw_content = self._get_fallback_response(session, child_response, str(e))

        # Safe Parse JSON
        try:
            clean_json = raw_content
            if clean_json.startswith("\`\`\`json"):
                clean_json = clean_json.split("\`\`\`json")[1].split("\`\`\`")[0].strip()
            parsed_output = json.loads(clean_json)
            socratic_text = parsed_output.get("socratic_response_to_child", "")
            child_emotion = parsed_output.get("child_emotion", "unknown")
            agreed = parsed_output.get("agreed_to_boundary", False)
        except Exception:
            socratic_text = raw_content
            child_emotion = "unclear"
            agreed = False

        # State Transitions
        previous_phase = session["current_phase"]
        if session["current_phase"] == "Acknowledge":
            session["current_phase"] = "Reason"
        elif session["current_phase"] == "Reason":
            session["current_phase"] = "Contract"
        elif session["current_phase"] == "Contract" and agreed:
            session["completed"] = True

        session["history"].append({"role": "assistant", "content": json.dumps({
            "socratic_response_to_child": socratic_text,
            "child_emotion": child_emotion,
            "agreed_to_boundary": agreed
        })})

        return session, {
            "socratic_response_to_child": socratic_text,
            "child_emotion": child_emotion,
            "agreed_to_boundary": agreed,
            "state_info": {
                "previous_phase": previous_phase,
                "next_phase": session["current_phase"],
                "completed": session["completed"]
            }
        }`;

const codeMainPy = `from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid
from socratic_agent import SocraticAgentManager

app = FastAPI(title="Socratic Digital Child Safety Engine")

# CORS middleware for React / Electron dialogue UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_manager = SocraticAgentManager(base_url="http://localhost:1234/v1", api_key="lm-studio")

class ThreatTriggerRequest(BaseModel):
    violence_score: float
    hate_speech_score: float
    adult_content_score: float
    child_age: int

class ThreatTriggerResponse(BaseModel):
    threat_detected: bool
    threat_type: str
    session_id: str
    initial_response: str
    child_age: int
    current_phase: str

class DialogueTurnRequest(BaseModel):
    session_id: str
    child_response: str

class DialogueTurnResponse(BaseModel):
    socratic_response_to_child: str
    child_emotion: str
    agreed_to_boundary: bool
    current_phase: str
    completed: bool

@app.post("/api/perception/trigger", response_model=ThreatTriggerResponse)
def trigger_threat(payload: ThreatTriggerRequest):
    scores = {
        "violence": payload.violence_score,
        "hate_speech": payload.hate_speech_score,
        "adult_content": payload.adult_content_score
    }
    max_threat = max(scores, key=scores.get)
    max_score = scores[max_threat]

    # Threshold Gate (> 0.85)
    if max_score > 0.85:
        threat_detected = True
        threat_type = max_threat
    else:
        return ThreatTriggerResponse(
            threat_detected=False, threat_type="none",
            session_id="", initial_response="",
            child_age=payload.child_age, current_phase="none"
        )

    session_id = str(uuid.uuid4())
    session_state = agent_manager.init_session(
        session_id=session_id, child_age=payload.child_age, threat_type=threat_type
    )

    # Initial Acknowledge Dialog based on Age
    if payload.child_age <= 10:
        initial_question = "Hi there! I noticed some scary things on your screen, so I've covered it to keep you safe. Can you tell me what you were looking at, okay?"
    else:
        initial_question = f"Hey. I've temporarily intercepted the screen because I detected material that looks like {threat_type}. Socratic Buddy is here. How did you end up here?"

    return ThreatTriggerResponse(
        threat_detected=True, threat_type=threat_type,
        session_id=session_id, initial_response=initial_question,
        child_age=payload.child_age, current_phase="Acknowledge"
    )

@app.post("/api/dialogue/turn", response_model=DialogueTurnResponse)
def execute_dialogue_turn(payload: DialogueTurnRequest):
    session = agent_manager.get_session(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    updated_session, result = agent_manager.execute_turn(
        session_id=payload.session_id, child_response=payload.child_response
    )

    return DialogueTurnResponse(
        socratic_response_to_child=result["socratic_response_to_child"],
        child_emotion=result["child_emotion"],
        agreed_to_boundary=result["agreed_to_boundary"],
        current_phase=result["state_info"]["next_phase"],
        completed=result["state_info"]["completed"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)`;

const codeRequirements = `fastapi>=0.100.0
uvicorn>=0.22.0
openai>=1.0.0
pydantic>=2.0.0`;

const codeReadme = `# Offline Socratic Child Safety Engine

This system executes a Socratic dialogue to safely scaffold kids away from harmful content without scolding them, keeping all data fully local and private.

## Quickstart

1. Launch LM Studio, download and load **google/gemma-3-1b** (or Llama-3.2-1B-Instruct)
2. Enable "Local Server" inside LM Studio on port 1234
3. Create venv, activate, and install packages:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
4. Run FastAPI server:
   \`\`\`bash
   python main.py
   \`\`\`
5. Enable 'Local Python Backend' mode in the Prototype Sandbox to link them directly!`;

interface ChatMessage {
  role: "user" | "assistant";
  content: string; // can be serialized JSON if parsed from state machine
}

export default function SocraticPrototype() {
  // Navigation Tabs
  const [activeTab, setActiveTab] = useState<"sandbox" | "codeHub" | "videoGuard">("sandbox");

  // Zero-Trust Guard States
  const [guardState, setGuardState] = useState<any>(null);
  const [guardSimMode, setGuardSimMode] = useState<boolean>(true);
  const [guardActive, setGuardActive] = useState<boolean>(true);

  // Custom Video Upload & Analysis States
  const [videoUrl, setVideoUrl] = useState<string>("");
  const [videoFilename, setVideoFilename] = useState<string>("");
  const [analyzingVideo, setAnalyzingVideo] = useState<boolean>(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Parent Control Panel Inputs
  const [childAge, setChildAge] = useState<number>(10);
  const [violenceScore, setViolenceScore] = useState<number>(0.1);
  const [weaponsScore, setWeaponsScore] = useState<number>(0.1);
  const [adultScore, setAdultScore] = useState<number>(0.1);

  // Simulated active desktop behind the overlay
  const [childActiveApp, setChildActiveApp] = useState<"game" | "browser" | "video">("video");
  const [lastInterceptionStatus, setLastInterceptionStatus] = useState<string>("Ready & Guarding System");

  // State Machine Dialog States
  const [interceptActive, setInterceptActive] = useState<boolean>(false);
  const [activeThreatType, setActiveThreatType] = useState<string>("none");
  const [sessionId, setSessionId] = useState<string>("");
  const [currentPhase, setCurrentPhase] = useState<string>("Acknowledge"); // Acknowledge -> Reason -> Contract
  const [childEmotion, setChildEmotion] = useState<string>("neutral");
  const [agreedToBoundary, setAgreedToBoundary] = useState<boolean>(false);
  const [isCompleted, setIsCompleted] = useState<boolean>(false);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [userInput, setUserInput] = useState<string>("");
  const [loadingTurn, setLoadingTurn] = useState<boolean>(false);
  const [apiError, setApiError] = useState<string | null>(null);

  // Auto Scroll Chat Ref
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, loadingTurn]);

  // Background Polling of Zero-Trust Guard state
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    const pollGuardState = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/guard/state`);
        if (res.ok) {
          const data = await res.json();
          setGuardState(data);
          
          const nsfwBreached = data.nsfw_score > 0.80;
          const violenceBreached = data.violence_score > 0.80;
          const weaponsBreached = data.weapons_score > 0.75;
          const anyBreached = nsfwBreached || violenceBreached || weaponsBreached;

          // Auto block if simulation detects a threat in the background
          if (anyBreached && guardSimMode && !interceptActive && !isCompleted) {
            let detectedType = "none";
            if (nsfwBreached) detectedType = "adult_content";
            else if (violenceBreached) detectedType = "violence";
            else if (weaponsBreached) detectedType = "weapons";

            setLastInterceptionStatus(`ALERT: Zero-Trust Guard intercepted threat: ${detectedType}!`);
            
            if (nsfwBreached) setAdultScore(data.nsfw_score);
            if (violenceBreached) setViolenceScore(data.violence_score);
            if (weaponsBreached) setWeaponsScore(data.weapons_score);

            const triggerPayload = {
              nsfw_score: data.nsfw_score || 0.1,
              violence_score: data.violence_score || 0.1,
              weapons_score: data.weapons_score || 0.1,
              child_age: childAge,
            };
            
            const triggerRes = await fetch(`${BACKEND_URL}/api/perception/trigger`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(triggerPayload),
            });
            
            if (triggerRes.ok) {
              const triggerData = await triggerRes.json();
              if (triggerData.threat_detected) {
                setActiveThreatType(triggerData.threat_type);
                setSessionId(triggerData.session_id);
                setCurrentPhase("Acknowledge");
                setChildEmotion("neutral");
                setAgreedToBoundary(false);
                setIsCompleted(false);
                setChatHistory([
                  { role: "assistant", content: JSON.stringify({ socratic_response_to_child: triggerData.initial_response, child_emotion: "neutral", agreed_to_boundary: false }) }
                ]);
                setInterceptActive(true);
              }
            }
          }
        }
      } catch (err) {
        // Silent connection issues
      }
    };
    
    interval = setInterval(pollGuardState, 2000);
    return () => clearInterval(interval);
  }, [guardSimMode, interceptActive, isCompleted, childAge]);

  const toggleSimulationMode = async (simMode: boolean) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/guard/toggle`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ simulation_mode: simMode }),
      });
      if (res.ok) {
        setGuardSimMode(simMode);
      }
    } catch (err) {
      console.error("Failed to toggle simulation mode:", err);
    }
  };

  const toggleMonitorActive = async (active: boolean) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/guard/toggle-monitor`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ active: active }),
      });
      if (res.ok) {
        setGuardActive(active);
      }
    } catch (err) {
      console.error("Failed to toggle monitor:", err);
    }
  };

  // Custom Video Frame Grabber & Analysis Loop
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    const grabFrameAndProcess = async () => {
      const video = videoRef.current;
      if (!video || video.paused || video.ended || guardSimMode) return;
      
      try {
        const canvas = document.createElement("canvas");
        canvas.width = Math.min(640, video.videoWidth || 640);
        canvas.height = Math.min(360, video.videoHeight || 360);
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frameB64 = canvas.toDataURL("image/jpeg", 0.65);
        
        setAnalyzingVideo(true);
        const res = await fetch(`${BACKEND_URL}/api/guard/process-frame`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_b64: frameB64,
            filename: videoFilename
          })
        });
        
        if (res.ok) {
          const data = await res.json();
          setGuardState(data);
          
          const nsfwBreached = data.nsfw_score > 0.80;
          const violenceBreached = data.violence_score > 0.80;
          const weaponsBreached = data.weapons_score > 0.75;
          const anyBreached = nsfwBreached || violenceBreached || weaponsBreached;

          if (anyBreached && !interceptActive && !isCompleted) {
            video.pause();
            
            let detectedType = "none";
            if (nsfwBreached) detectedType = "adult_content";
            else if (violenceBreached) detectedType = "violence";
            else if (weaponsBreached) detectedType = "weapons";

            setLastInterceptionStatus(`ALERT: Zero-Trust Guard intercepted threat: ${detectedType}!`);
            
            if (nsfwBreached) setAdultScore(data.nsfw_score);
            if (violenceBreached) setViolenceScore(data.violence_score);
            if (weaponsBreached) setWeaponsScore(data.weapons_score);

            const triggerPayload = {
              nsfw_score: data.nsfw_score || 0.1,
              violence_score: data.violence_score || 0.1,
              weapons_score: data.weapons_score || 0.1,
              child_age: childAge,
            };
            
            const triggerRes = await fetch(`${BACKEND_URL}/api/perception/trigger`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(triggerPayload),
            });
            
            if (triggerRes.ok) {
              const triggerData = await triggerRes.json();
              if (triggerData.threat_detected) {
                setActiveThreatType(triggerData.threat_type);
                setSessionId(triggerData.session_id);
                setCurrentPhase("Acknowledge");
                setChildEmotion("neutral");
                setAgreedToBoundary(false);
                setIsCompleted(false);
                setChatHistory([
                  { role: "assistant", content: JSON.stringify({ socratic_response_to_child: triggerData.initial_response, child_emotion: "neutral", agreed_to_boundary: false }) }
                ]);
                setInterceptActive(true);
              }
            }
          }
        }
      } catch (err) {
        console.error("Frame capture processing failed:", err);
      } finally {
        setAnalyzingVideo(false);
      }
    };
    
    if (!guardSimMode && videoUrl) {
      interval = setInterval(grabFrameAndProcess, 2000);
    }
    
    return () => clearInterval(interval);
  }, [guardSimMode, videoUrl, videoFilename, interceptActive, isCompleted, childAge]);

  const resetGuardSession = async () => {
    try {
      await fetch(`${BACKEND_URL}/api/guard/reset`, { method: "POST" });
      setInterceptActive(false);
      setIsCompleted(false);
      setChatHistory([]);
      setChildEmotion("neutral");
      setAgreedToBoundary(false);
      setVideoUrl("");
      setVideoFilename("");
      
      const res = await fetch(`${BACKEND_URL}/api/guard/state`);
      if (res.ok) {
        const data = await res.json();
        setGuardState(data);
      }
    } catch (err) {
      console.error("Failed to reset guard state:", err);
    }
  };

  const handleVideoUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    if (videoUrl && videoUrl.startsWith("blob:")) {
      URL.revokeObjectURL(videoUrl);
    }
    
    await resetGuardSession();
    
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    setVideoFilename(file.name);
    toggleSimulationMode(false);
  };

  const selectPresetVideo = async (type: "nature" | "combat") => {
    if (videoUrl && videoUrl.startsWith("blob:")) {
      URL.revokeObjectURL(videoUrl);
    }
    
    await resetGuardSession();
    
    if (type === "nature") {
      setVideoUrl("https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4");
      setVideoFilename("nature_wildlife_doc.mp4");
    } else {
      setVideoUrl("https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4");
      setVideoFilename("combat_match_battle_violence.mp4");
    }
    toggleSimulationMode(false);
  };

  // Code Hub state
  const [selectedCodeFile, setSelectedCodeFile] = useState<"agent" | "main" | "req" | "readme">("agent");
  const [copied, setCopied] = useState<boolean>(false);

  const getCodeText = () => {
    switch (selectedCodeFile) {
      case "agent": return codeSocraticAgent;
      case "main": return codeMainPy;
      case "req": return codeRequirements;
      case "readme": return codeReadme;
    }
  };

  const handleCopyCode = () => {
    navigator.clipboard.writeText(getCodeText());
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Trigger Socratic Intercept Simulation (Local FastAPI + LM Studio)
  const handleTriggerIntercept = async () => {
    setApiError(null);
    setLastInterceptionStatus("Analyzing content vectors...");

    const payload = {
      nsfw_score: adultScore,
      violence_score: violenceScore,
      weapons_score: weaponsScore,
      child_age: childAge,
    };

    try {
      // Direct local connection to FastAPI
      const res = await fetch(`${BACKEND_URL}/api/perception/trigger`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error(`Could not connect to local FastAPI backend. Ensure python main.py is running on ${BACKEND_URL}`);
      const data = await res.json();

      if (data.threat_detected) {
        setActiveThreatType(data.threat_type);
        setSessionId(data.session_id);
        setCurrentPhase("Acknowledge");
        setChildEmotion("neutral");
        setAgreedToBoundary(false);
        setIsCompleted(false);
        setChatHistory([
          { role: "assistant", content: JSON.stringify({ socratic_response_to_child: data.initial_response, child_emotion: "neutral", agreed_to_boundary: false }) }
        ]);
        setInterceptActive(true);
        setLastInterceptionStatus(`ALERT: Intercepted ${data.threat_type} locally! Dialog activated.`);
      } else {
        setLastInterceptionStatus("Local API says content is clean (max score <= 0.85).");
      }
    } catch (err: any) {
      console.error(err);
      setApiError(err.message || "Interception trigger failed.");
      setLastInterceptionStatus("Interception failed. Check local FastAPI server connection.");
    }
  };

  // Child Dialogue Turn (Local FastAPI + LM Studio)
  const handleSendDialogue = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!userInput.trim() || loadingTurn) return;

    const childText = userInput;
    setUserInput("");
    setLoadingTurn(true);
    setApiError(null);
    setChildEmotion("Analyzing...");

    // Append child response
    const nextHistory = [...chatHistory, { role: "user" as const, content: childText }];
    setChatHistory(nextHistory);

    try {
      // Local FastAPI dialogue turn
      const res = await fetch(`${BACKEND_URL}/api/dialogue/turn`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          child_response: childText
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Local FastAPI turn failed. Make sure python main.py is running on ${BACKEND_URL} and LM Studio server is started.`);
      }
      const data = await res.json();

      setCurrentPhase(data.current_phase);
      setChildEmotion(data.child_emotion);
      setAgreedToBoundary(data.agreed_to_boundary);
      setIsCompleted(data.completed);

      setChatHistory(prev => [
        ...prev,
        { 
          role: "assistant", 
          content: JSON.stringify({ 
            socratic_response_to_child: data.socratic_response_to_child, 
            child_emotion: data.child_emotion, 
            agreed_to_boundary: data.agreed_to_boundary 
          }) 
        }
      ]);
    } catch (err: any) {
      console.error(err);
      setApiError(err.message || "Dialogue turn failed.");
    } finally {
      setLoadingTurn(false);
    }
  };

  // Quick Preset Triggers for Testing
  const applyPreset = (type: "violence" | "weapons" | "adult" | "clean") => {
    if (type === "violence") {
      setViolenceScore(0.95);
      setWeaponsScore(0.15);
      setAdultScore(0.05);
    } else if (type === "weapons") {
      setViolenceScore(0.05);
      setWeaponsScore(0.92);
      setAdultScore(0.12);
    } else if (type === "adult") {
      setViolenceScore(0.02);
      setWeaponsScore(0.08);
      setAdultScore(0.98);
    } else {
      setViolenceScore(0.12);
      setWeaponsScore(0.22);
      setAdultScore(0.15);
    }
  };

  return (
    <div className="min-h-screen bg-[#F3F4ED] text-[#2D3025] flex flex-col font-sans antialiased" id="main-root">
      {/* Top Professional Header */}
      <header className="border-b border-[#DDE0D0] bg-white px-6 py-4 flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-sm" id="app-header">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-[#FAF9F6] text-[#5A5A40] rounded-lg border border-[#DDE0D0]" id="header-logo-container">
            <Shield className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-serif-natural font-semibold tracking-tight text-[#2D3025] flex items-center gap-2">
              Socratic Digital Guard <span className="text-xs bg-[#E6D5C3]/40 text-[#5A5A40] px-2 py-0.5 rounded border border-[#DDE0D0] font-sans">Prototype V1.0</span>
            </h1>
            <p className="text-xs text-[#6B705C]">Offline Pedagogical Cognitive Scaffold for Minor Safety</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center bg-[#FAF9F6] p-1 rounded-lg border border-[#DDE0D0]" id="header-nav">
          <button
            onClick={() => setActiveTab("sandbox")}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition ${
              activeTab === "sandbox" 
                ? "bg-[#5A5A40] text-white shadow-sm" 
                : "text-[#6B705C] hover:text-[#2D3025]"
            }`}
            id="tab-sandbox-btn"
          >
            <Sliders className="w-4 h-4" />
            Prototype Sandbox
          </button>
          <button
            onClick={() => setActiveTab("videoGuard")}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition ${
              activeTab === "videoGuard" 
                ? "bg-[#5A5A40] text-white shadow-sm" 
                : "text-[#6B705C] hover:text-[#2D3025]"
            }`}
            id="tab-guard-btn"
          >
            <Shield className="w-4 h-4" />
            Zero-Trust Video Guard
          </button>
          <button
            onClick={() => setActiveTab("codeHub")}
            className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition ${
              activeTab === "codeHub" 
                ? "bg-[#5A5A40] text-white shadow-sm" 
                : "text-[#6B705C] hover:text-[#2D3025]"
            }`}
            id="tab-code-btn"
          >
            <Code className="w-4 h-4" />
            Offline Python Hub
          </button>
          <Link
            href="/parent"
            className="flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition text-[#6B705C] hover:text-[#2D3025]"
            id="tab-parent-btn"
          >
            <User className="w-4 h-4" />
            Parent Dashboard
          </Link>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 p-6 max-w-[1700px] w-full mx-auto grid grid-cols-1 xl:grid-cols-12 gap-6" id="main-content">
        
        {activeTab === "sandbox" && (
          <>
            {/* Left Col - 5 Cols: Parental Control & State Telemetry */}
            <section className="xl:col-span-5 flex flex-col gap-6" id="controls-section">
              
              {/* Connection & General Config */}
              <div className="bg-white border border-[#DDE0D0] rounded-xl p-5 shadow-sm flex flex-col gap-4" id="conn-panel">
                <div className="flex items-center justify-between border-b border-[#DDE0D0] pb-3">
                  <h3 className="font-semibold text-[#2D3025] flex items-center gap-2 text-sm uppercase tracking-wider font-sans">
                    <Cpu className="w-4 h-4 text-[#5A5A40]" />
                    Cognitive Engine Connection
                  </h3>
                  <span className="flex items-center gap-1.5 text-xs font-semibold text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-full border border-emerald-200">
                    <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                    100% Offline Mode
                  </span>
                </div>

                <div className="p-3.5 bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg text-xs text-[#6B705C] flex items-start gap-2.5" id="local-backend-info">
                  <Terminal className="w-4 h-4 mt-0.5 shrink-0 text-[#5A5A40]" />
                  <div className="leading-relaxed">
                    Connected to <strong className="text-[#2D3025]">Local Python Backend</strong> (<code className="bg-[#F1F2EB] px-1 py-0.5 rounded text-[#2D3025]">{BACKEND_URL}</code>) &amp; <strong className="text-[#2D3025]">LM Studio</strong> (<code className="bg-[#F1F2EB] px-1 py-0.5 rounded text-[#2D3025]">http://localhost:1234</code>).
                  </div>
                </div>
              </div>

              {/* Perception Layer Slider Controls */}
              <div className="bg-white border border-[#DDE0D0] rounded-xl p-5 shadow-sm flex flex-col gap-5" id="perception-panel">
                <div className="flex items-center justify-between border-b border-[#DDE0D0] pb-3">
                  <h3 className="font-semibold text-[#2D3025] flex items-center gap-2 text-sm uppercase tracking-wider">
                    <Sliders className="w-4 h-4 text-[#5A5A40]" />
                    Perception Threat vectors
                  </h3>
                  <span className="text-xs text-[#6B705C]">Gate Threshold: &gt; 0.85</span>
                </div>

                {/* Quick Presets */}
                <div className="flex flex-wrap items-center gap-2" id="preset-buttons">
                  <span className="text-xs text-[#6B705C]">Quick Presets:</span>
                  <button 
                    onClick={() => applyPreset("clean")}
                    className="text-xs px-2.5 py-1 rounded bg-[#FAF9F6] border border-[#DDE0D0] text-[#5A5A40] hover:bg-[#E6D5C3]/40 hover:text-[#2D3025] transition"
                    id="preset-clean-btn"
                  >
                    🟢 Approved Content
                  </button>
                  <button 
                    onClick={() => applyPreset("violence")}
                    className="text-xs px-2.5 py-1 rounded bg-rose-50 border border-rose-200 text-rose-700 hover:bg-rose-100 transition"
                    id="preset-violence-btn"
                  >
                    💥 High Violence
                  </button>
                  <button 
                    onClick={() => applyPreset("weapons")}
                    className="text-xs px-2.5 py-1 rounded bg-orange-50 border border-orange-200 text-orange-700 hover:bg-orange-100 transition"
                    id="preset-weapons-btn"
                  >
                    🔫 Weapons / Arms
                  </button>
                  <button 
                    onClick={() => applyPreset("adult")}
                    className="text-xs px-2.5 py-1 rounded bg-purple-50 border border-purple-200 text-purple-700 hover:bg-purple-100 transition"
                    id="preset-adult-btn"
                  >
                    🔞 Adult Material
                  </button>
                </div>

                {/* Configuration Sliders */}
                <div className="flex flex-col gap-4" id="sliders-container">
                  {/* Age */}
                  <div className="flex flex-col gap-1.5" id="slider-age-container">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-[#2D3025] font-medium flex items-center gap-1">
                        <User className="w-3.5 h-3.5 text-[#5A5A40]" /> Child Age Context
                      </span>
                      <span className="font-bold text-[#5A5A40] bg-[#E6D5C3]/40 border border-[#DDE0D0] px-2 py-0.5 rounded text-xs">
                        {childAge} Years Old ({childAge <= 10 ? "Protective Prompt" : "Autonomy Prompt"})
                      </span>
                    </div>
                    <input 
                      type="range" 
                      min="5" 
                      max="16" 
                      value={childAge} 
                      onChange={(e) => setChildAge(parseInt(e.target.value))}
                      className="w-full accent-[#5A5A40] bg-[#FAF9F6] border border-[#DDE0D0] h-2 rounded-lg appearance-none cursor-pointer"
                      id="input-child-age"
                    />
                  </div>

                  {/* Violence */}
                  <div className="flex flex-col gap-1.5" id="slider-violence-container">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-[#2D3025] font-medium">Violence Score</span>
                      <span className={`font-bold ${violenceScore > 0.85 ? "text-rose-600 font-bold" : "text-[#6B705C]"}`}>
                        {violenceScore.toFixed(2)}
                      </span>
                    </div>
                    <input 
                      type="range" 
                      min="0.0" 
                      max="1.0" 
                      step="0.05"
                      value={violenceScore} 
                      onChange={(e) => setViolenceScore(parseFloat(e.target.value))}
                      className="w-full accent-[#5A5A40] bg-[#FAF9F6] border border-[#DDE0D0] h-2 rounded-lg appearance-none cursor-pointer"
                      id="input-violence"
                    />
                  </div>

                  {/* Weapons Detection */}
                  <div className="flex flex-col gap-1.5" id="slider-weapons-container">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-[#2D3025] font-medium">Weapons Score</span>
                      <span className={`font-bold ${weaponsScore > 0.75 ? "text-rose-600 font-bold" : "text-[#6B705C]"}`}>
                        {weaponsScore.toFixed(2)}
                      </span>
                    </div>
                    <input 
                      type="range" 
                      min="0.0" 
                      max="1.0" 
                      step="0.05"
                      value={weaponsScore} 
                      onChange={(e) => setWeaponsScore(parseFloat(e.target.value))}
                      className="w-full accent-[#5A5A40] bg-[#FAF9F6] border border-[#DDE0D0] h-2 rounded-lg appearance-none cursor-pointer"
                      id="input-weapons"
                    />
                  </div>

                  {/* Adult Content */}
                  <div className="flex flex-col gap-1.5" id="slider-adult-container">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-[#2D3025] font-medium">Adult Content Score</span>
                      <span className={`font-bold ${adultScore > 0.85 ? "text-rose-600 font-bold" : "text-[#6B705C]"}`}>
                        {adultScore.toFixed(2)}
                      </span>
                    </div>
                    <input 
                      type="range" 
                      min="0.0" 
                      max="1.0" 
                      step="0.05"
                      value={adultScore} 
                      onChange={(e) => setAdultScore(parseFloat(e.target.value))}
                      className="w-full accent-[#5A5A40] bg-[#FAF9F6] border border-[#DDE0D0] h-2 rounded-lg appearance-none cursor-pointer"
                      id="input-adult"
                    />
                  </div>
                </div>

                {/* Simulate Button */}
                <button
                  onClick={() => handleTriggerIntercept()}
                  className="w-full py-3 px-4 bg-[#5A5A40] hover:bg-[#454530] text-white font-medium rounded-lg shadow-md flex items-center justify-center gap-2 transition duration-200"
                  id="btn-trigger-intercept"
                >
                  <AlertTriangle className="w-4 h-4" />
                  Evaluate Content & Simulate Intercept
                </button>
              </div>

              {/* Live State Machine Telemetry Panel */}
              <div className="bg-white border border-[#DDE0D0] rounded-xl p-5 shadow-sm flex flex-col gap-4 flex-1" id="telemetry-panel">
                <div className="flex items-center justify-between border-b border-[#DDE0D0] pb-3">
                  <h3 className="font-semibold text-[#2D3025] flex items-center gap-2 text-sm uppercase tracking-wider">
                    <Terminal className="w-4 h-4 text-[#5A5A40]" />
                    Cognitive Telemetry Logs
                  </h3>
                  <button 
                    onClick={() => {
                      setInterceptActive(false);
                      setChatHistory([]);
                      setIsCompleted(false);
                      setCurrentPhase("Acknowledge");
                      setChildEmotion("neutral");
                      setLastInterceptionStatus("Ready & Guarding System");
                    }}
                    className="text-[#6B705C] hover:text-[#2D3025] text-xs flex items-center gap-1 transition"
                    id="btn-reset-session"
                  >
                    <RefreshCw className="w-3 h-3" /> Reset Session
                  </button>
                </div>

                {/* Status indicator bar */}
                <div className="p-3 rounded-lg bg-[#FAF9F6] border border-[#DDE0D0] text-xs font-mono flex items-center gap-2 text-[#2D3025]" id="telemetry-bar">
                  <div className={`w-2.5 h-2.5 rounded-full ${interceptActive ? "bg-rose-500 animate-pulse" : "bg-emerald-600"}`} />
                  <span className="text-[#6B705C]">System Status:</span>
                  <span className={interceptActive ? "text-rose-700 font-bold" : "text-emerald-700 font-medium"}>
                    {lastInterceptionStatus}
                  </span>
                </div>

                {apiError && (
                  <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs" id="api-error-display">
                    <strong>Error:</strong> {apiError}
                  </div>
                )}

                {/* State Variables Table */}
                <div className="grid grid-cols-2 gap-3" id="telemetry-grid">
                  <div className="bg-[#FAF9F6] p-3 rounded-lg border border-[#DDE0D0] text-xs" id="tel-session-id">
                    <p className="text-[#6B705C] mb-1 font-medium">Active Session</p>
                    <p className="font-mono text-[#2D3025] truncate font-semibold">
                      {sessionId ? sessionId : "no_active_session"}
                    </p>
                  </div>
                  <div className="bg-[#FAF9F6] p-3 rounded-lg border border-[#DDE0D0] text-xs" id="tel-threat-type">
                    <p className="text-[#6B705C] mb-1 font-medium">Active Threat Type</p>
                    <p className={`font-mono font-bold capitalize ${activeThreatType !== "none" ? "text-rose-700" : "text-[#6B705C]"}`}>
                      {activeThreatType}
                    </p>
                  </div>
                  <div className="bg-[#FAF9F6] p-3 rounded-lg border border-[#DDE0D0] text-xs" id="tel-phase">
                    <p className="text-[#6B705C] mb-1 font-medium">State Phase</p>
                    <div className="flex items-center gap-1.5">
                      <span className="font-bold text-[#5A5A40]">
                        {interceptActive ? currentPhase : "Guarding"}
                      </span>
                    </div>
                  </div>
                  <div className="bg-[#FAF9F6] p-3 rounded-lg border border-[#DDE0D0] text-xs" id="tel-emotion">
                    <p className="text-[#6B705C] mb-1 font-medium">Extracted Emotion</p>
                    <p className="font-mono font-semibold capitalize text-[#5A5A40] flex items-center gap-1">
                      {childEmotion}
                    </p>
                  </div>
                </div>

                {/* Memory window tracker */}
                <div className="flex flex-col gap-1.5 flex-1 min-h-[140px]" id="memory-window-panel">
                  <div className="flex items-center justify-between text-xs text-[#6B705C]">
                    <span>Sliding Window Context (Max 4 turns)</span>
                    <span className="text-[#6B705C] opacity-75">Prunes automatically</span>
                  </div>
                  <div className="bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg p-3 font-mono text-[11px] overflow-y-auto max-h-[180px] flex-1 flex flex-col gap-2 text-[#2D3025]" id="memory-log-list">
                    {chatHistory.length === 0 ? (
                      <span className="text-[#6B705C] italic">Memory empty. Waiting for intercept...</span>
                    ) : (
                      chatHistory.map((msg, i) => {
                        let displayText = msg.content;
                        try {
                          const parsed = JSON.parse(msg.content);
                          if (parsed && parsed.socratic_response_to_child) {
                            displayText = parsed.socratic_response_to_child;
                          }
                        } catch {
                          // Not a JSON turn
                        }

                        return (
                          <div key={i} className={`pb-2 ${i !== chatHistory.length - 1 ? "border-b border-[#DDE0D0]" : ""}`}>
                            <span className={msg.role === "assistant" ? "text-[#5A5A40] font-bold" : "text-[#6B705C] font-semibold"}>
                              [{msg.role.toUpperCase()}]:
                            </span>{" "}
                            <span className="text-[#2D3025]">{displayText}</span>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>

              </div>
            </section>

            {/* Right Col - 7 Cols: The Interactive Active Screen / Intercept Overlay */}
            <section className="xl:col-span-7 flex flex-col" id="simulator-section">
              <div className="bg-white border border-[#DDE0D0] rounded-xl overflow-hidden shadow-sm flex-1 flex flex-col min-h-[600px] relative" id="desktop-simulator">
                
                {/* Simulated Desktop Window Title bar */}
                <div className="bg-[#FAF9F6] px-4 py-2.5 border-b border-[#DDE0D0] flex items-center justify-between" id="desktop-titlebar">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-rose-400" />
                    <span className="w-3 h-3 rounded-full bg-amber-400" />
                    <span className="w-3 h-3 rounded-full bg-emerald-400" />
                    <span className="text-xs text-[#6B705C] font-semibold ml-2 flex items-center gap-1.5 font-mono">
                      <Chrome className="w-3 h-3 text-[#5A5A40]" /> SIMULATED_CHILD_DESKTOP
                    </span>
                  </div>
                  <div className="flex gap-1" id="active-app-selectors">
                    <button 
                      onClick={() => {
                        if (interceptActive) return;
                        setChildActiveApp("video");
                      }}
                      className={`text-[11px] px-2.5 py-1 rounded transition font-medium ${
                        childActiveApp === "video" ? "bg-[#5A5A40] text-white shadow-sm" : "text-[#6B705C] hover:text-[#2D3025]"
                      } ${interceptActive ? "opacity-50 cursor-not-allowed" : ""}`}
                      disabled={interceptActive}
                      id="sim-app-video-btn"
                    >
                      Video Portal
                    </button>
                    <button 
                      onClick={() => {
                        if (interceptActive) return;
                        setChildActiveApp("game");
                      }}
                      className={`text-[11px] px-2.5 py-1 rounded transition font-medium ${
                        childActiveApp === "game" ? "bg-[#5A5A40] text-white shadow-sm" : "text-[#6B705C] hover:text-[#2D3025]"
                      } ${interceptActive ? "opacity-50 cursor-not-allowed" : ""}`}
                      disabled={interceptActive}
                      id="sim-app-game-btn"
                    >
                      Safe Games
                    </button>
                  </div>
                </div>

                {/* Desktop Screen Contents (Blurred when socratic buddy takes over) */}
                <div className={`flex-1 p-6 flex flex-col bg-[#FAF9F6] transition duration-500 ${interceptActive ? "blur-md select-none pointer-events-none scale-98" : ""}`} id="desktop-screen">
                  {childActiveApp === "video" && (
                    <div className="flex flex-col gap-4 flex-1" id="app-video-view">
                      <div className="flex items-center justify-between border-b border-[#DDE0D0] pb-3">
                        <div>
                          <h2 className="text-base font-semibold text-[#2D3025]">MeTube Kids View</h2>
                          <p className="text-xs text-[#6B705C]">Welcome to your child&apos;s media portal</p>
                        </div>
                        <span className="text-xs text-emerald-700 flex items-center gap-1 font-semibold">
                          <CheckCircle className="w-3 h-3 text-emerald-600" /> Safe Filter Enabled
                        </span>
                      </div>

                      {/* Mock Videos Grid */}
                      <div className="grid grid-cols-2 gap-4 flex-1" id="videos-grid">
                        <div 
                          onClick={() => {
                            applyPreset("clean");
                            handleTriggerIntercept();
                          }}
                          className="group border border-[#DDE0D0] rounded-lg overflow-hidden bg-white cursor-pointer hover:border-[#5A5A40] transition flex flex-col shadow-xs"
                          id="video-clean-card"
                        >
                          <div className="aspect-video bg-[#E6D5C3]/15 relative flex items-center justify-center border-b border-[#DDE0D0]">
                            <span className="text-xl text-[#5A5A40] font-serif-natural font-medium italic">Nature Documentary</span>
                            <span className="absolute bottom-2 right-2 bg-[#2D3025]/85 text-[10px] px-1.5 py-0.5 rounded text-white">10:45</span>
                          </div>
                          <div className="p-3">
                            <h4 className="text-xs font-semibold text-[#2D3025] group-hover:text-[#5A5A40] truncate">The Secret Life of Honeybees</h4>
                            <p className="text-[10px] text-[#6B705C] mt-1">Science & Nature channel</p>
                          </div>
                        </div>

                        <div 
                          onClick={() => {
                            applyPreset("violence");
                            setTimeout(() => {
                              handleTriggerIntercept();
                            }, 100);
                          }}
                          className="group border border-rose-200 rounded-lg overflow-hidden bg-white cursor-pointer hover:border-rose-500 transition flex flex-col shadow-xs"
                          id="video-violence-card"
                        >
                          <div className="aspect-video bg-rose-50/50 relative flex items-center justify-center border-b border-rose-100">
                            <span className="text-xl text-rose-600 font-bold">Extreme Battle</span>
                            <span className="absolute bottom-2 right-2 bg-rose-600 text-[10px] px-1.5 py-0.5 rounded text-white font-bold">18+</span>
                          </div>
                          <div className="p-3">
                            <h4 className="text-xs font-semibold text-[#2D3025] group-hover:text-rose-600 truncate">Mega Combat Arena - Blood Match 7</h4>
                            <p className="text-[10px] text-rose-600 font-bold mt-1">Unfiltered Fighting Hub</p>
                          </div>
                        </div>

                        <div 
                          onClick={() => {
                            applyPreset("adult");
                            setTimeout(() => {
                              handleTriggerIntercept();
                            }, 100);
                          }}
                          className="group border border-purple-200 rounded-lg overflow-hidden bg-white cursor-pointer hover:border-purple-500 transition flex flex-col shadow-xs"
                          id="video-adult-card"
                        >
                          <div className="aspect-video bg-purple-50/50 relative flex items-center justify-center border-b border-purple-100">
                            <span className="text-xl text-purple-600 font-bold">Adult Content</span>
                            <span className="absolute bottom-2 right-2 bg-purple-600 text-[10px] px-1.5 py-0.5 rounded text-white font-bold">Unfiltered</span>
                          </div>
                          <div className="p-3">
                            <h4 className="text-xs font-semibold text-[#2D3025] group-hover:text-purple-600 truncate">Forbidden Chats & Mature Forums Link</h4>
                            <p className="text-[10px] text-purple-600 font-bold mt-1">Adult Community</p>
                          </div>
                        </div>

                        <div 
                          onClick={() => {
                            applyPreset("weapons");
                            setTimeout(() => {
                              handleTriggerIntercept();
                            }, 100);
                          }}
                          className="group border border-amber-200 rounded-lg overflow-hidden bg-white cursor-pointer hover:border-amber-600 transition flex flex-col shadow-xs"
                          id="video-weapons-card"
                        >
                          <div className="aspect-video bg-amber-50/50 relative flex items-center justify-center border-b border-amber-100">
                            <span className="text-xl text-amber-700 font-bold">Tactical Firearms Demo</span>
                            <span className="absolute bottom-2 right-2 bg-amber-600 text-[10px] px-1.5 py-0.5 rounded text-white font-bold">Weapons</span>
                          </div>
                          <div className="p-3">
                            <h4 className="text-xs font-semibold text-[#2D3025] group-hover:text-amber-600 truncate">Tactical Gear & Banned Weapons Demo</h4>
                            <p className="text-[10px] text-amber-600 font-bold mt-1">Weapons / Arms</p>
                          </div>
                        </div>
                      </div>

                      <div className="text-center p-3 bg-[#E6D5C3]/20 rounded-lg border border-[#DDE0D0] text-[11px] text-[#6B705C]" id="video-info-footer">
                        💡 Clicking on any of the cards above will simulate threat detection and immediately trigger Socratic Buddy!
                      </div>
                    </div>
                  )}

                  {childActiveApp === "game" && (
                    <div className="flex flex-col gap-4 flex-1 text-center justify-center max-w-md mx-auto" id="app-game-view">
                      <div className="p-6 bg-white rounded-xl border border-[#DDE0D0] flex flex-col gap-3 items-center shadow-xs">
                        <Gamepad2 className="w-12 h-12 text-[#5A5A40]" />
                        <h2 className="text-lg font-bold text-[#2D3025]">Fun Educational Games</h2>
                        <p className="text-xs text-[#6B705C] leading-relaxed">
                          Your child is playing safe, curated games. These are completely approved and will never trigger an interception.
                        </p>
                        <div className="mt-2 w-full bg-[#FAF9F6] border border-[#DDE0D0] h-3 rounded-full overflow-hidden">
                          <div className="bg-[#5A5A40] h-full w-[45%]" />
                        </div>
                        <span className="text-[10px] text-[#6B705C] font-mono">Curriculum Level: Grade 4 Math Challenge</span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Secure Screen Interception Overlay */}
                <AnimatePresence>
                  {interceptActive && (
                    <motion.div 
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="absolute inset-0 bg-[#2D3025]/80 backdrop-blur-xs flex flex-col items-center justify-center p-6 z-50 overflow-hidden"
                      id="socratic-buddy-interceptor"
                    >
                      <motion.div 
                        initial={{ scale: 0.95, y: 15 }}
                        animate={{ scale: 1, y: 0 }}
                        exit={{ scale: 0.95, y: 15 }}
                        className="bg-white border border-[#DDE0D0] rounded-2xl w-full max-w-lg shadow-xl flex flex-col overflow-hidden h-[95%] relative"
                        id="interceptor-dialog-card"
                      >
                        
                        {/* Safe Intervention Banner */}
                        <div className="bg-[#5A5A40] p-4 flex items-center justify-between text-white border-b border-[#DDE0D0] shrink-0" id="buddy-banner">
                          <div className="flex items-center gap-3">
                            <div className="p-1.5 bg-white/10 rounded-lg">
                              <Shield className="w-5 h-5 text-white" />
                            </div>
                            <div>
                              <h3 className="text-sm font-bold tracking-tight text-white">Socratic Buddy Safe Guard</h3>
                              <p className="text-[10px] text-[#E6D5C3]">Screen Shield Active</p>
                            </div>
                          </div>
                          
                          <div className="flex items-center gap-1 bg-white/10 px-2 py-1 rounded text-[10px] font-mono border border-white/5">
                            <Lock className="w-3 h-3 text-white" /> Secure Offline Intercept
                          </div>
                        </div>

                        {/* Dialogue Timeline Progress Header */}
                        <div className="bg-[#FAF9F6] border-b border-[#DDE0D0] px-4 py-3 flex items-center justify-between text-[11px] shrink-0 text-[#6B705C]" id="dialogue-timeline">
                          <span className="text-[#6B705C] font-semibold uppercase tracking-wider">State Machine progression:</span>
                          <div className="flex items-center gap-1" id="progress-states-list">
                            <span className={`px-2 py-0.5 rounded font-medium ${
                              currentPhase === "Acknowledge" 
                                ? "bg-[#E6D5C3]/40 text-[#2D3025] border border-[#5A5A40] font-bold" 
                                : "bg-white text-[#6B705C] border border-transparent"
                            }`}>
                              1. Ack
                            </span>
                            <ChevronRight className="w-3 h-3 text-[#6B705C]" />
                            <span className={`px-2 py-0.5 rounded font-medium ${
                              currentPhase === "Reason" 
                                ? "bg-[#E6D5C3]/40 text-[#2D3025] border border-[#5A5A40] font-bold" 
                                : "bg-white text-[#6B705C] border border-transparent"
                            }`}>
                              2. Reason
                            </span>
                            <ChevronRight className="w-3 h-3 text-[#6B705C]" />
                            <span className={`px-2 py-0.5 rounded font-medium ${
                              currentPhase === "Contract" 
                                ? "bg-[#E6D5C3]/40 text-[#2D3025] border border-[#5A5A40] font-bold" 
                                : "bg-white text-[#6B705C] border border-transparent"
                            }`}>
                              3. Contract
                            </span>
                          </div>
                        </div>

                        {/* Interactive Chat Board area */}
                        <div className="flex-1 p-4 overflow-y-auto flex flex-col gap-4 bg-[#FAF9F6]" id="buddy-chat-history">
                          
                          {/* Socratic Avatar Greeting Bubble */}
                          <div className="flex items-start gap-3" id="buddy-avatar-greeting">
                            <div className="w-10 h-10 rounded-xl bg-[#5A5A40] flex items-center justify-center font-bold text-white shadow-xs shrink-0">
                              SB
                            </div>
                            <div className="bg-white border border-[#DDE0D0] rounded-xl p-3.5 max-w-[85%] text-[#2D3025] text-xs shadow-xs leading-relaxed">
                              <span className="font-bold text-[#5A5A40] block mb-1">Socratic Buddy</span>
                              Hi! Socratic Buddy here. Socratic Buddy always protects minors, so I&apos;ve covered the screen to keep you safe. Don&apos;t worry, you aren&apos;t in any trouble. Let&apos;s talk this through together.
                            </div>
                          </div>

                          {chatHistory.map((item, index) => {
                            let textContent = item.content;
                            let metaInfo = null;

                            // Parse structured telemetry response to render only the speech to child
                            try {
                              const parsed = JSON.parse(item.content);
                              if (parsed && parsed.socratic_response_to_child) {
                                textContent = parsed.socratic_response_to_child;
                                metaInfo = {
                                  emotion: parsed.child_emotion,
                                  agreed: parsed.agreed_to_boundary
                                };
                              }
                            } catch {
                              // Plain string (child user typed response)
                            }

                            return (
                              <div 
                                key={index} 
                                className={`flex items-start gap-3 ${item.role === "user" ? "flex-row-reverse" : ""}`}
                                id={`chat-turn-${index}`}
                              >
                                <div className={`w-10 h-10 rounded-xl flex items-center justify-center font-bold shadow-xs shrink-0 ${
                                  item.role === "user" 
                                    ? "bg-[#E6D5C3] text-[#2D3025] border border-[#5A5A40]" 
                                    : "bg-[#5A5A40] text-white border border-[#5A5A40]"
                                }`}>
                                  {item.role === "user" ? "ME" : "SB"}
                                </div>
                                
                                <div className={`border rounded-xl p-3.5 max-w-[85%] text-xs shadow-xs leading-relaxed ${
                                  item.role === "user"
                                    ? "bg-[#E6D5C3]/20 border-[#DDE0D0] text-[#2D3025]"
                                    : "bg-white border-[#DDE0D0] text-[#2D3025]"
                                }`}>
                                  <span className={`font-bold block mb-1 ${item.role === "user" ? "text-[#5A5A40]" : "text-[#5A5A40]"}`}>
                                    {item.role === "user" ? "My Response" : "Socratic Buddy"}
                                  </span>
                                  {textContent}

                                  {metaInfo && (
                                    <div className="mt-2.5 pt-2 border-t border-[#DDE0D0] flex items-center gap-2 text-[10px] text-[#6B705C]">
                                      <span className="bg-[#FAF9F6] px-1.5 py-0.5 rounded border border-[#DDE0D0]">
                                        Emotion: <span className="text-[#5A5A40] font-bold capitalize">{metaInfo.emotion}</span>
                                      </span>
                                      {metaInfo.agreed && (
                                        <span className="bg-emerald-50 text-emerald-800 border border-emerald-200 px-1.5 py-0.5 rounded font-bold">
                                          ✓ Boundary Agreed
                                        </span>
                                      )}
                                    </div>
                                  )}
                                </div>
                              </div>
                            );
                          })}

                          {loadingTurn && (
                            <div className="flex items-center gap-3" id="buddy-loading-indicator">
                              <div className="w-10 h-10 rounded-xl bg-[#5A5A40] flex items-center justify-center font-bold text-white border border-[#5A5A40] shrink-0">
                                SB
                              </div>
                              <div className="bg-white border border-[#DDE0D0] rounded-xl p-3 max-w-[85%] text-[#6B705C] text-xs shadow-xs italic flex items-center gap-2">
                                <RefreshCw className="w-3.5 h-3.5 animate-spin text-[#5A5A40]" />
                                Analyzing and formulating safety response...
                              </div>
                            </div>
                          )}

                          <div ref={chatEndRef} />
                        </div>

                        {/* Action Redirection Screen once child agreed (Is Completed) */}
                        {isCompleted && (
                          <motion.div 
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="absolute inset-0 bg-[#FAF9F6] flex flex-col items-center justify-center p-6 text-center z-50 border-t border-[#DDE0D0]"
                            id="pivot-redirection-overlay"
                          >
                            <div className="bg-white border border-emerald-200 p-4 rounded-2xl max-w-sm flex flex-col items-center gap-4 shadow-sm">
                              <CheckCircle className="w-16 h-16 text-emerald-600" />
                              <div>
                                <h3 className="text-base font-bold text-[#2D3025]">Safety Agreement Confirmed!</h3>
                                <p className="text-xs text-[#6B705C] mt-1">
                                  You did an amazing job talking this through. We have closed that unsafe window. Let&apos;s redirect to a fun learning site together!
                                </p>
                              </div>
                              <button
                                onClick={() => {
                                  setInterceptActive(false);
                                  setChatHistory([]);
                                  setIsCompleted(false);
                                  setChildActiveApp("game");
                                  setLastInterceptionStatus("Successfully scaffolded child. Safe app loaded.");
                                }}
                                className="w-full py-2.5 px-4 bg-[#5A5A40] hover:bg-[#454530] text-white font-bold rounded-lg text-xs flex items-center justify-center gap-2 transition duration-200 shadow-sm"
                                id="btn-redirect-fun"
                              >
                                Redirect to Currie&apos;s Sandbox <ArrowRight className="w-3.5 h-3.5" />
                              </button>
                            </div>
                          </motion.div>
                        )}

                        {/* Input Area */}
                        <form 
                          onSubmit={handleSendDialogue}
                          className="bg-white border-t border-[#DDE0D0] p-3.5 flex items-center gap-2 shrink-0"
                          id="chat-input-form"
                        >
                          <input 
                            type="text" 
                            placeholder="Type your response to Socratic Buddy..."
                            value={userInput}
                            onChange={(e) => setUserInput(e.target.value)}
                            className="flex-1 bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg py-2.5 px-3.5 text-xs text-[#2D3025] focus:outline-none focus:border-[#5A5A40] focus:ring-1 focus:ring-[#5A5A40]"
                            disabled={loadingTurn || isCompleted}
                            id="input-chat-text"
                          />
                          <button
                            type="submit"
                            className="p-2.5 rounded-lg bg-[#5A5A40] hover:bg-[#454530] text-white transition flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
                            disabled={!userInput.trim() || loadingTurn || isCompleted}
                            id="btn-send-chat"
                          >
                            <Send className="w-4 h-4" />
                          </button>
                        </form>

                      </motion.div>
                    </motion.div>
                  )}
                </AnimatePresence>

              </div>
            </section>
          </>
        )}

        {/* Offline Code Hub View */}
        {activeTab === "codeHub" && (
          <section className="col-span-12 flex flex-col gap-6" id="code-hub-section">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl" id="code-hub-container">
              <div className="border-b border-slate-800 pb-4 mb-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                    <BookOpen className="w-5 h-5 text-indigo-400" />
                    Offline Integration Hub
                  </h2>
                  <p className="text-xs text-slate-400">Complete modular source code for local zero-data-transit deployment on child computers</p>
                </div>
                
                <div className="flex items-center gap-2 bg-slate-950 p-1 rounded-lg border border-slate-800" id="file-selector-hub">
                  <button
                    onClick={() => { setSelectedCodeFile("agent"); setCopied(false); }}
                    className={`px-3 py-1.5 rounded text-xs font-semibold flex items-center gap-1.5 transition ${
                      selectedCodeFile === "agent" ? "bg-indigo-500/10 border border-indigo-500/30 text-indigo-300" : "text-slate-400 hover:text-slate-200"
                    }`}
                    id="select-agent-file"
                  >
                    <FileText className="w-3.5 h-3.5" />
                    socratic_agent.py
                  </button>
                  <button
                    onClick={() => { setSelectedCodeFile("main"); setCopied(false); }}
                    className={`px-3 py-1.5 rounded text-xs font-semibold flex items-center gap-1.5 transition ${
                      selectedCodeFile === "main" ? "bg-indigo-500/10 border border-indigo-500/30 text-indigo-300" : "text-slate-400 hover:text-slate-200"
                    }`}
                    id="select-main-file"
                  >
                    <FileText className="w-3.5 h-3.5" />
                    main.py
                  </button>
                  <button
                    onClick={() => { setSelectedCodeFile("req"); setCopied(false); }}
                    className={`px-3 py-1.5 rounded text-xs font-semibold flex items-center gap-1.5 transition ${
                      selectedCodeFile === "req" ? "bg-indigo-500/10 border border-indigo-500/30 text-indigo-300" : "text-slate-400 hover:text-slate-200"
                    }`}
                    id="select-req-file"
                  >
                    <FileText className="w-3.5 h-3.5" />
                    requirements.txt
                  </button>
                  <button
                    onClick={() => { setSelectedCodeFile("readme"); setCopied(false); }}
                    className={`px-3 py-1.5 rounded text-xs font-semibold flex items-center gap-1.5 transition ${
                      selectedCodeFile === "readme" ? "bg-[#E6D5C3]/40 border border-[#5A5A40] text-[#2D3025]" : "text-[#6B705C] hover:text-[#2D3025]"
                    }`}
                    id="select-readme-file"
                  >
                    <FileText className="w-3.5 h-3.5" />
                    README.md
                  </button>
                </div>
              </div>

              {/* Code Viewer Panel */}
              <div className="bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg overflow-hidden flex flex-col h-[550px]" id="code-viewer-panel">
                <div className="bg-[#FAF9F6] px-4 py-2 flex items-center justify-between border-b border-[#DDE0D0] text-xs shrink-0" id="viewer-header">
                  <span className="font-mono text-[#6B705C]">
                    {selectedCodeFile === "agent" && "socratic_agent.py"}
                    {selectedCodeFile === "main" && "main.py"}
                    {selectedCodeFile === "req" && "requirements.txt"}
                    {selectedCodeFile === "readme" && "README.md"}
                  </span>
                  <button
                    onClick={handleCopyCode}
                    className="flex items-center gap-1.5 px-3 py-1 rounded bg-white text-[#6B705C] hover:text-[#2D3025] border border-[#DDE0D0] text-xs transition shadow-xs"
                    id="btn-copy-code"
                  >
                    {copied ? (
                      <>
                        <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                        Copied!
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5 text-[#5A5A40]" />
                        Copy Code
                      </>
                    )}
                  </button>
                </div>
                
                <div className="flex-1 p-4 overflow-auto font-mono text-xs leading-relaxed text-[#2D3025] bg-[#FAF9F6]" id="code-block-viewer">
                  <pre className="whitespace-pre">
                    <code>{getCodeText()}</code>
                  </pre>
                </div>
              </div>

              {/* Step by Step Running Guides */}
              <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6" id="guides-grid">
                <div className="p-4 bg-white border border-[#DDE0D0] rounded-lg text-xs" id="guide-step-1">
                  <h4 className="font-bold text-[#5A5A40] flex items-center gap-1.5 mb-2 uppercase">
                    <span>1.</span> Load Local Language Model
                  </h4>
                  <p className="text-[#6B705C] leading-relaxed">
                    Download and open <strong>LM Studio</strong>. Search and load <strong>google/gemma-3-1b</strong> (or <strong>Llama-3.2-1B-Instruct</strong>). Navigate to the local server tab, verify port is set to <code>1234</code>, and click <strong>Start Server</strong>.
                  </p>
                </div>
                <div className="p-4 bg-white border border-[#DDE0D0] rounded-lg text-xs" id="guide-step-2">
                  <h4 className="font-bold text-[#5A5A40] flex items-center gap-1.5 mb-2 uppercase">
                    <span>2.</span> Run Python Socratic Agent
                  </h4>
                  <p className="text-[#6B705C] leading-relaxed">
                    Inside your project, navigate to the <code>offline_backend</code> folder. Create your Python virtual environment, execute <code>pip install -r requirements.txt</code>, and start the FastAPI service with <code>python main.py</code>.
                  </p>
                </div>
                <div className="p-4 bg-white border border-[#DDE0D0] rounded-lg text-xs" id="guide-step-3">
                  <h4 className="font-bold text-[#5A5A40] flex items-center gap-1.5 mb-2 uppercase">
                    <span>3.</span> Connect Interceptor Dialogue
                  </h4>
                  <p className="text-[#6B705C] leading-relaxed">
                    Once the Python backend is running on <code>localhost:8000</code>, return to the **Prototype Sandbox** tab of this page and toggle the connection mode to <strong>Local Python Backend</strong>. All threat triggering and conversation loops will query your local model!
                  </p>
                </div>
              </div>

            </div>
          </section>
        )}
        {/* Simplified Zero-Trust Video Guard View */}
        {activeTab === "videoGuard" && (
          <section className="col-span-12 grid grid-cols-1 xl:grid-cols-12 gap-6 animate-fade-in" id="video-guard-section">
            
            {/* Left Column: 6 Cols - Child Video Portal */}
            <div className="xl:col-span-6 flex flex-col gap-6">
              
              {/* Active Player Card */}
              <div className="bg-white border border-[#DDE0D0] rounded-xl p-5 shadow-sm flex flex-col gap-4">
                <h3 className="font-semibold text-[#2D3025] flex items-center gap-2 text-sm uppercase tracking-wider font-sans border-b border-[#DDE0D0] pb-2">
                  <Sliders className="w-4 h-4 text-[#5A5A40]" />
                  Child Video Portal
                </h3>
                
                <div className="flex flex-col gap-1.5 flex-1 justify-center">
                  <span className="text-[10px] uppercase font-bold text-[#6B705C] font-sans">Active Video Player</span>
                  <div className="relative aspect-video rounded-lg border border-[#DDE0D0] overflow-hidden bg-black flex items-center justify-center shadow-inner w-full">
                    {videoUrl ? (
                      <video 
                        ref={videoRef}
                        src={videoUrl}
                        controls
                        className={`w-full h-full object-contain transition duration-500 ${
                          interceptActive ? "blur-[25px] pointer-events-none" : ""
                        }`}
                      />
                    ) : (
                      <div className="text-[#6B705C] text-xs text-center p-6 italic leading-relaxed">
                        No video loaded. Select a demo preset or upload a custom file below to start!
                      </div>
                    )}
                    
                    {interceptActive && (
                      <div className="fixed inset-0 bg-[#2D3025]/80 backdrop-blur-md flex items-center justify-center p-4 md:p-6 z-[9999] overflow-hidden" id="socratic-buddy-interceptor-vg">
                        <div className="bg-white border border-[#DDE0D0] rounded-2xl w-full max-w-lg shadow-2xl flex flex-col overflow-hidden h-[90vh] max-h-[600px] relative animate-fade-in" id="interceptor-dialog-card-vg">
                          {/* Safe Intervention Banner */}
                          <div className="bg-[#5A5A40] px-3 py-1.5 flex items-center justify-between text-white border-b border-[#DDE0D0] shrink-0">
                            <div className="flex items-center gap-1.5">
                              <Shield className="w-3.5 h-3.5 text-white" />
                              <span className="text-[10px] font-bold tracking-tight text-white font-sans uppercase">Socratic Shield Intercept</span>
                            </div>
                            <div className="flex items-center gap-1 bg-white/10 px-1.5 py-0.5 rounded text-[8px] font-mono">
                              <Lock className="w-2.5 h-2.5 text-white" /> Offline
                            </div>
                          </div>

                          {/* Dialogue progress header */}
                          <div className="bg-[#FAF9F6] border-b border-[#DDE0D0] px-3 py-1 flex items-center justify-between text-[9px] shrink-0 text-[#6B705C]">
                            <span className="font-semibold uppercase tracking-wider text-[8px]">State:</span>
                            <div className="flex items-center gap-1">
                              <span className={`px-1 rounded ${currentPhase === "Acknowledge" ? "bg-[#E6D5C3] text-[#2D3025] font-bold" : ""}`}>1. Ack</span>
                              <span className={`px-1 rounded ${currentPhase === "Reason" ? "bg-[#E6D5C3] text-[#2D3025] font-bold" : ""}`}>2. Reason</span>
                              <span className={`px-1 rounded ${currentPhase === "Contract" ? "bg-[#E6D5C3] text-[#2D3025] font-bold" : ""}`}>3. Contract</span>
                            </div>
                          </div>

                          {/* Interactive Chat Board area */}
                          <div className="flex-1 p-2 overflow-y-auto flex flex-col gap-2 bg-[#FAF9F6] scrollbar-thin">
                            <div className="flex items-start gap-2">
                              <div className="w-7 h-7 rounded bg-[#5A5A40] flex items-center justify-center font-bold text-white text-[9px] shrink-0">SB</div>
                              <div className="bg-white border border-[#DDE0D0] rounded-lg p-2 max-w-[85%] text-[#2D3025] text-[10px] leading-relaxed">
                                <span className="font-bold text-[#5A5A40] block mb-0.5">Socratic Buddy</span>
                                Hi! Socratic Buddy here. A video containing {activeThreatType.replace('_', ' ')} was blocked. Let&apos;s talk about what was on the screen.
                              </div>
                            </div>

                            {chatHistory.map((item, index) => {
                              let textContent = item.content;
                              let emotion = "";
                              let agreed = false;
                              try {
                                const parsed = JSON.parse(item.content);
                                if (parsed && parsed.socratic_response_to_child) {
                                  textContent = parsed.socratic_response_to_child;
                                  emotion = parsed.child_emotion;
                                  agreed = parsed.agreed_to_boundary;
                                }
                              } catch {}

                              return (
                                <div key={index} className={`flex items-start gap-2 ${item.role === "user" ? "flex-row-reverse" : ""}`}>
                                  <div className={`w-7 h-7 rounded flex items-center justify-center font-bold text-[9px] shrink-0 ${item.role === "user" ? "bg-[#E6D5C3] text-[#2D3025] border border-[#5A5A40]" : "bg-[#5A5A40] text-white"}`}>
                                    {item.role === "user" ? "ME" : "SB"}
                                  </div>
                                  <div className={`border rounded-lg p-2 max-w-[85%] text-[10px] leading-relaxed ${item.role === "user" ? "bg-[#E6D5C3]/20 border-[#DDE0D0] text-[#2D3025]" : "bg-white border-[#DDE0D0] text-[#2D3025]"}`}>
                                    <span className="font-bold block mb-0.5 text-[#5A5A40]">{item.role === "user" ? "My Response" : "Socratic Buddy"}</span>
                                    {textContent}
                                    {emotion && (
                                      <div className="mt-1 pt-1 border-t border-[#DDE0D0] flex items-center gap-1 text-[8px] text-[#6B705C]">
                                        <span>Emotion: <strong className="capitalize">{emotion}</strong></span>
                                        {agreed && <span className="text-emerald-700 font-bold ml-1">✓ Boundary Agreed</span>}
                                      </div>
                                    )}
                                  </div>
                                </div>
                              );
                            })}

                            {loadingTurn && (
                              <div className="flex items-center gap-2">
                                <div className="w-7 h-7 rounded bg-[#5A5A40] flex items-center justify-center font-bold text-white text-[9px] shrink-0">SB</div>
                                <div className="bg-white border border-[#DDE0D0] rounded-lg p-2 max-w-[85%] text-[#6B705C] text-[10px] italic flex items-center gap-1.5">
                                  <RefreshCw className="w-3 animate-spin text-[#5A5A40]" /> Thinking...
                                </div>
                              </div>
                            )}
                          </div>

                          {/* Completion Screen Overlay inside Chat */}
                          {isCompleted && (
                            <div className="absolute inset-0 bg-white/95 flex flex-col items-center justify-center p-3 text-center z-20">
                              <CheckCircle className="w-10 h-10 text-emerald-600 mb-1" />
                              <h3 className="text-xs font-bold text-[#2D3025] font-sans">Pedagogical Boundary Confirmed</h3>
                              <p className="text-[10px] text-[#6B705C] mt-1 max-w-[200px] leading-relaxed text-center">
                                Socratic dialogue completed. We have pivoted away from the flagged video.
                              </p>
                              <button
                                onClick={async () => {
                                  await resetGuardSession();
                                  setChildActiveApp("game");
                                  setActiveTab("sandbox");
                                  setLastInterceptionStatus("Successfully scaffolded child. Redirected to Sandbox.");
                                }}
                                className="mt-3 py-1.5 px-3 bg-[#5A5A40] hover:bg-[#454530] text-white font-bold rounded text-[10px] transition duration-200"
                              >
                                Go to Sandbox Game
                              </button>
                            </div>
                          )}

                          {/* Chat Input form */}
                          {!isCompleted && (
                            <form onSubmit={handleSendDialogue} className="bg-white border-t border-[#DDE0D0] p-1.5 flex items-center gap-1.5 shrink-0">
                              <input 
                                type="text" 
                                placeholder="Type response..." 
                                value={userInput}
                                onChange={(e) => setUserInput(e.target.value)}
                                disabled={loadingTurn}
                                className="flex-1 border border-[#DDE0D0] rounded px-2 py-1 text-[11px] focus:outline-none focus:border-[#5A5A40] disabled:opacity-50"
                              />
                              <button
                                type="submit"
                                disabled={loadingTurn || !userInput.trim()}
                                className="px-3 py-1 bg-[#5A5A40] hover:bg-[#454530] text-white font-bold rounded text-[11px] transition"
                              >
                                Send
                              </button>
                            </form>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Upload & Preset controls */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 border-t border-[#DDE0D0] pt-4">
                  {/* Upload */}
                  <div className="flex flex-col gap-1.5">
                    <span className="text-[10px] uppercase font-bold text-[#6B705C] font-sans">Upload Video File</span>
                    <label className="flex items-center justify-center gap-1.5 px-3 py-2 border border-[#DDE0D0] hover:border-[#5A5A40] rounded-lg bg-[#FAF9F6] text-xs font-semibold cursor-pointer text-[#6B705C] hover:text-[#2D3025] transition shadow-2xs">
                      📁 Choose Video File...
                      <input 
                        type="file" 
                        accept="video/*" 
                        onChange={handleVideoUpload} 
                        className="hidden" 
                      />
                    </label>
                    {videoFilename && (
                      <div className="text-[10px] text-[#5A5A40] font-mono break-all leading-normal bg-[#E6D5C3]/15 border border-[#DDE0D0] p-1.5 rounded">
                        Playing: {videoFilename}
                      </div>
                    )}
                  </div>

                  {/* Presets */}
                  <div className="flex flex-col gap-1.5">
                    <span className="text-[10px] uppercase font-bold text-[#6B705C] font-sans">Pre-loaded Test Clips</span>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        onClick={() => selectPresetVideo("nature")}
                        className={`text-xs py-2 px-1.5 border rounded-md font-semibold transition ${
                          videoFilename === "nature_wildlife_doc.mp4" 
                            ? "bg-emerald-600 text-white border-emerald-600 shadow-2xs" 
                            : "bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border-emerald-200"
                        }`}
                      >
                        🟢 Nature Video
                      </button>
                      <button
                        onClick={() => selectPresetVideo("combat")}
                        className={`text-xs py-2 px-1.5 border rounded-md font-semibold transition ${
                          videoFilename === "combat_match_battle_violence.mp4"
                            ? "bg-rose-600 text-white border-rose-600 shadow-2xs"
                            : "bg-rose-50 hover:bg-rose-100 text-rose-800 border-rose-200"
                        }`}
                      >
                        💥 Combat Video
                      </button>
                    </div>
                  </div>
                </div>

              </div>
            </div>

            {/* Right Column: 6 Cols - Parent AI Visual Stream */}
            <div className="xl:col-span-6 flex flex-col gap-6">
              
              {/* Real-time AI Visual Feed Monitor */}
              <div className="bg-white border border-[#DDE0D0] rounded-xl p-5 shadow-sm flex flex-col gap-4">
                <div className="flex items-center justify-between border-b border-[#DDE0D0] pb-2">
                  <h3 className="font-semibold text-[#2D3025] flex items-center gap-2 text-sm uppercase tracking-wider font-sans">
                    <Shield className="w-4 h-4 text-[#5A5A40]" />
                    AI Visual Scan Feed
                  </h3>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-bold text-[#6B705C] uppercase tracking-wider font-sans">
                      {guardActive ? "🟢 Active Monitor" : "🔴 Paused"}
                    </span>
                    <button
                      onClick={() => toggleMonitorActive(!guardActive)}
                      className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                        guardActive ? "bg-[#5A5A40]" : "bg-slate-300"
                      }`}
                      id="btn-toggle-monitor-active"
                    >
                      <span
                        className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow-sm ring-0 transition duration-200 ease-in-out ${
                          guardActive ? "translate-x-4" : "translate-x-0"
                        }`}
                      />
                    </button>
                  </div>
                </div>

                <div className="relative aspect-video w-full rounded-lg border border-[#DDE0D0] overflow-hidden bg-slate-950 flex items-center justify-center shadow-inner">
                  {guardState?.last_frame ? (
                    <img 
                      src={guardState.last_frame} 
                      alt="AI Visual Bounding Boxes" 
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="text-slate-500 text-xs flex flex-col items-center gap-2">
                      <RefreshCw className="w-6 h-6 animate-spin text-[#5A5A40]" />
                      Waiting for active video frames...
                    </div>
                  )}
                  {analyzingVideo && (
                    <span className="absolute top-2 right-2 bg-emerald-500 text-white font-mono text-[9px] px-1.5 py-0.5 rounded animate-pulse shadow-sm z-10">
                      AI SCANNING...
                    </span>
                  )}
                </div>

                {/* Threat score progress bar */}
                <div className="p-4 bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg flex flex-col gap-3">
                  <div className="flex items-center justify-between text-xs font-semibold">
                    <span className="text-[#2D3025] font-sans">Safety Evaluation Threat Score</span>
                    <span className={guardState?.threat_score > 0.85 ? "text-rose-600 font-bold" : "text-[#5A5A40]"}>
                      {guardState ? (guardState.threat_score * 100).toFixed(0) : 0}% / 85% Gate
                    </span>
                  </div>
                  <div className="w-full bg-[#E1E2D9] h-3 rounded-full overflow-hidden">
                    <div 
                      className={`h-full transition-all duration-500 ${
                        guardState?.threat_score > 0.85 ? "bg-rose-500" : "bg-[#5A5A40]"
                      }`}
                      style={{ width: `${guardState ? guardState.threat_score * 100 : 0}%` }}
                    />
                  </div>
                  <div className="flex justify-between items-center text-[10px] text-[#6B705C] font-sans">
                    <span>Safety Gate limit: 85%</span>
                    <span className={guardState?.threat_score > 0.85 ? "text-rose-600 font-bold" : "text-emerald-700 font-bold"}>
                      {guardState?.threat_score > 0.85 ? "🚨 INTERCEPT LOCKED" : "🛡️ ACTIVE MONITORING"}
                    </span>
                  </div>
                  
                  <button 
                    onClick={async () => {
                      await resetGuardSession();
                      setLastInterceptionStatus("Ready & Guarding System");
                    }}
                    className="mt-2 py-2 px-3 bg-white hover:bg-slate-50 border border-[#DDE0D0] text-[#5A5A40] hover:text-[#2D3025] text-xs font-semibold rounded-lg transition flex items-center justify-center gap-1.5 shadow-2xs"
                  >
                    <RefreshCw className="w-3.5 h-3.5" /> Reset Detection State
                  </button>
                </div>
              </div>

            </div>

            {/* Real-time Telemetry Dashboard Card */}
            <div className="xl:col-span-6 flex flex-col gap-6">
              <div className="bg-white border border-[#DDE0D0] rounded-xl p-5 shadow-sm flex flex-col gap-4">
                <h3 className="font-semibold text-[#2D3025] flex items-center gap-2 text-sm uppercase tracking-wider font-sans border-b border-[#DDE0D0] pb-2">
                  <Activity className="w-4 h-4 text-[#5A5A40]" />
                  Zero-Trust Telemetry Dashboard
                </h3>
                
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
                  <div className="bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg p-2.5 flex flex-col gap-1">
                    <span className="text-[10px] text-[#6B705C] uppercase font-semibold">Active File</span>
                    <span className="font-bold text-[#2D3025] truncate" title={guardState?.video_name || "None"}>
                      {guardState?.video_name || "No file playing"}
                    </span>
                  </div>
                  
                  <div className="bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg p-2.5 flex flex-col gap-1">
                    <span className="text-[10px] text-[#6B705C] uppercase font-semibold">RAM Usage</span>
                    <span className="font-bold text-[#2D3025] flex items-center gap-1">
                      <Cpu className="w-3.5 h-3.5 text-[#5A5A40]" />
                      {guardState?.ram_usage || 0} MB
                    </span>
                  </div>

                  <div className="bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg p-2.5 flex flex-col gap-1">
                    <span className="text-[10px] text-[#6B705C] uppercase font-semibold">Pipeline Latency</span>
                    <span className="font-bold text-[#2D3025] flex items-center gap-1">
                      <Clock className="w-3.5 h-3.5 text-[#5A5A40]" />
                      {guardState?.fps_latency || 0} ms
                    </span>
                  </div>

                  <div className="bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg p-2.5 flex flex-col gap-1">
                    <span className="text-[10px] text-[#6B705C] uppercase font-semibold">Start Time</span>
                    <span className="font-bold text-[#2D3025]">
                      {guardState?.start_time || "Not Started"}
                    </span>
                  </div>

                  <div className="bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg p-2.5 flex flex-col gap-1">
                    <span className="text-[10px] text-[#6B705C] uppercase font-semibold">Elapsed Time</span>
                    <span className="font-bold text-[#2D3025]">
                      {guardState?.elapsed_time || 0} seconds
                    </span>
                  </div>

                  <div className="bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg p-2.5 flex flex-col gap-1">
                    <span className="text-[10px] text-[#6B705C] uppercase font-semibold">Overall Status</span>
                    <span className={`font-bold ${interceptActive ? "text-rose-600 animate-pulse animate-duration-1000" : "text-emerald-700"}`}>
                      {interceptActive ? "🚨 Threat Alert" : "🛡️ Guarding"}
                    </span>
                  </div>
                </div>

                {/* Model-specific score metrics bars */}
                <div className="mt-2 bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg p-3 flex flex-col gap-3">
                  <span className="text-[10px] text-[#6B705C] uppercase font-bold tracking-wider font-sans">ONNX Ensemble Model Scores</span>
                  
                  {/* NSFW Score */}
                  <div className="flex flex-col gap-1">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="font-medium text-[#2D3025]">nsfw-classifier-ONNX (Adult Content)</span>
                      <span className={guardState?.nsfw_score > 0.80 ? "text-rose-600 font-bold" : "text-[#5A5A40]"}>
                        {((guardState?.nsfw_score || 0) * 100).toFixed(0)}% / 80% Threshold
                      </span>
                    </div>
                    <div className="w-full bg-[#E1E2D9] h-2 rounded-full overflow-hidden">
                      <div 
                        className={`h-full transition-all duration-300 ${guardState?.nsfw_score > 0.80 ? "bg-rose-500" : "bg-emerald-600"}`}
                        style={{ width: `${Math.min(100, (guardState?.nsfw_score || 0) * 100)}%` }}
                      />
                    </div>
                  </div>

                  {/* MoViNet-A0 Score */}
                  <div className="flex flex-col gap-1">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="font-medium text-[#2D3025]">MoViNet-A0 (Physical Violence/Fighting)</span>
                      <span className={guardState?.violence_score > 0.80 ? "text-rose-600 font-bold" : "text-[#5A5A40]"}>
                        {((guardState?.violence_score || 0) * 100).toFixed(0)}% / 80% Threshold
                      </span>
                    </div>
                    <div className="w-full bg-[#E1E2D9] h-2 rounded-full overflow-hidden">
                      <div 
                        className={`h-full transition-all duration-300 ${guardState?.violence_score > 0.80 ? "bg-rose-500" : "bg-[#5A5A40]"}`}
                        style={{ width: `${Math.min(100, (guardState?.violence_score || 0) * 100)}%` }}
                      />
                    </div>
                  </div>

                  {/* YOLOv8-nano Score */}
                  <div className="flex flex-col gap-1">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="font-medium text-[#2D3025]">YOLOv8-nano (Weapons detection)</span>
                      <span className={guardState?.weapons_score > 0.75 ? "text-rose-600 font-bold" : "text-[#5A5A40]"}>
                        {((guardState?.weapons_score || 0) * 100).toFixed(0)}% / 75% Threshold
                      </span>
                    </div>
                    <div className="w-full bg-[#E1E2D9] h-2 rounded-full overflow-hidden">
                      <div 
                        className={`h-full transition-all duration-300 ${guardState?.weapons_score > 0.75 ? "bg-rose-500" : "bg-indigo-600"}`}
                        style={{ width: `${Math.min(100, (guardState?.weapons_score || 0) * 100)}%` }}
                      />
                    </div>
                  </div>
                </div>

              </div>
            </div>

          </section>
        )}

      </main>

      {/* Footer credits and information */}
      <footer className="border-t border-[#DDE0D0] bg-[#FAF9F6] p-6 text-center text-[#6B705C] text-xs mt-auto" id="app-footer">
        <div className="max-w-[1200px] mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="flex items-center gap-1.5 justify-center">
            Socratic Buddy is an offline child safety cognitive system protecting youth autonomy and digital wellbeing.
          </p>
          <div className="flex items-center gap-4 text-[11px]" id="footer-links">
            <span>License: MIT Offline Non-transit</span>
            <span>Security Status: 100% Local Guard</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
