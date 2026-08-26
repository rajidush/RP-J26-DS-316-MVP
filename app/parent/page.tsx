"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  Shield,
  User,
  Clock,
  AlertTriangle,
  TrendingUp,
  Sliders,
  Plus,
  Edit2,
  Trash2,
  Save,
  CheckCircle,
  ArrowLeft,
  ChevronRight,
  Monitor,
  Heart,
  Briefcase,
  HelpCircle,
  Calendar,
  Sparkles,
  Info,
  Check
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { BACKEND_URL, ANALYST_URL } from "../lib/backend";

interface ChildPersona {
  id: string;
  name: string;
  age: number;
  grade: string;
  personality: string[];
  interests: string[];
  devices: string;
  memberSince: string;
}

const DEFAULT_PERSONAS: ChildPersona[] = [
  {
    id: "sandeep",
    name: "Sandeep Perera",
    age: 10,
    grade: "Grade 5",
    personality: ["Curious", "Sensitive"],
    interests: ["Nature", "Science", "Space"],
    devices: "1 Desktop, 1 Tablet",
    memberSince: "01 Aug 2026",
  },
  {
    id: "senura",
    name: "Senura Perera",
    age: 14,
    grade: "Grade 9",
    personality: ["Independent", "Creative"],
    interests: ["Gaming", "Coding", "Music"],
    devices: "1 Laptop, 1 Smartphone",
    memberSince: "10 Aug 2026",
  }
];

export default function ParentDashboard() {
  const [activeSidebarTab, setActiveSidebarTab] = useState<"overview" | "profile">("overview");
  
  // Personas list state
  const [personas, setPersonas] = useState<ChildPersona[]>([]);
  const [activePersonaId, setActivePersonaId] = useState<string>("sandeep");
  const activePersona = personas.find(p => p.id === activePersonaId) || personas[0] || DEFAULT_PERSONAS[0];
  
  // Profile Editor state
  const [isEditing, setIsEditing] = useState<boolean>(false);
  const [editName, setEditName] = useState("");
  const [editAge, setEditAge] = useState(10);
  const [editGrade, setEditGrade] = useState("");
  const [editPersonality, setEditPersonality] = useState("");
  const [editInterests, setEditInterests] = useState("");
  const [editDevices, setEditDevices] = useState("");

  // New Persona creation state
  const [isCreatingNew, setIsCreatingNew] = useState(false);

  // Backend Integration State
  const [backendData, setBackendData] = useState<{
    analyst_runs: any[];
    socratic_sessions: any[];
    content_intercepted?: number;
    high_severity_alerts?: number;
    hate_speech_detected?: number;
    screen_time_minutes?: number;
    analyst_status?: { online?: boolean; capturing?: boolean; panel_url?: string };
    analyst_panel_url?: string;
    analyst_stats?: { total?: number; hate?: number };
  } | null>(null);
  const [backendAvailable, setBackendAvailable] = useState<boolean>(false);

  // Poll parent dashboard data from Python backend
  useEffect(() => {
    let active = true;
    const fetchData = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/parent/dashboard-data?child_age=${activePersona.age}`);
        if (!res.ok) throw new Error("Backend connection failed");
        const data = await res.json();
        if (active) {
          setBackendData(data);
          setBackendAvailable(true);
        }
      } catch (err) {
        if (active) {
          setBackendAvailable(false);
        }
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 3000); // poll every 3 seconds

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [activePersona.age]);

  // Initialize personas from localStorage or defaults
  useEffect(() => {
    const saved = localStorage.getItem("socratic_parent_personas");
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setPersonas(parsed);
        if (parsed.length > 0) {
          setActivePersonaId(parsed[0].id);
        }
      } catch (e) {
        setPersonas(DEFAULT_PERSONAS);
      }
    } else {
      setPersonas(DEFAULT_PERSONAS);
      localStorage.setItem("socratic_parent_personas", JSON.stringify(DEFAULT_PERSONAS));
    }
  }, []);
  // Set edit form values when editing starts or active persona changes
  useEffect(() => {
    if (activePersona) {
      setEditName(activePersona.name);
      setEditAge(activePersona.age);
      setEditGrade(activePersona.grade);
      setEditPersonality(activePersona.personality.join(", "));
      setEditInterests(activePersona.interests.join(", "));
      setEditDevices(activePersona.devices);
    }
  }, [activePersona, isEditing]);

  const savePersona = () => {
    const updated = personas.map(p => {
      if (p.id === activePersonaId) {
        return {
          ...p,
          name: editName,
          age: editAge,
          grade: editGrade,
          personality: editPersonality.split(",").map(s => s.trim()).filter(Boolean),
          interests: editInterests.split(",").map(s => s.trim()).filter(Boolean),
          devices: editDevices
        };
      }
      return p;
    });
    setPersonas(updated);
    localStorage.setItem("socratic_parent_personas", JSON.stringify(updated));
    setIsEditing(false);
  };

  const createNewPersona = (e: React.FormEvent) => {
    e.preventDefault();
    const newId = "persona-" + Date.now();
    const newPersona: ChildPersona = {
      id: newId,
      name: editName || "New Profile",
      age: editAge || 8,
      grade: editGrade || "Grade 3",
      personality: editPersonality ? editPersonality.split(",").map(s => s.trim()).filter(Boolean) : ["Energetic"],
      interests: editInterests ? editInterests.split(",").map(s => s.trim()).filter(Boolean) : ["Art", "Play"],
      devices: editDevices || "1 Tablet",
      memberSince: new Date().toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" }),
    };

    const updated = [...personas, newPersona];
    setPersonas(updated);
    localStorage.setItem("socratic_parent_personas", JSON.stringify(updated));
    setActivePersonaId(newId);
    setIsCreatingNew(false);
    setIsEditing(false);
    setActiveSidebarTab("profile");
  };

  const deletePersona = (id: string) => {
    if (personas.length <= 1) {
      alert("At least one child profile must remain.");
      return;
    }
    if (confirm(`Are you sure you want to delete ${personas.find(p => p.id === id)?.name}'s profile?`)) {
      const updated = personas.filter(p => p.id !== id);
      setPersonas(updated);
      localStorage.setItem("socratic_parent_personas", JSON.stringify(updated));
      setActivePersonaId(updated[0].id);
    }
  };

  // Mock/Real Overview Analytics Data
  const getMetrics = () => {
    const defaults = activePersona.age <= 10
      ? {
          intercepts: 0,
          interceptsTrend: "Polling...",
          alerts: 0,
          alertsTrend: "Polling...",
          hateSpeech: 0,
          hateSpeechTrend: "Polling...",
          screenTime: "0m",
          screenTimeTrend: "Polling...",
          anomalyScore: 0.28,
          threats: { violence: 0, hate: 0, adult: 0, controversial: 0, other: 0 }
        }
      : {
          intercepts: 0,
          interceptsTrend: "Polling...",
          alerts: 0,
          alertsTrend: "Polling...",
          hateSpeech: 0,
          hateSpeechTrend: "Polling...",
          screenTime: "0m",
          screenTimeTrend: "Polling...",
          anomalyScore: 0.54,
          threats: { violence: 0, hate: 0, adult: 0, controversial: 0, other: 0 }
        };

    if (backendAvailable && backendData) {
      const runsForChild = (backendData.analyst_runs || []).filter(r => r.child_age === activePersona.age);
      const sessionsForChild = (backendData.socratic_sessions || []).filter(s => s.child_age === activePersona.age);

      const threats = { violence: 0, hate: 0, adult: 0, controversial: 0, other: 0 };
      runsForChild.forEach(r => {
        if (r.decision === "hate") {
          const cat = r.category || "";
          if (cat.includes("violence")) threats.violence++;
          else if (cat.includes("hate")) threats.hate++;
          else if (cat.includes("adult")) threats.adult++;
          else if (cat.includes("controversial")) threats.controversial++;
          else threats.other++;
        }
      });
      sessionsForChild.forEach(s => {
        const tType = s.threat_type || "";
        if (tType.includes("violence")) threats.violence++;
        else if (tType.includes("hate")) threats.hate++;
        else if (tType.includes("adult")) threats.adult++;
        else if (tType.includes("controversial")) threats.controversial++;
        else threats.other++;
      });

      let anomalyScore = defaults.anomalyScore;
      const allEmotions: string[] = [];
      sessionsForChild.forEach(s => {
        (s.turns || []).forEach((t: any) => {
          if (t.child_emotion) allEmotions.push(t.child_emotion.toLowerCase());
        });
      });
      if (allEmotions.length > 0) {
        const negativeCount = allEmotions.filter(e => ["scared", "defensive", "frustrated", "angry"].includes(e)).length;
        anomalyScore = parseFloat((negativeCount / allEmotions.length).toFixed(2));
        anomalyScore = Math.max(0.1, Math.min(0.95, anomalyScore));
      }

      // Read real-time values from the backend response
      const realInterceptsCount = backendData.content_intercepted ?? 0;
      const realAlertsCount = backendData.high_severity_alerts ?? 0;
      const realHateSpeechCount = backendData.hate_speech_detected ?? 0;
      const realScreenTimeMinutes = backendData.screen_time_minutes ?? 0;

      return {
        intercepts: realInterceptsCount,
        interceptsTrend: "Real-time count",
        alerts: realAlertsCount,
        alertsTrend: "Real-time count",
        hateSpeech: realHateSpeechCount,
        hateSpeechTrend: "Real-time count",
        screenTime: `${realScreenTimeMinutes}m`,
        screenTimeTrend: "Real-time count",
        anomalyScore: anomalyScore,
        threats: threats
      };
    }

    return defaults;
  };

  // Dynamic Socratic Emotion Trendpoints
  const getEmotionTrendPoints = () => {
    if (!backendAvailable || !backendData) return null;
    const sessions = [...backendData.socratic_sessions]
      .filter(s => s.child_age === activePersona.age)
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

    // 1. Group all turns by day string (e.g. "18 Aug")
    const dayGroups: { [dateStr: string]: { positive: number; neutral: number; negative: number; dateObj: Date } } = {};

    sessions.forEach(session => {
      if (!session.timestamp) return;
      const date = new Date(session.timestamp);
      if (isNaN(date.getTime())) return;
      
      const day = date.getDate();
      const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      const dateStr = `${day} ${monthNames[date.getMonth()]}`;

      if (!dayGroups[dateStr]) {
        dayGroups[dateStr] = { positive: 0, neutral: 0, negative: 0, dateObj: date };
      }

      (session.turns || []).forEach((turn: any) => {
        const emo = (turn.child_emotion || "").toLowerCase();
        if (!emo || emo === "none") return;

        if (["resolved", "compliant", "happy", "reflective", "curious", "positive"].includes(emo)) {
          dayGroups[dateStr].positive++;
        } else if (["scared", "defensive", "frustrated", "angry", "negative"].includes(emo)) {
          dayGroups[dateStr].negative++;
        } else if (["neutral", "unknown", "unclear"].includes(emo)) {
          dayGroups[dateStr].neutral++;
        }
      });
    });

    const sortedDateStrs = Object.keys(dayGroups).sort((a, b) => {
      return dayGroups[a].dateObj.getTime() - dayGroups[b].dateObj.getTime();
    });

    // Limit to the last 7 days of historical data
    const last7Days = sortedDateStrs.slice(-7);

    if (last7Days.length === 0) return null;

    const emotionToY = (emo: string): number => {
      switch (emo) {
        case "positive":
          return 50; // maps to Positive line
        case "negative":
          return 110; // maps to Negative line
        case "neutral":
        default:
          return 80; // maps to Neutral line
      }
    };

    const emotionToColor = (emo: string): string => {
      switch (emo) {
        case "positive":
          return "#10B981";
        case "negative":
          return "#F43F5E";
        case "neutral":
        default:
          return "#F59E0B";
      }
    };

    const points = last7Days.map((dateStr, idx) => {
      const group = dayGroups[dateStr];
      let dominant: "positive" | "neutral" | "negative" = "neutral";
      
      if (group.positive > group.neutral && group.positive > group.negative) {
        dominant = "positive";
      } else if (group.negative > group.positive && group.negative > group.neutral) {
        dominant = "negative";
      } else if (group.positive === group.negative && group.positive > 0) {
        dominant = "neutral";
      }

      const x = last7Days.length > 1
        ? 10 + idx * (280 / (last7Days.length - 1))
        : 150;
      const y = emotionToY(dominant);

      return {
        x,
        y,
        color: emotionToColor(dominant),
        emotion: dominant.charAt(0).toUpperCase() + dominant.slice(1),
        timeStr: dateStr
      };
    });

    let pathD = "";
    if (points.length > 0) {
      pathD = `M ${points[0].x} ${points[0].y}`;
      for (let i = 1; i < points.length; i++) {
        pathD += ` L ${points[i].x} ${points[i].y}`;
      }
    }

    return { points, pathD };
  };

  const emotionTrend = getEmotionTrendPoints();

  // Dynamic Recent Alerts
  const getRecentAlerts = () => {
    const list: any[] = [];
    if (backendAvailable && backendData) {
      const runs = (backendData.analyst_runs || [])
        .filter(r => r.child_age === activePersona.age && r.decision === "hate");
      runs.forEach(run => {
        const src = run.source || {};
        const sourceName = src.ocr ? "OCR Search" : src.overlay ? "Overlay Text" : src.asr ? "Audio Transcript" : "Vision Screen Grab";
        const cat = String(run.category || "none").replace(/_/g, " ");
        const risk = Number(run.risk_score ?? 0);
        list.push({
          id: run.id,
          timestamp: new Date(run.timestamp),
          timeStr: run.time_str,
          title: `Hate Speech Detected (${cat})`,
          app: `Analyst: ${sourceName}${run.app_exe ? ` · ${run.app_exe}` : ""}`,
          severity: risk > 0.9 ? "High" : "Medium",
          severityColor: "bg-rose-50 text-rose-700 border border-rose-100",
          iconBg: "bg-rose-50 text-rose-600"
        });
      });

      const sessions = (backendData.socratic_sessions || [])
        .filter(s => s.child_age === activePersona.age);
      sessions.forEach(sess => {
        list.push({
          id: `sess-${sess.session_id}`,
          timestamp: new Date(sess.timestamp),
          timeStr: sess.time_str,
          title: `Socratic Intercept Activated`,
          app: `Perception: ${sess.threat_type.replace('_', ' ')}`,
          severity: "High",
          severityColor: "bg-orange-50 text-orange-700 border border-orange-100",
          iconBg: "bg-orange-50 text-orange-600"
        });
      });
    }

    list.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
    return list.length > 0 ? list.slice(0, 5) : null;
  };

  const recentAlerts = getRecentAlerts();

  // Dynamic Explainable AI
  const getExplainableAI = () => {
    const list: any[] = [];
    if (backendAvailable && backendData) {
      const runs = (backendData.analyst_runs || [])
        .filter(r => r.child_age === activePersona.age && r.decision === "hate");
      runs.forEach(run => {
        const textCtx = run.overlay_text || run.ocr_text || run.transcript;
        const truncatedText = textCtx ? (textCtx.length > 80 ? textCtx.slice(0, 80) + "..." : textCtx) : "N/A";
        const risk = Number(run.risk_score ?? 0);
        list.push({
          id: run.id,
          timestamp: new Date(run.timestamp),
          title: "Hate Speech Flagged",
          details: `Analyst detected category "${String(run.category || "none")}" (risk score ${risk.toFixed(2)}). Content flagged: "${truncatedText}"`,
          impact: "Negative",
          impactColor: "bg-rose-50 text-rose-800 border border-rose-100",
          iconBg: "bg-rose-50 text-rose-600"
        });
      });

      const sessions = (backendData.socratic_sessions || [])
        .filter(s => s.child_age === activePersona.age);
      sessions.forEach(sess => {
        const latestTurn = sess.turns && sess.turns.length > 0 ? sess.turns[sess.turns.length - 1] : null;
        const currentEmotion = latestTurn ? latestTurn.child_emotion : "neutral";
        const currentPhase = latestTurn ? latestTurn.current_phase : "Acknowledge";

        let impact = "Neutral";
        let impactColor = "bg-orange-50 text-orange-800 border border-orange-100";
        const emoLower = currentEmotion.toLowerCase();
        if (["resolved", "compliant", "happy"].includes(emoLower)) {
          impact = "Positive";
          impactColor = "bg-emerald-50 text-emerald-800 border border-emerald-100";
        } else if (["reflective", "curious"].includes(emoLower)) {
          impact = "Favorable";
          impactColor = "bg-emerald-50 text-emerald-800 border border-emerald-100";
        } else if (["scared", "defensive"].includes(emoLower)) {
          impact = "Negative";
          impactColor = "bg-rose-50 text-rose-800 border border-rose-100";
        } else if (["frustrated", "angry"].includes(emoLower)) {
          impact = "Very Negative";
          impactColor = "bg-rose-50 text-rose-800 border border-rose-100";
        }

        list.push({
          id: `xai-${sess.session_id}`,
          timestamp: new Date(sess.timestamp),
          title: "Socratic Safety Intercept",
          details: `Socratic agent intercepted a "${sess.threat_type}" threat. Session status: "${sess.completed ? 'Completed' : 'Active Dialog'}" in phase "${currentPhase}". Child response emotional index: "${currentEmotion}".`,
          impact: impact,
          impactColor: impactColor,
          iconBg: "bg-orange-50 text-orange-600"
        });
      });
    }

    list.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
    return list.length > 0 ? list.slice(0, 3) : null;
  };

  const explainableAI = getExplainableAI();

  // Dynamic Suggestion Banner Recommendations
  const getAIRecommendation = () => {
    let base = activePersona.age <= 10
      ? "Limit violent video content exposure to less than 15 mins daily. Socratic dialogue suggests Sandeep is highly receptive to boundaries when discussed in a supportive tone. Encourage early bedroom screens curfew at 9:30 PM."
      : "Discuss social circle dynamic on Discord. Young adult exhibits stress response (Elevated Anomaly Index) following late night gaming. Pivot safety agreements collaboratively rather than executing strict lockouts.";

    if (backendAvailable && backendData) {
      const sessions = (backendData.socratic_sessions || [])
        .filter(s => s.child_age === activePersona.age);
      const allEmotions: string[] = [];
      sessions.forEach(s => {
        (s.turns || []).forEach((t: any) => {
          if (t.child_emotion) allEmotions.push(t.child_emotion.toLowerCase());
        });
      });

      if (allEmotions.includes("frustrated") || allEmotions.includes("defensive")) {
        base += " Note: Recent Socratic interactions show signs of frustration or defensive behavior. It is recommended to approach boundaries gently without direct confrontation.";
      } else if (allEmotions.includes("resolved") || allEmotions.includes("compliant")) {
        base += " Note: Dialogue history shows positive boundary compliance and resolution. Cognitive scaffolding is working effectively.";
      }
    }
    return base;
  };

  const metrics = getMetrics();
  const totalThreats = Object.values(metrics.threats || {}).reduce((a, b) => a + b, 0);

  // Custom SVG donut chart segments calculation
  const getDonutSegments = () => {
    let currentOffset = 0;
    const threats = metrics.threats || { violence: 0, hate: 0, adult: 0, controversial: 0, other: 0 };
    const categories = [
      { name: "High Violence", count: Number(threats.violence || 0), color: "#F43F5E" },
      { name: "Hate Speech", count: Number(threats.hate || 0), color: "#F59E0B" },
      { name: "Adult Content", count: Number(threats.adult || 0), color: "#A855F7" },
      { name: "Controversial", count: Number(threats.controversial || 0), color: "#10B981" },
      { name: "Other", count: Number(threats.other || 0), color: "#3B82F6" },
    ];
    
    const sumThreats = categories.reduce((sum, cat) => sum + cat.count, 0);
    
    return categories.map(cat => {
      const percentage = sumThreats > 0 ? (cat.count / sumThreats) * 100 : 0;
      const strokeLength = sumThreats > 0 ? (percentage / 100) * 314.16 : 0;
      const strokeOffset = 314.16 - strokeLength + currentOffset;
      currentOffset -= strokeLength;
      return {
        ...cat,
        percentage: sumThreats > 0 ? Math.round(percentage) : 0,
        strokeLength,
        strokeOffset: isNaN(strokeOffset) || !isFinite(strokeOffset) ? 314.16 : strokeOffset
      };
    });
  };

  const donutSegments = getDonutSegments();

  return (
    <div className="min-h-screen bg-[#F3F4ED] text-[#2D3025] flex flex-col font-sans antialiased" id="parent-dashboard-root">
      
      {/* Top Header */}
      <header className="border-b border-[#DDE0D0] bg-white px-6 py-4 flex items-center justify-between shadow-sm shrink-0" id="parent-header">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-[#FAF9F6] text-[#5A5A40] rounded-lg border border-[#DDE0D0]">
            <Shield className="w-6 h-6 text-[#5A5A40]" />
          </div>
          <div>
            <h1 className="text-xl font-serif-natural font-semibold tracking-tight text-[#2D3025] flex items-center gap-2">
              Socratic Digital Guard <span className="text-xs bg-[#5A5A40] text-white px-2 py-0.5 rounded font-sans">Parental Control</span>
            </h1>
            <p className="text-xs text-[#6B705C]">System Behavior Analytics &amp; Child Safety Dashboard</p>
          </div>
        </div>

        {/* Global Controls / Avatar */}
        <div className="flex items-center gap-4">
          <a
            href={backendData?.analyst_panel_url || ANALYST_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg border border-[#DDE0D0] bg-[#FAF9F6] text-xs font-medium text-[#5A5A40] hover:bg-white transition"
            title="Open C2 Analyst — screen scan & whitebox"
          >
            <span className={`w-2 h-2 rounded-full ${backendData?.analyst_status?.online ? "bg-emerald-500" : "bg-amber-400"}`} />
            Analyst Panel
          </a>
          <div className="text-right hidden md:block">
            <p className="text-xs font-semibold text-[#2D3025]">Nuwan Perera</p>
            <p className="text-[10px] text-[#6B705C]">Family Administrator</p>
          </div>
          <div className="w-9 h-9 rounded-full bg-[#E6D5C3] border border-[#5A5A40] flex items-center justify-center font-bold text-xs">
            NP
          </div>
        </div>
      </header>

      <div className="flex-1 flex flex-col lg:flex-row" id="parent-layout">
        
        {/* Left Sidebar */}
        <aside className="w-full lg:w-64 bg-white border-b lg:border-b-0 lg:border-r border-[#DDE0D0] p-5 flex flex-col justify-between shrink-0" id="parent-sidebar">
          <div className="flex flex-col gap-6">
            
            {/* Quick Profile Switcher */}
            <div>
              <label className="text-[10px] uppercase tracking-wider font-semibold text-[#6B705C] block mb-2">
                Monitoring Target
              </label>
              <div className="flex flex-col gap-1.5">
                {personas.map(p => (
                  <button
                    key={p.id}
                    onClick={() => {
                      setActivePersonaId(p.id);
                      setIsEditing(false);
                      setIsCreatingNew(false);
                    }}
                    className={`w-full p-2.5 rounded-lg border text-left text-xs transition flex items-center justify-between ${
                      p.id === activePersonaId
                        ? "bg-[#FAF9F6] border-[#5A5A40] text-[#2D3025] font-semibold"
                        : "bg-white border-transparent text-[#6B705C] hover:bg-[#FAF9F6]"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <div className="w-6 h-6 rounded-full bg-[#5A5A40]/10 text-[#5A5A40] flex items-center justify-center font-bold text-[10px]">
                        {p.name.charAt(0)}
                      </div>
                      <span className="truncate">{p.name}</span>
                    </div>
                    {p.id === activePersonaId && (
                      <span className="w-1.5 h-1.5 rounded-full bg-[#5A5A40]" />
                    )}
                  </button>
                ))}
                
                <button
                  onClick={() => {
                    setEditName("");
                    setEditAge(8);
                    setEditGrade("");
                    setEditPersonality("");
                    setEditInterests("");
                    setEditDevices("");
                    setIsCreatingNew(true);
                    setIsEditing(true);
                    setActiveSidebarTab("profile");
                  }}
                  className="w-full p-2.5 rounded-lg border border-dashed border-[#DDE0D0] text-[#5A5A40] hover:bg-[#FAF9F6] transition text-xs font-medium flex items-center justify-center gap-1.5"
                >
                  <Plus className="w-3.5 h-3.5" />
                  Add Child Profile
                </button>
              </div>
            </div>

            {/* Main Tabs */}
            <nav className="flex flex-col gap-1">
              <button
                onClick={() => {
                  setActiveSidebarTab("overview");
                  setIsCreatingNew(false);
                }}
                className={`flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-xs font-semibold transition ${
                  activeSidebarTab === "overview" && !isCreatingNew
                    ? "bg-[#5A5A40] text-white"
                    : "text-[#6B705C] hover:bg-[#FAF9F6] hover:text-[#2D3025]"
                }`}
              >
                <TrendingUp className="w-4 h-4" />
                Overview
              </button>
              <a
                href={backendData?.analyst_panel_url || ANALYST_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-xs font-semibold transition text-[#6B705C] hover:bg-[#FAF9F6] hover:text-[#2D3025]"
              >
                <Monitor className="w-4 h-4" />
                Hate Analyst
                <span className={`ml-auto w-2 h-2 rounded-full ${backendData?.analyst_status?.online ? "bg-emerald-500" : "bg-amber-400"}`} />
              </a>
              <button
                onClick={() => {
                  setActiveSidebarTab("profile");
                  setIsCreatingNew(false);
                }}
                className={`flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-xs font-semibold transition ${
                  activeSidebarTab === "profile" && !isCreatingNew
                    ? "bg-[#5A5A40] text-white"
                    : "text-[#6B705C] hover:bg-[#FAF9F6] hover:text-[#2D3025]"
                }`}
              >
                <User className="w-4 h-4" />
                Child Profile &amp; Persona
              </button>
            </nav>
          </div>

          {/* Bottom Connection Status & Back Button */}
          <div className="flex flex-col gap-3 mt-6 pt-5 border-t border-[#DDE0D0]">
            <div className={backendAvailable ? "p-3 bg-emerald-50 rounded-lg border border-emerald-200 flex items-center gap-2 text-emerald-800" : "p-3 bg-amber-50 rounded-lg border border-amber-200 flex items-center gap-2 text-amber-800"}>
              <div className={backendAvailable ? "w-2 h-2 rounded-full bg-emerald-500 animate-pulse" : "w-2 h-2 rounded-full bg-amber-500 animate-pulse"} />
              <div className="text-[10px] leading-tight">
                <p className="font-bold">{backendAvailable ? "System Connected" : "System Offline"}</p>
                <p className="opacity-80">{backendAvailable ? "100% Offline Mode" : "Running on Fallback Demo Data"}</p>
              </div>
            </div>

            <a
              href={backendData?.analyst_panel_url || ANALYST_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-1.5 py-2.5 px-4 bg-white hover:bg-[#FAF9F6] border border-[#DDE0D0] text-[#5A5A40] text-xs font-medium rounded-lg transition"
            >
              <Monitor className="w-3.5 h-3.5" />
              Open Analyst Panel
            </a>

            <Link
              href="/"
              className="flex items-center justify-center gap-1.5 py-2.5 px-4 bg-[#FAF9F6] hover:bg-[#E6D5C3]/40 border border-[#DDE0D0] text-[#5A5A40] text-xs font-medium rounded-lg transition"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              Back to Simulator
            </Link>
          </div>
        </aside>

        {/* Main Panel */}
        <main className="flex-1 p-6 overflow-y-auto max-w-[1440px] w-full mx-auto" id="parent-main-content">
          
          <AnimatePresence mode="wait">
            
            {/* Overview Dashboard Tab */}
            {activeSidebarTab === "overview" && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                transition={{ duration: 0.15 }}
                className="flex flex-col gap-6"
                key="overview-tab"
              >
                {/* Row 1: Profile Summary (Left) & Metrics (Right) */}
                <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
                  
                  {/* Child Profile Summary Box */}
                  <div className="xl:col-span-4 bg-white border border-[#DDE0D0] rounded-xl p-5 shadow-sm flex flex-col justify-between">
                    <div>
                      <div className="flex items-center justify-between border-b border-[#DDE0D0] pb-3 mb-4">
                        <h3 className="font-semibold text-sm uppercase tracking-wider text-[#2D3025]">Child Profile</h3>
                        <span className="text-[10px] bg-[#E6D5C3]/50 border border-[#DDE0D0] px-2 py-0.5 rounded text-[#5A5A40] font-bold font-mono">
                          Active Target
                        </span>
                      </div>
                      
                      <div className="flex items-center gap-4 mb-4">
                        <div className="w-14 h-14 rounded-full bg-[#5A5A40]/10 border border-[#5A5A40]/25 flex items-center justify-center text-xl font-bold text-[#5A5A40]">
                          {activePersona.name.charAt(0)}
                        </div>
                        <div>
                          <h4 className="font-bold text-[#2D3025] flex items-center gap-1.5">
                            {activePersona.name}
                          </h4>
                          <span className="text-xs bg-[#5A5A40]/10 text-[#5A5A40] px-2 py-0.5 rounded font-semibold inline-block mt-0.5">
                            {activePersona.age} Years Old
                          </span>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 gap-2.5 text-xs text-[#2D3025] border-t border-[#DDE0D0] pt-4">
                        <div className="flex justify-between py-1 border-b border-[#DDE0D0]/50">
                          <span className="text-[#6B705C]">Grade</span>
                          <span className="font-semibold">{activePersona.grade}</span>
                        </div>
                        <div className="flex justify-between py-1 border-b border-[#DDE0D0]/50">
                          <span className="text-[#6B705C]">Personality</span>
                          <span className="font-semibold">{activePersona.personality.join(", ")}</span>
                        </div>
                        <div className="flex justify-between py-1 border-b border-[#DDE0D0]/50">
                          <span className="text-[#6B705C]">Interests</span>
                          <span className="font-semibold truncate max-w-[200px] text-right">{activePersona.interests.join(", ")}</span>
                        </div>
                        <div className="flex justify-between py-1 border-b border-[#DDE0D0]/50">
                          <span className="text-[#6B705C]">Devices</span>
                          <span className="font-semibold">{activePersona.devices}</span>
                        </div>
                        <div className="flex justify-between py-1">
                          <span className="text-[#6B705C]">Member Since</span>
                          <span className="font-semibold">{activePersona.memberSince}</span>
                        </div>
                      </div>
                    </div>

                    <button
                      onClick={() => {
                        setActiveSidebarTab("profile");
                        setIsEditing(true);
                      }}
                      className="w-full mt-5 py-2.5 px-4 bg-[#FAF9F6] hover:bg-[#E6D5C3]/40 border border-[#DDE0D0] text-[#5A5A40] font-semibold text-xs rounded-lg transition flex items-center justify-center gap-1.5"
                    >
                      <Edit2 className="w-3.5 h-3.5" />
                      Edit Profile / Persona
                    </button>
                  </div>

                  {/* Overview Stats Metrics Grid */}
                  <div className="xl:col-span-8 grid grid-cols-1 sm:grid-cols-2 gap-4">
                    
                    {/* Content Intercepted */}
                    <div className="bg-white border border-[#DDE0D0] rounded-xl p-5 shadow-sm flex flex-col justify-between">
                      <div className="flex items-center justify-between text-[#6B705C]">
                        <span className="text-xs font-semibold uppercase tracking-wider">Content Intercepted</span>
                        <div className="p-1.5 bg-emerald-50 rounded-lg text-emerald-600">
                          <Shield className="w-4 h-4" />
                        </div>
                      </div>
                      <div className="my-3">
                        <span className="text-3xl font-bold font-serif-natural text-[#2D3025]">{metrics.intercepts}</span>
                        <span className="text-[10px] text-[#6B705C] block mt-0.5">Today</span>
                      </div>
                      <span className="text-xs text-emerald-600 font-semibold flex items-center gap-1">
                        {metrics.interceptsTrend}
                      </span>
                    </div>

                    {/* High Severity Alerts */}
                    <div className="bg-white border border-[#DDE0D0] rounded-xl p-5 shadow-sm flex flex-col justify-between">
                      <div className="flex items-center justify-between text-[#6B705C]">
                        <span className="text-xs font-semibold uppercase tracking-wider">High Severity Alerts</span>
                        <div className="p-1.5 bg-rose-50 rounded-lg text-rose-600">
                          <AlertTriangle className="w-4 h-4" />
                        </div>
                      </div>
                      <div className="my-3">
                        <span className="text-3xl font-bold font-serif-natural text-[#2D3025]">{metrics.alerts}</span>
                        <span className="text-[10px] text-[#6B705C] block mt-0.5">Today</span>
                      </div>
                      <span className="text-xs text-rose-600 font-semibold flex items-center gap-1">
                        {metrics.alertsTrend}
                      </span>
                    </div>

                    {/* Hate Speech */}
                    <div className="bg-white border border-[#DDE0D0] rounded-xl p-5 shadow-sm flex flex-col justify-between">
                      <div className="flex items-center justify-between text-[#6B705C]">
                        <span className="text-xs font-semibold uppercase tracking-wider">Hate Speech Detected</span>
                        <div className="p-1.5 bg-orange-50 rounded-lg text-orange-600">
                          <HelpCircle className="w-4 h-4" />
                        </div>
                      </div>
                      <div className="my-3">
                        <span className="text-3xl font-bold font-serif-natural text-[#2D3025]">{metrics.hateSpeech}</span>
                        <span className="text-[10px] text-[#6B705C] block mt-0.5">Today</span>
                      </div>
                      <span className="text-xs text-orange-600 font-semibold flex items-center gap-1">
                        {metrics.hateSpeechTrend}
                      </span>
                    </div>

                    {/* Screen Time */}
                    <div className="bg-white border border-[#DDE0D0] rounded-xl p-5 shadow-sm flex flex-col justify-between">
                      <div className="flex items-center justify-between text-[#6B705C]">
                        <span className="text-xs font-semibold uppercase tracking-wider">Screen Time</span>
                        <div className="p-1.5 bg-purple-50 rounded-lg text-purple-600">
                          <Clock className="w-4 h-4" />
                        </div>
                      </div>
                      <div className="my-3">
                        <span className="text-3xl font-bold font-serif-natural text-[#2D3025]">{metrics.screenTime}</span>
                        <span className="text-[10px] text-[#6B705C] block mt-0.5">Today</span>
                      </div>
                      <span className="text-xs text-purple-600 font-semibold flex items-center gap-1">
                        {metrics.screenTimeTrend}
                      </span>
                    </div>

                  </div>
                </div>

                {/* Row 2: Analytics Visualizations */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  
                  {/* Threat Category Summary Circular SVG Chart */}
                  <div className="bg-white border border-[#DDE0D0] rounded-xl p-5 shadow-sm flex flex-col">
                    <h3 className="font-semibold text-xs uppercase tracking-wider text-[#6B705C] border-b border-[#DDE0D0] pb-2 mb-4">
                      Threat Category Distribution
                    </h3>
                    
                    <div className="flex items-center justify-center flex-1 py-4">
                      <div className="relative w-36 h-36 flex items-center justify-center">
                        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 120 120">
                          <circle cx="60" cy="60" r="50" fill="transparent" stroke="#E6E8E0" strokeWidth="12" />
                          {donutSegments.map((segment, idx) => (
                            <circle
                              key={idx}
                              cx="60"
                              cy="60"
                              r="50"
                              fill="transparent"
                              stroke={segment.color}
                              strokeWidth="12"
                              strokeDasharray="314.16"
                              strokeDashoffset={segment.strokeOffset}
                              strokeLinecap="round"
                              className="transition-all duration-300"
                            />
                          ))}
                        </svg>
                        <div className="absolute flex flex-col items-center justify-center">
                          <span className="text-2xl font-bold font-serif-natural text-[#2D3025]">{totalThreats}</span>
                          <span className="text-[10px] text-[#6B705C] font-semibold">Total Flags</span>
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2 mt-4 text-[11px] text-[#2D3025]">
                      {donutSegments.map((segment, idx) => (
                        <div key={idx} className="flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: segment.color }} />
                          <span className="truncate text-[#6B705C]">{segment.name}</span>
                          <span className="font-bold ml-auto">{segment.count} ({segment.percentage}%)</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Emotion Change Trend SVG Line Chart */}
                  <div className="bg-white border border-[#DDE0D0] rounded-xl p-5 shadow-sm flex flex-col">
                    <h3 className="font-semibold text-xs uppercase tracking-wider text-[#6B705C] border-b border-[#DDE0D0] pb-2 mb-4">
                      Emotion Change Trend <span className="text-[10px] lowercase font-normal">(from behavioral data)</span>
                    </h3>

                    <div className="flex-1 flex flex-col justify-between py-2">
                      <div className="relative h-40 w-full flex items-end">
                        
                        {/* Y-axis Labels on Left */}
                        <div className="absolute left-0 top-0 bottom-0 flex flex-col justify-between text-[9px] text-[#6B705C] h-full pointer-events-none z-10">
                          <span>V. Positive</span>
                          <span>Positive</span>
                          <span>Neutral</span>
                          <span>Negative</span>
                          <span>V. Negative</span>
                        </div>

                        {/* Y-axis Gridlines */}
                        <div className="absolute inset-0 flex flex-col justify-between pl-12 h-full pointer-events-none">
                          <div className="border-t border-[#DDE0D0]/40 w-full h-0" />
                          <div className="border-t border-[#DDE0D0]/40 w-full h-0" />
                          <div className="border-t border-[#DDE0D0]/40 w-full h-0" />
                          <div className="border-t border-[#DDE0D0]/40 w-full h-0" />
                          <div className="border-t border-[#DDE0D0]/40 w-full h-0" />
                        </div>

                        {/* SVG Line path & points */}
                        <svg className="w-full h-full pl-12 overflow-visible" viewBox="0 0 300 160" preserveAspectRatio="none">
                          {emotionTrend && (
                            <>
                              <path
                                d={emotionTrend.pathD}
                                fill="none"
                                stroke="#5A5A40"
                                strokeWidth="2.5"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              />
                              {emotionTrend.points.map((pt, idx) => (
                                <circle key={idx} cx={pt.x} cy={pt.y} r="4.5" fill={pt.color} />
                              ))}
                            </>
                          )}
                        </svg>
                      </div>

                      {/* X-axis Labels */}
                      <div className="flex justify-between pl-12 text-[10px] text-[#6B705C] font-semibold mt-2">
                        {emotionTrend && (
                          emotionTrend.points.map((pt, idx) => (
                            <span key={idx}>{pt.timeStr}</span>
                          ))
                        )}
                      </div>
                    </div>

                    <div className="flex justify-center gap-4 text-[10px] mt-2 border-t border-[#DDE0D0]/50 pt-2">
                      <span className="flex items-center gap-1 text-[#10B981] font-bold">
                        <span className="w-2 h-2 rounded-full bg-[#10B981]" /> Positive
                      </span>
                      <span className="flex items-center gap-1 text-[#F59E0B] font-bold">
                        <span className="w-2 h-2 rounded-full bg-[#F59E0B]" /> Neutral
                      </span>
                      <span className="flex items-center gap-1 text-[#F43F5E] font-bold">
                        <span className="w-2 h-2 rounded-full bg-[#F43F5E]" /> Negative
                      </span>
                    </div>
                  </div>

                </div>

                {/* Row 3: Explainable AI & Recent Logs */}
                <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
                  
                  {/* Left Column: Why was this flagged (Explainable AI) */}
                  <div className="xl:col-span-7 bg-white border border-[#DDE0D0] rounded-xl p-5 shadow-sm flex flex-col gap-4">
                    <h3 className="font-semibold text-xs uppercase tracking-wider text-[#6B705C] border-b border-[#DDE0D0] pb-2">
                      Explainable AI (XAI) — Why this happened?
                    </h3>

                    {/* Explanations List */}
                    <div className="flex flex-col gap-3">
                      {explainableAI ? (
                        explainableAI.map((item) => (
                          <div key={item.id} className="p-3 bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg flex items-start justify-between gap-3 text-xs">
                            <div className="flex items-start gap-3">
                              <div className={`p-1.5 border rounded-lg mt-0.5 ${item.iconBg}`}>
                                <AlertTriangle className="w-4 h-4" />
                              </div>
                              <div>
                                <h4 className="font-bold text-[#2D3025]">{item.title}</h4>
                                <p className="text-[#6B705C] mt-0.5 leading-relaxed">
                                  {item.details}
                                </p>
                              </div>
                            </div>
                            <div className="text-right shrink-0">
                              <p className="text-[10px] text-[#6B705C] font-medium">Impact on Emotion</p>
                              <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded block mt-1 ${item.impactColor}`}>
                                {item.impact}
                              </span>
                            </div>
                          </div>
                        ))
                      ) : (
                        <>
                          {/* Item 1 */}
                          <div className="p-3 bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg flex items-start justify-between gap-3 text-xs">
                            <div className="flex items-start gap-3">
                              <div className="p-1.5 bg-rose-50 border border-rose-200 text-rose-600 rounded-lg mt-0.5">
                                <AlertTriangle className="w-4 h-4" />
                              </div>
                              <div>
                                <h4 className="font-bold text-[#2D3025]">Violent Video Detected</h4>
                                <p className="text-[#6B705C] mt-0.5 leading-relaxed">
                                  A violent gameplay match was played on YouTube Desktop app at 8:15 PM.
                                </p>
                              </div>
                            </div>
                            <div className="text-right shrink-0">
                              <p className="text-[10px] text-[#6B705C] font-medium">Impact on Emotion</p>
                              <span className="bg-rose-50 text-rose-800 border border-rose-100 text-[10px] font-bold px-1.5 py-0.5 rounded block mt-1">
                                Very Negative
                              </span>
                            </div>
                          </div>

                          {/* Item 2 */}
                          <div className="p-3 bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg flex items-start justify-between gap-3 text-xs">
                            <div className="flex items-start gap-3">
                              <div className="p-1.5 bg-orange-50 border border-orange-200 text-orange-600 rounded-lg mt-0.5">
                                <HelpCircle className="w-4 h-4" />
                              </div>
                              <div>
                                <h4 className="font-bold text-[#2D3025]">Hate Speech Content</h4>
                                <p className="text-[#6B705C] mt-0.5 leading-relaxed">
                                  Toxic verbal interaction and hate speech words detected in Discord text channel at 7:40 PM.
                                </p>
                              </div>
                            </div>
                            <div className="text-right shrink-0">
                              <p className="text-[10px] text-[#6B705C] font-medium">Impact on Emotion</p>
                              <span className="bg-orange-50 text-orange-800 border border-orange-100 text-[10px] font-bold px-1.5 py-0.5 rounded block mt-1">
                                Negative
                              </span>
                            </div>
                          </div>

                          {/* Item 3 */}
                          <div className="p-3 bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg flex items-start justify-between gap-3 text-xs">
                            <div className="flex items-start gap-3">
                              <div className="p-1.5 bg-purple-50 border border-purple-200 text-purple-600 rounded-lg mt-0.5">
                                <Clock className="w-4 h-4" />
                              </div>
                              <div>
                                <h4 className="font-bold text-[#2D3025]">Late Night Screen Activity</h4>
                                <p className="text-[#6B705C] mt-0.5 leading-relaxed">
                                  Device usage detected after safe limits boundary (10:30 PM) on {activePersona.name}&apos;s Desktop.
                                </p>
                              </div>
                            </div>
                            <div className="text-right shrink-0">
                              <p className="text-[10px] text-[#6B705C] font-medium">Impact on Emotion</p>
                              <span className="bg-orange-50 text-orange-800 border border-orange-100 text-[10px] font-bold px-1.5 py-0.5 rounded block mt-1">
                                Negative
                              </span>
                            </div>
                          </div>
                        </>
                      )}

                      {/* Parent Suggestion Banner */}
                      <div className="p-4 bg-[#E6D5C3]/20 border border-[#DDE0D0] rounded-lg flex items-start gap-3.5 mt-2">
                        <Sparkles className="w-5 h-5 text-[#5A5A40] shrink-0 mt-0.5 animate-pulse" />
                        <div className="text-xs">
                          <h4 className="font-bold text-[#5A5A40] uppercase tracking-wide text-[10px]">AI Assistant Recommendation</h4>
                          <p className="text-[#2D3025] mt-1 font-medium leading-relaxed">
                            {getAIRecommendation()}
                          </p>
                        </div>
                      </div>

                    </div>
                  </div>

                  {/* Right Column: Recent Alerts & Intercepts */}
                  <div className="xl:col-span-5 bg-white border border-[#DDE0D0] rounded-xl p-5 shadow-sm flex flex-col gap-4">
                    <div className="flex items-center justify-between border-b border-[#DDE0D0] pb-2">
                      <h3 className="font-semibold text-xs uppercase tracking-wider text-[#6B705C]">
                        Recent Alerts &amp; Intercepts
                      </h3>
                      <span className="text-[10px] text-[#6B705C] hover:text-[#2D3025] cursor-pointer font-bold flex items-center gap-0.5">
                        View All <ChevronRight className="w-3 h-3" />
                      </span>
                    </div>

                    <div className="flex flex-col gap-2.5">
                      {recentAlerts ? (
                        recentAlerts.map((alert) => (
                          <div key={alert.id} className="flex items-center justify-between p-2.5 hover:bg-[#FAF9F6] rounded-lg transition border border-transparent hover:border-[#DDE0D0]">
                            <div className="flex items-center gap-3">
                              <div className={`p-1.5 rounded ${alert.iconBg}`}>
                                <AlertTriangle className="w-3.5 h-3.5" />
                              </div>
                              <div>
                                <p className="text-xs font-bold text-[#2D3025]">{alert.title}</p>
                                <p className="text-[10px] text-[#6B705C]">{alert.app}</p>
                              </div>
                            </div>
                            <div className="text-right">
                              <p className="text-[10px] text-[#6B705C] font-mono">{alert.timeStr}</p>
                              <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded inline-block mt-0.5 ${alert.severityColor}`}>
                                {alert.severity}
                              </span>
                            </div>
                          </div>
                        ))
                      ) : (
                        <>
                          {/* Log 1 */}
                          <div className="flex items-center justify-between p-2.5 hover:bg-[#FAF9F6] rounded-lg transition border border-transparent hover:border-[#DDE0D0]">
                            <div className="flex items-center gap-3">
                              <div className="p-1.5 bg-rose-50 text-rose-600 rounded">
                                <AlertTriangle className="w-3.5 h-3.5" />
                              </div>
                              <div>
                                <p className="text-xs font-bold text-[#2D3025]">Violent Video Detected</p>
                                <p className="text-[10px] text-[#6B705C]">YouTube Desktop</p>
                              </div>
                            </div>
                            <div className="text-right">
                              <p className="text-[10px] text-[#6B705C] font-mono">9:21 PM</p>
                              <span className="text-[9px] bg-rose-50 text-rose-700 font-bold px-1.5 py-0.5 rounded border border-rose-100 inline-block mt-0.5">
                                High
                              </span>
                            </div>
                          </div>

                          {/* Log 2 */}
                          <div className="flex items-center justify-between p-2.5 hover:bg-[#FAF9F6] rounded-lg transition border border-transparent hover:border-[#DDE0D0]">
                            <div className="flex items-center gap-3">
                              <div className="p-1.5 bg-rose-50 text-rose-600 rounded">
                                <AlertTriangle className="w-3.5 h-3.5" />
                              </div>
                              <div>
                                <p className="text-xs font-bold text-[#2D3025]">Hate Speech Content</p>
                                <p className="text-[10px] text-[#6B705C]">Discord App</p>
                              </div>
                            </div>
                            <div className="text-right">
                              <p className="text-[10px] text-[#6B705C] font-mono">8:45 PM</p>
                              <span className="text-[9px] bg-rose-50 text-rose-700 font-bold px-1.5 py-0.5 rounded border border-rose-100 inline-block mt-0.5">
                                High
                              </span>
                            </div>
                          </div>

                          {/* Log 3 */}
                          <div className="flex items-center justify-between p-2.5 hover:bg-[#FAF9F6] rounded-lg transition border border-transparent hover:border-[#DDE0D0]">
                            <div className="flex items-center gap-3">
                              <div className="p-1.5 bg-purple-50 text-purple-600 rounded">
                                <Monitor className="w-3.5 h-3.5" />
                              </div>
                              <div>
                                <p className="text-xs font-bold text-[#2D3025]">Adult Content Blocked</p>
                                <p className="text-[10px] text-[#6B705C]">Website Browser</p>
                              </div>
                            </div>
                            <div className="text-right">
                              <p className="text-[10px] text-[#6B705C] font-mono">7:30 PM</p>
                              <span className="text-[9px] bg-purple-50 text-purple-700 font-bold px-1.5 py-0.5 rounded border border-purple-100 inline-block mt-0.5">
                                Medium
                              </span>
                            </div>
                          </div>

                          {/* Log 4 */}
                          <div className="flex items-center justify-between p-2.5 hover:bg-[#FAF9F6] rounded-lg transition border border-transparent hover:border-[#DDE0D0]">
                            <div className="flex items-center gap-3">
                              <div className="p-1.5 bg-rose-50 text-rose-600 rounded">
                                <AlertTriangle className="w-3.5 h-3.5" />
                              </div>
                              <div>
                                <p className="text-xs font-bold text-[#2D3025]">Extreme Battle Video</p>
                                <p className="text-[10px] text-[#6B705C]">USB Drive Playback</p>
                              </div>
                            </div>
                            <div className="text-right">
                              <p className="text-[10px] text-[#6B705C] font-mono">6:15 PM</p>
                              <span className="text-[9px] bg-rose-50 text-rose-700 font-bold px-1.5 py-0.5 rounded border border-rose-100 inline-block mt-0.5">
                                High
                              </span>
                            </div>
                          </div>

                          {/* Log 5 */}
                          <div className="flex items-center justify-between p-2.5 hover:bg-[#FAF9F6] rounded-lg transition border border-transparent hover:border-[#DDE0D0]">
                            <div className="flex items-center gap-3">
                              <div className="p-1.5 bg-emerald-50 text-emerald-600 rounded">
                                <CheckCircle className="w-3.5 h-3.5" />
                              </div>
                              <div>
                                <p className="text-xs font-bold text-[#2D3025]">Controversial Rant</p>
                                <p className="text-[10px] text-[#6B705C]">Social Media App</p>
                              </div>
                            </div>
                            <div className="text-right">
                              <p className="text-[10px] text-[#6B705C] font-mono">5:02 PM</p>
                              <span className="text-[9px] bg-emerald-50 text-emerald-800 font-bold px-1.5 py-0.5 rounded border border-emerald-100 inline-block mt-0.5">
                                Low
                              </span>
                            </div>
                          </div>
                        </>
                      )}
                    </div>
                  </div>

                </div>

              </motion.div>
            )}

            {/* Child Profile & Persona Creator Tab */}
            {activeSidebarTab === "profile" && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                transition={{ duration: 0.15 }}
                className="grid grid-cols-1 xl:grid-cols-12 gap-6"
                key="profile-tab"
              >
                
                {/* Persona Listing & Viewer Details Card (Left) */}
                <div className="xl:col-span-5 flex flex-col gap-6">
                  
                  {/* Active Profile Info Panel */}
                  <div className="bg-white border border-[#DDE0D0] rounded-xl p-6 shadow-sm">
                    <div className="flex items-center justify-between border-b border-[#DDE0D0] pb-3 mb-5">
                      <h3 className="font-semibold text-xs uppercase tracking-wider text-[#6B705C]">Child Safety Persona</h3>
                      {!isEditing && (
                        <button
                          onClick={() => setIsEditing(true)}
                          className="text-xs text-[#5A5A40] hover:text-[#2D3025] font-bold flex items-center gap-1 transition"
                        >
                          <Edit2 className="w-3.5 h-3.5" /> Edit Persona
                        </button>
                      )}
                    </div>

                    <div className="flex flex-col items-center text-center pb-5 mb-5 border-b border-[#DDE0D0]/50">
                      <div className="w-20 h-20 rounded-full bg-[#5A5A40]/10 border-2 border-[#5A5A40]/30 flex items-center justify-center text-3xl font-bold text-[#5A5A40] shadow-xs mb-3">
                        {activePersona.name.charAt(0)}
                      </div>
                      <h4 className="text-lg font-bold text-[#2D3025]">{activePersona.name}</h4>
                      <p className="text-xs text-[#6B705C] font-semibold mt-1">
                        {activePersona.age} Years Old &bull; {activePersona.grade}
                      </p>
                    </div>

                    <div className="flex flex-col gap-4 text-xs">
                      <div>
                        <span className="text-[#6B705C] uppercase tracking-wider text-[10px] font-bold block mb-1">Personality Characteristics</span>
                        <div className="flex flex-wrap gap-1.5">
                          {activePersona.personality.map((tag, i) => (
                            <span key={i} className="bg-[#5A5A40]/10 text-[#5A5A40] border border-[#5A5A40]/15 px-2.5 py-1 rounded-full font-semibold">
                              {tag}
                            </span>
                          ))}
                          {activePersona.personality.length === 0 && (
                            <span className="text-[#6B705C] italic">No personality traits defined.</span>
                          )}
                        </div>
                      </div>

                      <div>
                        <span className="text-[#6B705C] uppercase tracking-wider text-[10px] font-bold block mb-1">Interests &amp; Hobbies</span>
                        <div className="flex flex-wrap gap-1.5">
                          {activePersona.interests.map((tag, i) => (
                            <span key={i} className="bg-[#E6D5C3]/40 text-[#5A5A40] border border-[#DDE0D0] px-2.5 py-1 rounded-full font-semibold">
                              {tag}
                            </span>
                          ))}
                          {activePersona.interests.length === 0 && (
                            <span className="text-[#6B705C] italic">No interests defined.</span>
                          )}
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4 border-t border-[#DDE0D0]/50 pt-4 mt-2">
                        <div>
                          <span className="text-[#6B705C] uppercase tracking-wider text-[10px] font-bold block mb-0.5">Monitoring Devices</span>
                          <span className="font-semibold text-[#2D3025]">{activePersona.devices}</span>
                        </div>
                        <div>
                          <span className="text-[#6B705C] uppercase tracking-wider text-[10px] font-bold block mb-0.5">Registered Since</span>
                          <span className="font-semibold text-[#2D3025]">{activePersona.memberSince}</span>
                        </div>
                      </div>

                    </div>
                  </div>

                  {/* Persona Switcher panel and actions */}
                  <div className="bg-white border border-[#DDE0D0] rounded-xl p-5 shadow-sm flex flex-col gap-4">
                    <h3 className="font-semibold text-xs uppercase tracking-wider text-[#6B705C] border-b border-[#DDE0D0] pb-2">
                      Manage Profiles
                    </h3>
                    <div className="flex flex-col gap-2">
                      {personas.map(p => (
                        <div key={p.id} className="flex items-center justify-between p-2 rounded-lg border border-[#DDE0D0]/50 hover:bg-[#FAF9F6]">
                          <span className="text-xs font-semibold text-[#2D3025]">{p.name}</span>
                          <div className="flex items-center gap-1.5">
                            <button
                              onClick={() => {
                                setActivePersonaId(p.id);
                                setIsEditing(false);
                                setIsCreatingNew(false);
                              }}
                              className={`px-2.5 py-1 rounded text-[10px] font-semibold transition ${
                                p.id === activePersonaId
                                  ? "bg-[#5A5A40] text-white"
                                  : "bg-[#FAF9F6] text-[#6B705C] border border-[#DDE0D0] hover:text-[#2D3025]"
                              }`}
                            >
                              Select
                            </button>
                            <button
                              onClick={() => deletePersona(p.id)}
                              className="p-1.5 rounded text-rose-600 hover:bg-rose-50 border border-transparent hover:border-rose-100 transition"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                </div>

                {/* Interactive Editor Form Column (Right) */}
                <div className="xl:col-span-7">
                  <div className="bg-white border border-[#DDE0D0] rounded-xl p-6 shadow-sm h-full flex flex-col justify-between">
                    
                    <div>
                      <div className="flex items-center justify-between border-b border-[#DDE0D0] pb-3 mb-5">
                        <h3 className="font-semibold text-sm uppercase tracking-wider text-[#2D3025]">
                          {isCreatingNew ? "Create Persona" : isEditing ? "Edit Persona Configuration" : "Persona Dashboard Actions"}
                        </h3>
                        <span className="text-xs text-[#6B705C] flex items-center gap-1">
                          <Info className="w-3.5 h-3.5 text-[#5A5A40]" /> Configure cognitive prompts
                        </span>
                      </div>

                      {isEditing ? (
                        <form onSubmit={isCreatingNew ? createNewPersona : (e) => { e.preventDefault(); savePersona(); }} className="flex flex-col gap-4">
                          
                          {/* Name Input */}
                          <div className="flex flex-col gap-1.5">
                            <label className="text-xs font-bold text-[#6B705C]">Child&apos;s Name</label>
                            <input
                              type="text"
                              required
                              value={editName}
                              onChange={(e) => setEditName(e.target.value)}
                              className="w-full bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg py-2 px-3 text-xs text-[#2D3025] focus:outline-none focus:border-[#5A5A40]"
                              placeholder="e.g. Sandeep Perera"
                            />
                          </div>

                          {/* Age & Grade Inputs */}
                          <div className="grid grid-cols-2 gap-4">
                            <div className="flex flex-col gap-1.5">
                              <label className="text-xs font-bold text-[#6B705C]">Age (Context Trigger)</label>
                              <select
                                value={editAge}
                                onChange={(e) => setEditAge(parseInt(e.target.value))}
                                className="w-full bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg py-2 px-3 text-xs text-[#2D3025] focus:outline-none focus:border-[#5A5A40]"
                              >
                                {Array.from({ length: 12 }, (_, i) => i + 5).map(ageVal => (
                                  <option key={ageVal} value={ageVal}>{ageVal} Years Old</option>
                                ))}
                              </select>
                            </div>
                            
                            <div className="flex flex-col gap-1.5">
                              <label className="text-xs font-bold text-[#6B705C]">Grade Level</label>
                              <input
                                type="text"
                                required
                                value={editGrade}
                                onChange={(e) => setEditGrade(e.target.value)}
                                className="w-full bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg py-2 px-3 text-xs text-[#2D3025] focus:outline-none focus:border-[#5A5A40]"
                                placeholder="e.g. Grade 5"
                              />
                            </div>
                          </div>

                          {/* Personality Traits */}
                          <div className="flex flex-col gap-1.5">
                            <label className="text-xs font-bold text-[#6B705C]">Personality Characteristics (Comma separated)</label>
                            <input
                              type="text"
                              value={editPersonality}
                              onChange={(e) => setEditPersonality(e.target.value)}
                              className="w-full bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg py-2 px-3 text-xs text-[#2D3025] focus:outline-none focus:border-[#5A5A40]"
                              placeholder="e.g. Curious, Sensitive, Analytical"
                            />
                            <p className="text-[10px] text-[#6B705C]">These attributes customize the Socratic Buddy&apos;s guiding vocabulary.</p>
                          </div>

                          {/* Interests */}
                          <div className="flex flex-col gap-1.5">
                            <label className="text-xs font-bold text-[#6B705C]">Interests &amp; Hobbies (Comma separated)</label>
                            <input
                              type="text"
                              value={editInterests}
                              onChange={(e) => setEditInterests(e.target.value)}
                              className="w-full bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg py-2 px-3 text-xs text-[#2D3025] focus:outline-none focus:border-[#5A5A40]"
                              placeholder="e.g. Nature, Space, Music, Gaming"
                            />
                          </div>

                          {/* Devices */}
                          <div className="flex flex-col gap-1.5">
                            <label className="text-xs font-bold text-[#6B705C]">Active Monitored Devices</label>
                            <input
                              type="text"
                              value={editDevices}
                              onChange={(e) => setEditDevices(e.target.value)}
                              className="w-full bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg py-2 px-3 text-xs text-[#2D3025] focus:outline-none focus:border-[#5A5A40]"
                              placeholder="e.g. 1 Desktop, 1 Tablet"
                            />
                          </div>

                          {/* Action Buttons */}
                          <div className="flex items-center gap-2.5 mt-4">
                            <button
                              type="submit"
                              className="flex-1 py-2.5 px-4 bg-[#5A5A40] hover:bg-[#454530] text-white font-bold rounded-lg text-xs transition flex items-center justify-center gap-1.5"
                            >
                              <Save className="w-3.5 h-3.5" /> Save Changes
                            </button>
                            <button
                              type="button"
                              onClick={() => {
                                setIsEditing(false);
                                setIsCreatingNew(false);
                              }}
                              className="py-2.5 px-4 bg-white border border-[#DDE0D0] hover:bg-[#FAF9F6] text-[#6B705C] hover:text-[#2D3025] font-bold rounded-lg text-xs transition"
                            >
                              Cancel
                            </button>
                          </div>

                        </form>
                      ) : (
                        <div className="flex flex-col gap-6 text-xs text-[#2D3025]">
                          <div className="bg-[#FAF9F6] p-4 rounded-lg border border-[#DDE0D0] flex flex-col gap-2">
                            <h4 className="font-bold text-[#5A5A40] flex items-center gap-1.5 uppercase text-[10px] tracking-wide">
                              <Sparkles className="w-3.5 h-3.5 text-[#5A5A40]" /> Dynamic Age Routing &amp; Guard Prompts
                            </h4>
                            <p className="text-[#6B705C] leading-relaxed">
                              Depending on the child profile&apos;s age, the guard engine scales the severity threshold and alters vocabulary structures:
                            </p>
                            <div className="grid grid-cols-2 gap-3 mt-1.5">
                              <div className="bg-white p-2.5 border border-[#DDE0D0] rounded">
                                <p className="font-bold text-[#2D3025]">Age &le; 10 Years Old</p>
                                <p className="text-[10px] text-[#6B705C] mt-0.5 leading-relaxed">Warm, elementary vocabulary buddy. Validates emotions with low threshold gating.</p>
                              </div>
                              <div className="bg-white p-2.5 border border-[#DDE0D0] rounded">
                                <p className="font-bold text-[#2D3025]">Age 11+ Years Old</p>
                                <p className="text-[10px] text-[#6B705C] mt-0.5 leading-relaxed">Autonomy builder. High-level negotiation, respecting young adult critical inquiry.</p>
                              </div>
                            </div>
                          </div>

                          <div className="flex flex-col gap-2">
                            <h4 className="font-bold text-[#6B705C] uppercase text-[10px] tracking-wide">AI Recommendation Rules</h4>
                            <div className="p-3.5 border border-[#DDE0D0] rounded-lg bg-[#E6D5C3]/10 flex items-start gap-3">
                              <Info className="w-4 h-4 text-[#5A5A40] shrink-0 mt-0.5" />
                              <div className="leading-relaxed text-[#2D3025]">
                                Parental suggestions adapt to anomalous scores calculated locally using Isolation Forests. When the Behavior Anomaly Score surpasses <strong className="text-rose-700 font-bold">0.85</strong>, behavioral warning suggestions will display.
                              </div>
                            </div>
                          </div>

                          <button
                            onClick={() => setIsEditing(true)}
                            className="w-full py-3 px-4 bg-[#5A5A40] hover:bg-[#454530] text-white font-bold rounded-lg text-xs transition duration-200 flex items-center justify-center gap-1.5 mt-4"
                          >
                            <Sliders className="w-4 h-4" /> Edit Profile Persona Data
                          </button>
                        </div>
                      )}
                    </div>
                    
                    {!isEditing && (
                      <div className="mt-8 p-3 bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg text-[10px] text-[#6B705C] text-center">
                        All profiles and custom persona models are stored in local offline sandbox tables to guarantee student/minor privacy compliance.
                      </div>
                    )}

                  </div>
                </div>

              </motion.div>
            )}

          </AnimatePresence>

        </main>
      </div>

      {/* Footer */}
      <footer className="border-t border-[#DDE0D0] bg-[#FAF9F6] p-5 text-center text-[#6B705C] text-xs shrink-0" id="parent-footer">
        <p>Socratic Parental Control Dashboard &bull; Google AI Studio Applet &bull; Feature: behavioral-profiling</p>
      </footer>

    </div>
  );
}
