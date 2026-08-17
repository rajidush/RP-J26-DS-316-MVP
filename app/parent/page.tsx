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

  const activePersona = personas.find(p => p.id === activePersonaId) || personas[0] || DEFAULT_PERSONAS[0];

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

  // Mock Overview Analytics Data (Dynamic overrides based on active child age)
  const getMetrics = () => {
    if (activePersona.age <= 10) {
      return {
        intercepts: 24,
        interceptsTrend: "↓ 12% vs yesterday",
        alerts: 3,
        alertsTrend: "↓ 25% vs yesterday",
        hateSpeech: 7,
        hateSpeechTrend: "↓ 8% vs yesterday",
        screenTime: "2h 45m",
        screenTimeTrend: "↓ 15% vs yesterday",
        anomalyScore: 0.28,
        threats: { violence: 8, hate: 6, adult: 5, controversial: 3, other: 2 }
      };
    } else {
      // Teenagers have different baseline stats
      return {
        intercepts: 41,
        interceptsTrend: "↑ 8% vs yesterday",
        alerts: 6,
        alertsTrend: "↓ 10% vs yesterday",
        hateSpeech: 12,
        hateSpeechTrend: "↑ 14% vs yesterday",
        screenTime: "4h 20m",
        screenTimeTrend: "↑ 5% vs yesterday",
        anomalyScore: 0.54,
        threats: { violence: 10, hate: 15, adult: 9, controversial: 5, other: 2 }
      };
    }
  };

  const metrics = getMetrics();
  const totalThreats = Object.values(metrics.threats).reduce((a, b) => a + b, 0);

  // Custom SVG donut chart segments calculation
  const getDonutSegments = () => {
    let currentOffset = 0;
    const categories = [
      { name: "High Violence", count: metrics.threats.violence, color: "#F43F5E" },
      { name: "Hate Speech", count: metrics.threats.hate, color: "#F59E0B" },
      { name: "Adult Content", count: metrics.threats.adult, color: "#A855F7" },
      { name: "Controversial", count: metrics.threats.controversial, color: "#10B981" },
      { name: "Other", count: metrics.threats.other, color: "#3B82F6" },
    ];
    
    return categories.map(cat => {
      const percentage = (cat.count / totalThreats) * 100;
      const strokeLength = (percentage / 100) * 314.16;
      const strokeOffset = 314.16 - strokeLength + currentOffset;
      currentOffset -= strokeLength;
      return {
        ...cat,
        percentage: Math.round(percentage),
        strokeLength,
        strokeOffset
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
            <div className="p-3 bg-emerald-50 rounded-lg border border-emerald-200 flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <div className="text-[10px] text-emerald-800 leading-tight">
                <p className="font-bold">System Connected</p>
                <p className="opacity-80">100% Offline Mode</p>
              </div>
            </div>

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
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                  
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
                          {/* Guideline path */}
                          <path
                            d={
                              activePersona.age <= 10
                                ? "M 10 30 L 58 10 L 106 30 L 154 130 L 202 90 L 250 10 L 290 30"
                                : "M 10 90 L 58 70 L 106 90 L 154 130 L 202 110 L 250 50 L 290 90"
                            }
                            fill="none"
                            stroke="#5A5A40"
                            strokeWidth="2.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                          
                          {/* Points */}
                          {activePersona.age <= 10 ? (
                            <>
                              <circle cx="10" cy="30" r="4.5" fill="#10B981" />
                              <circle cx="58" cy="10" r="4.5" fill="#10B981" />
                              <circle cx="106" cy="30" r="4.5" fill="#10B981" />
                              <circle cx="154" cy="130" r="4.5" fill="#F43F5E" />
                              <circle cx="202" cy="90" r="4.5" fill="#F59E0B" />
                              <circle cx="250" cy="10" r="4.5" fill="#10B981" />
                              <circle cx="290" cy="30" r="4.5" fill="#10B981" />
                            </>
                          ) : (
                            <>
                              <circle cx="10" cy="90" r="4.5" fill="#F59E0B" />
                              <circle cx="58" cy="70" r="4.5" fill="#10B981" />
                              <circle cx="106" cy="90" r="4.5" fill="#F59E0B" />
                              <circle cx="154" cy="130" r="4.5" fill="#F43F5E" />
                              <circle cx="202" cy="110" r="4.5" fill="#F59E0B" />
                              <circle cx="250" cy="50" r="4.5" fill="#10B981" />
                              <circle cx="290" cy="90" r="4.5" fill="#F59E0B" />
                            </>
                          )}
                        </svg>
                      </div>

                      {/* X-axis Labels */}
                      <div className="flex justify-between pl-12 text-[10px] text-[#6B705C] font-semibold mt-2">
                        <span>6 Aug</span>
                        <span>7 Aug</span>
                        <span>8 Aug</span>
                        <span>9 Aug</span>
                        <span>10 Aug</span>
                        <span>11 Aug</span>
                        <span>12 Aug</span>
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

                  {/* Behavior Anomaly Score SVG Gauge */}
                  <div className="bg-white border border-[#DDE0D0] rounded-xl p-5 shadow-sm flex flex-col">
                    <h3 className="font-semibold text-xs uppercase tracking-wider text-[#6B705C] border-b border-[#DDE0D0] pb-2 mb-4">
                      Behavior Anomaly Score <span className="text-[10px] lowercase font-normal">(Isolation Forest)</span>
                    </h3>

                    <div className="flex-1 flex flex-col items-center justify-center py-4">
                      <div className="relative w-44 h-24 flex flex-col items-center justify-end overflow-hidden">
                        
                        {/* Speedometer SVG Arc */}
                        <svg className="w-full h-full" viewBox="0 0 100 50">
                          {/* Background Arc */}
                          <path
                            d="M 10 45 A 40 40 0 0 1 90 45"
                            fill="none"
                            stroke="#E6E8E0"
                            strokeWidth="8"
                            strokeLinecap="round"
                          />
                          {/* Value Arc colored base on score */}
                          <path
                            d="M 10 45 A 40 40 0 0 1 90 45"
                            fill="none"
                            stroke={metrics.anomalyScore > 0.85 ? "#F43F5E" : metrics.anomalyScore > 0.5 ? "#F59E0B" : "#10B981"}
                            strokeWidth="8"
                            strokeLinecap="round"
                            strokeDasharray={125.66}
                            strokeDashoffset={125.66 * (1 - metrics.anomalyScore)}
                            className="transition-all duration-500"
                          />
                          {/* Threshold Dash (0.85 of half-circle which is 180deg) */}
                          <line
                            x1="50"
                            y1="45"
                            x2={50 + 40 * Math.cos(Math.PI * (1 - 0.85))}
                            y2={45 - 40 * Math.sin(Math.PI * (1 - 0.85))}
                            stroke="#EF4444"
                            strokeWidth="1.5"
                            strokeDasharray="2,2"
                          />
                        </svg>

                        {/* Needle pointing at score */}
                        <div
                          className="absolute bottom-1 origin-bottom w-1 h-14 bg-slate-800 rounded-full transition-transform duration-500"
                          style={{
                            transform: `rotate(${(metrics.anomalyScore * 180) - 90}deg)`,
                          }}
                        />
                        <div className="w-3 h-3 rounded-full bg-slate-800 absolute bottom-0" />
                      </div>
                      
                      {/* Metric Values */}
                      <div className="text-center mt-3 flex flex-col">
                        <span className="text-2xl font-bold text-[#2D3025]">{metrics.anomalyScore}</span>
                        <span className={`text-xs font-bold ${
                          metrics.anomalyScore > 0.85
                            ? "text-rose-600"
                            : metrics.anomalyScore > 0.5
                            ? "text-orange-600"
                            : "text-emerald-700"
                        }`}>
                          {metrics.anomalyScore > 0.85 ? "Abnormal Behavior Flagged" : metrics.anomalyScore > 0.5 ? "Elevated Variance" : "Normal Behavior"}
                        </span>
                        <span className="text-[10px] text-[#6B705C] font-semibold mt-1">
                          Risk Threshold: 0.85
                        </span>
                      </div>
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

                      {/* Parent Suggestion Banner */}
                      <div className="p-4 bg-[#E6D5C3]/20 border border-[#DDE0D0] rounded-lg flex items-start gap-3.5 mt-2">
                        <Sparkles className="w-5 h-5 text-[#5A5A40] shrink-0 mt-0.5 animate-pulse" />
                        <div className="text-xs">
                          <h4 className="font-bold text-[#5A5A40] uppercase tracking-wide text-[10px]">AI Assistant Recommendation</h4>
                          <p className="text-[#2D3025] mt-1 font-medium leading-relaxed">
                            {activePersona.age <= 10 ? (
                              "Limit violent video content exposure to less than 15 mins daily. Socratic dialogue suggests Sandeep is highly receptive to boundaries when discussed in a supportive tone. Encourage early bedroom screens curfew at 9:30 PM."
                            ) : (
                              "Discuss social circle dynamic on Discord. Young adult exhibits stress response (Elevated Anomaly Index) following late night gaming. Pivot safety agreements collaboratively rather than executing strict lockouts."
                            )}
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
