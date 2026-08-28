"use client";

import React, { useState, useEffect } from "react";
import { motion } from "motion/react";

export interface SocraticAvatarReplicaProps {
  isSpeaking: boolean;
  isListening: boolean;
  visemeLevel: number; // 0 to 3
  childAge?: number;
  childEmotion?: string;
  currentPhase?: string;
  loadingTurn?: boolean;
  isCompleted?: boolean;
  isMuted?: boolean;
  onToggleMute?: () => void;
  onReplaySpeech?: () => void;
  voices?: SpeechSynthesisVoice[];
  selectedVoice?: SpeechSynthesisVoice | null;
  onSelectVoice?: (voice: SpeechSynthesisVoice) => void;
  voiceRate?: number;
  onChangeRate?: (rate: number) => void;
  voicePitch?: number;
  onChangePitch?: (pitch: number) => void;
  lastSpokenSnippet?: string;
}

export const SocraticAvatarReplica: React.FC<SocraticAvatarReplicaProps> = ({
  isSpeaking,
  isListening,
  visemeLevel,
  childAge = 10,
  childEmotion = "neutral",
  loadingTurn = false,
  isCompleted = false,
  isMuted = false,
}) => {
  const [isBlinking, setIsBlinking] = useState(false);

  // Periodic natural blinking
  useEffect(() => {
    const blinkInterval = setInterval(() => {
      setIsBlinking(true);
      setTimeout(() => setIsBlinking(false), 180);
    }, 3600);

    return () => clearInterval(blinkInterval);
  }, []);

  const isCelebratory = isCompleted || childEmotion === "compliant";
  const isComforting = ["scared", "defensive", "frustrated"].includes(childEmotion.toLowerCase());
  const isReflective = ["curious", "reflective"].includes(childEmotion.toLowerCase());

  return (
    <div className="bg-gradient-to-b from-[#FAF9F6] to-[#F3F4ED] border-b border-[#DDE0D0] py-5 px-4 flex flex-col items-center justify-center relative select-none">
      {/* Main Avatar Character Stage */}
      <div className="relative flex flex-col items-center justify-center">
        {/* Pulsing Concentric Audio Rings during Speech */}
        {isSpeaking && !isMuted && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <motion.div
              animate={{ scale: [1, 1.45, 1.8], opacity: [0.65, 0.35, 0] }}
              transition={{ repeat: Infinity, duration: 1.6, ease: "easeOut" }}
              className="w-24 h-24 rounded-full border-2 border-[#5A5A40]/40 bg-[#5A5A40]/10"
            />
            <motion.div
              animate={{ scale: [1, 1.3, 1.6], opacity: [0.55, 0.25, 0] }}
              transition={{ repeat: Infinity, duration: 1.6, delay: 0.5, ease: "easeOut" }}
              className="w-24 h-24 rounded-full border border-amber-600/30 bg-amber-500/5"
            />
          </div>
        )}

        {/* Listening Ambient Glow during STT Microphone Input */}
        {isListening && (
          <motion.div
            animate={{ scale: [0.95, 1.15, 0.95], opacity: [0.4, 0.8, 0.4] }}
            transition={{ repeat: Infinity, duration: 1.2, ease: "easeInOut" }}
            className="absolute inset-0 flex items-center justify-center pointer-events-none"
          >
            <div className="w-28 h-28 rounded-full bg-emerald-400/20 blur-md border border-emerald-500/40" />
          </motion.div>
        )}

        {/* Avatar SVG Figure */}
        <motion.div
          animate={
            isSpeaking
              ? { y: [0, -3, 0, -2, 0], rotate: [0, -1, 1, 0] }
              : loadingTurn
              ? { y: [0, -4, 0], scale: [1, 1.02, 1] }
              : isListening
              ? { scale: [1, 1.04, 1] }
              : { y: [0, -2, 0] }
          }
          transition={{
            repeat: Infinity,
            duration: isSpeaking ? 0.8 : loadingTurn ? 1.2 : 3.0,
            ease: "easeInOut"
          }}
          className="relative z-10 w-24 h-24 rounded-3xl bg-gradient-to-b from-[#5A5A40] to-[#454530] p-1 shadow-md border-2 border-white flex items-center justify-center cursor-default"
        >
          {/* Guardian Shield Antenna / Halo Indicator */}
          <div className="absolute -top-3.5 flex flex-col items-center">
            <motion.div 
              animate={isSpeaking ? { scale: [1, 1.3, 1] } : {}}
              transition={{ repeat: Infinity, duration: 0.5 }}
              className={`w-3.5 h-3.5 rounded-full border-2 border-white shadow-xs flex items-center justify-center ${
                isCelebratory 
                  ? "bg-emerald-500" 
                  : isSpeaking 
                  ? "bg-amber-400" 
                  : isListening 
                  ? "bg-emerald-400" 
                  : "bg-[#E6D5C3]"
              }`}
            >
              <div className="w-1.5 h-1.5 rounded-full bg-white animate-ping opacity-75" />
            </motion.div>
            <div className="w-0.5 h-2 bg-white/60" />
          </div>

          {/* Avatar Face Container */}
          <div className="w-full h-full rounded-2xl bg-[#3F402C] flex flex-col items-center justify-center relative overflow-hidden px-2 pt-1">
            {/* Ambient Background Gradient inside Face */}
            <div className="absolute inset-0 bg-radial from-white/10 to-transparent pointer-events-none" />

            {/* Eyebrows */}
            <div className="w-14 flex justify-between items-center px-1 mb-1 z-10">
              {/* Left Eyebrow */}
              <motion.div 
                animate={
                  isComforting 
                    ? { rotate: 12, y: -1 } 
                    : isReflective 
                    ? { rotate: -8, y: -2 } 
                    : { rotate: 0, y: 0 }
                }
                className="w-3.5 h-1 bg-[#E6D5C3] rounded-full" 
              />
              {/* Right Eyebrow */}
              <motion.div 
                animate={
                  isComforting 
                    ? { rotate: -12, y: -1 } 
                    : isReflective 
                    ? { rotate: 10, y: 1 } 
                    : { rotate: 0, y: 0 }
                }
                className="w-3.5 h-1 bg-[#E6D5C3] rounded-full" 
              />
            </div>

            {/* Eyes */}
            <div className="w-14 flex justify-between items-center px-1.5 z-10">
              {/* Left Eye */}
              <div className="w-3.5 h-3.5 rounded-full bg-white flex items-center justify-center relative overflow-hidden shadow-inner">
                {isBlinking ? (
                  <div className="w-full h-0.5 bg-[#2D3025]" />
                ) : isCelebratory ? (
                  <div className="w-3 h-2 rounded-t-full border-t-2 border-[#2D3025] -mt-1" />
                ) : (
                  <motion.div 
                    animate={loadingTurn ? { x: [-1, 1, -1] } : { x: 0 }}
                    transition={{ repeat: Infinity, duration: 1 }}
                    className="w-2.5 h-2.5 rounded-full bg-[#1F2016] flex items-start justify-end p-0.5"
                  >
                    <div className="w-1 h-1 rounded-full bg-white" />
                  </motion.div>
                )}
              </div>

              {/* Right Eye */}
              <div className="w-3.5 h-3.5 rounded-full bg-white flex items-center justify-center relative overflow-hidden shadow-inner">
                {isBlinking ? (
                  <div className="w-full h-0.5 bg-[#2D3025]" />
                ) : isCelebratory ? (
                  <div className="w-3 h-2 rounded-t-full border-t-2 border-[#2D3025] -mt-1" />
                ) : (
                  <motion.div 
                    animate={loadingTurn ? { x: [-1, 1, -1] } : { x: 0 }}
                    transition={{ repeat: Infinity, duration: 1 }}
                    className="w-2.5 h-2.5 rounded-full bg-[#1F2016] flex items-start justify-end p-0.5"
                  >
                    <div className="w-1 h-1 rounded-full bg-white" />
                  </motion.div>
                )}
              </div>
            </div>

            {/* Cheeks (for friendly demeanor) */}
            <div className="w-16 flex justify-between px-1 mt-1 z-10 opacity-70">
              <div className="w-2.5 h-1.5 rounded-full bg-rose-400/50 blur-[0.5px]" />
              <div className="w-2.5 h-1.5 rounded-full bg-rose-400/50 blur-[0.5px]" />
            </div>

            {/* Dynamic Lip-Sync Mouth */}
            <div className="h-4 flex items-center justify-center mt-0.5 z-10">
              {isSpeaking && !isMuted ? (
                /* Dynamic Viseme Mouth States */
                visemeLevel === 3 ? (
                  <motion.div 
                    layout
                    className="w-4 h-3.5 rounded-full bg-[#E6D5C3] border border-[#2D3025] flex items-center justify-center overflow-hidden"
                  >
                    <div className="w-2.5 h-1.5 rounded-full bg-rose-500 -mb-1" />
                  </motion.div>
                ) : visemeLevel === 2 ? (
                  <motion.div 
                    layout
                    className="w-3.5 h-2.5 rounded-full bg-[#E6D5C3] border border-[#2D3025]" 
                  />
                ) : visemeLevel === 1 ? (
                  <motion.div 
                    layout
                    className="w-3 h-1.5 rounded-full bg-[#E6D5C3]" 
                  />
                ) : (
                  <div className="w-3 h-0.5 bg-[#E6D5C3] rounded-full" />
                )
              ) : isCelebratory ? (
                /* Big warm smile */
                <div className="w-4 h-2 rounded-b-full border-b-2 border-[#E6D5C3]" />
              ) : isComforting ? (
                /* Gentle curved warm reassurance */
                <div className="w-3 h-1.5 rounded-b-full bg-[#E6D5C3]/80" />
              ) : loadingTurn ? (
                /* Thinking small dot */
                <div className="w-1.5 h-1.5 rounded-full bg-[#E6D5C3] animate-pulse" />
              ) : (
                /* Gentle friendly neutral curve */
                <div className="w-3 h-1 rounded-b-full border-b border-[#E6D5C3]" />
              )}
            </div>

            {/* Bottom Socratic Initial Tag */}
            <div className="absolute bottom-0.5 text-[8px] font-mono tracking-widest text-[#E6D5C3]/40 uppercase">
              BUDDY
            </div>
          </div>
        </motion.div>

        {/* Live Audio Equalizer Wave when Listening (Microphone Active) */}
        {isListening && (
          <div className="flex items-center gap-1 mt-2.5">
            {[40, 75, 100, 60, 90, 45, 80, 50].map((height, i) => (
              <motion.div
                key={i}
                animate={{ height: ["4px", `${height * 0.18}px`, "4px"] }}
                transition={{
                  repeat: Infinity,
                  duration: 0.5 + (i % 3) * 0.15,
                  ease: "easeInOut",
                }}
                className="w-1 bg-emerald-600 rounded-full"
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default SocraticAvatarReplica;
