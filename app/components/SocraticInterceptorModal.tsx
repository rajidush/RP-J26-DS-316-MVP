"use client";

import React, { useRef, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import { 
  Shield, 
  Lock, 
  ChevronRight, 
  Volume2, 
  Mic, 
  Send, 
  RefreshCw, 
  CheckCircle, 
  ArrowRight 
} from "lucide-react";
import { SocraticAvatarReplica } from "./SocraticAvatarReplica";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface SocraticInterceptorModalProps {
  isOpen: boolean;
  activeThreatType: string;
  currentPhase: string;
  chatHistory: ChatMessage[];
  loadingTurn: boolean;
  isCompleted: boolean;
  userInput: string;
  setUserInput: (val: string) => void;
  onSendDialogue: (e?: React.FormEvent) => void;
  voice: {
    isSpeaking: boolean;
    isListening: boolean;
    visemeLevel: number;
    isMuted: boolean;
    sttSupported: boolean;
    speak: (text: string) => void;
    stopSpeaking: () => void;
    startListening: (onResult: (text: string) => void) => void;
    stopListening: () => void;
  };
  childAge: number;
  childEmotion: string;
  onRedirect: () => void;
}

export const SocraticInterceptorModal: React.FC<SocraticInterceptorModalProps> = ({
  isOpen,
  activeThreatType,
  currentPhase,
  chatHistory,
  loadingTurn,
  isCompleted,
  userInput,
  setUserInput,
  onSendDialogue,
  voice,
  childAge,
  childEmotion,
  onRedirect,
}) => {
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen) {
      chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [chatHistory, loadingTurn, isOpen]);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-[#2D3025]/80 backdrop-blur-xs flex flex-col items-center justify-center p-4 md:p-6 z-50 overflow-hidden"
          id="socratic-buddy-interceptor"
        >
          <motion.div 
            initial={{ scale: 0.95, y: 15 }}
            animate={{ scale: 1, y: 0 }}
            exit={{ scale: 0.95, y: 15 }}
            className="bg-white border border-[#DDE0D0] rounded-2xl w-full max-w-lg shadow-2xl flex flex-col overflow-hidden h-[95%] max-h-[680px] relative"
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
            <div className="bg-[#FAF9F6] border-b border-[#DDE0D0] px-4 py-2.5 flex items-center justify-between text-[11px] shrink-0 text-[#6B705C]" id="dialogue-timeline">
              <span className="text-[#6B705C] font-semibold uppercase tracking-wider text-[10px]">State Machine progression:</span>
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

            {/* Interactive Socratic Hero Avatar Stage */}
            <SocraticAvatarReplica
              isSpeaking={voice.isSpeaking}
              isListening={voice.isListening}
              visemeLevel={voice.visemeLevel}
              childAge={childAge}
              childEmotion={childEmotion}
              loadingTurn={loadingTurn}
              isCompleted={isCompleted}
              isMuted={voice.isMuted}
            />

            {/* Interactive Chat Board area */}
            <div className="flex-1 p-4 overflow-y-auto flex flex-col gap-4 bg-[#FAF9F6]" id="buddy-chat-history">
              
              {/* Socratic Avatar Greeting Bubble */}
              <div className="flex items-start gap-3" id="buddy-avatar-greeting">
                <div className="w-10 h-10 rounded-xl bg-[#5A5A40] flex items-center justify-center font-bold text-white shadow-xs shrink-0 border border-[#5A5A40] text-xs">
                  SB
                </div>
                <div className="bg-white border border-[#DDE0D0] rounded-xl p-3.5 max-w-[85%] text-[#2D3025] text-xs shadow-xs leading-relaxed relative group">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-bold text-[#5A5A40]">Socratic Buddy</span>
                    <button
                      type="button"
                      onClick={() => voice.speak("Hi! Socratic Buddy here. Socratic Buddy always protects minors, so I've covered the screen to keep you safe. Don't worry, you aren't in any trouble. Let's talk this through together.")}
                      title="Replay greeting voice"
                      className="text-[#6B705C] hover:text-[#5A5A40] p-1 rounded hover:bg-[#FAF9F6] transition flex items-center gap-1 text-[10px]"
                    >
                      <Volume2 className="w-3 h-3 text-[#5A5A40]" />
                      <span className="hidden sm:inline">Play</span>
                    </button>
                  </div>
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
                      <div className="flex items-center justify-between mb-1">
                        <span className={`font-bold ${item.role === "user" ? "text-[#5A5A40]" : "text-[#5A5A40]"}`}>
                          {item.role === "user" ? "My Response" : "Socratic Buddy"}
                        </span>
                        {item.role !== "user" && textContent && (
                          <button
                            type="button"
                            onClick={() => voice.speak(textContent)}
                            title="Play this message voice"
                            className="text-[#6B705C] hover:text-[#5A5A40] p-1 rounded hover:bg-[#FAF9F6] transition flex items-center gap-1 text-[10px]"
                          >
                            <Volume2 className="w-3 h-3 text-[#5A5A40]" />
                            <span className="hidden sm:inline">Play</span>
                          </button>
                        )}
                      </div>
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
                <div className="bg-white border border-emerald-200 p-5 rounded-2xl max-w-sm flex flex-col items-center gap-4 shadow-md">
                  <CheckCircle className="w-16 h-16 text-emerald-600" />
                  <div>
                    <h3 className="text-base font-bold text-[#2D3025]">Safety Agreement Confirmed!</h3>
                    <p className="text-xs text-[#6B705C] mt-1.5 leading-relaxed">
                      You did an amazing job talking this through. We have closed that unsafe window. Let&apos;s redirect to a fun learning site together!
                    </p>
                  </div>
                  <button
                    onClick={onRedirect}
                    className="w-full py-2.5 px-4 bg-[#5A5A40] hover:bg-[#454530] text-white font-bold rounded-lg text-xs flex items-center justify-center gap-2 transition duration-200 shadow-sm cursor-pointer"
                    id="btn-redirect-fun"
                  >
                    Redirect to Currie&apos;s Sandbox <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </motion.div>
            )}

            {/* Input Area */}
            <form 
              onSubmit={onSendDialogue}
              className="bg-white border-t border-[#DDE0D0] p-3.5 flex flex-col gap-2 shrink-0"
              id="chat-input-form"
            >
              {voice.isListening && (
                <div className="flex items-center justify-between px-3 py-1.5 bg-emerald-50 border border-emerald-200 rounded-lg text-emerald-800 text-xs animate-pulse">
                  <span className="flex items-center gap-2 font-medium">
                    <Mic className="w-3.5 h-3.5 text-emerald-600 animate-bounce" />
                    Listening to child voice... speak your reply now
                  </span>
                  <button
                    type="button"
                    onClick={() => voice.stopListening()}
                    className="text-[10px] font-bold uppercase underline hover:text-emerald-950"
                  >
                    Stop
                  </button>
                </div>
              )}

              <div className="flex items-center gap-2">
                {voice.sttSupported && (
                  <button
                    type="button"
                    onClick={() => {
                      if (voice.isListening) {
                        voice.stopListening();
                      } else {
                        voice.startListening((spokenText) => {
                          setUserInput(spokenText);
                        });
                      }
                    }}
                    disabled={loadingTurn || isCompleted}
                    title={voice.isListening ? "Listening... click to stop" : "Speak your response (Mic)"}
                    className={`p-2.5 rounded-lg transition flex items-center justify-center shrink-0 disabled:opacity-50 disabled:cursor-not-allowed ${
                      voice.isListening
                        ? "bg-emerald-600 hover:bg-emerald-700 text-white animate-pulse shadow-md"
                        : "bg-[#FAF9F6] border border-[#DDE0D0] hover:bg-[#E6D5C3]/40 text-[#5A5A40]"
                    }`}
                    id="btn-voice-mic"
                  >
                    <Mic className="w-4 h-4" />
                  </button>
                )}

                <input 
                  type="text" 
                  placeholder={voice.isListening ? "Listening to your voice..." : "Type or speak your response to Socratic Buddy..."}
                  value={userInput}
                  onChange={(e) => setUserInput(e.target.value)}
                  className="flex-1 bg-[#FAF9F6] border border-[#DDE0D0] rounded-lg py-2.5 px-3.5 text-xs text-[#2D3025] focus:outline-none focus:border-[#5A5A40] focus:ring-1 focus:ring-[#5A5A40]"
                  disabled={loadingTurn || isCompleted}
                  id="input-chat-text"
                />
                <button
                  type="submit"
                  className="p-2.5 rounded-lg bg-[#5A5A40] hover:bg-[#454530] text-white transition flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
                  disabled={!userInput.trim() || loadingTurn || isCompleted}
                  id="btn-send-chat"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </form>

          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default SocraticInterceptorModal;
