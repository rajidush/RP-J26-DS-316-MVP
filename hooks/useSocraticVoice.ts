"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { BACKEND_URL } from "@/app/lib/backend";

export interface SocraticVoiceOptions {
  childAge: number;
  autoSpeak?: boolean;
}

export function useSocraticVoice({ childAge }: SocraticVoiceOptions) {
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<SpeechSynthesisVoice | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [voiceRate, setVoiceRate] = useState(childAge <= 10 ? 0.95 : 1.0);
  const [voicePitch, setVoicePitch] = useState(childAge <= 10 ? 1.25 : 1.0);
  const [sttSupported, setSttSupported] = useState(false);
  const [ttsSupported, setTtsSupported] = useState(false);
  const [visemeLevel, setVisemeLevel] = useState<number>(0); // 0 (closed) to 3 (wide open) for mouth animation
  const [speechStatus, setSpeechStatus] = useState<"idle" | "speaking" | "listening" | "thinking">("idle");
  const [lastSpokenText, setLastSpokenText] = useState<string>("");

  const recognitionRef = useRef<any>(null);
  const animIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceNodeRef = useRef<MediaElementAudioSourceNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const currentBlobUrlRef = useRef<string | null>(null);

  // Initialize SpeechSynthesis Voices (as offline fallback)
  useEffect(() => {
    if (typeof window === "undefined") return;
    if ("speechSynthesis" in window) {
      setTtsSupported(true);

      const updateVoices = () => {
        const availableVoices = window.speechSynthesis.getVoices();
        if (availableVoices && availableVoices.length > 0) {
          setVoices(availableVoices);
          const englishVoices = availableVoices.filter((v) =>
            v.lang.toLowerCase().startsWith("en")
          );
          const preferred =
            englishVoices.find(
              (v) =>
                v.name.toLowerCase().includes("natural") ||
                v.name.toLowerCase().includes("friendly") ||
                v.name.toLowerCase().includes("junior") ||
                v.name.toLowerCase().includes("zira") ||
                v.name.toLowerCase().includes("samantha") ||
                v.name.toLowerCase().includes("google us english")
            ) ||
            englishVoices[0] ||
            availableVoices[0];

          setSelectedVoice(preferred || null);
        }
      };

      updateVoices();
      window.speechSynthesis.onvoiceschanged = updateVoices;

      return () => {
        if (typeof window !== "undefined" && "speechSynthesis" in window) {
          window.speechSynthesis.onvoiceschanged = null;
        }
      };
    } else {
      setTtsSupported(false);
    }
  }, []);

  // Update pitch & rate automatically when childAge changes
  useEffect(() => {
    if (childAge <= 10) {
      setVoicePitch(1.25);
      setVoiceRate(0.95);
    } else {
      setVoicePitch(1.0);
      setVoiceRate(1.0);
    }
  }, [childAge]);

  // Setup SpeechRecognition (STT)
  useEffect(() => {
    if (typeof window === "undefined") return;

    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (SpeechRecognition) {
      setSttSupported(true);
      try {
        const recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.lang = "en-US";
        recognitionRef.current = recognition;
      } catch (e) {
        console.warn("SpeechRecognition init error:", e);
      }
    } else {
      setSttSupported(false);
    }

    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.abort();
        } catch {
          // ignore
        }
      }
    };
  }, []);

  // Timer-based mouth viseme oscillator (for fallback or synthetic cadence)
  const startTimerMouthAnimation = useCallback(() => {
    if (animIntervalRef.current) clearInterval(animIntervalRef.current);
    let step = 0;
    const levels = [1, 2, 3, 2, 1, 0, 2, 3, 1, 2, 0];
    animIntervalRef.current = setInterval(() => {
      setVisemeLevel(levels[step % levels.length]);
      step++;
    }, 120);
  }, []);

  const stopMouthAnimation = useCallback(() => {
    if (animIntervalRef.current) {
      clearInterval(animIntervalRef.current);
      animIntervalRef.current = null;
    }
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    setVisemeLevel(0);
  }, []);

  // Acoustic Frequency Analyzer Loop using Web Audio API
  const startAcousticAnalysis = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) {
      startTimerMouthAnimation();
      return;
    }

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const checkAudio = () => {
      analyser.getByteFrequencyData(dataArray);

      // Compute average energy across vocal frequency bins (approx 100Hz - 4kHz)
      let sum = 0;
      const count = Math.min(bufferLength, 32);
      for (let i = 0; i < count; i++) {
        sum += dataArray[i];
      }
      const avg = sum / count;

      // Map acoustic energy directly to viseme mouth openness levels (0 to 3)
      if (avg > 70) {
        setVisemeLevel(3); // Wide open
      } else if (avg > 40) {
        setVisemeLevel(2); // Medium open
      } else if (avg > 15) {
        setVisemeLevel(1); // Slight open
      } else {
        setVisemeLevel(0); // Closed
      }

      rafRef.current = requestAnimationFrame(checkAudio);
    };

    rafRef.current = requestAnimationFrame(checkAudio);
  }, [startTimerMouthAnimation]);

  // Stop speaking and audio playback
  const stopSpeaking = useCallback(() => {
    // 1. Stop HTML5 Audio if playing
    if (audioRef.current) {
      try {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
      } catch {
        // ignore
      }
    }
    // 2. Revoke blob URL
    if (currentBlobUrlRef.current) {
      try {
        URL.revokeObjectURL(currentBlobUrlRef.current);
      } catch {
        // ignore
      }
      currentBlobUrlRef.current = null;
    }
    // 3. Stop native SpeechSynthesis
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        // ignore
      }
    }
    setIsSpeaking(false);
    stopMouthAnimation();
    setSpeechStatus((prev) => (prev === "speaking" ? "idle" : prev));
  }, [stopMouthAnimation]);

  // Fallback to client-side browser SpeechSynthesis
  const speakWithBrowserFallback = useCallback(
    (cleanText: string, onEndCallback?: () => void) => {
      if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
      try {
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(cleanText);
        if (selectedVoice) {
          utterance.voice = selectedVoice;
        }
        utterance.pitch = voicePitch;
        utterance.rate = voiceRate;
        utterance.volume = 1.0;

        utterance.onstart = () => {
          setIsSpeaking(true);
          setSpeechStatus("speaking");
          startTimerMouthAnimation();
        };

        utterance.onboundary = () => {
          setVisemeLevel(Math.floor(Math.random() * 3) + 1);
        };

        utterance.onend = () => {
          setIsSpeaking(false);
          stopMouthAnimation();
          setSpeechStatus("idle");
          if (onEndCallback) onEndCallback();
        };

        utterance.onerror = () => {
          setIsSpeaking(false);
          stopMouthAnimation();
          setSpeechStatus("idle");
        };

        window.speechSynthesis.speak(utterance);
      } catch (err) {
        console.warn("Fallback speech error:", err);
        setIsSpeaking(false);
        stopMouthAnimation();
        setSpeechStatus("idle");
      }
    },
    [selectedVoice, voicePitch, voiceRate, startTimerMouthAnimation, stopMouthAnimation]
  );

  // Main Speech function: Tries Edge Neural TTS endpoint first, falls back to Web Speech API
  const speak = useCallback(
    async (text: string, onEndCallback?: () => void) => {
      if (isMuted || !text || !text.trim()) return;

      // Clean text: strip json braces, markdown formatting, or internal bracket tokens
      let cleanText = text
        .replace(/\{[\s\S]*?\}/g, "")
        .replace(/```[\s\S]*?```/g, "")
        .replace(/[*_#`~>\[\]]/g, " ")
        .replace(/\s+/g, " ")
        .trim();

      if (!cleanText) return;

      setLastSpokenText(cleanText);
      stopSpeaking();

      // Attempt Edge Neural TTS from FastAPI backend
      try {
        const response = await fetch(`${BACKEND_URL}/api/tts/synthesize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: cleanText,
            child_age: childAge,
          }),
        });

        if (!response.ok) {
          throw new Error(`TTS backend returned ${response.status}`);
        }

        const audioBlob = await response.blob();
        if (!audioBlob || audioBlob.size === 0) {
          throw new Error("Empty audio response from TTS engine");
        }

        const blobUrl = URL.createObjectURL(audioBlob);
        currentBlobUrlRef.current = blobUrl;

        // Initialize or recycle Audio Element
        if (!audioRef.current) {
          const audio = new Audio();
          audio.crossOrigin = "anonymous";
          audioRef.current = audio;

          // Connect to Web Audio API for real-time acoustic mouth analysis
          try {
            const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
            if (AudioCtx) {
              const audioCtx = new AudioCtx();
              const analyser = audioCtx.createAnalyser();
              analyser.fftSize = 64;
              analyser.smoothingTimeConstant = 0.5;

              const source = audioCtx.createMediaElementSource(audio);
              source.connect(analyser);
              analyser.connect(audioCtx.destination);

              audioContextRef.current = audioCtx;
              analyserRef.current = analyser;
              sourceNodeRef.current = source;
            }
          } catch (audioCtxErr) {
            console.warn("Web Audio API analyser init warning:", audioCtxErr);
          }
        }

        const audio = audioRef.current;
        audio.src = blobUrl;

        audio.onplay = () => {
          // Resume AudioContext if suspended (browser autoplay policy)
          if (audioContextRef.current && audioContextRef.current.state === "suspended") {
            audioContextRef.current.resume().catch(() => {});
          }
          setIsSpeaking(true);
          setSpeechStatus("speaking");
          startAcousticAnalysis();
        };

        audio.onended = () => {
          setIsSpeaking(false);
          stopMouthAnimation();
          setSpeechStatus("idle");
          if (currentBlobUrlRef.current) {
            URL.revokeObjectURL(currentBlobUrlRef.current);
            currentBlobUrlRef.current = null;
          }
          if (onEndCallback) onEndCallback();
        };

        audio.onerror = (e) => {
          console.warn("Audio playback error, falling back to Web Speech:", e);
          stopMouthAnimation();
          speakWithBrowserFallback(cleanText, onEndCallback);
        };

        await audio.play();
      } catch (err) {
        console.warn("Neural TTS backend failed or offline, using browser fallback:", err);
        speakWithBrowserFallback(cleanText, onEndCallback);
      }
    },
    [isMuted, childAge, stopSpeaking, startAcousticAnalysis, stopMouthAnimation, speakWithBrowserFallback]
  );

  // Replay speech
  const replayLastSpeech = useCallback(() => {
    if (lastSpokenText) {
      speak(lastSpokenText);
    }
  }, [lastSpokenText, speak]);

  // Toggle Mute
  const toggleMute = useCallback(() => {
    setIsMuted((prev) => {
      const next = !prev;
      if (next) {
        stopSpeaking();
      }
      return next;
    });
  }, [stopSpeaking]);

  // Speech-To-Text (Microphone input)
  const startListening = useCallback(
    (onResult: (text: string) => void, onEnd?: () => void) => {
      if (!recognitionRef.current) return;

      // If Buddy is currently speaking, stop so mic doesn't catch it
      stopSpeaking();

      try {
        const recognition = recognitionRef.current;
        recognition.onstart = () => {
          setIsListening(true);
          setSpeechStatus("listening");
        };

        recognition.onresult = (event: any) => {
          let currentTranscript = "";
          for (let i = event.resultIndex; i < event.results.length; i++) {
            currentTranscript += event.results[i][0].transcript;
          }
          if (currentTranscript.trim()) {
            onResult(currentTranscript.trim());
          }
        };

        recognition.onerror = (event: any) => {
          console.warn("SpeechRecognition error:", event.error);
          setIsListening(false);
          setSpeechStatus("idle");
          if (onEnd) onEnd();
        };

        recognition.onend = () => {
          setIsListening(false);
          setSpeechStatus("idle");
          if (onEnd) onEnd();
        };

        recognition.start();
      } catch (err) {
        console.warn("Failed to start speech recognition:", err);
        setIsListening(false);
        setSpeechStatus("idle");
      }
    },
    [stopSpeaking]
  );

  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch {
        // ignore
      }
    }
    setIsListening(false);
    setSpeechStatus("idle");
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        try {
          audioRef.current.pause();
        } catch {
          // ignore
        }
      }
      if (currentBlobUrlRef.current) {
        try {
          URL.revokeObjectURL(currentBlobUrlRef.current);
        } catch {
          // ignore
        }
      }
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        try {
          window.speechSynthesis.cancel();
        } catch {
          // ignore
        }
      }
      if (animIntervalRef.current) {
        clearInterval(animIntervalRef.current);
      }
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, []);

  return {
    voices,
    selectedVoice,
    setSelectedVoice,
    isSpeaking,
    isListening,
    isMuted,
    toggleMute,
    voiceRate,
    setVoiceRate,
    voicePitch,
    setVoicePitch,
    visemeLevel,
    speechStatus,
    setSpeechStatus,
    sttSupported,
    ttsSupported,
    speak,
    stopSpeaking,
    replayLastSpeech,
    startListening,
    stopListening,
    lastSpokenText,
  };
}
