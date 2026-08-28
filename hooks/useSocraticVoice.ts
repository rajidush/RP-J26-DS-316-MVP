"use client";

import { useState, useEffect, useRef, useCallback } from "react";

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

  // Initialize SpeechSynthesis Voices
  useEffect(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      setTtsSupported(false);
      return;
    }
    setTtsSupported(true);

    const updateVoices = () => {
      const availableVoices = window.speechSynthesis.getVoices();
      if (availableVoices && availableVoices.length > 0) {
        setVoices(availableVoices);
        // Find best default English voice: prefer friendly/child-like or natural English female/male
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

  // Mouth viseme oscillator when speaking
  const startMouthAnimation = useCallback(() => {
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
    setVisemeLevel(0);
  }, []);

  // Stop speaking
  const stopSpeaking = useCallback(() => {
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    setIsSpeaking(false);
    stopMouthAnimation();
    setSpeechStatus((prev) => (prev === "speaking" ? "idle" : prev));
  }, [stopMouthAnimation]);

  // Text-To-Speech speak method
  const speak = useCallback(
    (text: string, onEndCallback?: () => void) => {
      if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
      if (isMuted || !text || !text.trim()) return;

      // Clean text: strip json braces, markdown formatting, or internal brackets
      let cleanText = text
        .replace(/\{[\s\S]*?\}/g, "")
        .replace(/```[\s\S]*?```/g, "")
        .replace(/[*_#`]/g, "")
        .trim();

      if (!cleanText) return;

      setLastSpokenText(cleanText);
      try {
        window.speechSynthesis.cancel(); // Cancel ongoing utterance

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
          startMouthAnimation();
        };

        utterance.onboundary = () => {
          // dynamic mouth variation on word boundaries
          setVisemeLevel(Math.floor(Math.random() * 3) + 1);
        };

        utterance.onend = () => {
          setIsSpeaking(false);
          stopMouthAnimation();
          setSpeechStatus("idle");
          if (onEndCallback) onEndCallback();
        };

        utterance.onerror = (e) => {
          console.warn("SpeechSynthesis error:", e);
          setIsSpeaking(false);
          stopMouthAnimation();
          setSpeechStatus("idle");
        };

        window.speechSynthesis.speak(utterance);
      } catch (err) {
        console.warn("Error calling speak:", err);
      }
    },
    [isMuted, selectedVoice, voicePitch, voiceRate, startMouthAnimation, stopMouthAnimation]
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
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
      if (animIntervalRef.current) {
        clearInterval(animIntervalRef.current);
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
