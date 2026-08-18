"use client";

import React, { useEffect, useRef, useState } from "react";
import { Eye, Image as ImageIcon, Mic, ScanSearch, Trash2 } from "lucide-react";

import { BACKEND_URL } from "./lib/backend";

const ANALYST = `${BACKEND_URL}/api/analyst`;

type BackendStatus = {
  ocr: string;
  asr: string;
  text: string;
  vision: string;
  capture: string;
};

type AnalystResult = {
  decision: "hate" | "not-hate";
  risk_score: number;
  category: string;
  threshold: number;
  ocr_text: string;
  transcript: string;
  escalated: boolean;
  media_deleted: boolean;
  stage1: { text_score: number; vision_score: number };
  stage2: { fused: number } | null;
  backends: BackendStatus;
  notes: string[];
  source: Record<string, boolean>;
};

type Health = {
  ready: boolean;
  backends: BackendStatus;
  notes: string[];
};

export default function AnalystPanel({
  childAge,
  onScore,
  onIntercept,
}: {
  childAge: number;
  onScore: (score: number, decision: string) => void;
  onIntercept: (hateScore: number) => void;
}) {
  const [health, setHealth] = useState<Health | null>(null);
  const [overlayText, setOverlayText] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [captureScreen, setCaptureScreen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalystResult | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch(`${ANALYST}/health`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => setHealth(data))
      .catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const clearPreview = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setImageFile(null);
    if (fileRef.current) fileRef.current.value = "";
  };

  const onPickImage = (file: File | null) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setImageFile(file);
    setPreviewUrl(file ? URL.createObjectURL(file) : null);
  };

  const analyze = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    const body = new FormData();
    body.append("child_age", String(childAge));
    body.append("overlay_text", overlayText);
    body.append("capture_screen", captureScreen ? "true" : "false");
    if (imageFile) body.append("image", imageFile);
    if (audioFile) body.append("audio", audioFile);

    try {
      const res = await fetch(`${ANALYST}/analyze`, { method: "POST", body });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Analyst is not reachable on :8000");
      }
      const data: AnalystResult = await res.json();
      setResult(data);
      onScore(data.risk_score, data.decision);
      clearPreview();
      setAudioFile(null);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Analyze failed");
    } finally {
      setLoading(false);
    }
  };

  const backends = health?.backends;

  return (
    <div className="bg-white border border-[#DDE0D0] rounded-xl p-5 shadow-sm flex flex-col gap-4" id="analyst-panel">
      <div className="flex items-center justify-between border-b border-[#DDE0D0] pb-3">
        <h3 className="font-semibold text-[#2D3025] flex items-center gap-2 text-sm uppercase tracking-wider">
          <ScanSearch className="w-4 h-4 text-[#5A5A40]" />
          Analyst — Hate Speech
        </h3>
        <span className="text-[10px] font-semibold uppercase tracking-wider text-[#6B705C] bg-[#FAF9F6] border border-[#DDE0D0] px-2 py-0.5 rounded">
          RAM buffer · then delete
        </span>
      </div>

      <p className="text-xs text-[#6B705C] leading-relaxed">
        Frame and audio stay in the local Python process until this run finishes, then they are wiped.
        The browser only keeps a preview until Analyze; the parent UI never stores the picture.
      </p>

      {backends && (
        <div className="grid grid-cols-2 gap-1.5 text-[10px] font-mono text-[#6B705C]" id="analyst-backends">
          <span>ocr: {backends.ocr}</span>
          <span>asr: {backends.asr}</span>
          <span>text: {backends.text}</span>
          <span>vision: {backends.vision}</span>
        </div>
      )}
      {!health && (
        <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
          Analyst health not reachable. Start <code>python main.py</code> on port 8000.
        </p>
      )}

      <label className="text-xs font-medium text-[#2D3025]" htmlFor="analyst-overlay">
        Overlay / chat text (works with no OCR)
      </label>
      <textarea
        id="analyst-overlay"
        value={overlayText}
        onChange={(e) => setOverlayText(e.target.value)}
        rows={2}
        placeholder="Paste Discord / game chat, or leave empty and upload a frame"
        className="w-full text-sm border border-[#DDE0D0] rounded-lg px-3 py-2 bg-[#FAF9F6] text-[#2D3025] focus:outline-none focus:ring-1 focus:ring-[#5A5A40]"
      />

      <div className="flex flex-wrap gap-2">
        <label className="cursor-pointer text-xs px-2.5 py-1.5 rounded bg-[#FAF9F6] border border-[#DDE0D0] text-[#5A5A40] hover:bg-[#E6D5C3]/40 flex items-center gap-1.5">
          <ImageIcon className="w-3.5 h-3.5" />
          Upload frame
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => onPickImage(e.target.files?.[0] ?? null)}
          />
        </label>
        <label className="cursor-pointer text-xs px-2.5 py-1.5 rounded bg-[#FAF9F6] border border-[#DDE0D0] text-[#5A5A40] hover:bg-[#E6D5C3]/40 flex items-center gap-1.5">
          <Mic className="w-3.5 h-3.5" />
          Audio clip
          <input
            type="file"
            accept="audio/*"
            className="hidden"
            onChange={(e) => setAudioFile(e.target.files?.[0] ?? null)}
          />
        </label>
        <button
          type="button"
          onClick={() => setCaptureScreen((v) => !v)}
          className={`text-xs px-2.5 py-1.5 rounded border flex items-center gap-1.5 ${
            captureScreen
              ? "bg-[#5A5A40] text-white border-[#5A5A40]"
              : "bg-[#FAF9F6] border-[#DDE0D0] text-[#5A5A40]"
          }`}
        >
          <Eye className="w-3.5 h-3.5" />
          Grab server screen
        </button>
        {(imageFile || previewUrl) && (
          <button
            type="button"
            onClick={clearPreview}
            className="text-xs px-2.5 py-1.5 rounded bg-white border border-[#DDE0D0] text-[#6B705C] flex items-center gap-1.5"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Drop preview
          </button>
        )}
      </div>

      {previewUrl && (
        <img
          src={previewUrl}
          alt="Local preview only — revoked after analyze"
          className="max-h-32 rounded-lg border border-[#DDE0D0] object-contain bg-[#FAF9F6]"
        />
      )}
      {audioFile && (
        <p className="text-[10px] text-[#6B705C]">Audio selected: {audioFile.name}</p>
      )}

      <button
        type="button"
        onClick={analyze}
        disabled={loading}
        className="w-full py-2.5 px-4 bg-[#6B4F3A] hover:bg-[#5A3F2E] disabled:opacity-60 text-white text-sm font-medium rounded-lg flex items-center justify-center gap-2"
        id="btn-analyst-analyze"
      >
        {loading ? "Analysing & wiping buffer…" : "Analyse locally"}
      </button>

      {error && (
        <p className="text-xs text-rose-700 bg-rose-50 border border-rose-200 rounded-lg px-3 py-2">{error}</p>
      )}

      {result && (
        <div className="flex flex-col gap-2 text-xs border border-[#DDE0D0] rounded-lg p-3 bg-[#FAF9F6]" id="analyst-result">
          <div className="flex items-center justify-between">
            <span className={`font-bold ${result.decision === "hate" ? "text-rose-700" : "text-emerald-700"}`}>
              {result.decision.toUpperCase()} · {result.risk_score.toFixed(2)} · {result.category}
            </span>
            <span className="text-[#6B705C]">
              {result.media_deleted ? "media deleted" : "media still held"}
            </span>
          </div>
          <p className="text-[#6B705C] font-mono">
            stage1 text {result.stage1.text_score.toFixed(2)} · vision {result.stage1.vision_score.toFixed(2)}
            {result.stage2 ? ` · fused ${result.stage2.fused.toFixed(2)}` : " · stopped stage 1"}
          </p>
          {result.ocr_text && (
            <p className="text-[#2D3025]">
              <span className="text-[#6B705C]">OCR: </span>
              {result.ocr_text}
            </p>
          )}
          {result.notes.length > 0 && (
            <p className="text-[#6B705C]">{result.notes.join(" · ")}</p>
          )}
          {result.decision === "hate" && (
            <button
              type="button"
              onClick={() => onIntercept(result.risk_score)}
              className="mt-1 w-full py-2 rounded-md bg-rose-700 hover:bg-rose-800 text-white text-xs font-medium"
            >
              Send score to Socratic intercept
            </button>
          )}
        </div>
      )}
    </div>
  );
}
