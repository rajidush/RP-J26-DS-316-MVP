"""Read live C2 Analyst runs from analyst/data/analyst.db for the C4 parent dashboard.

Read-only bridge — does not import the analyst package (avoids heavy ML deps here).
"""

from __future__ import annotations

import json
import os
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "analyst" / "data" / "analyst.db"
ANALYST_DB = Path(os.environ.get("ANALYST_DB_PATH", str(DEFAULT_DB)))
ANALYST_PANEL_URL = os.environ.get("ANALYST_PANEL_URL", "http://127.0.0.1:8765")


def _parse_json(raw: Optional[str]) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _time_str(ts: str) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%I:%M %p")
    except Exception:
        return ts[11:16] if len(ts) >= 16 else ""


def _source_from_row(row: Dict[str, Any]) -> Dict[str, bool]:
    ocr = bool((row.get("ocr_snippet") or "").strip())
    transcript = bool((row.get("transcript_snippet") or "").strip())
    mods = _parse_json(row.get("modalities_json")) or {}
    image = float(mods.get("image") or 0.0)
    audio = float(mods.get("audio") or 0.0)
    return {
        "ocr": ocr,
        "overlay": False,
        "asr": transcript or audio > 0.05,
        "vision": image > 0.05,
    }


def row_to_dashboard_run(row: Dict[str, Any]) -> Dict[str, Any]:
    category = row.get("category") or "none"
    if category == "none" and row.get("decision") == "hate":
        category = "hate_identity"
    return {
        "id": row.get("id") or "",
        "timestamp": row.get("ts") or "",
        "time_str": _time_str(str(row.get("ts") or "")),
        "child_age": int(row.get("child_age") or 10),
        "decision": row.get("decision") or "not-hate",
        "risk_score": float(row.get("score") or 0.0),
        "category": category,
        "ocr_text": row.get("ocr_snippet") or "",
        "transcript": row.get("transcript_snippet") or "",
        "overlay_text": "",
        "source": _source_from_row(row),
        "session_hint": row.get("app_exe") or "unknown",
        "app_exe": row.get("app_exe") or "unknown",
        "child_safe_summary": row.get("child_safe_summary") or "",
    }


def db_available() -> bool:
    return ANALYST_DB.is_file()


def list_runs(limit: int = 100) -> List[Dict[str, Any]]:
    if not db_available():
        return []
    limit = max(1, min(int(limit), 200))
    try:
        conn = sqlite3.connect(f"file:{ANALYST_DB}?mode=ro", uri=True, timeout=2.0)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                "SELECT id, ts, decision, category, score, child_age, "
                "ocr_snippet, transcript_snippet, app_exe, modalities_json, child_safe_summary "
                "FROM runs ORDER BY ts DESC LIMIT ?",
                (limit,),
            )
            return [row_to_dashboard_run(dict(r)) for r in cur.fetchall()]
        finally:
            conn.close()
    except Exception:
        return []


def stats_by_age() -> Dict[str, int]:
    """Hate counts keyed by child_age string (matches dashboard_store shape)."""
    if not db_available():
        return {}
    out: Dict[str, int] = {}
    try:
        conn = sqlite3.connect(f"file:{ANALYST_DB}?mode=ro", uri=True, timeout=2.0)
        try:
            cur = conn.execute(
                "SELECT child_age, COUNT(*) FROM runs WHERE decision = 'hate' "
                "GROUP BY child_age"
            )
            for age, count in cur.fetchall():
                out[str(int(age))] = int(count)
        finally:
            conn.close()
    except Exception:
        pass
    return out


def db_stats() -> Dict[str, int]:
    if not db_available():
        return {"total": 0, "hate": 0, "not_hate": 0}
    try:
        conn = sqlite3.connect(f"file:{ANALYST_DB}?mode=ro", uri=True, timeout=2.0)
        try:
            total = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            hate = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE decision = 'hate'"
            ).fetchone()[0]
            return {"total": int(total), "hate": int(hate), "not_hate": int(total) - int(hate)}
        finally:
            conn.close()
    except Exception:
        return {"total": 0, "hate": 0, "not_hate": 0}


def live_panel_status(timeout_s: float = 1.5) -> Dict[str, Any]:
    url = f"{ANALYST_PANEL_URL.rstrip('/')}/api/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return {
            "online": True,
            "panel_url": ANALYST_PANEL_URL,
            "capturing": body.get("capturing"),
            "protection": body.get("protection"),
            "stats": body.get("stats"),
            "last_error": body.get("last_error") or "",
        }
    except Exception as exc:
        return {
            "online": False,
            "panel_url": ANALYST_PANEL_URL,
            "error": str(exc)[:120],
        }


def latest_run() -> Optional[Dict[str, Any]]:
    runs = list_runs(limit=1)
    return runs[0] if runs else None


def hate_speech_score_from_latest(run: Optional[Dict[str, Any]] = None) -> float:
    """Score for C3 perception trigger. Only elevated when decision is hate."""
    row = run if run is not None else latest_run()
    if not row:
        return 0.0
    if row.get("decision") != "hate":
        return 0.0
    try:
        return float(row.get("risk_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def get_merged_analyst_runs(limit: int = 100) -> Dict[str, Any]:
    runs = list_runs(limit=limit)
    latest = runs[0] if runs else None
    return {
        "analyst_runs": runs,
        "latest_run": latest,
        "hate_speech_score": hate_speech_score_from_latest(latest),
        "analyst_db_path": str(ANALYST_DB),
        "analyst_db_available": db_available(),
        "analyst_stats": db_stats(),
        "hate_speech_by_age": stats_by_age(),
        "analyst_status": live_panel_status(),
    }
