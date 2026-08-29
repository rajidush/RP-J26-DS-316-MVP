"""Analyst SQLite store — detection events only (no raw media files)."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_DB = DATA_DIR / "analyst.db"

# Retention. Blurred thumbnails are the most sensitive thing we keep, so they
# expire first and fastest; the numeric/telemetry row lives longer for the
# dashboard and evaluation. Override with ANALYST_THUMB_TTL_H /
# ANALYST_RUN_TTL_DAYS / ANALYST_MAX_RUNS.
THUMB_TTL_HOURS = int(os.environ.get("ANALYST_THUMB_TTL_H", "24"))
RUN_TTL_DAYS = int(os.environ.get("ANALYST_RUN_TTL_DAYS", "30"))
MAX_RUNS = int(os.environ.get("ANALYST_MAX_RUNS", "5000"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    decision TEXT NOT NULL,
    category TEXT,
    score REAL,
    child_age INTEGER,
    ocr_snippet TEXT,
    transcript_snippet TEXT,
    image_caption TEXT,
    image_text TEXT,
    lexicon_hits_json TEXT,
    stage1_json TEXT,
    stage2_json TEXT,
    backends_json TEXT,
    latency_json TEXT,
    notes_json TEXT,
    app_exe TEXT,
    modalities_json TEXT,
    child_safe_summary TEXT,
    recommended_action TEXT,
    thumb_jpeg BLOB,
    envelope_json TEXT,
    trace_json TEXT,
    evidence_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(ts DESC);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


class AnalystStore:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            # Overwrite deleted content instead of leaving it in free pages.
            # Without this, a "cleared" screen thumbnail is still carvable out
            # of the database file.
            conn.execute("PRAGMA secure_delete=ON")
        except Exception:
            pass
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(_SCHEMA)
                self._migrate(conn)
                conn.commit()
            finally:
                conn.close()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        if "trace_json" not in cols:
            conn.execute("ALTER TABLE runs ADD COLUMN trace_json TEXT")
        # What the vision branch read, so the log view can answer "what was in
        # that picture" after the thumbnail has expired.
        if "image_caption" not in cols:
            conn.execute("ALTER TABLE runs ADD COLUMN image_caption TEXT")
        if "image_text" not in cols:
            conn.execute("ALTER TABLE runs ADD COLUMN image_text TEXT")
        # Why the score is what it is: which detector fired, whether the framing
        # guard discounted it, whether the image branch was allowed to count.
        # One JSON column rather than six, so adding a signal is not a migration.
        if "evidence_json" not in cols:
            conn.execute("ALTER TABLE runs ADD COLUMN evidence_json TEXT")

    def insert_run(self, row: Dict[str, Any]) -> str:
        run_id = row.get("id") or _new_id()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO runs (
                        id, ts, decision, category, score, child_age,
                        ocr_snippet, transcript_snippet, image_caption, image_text, lexicon_hits_json,
                        stage1_json, stage2_json, backends_json, latency_json, notes_json,
                        app_exe, modalities_json, child_safe_summary, recommended_action,
                        thumb_jpeg, envelope_json, trace_json, evidence_json
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        row.get("ts") or _now_iso(),
                        row["decision"],
                        row.get("category"),
                        row.get("score"),
                        row.get("child_age"),
                        row.get("ocr_snippet") or "",
                        row.get("transcript_snippet") or "",
                        row.get("image_caption") or "",
                        row.get("image_text") or "",
                        _json(row.get("lexicon_hits")),
                        _json(row.get("stage1")),
                        _json(row.get("stage2")),
                        _json(row.get("backends")),
                        _json(row.get("latency_ms")),
                        _json(row.get("notes")),
                        row.get("app_exe") or "unknown",
                        _json(row.get("modalities")),
                        row.get("child_safe_summary") or "",
                        row.get("recommended_action") or "",
                        row.get("thumb_jpeg"),
                        _json(row.get("envelope")),
                        _json(row.get("trace")),
                        _json(row.get("evidence")),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        self._maybe_prune()
        return run_id

    def _maybe_prune(self) -> None:
        """Cheap amortised cleanup: run every 25th insert."""
        self._since_prune = getattr(self, "_since_prune", 0) + 1
        if self._since_prune < 25:
            return
        self._since_prune = 0
        try:
            self.prune()
        except Exception:
            pass

    def prune(
        self,
        *,
        thumb_ttl_hours: Optional[int] = None,
        run_ttl_days: Optional[int] = None,
        max_runs: Optional[int] = None,
    ) -> Dict[str, int]:
        """Drop expired screen thumbnails and old rows. Returns what it removed."""
        th = THUMB_TTL_HOURS if thumb_ttl_hours is None else thumb_ttl_hours
        rd = RUN_TTL_DAYS if run_ttl_days is None else run_ttl_days
        mx = MAX_RUNS if max_runs is None else max_runs
        now = datetime.now(timezone.utc)
        thumb_cut = (now - timedelta(hours=th)).astimezone().isoformat(timespec="milliseconds")
        run_cut = (now - timedelta(days=rd)).astimezone().isoformat(timespec="milliseconds")

        with self._lock:
            conn = self._connect()
            try:
                # 1. blank expired thumbnails but keep the telemetry row
                cur = conn.execute(
                    "UPDATE runs SET thumb_jpeg = NULL "
                    "WHERE thumb_jpeg IS NOT NULL AND ts < ?",
                    (thumb_cut,),
                )
                thumbs = cur.rowcount or 0
                # 2. delete rows past the run TTL
                cur = conn.execute("DELETE FROM runs WHERE ts < ?", (run_cut,))
                rows = cur.rowcount or 0
                # 3. enforce a hard ceiling, oldest first
                cur = conn.execute(
                    "DELETE FROM runs WHERE id IN ("
                    "  SELECT id FROM runs ORDER BY ts DESC LIMIT -1 OFFSET ?"
                    ")",
                    (mx,),
                )
                rows += cur.rowcount or 0
                conn.commit()
                if thumbs or rows:
                    conn.execute("VACUUM")
            finally:
                conn.close()
        return {"thumbs_cleared": thumbs, "rows_deleted": rows}

    def purge_all(self) -> int:
        """Operator panic button: delete every recorded run."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("DELETE FROM runs")
                conn.commit()
                conn.execute("VACUUM")
                return cur.rowcount or 0
            finally:
                conn.close()

    def list_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT * FROM runs ORDER BY ts DESC LIMIT ?",
                    (limit,),
                )
                return [_row_to_dict(r) for r in cur.fetchall()]
            finally:
                conn.close()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
                row = cur.fetchone()
                return _row_to_dict(row) if row else None
            finally:
                conn.close()

    def latest_run(self) -> Optional[Dict[str, Any]]:
        rows = self.list_runs(limit=1)
        return rows[0] if rows else None

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                total = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
                hate = conn.execute(
                    "SELECT COUNT(*) FROM runs WHERE decision = 'hate'"
                ).fetchone()[0]
                return {"total": int(total), "hate": int(hate), "not_hate": int(total) - int(hate)}
            finally:
                conn.close()


def _new_id() -> str:
    from uuid import uuid4

    return uuid4().hex


def _json(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _parse_json(raw: Optional[str]) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["lexicon_hits"] = _parse_json(d.pop("lexicon_hits_json", None)) or []
    d["stage1"] = _parse_json(d.pop("stage1_json", None)) or {}
    d["stage2"] = _parse_json(d.pop("stage2_json", None))
    d["backends"] = _parse_json(d.pop("backends_json", None)) or {}
    d["latency_ms"] = _parse_json(d.pop("latency_json", None)) or {}
    d["notes"] = _parse_json(d.pop("notes_json", None)) or []
    d["modalities"] = _parse_json(d.pop("modalities_json", None)) or {}
    d["envelope"] = _parse_json(d.pop("envelope_json", None))
    d["trace"] = _parse_json(d.pop("trace_json", None))
    d["evidence"] = _parse_json(d.pop("evidence_json", None)) or {}
    # Keep thumb_jpeg as bytes; API layer may base64 it
    return d
