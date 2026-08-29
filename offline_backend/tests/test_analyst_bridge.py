"""Tests for C2 → C4 analyst bridge."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import analyst_bridge


class AnalystBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "analyst.db"
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript(
            """
            CREATE TABLE runs (
                id TEXT PRIMARY KEY, ts TEXT, decision TEXT, category TEXT,
                score REAL, child_age INTEGER, ocr_snippet TEXT, transcript_snippet TEXT,
                app_exe TEXT, modalities_json TEXT, child_safe_summary TEXT
            );
            INSERT INTO runs VALUES (
                'r1', '2026-08-25T10:00:00+05:30', 'hate', 'threat', 0.91, 10,
                'you should kys', '', 'chrome.exe',
                '{"text":0.9,"image":0.0,"audio":0.0}', 'Threat detected.'
            );
            """
        )
        conn.commit()
        conn.close()
        self._orig = analyst_bridge.ANALYST_DB
        analyst_bridge.ANALYST_DB = self.db_path

    def tearDown(self):
        analyst_bridge.ANALYST_DB = self._orig
        self.tmp.cleanup()

    def test_list_runs_maps_fields(self):
        runs = analyst_bridge.list_runs(limit=5)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["decision"], "hate")
        self.assertEqual(runs[0]["risk_score"], 0.91)
        self.assertTrue(runs[0]["source"]["ocr"])
        self.assertEqual(runs[0]["app_exe"], "chrome.exe")

    def test_hate_score_only_on_hate(self):
        score = analyst_bridge.hate_speech_score_from_latest()
        self.assertGreater(score, 0.8)
        runs = analyst_bridge.list_runs(limit=1)
        runs[0]["decision"] = "not-hate"
        self.assertEqual(analyst_bridge.hate_speech_score_from_latest(runs[0]), 0.0)

    def test_merged_includes_latest(self):
        merged = analyst_bridge.get_merged_analyst_runs(limit=5)
        self.assertIn("latest_run", merged)
        self.assertIn("hate_speech_score", merged)
        self.assertIsNotNone(merged["latest_run"])

    def test_older_schema_without_envelope_still_reads(self):
        """setUp builds a table with no envelope_json / recommended_action.
        The bridge must fall back to the age rule rather than raising."""
        verdict = analyst_bridge.hate_verdict()
        self.assertTrue(verdict["detected"])
        self.assertEqual(verdict["persona_threshold"], 0.55)  # child_age 10


class HateVerdictTests(unittest.TestCase):
    """C3 must act on the Analyst's decision, not re-threshold its score."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "analyst.db"
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript(
            """
            CREATE TABLE runs (
                id TEXT PRIMARY KEY, ts TEXT, decision TEXT, category TEXT,
                score REAL, child_age INTEGER, ocr_snippet TEXT,
                transcript_snippet TEXT, app_exe TEXT, modalities_json TEXT,
                child_safe_summary TEXT, recommended_action TEXT,
                envelope_json TEXT
            );
            """
        )
        conn.commit()
        conn.close()
        self._orig = analyst_bridge.ANALYST_DB
        analyst_bridge.ANALYST_DB = self.db_path

    def tearDown(self):
        analyst_bridge.ANALYST_DB = self._orig
        self.tmp.cleanup()

    def _insert(self, run_id, ts, decision, category, score, age, theta):
        envelope = json.dumps(
            {"topic": "hate.detected", "payload": {"persona_threshold": theta}}
        )
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id, ts, decision, category, score, age, "", "", "chrome.exe",
                '{"text":1.0,"image":0.0,"audio":0.0}', "Someone is being unkind.",
                "blur_region", envelope,
            ),
        )
        conn.commit()
        conn.close()

    def test_bullying_below_080_is_still_a_detection(self):
        """The regression this fix exists for: the demo bullying poster scores
        0.64, which clears the 8-10 persona threshold of 0.55. The old C3 gate
        of 0.80 discarded it."""
        self._insert("r-bully", "2026-08-26T10:00:00+05:30", "hate", "bullying", 0.64, 10, 0.55)
        verdict = analyst_bridge.hate_verdict()
        self.assertTrue(verdict["detected"])
        self.assertEqual(verdict["category"], "bullying")
        self.assertEqual(verdict["score"], 0.64)
        self.assertEqual(verdict["persona_threshold"], 0.55)
        self.assertLess(verdict["score"], 0.80)  # would have been dropped before

    def test_not_hate_is_never_a_detection(self):
        self._insert("r-clear", "2026-08-26T10:00:00+05:30", "not-hate", "none", 0.18, 10, 0.55)
        verdict = analyst_bridge.hate_verdict()
        self.assertFalse(verdict["detected"])
        self.assertEqual(verdict["score"], 0.0)
        self.assertEqual(verdict["category"], "none")

    def test_persona_threshold_comes_from_the_envelope(self):
        self._insert("r-teen", "2026-08-26T10:00:00+05:30", "hate", "threat", 0.88, 15, 0.75)
        verdict = analyst_bridge.hate_verdict()
        self.assertEqual(verdict["persona_threshold"], 0.75)
        self.assertEqual(verdict["child_age"], 15)

    def test_run_id_pins_the_exact_detection(self):
        """A newer run must not hijack an interception raised for an older one."""
        self._insert("r-old", "2026-08-26T10:00:00+05:30", "hate", "bullying", 0.64, 10, 0.55)
        self._insert("r-new", "2026-08-26T10:05:00+05:30", "not-hate", "none", 0.08, 10, 0.55)
        self.assertFalse(analyst_bridge.hate_verdict()["detected"])
        pinned = analyst_bridge.hate_verdict(run_id="r-old")
        self.assertTrue(pinned["detected"])
        self.assertEqual(pinned["run_id"], "r-old")

    def test_missing_run_id_is_not_a_detection(self):
        verdict = analyst_bridge.hate_verdict(run_id="does-not-exist")
        self.assertFalse(verdict["detected"])
        self.assertEqual(verdict["run_id"], "")

    def test_recommended_action_reaches_c3(self):
        self._insert("r-act", "2026-08-26T10:00:00+05:30", "hate", "threat", 0.91, 10, 0.55)
        self.assertEqual(analyst_bridge.hate_verdict()["recommended_action"], "blur_region")


if __name__ == "__main__":
    unittest.main()
