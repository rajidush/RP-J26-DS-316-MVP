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


if __name__ == "__main__":
    unittest.main()
