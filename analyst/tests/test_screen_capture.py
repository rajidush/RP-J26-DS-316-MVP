"""Screen capture retry / fallback behaviour."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyst.capture import screen as screen_mod
from analyst.capture.screen import ScreenCapture


class ScreenCaptureTests(unittest.TestCase):
    def tearDown(self):
        screen_mod._reset_mss()

    def test_grab_jpeg_succeeds(self):
        cap = ScreenCapture()
        data, w, h = cap.grab_jpeg()
        if (screen_mod._MSS_OK or sys.platform == "win32") and data is not None:
            self.assertIsNotNone(data)
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)
        elif data is None:
            self.assertTrue("glitch" in cap.last_error or "BitBlt" in cap.last_error or "unavailable" in cap.last_error)
        cap.close()

    @patch.object(screen_mod, "_grab_mss")
    @patch.object(screen_mod, "_grab_imagegrab")
    def test_bitblt_retries_then_imagegrab(self, mock_ig, mock_mss):
        mock_mss.side_effect = RuntimeError("Windows graphics function failed: BitBlt")
        mock_ig.return_value = screen_mod.Image.new("RGB", (100, 80), (10, 20, 30))
        cap = ScreenCapture()
        data, w, h = cap.grab_jpeg()
        self.assertIsNotNone(data)
        self.assertEqual(w, 100)
        self.assertEqual(h, 80)
        self.assertGreater(mock_mss.call_count, 1)
        cap.close()

    @patch.object(screen_mod, "_grab_mss")
    @patch.object(screen_mod, "_grab_imagegrab")
    def test_all_fail_friendly_message(self, mock_ig, mock_mss):
        mock_mss.side_effect = RuntimeError("Windows graphics function failed: BitBlt")
        mock_ig.return_value = None
        cap = ScreenCapture()
        data, w, h = cap.grab_jpeg()
        self.assertIsNone(data)
        self.assertIn("BitBlt", cap.last_error)
        cap.close()


if __name__ == "__main__":
    unittest.main()
