"""Regression tests for the C2 audit findings. Each test names the defect it locks down."""

import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyst.buffer import TransientMediaBuffer
from analyst.decide import PERSONA_THETA2, STAGE1_THETA
from analyst.pipeline import AnalystPipeline
from analyst.stage1.lexicon import MID_BASE, MID_STEP, score_text


class LexiconFalsePositiveTests(unittest.TestCase):
    """Multi-word phrases were matched as raw substrings, so ordinary idioms
    containing a trigger word ("gas the car", "go back to the hotel") scored
    0.77-0.88 and fired hate.detected."""

    BENIGN = [
        "lets gas the car up before the trip",
        "we should go back to the hotel",
        "your kind of music is great",
        "my therapist appointment is friday",
        "grape juice",
        "hey want to play minecraft after school",
    ]

    def test_benign_phrases_do_not_escalate(self):
        for text in self.BENIGN:
            with self.subTest(text=text):
                score, category, hits = score_text(text)
                self.assertLess(score, STAGE1_THETA, f"{text!r} escalated on {hits}")
                self.assertEqual(category, "none")


class LexiconRecallTests(unittest.TestCase):
    """Word-boundary matching missed inflections and obfuscation."""

    def test_inflected_and_obfuscated_forms_are_caught(self):
        cases = {
            "he raped her": "sexual_harassment",
            "kill your self": "threat",
            "k y s": "threat",
            "kiiill yourself": "threat",
            "you are a f4ggot": "hate_identity",
            "n1gger": "hate_identity",
            "just  kys  already": "threat",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                score, category, hits = score_text(text)
                self.assertGreaterEqual(score, STAGE1_THETA, f"{text!r} missed")
                self.assertEqual(category, expected)
                self.assertTrue(hits)


class PersonaThresholdTests(unittest.TestCase):
    """Every reachable escalated score exceeded every theta2 (min 0.77 vs max
    0.75), so child_age never changed a decision and hate.cleared was dead."""

    def setUp(self):
        self.pipe = AnalystPipeline()

    def test_mid_band_straddles_persona_thresholds(self):
        one_hit = MID_BASE + MID_STEP
        self.assertGreater(one_hit, PERSONA_THETA2["8-10"])
        self.assertLess(one_hit, PERSONA_THETA2["11-13"])

    def test_age_changes_the_decision(self):
        text = "nobody likes you"
        self.assertEqual(self.pipe.analyze(child_age=9, overlay_text=text).decision, "hate")
        self.assertEqual(self.pipe.analyze(child_age=15, overlay_text=text).decision, "not-hate")

    def test_high_band_trips_for_every_age(self):
        for age in (8, 12, 15):
            with self.subTest(age=age):
                r = self.pipe.analyze(child_age=age, overlay_text="you should kys")
                self.assertEqual(r.decision, "hate")


class ClearedContractTests(unittest.TestCase):
    """hate.cleared had a payload but no Envelope, so C4 received no
    false-positive telemetry on the bus."""

    def setUp(self):
        self.pipe = AnalystPipeline()

    def test_escalated_but_below_threshold_emits_cleared_envelope(self):
        r = self.pipe.analyze(child_age=15, overlay_text="nobody likes you")
        self.assertEqual(r.decision, "not-hate")
        self.assertIsNotNone(r.envelope)
        self.assertEqual(r.envelope.topic, "hate.cleared")
        self.assertEqual(r.cleared.reason, "below_persona_threshold")

    def test_stage1_stop_emits_no_envelope(self):
        r = self.pipe.analyze(child_age=10, overlay_text="gg ez noob")
        self.assertIsNone(r.envelope)
        self.assertIsNone(r.cleared)
        self.assertIn("stopped_at_stage1", r.notes)


class PrivacyTests(unittest.TestCase):
    """Engineering Plan hard rule: raw media stays in RAM and is wiped."""

    def test_wipe_reaches_the_buffer_the_caller_read(self):
        buf = TransientMediaBuffer()
        held = {}
        with buf.hold("t1", frame=b"TOPSECRETFRAME") as slot:
            held["frame"] = slot["frame"]
            self.assertEqual(bytes(held["frame"]), b"TOPSECRETFRAME")
        self.assertEqual(bytes(held["frame"]), b"", "caller kept readable media")
        self.assertEqual(buf.occupied(), 0)

    def test_asr_never_writes_media_to_disk(self):
        source = (Path(__file__).resolve().parents[1] / "extract" / "asr.py").read_text(
            encoding="utf-8"
        )
        for banned in ("tempfile", "mkstemp", "NamedTemporaryFile"):
            self.assertNotIn(banned, source, f"asr.py reintroduced {banned}")


class OcrCompatTests(unittest.TestCase):
    """rapidocr-onnxruntime caps at Python <3.13; extract/ocr.py must also
    drive the successor `rapidocr` package, whose result is an object."""

    def test_engine_reports_a_backend_when_a_rapidocr_is_installed(self):
        try:
            import rapidocr_onnxruntime  # noqa: F401

            installed = True
        except Exception:
            try:
                import rapidocr  # noqa: F401

                installed = True
            except Exception:
                installed = False
        if not installed:
            self.skipTest("no RapidOCR package installed")

        from analyst.extract.ocr import OcrEngine

        self.assertIn(OcrEngine().name, {"rapidocr", "rapidocr3"})

    def test_both_result_shapes_are_handled(self):
        from analyst.extract.ocr import OcrEngine

        engine = OcrEngine.__new__(OcrEngine)
        engine._tesseract = None

        class FakeObj:
            txts = ("you should kys",)

        engine._rapid = lambda _img: FakeObj()
        engine._rapid_api = "object"
        self.assertEqual(engine.extract(_blank()), "you should kys")

        engine._rapid = lambda _img: ([[None, "you should kys", 0.99]], None)
        engine._rapid_api = "tuple"
        self.assertEqual(engine.extract(_blank()), "you should kys")


def _blank():
    from PIL import Image

    return Image.new("RGB", (64, 32), "white")



class LoopbackAudioTests(unittest.TestCase):
    """A Speaker object only plays; `_Speaker.recorder()` raises AttributeError.
    The capture thread died instantly while start() still returned True, so the
    panel reported a healthy audio branch that recorded nothing."""

    def test_uses_loopback_microphone_not_speaker_recorder(self):
        source = (
            Path(__file__).resolve().parents[1] / "capture" / "audio.py"
        ).read_text(encoding="utf-8")
        self.assertIn("include_loopback=True", source)
        self.assertNotIn("default_speaker().recorder", source)

    def test_start_reports_failure_when_device_cannot_open(self):
        from analyst.capture.audio import LoopbackCapture

        cap = LoopbackCapture()
        if not cap.available:
            self.skipTest("soundcard not installed")

        def boom() -> None:
            cap._last_error = "simulated device failure"
            cap._running = False
            cap._ready.set()

        cap._loop = boom  # type: ignore[method-assign]
        self.assertFalse(cap.start(), "start() claimed success on a dead device")
        self.assertTrue(cap.last_error)
        self.assertFalse(cap.running)

    def test_start_then_stop_buffers_and_clears(self):
        from analyst.capture.audio import LoopbackCapture

        cap = LoopbackCapture()
        if not cap.available:
            self.skipTest("soundcard not installed")
        if not cap.start():
            self.skipTest(f"no loopback device here: {cap.last_error}")
        try:
            time.sleep(1.0)
            self.assertTrue(cap.running)
            self.assertGreater(cap._concat().size, 0, "loopback buffered no samples")
        finally:
            cap.stop()
        self.assertFalse(cap.running)


class RetentionTests(unittest.TestCase):
    """The store kept every blurred screen thumbnail and OCR snippet forever —
    there was no prune, TTL, or row cap anywhere in the codebase."""

    def _store(self):
        import tempfile
        from analyst.store.db import AnalystStore

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        return AnalystStore(Path(self._tmp.name) / "t.db")

    def _insert(self, store, run_id, ts, thumb=b"fake-jpeg-bytes"):
        import sqlite3

        conn = sqlite3.connect(str(store.db_path))
        conn.execute(
            "INSERT INTO runs (id, ts, decision, thumb_jpeg, ocr_snippet) VALUES (?,?,?,?,?)",
            (run_id, ts, "not-hate", thumb, "text read off the screen"),
        )
        conn.commit()
        conn.close()

    def _count(self, store, sql):
        import sqlite3

        conn = sqlite3.connect(str(store.db_path))
        n = conn.execute(sql).fetchone()[0]
        conn.close()
        return n

    def test_expired_thumbnails_are_cleared_but_row_survives(self):
        from datetime import datetime, timedelta, timezone

        store = self._store()
        old = (datetime.now(timezone.utc) - timedelta(hours=48)).astimezone().isoformat(
            timespec="milliseconds"
        )
        new = datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")
        self._insert(store, "old", old)
        self._insert(store, "new", new)

        store.prune(thumb_ttl_hours=24, run_ttl_days=3650, max_runs=10_000)

        self.assertEqual(
            self._count(store, "SELECT COUNT(*) FROM runs WHERE thumb_jpeg IS NOT NULL"),
            1,
            "expired thumbnail was not cleared",
        )
        self.assertEqual(self._count(store, "SELECT COUNT(*) FROM runs"), 2)

    def test_rows_past_ttl_are_deleted(self):
        from datetime import datetime, timedelta, timezone

        store = self._store()
        ancient = (datetime.now(timezone.utc) - timedelta(days=90)).astimezone().isoformat(
            timespec="milliseconds"
        )
        self._insert(store, "ancient", ancient)
        self._insert(
            store,
            "fresh",
            datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
        )
        store.prune(thumb_ttl_hours=24, run_ttl_days=30, max_runs=10_000)
        self.assertEqual(self._count(store, "SELECT COUNT(*) FROM runs"), 1)

    def test_row_cap_drops_oldest_first(self):
        from datetime import datetime, timedelta, timezone

        store = self._store()
        base = datetime.now(timezone.utc)
        for i in range(10):
            ts = (base - timedelta(minutes=i)).astimezone().isoformat(timespec="milliseconds")
            self._insert(store, f"r{i}", ts)
        store.prune(thumb_ttl_hours=99999, run_ttl_days=3650, max_runs=4)
        self.assertEqual(self._count(store, "SELECT COUNT(*) FROM runs"), 4)

    def test_purge_all_removes_everything(self):
        from datetime import datetime, timezone

        store = self._store()
        self._insert(
            store,
            "a",
            datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
        )
        self.assertEqual(store.purge_all(), 1)
        self.assertEqual(self._count(store, "SELECT COUNT(*) FROM runs"), 0)



def _poster(a="NOBODY LIKES YOU", b="GO BACK TO YOUR COUNTRY"):
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (760, 560), (140, 30, 30))
    d = ImageDraw.Draw(im)
    d.rectangle([40, 40, 720, 520], outline=(255, 255, 255), width=3)
    d.text((90, 150), a, fill=(255, 220, 90))
    d.text((90, 250), b, fill=(255, 255, 255))
    d.ellipse([520, 330, 640, 450], fill=(230, 180, 60))
    return im


def _desktop_with(poster):
    from PIL import Image, ImageDraw

    desk = Image.new("RGB", (1600, 900), (245, 245, 245))
    d = ImageDraw.Draw(desk)
    d.rectangle([0, 0, 1600, 70], fill=(228, 228, 228))
    d.text((24, 92), "https://www.bing.com/images/search?q=hate", fill=(60, 60, 60))
    desk.paste(poster, (420, 160))
    return desk


class ImageRegionTests(unittest.TestCase):
    """A 450M VLM handed a whole desktop invents content confidently
    ("a red square with the text GUARANTEE"). Cropping to the picture first is
    what makes the vision branch trustworthy, so these lock the crop down."""

    def test_finds_the_poster_inside_a_desktop(self):
        from analyst.extract.region import find_image_region

        box = find_image_region(_desktop_with(_poster()))
        self.assertIsNotNone(box, "poster region not found")
        true_box = (420, 160, 1180, 720)
        ix0, iy0 = max(box[0], true_box[0]), max(box[1], true_box[1])
        ix1, iy1 = min(box[2], true_box[2]), min(box[3], true_box[3])
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        poster_area = (true_box[2] - true_box[0]) * (true_box[3] - true_box[1])
        self.assertGreater(inter / poster_area, 0.9, "crop missed part of the poster")

    def test_plain_text_screen_is_not_a_picture(self):
        """Text is high-contrast but colourless; OCR already covers it."""
        from PIL import Image, ImageDraw

        from analyst.extract.region import find_image_region

        page = Image.new("RGB", (1600, 900), (250, 250, 250))
        d = ImageDraw.Draw(page)
        for i in range(28):
            d.text((60, 60 + i * 28), "the quick brown fox jumps over the lazy dog " * 2,
                   fill=(40, 40, 40))
        self.assertIsNone(find_image_region(page))

    def test_none_and_tiny_images_are_safe(self):
        from PIL import Image

        from analyst.extract.region import crop_to_region, find_image_region

        self.assertIsNone(find_image_region(None))
        self.assertEqual(crop_to_region(None), (None, None))
        self.assertIsNone(find_image_region(Image.new("RGB", (10, 10), "red")))


class VisionReadingTests(unittest.TestCase):
    def test_declines_when_there_is_no_picture(self):
        """No region => no VLM call. Asking anyway is how hallucinations enter."""
        from PIL import Image

        from analyst.extract.vision_meaning import VisionMeaning

        vm = VisionMeaning(base_url="http://127.0.0.1:9/v1", timeout=2)
        reading = vm.read(Image.new("RGB", (800, 600), (250, 250, 250)))
        self.assertIn("no_image_region", reading.notes)
        self.assertEqual(reading.caption, "")
        self.assertEqual(reading.error, "", "must not even attempt the call")

    def test_reply_scaffolding_is_stripped(self):
        from analyst.extract.vision_meaning import _clean

        self.assertEqual(_clean("Sure! Here's the transcription: **KYS**"), "KYS")
        self.assertEqual(_clean("NONE"), "")
        self.assertEqual(_clean("  NONE.  "), "")

    def test_result_carries_what_the_vision_branch_read(self):
        from analyst.schemas import AnalystRunResult

        r = AnalystRunResult(decision="not-hate")
        for field in ("image_caption", "image_text", "image_region"):
            self.assertTrue(hasattr(r, field), f"{field} missing from the result")


class AsrHallucinationTests(unittest.TestCase):
    """Whisper invents fluent speech from silence — on a quiet desktop it was
    emitting Japanese, which then scored as real audio evidence."""

    def _wav(self, samples, sr=16000):
        import io as _io
        import wave

        import numpy as np

        b = _io.BytesIO()
        with wave.open(b, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
        return b.getvalue()

    def test_silence_transcribes_to_nothing(self):
        import numpy as np

        from analyst.extract.asr import AsrEngine

        engine = AsrEngine(model_size="tiny")
        if engine.name == "none":
            self.skipTest("faster-whisper not installed")
        rng = np.random.default_rng(0)
        for label, sig in (
            ("silence", np.zeros(16000 * 3, dtype=np.float32)),
            ("room tone", (rng.standard_normal(16000 * 3) * 0.006).astype(np.float32)),
        ):
            with self.subTest(signal=label):
                self.assertEqual(engine.transcribe(self._wav(sig)).strip(), "")

    def test_language_is_pinned_to_english(self):
        source = (Path(__file__).resolve().parents[1] / "extract" / "asr.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('language="en"', source)
        self.assertIn("no_speech_prob", source)

if __name__ == "__main__":
    unittest.main()


class VisionMeaningTests(unittest.TestCase):
    """Meme case: harm in the picture, not in OCR-able text. Closed with a
    local VLM whose caption is scored by the SAME stage-1 text scorer, so the
    definition of hate stays in one auditable place."""

    def test_disabled_by_default(self):
        """With no ANALYST_VLM_URL in the environment the branch is inert."""
        import os
        from unittest import mock

        from analyst.extract.vision_meaning import VisionMeaning

        with mock.patch.dict(os.environ, {}, clear=True):
            vm = VisionMeaning()
            self.assertFalse(vm.enabled)
            self.assertEqual(vm.name, "none")
            self.assertEqual(vm.caption(_blank()), "", "must be inert when unconfigured")

    def test_unreachable_server_degrades_without_raising(self):
        """Needs a real picture: a blank frame is declined before any call."""
        from analyst.extract.vision_meaning import VisionMeaning

        vm = VisionMeaning(base_url="http://127.0.0.1:9/v1", timeout=2)
        self.assertTrue(vm.enabled)
        reading = vm.read(_desktop_with(_poster()))
        self.assertIsNotNone(reading.box, "region should have been found")
        self.assertEqual(reading.caption, "")
        self.assertTrue(reading.error, "failure must be recorded, not swallowed")
        self.assertTrue(vm.last_error)

    def test_backend_is_reported_to_the_panel(self):
        from analyst.pipeline import AnalystPipeline

        self.assertIn("vision_meaning", AnalystPipeline().backends())

    def test_caption_is_scored_by_the_shared_text_scorer(self):
        """A hateful caption must reach the same lexicon the text path uses."""
        from analyst.stage1.text_fast import TextFast

        caption = 'A meme showing the words "you should kys" over a photo.'
        score, category, _ = TextFast().score(caption)
        self.assertGreaterEqual(score, STAGE1_THETA)
        self.assertEqual(category, "threat")
