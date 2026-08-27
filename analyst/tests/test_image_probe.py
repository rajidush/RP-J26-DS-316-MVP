"""The vision probe's admission rules.

A trained probe is the only thing allowed to make the image branch count toward
the score. These tests guard the gate, because the failure they prevent already
happened once: an uncalibrated vision score averaged into fusion cleared a
confirmed "you should kys" for ages 14-15.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyst.stage1 import image_fast as mod
from analyst.stage1.image_fast import ImageFast


def _probe(dim=512, auc=0.71, meets_bar=True, coef=None):
    return {
        "coef": coef if coef is not None else [0.01] * dim,
        "intercept": 0.0,
        "dim": dim,
        "dev_auc": auc,
        "meets_bar": meets_bar,
    }


class ProbeAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "image_probe.json"
        self._orig = mod.PROBE_PATH
        mod.PROBE_PATH = self.path

    def tearDown(self):
        mod.PROBE_PATH = self._orig
        self.tmp.cleanup()

    def _write(self, probe):
        self.path.write_text(json.dumps(probe), encoding="utf-8")

    def test_no_probe_means_uncalibrated(self):
        """A fresh clone has no probe, so the branch must not count."""
        fast = ImageFast()
        self.assertFalse(fast.calibrated)
        self.assertIsNone(fast._probe)
        self.assertNotIn("clip-probe", fast.name)

    def test_probe_below_the_bar_is_refused(self):
        """Saved for the record, never loaded — the whole point of the gate."""
        self._write(_probe(auc=0.58, meets_bar=False))
        fast = ImageFast()
        self.assertFalse(fast.calibrated)
        self.assertIsNone(fast._probe)

    def test_probe_above_the_bar_is_admitted(self):
        self._write(_probe(auc=0.71, meets_bar=True))
        fast = ImageFast()
        self.assertTrue(fast.calibrated)
        self.assertIn("clip-probe", fast.name)

    def test_corrupt_probe_degrades_instead_of_crashing(self):
        self.path.write_text("{ not json", encoding="utf-8")
        fast = ImageFast()
        self.assertFalse(fast.calibrated)

    def test_probe_without_weights_is_refused(self):
        self._write(_probe(coef=[]))
        self.assertFalse(ImageFast().calibrated)


class ProbeScoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "image_probe.json"
        self._orig = mod.PROBE_PATH
        mod.PROBE_PATH = self.path

    def tearDown(self):
        mod.PROBE_PATH = self._orig
        self.tmp.cleanup()

    def _fast(self, probe):
        self.path.write_text(json.dumps(probe), encoding="utf-8")
        from analyst.extract.embed import ImageEmbedder

        fast = ImageFast(embedder=ImageEmbedder(embed_fn=lambda _i: [0.0] * probe["dim"]))
        return fast

    def test_score_is_a_probability(self):
        fast = self._fast(_probe(dim=8))
        score = fast.score(embedding=[0.5] * 8)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_zero_intercept_zero_input_is_one_half(self):
        fast = self._fast(_probe(dim=8, coef=[0.0] * 8))
        self.assertAlmostEqual(fast.score(embedding=[0.0] * 8), 0.5, places=4)

    def test_positive_weights_raise_the_score(self):
        fast = self._fast(_probe(dim=8, coef=[1.0] * 8))
        self.assertGreater(fast.score(embedding=[1.0] * 8), 0.9)

    def test_dimension_mismatch_refuses_rather_than_scoring_noise(self):
        """If the embedder changes under the probe, the weights are meaningless.
        Scoring anyway would feed garbage into a safety decision."""
        fast = self._fast(_probe(dim=512))
        self.assertTrue(fast.calibrated)
        score = fast.score(embedding=[0.1] * 8)  # wrong width
        self.assertEqual(score, 0.0)
        self.assertFalse(fast.calibrated)
        self.assertIn("mismatch", fast.name)

    def test_bind_embedder_does_not_downgrade_a_loaded_probe(self):
        from analyst.extract.embed import ImageEmbedder

        fast = self._fast(_probe(dim=8))
        fast.bind_embedder(ImageEmbedder(embed_fn=lambda _i: [0.0] * 8))
        self.assertTrue(fast.calibrated)
        self.assertIn("clip-probe", fast.name)


class BarConstantTests(unittest.TestCase):
    def test_bar_is_above_chance_by_a_clear_margin(self):
        from analyst.evaluation.train_image_probe import MIN_AUC

        self.assertGreater(MIN_AUC, 0.5)
        self.assertLess(MIN_AUC, 1.0)


if __name__ == "__main__":
    unittest.main()
