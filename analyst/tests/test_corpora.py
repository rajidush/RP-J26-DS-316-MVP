"""External corpus loader — split discipline and label mapping.

These run offline: the split function and the adapters are pure, so the only
tests that need the network are skipped when the cache is cold.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyst.evaluation import corpora


class SplitDisciplineTests(unittest.TestCase):
    """The whole value of the loader rests on splits being content-derived."""

    def test_split_is_stable_for_the_same_text(self):
        text = "you should kys nobody likes you"
        self.assertEqual(corpora.split_of(text), corpora.split_of(text))

    def test_split_ignores_surrounding_whitespace_and_case(self):
        self.assertEqual(
            corpora.split_of("  Some Sentence Here "),
            corpora.split_of("some sentence here"),
        )

    def test_every_split_is_reachable(self):
        seen = {corpora.split_of(f"sample sentence number {i}") for i in range(400)}
        self.assertEqual(seen, {"train", "dev", "test"})

    def test_distribution_is_roughly_60_20_20(self):
        counts = {"train": 0, "dev": 0, "test": 0}
        n = 4000
        for i in range(n):
            counts[corpora.split_of(f"row {i} of the corpus")] += 1
        self.assertAlmostEqual(counts["train"] / n, 0.60, delta=0.04)
        self.assertAlmostEqual(counts["dev"] / n, 0.20, delta=0.04)
        self.assertAlmostEqual(counts["test"] / n, 0.20, delta=0.04)

    def test_a_tuned_case_cannot_drift_into_test(self):
        """Split must not depend on how many rows were requested."""
        text = "their kind always ruins everything they touch"
        first = corpora.split_of(text)
        for _ in range(50):
            self.assertEqual(corpora.split_of(text), first)


class RegistryTests(unittest.TestCase):
    def test_every_registered_corpus_has_an_adapter(self):
        for key in corpora.available():
            self.assertIn(key, corpora._ADAPTERS, f"{key} has no adapter")

    def test_every_corpus_carries_a_citation(self):
        for key in corpora.available():
            self.assertTrue(corpora.REGISTRY[key].citation.strip())

    def test_unknown_corpus_and_split_raise(self):
        with self.assertRaises(KeyError):
            corpora.load("not-a-corpus")
        with self.assertRaises(KeyError):
            corpora.load("davidson", split="not-a-split")


class OfflineBehaviourTests(unittest.TestCase):
    def test_missing_download_returns_empty_not_an_exception(self):
        """A cold cache with no network must degrade, not crash the harness."""
        original = corpora._download
        corpora._download = lambda info: None
        try:
            self.assertEqual(corpora.load("jigsaw"), [])
        finally:
            corpora._download = original


class BalancingTests(unittest.TestCase):
    def test_interleave_alternates_and_keeps_everything(self):
        from analyst.evaluation.dataset import TextCase

        a = [TextCase(f"h{i}", True, "threat", "threat") for i in range(3)]
        b = [TextCase(f"s{i}", False, "none", "benign") for i in range(3)]
        out = corpora._interleave(a, b)
        self.assertEqual(len(out), 6)
        self.assertTrue(out[0].hate)
        self.assertFalse(out[1].hate)

    def test_interleave_handles_uneven_sides(self):
        from analyst.evaluation.dataset import TextCase

        a = [TextCase("h", True, "threat", "threat")]
        b = [TextCase(f"s{i}", False, "none", "benign") for i in range(4)]
        self.assertEqual(len(corpora._interleave(a, b)), 5)


if __name__ == "__main__":
    unittest.main()
