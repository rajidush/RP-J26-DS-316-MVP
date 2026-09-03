"""The "What it read" panel must show everything it read, and mark the risk.

Two regressions this guards:

1. The list was `meaningful.slice(0, 10)` with a "+ 48 more regions read" tail.
   A parent asking what was on the child's screen got ten rows and a number.

2. Highlighting was binary on a hardcoded `score >= 0.5`, which does not match
   any actual alert limit (0.55 / 0.65 / 0.75 by age). A line could be styled
   as harmless while sitting above the limit that raised the alert.

These are string assertions against the panel source rather than DOM tests —
there is no JS runner in this project, and the point is to notice if the
behaviour is quietly reverted, not to re-test the browser.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PANEL = ROOT / "analyst" / "panel" / "index.html"


@unittest.skipUnless(PANEL.is_file(), "panel/index.html missing")
class ReadListTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = PANEL.read_text(encoding="utf-8")

    def test_list_is_not_truncated_to_a_head(self):
        self.assertNotIn(
            "meaningful.slice(0, 10)",
            self.html,
            "the read-list is sliced again — it must list every region",
        )

    def test_no_more_regions_placeholder(self):
        self.assertNotIn(
            "more region${rest",
            self.html,
            'the "+ N more regions read" tail is back instead of the regions',
        )

    def test_text_wraps_instead_of_truncating(self):
        row = re.search(r"\.rows \.t\{([^}]*)\}", self.html)
        self.assertIsNotNone(row, ".rows .t rule not found")
        css = row.group(1)
        self.assertNotIn(
            "white-space:nowrap",
            css,
            "read rows truncate again; the end of a sentence is where the "
            "hurtful part usually sits",
        )
        self.assertIn("overflow-wrap:anywhere", css)


@unittest.skipUnless(PANEL.is_file(), "panel/index.html missing")
class RiskTierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = PANEL.read_text(encoding="utf-8")

    def test_tier_uses_the_age_limit_not_a_constant(self):
        self.assertIn("age <= 10 ? 0.55 : (age <= 13 ? 0.65 : 0.75)", self.html)
        self.assertIn("sc >= limit", self.html)

    def test_no_hardcoded_half_threshold(self):
        self.assertNotIn(
            "Number(d.score) >= 0.5",
            self.html,
            "highlighting is back on a hardcoded 0.5, which matches no age limit",
        )

    def test_all_three_tiers_are_defined(self):
        for token in ("ov-hit", "ov-warn", "ov-q", "ov-pic"):
            self.assertIn(token, self.html, f"overlay tier {token} missing")
        for rule in (".rows li.hit", ".rows li.warn"):
            self.assertIn(rule, self.html, f"row style {rule} missing")

    def test_overlay_and_list_share_one_tier_function(self):
        """A red box on the picture and a red row in the list must mean the
        same thing, so both have to come from the same classifier."""
        self.assertIn("const tierOf = d =>", self.html)
        self.assertGreaterEqual(
            self.html.count("tierOf("),
            3,
            "tierOf should drive the overlay, the ordering and the row classes",
        )

    def test_legend_names_every_tier(self):
        for token in ("lg-q", "lg-warn", "lg-hit", "lg-pic"):
            self.assertIn(token, self.html, f"legend entry {token} missing")

    def test_counts_are_surfaced(self):
        """A count of what was read, and of what crossed the limit, so the
        reader can tell "nothing found" from "nothing looked at"."""
        self.assertIn('class="readsum"', self.html)
        self.assertIn("${listed.length}", self.html)
        self.assertIn("at or over the limit", self.html)
        self.assertIn("${nAlert}", self.html)
        self.assertIn("${nWarn}", self.html)


if __name__ == "__main__":
    unittest.main()
