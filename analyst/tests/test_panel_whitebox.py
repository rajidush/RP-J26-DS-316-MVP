"""The White box diagram must stay wired to the real pipeline.

The diagram is data-driven: each box names a trace step id (what it did on the
last check) and a backend key (which model is loaded behind it). Both are
strings shared with Python code that has no idea the panel exists. Rename
`_step("clip", ...)` in whitebox/trace.py, or a key in pipeline.backends(),
and the box does not error — it silently goes grey forever, which is the worst
failure mode a transparency feature can have.

These tests parse both sides and assert they still agree.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PANEL = ROOT / "analyst" / "panel" / "index.html"
TRACE = ROOT / "analyst" / "whitebox" / "trace.py"
PIPELINE = ROOT / "analyst" / "pipeline.py"
WORKER = ROOT / "analyst" / "capture" / "worker.py"


def _panel() -> str:
    return PANEL.read_text(encoding="utf-8")


def _nodes():
    """Parse WB_NODES entries into {id, step, backend}."""
    html = _panel()
    block = re.search(r"const WB_NODES = \[(.*?)\n\];", html, re.S)
    assert block, "WB_NODES not found in the panel"
    out = []
    for line in block.group(1).splitlines():
        m = re.search(r'id:"(\w+)"', line)
        if not m:
            continue
        step = re.search(r'step:"(\w+)"', line)
        backend = re.search(r"backend:(?:\"(\w+)\"|null)", line)
        out.append({
            "id": m.group(1),
            "step": step.group(1) if step else None,
            "backend": backend.group(1) if backend and backend.group(1) else None,
        })
    return out


def _primary_flags():
    """WB_NODES entries with their `primary` (owns-the-timing) flag."""
    block = re.search(r"const WB_NODES = \[(.*?)\n\];", _panel(), re.S)
    out = []
    for line in block.group(1).splitlines():
        m = re.search(r'id:"(\w+)"', line)
        if not m:
            continue
        step = re.search(r'step:"(\w+)"', line)
        out.append({
            "id": m.group(1),
            "step": step.group(1) if step else None,
            "primary": "primary:true" in line.replace(" ", ""),
        })
    return out


def _live_stage_ids():
    """Every stage id the running pipeline actually marks."""
    ids = set()
    for f in (PIPELINE, WORKER):
        src = f.read_text(encoding="utf-8")
        ids |= set(re.findall(r'live_stage\(\s*"(\w+)"', src))
        ids |= set(re.findall(r'LIVE\.begin\(\s*"(\w+)"', src))
    return ids


def _node_live_ids():
    block = re.search(r"const WB_NODES = \[(.*?)\n\];", _panel(), re.S)
    out = {}
    for line in block.group(1).splitlines():
        m = re.search(r'id:"(\w+)"', line)
        if not m:
            continue
        live = re.search(r'live:(?:"(\w+)"|null)', line)
        out[m.group(1)] = live.group(1) if live and live.group(1) else None
    return out


def _trace_step_ids():
    return set(re.findall(r'_step\(\s*"(\w+)"', TRACE.read_text(encoding="utf-8")))


def _backend_keys():
    body = re.search(r"def backends\(self\) -> dict:\s*return \{(.*?)\n        \}",
                     PIPELINE.read_text(encoding="utf-8"), re.S)
    assert body, "pipeline.backends() not found"
    keys = set(re.findall(r'"(\w+)":', body.group(1)))
    # worker.whitebox() merges these in alongside the pipeline's own keys
    wb = re.search(r'"modules": \{(.*?)\n            \}',
                   WORKER.read_text(encoding="utf-8"), re.S)
    if wb:
        keys |= set(re.findall(r'"(\w+)":', wb.group(1)))
    return keys


@unittest.skipUnless(PANEL.is_file(), "panel/index.html missing")
class DiagramContractTests(unittest.TestCase):
    def test_nodes_were_parsed(self):
        self.assertGreaterEqual(len(_nodes()), 8, "WB_NODES failed to parse")

    def test_every_node_names_a_real_trace_step(self):
        known = _trace_step_ids()
        self.assertTrue(known, "no _step ids parsed from trace.py")
        for n in _nodes():
            if n["step"] is None:
                # A pure input (the typed message) is not a pipeline stage and
                # has no trace step of its own. Allowed, but it must then be a
                # source box — anything downstream has to report a real stage.
                self.assertIn(n["id"], {"typed"}, f'{n["id"]} has no trace step')
                continue
            self.assertIn(
                n["step"], known,
                f'diagram box "{n["id"]}" waits on step "{n["step"]}", which '
                f"whitebox/trace.py never emits — the box would stay grey forever",
            )

    def test_every_node_backend_is_a_real_module_key(self):
        known = _backend_keys()
        self.assertTrue(known, "no backend keys parsed")
        for n in _nodes():
            if n["backend"] is None:
                continue
            self.assertIn(
                n["backend"], known,
                f'diagram box "{n["id"]}" reads module "{n["backend"]}", which '
                f"is not reported by backends()/whitebox()",
            )

    def test_every_emitted_step_is_shown_somewhere(self):
        """The reverse direction: a stage the engine runs but the diagram never
        draws is a stage the panel is hiding."""
        shown = {n["step"] for n in _nodes()}
        for step in _trace_step_ids():
            self.assertIn(
                step, shown,
                f'the engine emits step "{step}" but no diagram box shows it',
            )

    def test_every_stage_has_a_live_status_label(self):
        """The live line names the stage, not a box. Several boxes share one
        stage id, so an unlabelled stage would surface as a raw id like
        "stage1" in front of the user."""
        block = re.search(r"const WB_STAGE_LABELS = \{(.*?)\};", _panel(), re.S)
        self.assertIsNotNone(block, "WB_STAGE_LABELS not found")
        labelled = set(re.findall(r"^\s*(\w+):", block.group(1), re.M))
        for step in _trace_step_ids():
            self.assertIn(step, labelled, f'stage "{step}" has no live label')

    def test_timing_badge_is_not_repeated_across_shared_steps(self):
        """Stage 1 is timed as a whole. Showing that one number on all six of
        its boxes would read as six separate costs."""
        nodes = _primary_flags()
        by_step = {}
        for n in nodes:
            by_step.setdefault(n["step"], []).append(n)
        for step, group in by_step.items():
            if step is None:
                continue
            primaries = [n for n in group if n["primary"]]
            self.assertLessEqual(
                len(primaries), 1,
                f'step "{step}" shows its duration on {len(primaries)} boxes',
            )

    def test_every_node_live_id_is_actually_marked(self):
        """A box whose live id the pipeline never marks can never light up —
        it would sit dim through every check and nothing would say why."""
        marked = _live_stage_ids()
        self.assertTrue(marked, "no live stage ids parsed from the pipeline")
        for node, live in _node_live_ids().items():
            if live is None:
                continue  # a pure input, not a stage
            self.assertIn(
                live, marked,
                f'box "{node}" waits on live stage "{live}", which the pipeline '
                f"never marks",
            )

    def test_every_marked_stage_is_shown_somewhere(self):
        shown = set(_node_live_ids().values())
        for live in _live_stage_ids():
            self.assertIn(
                live, shown,
                f'the pipeline marks live stage "{live}" but no box shows it',
            )

    def test_live_ids_have_status_labels(self):
        block = re.search(r"const WB_STAGE_LABELS = \{(.*?)\};", _panel(), re.S)
        labelled = set(re.findall(r"^\s*(\w+):", block.group(1), re.M))
        for live in _live_stage_ids():
            self.assertIn(live, labelled, f'live stage "{live}" has no label')

    def test_sub_steps_do_not_share_a_live_id(self):
        """The bug this replaced: six boxes shared one coarse stage, so they
        all lit at the same instant showing the same duration."""
        lives = [v for v in _node_live_ids().values() if v]
        dupes = {v for v in lives if lives.count(v) > 1}
        # "audio" is legitimately shared: the speaker source and the transcriber
        # are two faces of one capture-and-transcribe step.
        self.assertEqual(dupes - {"audio"}, set(), f"boxes share a live stage: {dupes}")

    def test_init_time_state_is_declared_before_show(self):
        """show() runs during page init and starts the white-box poller, so
        every `let` that poller touches must already be initialised.

        A `let` declared further down the file is in its temporal dead zone at
        that moment and throws a ReferenceError, which takes the whole init
        block with it and leaves the panel blank with no visible clue. This has
        happened twice; the ordering is now asserted rather than remembered.
        """
        html = _panel()
        show_at = html.index("function show(name)")
        body = re.search(r"function wbPollStart\(\)[^\n]*", html).group(0)
        body += re.search(r"function wbPollStop\(\)[^\n]*", html).group(0)
        touched = set(re.findall(r"\b(wb[A-Za-z]\w*)\b", body))
        for name in sorted(touched):
            m = re.search(r"^(?:let|const|var)\s+%s\b" % re.escape(name),
                          html, re.M)
            if not m:
                continue  # a function, not module state
            self.assertLess(
                m.start(), show_at,
                f'`{name}` is declared after show(), so calling wbPollStart() '
                f"during init throws a temporal-dead-zone ReferenceError",
            )

    def test_panel_marks_are_set_by_the_pipeline(self):
        """Stage 1 skips the language models when the lexicon has already
        decided, so an explicit slur is the *fastest* path through the engine,
        not the slowest. The box must say which path ran instead of always
        advertising models. The panel reads these keys by name; the pipeline
        has to set them."""
        html = _panel()
        pipeline = PIPELINE.read_text(encoding="utf-8")
        read = set(re.findall(r'marks(?:\.|\[")(\w+)', html))
        self.assertIn("s1_models_ran", read, "panel no longer reads the mark")
        for key in read:
            self.assertIn(
                'LIVE.mark("%s"' % key, pipeline,
                f'panel reads mark "{key}" that the pipeline never sets',
            )

    def test_marks_are_cleared_per_check(self):
        """A mark left over from the previous check would describe the wrong
        run — worse than showing nothing."""
        live = (ROOT / "analyst" / "whitebox" / "live.py").read_text(encoding="utf-8")
        start = live.index("def start_check")
        end = live.index("def end_check")
        self.assertIn("_marks.clear()", live[start:end],
                      "start_check does not clear the previous check's marks")

    def test_edges_only_reference_declared_nodes(self):
        ids = {n["id"] for n in _nodes()}
        block = re.search(r"const WB_EDGES = \[(.*?)\n\];", _panel(), re.S)
        self.assertIsNotNone(block, "WB_EDGES not found")
        pairs = re.findall(r'\["(\w+)","(\w+)"', block.group(1).replace(" ", ""))
        self.assertTrue(pairs, "no edges parsed")
        for a, b in pairs:
            self.assertIn(a, ids, f"edge starts at unknown node {a}")
            self.assertIn(b, ids, f"edge ends at unknown node {b}")

    def test_diagram_reaches_a_decision(self):
        """Every reader must lead somewhere; a dangling box teaches nothing."""
        block = re.search(r"const WB_EDGES = \[(.*?)\n\];", _panel(), re.S)
        pairs = re.findall(r'\["(\w+)","(\w+)"', block.group(1).replace(" ", ""))
        targets = {b for _, b in pairs}
        sources = {a for a, _ in pairs}
        for n in _nodes():
            if n["id"] == "decide":
                self.assertIn(n["id"], targets, "decision node has no input")
            else:
                self.assertIn(
                    n["id"], sources | targets,
                    f'node "{n["id"]}" is not connected to anything',
                )


@unittest.skipUnless(PANEL.is_file(), "panel/index.html missing")
class NavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = _panel()

    def test_white_box_tab_exists(self):
        self.assertIn('id="t-wb"', self.html)
        self.assertIn('id="p-wb"', self.html)

    def test_test_a_message_moved_to_the_footer(self):
        rail = re.search(r"<nav role=\"tablist\".*?</nav>", self.html, re.S)
        self.assertIsNotNone(rail)
        self.assertNotIn('id="t-test"', rail.group(0),
                         "Test a message is back in the rail")
        foot = re.search(r'<div class="foot">.*?</div>', self.html, re.S)
        self.assertIsNotNone(foot)
        self.assertIn('id="testBtn"', foot.group(0))

    def test_test_panel_is_still_reachable(self):
        """It has no rail button now, so show() and the footer must cover it."""
        self.assertIn('id="p-test"', self.html)
        self.assertIn('$("testBtn").onclick = () => show("test")', self.html)
        self.assertIn('"test"', self.html)

    def test_show_tolerates_a_missing_tab_button(self):
        self.assertIn("if (t) { t.setAttribute", self.html,
                      "show() would throw on the buttonless test panel")

    def test_arrow_keys_cycle_only_rail_tabs(self):
        self.assertIn("RAIL_TABS", self.html)
        self.assertIn('TABS.filter(n => n !== "test")', self.html)


if __name__ == "__main__":
    unittest.main()
