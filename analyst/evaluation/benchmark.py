"""Milestone A4 measurement harness — accuracy, false positives, latency, RAM.

    python -m analyst.evaluation.benchmark                  # current stack
    python -m analyst.evaluation.benchmark --scorer lexicon # ablation
    python -m analyst.evaluation.benchmark --compare        # every scorer, one table

Everything runs CPU-only against `dataset.py`. The harness takes a *scorer*
(any callable text -> score) so ablations, candidate models and the shipped
cascade are all measured the same way — no separate code path can flatter one
of them.

The headline number is deliberately not accuracy. For a tool that interrupts a
child, the false-positive rate on `gaming` / `reporting` / `figurative` matters
more: those are the cases where a wrong alert teaches the child that ordinary
play, or asking an adult for help, triggers surveillance.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analyst.evaluation import heldout
from analyst.evaluation.dataset import TextCase, all_cases, buckets

# Which corpus the harness is pointed at. `dev` designed the detectors, so only
# `heldout` speaks to generalisation — see heldout.py.
CORPORA = ("dev", "heldout", "both")
_active_cases = all_cases


def set_corpus(name: str) -> None:
    global _active_cases, _corpus_name
    _corpus_name = name
    if name == "heldout":
        _active_cases = heldout.all_cases
    elif name == "both":
        _active_cases = lambda: [*all_cases(), *heldout.all_cases()]
    else:
        _active_cases = all_cases


def cases():
    return _active_cases()


def active_buckets():
    seen = []
    for case in cases():
        if case.bucket not in seen:
            seen.append(case.bucket)
    return seen

_REPORT_DIR = Path(__file__).resolve().parent
_corpus_name = "dev"


def report_path() -> Path:
    return _REPORT_DIR / f"accuracy_report_{_corpus_name}.md"

# Persona thresholds under test (analyst.decide.PERSONA_THETA2).
PERSONAS: Dict[str, float] = {"8-10": 0.55, "11-13": 0.65, "14-15": 0.75}

# A false positive here is worse than elsewhere — see module docstring.
SENSITIVE_BUCKETS = ("gaming", "reporting", "figurative")

Scorer = Callable[[str], float]


# --- metrics -----------------------------------------------------------------


class Confusion:
    def __init__(self) -> None:
        self.tp = self.fp = self.tn = self.fn = 0

    def add(self, predicted_hate: bool, actual_hate: bool) -> None:
        if predicted_hate and actual_hate:
            self.tp += 1
        elif predicted_hate and not actual_hate:
            self.fp += 1
        elif not predicted_hate and actual_hate:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def fp_rate(self) -> float:
        """Share of genuinely safe content that was wrongly flagged."""
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0

    def as_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fp_rate": round(self.fp_rate, 4),
        }


class Result:
    def __init__(self, name: str) -> None:
        self.name = name
        self.scores: List[Tuple[TextCase, float]] = []
        self.latencies_ms: List[float] = []
        self.peak_mb = 0.0
        self.load_s = 0.0

    def confusion(self, theta: float, only_bucket: Optional[str] = None) -> Confusion:
        c = Confusion()
        for case, score in self.scores:
            if only_bucket and case.bucket != only_bucket:
                continue
            c.add(score >= theta, case.hate)
        return c

    def failures(self, theta: float) -> List[Tuple[TextCase, float, str]]:
        out = []
        for case, score in self.scores:
            predicted = score >= theta
            if predicted != case.hate:
                out.append((case, score, "false_positive" if predicted else "missed_hate"))
        return out

    @property
    def p50(self) -> float:
        return round(statistics.median(self.latencies_ms), 2) if self.latencies_ms else 0.0

    @property
    def p95(self) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return round(ordered[idx], 2)


def _peak_rss_mb() -> float:
    """Process RSS if psutil is available; else 0.0 (reported as n/a)."""
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return 0.0


def run_scorer(name: str, build: Callable[[], Scorer], warmup: bool = True) -> Result:
    result = Result(name)

    gc.collect()
    rss_before = _peak_rss_mb()
    t0 = time.perf_counter()
    scorer = build()
    result.load_s = round(time.perf_counter() - t0, 2)

    if warmup:  # first call pays lazy init; keep it out of the latency stats
        try:
            scorer("warmup text")
        except Exception:
            pass

    tracemalloc.start()
    for case in cases():
        t = time.perf_counter()
        try:
            score = float(scorer(case.text))
        except Exception:
            score = 0.0
        result.latencies_ms.append((time.perf_counter() - t) * 1000)
        result.scores.append((case, score))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rss_after = _peak_rss_mb()
    result.peak_mb = round(rss_after - rss_before, 1) if rss_after and rss_before else round(peak / 1e6, 1)
    return result


# --- scorers under test ------------------------------------------------------


def _lexicon_scorer() -> Scorer:
    from analyst.stage1.lexicon import score_text

    return lambda t: score_text(t)[0]


def _text_fast_scorer() -> Scorer:
    from analyst.stage1.text_fast import TextFast

    tf = TextFast()
    return lambda t: tf.score(t)[0]


def _cascade_scorer() -> Scorer:
    """Full Stage-1 + Stage-2 text path, exactly as the pipeline runs it."""
    from analyst.stage1.text_fast import TextFast
    from analyst.stage2.fusion import TextFull

    tf = TextFast()
    full = TextFull()

    def score(text: str) -> float:
        s1, _cat, _hits = tf.score(text)
        return full.score(text, s1)

    return score


SCORERS: Dict[str, Callable[[], Scorer]] = {
    "lexicon": _lexicon_scorer,
    "text_fast": _text_fast_scorer,
    "cascade": _cascade_scorer,
}


# Label vocabularies differ per model, so map by name rather than by index.
# Anything unrecognised is ignored instead of guessed — inventing a score from
# an unknown label is how a model silently scores everything 0.5.
_HATE_LABELS = {
    "hate", "hateful", "hate_speech", "hatespeech", "toxic", "severe_toxic",
    "obscene", "threat", "insult", "identity_hate", "identity_attack",
    "offensive", "abusive", "harassment", "sexual_explicit", "label_1", "1",
}
_SAFE_LABELS = {
    "nothate", "not_hate", "not-hate", "non_toxic", "non-toxic", "nontoxic",
    "neutral", "benign", "normal", "clean", "ok", "label_0", "0",
}


def hate_score_from_labels(rows: List[dict]) -> Optional[float]:
    """Collapse a model's label distribution into one hate probability.

    Multi-label heads (toxic-bert) emit independent sigmoid scores, so the max
    over harmful labels is the right reduction. Single-label heads emit a
    softmax, where the safe-label complement is equivalent. Returns None when
    no label is recognised, so callers can fall back rather than fabricate.
    """
    hate = [float(r["score"]) for r in rows if _norm_label(r.get("label")) in _HATE_LABELS]
    if hate:
        return max(hate)
    safe = [float(r["score"]) for r in rows if _norm_label(r.get("label")) in _SAFE_LABELS]
    if safe:
        return round(1.0 - max(safe), 4)
    return None


def _norm_label(label) -> str:
    return str(label or "").strip().lower().replace(" ", "_").replace("-", "_")


def hf_scorer(model_id: str) -> Scorer:
    """Any HF text-classification model as a plain text -> score callable."""
    from transformers import pipeline

    pipe = pipeline(
        "text-classification",
        model=model_id,
        tokenizer=model_id,
        device=-1,  # CPU only — the whole project is a CPU claim
        truncation=True,
        max_length=128,
        top_k=None,  # all labels, so multi-label heads are handled properly
    )

    def score(text: str) -> float:
        if not (text or "").strip():
            return 0.0
        out = pipe(text[:512])
        rows = out[0] if out and isinstance(out[0], list) else out
        value = hate_score_from_labels(rows or [])
        return float(value) if value is not None else 0.0

    return score


# --- reporting ---------------------------------------------------------------


def print_report(results: List[Result], theta_name: str = "8-10") -> None:
    theta = PERSONAS[theta_name]
    print(f"\n=== Accuracy @ persona {theta_name} (theta={theta}) ===\n")
    head = f"{'scorer':<14} {'acc':>6} {'prec':>6} {'rec':>6} {'F1':>6} {'FP%':>6} {'p50ms':>8} {'p95ms':>8} {'RAM MB':>8}"
    print(head)
    print("-" * len(head))
    for r in results:
        c = r.confusion(theta)
        print(
            f"{r.name:<14} {c.accuracy:>6.1%} {c.precision:>6.1%} {c.recall:>6.1%} "
            f"{c.f1:>6.1%} {c.fp_rate:>6.1%} {r.p50:>8.2f} {r.p95:>8.2f} {r.peak_mb:>8.1f}"
        )

    print(f"\n=== Per-bucket accuracy @ {theta_name} ===\n")
    bucket_list = active_buckets()
    print(f"{'scorer':<14}" + "".join(f"{b[:9]:>11}" for b in bucket_list))
    print("-" * (14 + 11 * len(bucket_list)))
    for r in results:
        row = f"{r.name:<14}"
        for b in bucket_list:
            c = r.confusion(theta, only_bucket=b)
            row += f"{c.accuracy:>10.0%} "
        print(row)

    print(f"\n=== Sensitive-bucket false positives @ {theta_name} ===")
    print("(gaming / reporting / figurative — a wrong alert here is the costly kind)\n")
    for r in results:
        fps = [
            (case, score)
            for case, score, kind in r.failures(theta)
            if kind == "false_positive" and case.bucket in SENSITIVE_BUCKETS
        ]
        print(f"{r.name}: {len(fps)} false positive(s)")
        for case, score in fps[:6]:
            print(f"    {score:.2f}  [{case.bucket}] {case.text}")

    print(f"\n=== Missed hate @ {theta_name} ===\n")
    for r in results:
        misses = [(c, s) for c, s, k in r.failures(theta) if k == "missed_hate"]
        print(f"{r.name}: {len(misses)} missed")
        for case, score in misses[:8]:
            print(f"    {score:.2f}  [{case.bucket}] {case.text}")


SWEEP_THETAS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.75)


def print_sweep(results: List[Result]) -> None:
    """Precision/recall across thresholds.

    Stage 1 is judged at theta1 = 0.35 (escalation gate — recall is what
    matters, a miss here can never be recovered downstream). Stage 2 is judged
    at the persona thresholds, where precision governs.
    """
    for r in results:
        print(f"\n=== {r.name} — threshold sweep ({len(cases())} cases) ===\n")
        print(f"{'theta':>6} {'acc':>7} {'prec':>7} {'rec':>7} {'F1':>7} {'FP%':>7}   note")
        print("-" * 60)
        for theta in SWEEP_THETAS:
            c = r.confusion(theta)
            note = ""
            if abs(theta - 0.35) < 1e-9:
                note = "<- theta1 stage-1 gate"
            elif theta in (0.55, 0.65, 0.75):
                note = "<- persona theta2"
            print(f"{theta:>6.2f} {c.accuracy:>7.1%} {c.precision:>7.1%} {c.recall:>7.1%} "
                  f"{c.f1:>7.1%} {c.fp_rate:>7.1%}   {note}")


def write_report(results: List[Result]) -> None:
    lines: List[str] = [
        "# C2 Analyst — text accuracy report",
        "",
        f"Generated: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        f"Corpus: **{_corpus_name}** ({len(cases())} cases)",
        "",
        "CPU only. Development set from `analyst/evaluation/dataset.py` — a hand-built",
        "set for catching regressions, **not** external validation. Milestone A3/A4",
        "replaces these headline numbers with HateXplain / Jigsaw / CONDA.",
        "",
    ]
    for theta_name, theta in PERSONAS.items():
        lines += [
            f"## Persona {theta_name} (theta = {theta})",
            "",
            "| scorer | accuracy | precision | recall | F1 | FP rate | p50 ms | p95 ms | load s |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in results:
            c = r.confusion(theta)
            lines.append(
                f"| `{r.name}` | {c.accuracy:.1%} | {c.precision:.1%} | {c.recall:.1%} | "
                f"{c.f1:.1%} | {c.fp_rate:.1%} | {r.p50:.2f} | {r.p95:.2f} | {r.load_s:.2f} |"
            )
        lines.append("")

    theta = PERSONAS["8-10"]
    bl = active_buckets()
    lines += ["## Per-bucket accuracy (persona 8-10)", "",
              "| scorer | " + " | ".join(bl) + " |",
              "|---" * (len(bl) + 1) + "|"]
    for r in results:
        cells = " | ".join(f"{r.confusion(theta, only_bucket=b).accuracy:.0%}" for b in bl)
        lines.append(f"| `{r.name}` | {cells} |")
    lines.append("")

    lines += ["## Remaining failures (persona 8-10)", ""]
    for r in results:
        lines.append(f"### `{r.name}`")
        lines.append("")
        fails = r.failures(theta)
        if not fails:
            lines += ["_None._", ""]
            continue
        lines += ["| kind | score | bucket | text |", "|---|---|---|---|"]
        for case, score, kind in fails:
            safe_text = case.text.replace("|", "\\|")
            lines.append(f"| {kind} | {score:.2f} | {case.bucket} | {safe_text} |")
        lines.append("")

    out = report_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written -> {out}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="C2 Analyst accuracy / latency / RAM benchmark")
    p.add_argument("--corpus", default="dev", choices=CORPORA, help="dev (tuned on) | heldout (never tuned on) | both")
    p.add_argument("--scorer", default="cascade", choices=sorted(SCORERS), help="single scorer to run")
    p.add_argument("--compare", action="store_true", help="run every registered scorer")
    p.add_argument(
        "--hf",
        action="append",
        default=[],
        metavar="MODEL_ID",
        help="also benchmark a raw HF text-classification model (repeatable)",
    )
    p.add_argument("--persona", default="8-10", choices=sorted(PERSONAS), help="persona for the console table")
    p.add_argument("--sweep", action="store_true",
                   help="print a threshold sweep — the cascade's stage-1 gate (0.35) and "
                        "decision thresholds trade off differently, so one theta hides the picture")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of tables")
    p.add_argument("--no-report", action="store_true", help="skip writing the markdown report")
    args = p.parse_args(argv)

    set_corpus(args.corpus)
    names = sorted(SCORERS) if args.compare else ([args.scorer] if not args.hf else [])
    results = [run_scorer(n, SCORERS[n]) for n in names]
    for model_id in args.hf:
        short = model_id.split("/")[-1][:14]
        results.append(run_scorer(short, lambda m=model_id: hf_scorer(m)))

    if args.json:
        print(json.dumps(
            {
                r.name: {
                    "personas": {k: r.confusion(v).as_dict() for k, v in PERSONAS.items()},
                    "p50_ms": r.p50, "p95_ms": r.p95, "peak_mb": r.peak_mb, "load_s": r.load_s,
                }
                for r in results
            },
            indent=2,
        ))
    elif args.sweep:
        print_sweep(results)
    else:
        print_report(results, args.persona)

    if not args.no_report:
        write_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
