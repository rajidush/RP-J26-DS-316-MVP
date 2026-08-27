# C2 Analyst — text accuracy report

Generated: 2026-08-27T12:19:04+05:30
Corpus: **heldout** (40 cases)

CPU only. Development set from `analyst/evaluation/dataset.py` — a hand-built
set for catching regressions, **not** external validation. Milestone A3/A4
replaces these headline numbers with HateXplain / Jigsaw / CONDA.

## Persona 8-10 (theta = 0.55)

| scorer | accuracy | precision | recall | F1 | FP rate | p50 ms | p95 ms | load s |
|---|---|---|---|---|---|---|---|---|
| `cascade` | 67.5% | 88.9% | 40.0% | 55.2% | 5.0% | 62.05 | 71.42 | 1.00 |

## Persona 11-13 (theta = 0.65)

| scorer | accuracy | precision | recall | F1 | FP rate | p50 ms | p95 ms | load s |
|---|---|---|---|---|---|---|---|---|
| `cascade` | 65.0% | 87.5% | 35.0% | 50.0% | 5.0% | 62.05 | 71.42 | 1.00 |

## Persona 14-15 (theta = 0.75)

| scorer | accuracy | precision | recall | F1 | FP rate | p50 ms | p95 ms | load s |
|---|---|---|---|---|---|---|---|---|
| `cascade` | 65.0% | 87.5% | 35.0% | 50.0% | 5.0% | 62.05 | 71.42 | 1.00 |

## Per-bucket accuracy (persona 8-10)

| scorer | bullying | threat | implicit | sexual | obfuscated | gaming | figurative | reporting | benign |
|---|---|---|---|---|---|---|---|---|---|
| `cascade` | 60% | 0% | 60% | 0% | 50% | 100% | 75% | 100% | 100% |

## Remaining failures (persona 8-10)

### `cascade`

| kind | score | bucket | text |
|---|---|---|---|
| missed_hate | 0.08 | bullying | everyone would celebrate if you left this school |
| missed_hate | 0.08 | bullying | do everyone a favour and disappear |
| missed_hate | 0.08 | threat | im waiting outside for you after class |
| missed_hate | 0.08 | threat | youll regret ever talking to me |
| missed_hate | 0.45 | threat | ill make sure you never come back here |
| missed_hate | 0.08 | threat | bring your friends youll need them |
| missed_hate | 0.24 | implicit | those people are a disease on this country |
| missed_hate | 0.08 | implicit | the world would be cleaner without them |
| missed_hate | 0.08 | sexual | send a photo of yourself without a shirt |
| missed_hate | 0.08 | sexual | keep this between us dont mention it to mum |
| missed_hate | 0.08 | obfuscated | d1e already |
| missed_hate | 0.08 | obfuscated | unalive yourself |
| false_positive | 0.94 | figurative | my brother keeps annoying me i hate it |
