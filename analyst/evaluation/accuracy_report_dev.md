# C2 Analyst — text accuracy report

Generated: 2026-08-27T11:00:39+05:30
Corpus: **dev** (67 cases)

CPU only. Development set from `analyst/evaluation/dataset.py` — a hand-built
set for catching regressions, **not** external validation. Milestone A3/A4
replaces these headline numbers with HateXplain / Jigsaw / CONDA.

## Persona 8-10 (theta = 0.55)

| scorer | accuracy | precision | recall | F1 | FP rate | p50 ms | p95 ms | load s |
|---|---|---|---|---|---|---|---|---|
| `cascade` | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 63.92 | 77.75 | 1.03 |
| `lexicon` | 94.0% | 89.2% | 100.0% | 94.3% | 11.8% | 0.77 | 0.95 | 0.00 |
| `text_fast` | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 65.33 | 77.62 | 0.00 |
| `toxic-comment-` | 58.2% | 85.7% | 18.2% | 30.0% | 2.9% | 16.02 | 18.74 | 3.10 |

## Persona 11-13 (theta = 0.65)

| scorer | accuracy | precision | recall | F1 | FP rate | p50 ms | p95 ms | load s |
|---|---|---|---|---|---|---|---|---|
| `cascade` | 86.6% | 100.0% | 72.7% | 84.2% | 0.0% | 63.92 | 77.75 | 1.03 |
| `lexicon` | 74.6% | 90.0% | 54.5% | 67.9% | 5.9% | 0.77 | 0.95 | 0.00 |
| `text_fast` | 86.6% | 100.0% | 72.7% | 84.2% | 0.0% | 65.33 | 77.62 | 0.00 |
| `toxic-comment-` | 58.2% | 100.0% | 15.2% | 26.3% | 0.0% | 16.02 | 18.74 | 3.10 |

## Persona 14-15 (theta = 0.75)

| scorer | accuracy | precision | recall | F1 | FP rate | p50 ms | p95 ms | load s |
|---|---|---|---|---|---|---|---|---|
| `cascade` | 85.1% | 100.0% | 69.7% | 82.1% | 0.0% | 63.92 | 77.75 | 1.03 |
| `lexicon` | 73.1% | 89.5% | 51.5% | 65.4% | 5.9% | 0.77 | 0.95 | 0.00 |
| `text_fast` | 85.1% | 100.0% | 69.7% | 82.1% | 0.0% | 65.33 | 77.62 | 0.00 |
| `toxic-comment-` | 58.2% | 100.0% | 15.2% | 26.3% | 0.0% | 16.02 | 18.74 | 3.10 |

## Per-bucket accuracy (persona 8-10)

| scorer | explicit | implicit | threat | bullying | sexual | obfuscated | gaming | reporting | figurative | benign |
|---|---|---|---|---|---|---|---|---|---|---|
| `cascade` | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| `lexicon` | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 43% | 100% | 100% |
| `text_fast` | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| `toxic-comment-` | 43% | 0% | 0% | 40% | 0% | 20% | 100% | 100% | 86% | 100% |

## Remaining failures (persona 8-10)

### `cascade`

_None._

### `lexicon`

| kind | score | bucket | text |
|---|---|---|---|
| false_positive | 0.88 | reporting | you should never tell someone to kys thats awful |
| false_positive | 0.88 | reporting | miss he keeps calling me a retard what do i do |
| false_positive | 0.64 | reporting | is it bullying if someone says nobody likes you |
| false_positive | 0.58 | reporting | mum someone wrote go back to your country on my locker |

### `text_fast`

_None._

### `toxic-comment-`

| kind | score | bucket | text |
|---|---|---|---|
| missed_hate | 0.04 | explicit | you should kys |
| missed_hate | 0.06 | explicit | go die in a hole |
| missed_hate | 0.48 | explicit | you're a retard and everyone knows it |
| missed_hate | 0.09 | explicit | hang yourself you waste of space |
| missed_hate | 0.02 | implicit | all of them are animals and should be caged |
| missed_hate | 0.05 | implicit | people like you shouldn't be allowed in this country |
| missed_hate | 0.01 | implicit | their kind always ruins everything they touch |
| missed_hate | 0.00 | implicit | we'd all be better off without your sort around here |
| missed_hate | 0.13 | implicit | they breed like vermin and take our jobs |
| missed_hate | 0.00 | implicit | everyone would be happier if you were gone |
| missed_hate | 0.01 | implicit | nobody wants you here go away |
| missed_hate | 0.00 | implicit | why do you even bother showing up |
| missed_hate | 0.00 | threat | i'm going to hurt you after school |
| missed_hate | 0.00 | threat | i'll find you and make you regret it |
| missed_hate | 0.01 | threat | watch your back tomorrow |
| missed_hate | 0.02 | threat | me and my friends are gonna jump you |
| missed_hate | 0.00 | threat | i know where you live and i'm coming |
| missed_hate | 0.01 | bullying | everyone in class laughs at you behind your back |
| missed_hate | 0.15 | bullying | you're so ugly it hurts to look at you |
| missed_hate | 0.01 | bullying | no wonder you have no friends |
| missed_hate | 0.00 | sexual | send me pics or i'll tell everyone |
| missed_hate | 0.00 | sexual | take your clothes off for the camera |
| missed_hate | 0.00 | sexual | don't tell your parents this is our secret |
| missed_hate | 0.07 | obfuscated | you should k y s |
| missed_hate | 0.03 | obfuscated | kiiiill yourself |
| missed_hate | 0.00 | obfuscated | g0 d1e already |
| missed_hate | 0.01 | obfuscated | n o b o d y likes you |
| false_positive | 0.60 | figurative | this homework is killing me |
