# Prism × nanoGPT — Results

> **First committed result (2026-07-18).** Every number below comes from one
> artifact — [`results/recipe_20260718T002717Z.json`](results/recipe_20260718T002717Z.json)
> — produced by `src/prism_eval.py` on an NVIDIA L4 (torch 2.13, CUDA) across
> three seeds (1337, 1338, 1339). It is reproducible from that file.
>
> This **supersedes and removes** the earlier tables (an "alpha run" with baseline
> 1.4636, a WikiText-2 "2.7×", "sprint 3.8–4.8× across seeds", "80+ runs", a
> "71% cross-data" figure). None had a committed artifact, and several were
> mechanically impossible — seeds were never varied in the old code, and the
> cross-data test never ran. They are deleted rather than quarantined.
>
> **Scope — read before the numbers.** This measures the Prism *recipe config*
> against nanoGPT's *default config*, not the spectral method against its absence.
> The recipe halves the learning rate (1e-3 → 5e-4) and adds a regularizer, so the
> improvement is **not yet attributable to Prism** — a schedule-matched control has
> not been run. And it is **same-data** (teacher and student share the 80% split),
> so even a clean control would not rule out content leakage. Two controls, in
> order, would settle it; neither has run (see below).

## The result — nanoGPT Shakespeare char-level (10.65M params)

Teacher trained 2000 steps; baseline and Prism-recipe students 5000 steps each,
evaluated every 100 steps on the held-out Shakespeare validation set. Three seeds.

| | Baseline | Prism Recipe |
|---|---|---|
| Best val loss (median of 3) | 1.782 | **1.656** |
| &nbsp;&nbsp;range across seeds | 1.778 – 1.785 | 1.655 – 1.658 |
| Val loss @ step 5000 | ~2.31 (overfit) | ~1.66 (stable) |
| Overfits within 5000 steps | **yes — all 3 seeds** | **no — 0 of 3** |
| Steps to baseline's best quality | — | **≤100 (≥13–14×, lower bound)** |

Per seed:

| seed | baseline best @step | baseline @5000 | recipe best @step | recipe @5000 | Prism Score |
|---|---|---|---|---|---|
| 1337 | 1.7783 @1400 | 2.3017 | 1.6577 @5000 | 1.6577 | ≥14× |
| 1338 | 1.7823 @1300 | 2.3225 | 1.6550 @3900 | 1.6634 | ≥13× |
| 1339 | 1.7854 @1400 | 2.3008 | 1.6557 @4700 | 1.6656 | ≥14× |

**Three findings, consistent across all three seeds** — and a reminder of what the
comparison actually is:

1. **Lower loss (directly measured, not censored).** The recipe reaches ~1.656 vs
   the baseline's best of ~1.782 — about **7% lower** best validation loss, every seed.
2. **No overfitting.** The baseline peaks near step 1350 and then degrades to
   ~2.31 by step 5000 on all three seeds. The recipe holds near its best and
   overfits on none.
3. **Faster to parity — but a lower bound.** The recipe crosses the baseline's
   *best* loss by the first eval (step 100) on all three seeds, so the Prism Score
   of 13–14× is **left-censored**: the true crossing lies below step 100 and is not
   resolved by this eval. Read 13–14× as a floor, not a point estimate.

**But the baseline here uses LR 1e-3 and the recipe uses 5e-4** (plus the mod-wheel
regularizer). Overfitting from 1.78 to 2.30 is the classic signature of a
learning rate too high for a tiny dataset — so findings 1 and 2 may be the
schedule, not the spectral method. That is the first thing the controls below test.

## Reproduce it

```bash
git clone https://github.com/timepointai/nanogpt-prism-shakespeare.git
cd nanogpt-prism-shakespeare/src
pip install torch numpy transformers tiktoken datasets
python prism_eval.py                 # seeds 1337,1338,1339; writes results/*.json
python prism_eval.py --report        # reprint the last artifact
```

The run is stepwise and resumable — each seed's stages bank to disk as they finish
— and it writes a full artifact (loss curves, git commit, GPU, argv) to `results/`.
The rule: a number in this file must have a matching `results/*.json`.

## What is NOT established

- **That Prism caused it.** The recipe differs from the baseline by learning rate
  and a regularizer *as well as* the spectral method, so the improvement may be the
  schedule. Nothing here isolates the spectral transfer. This is the biggest gap.
- **Structure vs. content.** Even with a clean control, same-data transfer cannot
  separate "Prism moved organizational structure" from "the teacher handed over
  answers." Only the cross-data test can.
- **Anti-overfitting vs. generic regularization.** The mod wheel is a regularizer;
  it has not been compared against tuned dropout / weight decay / early stopping.
- **The exact speedup.** Left-censored (above), and measured against a baseline
  that may be hobbled by too high a learning rate. Resolving it needs dense early
  evaluation (steps 10 / 25 / 50 / 75).
- **Scale.** Shakespeare is ~1M characters, 10.65M params. Nothing here has been
  tested at GPT-2 124M / OpenWebText scale.

## The two controls that decide it (in order)

**1. Schedule-matched control — does the spectral method do anything?**
Run the baseline again with `prism_init=False` but at the recipe's schedule
(learning rate 5e-4, warmup 50). If it also lands near ~1.66 and stops overfitting,
the result above was the learning rate and Prism adds nothing measurable here. If
it still overfits near ~1.78, the method is doing something on top of the schedule.
This is cheap and it comes first — nothing below matters until it passes.

**2. Cross-data test — is what it does structural, or a content leak?**
Only if control 1 passes. Extract the teacher's fingerprint from one half of the
training data and train the student on the disjoint other half, where no shared
content remains. If the advantage survives, the transfer is structural — and every
checkpoint ever trained holds a reusable prior nobody extracted. If it collapses,
Prism is distillation with extra steps.

## How the method works

**Spectral Imprint** — SVD each teacher weight matrix, group by type (attention,
FFN up, FFN down, embedding), average the singular-value distributions per group,
compress each to 8 DCT coefficients (~128 bytes total), and reshape the student's
spectrum to match.

**EigenTransfer** — blend the student's singular vectors 75% toward the teacher's,
then re-orthogonalize. The student starts with the teacher's directional scaffolding.

**Mod Wheel** — after each optimizer step, `W ← (1 − s)·W + s·W_target`, with
`s = 0.01` decaying by `0.9999` per step. A continuous, zero-storage spectral
regularizer.

The 128-byte figure is the spectrum only. The directional matrices (U, V) are
~500 MB uncompressed; compressing them is an open problem, so "128 bytes"
describes one tier of the method, not the whole thing.
