# Prism × nanoGPT — Results (v0.1)

Two committed runs, three seeds each, on an NVIDIA L4. Every number below is
reproducible from an artifact in [`results/`](results/). The earlier writeup, from
before the confound was ruled out, is archived at
[`archive/v0.0/RESULTS.md`](archive/v0.0/RESULTS.md).

## The two runs

**Run A — the recipe as tuned** ([`recipe_20260718T002717Z.json`](results/recipe_20260718T002717Z.json)):
recipe at LR 5e-4 vs. baseline at the config default LR 1e-3.

**Run B — the attribution control** ([`recipe_20260720T230405Z.json`](results/recipe_20260720T230405Z.json)):
recipe at LR **1e-3**, matched to the baseline. `schedule_matched: true` — the two
arms differ by *nothing but the spectral flags*.

Teacher 2,000 steps; students 5,000 steps, eval every 100, held-out Shakespeare
validation set.

| | Baseline (LR 1e-3) | Recipe (LR 5e-4) | Recipe (LR 1e-3, matched) |
|---|---|---|---|
| Best val loss (median of 3) | 1.782 | 1.656 | **1.671** |
| &nbsp;&nbsp;range across seeds | 1.781–1.783 | 1.655–1.658 | 1.671–1.674 |
| Val loss @ step 5,000 | ~2.31 (overfit) | ~1.66 (stable) | ~1.67 (stable) |
| Overfits within 5,000 steps | yes — 3/3 | no — 0/3 | no — 0/3 |
| Prism Score (steps to baseline's best) | — | ≥13–14× (left-censored) | **7× (median; measured, not censored)** |

Per seed, matched-LR run (Run B):

| seed | baseline best @step | baseline @5000 | recipe best @step | recipe @5000 | Prism Score |
|---|---|---|---|---|---|
| 1337 | 1.7822 @1400 | 2.3075 | 1.6709 @5000 | 1.6709 | 7.0× (hit @200) |
| 1338 | 1.7832 @1400 | 2.3062 | 1.6742 @5000 | 1.6742 | 7.0× (hit @200) |
| 1339 | 1.7806 @1300 | 2.3341 | 1.6712 @4700 | 1.6809 | 6.5× (hit @200) |

## What this establishes

**The effect is attributable to the spectral method, not the learning rate.** Run B
holds the learning rate identical to the baseline and toggles only the spectral
flags. At that matched schedule, where the baseline overfits and decays to ~2.31,
the recipe reaches ~1.67 and holds, and crosses the baseline's best quality ~7×
faster. So the "maybe it was just the lower learning rate" explanation is closed.

Two supporting points:

- **Two learning rates both favor Prism.** The recipe is slightly better at 5e-4
  (1.656) than at 1e-3 (1.671), so 5e-4 is mildly better-tuned for it — but it wins
  at both, which is stronger than a single operating point.
- **The speed number is now measured.** At 5e-4 the recipe crossed the baseline's
  best before the first eval (step 100), so the score was a left-censored lower
  bound. At 1e-3 the crossover moved to step 200, so the eval resolved it: 7×, a
  real measurement.

## What this does NOT establish

- **Structure vs. content.** Both runs are same-data (teacher and student share the
  80% split). This shows Prism transfers something real and useful; it does not show
  that what transfers is *structure* rather than the teacher leaking *content*. Only
  the cross-data test can (below).
- **Spectral vs. generic regularization.** The mod wheel is a regularizer. We've
  shown it beats a schedule-matched baseline, but not that the *spectral* nature
  specifically (vs. tuned dropout / weight decay / early stopping) is what prevents
  overfitting. The `spectral_only` / `dirs_only` arms and a reg-matched baseline
  address this.
- **A full LR sweep.** Two shared operating points is not the gold standard of
  comparing each arm at *its own best* learning rate. The matched-LR comparison is
  the decisive one for attribution, but a small sweep would make it airtight.
- **Scale.** Shakespeare only, 10.65M params.

## Reproduce it

```bash
cd src
pip install torch numpy transformers tiktoken datasets
python prism_eval.py                                      # Run A: recipe @ 5e-4
python prism_eval.py --method_lr=1e-3 --method_warmup=100 # Run B: matched — only spectral flags differ
python prism_eval.py --report                             # reprint the last artifact
```

The eval is stepwise and resumable (each seed's stages bank to disk), records
provenance and whether the schedule was matched, and refuses to score a partial or
crashed run. A number in these docs must have a matching `results/*.json`.

## The next experiments, in order

1. **Cross-data** — fingerprint the teacher on one half of the training data, train
   the student on the disjoint other half. If the advantage survives, the transfer
   is structural. If it collapses, Prism is distillation with extra steps. The one
   that decides how important this is.
2. **Endurance** — take the recipe well past 5,000 steps (20k–50k) to find where, if
   ever, it finally overfits. Either it holds (the ceiling is genuinely removed) or
   it breaks at a measurable step *N*.
3. **Ablations** — `spectral_only`, `dirs_only`, and a regularization-matched
   baseline, to separate the spectral contribution from generic regularization.
4. **Scale** — GPT-2 124M on OpenWebText.

## How the method works

**Spectral Imprint** — SVD each teacher weight matrix, group by type, average the
singular-value distributions per group, compress each to 8 DCT coefficients
(~128 bytes total), reshape the student's spectrum to match.

**EigenTransfer** — blend the student's singular vectors 75% toward the teacher's,
re-orthogonalize.

**Mod Wheel** — after each optimizer step, `W ← (1 − s)·W + s·W_target`, with
`s = 0.01` decaying by `0.9999` per step. A continuous, zero-storage spectral
regularizer.

The 128-byte figure is the spectrum only; the directional matrices (U, V) are
~500 MB uncompressed and compressing them is an open problem.
