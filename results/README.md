# results/

Eval artifacts. **This directory is the evidence.** Committed runs:

- `recipe_20260718T002717Z.json` — recipe at LR 5e-4 vs. baseline at LR 1e-3
  (the recipe as tuned; best loss ~1.66, no overfitting, but confounded by the LR
  difference; score left-censored at ≥13–14×).
- `recipe_20260720T230405Z.json` — the **attribution control**: recipe at the
  baseline's own LR 1e-3, `schedule_matched: true`, only the spectral flags differ.
  Best ~1.67, no overfitting, ~7× faster — so the effect is the spectral method,
  not the learning rate.
- `recipe_20260721T022342Z.json` — sliding-window overlap sweep. **Record only:
  its interpretation is retired** — window position correlated with slice
  difficulty (baseline drifted 1.88→1.55 across the sweep), so the apparent
  overlap dependence was a confound. Superseded by the next artifact.
- `recipe_20260721T050203Z.json` — difficulty-controlled overlap probe (random
  blocks, 12 overlaps × 3 seeds, 100 steps, matched LR): the early advantage is
  **flat** across overlap 1.0 → 0.0 (Δloss ~0.57–0.59, ~23%) — the head start
  does not depend on shared teacher/student content, within-domain.
- `recipe_20260721T142104Z.json` — **un-censored speed**: dense eval (every 10)
  over 1,500 steps, tuned recipe. Score resolved: **11.8× median (10.2–11.9×)**,
  `left_censored: false`. Crossover ~step 100 vs. baseline best ~step 1,020–1,400.
- `recipe_20260721T143246Z.json` — **teacher-strength sweep** (100→2,000 teacher
  steps, 300-step students): advantage is monotonic in teacher strength and
  saturating at ≈2,000 (see the saturation run below); a 100-step teacher is
  *actively worse than random init* (method 2.249 vs. baseline 2.180).
- `recipe_20260721T172238Z.json` — **teacher saturation** (2k/4k/8k): Δloss
  +0.458 / +0.465 / +0.456 — a plateau at ≈+0.46 from 2,000 steps on, right
  where the teacher itself converges. The 2k anchor reproduces the sweep's
  +0.451.
- `recipe_20260721T153218Z.json` — far-corpus sweep (student's fresh blocks from
  Sherlock Holmes, token-JS up to 0.027): Δloss flat across distance. **Scope
  caveat:** the student is scored on the *Shakespeare* val set in this protocol,
  so this shows the teacher-geometry init isn't washed out by far-domain
  training — it does not yet measure accelerated learning *of* the far domain.
  Superseded by the `far_val` run below.
- `recipe_20260721T162552Z.json` — lever eval, **grassmann alignment alone**
  (single variable vs. the plain `far_val` control): Δloss −0.012..−0.023 at
  every overlap — the geodesic pairing eliminates the head start.
- `recipe_20260721T164007Z.json` — lever eval, **top-k=128 alone**: head start
  survives (Δloss +0.54..+0.57) but uniformly ~0.05 worse than the plain
  full-direction blend; the discarded tail carries useful geometry.
- `recipe_20260721T161208Z.json` — **the cross-domain result** (`far_val: true`):
  same sweep, but each student is scored on a val set mirroring its *own* train
  mixture (pure held-out Sherlock at overlap 0.0). Sanity gate: the overlap-1.0
  row reproduces the base protocol exactly. Δloss is flat-to-growing across
  distance — 0.591 at overlap 1.0 → **0.627 at overlap 0.0** (recipe 1.786 vs.
  baseline 2.414 on Sherlock val, 3 seeds, matched LR): a Shakespeare teacher's
  geometry accelerates learning *of Sherlock* as much as it accelerates
  Shakespeare. Scores ≥9–10× at every overlap (left-censored at this probe's
  resolution). The head start is structural, and it transfers across domains.

See [RESULTS.md](../RESULTS.md) for the full comparison and what remains open.

Every file here is written by `src/prism_eval.py` and contains, for each seed:
the full baseline and method loss curves, best loss and step, the Prism Score
and whether it is left-censored, wall time, plus provenance — git commit, dirty
flag, GPU, torch version, and the exact argv.

## The rule

**A number in README.md, WHITEPAPER.md, or RESULTS.md must have a matching
`results/*.json`.** If it doesn't, it isn't a result — it's a note.

This rule exists because the project already failed this way once. The published
13x Prism Score, the 1.7704 / 1.6498 val losses, and the 71% cross-data figure
have no source in this repository: the notebooks were saved without outputs and
every run wrote to `/content/...` in Colab, which evaporated with the VM. The
numbers survived in prose; the evidence did not. Two documents ended up
reporting different results for the same experiment, and nobody could tell which
was real.

## Generating one

```bash
cd src
python prism_eval.py                    # seeds 1337,1338,1339 — ~80 min on an L4
git add ../results/recipe_*.json        # commit the artifact with the claim
```

From Colab, `nanogpt_prism_eval.ipynb` runs the same thing and downloads the
artifact at the end. Save the notebook **with outputs**.

## Reading an artifact

- `summary.prism_score` — median and range across seeds. A single seed is a
  sample, not a result.
- `summary.baseline_best` — **read this first.** The Prism Score is a ratio; a
  weak baseline inflates it without the method improving. If the baseline here
  is worse than a previous run's, the score is not comparable.
- `runs[].score.left_censored` — `true` means the method hit target at the first
  eval, so the score is a lower bound and the true crossing is unresolved below
  `config.eval_every` steps.
- `config.teacher_data_equals_student_data` — `true` for the standard eval. The
  teacher and student share a training split, so this measures same-data
  transfer and does **not** rule out content leakage. The cross-data test is the
  one that would.
