# results/

Eval artifacts. **This directory is the evidence.** Two committed runs:

- `recipe_20260718T002717Z.json` — recipe at LR 5e-4 vs. baseline at LR 1e-3
  (the recipe as tuned; best loss ~1.66, no overfitting, but confounded by the LR
  difference).
- `recipe_20260720T230405Z.json` — the **attribution control**: recipe at the
  baseline's own LR 1e-3, `schedule_matched: true`, only the spectral flags differ.
  Best ~1.67, no overfitting, ~7× faster — so the effect is the spectral method,
  not the learning rate.

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
