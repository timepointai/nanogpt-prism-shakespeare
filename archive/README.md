# Archive

Superseded documentation, kept verbatim for the record. Not maintained.

## v0.0 — before attribution

[`v0.0/`](v0.0/) holds `README.md`, `WHITEPAPER.md`, and `RESULTS.md` as they stood
**before the learning-rate confound was ruled out**.

v0.0 reported that the Prism recipe reached a lower loss and did not overfit — but
the recipe also lowered the learning rate versus the baseline (1e-3 → 5e-4), so the
comparison was "recipe config vs. default config," not "the spectral method vs. its
absence." A too-high baseline learning rate alone could have produced the whole
picture, and v0.0 said so honestly (it led with "a reproducible effect, not yet
attributed to the method").

**What changed in v0.1.** The control was run: the recipe at the baseline's *own*
learning rate, only the spectral flags different. The effect survived — lower loss,
no overfitting, ~7× faster to the baseline's best (a resolved number, not a censored
lower bound). So the effect is attributable to the spectral method. See the current
top-level [README.md](../README.md) and
[`results/recipe_20260720T230405Z.json`](../results/recipe_20260720T230405Z.json).

(The archived Markdown references `assets/*.svg` with repo-relative paths, so inline
images resolve from the repo root, not from inside the archive.)
