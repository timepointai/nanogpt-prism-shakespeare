# Archive

Superseded documentation and exploratory work, kept verbatim for the record. Not
maintained.

## experiments/ and notebooks/ — pre-ground-truth exploration

[`experiments/`](experiments/) holds the exploratory notebooks, plans, and scripts
from before the evidence-first rewrite — the "80+ runs" era. They contain **no
committed results** (the notebooks were saved without executed outputs; every run
wrote to ephemeral Colab paths), so they are exploratory history only, not
evidence. [`notebooks/`](notebooks/) holds the two root-level Colab eval notebooks
(`nanogpt_prism_eval.ipynb`, `nanogpt_prism_eval_gpt2.ipynb`), superseded by the
Modal runners (`prism_modal.py`) and the local `src/prism_eval.py`. None of this is
referenced by the current v0.2 docs; the authoritative path is the top-level
[README.md](../README.md) and the committed [`results/`](../results/) artifacts.

## v0.1 — attribution, before the transfer results

[`v0.1/`](v0.1/) holds the docs as they stood after the learning-rate confound was
ruled out (the matched-LR control, ~7×) but **before the probe campaign that
followed**: the un-censored speed measurement (11.8× median, resolved — retiring
v0.1's "≥13–14×, censored" floor language), the difficulty-controlled overlap sweep
(the head start is flat across teacher/student data overlap — content-independent),
the cross-domain test (a Shakespeare teacher accelerates learning *of Sherlock
Holmes*, scored on Sherlock — domain-independent), and the teacher-strength sweep
(the advantage grows monotonically with teacher training and is unsaturated; a weak
teacher actively hurts). v0.1's "what's next" list — the cross-data test above
all — is what v0.2 ran. See the current top-level [README.md](../README.md).

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
