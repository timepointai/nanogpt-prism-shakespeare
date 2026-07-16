# Prism
# PRE RELEASE
## CLAIMS UNDER RE-VERIFICATION — DO NOT CITE

> **Every number in this repo is suspended pending a re-run (2026-07-16).**
>
> The published figures — 13x Prism Score, val losses 1.7704 / 1.6498, 71%
> cross-data retention — have **no committed artifact backing them**. No
> notebook here was saved with outputs; every run wrote to ephemeral Colab
> paths and died with the VM. `RESULTS.md` and `WHITEPAPER.md` report
> *different* numbers for the same experiment: RESULTS.md has the baseline at
> **1.4636**, which is better than the whitepaper's headline *Prism* result of
> **1.6498**. The 71% figure describes a test RESULTS.md itself still lists as
> "(running)".
>
> Known measurement issues, independent of the missing artifacts:
> - **13x is a resolution artifact.** With eval every 100 steps, "reached
>   baseline quality at step 100" means "at or before the first eval." 1300/100
>   is a lower bound, not a measurement.
> - **The score is a ratio.** The 13x run's baseline (1.7704) is far weaker than
>   RESULTS.md's baseline (1.4636). A weaker baseline inflates the ratio without
>   the method improving.
> - **"Never overfits" is false for Sprint.** RESULTS.md's own table shows Prism
>   Sprint at 1.690 @5000 — overfit. Only Marathon held.
> - **Seeds were never varied.** `train.py` hardcoded `manual_seed(1337)` and did
>   not expose a seed flag; `--seed=42` would have raised `Unknown config key`.
>   The "seed 42" and "3.8-4.8x across seeds" claims had no mechanism. Seed is
>   now configurable.
> - **No regularization-matched control.** The mod wheel is a regularizer, but no
>   ablation compares it against tuned dropout / weight decay / early stopping.
>
> `src/prism_eval.py` has been rebuilt to run multiple seeds and emit auditable
> artifacts to `results/`. **Cite nothing here until a matching `results/*.json`
> exists.**

**A transfer learning primitive that aims to suppress overfitting and make training cumulative.**

Every trained neural network contains a spectral structure — a compact description of *how* it organizes its parameters — that is normally discarded. Prism extracts that structure and injects it into fresh models. The hypothesis under test: models that converge faster and keep improving where a baseline would overfit.

*Method described below. No validated results at this time — see the banner above.*

**[Run the eval in Colab →](https://colab.research.google.com/github/timepointai/nanogpt-prism-shakespeare/blob/master/nanogpt_prism_eval.ipynb)** Multi-seed. Writes a committable artifact.

## Why It Would Matter

*These are the motivations for the method, not findings. Nothing below is
established — that is what the re-run is for.*

**If training became cumulative rather than disposable.** Today a finished
training run yields one product: the weights. Prism aims to extract a second —
the spectral geometry — that would accelerate future runs on the same
architecture, so each run makes the next cheaper.

**If overfitting stopped being the ceiling.** Overfitting is what usually ends
training. A method that suppressed it would let you train longer, use bigger
models on smaller datasets, and shrink the regularization search. The open
question is whether the mod wheel does this any better than tuned dropout and
weight decay — an ablation nobody has run. Until it exists, the mod wheel is
simply an unusual regularizer with no demonstrated edge over standard ones.

**If late features had time to emerge.** Complex representations tend to appear
late in training, after simple ones saturate. A model that overfits first never
gets them. This is the payoff that would justify the method — and the reason the
result is worth verifying properly rather than asserting.

## The Result — UNVERIFIED, NO ARTIFACT

**The table below has no committed evidence and is contradicted by
[RESULTS.md](RESULTS.md). It is retained only so the re-run has something to
compare against. Do not cite it.**

Claimed on nanoGPT Shakespeare (char-level, 10.65M params):

```
┌───────────────┬──────────────┬────────────────────┐
│               │   Baseline   │   Prism Recipe     │
├───────────────┼──────────────┼────────────────────┤
│ Best val loss │     1.7704   │     1.6498         │
│ Best @ step   │       1300   │       4800         │
│ Val @ 5000    │     2.3613   │     1.6703         │
│ Overfitting   │        YES   │         no         │
├───────────────┴──────────────┴────────────────────┤
│  Prism reaches baseline quality at step 100       │
│  Baseline reaches it at step 1300                 │
│                                                   │
│  13x PRISM SCORE (steps to baseline quality)      │
│  7% better final quality (loss baseline never     │
│  reaches at any point in 5000 steps)              │
│  Zero overfitting (baseline collapses by 3000)    │
└───────────────────────────────────────────────────┘
```

*Why this is suspended: the 13x is `1300/100`, where 100 is the first eval step —
the measurement floor, not a resolved crossing. The baseline here (1.7704) is
markedly worse than the baseline RESULTS.md reports for the same rig (1.4636),
and a weaker baseline inflates a ratio without the method improving. Prism's
1.6498 is itself worse than that 1.4636 baseline.*

## What Prism Is

Prism is a new primitive in the transfer learning taxonomy:

```
Random init → Prism (spectral prior) → LoRA/adapters → Fine-tuning → Distillation
     ↑                                                                      ↑
  No knowledge                                                    Full knowledge
  transferred                                                     transferred
```

Existing methods transfer *content* — specific weights, activations, or outputs. Prism aims to transfer only *structure* — which directions in weight space matter and how energy distributes across them, leaving the student to learn its own content.

**This structure-not-content claim is currently unsupported.** It rests on the cross-data test, which was never run (see below). In the eval as written, the teacher and student train on the *same* 80% split, so content transfer is not ruled out.

Three ingredients:

1. **EigenTransfer** — Extract SVD from a trained model's weights. Blend
   the fresh model's singular vectors 75% toward the trained directions.
   This tells the optimizer *which directions in weight space matter*.

2. **Spectral Imprint** — Compress the trained model's singular value
   distribution to 8 DCT coefficients per weight group. Reshape the fresh
   model's spectrum to match. This tells the optimizer *how much energy
   goes in each direction*.

3. **Mod Wheel** — After each training step, gently pull weights back
   toward the spectral target (strength 0.01, decay 0.9999 per step).
   This is a spectral regularizer that prevents overfitting by maintaining
   structural coherence throughout training. It's the reason overfitting
   disappears.

## The Cross-Data Test — NOT RUN

**The "71% structural" claim is withdrawn.** It has no source: the string `71`
does not appear anywhere in `experiments/nanogpt_skeptic.ipynb`, that notebook
has zero executed cells, its verdict cell reads a `/content/skeptic_curves.json`
that was never committed, and [RESULTS.md](RESULTS.md) lists the skeptic test as
"(running)" under both Experiments and Next Steps.

The test itself is still the right one, and it is the load-bearing experiment for
this entire method: extract a fingerprint from one partition of Shakespeare,
train a student on a disjoint partition, and see how much of the advantage
survives. If most of it survives, the transfer is structural. If it collapses,
Prism is leaking content and the teacher is doing the work.

Until that runs, nothing here distinguishes spectral transfer from content
transfer.

## Unexplored Headroom

The current recipe (0.75 alignment, 0.01 mod, 0.9999 decay) is the first configuration that worked well. Nobody has yet:

- Made alignment strength per-layer or learned
- Made the mod wheel adaptive (stronger when drifting, weaker when on track)
- Stacked spectral priors from multiple teachers
- Tested generational compounding (model A → B → C, each extracting and improving the prior)
- Compressed the directional matrices (500MB → target <1MB)
- Pushed training beyond 5000 steps to find where Prism eventually plateaus

None of this headroom is worth exploring until the base result is verified.

## Test Rig

All results use the same rigorous eval setup:

- **Model**: nanoGPT Shakespeare char-level (6 layers, 384 hidden, 10.65M params)
- **Data partition**: Contiguous split — Train (80% of original train),
  Teacher-Val (20% of original train), Test (original val.bin). Teacher and
  student both train on Train. All reported numbers are on the held-out Test
  set, which is never seen during training.
- **Teacher**: Trained for 2000 steps on Train partition, checkpoint extracted.
  Spectral fingerprint = 8 DCT coefficients per weight group + full U/V
  directional matrices.
- **Student configs**: Prism Recipe (align 0.75, LR 5e-4, warmup 50,
  mod_strength 0.01, mod_decay 0.9999) vs standard Normal(0, 0.02) baseline.
- **Steps**: 5000 per run, eval every 100 steps.
- **Hardware**: NVIDIA A100 (Google Colab). ~120-200s per 5000-step run.
- **Seeds**: Historically **1337 for every run** — `train.py` hardcoded
  `torch.manual_seed(1337)` and exposed no seed flag, so `--seed=42` would have
  raised `Unknown config key`. Earlier claims of "seed 42" and "3.8-4.8x across
  seeds" had no mechanism to produce them. Seed is now a config key; the eval
  runs 1337/1338/1339 by default.
- **Reproducibility**: `src/prism_eval.py` writes a full artifact (loss curves,
  git commit, GPU, seeds) to `results/`. The notebooks in `experiments/` were
  saved without outputs and are not evidence of anything.

## Reproduce It

**Colab (easiest):** [Run the eval →](https://colab.research.google.com/github/timepointai/nanogpt-prism-shakespeare/blob/master/nanogpt_prism_eval.ipynb)

**Local:**
```bash
git clone https://github.com/timepointai/nanogpt-prism-shakespeare.git
cd nanogpt-prism-shakespeare/src
pip install transformers tiktoken datasets
python prism_eval.py
```

For each seed this trains a teacher, extracts the spectral fingerprint, runs
baseline and Prism, and scores them. It writes a full artifact — loss curves,
git commit, GPU, seeds — to `results/`, and prints a median and range across
seeds. Defaults to seeds 1337/1338/1339: roughly 45-60 min on an A100.

```bash
python prism_eval.py --seeds=1337              # single seed, fast, NOT publishable
python prism_eval.py --method=spectral_only    # ablation: 128-byte shape, no directions
python prism_eval.py --report                  # reprint the last artifact
```

**Commit the artifact.** A number in a doc without a matching `results/*.json`
is not a result — that is exactly how this repo ended up with a headline nobody
can reproduce.

## The Recipe

```python
# config/prism_recipe.py
prism_init = True
prism_align = 0.75     # EigenTransfer: 75% toward teacher directions
prism_spectra = '.prism_cache/teacher/spectra.json'
prism_directions = '.prism_cache/teacher/directions.pt'
learning_rate = 5e-4   # half the Shakespeare default
warmup_iters = 50
prism_mod = 0.01       # mod wheel strength
prism_mod_decay = 0.9999  # halves every ~7000 steps
```

## Repo Structure

```
README.md                    ← You are here
WHITEPAPER.md                ← Method description (claims suspended)
RESULTS.md                   ← Earlier findings (unreproduced, conflicts with README)
nanogpt_prism_eval.ipynb     ← Multi-seed Colab eval; writes a committable artifact
results/                     ← Eval artifacts. THE evidence. Empty = nothing proven.
config/prism_recipe.py       ← The 8-line config
src/
  prism_eval.py              ← Standardized benchmark (produces Prism Score)
  prism_init.py              ← Spectral Imprint + EigenTransfer + Mod Wheel
  prism_extract.py           ← Extract fingerprint from any checkpoint
  train.py                   ← nanoGPT + Prism hooks + configurable seed
  model.py                   ← nanoGPT model (unmodified)
experiments/                 ← Exploratory notebooks, saved WITHOUT outputs (not evidence)
```

## Limitations

- **No validated results.** Every published number lacks a committed artifact. This is the top limitation and it supersedes the rest.
- **Nothing was reproduced.** All 15 Prism notebooks have zero executed outputs and wrote to ephemeral Colab paths. The "80+ runs" left no trace in this repo.
- **Seeds never varied.** See Test Rig. Any claim of seed variance predates the code being able to vary a seed.
- **Cross-data untested.** The one experiment that would show the transfer is structural rather than content was never run.
- **No regularization-matched baseline.** The mod wheel is a regularizer. No ablation compares it against tuned dropout / weight decay / early stopping, so there is no evidence it beats standard regularization at its own job.
- **Shakespeare only.** Char-level, ~1M tokens, 10.65M params. Not validated at production scale.
- **Teacher required.** Prism is transfer learning. No teacher = no benefit.
- **The 128-byte headline is not the method.** The GitHub description says "extract 128 bytes, train again 13x faster." Those are different tiers: 128 bytes (spectral shape) is claimed at ~1.4x; the large speedups need ~500MB of directional matrices. The description conflates them and needs fixing.

## License

MIT — see [LICENSE](LICENSE).

This is a standalone clone of [nanoGPT](https://github.com/karpathy/nanoGPT) by
Andrej Karpathy (MIT, © 2022), not a fork. `model.py`, `train.py`,
`configurator.py`, `bench.py`, `sample.py`, and the `data/` preparers are his
work; `train.py` carries Prism modifications. The Prism additions
(`prism_init.py`, `prism_extract.py`, `prism_eval.py`, `config/prism_*.py`) are
Timepoint Labs' and are released under the same MIT terms.

---

*Created by [Sean McDonald](https://x.com/seanmcdonaldxyz) · A [Timepoint Labs](https://timepointai.com) project · April 2026.*
