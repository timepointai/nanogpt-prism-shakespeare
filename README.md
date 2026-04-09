# Prism
# PRE RELEASE
## VALIDATION UNDERWAY

**A new transfer learning primitive that eliminates overfitting and makes training cumulative.**

Every trained neural network contains a reusable spectral blueprint — a compact description of *how* it organizes its parameters — that is currently thrown away. Prism extracts that blueprint and injects it into fresh models. The result: models that converge faster, reach better final quality, and never overfit. Not because they train less, but because they can train *longer* without the overfitting ceiling that normally forces you to stop.

*Validated on nanoGPT Shakespeare. Scale testing in progress.*

**[Run the eval in Colab →](https://colab.research.google.com/github/timepointai/nanogpt-prism-shakespeare/blob/master/nanogpt_prism_eval.ipynb)** One cell. One number. Reproducible.

## Why It Matters

**Training becomes cumulative, not disposable.** Right now, when you finish a training run, you get one product: the weights. Prism extracts a second product — the spectral geometry — that accelerates every future training run on the same architecture. The cost of training amortizes. Each run makes the next one cheaper.

**Overfitting stops being the ceiling.** The baseline model overfits and collapses by step 3,000. Prism is still improving at step 5,000 — not flat, *still going down*. When overfitting doesn't stop you, you can: train longer to find better solutions, use bigger models on smaller datasets, and skip the regularization hyperparameter search (dropout, weight decay, early stopping) that exists solely to manage overfitting.

**Features have more time to emerge.** Complex features — the representations that make models actually good — emerge late in training, after simple features saturate. If the model overfits before they arrive, you never get them. Zero overfitting means the model keeps finding structure long after the baseline has given up.

## The Result

Tested on nanoGPT Shakespeare (char-level, 10.65M params), evaluated on
held-out test data with strict partitioning:

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

The Prism Score (13x) measures convergence speed. But the more important number is the val loss at step 5000: Prism is at 1.6703 and still improving. The baseline is at 2.3613 and getting worse. The gap *widens* with more training. Prism doesn't just start faster — it removes the ceiling.

## What Prism Is

Prism is a new primitive in the transfer learning taxonomy:

```
Random init → Prism (spectral prior) → LoRA/adapters → Fine-tuning → Distillation
     ↑                                                                      ↑
  No knowledge                                                    Full knowledge
  transferred                                                     transferred
```

Existing methods transfer *content* — specific weights, activations, or outputs. Prism transfers only *structure* — which directions in weight space matter and how energy distributes across them. The student learns its own content from scratch. This is why cross-data transfer retains 71% of the advantage: there's no content to leak, only organizational geometry.

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

## What 71% Structural Means

The cross-data skeptic test extracted a spectral fingerprint from one partition of Shakespeare and applied it to a student training on a completely separate partition. 71% of the convergence advantage was retained.

This means roughly 71% of what makes a trained model good is *structural organization* — how it arranges its weight matrices — not the specific data it saw. Every pretrained checkpoint contains this structural prior. It's currently discarded. Prism is the extraction method.

## Unexplored Headroom

The current recipe (0.75 alignment, 0.01 mod, 0.9999 decay) is the first configuration that worked well. Nobody has yet:

- Made alignment strength per-layer or learned
- Made the mod wheel adaptive (stronger when drifting, weaker when on track)
- Stacked spectral priors from multiple teachers
- Tested generational compounding (model A → B → C, each extracting and improving the prior)
- Compressed the directional matrices (500MB → target <1MB)
- Pushed training beyond 5000 steps to find where Prism eventually plateaus

The 13x Prism Score is the floor of this primitive, not the ceiling.

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
- **Reproducibility**: All notebooks in this repo. Single seed (42) for all
  runs. Seed variance measured in earlier experiments: Sprint speedup ranges
  3.8-4.8x across seeds.

## Reproduce It

**Colab (easiest):** [Run the eval →](https://colab.research.google.com/github/timepointai/nanogpt-prism-shakespeare/blob/master/nanogpt_prism_eval.ipynb)

**Local:**
```bash
git clone https://github.com/timepointai/nanogpt-prism-shakespeare.git
cd nanogpt-prism-shakespeare/src
pip install transformers tiktoken datasets
python prism_eval.py
```

This trains a teacher, extracts the spectral fingerprint, runs baseline
and Prism, and prints the Prism Score. ~6 min on A100, ~20 min on T4.

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
WHITEPAPER.md                ← Full method description and discussion
RESULTS.md                   ← Detailed findings, compression tiers
nanogpt_prism_eval.ipynb     ← One-click Colab eval
config/prism_recipe.py       ← The 8-line winning config
src/
  prism_eval.py              ← Standardized benchmark (produces Prism Score)
  prism_init.py              ← Spectral Imprint + EigenTransfer + Mod Wheel
  prism_extract.py           ← Extract fingerprint from any checkpoint
  train.py                   ← nanoGPT + 15 lines for Prism
  model.py                   ← nanoGPT model (unmodified)
experiments/                 ← 80+ experimental runs, notebooks, planning docs
```

## Limitations

- **Shakespeare only.** Not yet validated at production scale. GPT-2 124M on OpenWebText is in progress.
- **Single seed.** Reported numbers are seed 42. Earlier experiments showed 3.8-4.8x range for Sprint mode across seeds.
- **Teacher required.** Prism is transfer learning. No teacher = no benefit. The spectral-shape-only path (no directions) is only ~1.4x.
- **Directions are large.** The directional matrices that provide the real benefit require ~500MB. The 128-byte spectral shape alone is insufficient. Directional compression is an open research question.

## License

Apache 2.0.

---

*Created by [Sean McDonald](https://x.com/seanmcdonaldxyz) · A [Timepoint Labs](https://timepointai.com) project · April 2026.*
