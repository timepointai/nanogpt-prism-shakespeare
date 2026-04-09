# Prism

**13x faster convergence. Better final quality. Zero overfitting.**

Prism accelerates neural network training from scratch by transferring the
spectral fingerprint of a trained model's weight structure to a fresh
initialization.

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
│  >>> 13x FASTER <<<                               │
└───────────────────────────────────────────────────┘
```

Prism wins every single checkpoint from step 0 to step 5000.

## What It Does

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
   This prevents overfitting by maintaining spectral structure throughout
   training.

## Why It's Not Cheating

**"You need a trained model — why not just use it?"**

- The teacher is cheap. A 2000-step teacher + 100-step Prism student =
  2100 total steps vs 1300 for baseline to reach the same quality. But
  Prism keeps improving to 1.6498, which baseline never reaches. Net:
  better quality in fewer total steps.
- One teacher, many students. Extract once, initialize every experiment.
  The teacher cost amortizes across all subsequent runs.
- Cross-data transfer works. Spectra from non-overlapping data retain
  71% of the advantage (skeptic test, validated).
- Pre-trained models exist. For GPT-2, extract from HuggingFace in
  seconds, use forever.

Prism is transfer learning at the spectral level. The criticism that
"you need a teacher" applies equally to knowledge distillation, LoRA,
and every transfer method. The question is whether the transfer is
worth the cost. At 13x, it is.

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

## Quick Start

```bash
git clone https://github.com/realityinspector/nanogpt-prism.git
cd nanogpt-prism
pip install transformers tiktoken datasets

# Prepare data
python data/shakespeare_char/prepare.py

# Train teacher (once)
python train.py config/train_shakespeare_char.py --max_iters=2000 \
    --out_dir=out-teacher --always_save_checkpoint=True

# Extract spectral fingerprint (once)
python prism_extract.py --ckpt out-teacher/ckpt.pt --out .prism_cache/teacher

# Train with Prism Recipe
python train.py config/train_shakespeare_char.py config/prism_recipe.py
```

Or run the Colab notebook:
**[The Prism Recipe](https://colab.research.google.com/github/realityinspector/nanogpt-prism/blob/master/nanogpt_prism_recipe.ipynb)**

## How It Was Developed

80+ experimental runs over 10 days:

1. **MPS stairclimb** (27 configs): Discovered spectral shape + directional
   alignment + LR tolerance as three independent axes of improvement.
2. **CUDA validation** (20 configs, A100): Confirmed 2.8x at real scale,
   identified overfitting at high LR.
3. **nanoGPT integration**: Self-extraction from same-architecture teacher.
4. **Race v1-v3** (15 configs): Discovered the mod wheel — continuous spectral
   modulation as anti-overfitting regularizer.
5. **Skeptic test** (6 configs): Cross-data retains 71%. Not data leakage.
6. **Teacher × DCT sweep** (26 configs): Spectral shape alone doesn't
   beat baseline. Directions are the payload.
7. **Unified sweep** (9 configs): Directions + Marathon mod = 13x.

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

## Limitations

- **Shakespeare only.** Not yet validated on OpenWebText/GPT-2 124M at
  full scale (600K steps, 8×A100). The 13x is on a tiny dataset.
- **Single seed.** Reported numbers are seed 42. Earlier experiments showed
  3.8-4.8x range for Sprint mode across seeds.
- **Teacher required.** Prism is transfer learning. No teacher = no benefit.
  The spectral-shape-only path (no directions) doesn't beat baseline.
- **Directions are large.** The directional alignment that provides the
  real benefit requires ~500MB of U/V matrices. Compressing this is an
  open research question.

## Next Steps

- OpenWebText GPT-2 124M benchmark (the real test)
- Multi-seed validation
- Directional compression (500MB → target: <1MB)
- Cross-architecture transfer

## License

Apache 2.0.

---

*Created by [Sean McDonald](https://x.com/seanmcdonaldxyz) — unfunded indie
research, April 2026.*
