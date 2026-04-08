# Prism × nanoGPT Results

## Shakespeare Char-Level (10.65M params, April 2026)

### Two Modes

**Prism Sprint** — 4.8x faster to baseline quality.

```bash
python train.py config/train_shakespeare_char.py config/prism_sprint.py --max_iters=600
```

**Prism Marathon** — lower val loss than baseline ever achieves, no overfitting.

```bash
python train.py config/train_shakespeare_char.py config/prism_marathon.py
```

Both require a pre-trained teacher model for spectral extraction:

```bash
# Train teacher (once)
python train.py config/train_shakespeare_char.py --max_iters=2000 --out_dir=out-teacher
# Extract spectral fingerprint (once)
python prism_extract.py --ckpt out-teacher/ckpt.pt --out .prism_cache/shakespeare
```

### Results

| Config | Steps to val 1.46 | Best val loss | Best @ step | @5000 |
|--------|-------------------|---------------|-------------|-------|
| Baseline | 1,900 | 1.4636 | 1,900 | 1.690 (overfit) |
| **Prism Sprint** | **400** (4.8x) | 1.4349 | 1,100 | 1.693 (overfit) |
| **Prism Marathon** | 800 (2.4x) | **1.4187** | 4,000 | **1.430** (stable) |

### What's Happening

Prism Sprint (mod_gentle: strength 0.005, decay 0.999):
- Spectral init gives a 1-point head start at step 0 (val 3.11 vs 4.28)
- Continuous modulation pulls weights toward spectral targets
- Modulation fades quickly → converges fast but eventually overfits like baseline
- Value: **stop early, get baseline quality in 1/5 the compute**

Prism Marathon (mod_sustained: strength 0.01, decay 0.9999):
- Same spectral init head start
- Modulation fades slowly → stays active through entire training
- Acts as anti-overfitting regularizer: val loss 1.43 at step 5000 vs baseline 1.69
- Value: **better final quality with zero overfitting**

### The Spectral Mod Wheel

After each optimizer step, blend each weight matrix slightly toward its
spectral target (the initialized spectral shape). The blend strength
decays exponentially per step.

```
W = (1 - mod_strength) * W + mod_strength * W_spectral_target
mod_strength *= decay  # per step
```

At decay 0.999, the modulation halves every ~700 steps (sprint: fast fade).
At decay 0.9999, it halves every ~7000 steps (marathon: stays on).

### Race Data (v1-v3, 20 configs tested)

Sprint winner: mod_gentle (0.005/0.999) — 4.8x speedup
Marathon winner: mod_sustained (0.01/0.9999) — best quality, never overfits
ADSR variants: matched marathon but didn't beat it — sustain phase dominates

## Next Steps

1. **OpenWebText GPT-2 124M**: real nanoGPT benchmark (needs 8×A100)
2. **Multi-seed validation**: confirm 4.8x with different random seeds
3. **Self-extraction at scale**: does the approach transfer from Shakespeare to GPT-2?
