# Hypothesis: Prism Spectral Initialization Accelerates GPT-2 Training

## What We're Testing

Whether injecting the spectral fingerprint of a pretrained GPT-2's weight
matrices into a randomly-initialized GPT-2 produces faster convergence
when training from scratch on the nanoGPT benchmark.

## What We Know (Empirically, April 2026)

### MPS experiments (toy scale: batch 8, seq 256, WikiText-2)

27 configs tested. Best result: **3.33x faster convergence** at step 750
using Spectral Imprint + EigenTransfer + 2x LR. This was measured as
val PPL 572 (Prism) vs 1,904 (orthogonal baseline).

The 3.33x number was real but misleading — it was partly inflated by
overfitting at the toy-scale 2x LR. At step 1000 the advantage held
(543 vs 1,896 = 3.49x) but we had not tested longer.

### CUDA experiments (real scale: batch 64, seq 1024, WikiText-2, A100)

20 configs swept. Key findings:

**The spectral advantage is real at CUDA scale but smaller and LR-sensitive.**

| Config | val_ppl@750 | vs Ortho (610) | Overfitting? |
|--------|-------------|----------------|--------------|
| Prism UV align=0.75, 1.5x LR, spike_skip | **222** | **2.74x** | No |
| Prism UV align=0.65, 1.75x LR, clip=0.5 | 243 | 2.51x | No |
| Prism UV align=0.5, 2x LR | 301 | 2.02x | Slight |
| Prism UV align=0.5, 1.5x LR | 307 | 1.99x | No |
| Spectral only (no alignment), 2x LR | 380 | 1.60x | Yes |
| Spectral only, 1x LR | 448 | 1.36x | No |
| Orthogonal baseline | 610 | 1.00x | No |

**At 2x LR without stabilization, Prism overfits by step 750-1000.**
The earlier 2K-step Prism run at 2x LR hit val_ppl 281 at step 500
(amazing) but collapsed to 780 by step 1000 (catastrophic overfitting).

**The winning config uses three ingredients:**
1. Strong UV alignment (0.75) — pretrained singular vectors carry directional info
2. Moderate LR boost (1.5x) — spectral init tolerates higher LR than orthogonal
3. Spike-skip stabilization (threshold 50x) — prevents gradient spikes from derailing

**A longer validation (2K steps) of the winning config is running but
results are not yet available.** The critical open question is whether
2.74x holds or shrinks past 750 steps.

## Three Claims We're Testing on nanoGPT

### Claim 1: Spectral shape carries transferable task structure

Pretrained GPT-2 weight matrices have characteristic singular value
distributions that differ from random initialization. When we impose
this distribution on a fresh model, it converges faster. The signal is
in the *shape* of the spectrum (which subspaces carry more energy), not
the absolute scale. Evidence: shape-only init (imt_flat, all SVs=1)
is worse than shaped init. Scale-only init (flat SVs at pretrained norms)
causes gradient explosion.

**nanoGPT test**: Compare spectral-only init vs standard Normal(0, 0.02)
at the same LR. If spectral-only gives lower val loss at 1K-10K steps,
the spectral signal transfers to nanoGPT's training setup.

### Claim 2: Pretrained singular vector directions add signal beyond magnitudes

Blending the fresh model's right singular vectors 50-75% toward the
pretrained model's vectors provides additional convergence benefit.
On CUDA, spectral-only achieves ~1.4x while spectral + UV alignment
achieves ~2.7x — the alignment roughly doubles the advantage.

**nanoGPT test**: Compare spectral-only vs spectral + UV alignment.
If alignment helps, it confirms the directional information transfers
even when the training setup is very different (different dataset,
different LR, different weight decay).

### Claim 3: The advantage persists at nanoGPT's training scale

Our CUDA experiments used WikiText-2 (2M tokens) for 750-2000 steps.
nanoGPT trains on OpenWebText (9B tokens) for 600K steps. The advantage
could wash out because:
- The optimizer has 300x more steps to discover structure on its own
- The dataset is 4500x larger (less overfitting pressure)
- nanoGPT's LR (6e-4) is already 10x higher than our experiments

**nanoGPT test**: Run 10K steps and check if the Prism val loss curve
stays below the baseline curve. If the curves converge by 10K steps,
the advantage is front-loaded (useful for cost savings but not for
final quality). If the gap persists, the spectral prior provides
lasting value.

## Why nanoGPT Is the Right Testbed

1. **Community standard.** Everyone benchmarks GPT-2 training on nanoGPT.
   A result here is credible in a way our custom training loop isn't.

2. **Minimal code.** 330 lines of model, 340 lines of training. The
   initialization is explicit and easy to replace. No framework
   abstractions hiding behavior.

3. **Known baseline.** Val loss ~2.85 at convergence (600K steps) is
   well-established. We can compare against published loss curves.

4. **The init is trivially replaceable.** `_init_weights` uses
   `Normal(0, 0.02)` + residual scaling. Our `prism_init.py` replaces
   this in 7 lines of train.py after model construction.

## What Could Go Wrong

### The LR problem

nanoGPT uses 6e-4 peak LR — 10x what we tested. Our sweep found the
sweet spot at 1.5x *our* base LR (9.38e-5). At nanoGPT's LR, 1.5x
would be 9e-4, which might overfit. We might need to run Prism at
nanoGPT's default 6e-4 without a boost, accepting that the LR-tolerance
advantage doesn't apply here.

### The overfitting risk

Our best result used spike_skip=50. nanoGPT doesn't have spike-skip.
We added it to train.py but if it triggers frequently, it changes the
effective training dynamics in a way that makes the comparison unfair.
The cleaner test: run without spike-skip first and see if the advantage
holds. The tighter grad_clip (0.5 vs 1.0) is a less invasive alternative.

### The dataset scale difference

WikiText-2 has 2M tokens. OpenWebText has 9B tokens. On a small dataset,
spectral init's directional prior might matter more because there's less
data to learn the structure from. On a large dataset, the optimizer might
discover the same structure within the first few thousand steps regardless
of init, making the init prior redundant.

### The weight tying issue

nanoGPT ties wte and lm_head. Our Prism init shapes wte, which
automatically shapes lm_head. This is a stronger intervention than in
our experiments where they were separate. It could help (consistent
spectral structure in input and output) or hurt (constrains the output
projection too early).

### The Conv1D transpose

HuggingFace GPT-2 uses Conv1D (transposed weights). nanoGPT uses
nn.Linear. Our prism_init.py transposes during extraction to match.
But if the transposition is wrong for any specific matrix, the spectral
transfer will inject the wrong shape. This needs verification on the
first run by checking that the initialized weight spectra match the
pretrained spectra.

## Experiment Phases

1. **Smoke test** (1K steps, 1 GPU): Does it run? Does loss decrease?
2. **Signal test** (10K steps, 1 GPU): Does Prism beat baseline at any checkpoint?
3. **Stability test** (50K steps, 1 GPU): Does the advantage persist or collapse?
4. **Full benchmark** (600K steps, 8 GPU): Only if phases 1-3 show signal.

## What a Positive Result Looks Like

If at 10K steps, Prism achieves a val loss that the baseline doesn't reach
until 15K-20K steps, that's a **1.5-2x convergence speedup on the nanoGPT
benchmark**. This would be publishable and practically useful — it means
you could train GPT-2 in 2 days instead of 4 on the same hardware, or
achieve the same result on half the hardware.

If the advantage washes out by 10K steps, Prism is a "warm start" trick
that saves early compute but doesn't change the final result. Still useful
for iteration speed during research, but a weaker claim.

## What a Negative Result Looks Like

If Prism provides zero advantage at nanoGPT's LR and scale — val loss
curves are identical from step 1K onward — it means the spectral prior
is redundant when:
- The dataset is large enough (9B tokens)
- The LR is high enough (6e-4)
- Training runs long enough (600K steps)

This would still be an informative result. It would characterize the
boundary of spectral initialization's applicability: useful at small
scale / low LR / short training, but unnecessary at production scale.
