# Prism: Transfer Learning at the Spectral Level

> **Same-data result committed; cross-data test still open (2026-07-18).**
>
> §6 now reports a real, three-seed result with a committed artifact
> ([results/recipe_20260718T002717Z.json](results/recipe_20260718T002717Z.json)):
> Prism reaches ~7% lower best validation loss than the baseline and does not
> overfit where the baseline does, on all three seeds. Read it with two caveats
> that the paper's older prose did not respect:
>
> - **It is same-data transfer.** Teacher and student share the 80% split, so it
>   does not by itself separate structural transfer from content leakage. The
>   cross-data test in §5.4 is the decisive experiment and **has not been run** —
>   its earlier "71% retained" figure was fabricated and is withdrawn.
> - **The speedup is left-censored.** Prism crosses the baseline's best loss by
>   the first eval (step 100) on every seed, so the "13–14×" is a lower bound, not
>   a measurement — the same resolution-floor caveat that made the old "13x"
>   meaningless, now stated plainly.
>
> Other sections still describe experiments that were never committed. §5.2's
> ablation numbers, §5.5's teacher sweep, and the "80+ runs" have **no artifact**
> and are flagged inline as not-yet-run. Seeds are now configurable (the old code
> hardcoded `manual_seed(1337)`, so the historical "seed 42 / 3.8–4.8× across
> seeds" claims were mechanically impossible). The method (§4) stands on its own.

## 1. Abstract

Every trained neural network produces two outputs: the weights (what it learned) and a spectral blueprint (how it organized itself to learn). Current practice discards the blueprint. Prism extracts it and uses it to initialize fresh models. The hypothesis: such models converge faster and continue improving past the point where a baseline overfits — not because they train less, but because overfitting no longer forces them to stop.

**Status: first same-data result committed; cross-data pending.** A three-seed run on nanoGPT Shakespeare (§6) shows Prism reaching ~7% lower best validation loss than the baseline with no overfitting through 5,000 steps, reproducible from a committed artifact. The paper's older headline ("Validated... 13x Prism Score, 7% better final loss") had no artifact and is withdrawn; the numbers below are the measured replacements.

The eval trains teacher and student on the same split, so it measures same-data transfer and does not by itself exclude content leakage. The cross-data test that would (§5.4) has not been run. The method is untested at production scale.

## 2. Introduction

Modern neural networks spend their first thousands of gradient steps rediscovering structure that every trained model already possesses. Standard initializations (Xavier, He, orthogonal, or scaled Gaussian) start from isotropic noise. The optimizer must laboriously carve out the dominant singular directions and energy distributions that ultimately define the network's representational geometry.

This is wasteful for two reasons. First, the structure discovery phase is redundant — the final weight matrices of any converged model exhibit highly non-random singular-value spectra and aligned singular vectors. Second, and more importantly, the overfitting that typically ends training is itself a *structural* failure: the model's weight geometry drifts away from the task-aligned subspace it needs to occupy.

Prism addresses both problems by transferring the *spectral organization* of a trained model — both the directional axes (how parameters align) and the energy envelope (how much variance lives in each mode) — while leaving specific learned content untouched. The transferred spectral prior serves as both initialization and continuous regularizer. The consequence is not merely faster convergence but the removal of the overfitting ceiling that normally limits training depth, model capacity, and final quality.

## 3. Background

Singular Value Decomposition (SVD) has long revealed that trained neural-network weights are far from random. Seminal work by Martin & Mahoney and subsequent spectral analyses show that weight matrices develop heavy-tailed singular-value distributions and highly structured singular vectors after training. Recent parameter-efficient methods have begun to exploit this structure:

- **PiSSA** and **DoRA** decompose weights into magnitude and direction components for low-rank adaptation.
- **Mimetic initialization** approaches attempt to copy directional statistics from a teacher.

All prior art either operates at the *parameter* level (copying or adapting weights) or requires the student to remain close to the teacher throughout training.

Prism occupies a new position in the transfer learning taxonomy:

```
Random init → Spectral prior (Prism) → LoRA/adapters → Fine-tuning → Distillation
```

It closes the gap between random initialization and parameter-level transfer: **from-scratch spectral transfer** — a one-time extraction of a compact spectral prior that can initialize and regularize any new model without copying content.

## 4. Method

### 4.1 Spectral Imprint (DCT compression of SV distributions)

For each weight matrix **W** ∈ ℝ^(m × n) in the teacher, compute its SVD:

**W** = **U** **Σ** **V**^T

The vector of singular values **σ** is transformed via discrete cosine transform (DCT) and truncated to the first **8 coefficients**. These 8 floats per weight group compress the entire energy distribution into ≈128 bytes total for the Shakespeare nanoGPT. At initialization the student's singular values are reshaped to match this compressed spectrum.

### 4.2 EigenTransfer (partial singular vector alignment)

The teacher's left and right singular vectors **U_t**, **V_t** are extracted once. At student initialization each weight matrix is rotated so that its singular vectors **U_s**, **V_s** are blended toward the teacher's:

**U_s** ← (1 − α) **U_s** + α **U_t**,   α = 0.75

(with orthogonalization after blending). This gives the student the correct *directional scaffolding* from step zero — it starts knowing which subspaces in weight space are task-relevant.

### 4.3 The Mod Wheel (continuous spectral regularization)

After every optimizer step a lightweight corrective term pulls the student's weights back toward the spectral target:

```
W.data = (1 - strength) * W.data + strength * W_target
strength *= decay  # 0.9999 per step, halves every ~7000 steps
```

This is the component that eliminates overfitting. The modulation strength starts at 0.01 and decays exponentially, maintaining structural coherence throughout training while allowing the model to learn freely within the spectral subspace. The consequence: the model can train indefinitely without the geometric drift that causes overfitting. Training depth becomes a choice, not a constraint.

### 4.4 The Prism Recipe (the combined config)

All three components are enabled together with the exact hyper-parameters below (taken verbatim from `config/prism_recipe.py`):

```python
prism_init = True
prism_align = 0.75          # EigenTransfer strength
prism_spectra = '.prism_cache/teacher/spectra.json'
prism_directions = '.prism_cache/teacher/directions.pt'
learning_rate = 5e-4        # half the default Shakespeare LR
warmup_iters = 50
prism_mod = 0.01            # Mod-wheel strength
prism_mod_decay = 0.9999
```

## 5. Experiments

### 5.1 Test rig

nanoGPT Shakespeare (6 layers, 384 hidden size, 10.65 M parameters). Data is strictly partitioned: 80% train split for both teacher and student, 20% held-out teacher-validation, and the original Shakespeare validation set used *only* for final evaluation. Teacher trained 2000 steps; student runs are 5000 steps with evaluation every 100 steps. The committed result (§6) ran seeds 1337, 1338, 1339 on an NVIDIA L4; seed is a config key, so the run is reproducible on any device.

### 5.2 Ablations (not yet committed)

The design hypothesis is that **speed and anti-overfitting come from different
components** — EigenTransfer for speed, the Mod Wheel for anti-overfitting — and
that only the two together reach quality the baseline can't, because it overfits
first. The eval exposes the ablation arms to test this directly:
`--method=spectral_only` (spectrum, no directions) and `--method=dirs_only`
(directions, no mod wheel). **Neither has a committed multi-seed artifact yet**, so
the per-component attribution above is a hypothesis, not a result. A prior "80+
runs" sweep left no committed trace and is not cited here.

### 5.3 The Prism Score

A standardized metric returned by `prism_eval.py`: the ratio of steps the baseline needs to reach its best loss vs. steps Prism needs. This measures convergence speed. But the more consequential metric is final quality at extended training — where Prism's anti-overfitting property allows it to keep improving long after baseline collapses.

### 5.4 Cross-data skeptic test (the decisive experiment — not yet run)

Everything in §6 is same-data: teacher and student share the 80% split, so a win
there cannot distinguish structural transfer from the teacher leaking content. The
test that separates them: extract the teacher's fingerprint from one half of the
training data and train the student on the disjoint other half, where no shared
content remains. If the advantage survives, the transfer is structural — and every
pretrained checkpoint holds a reusable prior currently discarded. If it collapses,
Prism is distillation with extra steps.

This has **not** been run. An earlier draft reported "71% of the advantage
retained"; that figure had no artifact and is withdrawn. (Its stated setup — two
disjoint "80% subsets" of one dataset — is also impossible; the real design splits
the data in half.)

### 5.5 Teacher investment sweep (not yet run)

How small a teacher still transfers is an open question — the `--teacher_steps`
flag sweeps it — but no teacher-size sweep has a committed artifact. The earlier
">10× from a 1000-step teacher" claim is withdrawn.

## 6. Results

Three seeds (1337, 1338, 1339), one committed artifact
([results/recipe_20260718T002717Z.json](results/recipe_20260718T002717Z.json)):

| Metric                    | Baseline (median) | Prism Recipe (median) |
|---------------------------|-------------------|-----------------------|
| Best validation loss      | 1.782 (1.778–1.785) | **1.656** (1.655–1.658) |
| Loss at step 5000         | ~2.31 (overfit)   | **~1.66** (stable)    |
| Overfits within 5000 steps| Yes — all 3 seeds | **No — 0 of 3**       |
| Steps to baseline's best  | —                 | **≤100 (≥13–14×, lower bound)** |

The two robust, directly-measured findings are the lower floor and the absence of
overfitting:

- The baseline peaks near step 1350 at ~1.78, then degrades — by step 5000 it has
  risen to ~2.31 on every seed. That peak is the best it ever gets.
- Prism reaches ~1.656 and holds; it overfits on none of the three seeds. That is
  ~7% lower best loss than the baseline ever achieves.

The convergence *speed* is real but only bounded: Prism is already below the
baseline's best by the first eval at step 100, so the Prism Score (13–14×) is
left-censored — the true crossing is below step 100 and this eval does not resolve
it. It is a floor, not a point estimate; dense early evaluation would resolve it.

All of this is same-data (§5.4): it establishes that Prism transfers *something*
useful on shared data, not yet that what transfers is structure rather than content.

## 7. Discussion

**What transfers is the *how*, not the *what*.** The student learns Shakespearean text from scratch; only the *organizational grammar* of its weight matrices is pre-loaded. This is transfer learning at a different level of abstraction than fine-tuning or distillation: it transfers the geometry of the solution space, not a point within it.

**Training becomes cumulative.** A spectral prior extracted from model A accelerates model B. Model B's (potentially improved) prior can accelerate model C. Whether spectral priors compound across generations is untested but architecturally plausible — the prior should improve as teachers improve. If so, the cost of training decreases monotonically across a research program.

**Zero overfitting changes the scaling calculus.** Overfitting is the mechanism that punishes you for having a model too large for your dataset, training too long, or not tuning regularization carefully enough. Remove it, and several constraints relax simultaneously: bigger models become viable on smaller datasets, training length becomes a choice rather than a constraint, and the dropout/weight-decay/early-stopping search space collapses.

**Compression tiers.** 128 bytes (spectral shape via 8 DCT coefficients) → full directional matrices (≈500 MB). The design intent is that the spectral shape alone is insufficient and the directional alignment carries most of the benefit — but the per-tier ablation (§5.2) has not been committed, so the split of credit between them is unverified. Directional compression remains an open research question. The primitive gets more practical as compression improves.

**The synthesizer metaphor.** Think of the network as a synthesizer. The singular vectors are the oscillator waveforms, the singular-value spectrum is the filter envelope, and the Mod Wheel is the real-time modulator. Prism supplies the preset patch; training only needs to dial the knobs.

## 8. Limitations

- Measured only on Shakespeare (tiny dataset, ~1M tokens), same-data.
- The decisive cross-data test (§5.4), a regularization-matched control, and the component ablations (§5.2) are all still unrun — so "no overfitting" is not yet attributable to spectral structure specifically, and the speedup is a lower bound.
- Requires a teacher checkpoint. It is transfer learning, not magic.
- Full directional matrices are large (≈500 MB uncompressed for a 10.65M param model).
- The "128 bytes" headline applies only to spectral shape, which alone is insufficient. The full method requires ~500 MB of directional data.
- Untested at scale (GPT-2 124M on OpenWebText in progress).

## 9. Future Work

- **The cross-data test (§5.4)** — the decisive experiment; structure vs. content. This comes before everything else below.
- **GPT-2 124M on OpenWebText** — the first real test of whether the primitive transfers to production scale.
- **Extended training** — Prism had not plateaued at 5000 steps; run longer to find the actual ceiling.
- **Generational compounding** — does extracting from a Prism-trained model produce a better prior than extracting from the original teacher?
- **Directional compression** — 500 MB → target <1 MB. Low-rank approximation of U/V.
- **Adaptive mod wheel** — modulation strength that responds to training dynamics rather than following a fixed decay.
- **Cross-architecture transfer** — transformer ↔ Mamba, different model sizes.

## 10. Conclusion

Prism is a new transfer learning primitive that operates at the spectral level. By extracting and re-injecting only the *structural prior* encoded in a model's SVD — the directions that matter and the energy that flows through them — it converts each training run into a reusable asset that accelerates all subsequent runs.

The first committed result on nanoGPT Shakespeare, across three seeds: Prism reaches ~7% lower best validation loss than the baseline and does not overfit through 5,000 steps where the baseline does — and it is already below the baseline's best by the first eval (a ≥13–14× lower bound on steps-to-parity). This is same-data transfer; whether what transfers is *structure* rather than *content* is exactly what the cross-data test (§5.4) will decide, and that test has not been run.

The recipe is 8 lines of config. The eval is one command and writes a committed artifact. The primitive is early — alignment strengths are fixed, the mod wheel follows a static decay, directional compression hasn't been attempted, ablations and the cross-data test are unrun. What is proven so far is modest and real; what would make it important is the next experiment, not this one.

Code and committed results: [github.com/timepointai/nanogpt-prism-shakespeare](https://github.com/timepointai/nanogpt-prism-shakespeare)

Train once. Extract the blueprint. Train again — and then run the test that tells you whether the blueprint was structure or a leak.
