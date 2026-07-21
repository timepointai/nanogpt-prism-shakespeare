# Prism: Transfer Learning at the Spectral Level

**v0.1.** This version reports a result that is *attributed*: on nanoGPT
Shakespeare, a fresh model given only the spectral geometry of a trained one
reaches a lower validation loss, does not overfit where the baseline does, and
gets to the baseline's best quality several times faster — and this holds when the
learning rate is matched to the baseline, so only the spectral machinery differs.
The earlier writeup, before that control was run, is archived at
[`archive/v0.0/WHITEPAPER.md`](archive/v0.0/WHITEPAPER.md).

## 1. Abstract

Every trained neural network produces two outputs: the weights (what it learned)
and a spectral blueprint (how it organized itself to learn — the singular-value
spectrum and singular vectors of its weight matrices). Standard practice discards
the blueprint. Prism extracts it and uses it to initialize and regularize a fresh
model.

On nanoGPT Shakespeare (10.65M params), across three seeds, the Prism recipe
reaches a best validation loss of ~1.66–1.67 versus the baseline's ~1.78, and does
not overfit through 5,000 steps where the baseline decays to ~2.30. Critically, a
**schedule-matched control** — the recipe run at the baseline's own learning rate,
so the only difference is the spectral flags — reproduces the effect (best ~1.67,
no overfitting, ~7× faster to the baseline's best). The improvement is therefore
attributable to the spectral method, not to the learning rate the recipe happened
to use.

Two questions remain open and bound the claim. First, all runs are *same-data*:
teacher and student share the training split, so this does not yet distinguish
structural transfer from content leakage — the cross-data test (§7) decides that.
Second, the anti-overfitting has not been separated from generic regularization.
The method is untested at production scale.

## 2. Introduction

Modern networks spend their first thousands of gradient steps rediscovering
structure that every trained model already possesses. Standard initializations
start from isotropic noise; the optimizer must carve out the dominant singular
directions and energy distribution that define the network's representational
geometry. This is wasteful in two ways. The structure-discovery phase is redundant —
converged weight matrices have highly non-random spectra and aligned singular
vectors. And the overfitting that ends most training runs is itself *structural*:
the weight geometry drifts away from the task-aligned subspace it should occupy.

Prism transfers the *spectral organization* of a trained model — the directional
axes and the energy envelope — while leaving learned content untouched. The
transferred prior serves as both initialization and continuous regularizer. The
hoped-for consequence is not merely faster convergence but the removal of the
overfitting ceiling.

## 3. Background

Singular Value Decomposition has long shown that trained weights are far from
random: they develop heavy-tailed singular-value distributions and structured
singular vectors (Martin & Mahoney and subsequent spectral analyses). Parameter-
efficient methods exploit this — PiSSA and DoRA decompose weights into magnitude
and direction; mimetic initializations copy directional statistics. All prior art
operates at the *parameter* level or requires the student to stay close to the
teacher. Prism occupies a different slot: **from-scratch spectral transfer** — a
one-time extraction of a compact spectral prior that initializes and regularizes a
new model without copying content.

```
Random init → Spectral prior (Prism) → LoRA/adapters → Fine-tuning → Distillation
```

## 4. Method

### 4.1 Spectral Imprint (DCT compression of the SV distribution)

For each weight matrix **W** = **U Σ Vᵀ**, take the singular values **σ**,
normalize by the max, and average within each weight group (attention, FFN up, FFN
down, embedding). Each group's averaged spectrum is fit — in an inverse-softplus
space that keeps the reconstruction positive — to the first **8 cosine (DCT)
coefficients** by least squares. That is ~128 bytes total for the Shakespeare
nanoGPT. Reconstruction from those 8 coefficients tracks a real group-averaged
spectrum to a mean absolute error of ~0.03. At init the student's singular values
are reshaped to match.

### 4.2 EigenTransfer (partial singular-vector alignment)

At student init each weight matrix is rotated so its singular vectors blend toward
the teacher's, **U_s ← (1−α)U_s + αU_t**, α = 0.75, with re-orthogonalization. The
student starts with the teacher's directional scaffolding.

### 4.3 The Mod Wheel (continuous spectral regularization)

After every optimizer step a corrective term pulls weights back toward the spectral
target:

```
W.data = (1 - s) * W.data + s * W_target
s *= 0.9999   # per step
```

Strength starts at 0.01 and decays. This is the component intended to prevent
overfitting: the model learns freely within the spectral subspace but does not
drift out of it. It adds no storage.

### 4.4 The Prism Recipe

All three together (from `config/prism_recipe.py`): `prism_align = 0.75`,
`prism_mod = 0.01`, `prism_mod_decay = 0.9999`, plus the imprint and directions.
The recipe as originally tuned also set `learning_rate = 5e-4` (half the config
default) and `warmup_iters = 50` — and §6 shows why that detail had to be
controlled for.

## 5. Experimental protocol

nanoGPT Shakespeare (6 layers, 384 hidden, 10.65M params). Data is partitioned:
80% train (for both teacher and student), 20% held-out teacher-validation, and the
original Shakespeare validation set used only for final scoring. Teacher 2,000
steps; students 5,000 steps, eval every 100. Three seeds (1337, 1338, 1339) on an
NVIDIA L4. Seed and the student learning rate / warmup are configuration keys, so
every run is reproducible and the schedule is auditable. Each run writes a full
artifact — loss curves, git commit, GPU, argv, and whether the schedule was
matched — to `results/`.

## 6. Results

### 6.1 The recipe vs. the baseline (two learning rates)

| | Baseline (LR 1e-3) | Recipe (LR 5e-4) | Recipe (LR 1e-3, matched) |
|---|---|---|---|
| Best validation loss | 1.782 | 1.656 | **1.671** |
| Loss at step 5,000 | ~2.31 (overfit) | ~1.66 (stable) | ~1.67 (stable) |
| Overfits within 5,000 steps | yes — 3/3 | no — 0/3 | no — 0/3 |
| Steps to baseline's best | — | ≤100 (≥13×, censored) | 200 (7×, measured) |

The baseline peaks near step 1,350 at ~1.78 and then overfits, reaching ~2.31 by
step 5,000 on every seed. The recipe reaches ~1.66–1.67 and holds, overfitting on
none.

### 6.2 The attribution control (the point of v0.1)

The recipe's left column differs from the baseline in two ways at once: the
spectral machinery *and* a halved learning rate. Since a too-high learning rate is
itself a classic cause of overfitting, the confound is fatal to a naive reading:
the whole picture might be the learning rate.

So the recipe was re-run at the baseline's own learning rate (1e-3) and warmup
(100), with only the spectral flags changed (`schedule_matched: true` in the
artifact, run
[`recipe_20260720T230405Z.json`](../results/recipe_20260720T230405Z.json)). The
effect survives:

- Best validation loss **1.671** (range 1.671–1.674), vs. the baseline's 1.782.
- **No overfitting** on any of the three seeds; the baseline overfits on all three.
- Crosses the baseline's best quality at step **200 — a 7× speedup that is now
  resolved**, not the censored lower bound the 5e-4 run produced (it crossed before
  the first eval).

At the same learning rate the baseline fails at, toggling only the spectral flags
recovers the effect. The improvement is attributable to the spectral method.

### 6.3 What is not yet shown

- **Structure vs. content.** All runs are same-data (§7). This shows Prism transfers
  something real; not that it is structure rather than leakage.
- **Spectral vs. generic regularization.** The mod wheel is a regularizer; it beats
  a schedule-matched baseline, but has not been compared against tuned dropout /
  weight decay. The `spectral_only` / `dirs_only` arms (exposed by the eval) and a
  reg-matched baseline are unrun.
- **Component attribution.** The design hypothesis — EigenTransfer for speed, the
  Mod Wheel for anti-overfitting — has no committed ablation yet.

## 7. The decisive experiment (not yet run)

Everything in §6 is same-data. To separate structure from content: extract the
teacher's fingerprint from one half of the training data and train the student on
the disjoint other half, where no shared content remains. If the advantage
survives, the transfer is structural — implying every checkpoint ever trained holds
a reusable prior nobody extracts. If it collapses, Prism is distillation with extra
steps. This has not been run, and it is what would elevate the result from "a real
regularization/initialization effect" to "structural transfer."

## 8. Discussion

**What appears to transfer is the *how*, not the *what*.** The student learns its
content from scratch; only the organizational grammar of its weight matrices is
pre-loaded — and, per §6.2, that grammar alone (not the learning rate) drives the
effect. Whether it is genuinely content-independent is exactly what §7 tests.

**If overfitting is removed, the scaling calculus changes.** Overfitting is what
punishes a model too large for its data or trained too long. The recipe does not
overfit through 5,000 steps at either learning rate; whether it *never* does is the
endurance question. If it holds, training depth becomes a choice rather than a
constraint.

**Compression tiers.** 128 bytes (spectral shape) → ~500 MB directional matrices.
The spectral shape alone is the compact tier; the directional alignment is large.
Per-tier attribution is unverified (the ablation is unrun), so the split of credit
between them is a hypothesis. Directional compression is an open problem.

## 9. Limitations

- Same-data only; the cross-data test (§7) is unrun, so content leakage is not
  excluded.
- No regularization-matched control, so "the spectral structure prevents
  overfitting" (vs. generic regularization) is not isolated.
- Two shared learning rates, not a full per-arm sweep.
- Shakespeare only (~1M tokens, 10.65M params); untested at scale.
- Requires a teacher checkpoint. Transfer learning, not magic.
- The "128 bytes" headline is the spectrum; the directions are ~500 MB uncompressed.

## 10. Future work

- **Cross-data test (§7)** — structure vs. content. Comes first.
- **Endurance run** — recipe to 20k–50k steps; find where, if ever, it overfits.
- **Ablations** — `spectral_only` / `dirs_only` and a reg-matched baseline.
- **Scale** — GPT-2 124M on OpenWebText.

## 11. Conclusion

Prism transfers the spectral organization of a trained model — the directions that
matter and the energy that flows through them — into a fresh one. On nanoGPT
Shakespeare it reaches a lower loss than the baseline and does not overfit where the
baseline collapses, and this survives a schedule-matched control, so the effect is
the spectral method rather than the learning rate. That is a modest, real, and now
*attributed* result. Whether what transfers is structure rather than content — the
claim that would make it matter — is the next experiment, not this one.

Code and committed results:
[github.com/timepointai/nanogpt-prism-shakespeare](https://github.com/timepointai/nanogpt-prism-shakespeare)
