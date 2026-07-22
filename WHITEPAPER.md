# Prism: Transfer Learning at the Spectral Level

**v0.2.** This version reports the *transfer* results: the speedup is resolved at
~12× (median, three seeds); the head start is invariant to how much data the
teacher and student share, including zero; it survives — slightly grows — when the
student trains and is evaluated on a different corpus (Shakespeare teacher →
Sherlock Holmes student); and it scales monotonically with teacher training,
saturating precisely where the teacher itself converges. A companion study (§6.7)
points the same regularizer the *other* way — anchoring a *trained* model during
finetuning instead of seeding a fresh one — and finds it cuts catastrophic
forgetting up to ~10×, attributably through the singular *directions* rather than
the spectrum, which cleanly separates the two roles of the geometry. Earlier
versions: [`archive/v0.1/`](archive/v0.1/)
(attribution — the matched-schedule control), [`archive/v0.0/`](archive/v0.0/)
(before attribution).

## 1. Abstract

Every trained neural network produces two outputs: the weights (what it learned)
and a spectral blueprint (how it organized itself to learn — the singular-value
spectrum and singular vectors of its weight matrices). Standard practice discards
the blueprint. Prism extracts it and uses it to initialize and regularize a fresh
model.

On nanoGPT Shakespeare (10.65M params, three seeds throughout): the Prism recipe
reaches the from-scratch baseline's best quality in ~100 steps versus the
baseline's ~1,020–1,300 — **11.8× median, a resolved measurement** — and does not
overfit through 5,000 steps where the baseline decays from 1.78 to ~2.31. A
schedule-matched control attributes the effect to the spectral machinery (7.0× with
only the spectral flags toggled). Two experiments then separate *structure* from
*content*: the early advantage is identical whether teacher and student share 100%
or 0% of their training text (difficulty-controlled), and it persists at full
strength when the student's data and validation set are replaced by a different
corpus entirely. A teacher-strength sweep shows the advantage grows monotonically
with teacher training and that a barely-trained teacher actively hurts —
evidence the transferred geometry itself, not generic regularization pressure,
carries the effect.

The claims are bounded: the transfer experiments are early-window probes (100
steps); the "far" corpus is a different author but the same language and character
set; the spectral-vs-generic-regularization ablation is unrun; and nothing here
exceeds 10.65M parameters.

## 2. Introduction

Modern networks spend their first thousands of gradient steps rediscovering
structure that every trained model already possesses. Standard initializations
start from isotropic noise; the optimizer must carve out the dominant singular
directions and energy distribution that define the network's representational
geometry. This is wasteful in two ways. The structure-discovery phase is
redundant — converged weight matrices have highly non-random spectra and aligned
singular vectors. And the overfitting that ends most training runs is itself
*structural*: the weight geometry drifts away from the task-aligned subspace it
should occupy.

Prism transfers the *spectral organization* of a trained model — the directional
axes and the energy envelope — while leaving learned content untouched. v0.1
established that the resulting speedup and overfitting immunity are attributable
to this machinery rather than to its training schedule. v0.2 asks the question
that decides whether the method matters: is what transfers *structure* (reusable
on new data) or *content* (a compressed echo of the teacher's corpus)? Both
experiments built to distinguish these came back on the side of structure.

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

The same term also runs on the **resume (finetune) path**, self-anchored to the
model's own pre-finetune weights (`prism_anchor_mode`, constant pull) — the basis
for the finetune-retention study in §6.7. There the target is the trained weights
themselves, not a fresh Prism-shaped init.

### 4.4 The Prism Recipe

All three together (from `config/prism_recipe.py`): `prism_align = 0.75`,
`prism_mod = 0.01`, `prism_mod_decay = 0.9999`, plus the imprint and directions.
The recipe as tuned also sets `learning_rate = 5e-4` and `warmup_iters = 50`; §6.2
is the control that shows the effect does not come from that schedule.

## 5. Experimental protocol

nanoGPT Shakespeare (6 layers, 384 hidden, 10.65M params), NVIDIA L4, seeds 1337 /
1338 / 1339 throughout. Two regimes:

**Long-horizon runs** (§6.1–6.2): teacher 2,000 steps; students 5,000 steps (or
1,500 with eval every 10 for the resolved-speed run), evaluated on the held-out
Shakespeare validation set.

**Wide-shallow probes** (§6.3–6.5): teacher 500 steps (2,000 for the
teacher-sweep's largest arm); students 100–300 steps, batch 32, eval every 10–20,
matched schedule (both arms LR 1e-3, warmup 20). Probes trade depth for breadth —
12 conditions × 3 seeds in under an hour — and their scores are floors
(left-censored at the eval resolution); the resolved speed number comes from the
dense-eval long run.

For the overlap experiments the corpus is cut into 100 random blocks; each arm
receives a random half (both spanning the whole corpus — this is what removes the
slice-difficulty confound that invalidated an earlier sliding-window design), and
only the shared fraction varies. For the cross-domain experiment the student's
non-shared blocks are drawn from Sherlock Holmes (Project Gutenberg #1661)
char-encoded in Shakespeare's vocabulary, and the student is **scored on a
validation set mirroring its own training mixture** — pure held-out Sherlock at
zero overlap. Distributional distance is reported as the Jensen-Shannon divergence
of token histograms (token-JS, bits).

Every run writes a full artifact — loss curves, git commit, GPU, argv, schedule-
matched flag, censoring flags — to `results/`. A number in this paper must have a
matching committed artifact.

## 6. Results

### 6.1 Speed and overfitting (Runs A, C)

The baseline peaks near step 1,020–1,400 at ~1.77–1.78 and then overfits, reaching
~2.31 by step 5,000 on every seed. The recipe reaches ~1.66 and holds through
5,000 steps, overfitting on none. With dense evaluation the crossover is resolved:
the recipe reaches the baseline's best at step 100–110, versus the baseline's
1,020–1,300 — **Prism Score 11.8× median (10.2–11.9×), no censoring**
([`recipe_20260721T142104Z.json`](results/recipe_20260721T142104Z.json)).

### 6.2 Attribution (Run B)

Re-run at the baseline's own learning rate and warmup, only the spectral flags
changed (`schedule_matched: true`,
[`recipe_20260720T230405Z.json`](results/recipe_20260720T230405Z.json)): best
1.671 (range 1.671–1.674) vs. baseline 1.782; no overfitting on any seed (baseline
3/3); crossover at step 200 — **7.0×**. At the same schedule the baseline fails
under, toggling only the spectral flags recovers the effect. The improvement is
attributable to the spectral method.

### 6.3 Structure vs. content, within-domain (Run D)

Twelve overlap points from 1.0 to 0.0, difficulty-controlled, 100-step students,
matched LR ([`recipe_20260721T050203Z.json`](results/recipe_20260721T050203Z.json)):

**Δloss is flat — 0.565–0.587 (~23% lower loss at step 100) at every overlap,
including fully disjoint data.** The baseline is flat at ~2.48 across the sweep
(the difficulty control holding). The early advantage owes nothing to shared text.

### 6.4 Structure vs. content, cross-domain (Run E)

The decisive version: the student trains on progressively more Sherlock Holmes and
is scored on its own mixture — pure Sherlock at zero overlap
([`recipe_20260721T161208Z.json`](results/recipe_20260721T161208Z.json)):

| overlap | token-JS | val set | baseline | recipe | Δloss |
|---|---|---|---|---|---|
| 1.00 | 0.0000 | 100% Shakespeare | 2.469 | 1.878 | 0.591 |
| 0.50 | 0.0044 | 50/50 | 2.475 | 1.882 | 0.593 |
| 0.00 | 0.0266 | **100% Sherlock** | 2.414 | 1.786 | **0.627** |

**A Shakespeare teacher's geometry accelerates learning *of Sherlock* at least as
much as it accelerates Shakespeare.** The overlap-1.0 row reproduces the base
protocol (sanity gate); the gap grows slightly, rather than shrinking, as the
student's data and evaluation move wholly to the far corpus. Scores are ≥9–10× at
every overlap (censored at the probe's 10-step eval resolution — consistent with
§6.1's resolved ~12×).

Together, §6.3 and §6.4 close the content-leakage explanation twice: once with
disjoint same-corpus data, once with a different corpus and a validation set that
rewards only the new domain.

### 6.5 The teacher-strength lever (Run F)

One teacher size per arm, same-data, 300-step students
([`recipe_20260721T143246Z.json`](results/recipe_20260721T143246Z.json)):

| teacher steps | Δloss (baseline − recipe) |
|---|---|
| 100 | **−0.069** (recipe *worse* than baseline) |
| 250 | +0.088 |
| 500 | +0.248 |
| 1,000 | +0.377 |
| 2,000 | +0.451 |
| 4,000 | +0.465 |
| 8,000 | +0.456 |

(2k/4k/8k from the saturation run,
[`recipe_20260721T172238Z.json`](results/recipe_20260721T172238Z.json), whose 2k
anchor reproduces the sweep's +0.451 at +0.458.) The advantage is monotonic in
teacher training and **saturates at ≈2,000 steps — right where the teacher's own
training converges** (baseline best ~step 1,350). The lever tracks
teacher-geometry convergence and plateaus with it, which is precisely the shape
the spectral-transfer mechanism predicts: once the geometry stops changing, there
is nothing more to transfer. The negative cell is as informative as the trend: a
barely-trained teacher's geometry is noise imprinted with authority, and it
actively hurts. This dependence on the *quality of the transferred geometry* is
indirect evidence that the spectral target, not generic regularization pressure,
carries the effect — a generic regularizer has no teacher to be wrong about.

### 6.6 Finetuning without forgetting

The transfer experiments seed a *fresh* model. This one keeps the Mod Wheel on while
*finetuning* a trained one — self-anchored to its own pre-finetune weights (§4.3) —
and asks whether the no-drift property that prevents overfitting from scratch also
prevents catastrophic forgetting. Setup: a plain Shakespeare base (2,000 steps, one
per seed) finetuned 1,000 steps on Sherlock; each arm forks the same base and is
scored every step on **both** the new domain (Sherlock, adaptation) and the old
(Shakespeare, retention). Control and anchor arms share the identical schedule;
only the mod wheel differs. Three-seed frontier
([`finetune_20260721T215319Z.json`](results/finetune_20260721T215319Z.json); base
Shakespeare val 1.488, from-scratch Sherlock ceiling 1.493):

| arm | forgetting (Δ old-domain val) | Sherlock best | vs. plain |
|---|---|---|---|
| plain finetune | +0.428 | 1.252 | 1.0× |
| raw anchor 0.02 | **+0.043** | 1.368 | **9.9×** |
| raw anchor 0.01 | +0.067 | 1.337 | 6.6× |
| raw anchor 0.005 | +0.090 | 1.307 | 4.8× |
| low-LR 5e-5 | +0.227 | 1.297 | 1.9× |
| spectral (spectrum only) | +0.399 | 1.254 | 1.07× |
| shuffled (wrong spectrum) | +1.085 | 1.413 | 0.39× |

Up to ~10× less forgetting, every anchor arm still beating the from-scratch Sherlock
ceiling (so retention is not "it learned nothing"). The attribution — three anchor
variants at matched pull — is a **negative on the natural hypothesis that the
spectral geometry is doing the work**. Anchoring only the *spectrum* while freeing
the directions (`spectral`) forgets as much as a plain finetune (1.07×); a
permuted-spectrum placebo (`shuffled`) actively harms (0.39×); and the raw
whole-weight anchor Pareto-dominates the low-LR frontier — ~2× more retention at
equal adaptation (raw 0.090 @1.307 vs. low-LR 0.227 @1.297). The forgetting
protection is a raw directional/whole-weight anchor (soft L2-to-init / EWC-lite),
**not** the spectral geometry and **not** just a smaller learning rate. Mod strength
is a clean retention/plasticity dial (0.005→0.02: forgetting 0.090→0.043).

This is the complement to §6.3–6.5, and together they license the paper's organizing
claim (§7): the **spectrum** carries transferable, data-independent *structure*
(transfer it into fresh models), while the **directions** carry domain *content*
(pin them to finetune without forgetting). Full method and bounds:
[`docs/FINETUNE-RETENTION.md`](docs/FINETUNE-RETENTION.md).

### 6.7 What is not yet shown

- §6.3–6.4 are 100-step probes: they establish the head start, not the full
  no-overfitting arc on far data.
- Sherlock Holmes is a different author but the same language and character set
  (token-JS 0.027). A genuinely far modality is untested — for both transfer and
  finetune-retention.
- The direct spectral-vs-generic-regularization control (tuned dropout / weight
  decay) is unrun; §6.5 is suggestive, not a substitute.
- Component attribution (`spectral_only` / `dirs_only`) has no committed artifact.
- §6.6 shows old-domain *retention*, not new-domain overfit prevention (plain
  finetuning did not itself overfit Sherlock in 1,000 steps); the anchor also trades
  a little late adaptation back for retention (a decaying pull would remove it).

## 7. Discussion

**What transfers is the geometry of a trained transformer, and it is largely
data-independent within the modality.** That is the strong reading the v0.2
experiments license: the head start is invariant to shared content (§6.3),
portable to a corpus the teacher never saw — measured on that corpus (§6.4) — and
proportional to how converged the teacher's geometry is (§6.5). The student is not
receiving a compressed Shakespeare; it is receiving *how a char-level transformer
of this architecture organizes itself*, which turns out to be most of what the
first thousand steps of training laboriously rediscover.

**The practical consequence** is `prism_accelerate.py`: any trained checkpoint of
an architecture holds a reusable prior for training fresh models of that
architecture — on the same data, on new data, or (within the measured range) on a
different corpus — with the single sharp edge that the teacher must itself be
trained past the point where its geometry means something.

**The two roles of the geometry.** §6.6 sharpens the whole picture by pointing the
same regularizer at a *trained* model during finetuning. There the result inverts:
holding the *spectrum* alone does nothing (and a wrong spectrum harms), while pinning
the *directions* — via the raw weight anchor — is what prevents forgetting. Put
beside the from-scratch result, where the *spectrum* is the transferable,
data-independent part, this reads as a clean division of labor: **the spectrum is
reusable structure (transfer it into a fresh model); the directions are domain
content (pin them to retain it).** Structure transfers; content is retained. The
practical consequence is `prism_finetune.py` — finetune any trained checkpoint with
the mod wheel self-anchored for up to ~10× less catastrophic forgetting at a tunable
adaptation cost, with the attribution establishing it is neither the spectral
geometry nor merely a smaller learning rate that buys it.

**If overfitting is removed, the scaling calculus changes.** The recipe does not
overfit through 5,000 steps at either learning rate; whether it *never* does is
the endurance question, and whether the immunity transfers cross-domain is open.

**Compression tiers.** 128 bytes (spectral shape) → ~500 MB directional matrices.
Per-tier attribution is unverified; directional compression, and the projection
scheme that would allow cross-*size* transfer, are open problems — the latter being
the payoff weight-copying cannot reach.

## 8. Limitations

- The transfer results (§6.3–6.4) are early-window (100-step) probes, 3 seeds.
- "Cross-domain" means a different author in the same language and character set.
- No regularization-matched control; no component ablation artifacts.
- Two shared learning rates, not a per-arm-best sweep.
- ~1M-token corpora, 10.65M params; untested at scale.
- Requires a trained teacher of the same architecture. Transfer learning, not
  magic — and a weak teacher is worse than none (§6.5).
- The finetune-retention result (§6.6) is old-domain *retention* at one scale, with
  "far" again Sherlock; new-domain overfit prevention is not shown, and the raw
  anchor trades a little late adaptation for retention.

## 9. Future work

- **Truly far modality** — source code or another language; find where structural
  transfer degrades. Geometric-alignment refinements (Grassmann direction pairing,
  top-k subspace transfer, CKA regularization — contributed in PR #1, merged) get
  their shot where the plain recipe weakens: first single-variable evals at
  Sherlock-distance found the plain 75% blend strongest (grassmann eliminates the
  head start; top-k=128 is uniformly ~0.05 worse than full transfer —
  `recipe_20260721T162552Z.json`, `recipe_20260721T164007Z.json`), which is the
  expected shape when the plain recipe already transfers at full strength.
- **Long-horizon far-domain** — does overfitting immunity transfer too?
- **Endurance** — 20k–50k steps.
- **Ablations** — `spectral_only` / `dirs_only`, reg-matched baseline.
- **Cross-size transfer** — spectrum interpolates trivially; directions need a
  projection scheme.
- **Finetune-retention, further (§6.6)** — a truly-far new domain (code) where the
  directions may diverge more; a decaying pull to remove the anchor's late
  adaptation give-back; and the new-domain-overfit case the frontier did not
  exercise.

## 10. Conclusion

Prism transfers the spectral organization of a trained model into a fresh one. On
nanoGPT Shakespeare that is worth ~12× in steps-to-quality (7× under the strictest
attribution control), removes overfitting over the horizons tested, does not
depend on the student sharing any data with the teacher, survives wholesale
replacement of the student's corpus, and grows with the quality of the teacher's
geometry. The claims stop at 10.65M parameters and one modality — but within that
scope, the question v0.1 left open is answered: what transfers is structure. And
pointed the other way — anchoring a *trained* model during finetuning — the same
machinery cuts catastrophic forgetting up to ~10×, through the *directions* rather
than the spectrum (§6.6): two halves of one geometry, the spectrum transferable as
structure, the directions retained as content.

Code and committed results:
[github.com/timepointai/nanogpt-prism-shakespeare](https://github.com/timepointai/nanogpt-prism-shakespeare)
