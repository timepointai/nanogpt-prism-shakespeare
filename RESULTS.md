# Prism × nanoGPT — Results (v0.2)

Seven committed runs, three seeds each, on an NVIDIA L4. Every number below is
reproducible from an artifact in [`results/`](results/). Earlier writeups:
[`archive/v0.1/RESULTS.md`](archive/v0.1/RESULTS.md) (the attribution pass),
[`archive/v0.0/RESULTS.md`](archive/v0.0/RESULTS.md) (before attribution).

## The runs

| # | run | artifact |
|---|---|---|
| A | Recipe as tuned (LR 5e-4) vs. baseline (LR 1e-3), 5,000 steps | [`recipe_20260718T002717Z.json`](results/recipe_20260718T002717Z.json) |
| B | **Attribution control**: recipe at the baseline's own LR, 5,000 steps | [`recipe_20260720T230405Z.json`](results/recipe_20260720T230405Z.json) |
| C | **Un-censored speed**: tuned recipe, dense eval (every 10), 1,500 steps | [`recipe_20260721T142104Z.json`](results/recipe_20260721T142104Z.json) |
| D | **Overlap probe**: 12 overlaps × 3 seeds, random blocks, matched LR, 100 steps | [`recipe_20260721T050203Z.json`](results/recipe_20260721T050203Z.json) |
| E | **Cross-domain**: far corpus (Sherlock), student scored on its own data | [`recipe_20260721T161208Z.json`](results/recipe_20260721T161208Z.json) |
| F | **Teacher-strength sweep**: teachers 100→2,000 steps, 300-step students | [`recipe_20260721T143246Z.json`](results/recipe_20260721T143246Z.json) |
| G | **Lever eval — grassmann alignment alone** (single variable vs. Run E's plain control) | [`recipe_20260721T162552Z.json`](results/recipe_20260721T162552Z.json) |
| H | **Lever eval — top-k=128 alone** (single variable vs. Run E's plain control) | [`recipe_20260721T164007Z.json`](results/recipe_20260721T164007Z.json) |
| I | **Teacher saturation**: teachers 2,000 / 4,000 / 8,000 — finds the plateau | [`recipe_20260721T172238Z.json`](results/recipe_20260721T172238Z.json) |
| — | Sliding-window overlap sweep — **record only, interpretation retired** (difficulty confound; superseded by D) | [`recipe_20260721T022342Z.json`](results/recipe_20260721T022342Z.json) |
| — | Far-corpus sweep scored on Shakespeare val — superseded by E (see scope note in [`results/README.md`](results/README.md)) | [`recipe_20260721T153218Z.json`](results/recipe_20260721T153218Z.json) |
| J | **Finetune-retention frontier** (§7): 11 arms × 3 seeds — raw anchor vs. low-LR vs. spectral vs. shuffled | [`finetune_20260721T215319Z.json`](results/finetune_20260721T215319Z.json) |
| K | Finetune-retention core pair (§7): plain vs. raw-0.01 anchor, 3 seeds — the 5.73× headline | [`finetune_20260721T201955Z.json`](results/finetune_20260721T201955Z.json) |
| — | Finetune go/no-go probe (1 seed) — confirms plain finetuning forgets | [`finetune_20260721T200415Z.json`](results/finetune_20260721T200415Z.json) |
| — | Finetune attribution probe (1 seed) — superseded by J | [`finetune_20260721T212408Z.json`](results/finetune_20260721T212408Z.json) |

## 1. Speed: ~12×, resolved (Run C)

The tuned recipe's speed was left-censored in Run A (it crossed the baseline's best
before the first eval at step 100, so "≥13–14×" was a floor with coarse
resolution). Run C was built to resolve it: eval every 10 steps, 1,500-step
horizon.

| seed | baseline best @step | recipe reaches it @step | Prism Score |
|---|---|---|---|
| 1337 | 1.7663 @1,020 | @100 | 10.2× |
| 1338 | 1.7784 @1,190 | @100 | 11.9× |
| 1339 | 1.7674 @1,300 | @110 | 11.8× |

**Median 11.8× (10.2–11.9×), `left_censored: false` on all seeds.** The earlier
"≥13–14×" language is retired — the two runs don't conflict (different eval
resolution and horizon place the baseline's best differently), but the resolved
number is the number: **~12× as tuned, 7× at matched LR (Run B, below)**.

## 2. Attribution: it's the method, not the schedule (Runs A + B)

| | Baseline (LR 1e-3) | Recipe (LR 5e-4) | Recipe (LR 1e-3, matched) |
|---|---|---|---|
| Best val loss (median of 3) | 1.782 | 1.656 | **1.671** |
| Val loss @ step 5,000 | ~2.31 (overfit) | ~1.66 (stable) | ~1.67 (stable) |
| Overfits within 5,000 steps | yes — 3/3 | no — 0/3 | no — 0/3 |
| Prism Score | — | 11.8× (from Run C) | **7.0× (6.5–7.0×)** |

Run B holds the learning rate and warmup identical to the baseline
(`schedule_matched: true`) and toggles only the spectral flags. The effect
survives on every seed: lower best loss, no overfitting, 7× to the baseline's
best. The "maybe it was just the lower learning rate" explanation is closed.

## 3. Structure, not content (Run D)

The teacher/student overlap dial, difficulty-controlled: the corpus is cut into
100 random blocks; each arm gets a random half (both spanning the whole corpus, so
neither arm gets an "easier" slice); only the *shared fraction* varies. 12 overlap
points from 1.0 (identical data) to 0.0 (fully disjoint), 3 seeds each, 100-step
students, matched LR.

**The early advantage is flat: Δloss 0.565–0.587 (~23% lower loss at step 100) at
overlap 1.0, at 0.0, and everywhere between.** Baseline flat at ~2.48 across the
sweep (the difficulty control working). Nothing the student gains depends on
sharing text with the teacher.

(An earlier sliding-window version of this sweep showed the advantage *declining*
with overlap — that was a confound: window position correlated with text
difficulty, the baseline drifted 1.88→1.55 across the sweep. The artifact stays
committed as a record; its interpretation is retired.)

## 4. Cross-domain: the head start survives a change of corpus (Run E)

The student's non-shared blocks come from **Sherlock Holmes** (`data/far.txt`,
Project Gutenberg #1661), char-encoded in Shakespeare's vocabulary, and — the part
that makes the test decisive — the student is **scored on a validation set drawn
from its own training mixture** (`far_val: true`): pure held-out Shakespeare at
overlap 1.0, pure held-out Sherlock at overlap 0.0. Matched LR, 3 seeds, 100-step
students. The overlap-1.0 row reproduces the base protocol exactly (the sanity
gate).

| overlap | token-JS | val set | baseline | recipe | Δloss |
|---|---|---|---|---|---|
| 1.00 | 0.0000 | 100% Shakespeare | 2.469 | 1.878 | 0.591 |
| 0.75 | 0.0008 | 76% Shakespeare | 2.487 | 1.898 | 0.589 |
| 0.50 | 0.0044 | 50/50 | 2.475 | 1.882 | 0.593 |
| 0.25 | 0.0121 | 32% Shakespeare | 2.464 | 1.851 | 0.613 |
| 0.00 | 0.0266 | **100% Sherlock** | 2.414 | 1.786 | **0.627** |

**A Shakespeare teacher's geometry accelerates learning of Sherlock at least as
much as it accelerates Shakespeare.** The gap does not shrink with distance — it
grows slightly. Prism Scores are ≥9–10× at every overlap (left-censored at this
probe's 10-step resolution — consistent with Run C's resolved ~12×). Combined with
Run D, this closes the content-leakage explanation twice over: the transfer is
structural, and the structure is portable across corpora.

An earlier version of this run scored all students on the *Shakespeare* val set —
which conflates accelerated learning of the new domain with retention of the
teacher's. That artifact ([`…T153218Z`](results/recipe_20260721T153218Z.json)) is
kept with a scope note; Run E is the corrected protocol.

## 5. The lever: teacher strength (Run F)

Same-data probe, 300-step students, one teacher size per arm:

| teacher steps | baseline best | recipe best | Δloss | score |
|---|---|---|---|---|
| 100 | 2.180 | 2.249 | **−0.069** | never reached |
| 250 | 2.180 | 2.092 | 0.088 | 2.3× |
| 500 | 2.180 | 1.932 | 0.248 | 7.5× |
| 1,000 | 2.180 | 1.804 | 0.377 | 7.5× |
| 2,000 | 2.180 | 1.729 | 0.451 | 7.5× |

And the saturation run (Run I, teachers 2,000 / 4,000 / 8,000, same rig):

| teacher steps | Δloss |
|---|---|
| 2,000 | +0.458 (anchor — reproduces Run F's +0.451) |
| 4,000 | +0.465 |
| 8,000 | +0.456 |

**Monotonic to ≈2,000 teacher steps, then a plateau at Δloss ≈ +0.46** — the
lever saturates right where the teacher itself converges (its best is ~step
1,350). The advantage tracks teacher-geometry convergence: once the geometry has
converged, more teacher training adds nothing. And the sharp edge: a
barely-trained teacher is *actively worse than random init* — its geometry is
noise being imprinted with authority. If you use `prism_accelerate.py`, use a
trained teacher.

This is also indirect evidence *for* the spectral mechanism: the effect tracks the
quality of the teacher's geometry, which a generic regularizer has no access to.

## 6. Lever evals: the plain blend wins at this distance (Runs G, H)

Two geometric-alignment refinements (contributed in PR #1) evaluated one variable
at a time against Run E's plain control — same rig, same seeds, same overlaps,
only the lever flag added:

- **Grassmann geodesic pairing** (`--align_mode=grassmann`): Δloss **−0.012 to
  −0.023** at every overlap — the recipe never reaches the baseline's best. The
  geodesic re-pairing eliminates the head start the plain 75% blend delivers
  (+0.59 to +0.63 on the identical rig).
- **Top-k=128 subspace transfer** (`--align_topk=128`): the head start survives
  (Δloss +0.54 to +0.57, scores ~5×) but is uniformly **+0.04 to +0.06 worse**
  than transferring all ~384 directions. The discarded low-energy tail carries
  useful geometry at this scale.

Verdict at Sherlock-distance: **the plain full-direction blend is the strongest
configuration measured** — there is no headroom for geometric refinement here,
because the plain recipe already transfers at full strength (Run E). The levers'
real test is a corpus far enough that the plain recipe degrades (see next
experiments).

## 7. Finetuning without forgetting (Runs J, K)

Runs A–H are all *from-scratch*. A separate study points the Mod Wheel the other
way: kept on during a **finetune** and self-anchored to a trained model's own
pre-finetune weights, does it prevent catastrophic forgetting? Setup: a plain
Shakespeare base (2,000 steps, one per seed) finetuned 1,000 steps on Sherlock; each
arm forks the same base and is scored every step on **both** Sherlock (adaptation)
and Shakespeare (retention). The control and every anchor arm share the identical
schedule — only the Mod Wheel differs.

**The technique works: up to ~10× less forgetting** (Run J, 3-seed frontier,
[`finetune_20260721T215319Z.json`](results/finetune_20260721T215319Z.json); base
Shakespeare val 1.488, from-scratch Sherlock ceiling 1.493):

| arm | forgetting (Shakespeare val climb) | Sherlock best | vs. plain |
|---|---|---|---|
| plain finetune | +0.428 | 1.252 | 1.0× |
| raw anchor 0.02 | **+0.043** | 1.368 | **9.9×** |
| raw anchor 0.01 | +0.067 | 1.337 | 6.6× |
| raw anchor 0.005 | +0.090 | 1.307 | 4.8× |
| low-LR 5e-5 | +0.227 | 1.297 | 1.9× |
| low-LR 1e-4 | +0.282 | 1.277 | 1.5× |
| spectral (spectrum only) | +0.399 | 1.254 | 1.07× |
| shuffled (wrong spectrum) | +1.085 | 1.413 | 0.39× |

Every anchor arm still beats the from-scratch Sherlock ceiling (1.493), so retention
is never "it didn't learn Sherlock." Run K is the same-schedule core pair (plain vs.
the raw-0.01 anchor) at the headline **5.73×**
([`finetune_20260721T201955Z.json`](results/finetune_20260721T201955Z.json)).

**The attribution is a negative on the obvious guess.** Anchoring only the
*spectrum* (freeing the directions) does essentially nothing for retention (1.07×, ≈
plain); a *wrong*-spectrum placebo actively harms (0.39×); and the raw anchor
**Pareto-dominates** the low-LR frontier — at comparable adaptation it retains ~2×
more (raw 0.090 @1.307 vs. low-LR 0.227 @1.297). The protection is a raw
directional/whole-weight anchor (L2-to-init / EWC-lite), **not** the spectral
geometry and **not** just a smaller learning rate.

This separates PRISM's two regimes cleanly: **from scratch the *spectrum* carries
transferable, data-independent structure (§3, §4); in finetuning the *directions*
carry retained content and must be pinned.** Method, full frontier, and bounds:
[`docs/FINETUNE-RETENTION.md`](docs/FINETUNE-RETENTION.md).

## 8. The arc: PRISM pretraining + finetuning compound (Runs L, M)

Does combining the two directions buy anything? The test: a PRISM-pretrained base vs.
a plain base as the thing you finetune-anchor, at **matched** Shakespeare quality
(`train.py --stop_val_target`). 3 seeds, all bases matched at ~1.79, finetuned on
Sherlock with the *identical* raw anchor. Medians:

| base (matched ~1.79) | forgets Shakespeare | learns Sherlock | run |
|---|---|---|---|
| plain (LR 1e-3) | +0.050 | 1.572 | [`…T172023Z`](results/arc_20260724T172023Z.json) (L) |
| plain_fastlr (LR 5e-4, no spectral) | +0.084 | 1.571 | [`…T210817Z`](results/arc_20260724T210817Z.json) (M) |
| **prism** (spectral) | **−0.001** | **1.438** | (L, M) |

**A PRISM base finetunes with ≈ zero forgetting** (vs +0.05 for plain) **and adapts
~8% better**, at matched quality. And it's the **spectral geometry**: the
`plain_fastlr` control (plain at PRISM's learning rate, no spectral) forgets *more*
than plain and adapts identically — so the schedule doesn't do it, only the spectral
machinery does. Pretraining and finetuning with PRISM reinforce each other. Full study
and bounds: [`docs/UNIFIED-ARC.md`](docs/UNIFIED-ARC.md).

## 9. Prior-Fused PRISM: a T9-style statistical prior compounds (Run N)

Structure comes in more than one flavour. Fuse a fixed shared **n-gram prior** into the
logits (product of experts, `final = model_logits + λ·log p_ngram(next | last C chars)`),
so the model learns only the residual, and stack it with PRISM's spectral init. A startling
input: a context-3 char n-gram already predicts Shakespeare val at **2.5722 bits/char** ≈
the neural baseline's 2.565 best. 1 seed, steps-to-baseline-best
([`…T011443Z`](results/prior_20260725T011443Z.json)):

| arm | init (bits/ch) | best (bits/ch) | speedup |
|---|---|---|---|
| baseline | 6.18 | 2.565 | 1.0× |
| prism | 4.47 | 2.245 | 15× |
| prior (n-gram only) | 2.70 | 2.504 | 3.8× |
| **prism_prior** | 2.68 | **2.235** | **30×** |

**The hybrid doubles PRISM (15× → 30×) and reaches the best loss of all four** — below
baseline, PRISM-alone, and the n-gram floor (2.50). The T9 statistical prior and PRISM's
geometry compound: the prior pre-loads local structure for free, PRISM pre-loads the
representational geometry, and PRISM's geometry is what breaks the residual below the
n-gram floor. The "30×" is the compounding of two free priors (the fused arms start near
baseline), not PRISM training 30× faster alone — the defensible claim is that the hybrid
beats PRISM alone *and* reaches a better loss. The literal 1000× (reach baseline *at
init*) is a knife-edge here — context-3 tops out *at* baseline, and a logit-gate meant to
enable it backfired ([`…T013656Z`](results/prior_20260725T013656Z.json)); a clean version
needs a context-4+ prior. Full study: [`docs/PRIOR-FUSED-PRISM.md`](docs/PRIOR-FUSED-PRISM.md).

## What this does NOT establish

- **Early-window scope.** Runs D and E measure the head start at step 100
  (init-dominated). They prove the transfer is structural and domain-portable;
  they do not show the full no-overfitting arc to convergence on far data (Runs
  A–C are long-horizon but same-data).
- **Sherlock is still English prose** (token-JS 0.027, same character set). A
  genuinely different modality — code, another language — is untested.
- **Spectral vs. generic regularization** is still not isolated by a direct
  control (tuned dropout / weight decay). Run F's teacher-dependence is
  suggestive, not a substitute.
- **Two shared learning rates, not a per-arm-best sweep.**
- **Scale.** 10.65M params, ~1M-token corpora.
- **Finetune-retention (§7) shows old-domain *retention*, not new-domain overfit
  prevention** (plain finetuning didn't itself overfit Sherlock in 1,000 steps), and
  "far" is again Sherlock (same alphabet). The anchor also trades a little late
  adaptation back for retention (a decaying pull would remove it).

## The next experiments, in order

1. **Truly far modality** — the far-corpus arm pointed at source code or another
   language (token-JS ≫ 0.03); find where structural transfer degrades.
2. **Long-horizon far-domain** — the 0%-overlap Sherlock student to convergence:
   does no-overfitting transfer across domains too?
3. **Endurance** — the recipe to 20k–50k steps.
4. **Ablations** — `spectral_only` / `dirs_only` / reg-matched baseline.
5. **Cross-size transfer** — spectrum interpolates; directions need a projection.

## Reproduce it

```bash
cd src
pip install torch numpy transformers tiktoken datasets
python prism_eval.py --student_steps=1500 --eval_every=10 --eval_iters=50   # Run C
python prism_eval.py --method_lr=1e-3 --method_warmup=100                   # Run B
python prism_eval.py --teacher_steps=500 --student_steps=100 --eval_every=10 \
  --eval_iters=40 --batch_size=32 --method_lr=1e-3 --method_warmup=20 \
  --baseline_warmup=20 \
  --overlap=1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.0               # Run D
python prism_eval.py --teacher_sweep=100,250,500,1000,2000 \
  --student_steps=300 --eval_every=20 --eval_iters=40 --batch_size=32       # Run F
python prism_eval.py --teacher_steps=500 --student_steps=100 --eval_every=10 \
  --eval_iters=40 --batch_size=32 --method_lr=1e-3 --method_warmup=20 \
  --baseline_warmup=20 --overlap=1.0,0.75,0.5,0.25,0.0 \
  --far_corpus=data/far.txt --far_val                                       # Run E
python prism_eval.py --report                                               # reprint last artifact
```

The finetune-retention frontier (Runs J, K) runs headless on Modal:

```bash
modal run --detach prism_modal_finetune.py --extra \
 "--tag=r2b --base_steps=2000 --ft_steps=1000 --eval_every=25 --seeds=1337,1338,1339 \
  --arms=base,plain,raw_lo,raw_mid,raw_hi,lowlr_a,lowlr_b,lowlr_c,spectral,shuffled,scratch_ceiling \
  --learning_rate=3e-4 --min_lr=3e-5 --batch_size=32 --block_size=256 --far_corpus=data/far.txt"
```

The eval is stepwise and resumable, records provenance and whether the schedule
was matched, and refuses to score a partial or crashed run. **A number in these
docs must have a matching `results/*.json`.**
