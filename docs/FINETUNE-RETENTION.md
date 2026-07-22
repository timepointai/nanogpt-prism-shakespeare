# Finetuning without losing the advantage

*A PRISM side-study. Branch `prism-finetune-retention`. Every number below has a
committed `results/finetune_*.json`; single-seed runs are labelled probes, not
results.*

## The question

PRISM's headline result is about training **from scratch**: hand a fresh model the
spectral geometry of a trained one and it trains ~12× faster and stops overfitting.
Real models, though, get **finetuned** — adapted to new data after training. Does the
same machinery let you finetune a trained model on a new domain **without losing what
it already knew** (catastrophic forgetting)? And if so, *what* in the geometry is doing
the work?

## The answer

**Yes. Keep the Mod Wheel engaged during the finetune, self-anchored to the model's own
pre-finetune weights (a constant per-step pull). It cuts catastrophic forgetting by up
to ~10× at a small, tunable adaptation cost — and it beats the two obvious baselines it
could have been confused with.**

Setup for every number here: a plain nanoGPT model trained 2,000 steps on Shakespeare
(the "base", one per seed), then finetuned 1,000 steps on Sherlock Holmes (`data/far.txt`,
char-encoded in Shakespeare's vocabulary, token-JS ≈ 0.027). Each arm forks the *same*
base checkpoint and is scored every step on **both** validation sets: Sherlock
(*adaptation* — did it learn the new domain?) and Shakespeare (*retention* — did it keep
the old one?). "Forgetting" = the Shakespeare val loss climb from the base. The control
(`plain`) and every anchor arm share the identical LR/warmup/steps/data — they differ
only in the Mod Wheel — so a difference is attributable.

## The frontier (3 seeds, medians)

Base Shakespeare val **1.488**; from-scratch Sherlock ceiling **1.493** (best a fresh
model reaches on Sherlock in 1,000 steps). [`…T215319Z`](../results/finetune_20260721T215319Z.json)

| arm | what it does | forgetting ↓ | Sherlock best ↓ | less forgetting vs plain | overfits Sherlock |
|---|---|---|---|---|---|
| `plain` | finetune, no anchor | +0.428 | 1.252 | 1.0× | no |
| **`raw_hi`** | anchor, pull 0.02 | **+0.043** | 1.368 | **9.9×** | yes¹ |
| **`raw_mid`** | anchor, pull 0.01 | +0.067 | 1.337 | 6.6× | yes¹ |
| **`raw_lo`** | anchor, pull 0.005 | +0.090 | 1.307 | 4.8× | yes¹ |
| `lowlr_c` | no anchor, LR 5e-5 | +0.227 | 1.297 | 1.9× | no |
| `lowlr_b` | no anchor, LR 1e-4 | +0.282 | 1.277 | 1.5× | no |
| `lowlr_a` | no anchor, LR 1.5e-4 | +0.326 | 1.266 | 1.3× | no |
| `spectral` | anchor the *spectrum* only | +0.399 | 1.254 | 1.07× | no |
| `shuffled` | anchor a *wrong* spectrum | +1.085 | 1.413 | 0.39× | no |

*Every anchor arm still beats the from-scratch Sherlock ceiling (1.493) on adaptation —
so retention is never "it just didn't learn Sherlock." ¹The raw-anchor arms reach their
best Sherlock loss and then give a little of it back as the constant pull accumulates —
the stability/plasticity tradeoff made explicit, not classic overfitting.*

The earlier same-schedule 3-seed run of the core pair (`plain` vs the raw-0.01 anchor)
is the headline **5.73×** ([`…T201955Z`](../results/finetune_20260721T201955Z.json));
the go/no-go probe that first confirmed forgetting happens at all is
[`…T200415Z`](../results/finetune_20260721T200415Z.json); the single-seed attribution
probe is [`…T212408Z`](../results/finetune_20260721T212408Z.json).

## What it is — and, more usefully, what it is *not*

The forgetting protection is real and large, but two null hypotheses had to die before it
could be attributed to anything specific. Both did:

- **It is *not* the spectral geometry.** The `spectral` arm holds the base's
  singular-value *spectrum* fixed while letting the directions (U/V) adapt freely — the
  part of the geometry PRISM's from-scratch result says is the transferable, data-
  independent structure. It does essentially **nothing** for retention (1.07× — the same
  as a plain finetune). And a `shuffled` arm, which anchors a *permuted* base spectrum,
  actively **harms** (0.39× — worse than no anchor). So holding the spectrum is neither
  necessary nor sufficient; the *specific* spectrum only matters in that a wrong one hurts.

- **It is *not* just a smaller learning rate.** Lowering the LR also reduces forgetting
  (that is the `lowlr_*` frontier) — the obvious cheap alternative. But the raw-anchor
  frontier **Pareto-dominates** it: at comparable Sherlock adaptation, the anchor retains
  roughly **2× more** Shakespeare. `raw_lo` forgets **0.090** at Sherlock 1.307, while the
  best-retaining low-LR point `lowlr_c` forgets **0.227** at Sherlock 1.297 — nearly the
  same adaptation, less than half the forgetting. To match the anchor's retention by LR
  alone you would have to lower it until adaptation collapses.

What *does* do the work is the **raw, whole-weight anchor** — pulling every weight back
toward its pre-finetune value each step. That is soft **L2-to-init / EWC-lite**, delivered
through PRISM's Mod Wheel. Mod strength is a clean retention/plasticity dial: 0.005 → 0.02
moves you from (forget 0.090, adapt 1.307) to (forget 0.043, adapt 1.368).

## Why the negative is the interesting part

The result cleanly **separates PRISM's two regimes**, and they fit together:

| regime | what carries the signal | what you do with it |
|---|---|---|
| **From-scratch** (PRISM v0.2) | the **spectrum** — data-*independent* structure | *transfer* it → ~12× speedup, portable across corpora |
| **Finetuning** (this study) | the **directions** — domain *content* | *pin* them → up to ~10× less forgetting |

The spectrum is the reusable *shape* (why it transfers even to Sherlock); the directions
are where the *specific content* lives — which is exactly why holding the spectrum alone
doesn't protect old knowledge, and pinning the directions does. Structure transfers;
content must be retained.

## Honest bounds

- **Scale.** nanoGPT char-level, 10.65M params, Shakespeare→Sherlock (same alphabet,
  token-JS ≈ 0.027). A genuinely far modality (code, another language) is untested and may
  behave differently.
- **Forgetting, not new-domain overfitting.** Plain finetuning here forgets the old domain
  but does not itself overfit Sherlock in 1,000 steps (`overfit_averted` is not
  demonstrated). The shown claim is old-domain *retention*.
- **"Spectral" is dropped, deliberately.** Given the attribution above, this ships as a
  *proximal (raw) self-anchor*, not a spectral one — the evidence does not support the
  stronger word.
- **The anchor's late give-back.** The constant pull means the raw-anchor arms trade a
  little late-stage adaptation back for retention (the `overfits: yes` column). Their
  *best* adaptation is still strong; a decaying pull or an early-stop on the adaptation val
  would remove it if that tradeoff isn't wanted.

## Use it

`src/prism_finetune.py` applies the technique to any nanoGPT checkpoint:

```bash
cd src
python prism_finetune.py \
    --base_ckpt out-shakespeare-char/ckpt.pt \
    --new_data sherlock_ft --retain_val shakespeare_char \
    --out_dir out-finetuned --mod 0.01 --ft_steps 1000
# add --plain to see the same finetune forget, for comparison
```

## Reproduce it

```bash
# the full attribution frontier (what the table above is):
modal run --detach prism_modal_finetune.py --extra \
 "--tag=r2b --base_steps=2000 --ft_steps=1000 --eval_every=25 --seeds=1337,1338,1339 \
  --arms=base,plain,raw_lo,raw_mid,raw_hi,lowlr_a,lowlr_b,lowlr_c,spectral,shuffled,scratch_ceiling \
  --learning_rate=3e-4 --min_lr=3e-5 --batch_size=32 --block_size=256 --far_corpus=data/far.txt"
```

`src/prism_finetune_eval.py` is the benchmark; `src/prism_selftest.py` covers the mod-wheel
and spectral-target invariants (30 offline tests). The anchor modes live on `train.py`'s
resume path (`prism_anchor_mode` = `raw` | `spectral` | `shuffled`).
