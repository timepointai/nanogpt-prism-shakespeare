# PRISM × finetuning: a PRISM-pretrained base is a better anchor

*A PRISM side-study — "the unified arc." Every number below has a committed
`results/arc_*.json`; single-seed runs are labelled probes, not results.*

## The question

PRISM has two results, and they are the *same* mod-wheel operation
(`W ← (1−s)·W + s·W_target`) with a different target:

- **From scratch** ([RESULTS §1–6](../RESULTS.md)): transfer a teacher's *spectrum*
  → a fresh model trains ~12× faster. The spectrum is data-independent **structure**.
- **Finetuning** ([FINETUNE-RETENTION](FINETUNE-RETENTION.md)): keep the mod wheel on,
  self-anchored to a model's own weights → up to ~10× less catastrophic forgetting.
  The anchor pins the **directions** (content).

So the natural question: does *combining* them buy anything beyond running the two in
sequence? The sharpest, cheapest test:

> **Does a PRISM-pretrained base make a better finetune-anchor than a plain base at
> *matched* old-domain quality?**

If PRISM's healthier, non-overfit geometry is a better thing to anchor, the answer is
yes — *and* it should be attributable to the geometry, not to the prism base being a
better model (that's what "matched quality" controls for).

## The setup

Per seed: train two Shakespeare bases to the **same validation target** (the
matched-quality control, via `train.py --stop_val_target`) — `base_plain` (plain from
scratch) and `base_prism` (PRISM-accelerated: spectral init from a teacher + mod
wheel). PRISM reaches the target in far fewer steps; both stop at ≈ the same
Shakespeare val (~1.79). Then **finetune each on Sherlock (1,000 steps) with the
identical raw self-anchor** — only the base differs — and score adaptation (Sherlock)
and retention (Shakespeare) every step. An attribution arm, `base_plain_fastlr` (plain
at PRISM's learning rate, *no* spectral machinery), isolates the geometry from the
training schedule.

## The result (3 seeds, all bases matched at ~1.79)

[`…T172023Z`](../results/arc_20260724T172023Z.json) (the matched pair) ·
[`…T210817Z`](../results/arc_20260724T210817Z.json) (the attribution)

| base (all matched ~1.79 Shakespeare) | reached target | forgets Shakespeare | learns Sherlock |
|---|---|---|---|
| `plain` (LR 1e-3) | @~590 steps | +0.050 | 1.572 |
| **`plain_fastlr`** (LR 5e-4, *no spectral*) | @~730 steps | **+0.084** | 1.571 |
| **`prism`** (spectral) | **@~65 steps (8.8× faster)** | **−0.001** | **1.438** |

**At matched quality, the PRISM base essentially eliminates catastrophic forgetting**
(≈ 0 nats, vs +0.05 for plain — on two of three seeds Shakespeare val *improves*
slightly during the Sherlock finetune) **and adapts ~8% better** (1.438 vs 1.572,
tight across seeds). This is not "prism is a better model" — the bases are matched.

**And it is the spectral geometry, not the schedule.** The `plain_fastlr` control
(plain at PRISM's exact learning rate, no spectral machinery) forgets *more* than plain
and adapts identically — the faster LR alone does nothing. Only the spectral arm gets
the zero-forgetting + better-adaptation. (The `forget_ratio` is unstable because prism's
forgetting is ≈ 0; read the *difference*.)

The single-seed probe that opened the line — directionally positive before the match
was tight — is [`…T154528Z`](../results/arc_20260724T154528Z.json).

## Why it matters — the through-line

PRISM's three findings now tell one story, and it is always the same thing:

| result | what carries the signal | what you do |
|---|---|---|
| from-scratch transfer | the **spectrum** — data-independent structure | *transfer* it → ~12× faster |
| finetune retention | the **directions** — domain content | *pin* them → ~10× less forgetting |
| **the arc (this)** | a base's **spectral health** | *pretrain with PRISM* → a much better anchor |

Pretraining with PRISM and anchoring during finetuning **compound**: a spectrally
healthy base is a dramatically better thing to hold onto. Combining PRISM + finetuning
yields genuine synergy — and it traces back to the same place every PRISM result does:
the spectral geometry is special.

## Honest bounds

- **Matched only in a narrow band (~1.79).** PRISM reaches quality plain *cannot*
  (plain's best is ~1.78, non-overfit), so a clean match is only possible just above
  plain's best. All bases here sit at ~1.79.
- **The adaptation edge is geometry, not the residual val gap.** The prism base's
  matched-val gap to plain is ~0.018; its Sherlock adaptation is 0.13 better — far too
  large to be the 0.018. `plain_fastlr` at the *same* val as plain adapts the *same* as
  plain, confirming it.
- **Sherlock is mild-far** (token-JS 0.027, same alphabet). A truly-far domain (code)
  is untested — the top item in [NEXT-EXPERIMENTS](NEXT-EXPERIMENTS.md).
- **Scale:** nanoGPT char, 10.65M params.

## Reproduce

```bash
# the full 3-way attribution frontier (Round 2), headless on Modal:
modal run --detach prism_modal_arc.py --extra \
 "--base_target_val=1.80 --base_max_iters=3000 --base_eval_every=5 --teacher_steps=2000 \
  --ft_steps=1000 --eval_every=25 --eval_iters=200 --batch_size=32 --block_size=256 \
  --seeds=1337,1338,1339 --base_methods=plain,prism,plain_fastlr \
  --far_corpus=data/far.txt --tag=r2"
```

`src/prism_arc_eval.py` is the benchmark (trains matched bases, finetunes each with the
identical anchor, compares); `train.py --stop_val_target` is the matched-quality control
and the forced-eval-at-resume decouples the finetune eval cadence. Artifacts: R0
`…T154528Z`, R1 `…T172023Z`, R2 `…T210817Z`.
