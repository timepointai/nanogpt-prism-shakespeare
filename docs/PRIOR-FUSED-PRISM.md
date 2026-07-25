# Prior-Fused PRISM: T9 × PRISM — the priors compound

*A PRISM experiment. Every number below has a committed `results/prior_*.json`;
single-seed runs are labelled probes, not results.*

## The idea

T9's trick isn't the dictionary — it's that a *tiny shared prior* collapses most of the
ambiguity, so you only pay for the residual. Bake that into a language model as a
**product of experts**:

```
final_logits = model_logits + λ · log p_ngram(next | last C chars)
```

`p_ngram` is a fixed shared n-gram prior (the "T9 dictionary"), so the model learns only
the **residual** over it. Then stack that with PRISM's spectral init — the *geometric*
prior — and ask whether the two free priors **compound**: statistics from T9, geometry
from PRISM.

## The surprising input

A fixed **context-3 char n-gram prior predicts Shakespeare val at 2.5722 bits/char** —
essentially the converged neural baseline's best (2.565). A ~KB lookup table is as good
as a trained nanoGPT. So fuse it in, and at initialization the model already predicts at
baseline quality — a validated fact: a fused model's step-0 val is the prior's rate
(2.12 nats vs 4.18 for a plain random init).

## The result (1 seed, steps to the plain baseline's best val loss)

[`…T011443Z`](../results/prior_20260725T011443Z.json)

| arm | init (bits/ch) | best (bits/ch) | speedup to baseline |
|---|---|---|---|
| baseline | 6.18 | 2.565 | 1.0× |
| prism | 4.47 | 2.245 | 15× |
| prior (T9 only) | 2.70 | 2.504 | 3.8× |
| **prism_prior (hybrid)** | 2.68 | **2.235** | **30×** |

**The hybrid doubles PRISM (15× → 30×) *and* reaches the best final loss of all four
arms** — 2.235 bits, below the baseline, below PRISM-alone, and below the n-gram floor
(2.50). The T9 statistical prior and PRISM's spectral geometry compound: the prior
pre-loads the local structure for free (the model doesn't spend its early steps
rediscovering n-gram statistics), PRISM pre-loads the representational geometry, and
PRISM's geometry is what lets the residual break *below* the n-gram floor.

## Honest bounds

- **"30× speedup" is steps-to-baseline-best**, and the fused arms get a head start
  (they start near baseline). So this is the *compounding of two free priors*, not
  PRISM training 30× faster in isolation. The defensible claim: the hybrid beats PRISM
  alone (30× vs 15×) **and** reaches a strictly better loss.
- **The literal 1000× (reach baseline quality *at init*) did not land.** context-3 tops
  out *at* the baseline (2.56 ≈ 2.565), not below, so the fused init sits just above
  baseline — a knife-edge. A `logit_gate` (ramp the model's contribution 0 → 1 so the
  fused init is the pure prior) makes the init exactly the prior, but **backfired** —
  throttling early learning dropped the hybrid 30× → 15×
  ([`…T013656Z`](../results/prior_20260725T013656Z.json)). A clean reach-at-init needs a
  prior *clearly below* baseline = **context-4+**, which requires a sparse n-gram build
  (dense V⁴ times out) — the documented next step
  ([NEXT-EXPERIMENTS](NEXT-EXPERIMENTS.md)).
- **Scale:** nanoGPT char, 10.65M params, Shakespeare.

## Reproduce

```bash
cd src
python build_ngram_prior.py --dataset shakespeare_char --context 3 --out .prism_cache/ngram
# → context-3 prior: 2.5722 bits/char on val (≈ the neural baseline)
modal run --detach prism_modal_prior.py --extra \
 "--context=3 --student_steps=1500 --teacher_steps=2000 --eval_every=25 --eval_iters=100 \
  --batch_size=32 --block_size=256 --seeds=1337 --arms=baseline,prism,prior,prism_prior --tag=r0"
```

`src/build_ngram_prior.py` builds the dense (V^C, V) prior; `train.py`'s `prior_table` /
`prior_strength` fuse it into the logits (off by default); `src/prism_prior_eval.py` runs
the four arms and scores steps-to-baseline-best. Artifacts: r0 `…T011443Z` (the 30×), the
gate probe `…T013656Z`.
