# Prism

**Every trained neural network learns two things. We keep one and throw the other away.**

The weights are what a model learned. But *how* it organized itself to learn —
which directions in weight space it decided mattered, how it distributed energy
across them — is a second artifact, and it goes in the bin the moment training
ends. The next model starts from noise and rediscovers all of it.

Prism is an attempt to keep that second artifact and hand it to the next model.

> **Status: pre-result.** This repo contains a method and a benchmark. It does
> **not** yet contain a validated result — the numbers it used to publish had no
> evidence behind them and have been withdrawn. See [Status](#status). What's
> here is worth your attention only if you find the *question* interesting.

<img src="assets/prism-method.svg" alt="How Prism works: SVD a trained teacher into directions (U, V) and a spectrum (sigma), compress the spectrum to 8 DCT coefficients, inject both into a fresh student, then hold the student toward the spectral target with the mod wheel." width="100%">

## The idea

Take the SVD of a trained model's weights. You get two things: the **directions**
(which subspaces the model decided were worth using) and the **spectrum** (how
much energy it put in each). Together they describe the model's organization
without describing anything it knows.

Give both to a fresh model at init. Then, after every optimizer step, gently pull
it back toward that spectral shape. The student learns its own content from
scratch — Shakespeare, or whatever you point it at — but it doesn't spend its
first thousand steps rediscovering *how a transformer should be shaped*.

Three parts:

1. **Spectral Imprint** — compress the teacher's singular-value distribution to
   8 DCT coefficients per weight group (128 bytes total) and reshape the
   student's spectrum to match.
2. **EigenTransfer** — blend the student's singular vectors 75% toward the
   teacher's, then re-orthogonalize.
3. **Mod Wheel** — after each step, pull weights back toward the spectral target
   (strength 0.01, decay 0.9999).

No parameters are copied. Only geometry.

<img src="assets/prism-taxonomy.svg" alt="Transfer learning taxonomy: random init, then Prism (structure only, proposed), then LoRA/adapters, fine-tuning, and distillation at the far end." width="100%">

## Why it would matter

*These are motivations, not findings.*

**Training could become cumulative.** Today every run's organizational work dies
with it. If a spectral prior transfers, each run makes the next cheaper, and the
cost of a research program stops being linear in the number of runs.

**Overfitting might stop being the ceiling.** Overfitting is what ends most
training runs. A method that suppressed it would let you train longer, use bigger
models on smaller data, and shrink the regularization search. The honest caveat:
the mod wheel *is* a regularizer, and nobody has yet compared it against tuned
dropout and weight decay. Until that ablation exists, "it prevents overfitting"
is a description of what regularizers do, not a discovery.

**Late features would get time to arrive.** The representations that make models
good tend to emerge after the simple ones saturate. A model that overfits first
never gets them.

## The experiment that decides it

The whole method rests on one claim: that what transfers is *structure*, not
*content*. There is exactly one experiment that separates those, and it has not
been run.

<img src="assets/prism-skeptic.svg" alt="The decisive cross-data test: today the teacher and student share split A, so a win proves nothing. The test that matters extracts the fingerprint from split A and trains the student on disjoint split B, where only geometry can cross." width="100%">

As written, the eval trains the teacher and the student on the **same** 80%
split. A win there is unremarkable — the teacher may just be handing over
answers. Extract the fingerprint from one split, train the student on a disjoint
one, and the leak is closed: there is no shared content left, so anything that
survives is geometry.

If the advantage survives, that's a real result, and it implies every checkpoint
ever trained is sitting on a reusable prior nobody extracted. If it collapses,
Prism is distillation with extra steps and should be abandoned. Either way the
answer is worth having, which is the entire reason this repo is public.

## Status

Nothing quantitative is known. That is a stronger statement than it sounds, and
it's worth being precise about why.

This repo previously headlined a **13x Prism Score**, val losses of
**1.7704 / 1.6498**, and **71%** cross-data retention. An audit on 2026-07-16
found that none of those numbers had a source:

- They appear in **no notebook, script, log, or data file** — only in prose.
- All 15 Prism notebooks were saved with **zero executed outputs**, and every run
  wrote its curves to ephemeral Colab `/content` paths that no longer exist. The
  "80+ training runs" left no trace.
- [RESULTS.md](RESULTS.md) and [WHITEPAPER.md](WHITEPAPER.md) **disagree about
  the same experiment**: RESULTS.md's baseline (1.4636) is better than the
  whitepaper's headline *Prism* result (1.6498).
- The **13x** is `1300/100`, where 100 was the first eval step — a resolution
  floor, not a measurement. It's also a ratio against an unusually weak baseline,
  and a weak baseline inflates a ratio for free.
- The **71%** has no source at all: the figure appears nowhere in the skeptic
  notebook, whose verdict cell never ran, and RESULTS.md still lists that test as
  "(running)".
- **Seeds were never varied.** `train.py` hardcoded `manual_seed(1337)` and
  exposed no seed flag, so `--seed=42` would have raised `Unknown config key`.
  The old "seed 42" and "3.8-4.8x across seeds" claims had no mechanism.

The eval has been rebuilt to make that failure mode structurally impossible:
seeds are configurable, runs are multi-seed by default, scores are flagged when
they're left-censored, partial or crashed runs raise instead of being scored, and
every run writes a full artifact — curves, git commit, GPU, argv — to
[`results/`](results/).

**The rule now: a number in a doc must have a matching `results/*.json`.**
`results/` is currently empty of results. That's the honest state.

## Run it

**Colab:** [open the eval →](https://colab.research.google.com/github/timepointai/nanogpt-prism-shakespeare/blob/master/nanogpt_prism_eval.ipynb) — 3 seeds, ~45-60 min on an A100. Downloads an artifact at the end. Save the notebook with outputs.

**Local:**
```bash
git clone https://github.com/timepointai/nanogpt-prism-shakespeare.git
cd nanogpt-prism-shakespeare/src
pip install torch numpy transformers tiktoken datasets
python prism_eval.py                     # seeds 1337,1338,1339
```

```bash
python prism_eval.py --seeds=1337        # one seed — a sample, not a result
python prism_eval.py --method=spectral_only   # ablation: 128-byte shape, no directions
python prism_eval.py --device=mps        # cuda | mps | cpu (auto-detected)
python prism_eval.py --report            # reprint the last artifact
```

A GPU matters here: ~45-60 min on an A100 versus roughly 20 hours on Apple
silicon. Then commit the artifact alongside any claim you make from it.

### Reading a score

The Prism Score is `baseline_steps_to_best / method_steps_to_same_quality`.
It is a **ratio**, so read it next to `baseline_best` — a run whose baseline is
worse than another's produces a bigger score without a better method. That is
precisely how this repo talked itself into a 13x. A score marked `left_censored`
means the target was hit at the first eval and the true crossing is unresolved.

## What would make this real

In order:

1. **The cross-data test.** Fingerprint from split A, student on split B. Without
   it, nothing distinguishes spectral transfer from content leakage.
2. **A regularization-matched baseline.** Prism's mod wheel versus tuned dropout
   / weight decay / early stopping. Without it, "no overfitting" is not a claim
   about Prism.
3. **Multi-seed, with the range published.** Not the best seed. The range.
4. **Scale.** Shakespeare is ~1M characters. GPT-2 124M on OpenWebText is the
   first honest test of whether any of this survives contact with real data.

## Limitations

- **No validated results.** This supersedes everything else here.
- **Cross-data untested** — the load-bearing experiment.
- **No regularization-matched control**, so the anti-overfitting claim is unearned.
- **Shakespeare only** — char-level, ~1M tokens, 10.65M params.
- **A teacher is required.** Prism is transfer learning, not magic.
- **The 128-byte headline is not the method.** 128 bytes is the spectrum; the
  directions that appear to do the real work are ~500 MB. Compressing them is an
  open problem, and until it's solved the compression framing oversells.

## Repo map

```
README.md                  ← you are here
WHITEPAPER.md              ← method in full (results sections withdrawn)
RESULTS.md                 ← earlier findings, unreproduced
results/                   ← eval artifacts. the evidence. empty = nothing proven
nanogpt_prism_eval.ipynb   ← multi-seed Colab eval, writes a committable artifact
src/prism_eval.py          ← the benchmark
src/prism_init.py          ← Spectral Imprint + EigenTransfer + Mod Wheel
src/prism_extract.py       ← extract a fingerprint from any checkpoint
experiments/               ← exploratory notebooks, saved without outputs (not evidence)
```

## License

MIT — see [LICENSE](LICENSE).

A standalone clone of [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej
Karpathy (MIT, © 2022), not a fork. `model.py`, `configurator.py`, `bench.py`,
`sample.py`, and the `data/` preparers are his; `train.py` is his with Prism
hooks added. The Prism code (`prism_init.py`, `prism_extract.py`,
`prism_eval.py`, `config/prism_*.py`) is Timepoint Labs', under the same terms.

---

*A [Timepoint Labs](https://timepointai.com) project by [Sean McDonald](https://x.com/seanmcdonaldxyz).*
