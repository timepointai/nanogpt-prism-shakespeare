# Prism

**v0.1** — first *attributed* result. ([v0.0 archived here](archive/v0.0/) — the
earlier pass, before the learning-rate confound was ruled out.)

---

**Hand a fresh model nothing but the *spectral geometry* of a trained one — no
weights, no data — and it stops overfitting.** On nanoGPT Shakespeare it reaches a
lower loss than the baseline, holds it without overfitting, and gets to the
baseline's best quality several times faster. And — the thing v0.1 adds — this
survives the obvious objection: run it at the baseline's *own* learning rate, so
the only difference is the spectral machinery, and the effect is still there. **It's
the method, not the schedule.**

<img src="assets/prism-result.svg" alt="Validation loss over training, nanoGPT Shakespeare, three seeds each. The baseline (LR 1e-3) falls to ~1.78 near step 1,400 then overfits to ~2.30 by step 5,000. The Prism recipe at LR 5e-4 holds ~1.66; the recipe run at the baseline's own LR of 1e-3, where only the spectral flags differ, holds ~1.67 and also never overfits — so the effect is the spectral method, not the learning rate." width="100%">

| nanoGPT Shakespeare, 5,000 steps, 3 seeds | Baseline (LR 1e-3) | Recipe (LR 5e-4) | **Recipe (LR 1e-3, matched)** |
|---|---|---|---|
| Best validation loss | 1.782 | 1.656 | **1.671** |
| Loss at step 5,000 | **2.30 — overfit** | 1.66 — stable | **1.67 — stable** |
| Overfits within 5,000 steps | yes — 3/3 | no — 0/3 | **no — 0/3** |
| Steps to baseline's best | — | ≤100 (≥13×, *censored*) | **200 (7×, measured)** |

The rightmost column is the one that matters: **same learning rate as the
baseline, only the spectral flags flipped.** Where the baseline overfits and decays
to 2.30, the recipe reaches ~1.67 and holds — and crosses the baseline's best
quality ~7× faster (a resolved number this time, not a censored lower bound). Two
different learning rates, three seeds each, all committed under [results/](results/).

## What v0.1 settles that v0.0 did not

v0.0 reported the left two columns and was honest that they were **confounded**:
the recipe also lowered the learning rate, so "Prism vs. baseline" was really
"recipe config vs. default config," and a too-high baseline learning rate alone
could have produced the whole picture.

v0.1 ran the control. Holding the learning rate identical and toggling only the
spectral flags, the effect survives — lower loss, no overfitting, faster to parity.
So the confound is ruled out: **the improvement is attributable to the spectral
method.** That is the entire reason this version exists.

## The idea

Take the SVD of a trained model's weights. You get two things: the **directions**
(which subspaces the model decided were worth using) and the **spectrum** (how much
energy it put in each). Together they describe the model's organization without
describing anything it knows.

Give both to a fresh model at init. Then, after every optimizer step, gently pull
it back toward that spectral shape. The student learns its own content from
scratch — but it doesn't spend its first thousand steps rediscovering *how a
transformer should be shaped*, and it doesn't drift out of that shape later (which
is what overfitting looks like geometrically).

Three parts:

1. **Spectral Imprint** — compress the teacher's singular-value distribution to
   8 DCT coefficients per weight group (~128 bytes total) and reshape the student's
   spectrum to match.
2. **EigenTransfer** — blend the student's singular vectors 75% toward the
   teacher's, then re-orthogonalize.
3. **Mod Wheel** — after each step, pull weights back toward the spectral target
   (strength 0.01, decay 0.9999). A continuous, zero-storage spectral regularizer.

No parameters are copied. Only geometry.

<img src="assets/prism-method.svg" alt="How Prism works: SVD a trained teacher into directions (U, V) and a spectrum, compress the spectrum to 8 DCT coefficients, inject both into a fresh student, then hold the student toward the spectral target with the mod wheel." width="100%">

### How the imprint is derived

SVD every trained weight matrix, normalize its singular values, and average them
within each weight group (attention, FFN up, FFN down, embedding). Each group's
averaged spectrum is a smooth, decaying curve — and a smooth curve is what a few
cosine terms capture well. So it's least-squares fit to **8 cosine (DCT)
coefficients** in an inverse-softplus space (which keeps the reconstruction
positive). Those 8 numbers per group softplus-reconstruct back to the spectrum,
which the student's singular values are reshaped to match.

<img src="assets/prism-imprint.svg" alt="Deriving the spectral imprint: SVD each trained weight matrix, normalize and average the singular values per group, then least-squares fit 8 cosine (DCT) coefficients. The plot overlays a real group-averaged spectrum against its reconstruction from just 8 coefficients, mean absolute error about 0.03." width="100%">

The plot above is real: a group-averaged spectrum against its reconstruction from
8 coefficients (mean absolute error ~0.03). Eight numbers carry the shape; only the
extreme low-energy tail drifts. That smoothness is why the *spectrum* compresses so
hard — but note the *directions* (U, V) do not, which is why "128 bytes" describes
one tier of the method, not all of it (see [Limitations](#limitations)).

## Status

**Measured, committed, reproducible** (two artifacts under [results/](results/), an
NVIDIA L4, three seeds each):

- The recipe reaches a **lower** best validation loss than the baseline (1.66–1.67
  vs. 1.78) and **does not overfit** where the baseline collapses to ~2.30 — on
  every seed, at **both** learning rates tested.
- It reaches the baseline's best quality **~7× faster** at matched learning rate
  (measured), and even faster at LR 5e-4 (there only a lower bound, because it
  crossed before the first eval).
- **The effect is attributable to the spectral method**, not the learning rate:
  the matched-LR run isolates the spectral flags as the only difference and the
  effect holds.

**Not yet established** — and these are what would make it *important*, in order:

1. **Structure vs. content.** Every run so far is *same-data*: the teacher and
   student share the 80% training split. So this proves Prism transfers something
   real and useful — not yet that what transfers is *structure* rather than the
   teacher leaking *content*. The [cross-data test](#whats-next) is the decider.
2. **Spectral vs. generic regularization.** The mod wheel is a regularizer, and
   we've shown it beats a schedule-matched baseline — but not that the *spectral*
   nature specifically (vs. tuned dropout / weight decay) is what prevents the
   overfitting. The `spectral_only` / `dirs_only` ablations settle that.
3. **Scale.** Shakespeare is ~1M characters, 10.65M params. Nothing here has met
   real data.

**The rule:** a number in these docs has a matching `results/*.json`. The eval
enforces it — multi-seed, resumable, scores flagged when left-censored, partial or
crashed runs raise instead of being scored, and every run records its curves, git
commit, GPU, argv, and whether the schedule was matched.

## Run it

**Modal (headless, how the committed runs were produced):**

```bash
pip install modal && modal setup                        # one-time auth
modal run --detach prism_modal.py                       # recipe @ 5e-4, 3 seeds
modal run --detach prism_modal.py \
  --extra "--method_lr=1e-3 --method_warmup=100"        # the matched-LR control
```

It runs fire-and-forget (survives a dropped laptop connection), resumes from banked
stages, and leaves the artifact on a Volume to fetch and commit.

**Local:**

```bash
git clone https://github.com/timepointai/nanogpt-prism-shakespeare.git
cd nanogpt-prism-shakespeare/src
pip install torch numpy transformers tiktoken datasets
python prism_eval.py                                    # recipe @ 5e-4, seeds 1337,1338,1339
python prism_eval.py --method_lr=1e-3 --method_warmup=100   # matched-LR control (only spectral flags differ)
python prism_eval.py --report                           # reprint the last artifact
```

On an L4 a seed is ~27 min (teacher ~2.5 min + two 5,000-step students at ~12 min),
so a 3-seed run is ~80 min. The model is small enough that a bigger GPU buys little.

### Reading a score

The Prism Score is `baseline_steps_to_best / method_steps_to_same_quality` — a
**ratio**, so read it next to `baseline_best`; a weaker baseline inflates it. A
score flagged `left_censored` means the method hit target at the first eval, so the
number is a lower bound (that's what the ≥13× at LR 5e-4 is). At matched LR the
crossover moved to step 200, so the 7× is resolved.

## What's next

1. **The cross-data test** — the decisive experiment. Fingerprint the teacher on
   one half of the data, train the student on the disjoint other half, where no
   shared content remains. If the advantage survives, the transfer is *structural*,
   and every checkpoint ever trained holds a reusable prior nobody extracted. If it
   collapses, Prism is distillation with extra steps. This is the one that decides
   how big a deal this is.
2. **The endurance run** — the recipe hadn't overfit at 5,000 steps; take it to
   20,000–50,000 and find where, if ever, it destabilizes. Either it holds (the mod
   wheel truly removes the overfitting ceiling) or it breaks at step *N* (a real,
   measured ceiling).
3. **Ablations** — `spectral_only` / `dirs_only`, and a regularization-matched
   baseline, to separate the spectral contribution from generic regularization.
4. **Scale** — GPT-2 124M on OpenWebText.

## Limitations

- **Same-data.** Every committed run trains teacher and student on the same split;
  it does not yet exclude content leakage. This supersedes everything below.
- **Cross-data untested** — the experiment that separates structure from leakage.
- **No regularization-matched control**, so "the spectral structure prevents
  overfitting" (vs. generic regularization) is not yet isolated.
- **Shakespeare only** — char-level, ~1M tokens, 10.65M params.
- **A teacher is required.** Prism is transfer learning, not magic.
- **"128 bytes" is the spectrum, not the method.** The directions (U, V) are
  ~500 MB uncompressed; compressing them is an open problem.

## Repo map

```
README.md                  ← you are here (v0.1)
WHITEPAPER.md              ← method + experiments in full
RESULTS.md                 ← the committed runs and what they do / don't show
results/                   ← eval artifacts. the evidence. two committed runs
archive/v0.0/              ← the earlier pass, before attribution was established
nanogpt_prism_eval.ipynb   ← multi-seed Colab eval
prism_modal.py             ← headless Modal runner (used for the committed runs)
src/prism_eval.py          ← the benchmark (schedule knobs for the matched test)
src/prism_init.py          ← Spectral Imprint + EigenTransfer + Mod Wheel
src/prism_extract.py       ← extract a fingerprint from any checkpoint
```

## License

MIT — see [LICENSE](LICENSE).

A standalone clone of [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej
Karpathy (MIT, © 2022), not a fork. `model.py`, `configurator.py`, `bench.py`,
`sample.py`, and the `data/` preparers are his; `train.py` is his with Prism hooks
added. The Prism code (`prism_init.py`, `prism_extract.py`, `prism_eval.py`,
`config/prism_*.py`) is Timepoint Labs', under the same terms.

---

*A [Timepoint Labs](https://timepointai.com) project by [Sean McDonald](https://x.com/seanmcdonaldxyz).*
