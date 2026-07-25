# Prism

**v0.2** — the transfer results. ([v0.1 archived here](archive/v0.1/) — the
attribution pass; [v0.0 here](archive/v0.0/) — before the confound was ruled out.)

<img src="assets/prism-flashlight.svg" alt="A spectrographic flashlight for models: a trained checkpoint's raw weights enter a prism as white light and split into four spectral bands — attention, FFN up, FFN down, embedding — each carrying a spectrum and directions. A reversed prism recombines the bands into a fresh model that trains about 12 times faster. Geometry crosses; content never does." width="100%">

<a href="https://timepointai.github.io/nanogpt-prism-shakespeare/docs/how-prism-works.html"><img src="assets/prism-explainer-button.svg" alt="HOW PRISM WORKS — THE VISUAL EXPLAINER: the flashlight metaphor, node-level math, graph-level transfer, and the measurements" width="100%"></a>

---

**Hand a fresh model nothing but the *spectral geometry* of a trained one — no
weights, no data — and it trains ~12× faster, stops overfitting, and the advantage
survives moving to a different corpus.** On nanoGPT Shakespeare the recipe reaches
the from-scratch baseline's best quality in ~100 steps instead of ~1,200 (11.8×
median, three seeds, a resolved measurement — not a bound). The head start does not
depend on the student seeing the teacher's data: it is identical whether the two
share 100% or 0% of their training text, and it holds — slightly *grows* — when the
student trains and is evaluated on a different author entirely (Shakespeare teacher
→ Sherlock Holmes student). And it tracks the teacher's geometry exactly: the
advantage grows with teacher training and saturates right where the teacher itself
converges (≈2,000 steps; 4k and 8k teachers add nothing).

That is one of *two* things a trained model's geometry buys you. Point the same Mod
Wheel the other way — to *hold* a trained model in place rather than seed a fresh one —
and it lets you finetune onto a new domain with **up to ~10× less catastrophic
forgetting** ([the other direction](#the-other-direction-finetuning-without-forgetting)).
Structure transfers; content is retained.

<img src="assets/prism-transfer.svg" alt="Left: validation loss at step 100 versus teacher/student data overlap, with the student scored on its own data mixture — the from-scratch baseline sits at ~2.47 while the Prism recipe sits at ~1.88, and the gap grows from 0.591 at full overlap to 0.627 when the student trains and is scored entirely on Sherlock Holmes. Right: the advantage versus teacher training steps — negative at a 100-step teacher, rising monotonically and saturating at a +0.46 plateau around 2,000 steps, where the teacher itself converges; 4k and 8k teachers are flat." width="100%">

| The four measurements (all committed under [results/](results/)) | number | artifact |
|---|---|---|
| **Speed** — steps to the baseline's best quality, tuned recipe, dense eval | **11.8× median** (10.2–11.9×, resolved) | [`…T142104Z`](results/recipe_20260721T142104Z.json) |
| **Attribution** — same at the baseline's own LR, only spectral flags differ | **7.0×** (6.5–7.0×, resolved) | [`…T230405Z`](results/recipe_20260720T230405Z.json) |
| **Structure, not content** — early advantage vs. data overlap, difficulty-controlled | **flat**: Δloss 0.57–0.59 at overlap 1.0 *and* 0.0 | [`…T050203Z`](results/recipe_20260721T050203Z.json) |
| **Cross-domain** — student trains *and is scored* on Sherlock Holmes | Δloss **0.591 → 0.627** (grows with distance) | [`…T161208Z`](results/recipe_20260721T161208Z.json) |

Plus the lever, now fully mapped: the advantage grows monotonically with teacher
training (Δloss −0.069 at a 100-step teacher → +0.46 at 2,000) and **saturates at
≈2,000 steps — right where the teacher itself converges** (4k/8k teachers flat;
[`…T143246Z`](results/recipe_20260721T143246Z.json),
[`…T172238Z`](results/recipe_20260721T172238Z.json)). And the recipe **does not
overfit** through 5,000 steps at either learning rate tested, where the baseline
collapses from 1.78 to ~2.31 on every seed
([`…T002717Z`](results/recipe_20260718T002717Z.json)).

## Why this is interesting

The obvious explanation for any teacher→student speedup is that the teacher leaked
its *content*. v0.2's experiments were built to kill that explanation, and they did,
twice:

1. **Same corpus, disjoint data.** Cut Shakespeare into 100 random blocks, give the
   teacher and student overlapping or disjoint halves (both spanning the whole
   corpus, so difficulty is controlled). The early advantage is **identical at 100%
   and 0% overlap**. Nothing the student gains depends on shared text.
2. **Different corpus entirely.** Swap the student's data for Sherlock Holmes and —
   the part that matters — **score the student on held-out Sherlock, not
   Shakespeare**. A Shakespeare teacher's geometry accelerates learning *of
   Sherlock* exactly as much as it accelerates Shakespeare (Δloss 0.627 vs. 0.591;
   recipe 1.79 vs. baseline 2.41 on Sherlock validation at step 100).

So what transfers is not "a Shakespeare model." It is the geometry a trained
char-level transformer converges to — which appears to be largely
*data-independent within the modality*. Any trained checkpoint carries a reusable
structural prior; Prism extracts and applies it. That is also what a pruning
schedule or a generic regularizer cannot do: they have no teacher geometry to point
at, and the teacher-strength sweep shows the effect tracks exactly that geometry's
quality — a barely-trained teacher's geometry actively *hurts* (Δloss −0.069),
a well-trained one helps more the longer it trained.

## The other direction: finetuning without forgetting

Everything above hands a trained model's geometry to a *fresh* one. The same
machinery, pointed the other way, solves a different problem: adapting a *trained*
model to a new domain **without wrecking what it already knew** (catastrophic
forgetting). Keep the Mod Wheel on during the finetune, self-anchored to the model's
own pre-finetune weights, and it cuts forgetting by **up to ~10×** while it still
learns the new domain (3 seeds, Shakespeare → Sherlock, 1,000 finetune steps):

| finetune arm | forgets Shakespeare | learns Sherlock | vs. plain |
|---|---|---|---|
| plain finetune | +0.43 | 1.25 | 1× |
| **Mod Wheel anchor (strength 0.02)** | **+0.04** | 1.37 | **~10× less forgetting** |

*(forgetting = old-domain val-loss climb from the base, in nats; lower is better on
both. The anchored model still beats a from-scratch Sherlock model, so it genuinely
learns the new domain — it isn't just refusing to move.)*

### How it works

Finetuning is just more gradient descent, now on new data — and that descent walks
the weights away from the configuration that knew the old domain. **Forgetting is that
drift.** So snapshot the trained weights **W₀** before you start, and after *every*
optimizer step, nudge the live weights a little back toward that snapshot:

```
# after each optimizer step, for every weight matrix:
W  ←  (1 − s)·W  +  s·W₀        # s = 0.01, constant (no decay)
```

Gradient descent still dominates each step, so the model keeps learning the new
domain — but the nudge continuously pulls it back toward `W₀`, so it never drifts far
from what it already knew. That's the whole method, and it's not a new mechanism: it's
PRISM's **Mod Wheel**, unchanged, with its target set to the model's *own pre-finetune
weights* instead of a fresh spectrally-shaped init. Two knobs matter:

- **The pull is constant** (`s` fixed at ~0.01), where the from-scratch recipe *decays*
  it. From scratch, the mod wheel is annealing a one-time reshape and should fade; in a
  finetune there's no reshape to settle — you want the same gentle restoring force for
  the whole run.
- **`s` is the retention/plasticity dial.** Turn it up for more retention (less
  forgetting, slightly slower to adapt), down for more plasticity. The frontier in
  [`docs/FINETUNE-RETENTION.md`](docs/FINETUNE-RETENTION.md) walks it 0.005 → 0.02.

No teacher, no fingerprint, no extra storage — the anchor is the model's own starting
point. `prism_finetune.py` ([below](#use-it)) does exactly this to any checkpoint.

The attribution is the interesting part, and it lands on a **negative** to the
obvious guess. The protection is **not** the spectral geometry — anchoring only the
*spectrum* (freeing the directions) forgets as much as a plain finetune (1.07×), and
a *wrong*-spectrum placebo actively harms (0.39×). And it is **not** just a smaller
learning rate — the anchor Pareto-dominates the low-LR frontier (~2× more retention
at equal adaptation). What does it is a raw **directional / whole-weight** anchor.

That splits PRISM's geometry cleanly in two, and the halves are complementary:

- **Spectrum → structure.** Data-*independent*; *transfer* it into fresh models —
  which is why the head start ports even to Sherlock.
- **Directions → content.** Domain-*specific*; *pin* them to finetune without
  forgetting.

**Structure transfers; content must be retained.** Full study, 3-seed frontier, and
honest bounds: [`docs/FINETUNE-RETENTION.md`](docs/FINETUNE-RETENTION.md).

### They compound

And the two directions **stack**: a base that was itself PRISM-*pretrained* makes a
much better thing to anchor-*finetune*. At *matched* Shakespeare quality, a PRISM base
finetunes on Sherlock with **≈ zero forgetting** (vs +0.05 nats for a plain base) and
adapts **~8% better** — and a schedule-matched plain control confirms it's the **spectral
geometry**, not the learning rate. A spectrally-healthy base is a dramatically better
anchor; pretraining and finetuning with PRISM genuinely reinforce each other. Full study:
[`docs/UNIFIED-ARC.md`](docs/UNIFIED-ARC.md).

## Use it

Two entry points — one per direction. **Transfer** into a fresh model with
`prism_accelerate.py`:

```bash
cd src
python prism_accelerate.py \
    --teacher_ckpt path/to/trained/ckpt.pt \
    --out_dir out-accelerated \
    -- --max_iters=2000 --dataset=your_dataset
```

It extracts the spectral fingerprint from the checkpoint, then trains a fresh model
initialized and regularized by it. Everything after `--` passes through to
`train.py`. Two things the measurements say you should know:

- **The teacher must be trained.** A weak teacher (100 steps here) is *worse than
  random init*. The advantage grows with teacher training and saturates once the
  teacher converges (≈2,000 steps here) — so train the teacher to convergence,
  and no further.
- Teacher and student must share the same architecture — the directional transfer
  is dimension-specific. Cross-size transfer is future work.

**Finetune** a trained model without forgetting with `prism_finetune.py`:

```bash
cd src
python prism_finetune.py \
    --base_ckpt path/to/trained/ckpt.pt \
    --new_data your_new_dataset --retain_val your_old_dataset \
    --out_dir out-finetuned --mod 0.01 --ft_steps 1000
# add --plain to run the same finetune with the anchor off, for comparison
```

It forks the checkpoint and finetunes it with the Mod Wheel self-anchored (constant
pull). `--retain_val` scores the *old* domain alongside the new one, so you can see
the forgetting the anchor is preventing. `--mod` is the retention/plasticity dial
(higher = more retention, less adaptation).

## Start here (humans and agents)

Everything needed to verify or extend the results, in order:

```bash
git clone https://github.com/timepointai/nanogpt-prism-shakespeare.git
cd nanogpt-prism-shakespeare/src
pip install torch numpy transformers tiktoken datasets
python prism_selftest.py          # 25 offline invariant tests, CPU, ~1 min — start here
python prism_eval.py --teacher_steps=10 --student_steps=10 --eval_every=5 \
  --eval_iters=2 --seeds=1337 --batch_size=4     # tiny end-to-end smoke, CPU, ~1 min
```

Then pick a real experiment (each ~20 min on any modern GPU; commands under
[Reproduce it](#reproduce-it)). Ground rules the tooling enforces, worth knowing
before you run:

1. **Evidence = `results/*.json`.** Every run writes a full artifact (loss curves,
   provenance, git commit, argv, censoring flags). A claim without a committed
   artifact is not a result in this repo.
2. **Reading a score:** `prism_score` is a ratio — always read it against
   `baseline_best` (a weak baseline inflates it). `left_censored: true` means a
   floor, not a measurement. `schedule_matched: true` means only the spectral
   flags differed — the attribution-grade comparison.
3. **Runs are resumable:** stages bank to `.prism_runs/<run-key>/`; the same
   command resumes, any changed knob gets a fresh key. You cannot accidentally
   resume one experiment onto another.
4. **Probes must keep `warmup ≪ student_steps`** (the probe schedule uses
   warmup 20 for 100-step runs) — warmup that spans the whole run flattens both
   arms and voids the comparison.
5. **One variable at a time.** The lever knobs (`--align_mode`, `--align_topk`,
   `--cka`, …) fold into the run key; test each against the plain recipe on an
   otherwise-identical rig, never stacked.

A machine-readable summary of the method and all measurements is embedded at the
bottom of the [visual explainer](https://timepointai.github.io/nanogpt-prism-shakespeare/docs/how-prism-works.html) —
a standalone, self-contained page (no scripts, no external requests), served
first-party from GitHub Pages; the source is [`docs/how-prism-works.html`](docs/how-prism-works.html).

## The idea

Take the SVD of a trained model's weights. You get the **directions** (which
subspaces the model decided were worth using) and the **spectrum** (how much energy
it put in each). Together they describe the model's organization without describing
anything it knows.

Give both to a fresh model at init. Then, after every optimizer step, gently pull it
back toward that spectral shape. The student learns its own content from scratch —
but it doesn't spend its first thousand steps rediscovering *how a transformer
should be shaped*, and it doesn't drift out of that shape later (which is what
overfitting looks like geometrically).

1. **Spectral Imprint** — compress the teacher's singular-value distribution to
   8 DCT coefficients per weight group (~128 bytes total) and reshape the student's
   spectrum to match.
2. **EigenTransfer** — blend the student's singular vectors 75% toward the
   teacher's, then re-orthogonalize.
3. **Mod Wheel** — after each step, pull weights back toward the spectral target
   (strength 0.01, decay 0.9999). A continuous, zero-storage spectral regularizer.

No parameters are copied. Only geometry.

<img src="assets/prism-method.svg" alt="How Prism works: SVD a trained teacher into directions (U, V) and a spectrum, compress the spectrum to 8 DCT coefficients, inject both into a fresh student, then hold the student toward the spectral target with the mod wheel." width="100%">

<img src="assets/prism-imprint.svg" alt="Deriving the spectral imprint: SVD each trained weight matrix, normalize and average the singular values per group, then least-squares fit 8 cosine (DCT) coefficients. The plot overlays a real group-averaged spectrum against its reconstruction from just 8 coefficients, mean absolute error about 0.03." width="100%">

The long-horizon picture (from the v0.1 attribution runs, 5,000 steps):

<img src="assets/prism-result.svg" alt="Validation loss over training, nanoGPT Shakespeare, three seeds each. The baseline (LR 1e-3) falls to ~1.78 near step 1,400 then overfits to ~2.30 by step 5,000. The Prism recipe at LR 5e-4 holds ~1.66; the recipe run at the baseline's own LR of 1e-3, where only the spectral flags differ, holds ~1.67 and also never overfits." width="100%">

## Honest bounds on the claims

Every number above has a committed artifact, and every artifact has a scope:

- **The transfer results are early-window probes.** The overlap and cross-domain
  measurements are at step 100 (init-dominated), 3 seeds, matched LR. They prove
  the head start is structural and domain-portable; they do not yet show the full
  no-overfitting story to convergence on far data. (The 11.8× and no-overfitting
  results are longer runs — 1,500 and 5,000 steps — but same-data.)
- **"Different corpus" means Sherlock Holmes** — a different author, but still
  English prose over the same character vocabulary (token-JS 0.027). A genuinely
  different modality (code, another language) is the next frontier and may be where
  the plain recipe finally needs help.
- **Speed scores at the probe scale are floors.** The 100-step probes cross the
  baseline's best at or before the first eval (≥9–10×, left-censored); the resolved
  11.8× comes from the dense-eval run built to resolve it. Earlier "≥13–14×" floor
  language is retired: measured beats hoped, and measured is ~12×.
- **Spectral vs. generic regularization is not yet isolated.** The mod wheel beats
  a schedule-matched baseline, and the teacher-strength dependence argues the
  spectral *target* matters — but a tuned-dropout/weight-decay control has not been
  run.
- **Scale:** everything is nanoGPT Shakespeare-char, 10.65M params, plus Sherlock.
  Nothing here has met a production model.

**The rule:** a number in these docs has a matching `results/*.json`. The eval
enforces it — multi-seed, resumable, scores flagged when left-censored, partial or
crashed runs raise instead of being scored, and every run records its curves, git
commit, GPU, argv, and whether the schedule was matched.

## Reproduce it

**Modal (headless, how the committed runs were produced):**

```bash
pip install modal && modal setup                        # one-time auth
modal run --detach prism_modal.py \
  --extra "--student_steps=1500 --eval_every=10 --eval_iters=50"   # the 11.8× run
modal run --detach prism_modal.py \
  --extra "--method_lr=1e-3 --method_warmup=100"        # the matched-LR control (7×)
```

**The transfer probes (local or Modal):**

```bash
cd src
python prism_eval.py --teacher_steps=500 --student_steps=100 --eval_every=10 \
  --eval_iters=40 --batch_size=32 --method_lr=1e-3 --method_warmup=20 \
  --baseline_warmup=20 --overlap=1.0,0.75,0.5,0.25,0.0        # overlap sweep
python prism_eval.py --teacher_sweep=100,250,500,1000,2000 \
  --student_steps=300 --eval_every=20 --eval_iters=40 --batch_size=32   # teacher lever
python prism_eval.py --report                                  # reprint last artifact
```

The cross-domain arm reproduces with
`--overlap=1.0,0.75,0.5,0.25,0.0 --far_corpus=data/far.txt --far_val` (plus the
probe schedule above).

**The finetune-retention frontier (Modal):**

```bash
modal run --detach prism_modal_finetune.py --extra \
 "--tag=r2b --base_steps=2000 --ft_steps=1000 --eval_every=25 --seeds=1337,1338,1339 \
  --arms=base,plain,raw_lo,raw_mid,raw_hi,lowlr_a,lowlr_b,lowlr_c,spectral,shuffled,scratch_ceiling \
  --learning_rate=3e-4 --min_lr=3e-5 --batch_size=32 --block_size=256 --far_corpus=data/far.txt"
```

`prism_finetune_eval.py` trains one base per seed, forks it into each arm, and scores
adaptation (new domain) and retention (old domain) every step. See
[`docs/FINETUNE-RETENTION.md`](docs/FINETUNE-RETENTION.md) for the full frontier.

## What's next

1. **A truly far modality.** Sherlock is still English prose. Point the far-corpus
   arm at source code or another language, where token-JS is an order of magnitude
   larger, and find where structural transfer finally degrades — that's where the
   geometric-alignment refinements (below) get their shot.
2. **Long-horizon far-domain run.** Take the 0%-overlap Sherlock student to
   convergence: does the no-overfitting property also transfer across domains?
3. **The endurance run** — the recipe hadn't overfit at 5,000 steps; take it to
   20,000–50,000 and find where, if ever, it destabilizes.
4. **Ablations** — `spectral_only` / `dirs_only`, and a regularization-matched
   baseline.
5. **Cross-size transfer** — the spectrum interpolates trivially; the directions
   need a projection scheme. The most differentiated payoff: weight-copying can't
   change architecture, geometry might.
6. **Finetune-retention, further** — a truly-far new domain (code) where the
   directions may diverge more; a decaying pull to remove the anchor's late
   adaptation give-back; and the new-domain-overfit case the current frontier didn't
   exercise. See [`docs/FINETUNE-RETENTION.md`](docs/FINETUNE-RETENTION.md).
7. **Teacher-free init & the rest** — the biggest open lead: if the spectrum is a
   *modality* constant, you could PRISM-init a fresh model from an unrelated corpus's
   fingerprint — **no teacher** — and keep the ~12× head start. That plus the truly-far
   corpus, the continuous single-run, continual multi-domain, and the "unfold curve"
   are laid out, with run commands, for the next agent in
   [`docs/NEXT-EXPERIMENTS.md`](docs/NEXT-EXPERIMENTS.md).

Geometric-alignment refinements contributed by Leonard Wang (PR #1, now merged)
are available as opt-in flags — Grassmann geodesic direction pairing
(`--align_mode=grassmann`), top-k subspace transfer (`--align_topk`), per-layer
spectra, a CKA representational regularizer — alongside the far-corpus evaluation
infrastructure the cross-domain result was measured with. First single-variable
evaluations (committed): at Sherlock-distance the plain 75% blend is already the
strongest configuration — grassmann pairing eliminates the head start
([`…T162552Z`](results/recipe_20260721T162552Z.json)), and top-k=128 keeps it but
uniformly ~0.05 worse than transferring all directions
([`…T164007Z`](results/recipe_20260721T164007Z.json)). Their real test is the
truly-far modality above, where the plain recipe may finally need the help.

## Repo map

```
README.md                  ← you are here (v0.2)
docs/how-prism-works.html  ← the visual explainer (standalone, self-contained)
docs/FINETUNE-RETENTION.md ← finetune without forgetting — the study + frontier
docs/UNIFIED-ARC.md        ← PRISM pretraining + finetuning compound — the arc study
docs/NEXT-EXPERIMENTS.md   ← handoff: the ranked next experiments + run commands
WHITEPAPER.md              ← method + experiments in full
RESULTS.md                 ← the committed runs and what they do / don't show
results/                   ← eval artifacts. the evidence.
archive/v0.1/              ← the attribution pass (matched-LR control)
archive/v0.0/              ← the earliest pass, before attribution
prism_modal.py             ← headless Modal runner (used for the committed runs)
prism_modal_leo.py         ← isolated runner for the leo-test branch
prism_modal_finetune.py    ← isolated runner for the finetune-retention benchmark
prism_modal_arc.py         ← isolated runner for the arc (base-interaction) benchmark
unfold_curve.py            ← exploratory: the Σ*/"unfold curve" first cut (next-exp #5)
src/prism_eval.py          ← the benchmark (schedule/overlap/teacher-sweep knobs)
src/prism_finetune_eval.py ← the finetune-retention benchmark (anchor frontier)
src/prism_arc_eval.py      ← the arc benchmark (matched-quality base interaction)
src/prism_accelerate.py    ← apply Prism to any checkpoint (the "use it" entry point)
src/prism_finetune.py      ← finetune any checkpoint without forgetting (the technique)
src/prism_init.py          ← Spectral Imprint + EigenTransfer + Mod Wheel + spectral_target
src/prism_extract.py       ← extract a fingerprint from any checkpoint
src/prism_selftest.py      ← 30 offline invariant tests (transfer levers + finetune anchor)
data/far.txt               ← Sherlock Holmes (Project Gutenberg #1661, public
                             domain, boilerplate stripped) — the far corpus
```

## License

MIT — see [LICENSE](LICENSE).

A standalone clone of [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej
Karpathy (MIT, © 2022), not a fork. `model.py`, `configurator.py`, `bench.py`,
`sample.py`, and the `data/` preparers are his; `train.py` is his with Prism hooks
added. The Prism code (`prism_init.py`, `prism_extract.py`, `prism_eval.py`,
`prism_accelerate.py`, `config/prism_*.py`) is Timepoint Labs', under the same
terms.

---

*A [Timepoint Labs](https://timepointai.com) project by [Sean McDonald](https://x.com/seanmcdonaldxyz).*
