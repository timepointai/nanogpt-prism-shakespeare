# Next experiments — a handoff for the next agent

You are picking up a mature PRISM codebase with three committed, artifact-backed
results. This doc tells you what's proven, what machinery exists, and the ranked
next experiments — each with a hypothesis, a concrete run command, and what a win
looks like. Read [README](../README.md) "Start here" first for the ground rules;
they are non-negotiable (evidence = committed `results/*.json`, matched schedules for
attribution, wide/shallow probes before deep runs, 3 seeds for a result, no mocks).

## What's already proven

1. **From-scratch spectral transfer** ([RESULTS](../RESULTS.md)): a teacher's spectrum
   → a fresh model trains ~12× faster; the spectrum is data-independent *structure*.
2. **Finetune without forgetting** ([FINETUNE-RETENTION](FINETUNE-RETENTION.md)): keep
   the mod wheel on, self-anchored → up to ~10× less forgetting; it's a raw
   *directional* anchor (not spectral, not just a smaller LR).
3. **The arc** ([UNIFIED-ARC](UNIFIED-ARC.md)): a PRISM-pretrained base is a much
   better finetune-anchor than a plain base *at matched quality* (≈0 forgetting + ~8%
   better adaptation), and it's the **spectral geometry** (a schedule-matched plain base
   doesn't do it).

The through-line: **the spectral geometry is special.** Every experiment below tests a
consequence of that.

## The machinery (all on `master`)

- `src/prism_eval.py` — from-scratch benchmark (`prism-eval/1`). Knobs: `--overlap`,
  `--teacher_sweep`, `--far_corpus`/`--far_val`, `--method_lr`/`--method_warmup`.
- `src/prism_finetune_eval.py` — finetune-retention benchmark (`prism-finetune/1`).
- `src/prism_arc_eval.py` — the base-interaction / arc benchmark (`prism-arc/1`).
- `src/prism_accelerate.py` / `src/prism_finetune.py` — the "apply it" entry points.
- `src/train.py` knobs you'll reuse: `prism_init`, `prism_mod`/`prism_mod_decay`,
  `prism_anchor_mode` (raw|spectral|shuffled) + `prism_anchor_refresh`, `prism_unfold`,
  `val2_dir` (dual-val), `stop_val_target` (matched-quality stop), forced-eval-at-resume.
- Modal runners (own isolated volume each): `prism_modal.py` / `prism_modal_finetune.py`
  / `prism_modal_arc.py`. Pattern: fork a branch → make `prism_modal_<name>.py` with its
  own `Volume.from_name("prism-eval-<name>")` → `modal run --detach`. Fetch:
  `modal volume get prism-eval-<name> nanogpt-prism/results/<file> ./results/` and COMMIT it.
- Ritual before any GPU: `cd src && python prism_selftest.py` (offline, ~1 min) then a
  tiny `--device=cpu` smoke of your driver → a conforming artifact. Then Modal.

## The experiments, ranked

### 1. Teacher-free PRISM init — THE priority (cheap, high-payoff)

**Hypothesis (from [project_prism_sigma_star] / the Hutter side-quest):** the spectrum
Σ\* is a *modality* constant, not corpus-specific. If so, you can PRISM-init a fresh
model from a fingerprint extracted from an **unrelated** corpus — no matched teacher —
and the ~12× head start survives. That turns PRISM from "needs a trained teacher" into
a **drop-in universal init** (~128 bytes).

**Run:** extract a fingerprint from a model trained on corpus A, then accelerate a model
on corpus B with it, vs a from-scratch baseline on B:
```bash
cd src
# 1) train an A-teacher and extract its fingerprint (or reuse .prism_cache/)
python prism_extract.py --ckpt out-A/ckpt.pt --out .prism_cache/A
# 2) accelerate on B with A's fingerprint; compare to a plain B run
python prism_accelerate.py --teacher_ckpt out-A/ckpt.pt --out_dir out-B-prism \
    -- --dataset=B --max_iters=1500
```
Better: add a `--cross_teacher` mode to `prism_eval.py` (teacher trained on A, student on
B, scored on B) so it emits a conforming artifact with the prism_score. **Win:** the
prism_score / Δloss on B with A's fingerprint ≈ with B's own teacher. The overlap-0 +
far-corpus results already hint it holds — this makes it the explicit claim. If it holds,
δ-as-OOD-detector, free-half-of-cross-size-transfer, and the continual anchor all follow.

### 2. Truly-far modality (where the story could break)

**Hypothesis:** the spectral transfer + the arc synergy hold within a modality but
degrade at a real modality boundary (English → **code** or another language). This is the
one place the plain recipe might finally need Leo's geometric levers (grassmann/topk/CKA,
already merged as opt-in flags).

**Gotcha to solve first:** the current vocab is Shakespeare's 65 chars; `far.txt`
(Sherlock) shares it. Real code / other languages use characters outside that vocab, so
`_encode_corpus` drops them and destroys the structure. You must either (a) pick a
far corpus over a compatible alphabet, or (b) rebuild the dataset + teacher + models on a
**new, larger vocab** that covers the far corpus (a bigger change — retrain everything).
Flagged as the first design decision. Then rerun `prism_eval.py --far_corpus` and
`prism_arc_eval.py` at that distance and find where Δloss / the arc synergy fall off.

### 3. The continuous single-run (vs sequential)

**Hypothesis (from the unified-arc plan):** one trajectory that pretrains then adapts with
a *morphing* target (teacher-spectrum early via init → self late via a frozen
`prism_unfold` target) matches the sequential pipeline at equal compute — likely *parity*
(the value is operational elegance), not new synergy. Worth one clean run to confirm
parity and rule out a hidden win. Build it as: `prism_init` + `prism_unfold=N` during
phase A, then freeze the target (`prism_unfold=0`) and switch dataset to B, one run.

### 4. Continual A → B → C (multi-domain accumulation)

**Hypothesis:** re-anchor to self after each domain → learn A, then B, then C, retaining
all prior domains. This is Result-2's anchor applied repeatedly = a continual-learning
method sequential single-finetune can't match. Needs a 3rd char corpus (see #2's vocab
gotcha). Metric: retention across ALL prior domains + speed of each new acquisition.

### 5. The "unfold curve" (the Σ\* / Hutter-adjacent test) — FIRST CUT ALREADY RUN

**Hypothesis:** most of a modality's compressibility is a tiny shared prior. Build
size-bounded priors of increasing size, fit on corpus A, and measure held-out
cross-entropy (bits/byte) on a *disjoint* corpus B (same modality); find the knee.

**A CPU first cut has been run** (Shakespeare prior → held-out Sherlock; n-gram priors
by order, size = bz2 bytes):

| prior | size | held-out bits/byte |
|---|---|---|
| order-0 | 367 B | 5.23 |
| order-1 | 3.9 KB | 4.36 |
| order-2 | 27 KB | 3.77 |
| **order-3** | **112 KB** | **3.50 (best)** |
| order-4 | 339 KB | 3.58 |
| order-5 | 893 KB | 3.95 |

**The finding — a knee that then *reverses*.** A tiny shared prior genuinely unfolds
never-seen same-modality text (5.23 → 3.50), but past ~112 KB **bigger priors get
*worse* on held-out text** — high-order Shakespeare n-grams are corpus-specific and
don't transfer. So there is an *optimal shared-prior size, and it's small*: ~⅔ of
English text's compressibility is a shared modality prior you can unfold from a tiny
dictionary; ~⅓ is irreducibly corpus-specific. That's the universal-structure /
specific-content split, shown classically (no PRISM). Harness: `unfold_curve.py`
(stdlib-only, ~130 lines; was written to the session scratchpad — rebuild or recover it).

**Where PRISM earns its slot (the next arm):** the classical prior tops out at 3.5
because n-grams are a *weak* universal model. Swap in a **stronger tiny shared prior** —
a small frozen neural LM, or Σ\* seeding an online nanoGPT — and see if it pushes the
knee *down and left* (more compressibility captured as "shared"). **The decisive test:
does 128 B of Σ\* spectral geometry beat 128 B of n-gram at the tiny end?** If yes, PRISM
earns a slot in compression *and* it's a direct "Σ\* is a modality constant" test. Then
the enwik8-scale version. Not a Hutter entry. Trap to avoid: the lossy keypad-digit
"literal T9" ≤ direct coding (data-processing inequality) — demo only. Full spec in
`project_prism_sigma_star`.

### 6. The reach-at-init moonshot (from the 30× hybrid to 1000×)

[Prior-Fused PRISM](PRIOR-FUSED-PRISM.md) already showed the hybrid at **30×**. The
literal **1000×** — the fused model at baseline quality *before any training* — needs the
shared prior to sit *clearly below* the baseline's loss so the fused init already beats it.
Context-3 tops out *at* baseline (2.57 ≈ 2.565 bits/char), a knife-edge, and the `logit_gate`
that would enable reach-at-init backfired (throttled early learning, 30× → 15×). **The one
missing piece: a context-4+ n-gram prior**, which is clearly below baseline but needs a
**sparse build** — the dense V⁴ table (65⁴×65) is too big and `build_ngram_prior.py` times
out on it. Build a sparse (hashed-context) n-gram + a sparse gather in `train.py`'s
`_prior_logp`, confirm the prior dips below baseline on val, then re-run
`prism_prior_eval.py` with the gate *off* (it hurt) and read the reach-at-init speedup on
the `prior` / `prism_prior` arms. Also fix the block-edge inflation (first C-1 positions
per window lack in-window context → currently uniform; back off to a lower order instead).

## Ground rules (again, because they matter)

A number in these docs has a matching committed `results/*.json`. Attribution needs
matched schedules (only then does "only X differs"). Probe wide/shallow first (warmup ≪
steps). 3 seeds for a result; a single seed is a probe. Partial/crashed runs must raise,
never be scored. When you launch a Modal run, watch it, fetch the artifact, and **commit
it** — that's the evidence. And own your errors plainly; this project killed its own
"spectral finetune" hypothesis with a placebo, and overturned an author's prior on the
arc. Truth over hype.
