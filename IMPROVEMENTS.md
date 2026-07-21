# Prism — transfer improvements (experiment menu)

New, opt-in levers on the `improvement` branch. Every default is unchanged: with
no new flags, `prism_eval.py` reproduces the committed recipe byte-for-byte. Each
lever below is a runnable experiment through the standard harness, so it produces
a `results/*.json` artifact — no number enters the docs without one.

The levers split by which mechanism they touch and, therefore, which number they
are meant to move. The distinction matters: overfit-suppression on shared data is
something a generic regularizer or prune also buys, so a change that only improves
the full-overlap number is not differentiated from "any regularization helps." The
signal to watch is the **zero-overlap / large-token-JS advantage** — the part that
a generic method cannot produce because it has no teacher to point at.

Validate the new math first (seconds, no training):

```bash
cd src && python prism_selftest.py
```

## Direction transfer (the differentiated headroom)

These act on EigenTransfer — the singular vectors, where the teacher's learned
features live. The spectrum is already near-optimally compressed; the directions
are not.

- **Grassmann geodesic** — pair singular directions by geometry, not by index,
  then rotate each pair through its own principal angle.
  `--align_mode=grassmann`
- **Top-k structural transfer** — transfer only the leading k directions verbatim,
  leave the tail fresh.
  `--align_topk=32`
- **Adaptive (subspace) strength** — set the per-matrix blend from how far the
  student already is from the teacher (principal-angle distance); transfer more
  into unaligned subspaces.
  `--align_mode=subspace`
- **Per-group / depth-tapered blend** — one strength per weight group, and/or less
  teacher in later (more data-specific) layers.
  `--align_spec=attention:0.9,ffn_down:0.5`  ·  `--align_depth_gamma=0.5`

## Spectral imprint

- **Truer spectrum** — more DCT coefficients (lower reconstruction error).
  `--n_dct=16`
- **Per-layer spectra** — imprint each matrix's own spectrum instead of the group
  average.
  `--per_layer`

## Mod Wheel (regularizer — least differentiated)

- **Scheduled strength** — attack→sustain instead of a single decay.
  `--mod=0.02 --mod_decay=0.999 --mod_transition=200 --mod_sustain=0.005`

## Representational transfer (CKA)

- **Match the teacher's representation geometry**, not its weights — add a
  `(1 − linear-CKA)` distance between student and teacher block activations to the
  loss. Differentiated by construction: the signal is the teacher's own
  activations. Experimental.
  `--cka=0.1 --cka_layers=2,4`

## Distance-based evaluation (how to prove structure vs. content)

Overlap-fraction is a proxy; token-JS is the real axis, and `overlap 0.0` is still
same-corpus (small token-JS), so it cannot separate structure from content on its
own.

- **Far-corpus arm** — the student's non-shared blocks come from a different corpus
  (char-encoded with Shakespeare's vocab), so token-JS is large. This is the real
  structure-vs-content test.
  `--overlap=0.0 --far_corpus=path/to/other.txt`

The sweep report already prints the `tok-JS` column; read the recipe advantage
against it. Advantage that persists as token-JS grows is structural; advantage that
vanishes was content.

## Examples

```bash
cd src

# Grassmann + top-k, matched LR, 3 seeds (its own artifact via the run key)
python prism_eval.py --method_lr=1e-3 --method_warmup=100 \
    --align_mode=grassmann --align_topk=32

# Truer, per-layer spectrum
python prism_eval.py --method_lr=1e-3 --method_warmup=100 --n_dct=16 --per_layer

# The decisive test: does the advantage survive genuinely far data?
python prism_eval.py --method_lr=1e-3 --method_warmup=100 \
    --overlap=0.0 --far_corpus=data/far.txt
```
