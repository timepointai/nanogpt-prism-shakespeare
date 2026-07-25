"""
unfold_curve.py — TTT9 <> Hutter, exploratory first cut (CPU-only, stdlib only).

EXPLORATORY, not a validated benchmark. This is the harness for next-experiment #5 in
docs/NEXT-EXPERIMENTS.md — the Σ*/"unfold curve" idea from the Hutter side-quest. It is
NOT part of the curated benchmark suite (prism_eval / prism_finetune_eval / prism_arc_eval)
and its numbers are a rough first cut, not committed results.

TTT9 packs a *shared* language prior into 838 bz2 bytes and "unfolds" new input through
it. This measures the general version for full text: build a size-bounded shared prior
from a REFERENCE corpus R, then compress a DISJOINT held-out corpus H under it, and plot
prior-size (k, bytes) vs held-out rate (bits/byte). The knee = how small a shared prior
can be while still unfolding never-seen text at a good rate.

Prior = byte-level backoff n-gram (PPM-lite), orders 0..N, contexts pruned by min count.
Prior SIZE k = bz2 of the serialized model (TTT9-faithful: it bz2's its dictionary).
Held-out rate = cross-entropy of H under the prior = ideal lossless bits/byte (no coder
needed to measure). Backoff to lower orders, then uniform 1/256, so it is never infinite.

R = Shakespeare, H = Sherlock (far.txt): disjoint, same modality (English prose). NOT a
Hutter entry — it's the amortized-shared-prior question Hutter's single number hides.

First cut found a knee THAT REVERSES: rate improves 5.23 → 3.50 bits/byte as k grows to
~112 KB (order-3), then gets WORSE (high-order Shakespeare n-grams are corpus-specific
and don't transfer) — an optimal, small shared-prior size. The next arm (docs/NEXT-
EXPERIMENTS.md #5) swaps the weak n-gram prior for a stronger tiny one — a small frozen
neural LM, or 128 bytes of PRISM Σ* geometry — to test whether it pushes the knee down
and left. Requires the shakespeare_char dataset prepared (data/shakespeare_char/prepare.py).
"""
import bz2
import math
import os
import struct
import time
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
R_PATH = os.path.join(_HERE, 'data', 'shakespeare_char', 'input.txt')   # reference corpus
H_PATH = os.path.join(_HERE, 'data', 'far.txt')                         # disjoint held-out
ALPHA = 0.05          # add-alpha smoothing within a context's kept alphabet
VOCAB = 256


def build_counts(data, max_order):
    """orders[o][context_bytes] = {next_byte: count}, for o in 0..max_order."""
    orders = [defaultdict(lambda: defaultdict(int)) for _ in range(max_order + 1)]
    n = len(data)
    for i in range(n):
        b = data[i]
        for o in range(0, max_order + 1):
            if i - o < 0:
                break
            ctx = data[i - o:i]           # bytes of length o
            orders[o][ctx][b] += 1
    return orders


def prune(orders, min_count):
    """Keep only contexts whose total count >= min_count (order 0 always kept).
    Returns pruned {o: {ctx: {byte: count}}} as plain dicts."""
    kept = []
    for o, table in enumerate(orders):
        d = {}
        for ctx, dist in table.items():
            tot = sum(dist.values())
            if o == 0 or tot >= min_count:
                d[ctx] = dict(dist)
        kept.append(d)
    return kept


def serialize(kept):
    """Compact binary pack of the model, then bz2 — the shared 'dictionary blob'.
    Per order: [order][num_contexts]; per context: [ctxlen][ctx bytes][num_syms];
    per sym: [byte][varint count]. bz2 of the whole thing = prior size k."""
    out = bytearray()

    def put_varint(x):
        while True:
            b = x & 0x7F
            x >>= 7
            out.append(b | (0x80 if x else 0))
            if not x:
                break

    out += struct.pack('B', len(kept))                     # number of orders
    for o, table in enumerate(kept):
        out += struct.pack('B', o)
        put_varint(len(table))
        for ctx, dist in table.items():
            put_varint(len(ctx))
            out += ctx
            put_varint(len(dist))
            for byte, cnt in dist.items():
                out.append(byte)
                put_varint(cnt)
    return len(bz2.compress(bytes(out), 9))


def score(kept, H, max_order):
    """Cross-entropy bits/byte of H under the backoff model. Highest available kept
    order wins; add-alpha within its kept alphabet; else back off; else uniform."""
    bits = 0.0
    n = len(H)
    for i in range(n):
        b = H[i]
        p = None
        for o in range(min(max_order, i), -1, -1):
            ctx = H[i - o:i]
            dist = kept[o].get(ctx)
            if dist:
                tot = sum(dist.values())
                c = dist.get(b, 0)
                p = (c + ALPHA) / (tot + ALPHA * VOCAB)
                break
        if p is None:
            p = 1.0 / VOCAB
        bits += -math.log2(p)
    return bits / n


def main():
    R = open(R_PATH, 'rb').read()
    H = open(H_PATH, 'rb').read()
    print(f'R (prior corpus) = {len(R):,} bytes Shakespeare')
    print(f'H (held-out)     = {len(H):,} bytes Sherlock')
    print(f'raw byte size of H = {len(H)} B; uniform baseline = 8.000 bits/byte\n')

    MAXO = 5
    t0 = time.time()
    print('building counts (orders 0..%d) on R…' % MAXO, flush=True)
    orders = build_counts(R, MAXO)
    print(f'  done ({time.time()-t0:.1f}s)\n')

    # sweep (max_order N, min_count t) → a family of size-bounded shared priors
    grid = [(0, 1), (1, 1), (2, 8), (2, 2), (3, 8), (3, 2), (4, 4), (4, 2), (5, 3), (5, 2)]
    print(f'{"N":>2} {"mincnt":>6} | {"prior k (bz2 B)":>15} | {"held-out b/byte":>15} | '
          f'{"unfold |H|/k":>12}')
    print('-' * 64)
    rows = []
    for (N, t) in grid:
        kept = prune(orders[:N + 1], t)
        k = serialize(kept)
        bpb = score(kept, H, N)
        rows.append((N, t, k, bpb))
        print(f'{N:>2} {t:>6} | {k:>15,} | {bpb:>15.4f} | {len(H)/k:>12,.0f}', flush=True)

    print('-' * 64)
    best = min(rows, key=lambda r: r[3])
    print(f'\nbest held-out rate: {best[3]:.4f} bits/byte at prior k={best[2]:,} B '
          f'(order {best[0]}, mincnt {best[1]})')
    thresh = best[3] * 1.10
    knee = min((r for r in rows if r[3] <= thresh), key=lambda r: r[2])
    print(f'knee (within 10% of best): k={knee[2]:,} B → {knee[3]:.4f} bits/byte, '
          f'unfold ratio |H|/k = {len(H)/knee[2]:,.0f}×')
    print('NOTE: the curve reverses past the knee — bigger priors overfit R and transfer '
          'worse to H. Next arm: replace the n-gram prior with a stronger tiny prior '
          '(small neural LM / PRISM Σ* 128B) — see docs/NEXT-EXPERIMENTS.md #5.')


if __name__ == '__main__':
    main()
