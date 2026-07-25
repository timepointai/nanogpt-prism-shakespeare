"""
build_ngram_prior.py — build a fixed shared n-gram prior (the "T9 dictionary") over the
char vocab, as a dense (V^C, V) log-probability table that train.py fuses into the model's
logits: final = model_logits + strength · log p_ngram(next | last C chars).

The model then learns only the RESIDUAL over this prior. The prior is a linear
interpolation of orders 0..C (add-0 MLE per order, renormalized) — smooth, always a valid
distribution, and cheap to gather on GPU by context index.

    python build_ngram_prior.py --dataset shakespeare_char --context 3 --out .prism_cache/ngram

Also reports the prior's standalone cross-entropy on train and val (bits/char) — the
head start the fused model gets for free.
"""
import argparse
import os
import pickle
import numpy as np
import torch


def load_bin(path):
    return np.array(np.memmap(path, dtype=np.uint16, mode='r')).astype(np.int64)


def context_index(data, order, V):
    """Vectorized context index for every position >= order. Returns (idx, next) arrays of
    length n-order, where idx[m] encodes data[m..m+order-1] and next[m] = data[m+order]."""
    n = len(data)
    if order == 0:
        return np.zeros(n, dtype=np.int64), data
    idx = np.zeros(n - order, dtype=np.int64)
    for p in range(order):
        idx = idx * V + data[p:n - order + p]
    return idx, data[order:]


def order_table(data, order, V):
    """Dense (V^order, V) MLE next-char distribution; also a bool mask of observed rows."""
    ctx_space = V ** order
    counts = np.zeros((ctx_space, V), dtype=np.float64)
    idx, nxt = context_index(data, order, V)
    np.add.at(counts, (idx, nxt), 1.0)
    row = counts.sum(axis=1, keepdims=True)
    P = np.where(row > 0, counts / np.maximum(row, 1.0), 0.0)
    return P, (row > 0).reshape(-1)


def build(data, context, V, lambdas):
    """Interpolated dense (V^context, V) prob table, renormalized per row. Row idx encodes
    the `context` chars; lower orders index into it via idx % V^o (the most-recent o chars)."""
    Cspace = V ** context
    table = np.zeros((Cspace, V), dtype=np.float64)
    weight = np.zeros(Cspace, dtype=np.float64)
    ctx = np.arange(Cspace)
    for o in range(context + 1):
        lam = lambdas[o]
        P, obs = order_table(data, o, V)
        if o == 0:
            table += lam * P[0]
            weight += lam
        else:
            sub = ctx % (V ** o)                          # last o chars
            table += lam * P[sub]
            weight += lam * obs[sub]                       # weight only where observed
    table /= np.maximum(weight[:, None], 1e-12)            # renormalize the mixture
    return table


def xent_bits(table, data, context, V):
    """Cross-entropy (bits/char) of `data` under the table (skips first `context` chars)."""
    idx, nxt = context_index(data, context, V)
    p = np.clip(table[idx, nxt], 1e-12, 1.0)
    return float(-np.log2(p).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='shakespeare_char')
    ap.add_argument('--context', type=int, default=3)
    ap.add_argument('--out', default='.prism_cache/ngram')
    ap.add_argument('--lambdas', default='',
                    help='comma weights for orders 0..context (default: favor high order)')
    a = ap.parse_args()

    d = f'data/{a.dataset}'
    V = pickle.load(open(f'{d}/meta.pkl', 'rb'))['vocab_size']
    train = load_bin(f'{d}/train.bin')
    val = load_bin(f'{d}/val.bin')
    C = a.context
    if a.lambdas:
        lam = [float(x) for x in a.lambdas.split(',')]
    else:
        base = {0: [1.0], 1: [0.1, 0.9], 2: [0.05, 0.2, 0.75],
                3: [0.03, 0.1, 0.22, 0.65], 4: [0.02, 0.06, 0.12, 0.25, 0.55]}
        lam = base.get(C, [1.0 / (C + 1)] * (C + 1))
    print(f'vocab {V} · context {C} · lambdas {lam} · train {len(train):,} val {len(val):,}')

    print('building interpolated table…', flush=True)
    table = build(train, C, V, lam)
    tb = xent_bits(table, train, C, V)
    vb = xent_bits(table, val, C, V)
    print(f'  n-gram prior standalone:  train {tb:.4f} bits/char · VAL {vb:.4f} bits/char')
    print('  (nanoGPT char baseline best ≈ 2.5 bits/char / ~1.78 nats; a lower prior = a '
          'bigger free head start for the fused model)')

    os.makedirs(a.out, exist_ok=True)
    logtab = torch.tensor(np.log(np.clip(table, 1e-12, 1.0)), dtype=torch.float32)
    torch.save({'table': logtab, 'context_len': C, 'vocab_size': V,
                'val_bits_per_char': vb, 'train_bits_per_char': tb, 'lambdas': lam},
               f'{a.out}/prior_c{C}.pt')
    print(f'  saved {a.out}/prior_c{C}.pt  ({logtab.numel() * 4 / 1e6:.0f} MB, '
          f'shape {tuple(logtab.shape)})')


if __name__ == '__main__':
    main()
