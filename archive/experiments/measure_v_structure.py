"""
measure_v_structure.py — Measure the structure in inter-layer V matrices.

Track 2 research: is the directional information (500MB) compressible?

Measures:
1. Pairwise cosine similarity between V matrices of adjacent layers
2. Principal components of all V matrices stacked — what dimension do they live in?
3. Inter-layer delta norms — how much does V change from layer to layer?
4. Delta encoding savings — if we store deltas instead of raw V, how much smaller?

Run: python measure_v_structure.py --ckpt out-teacher/ckpt.pt
  or: python measure_v_structure.py --cache .prism_cache/shakespeare
"""
import argparse
import os
import json
import numpy as np
import torch


def load_directions(path):
    """Load cached directions from prism_extract output."""
    return torch.load(os.path.join(path, 'directions.pt'),
                      map_location='cpu', weights_only=False)


def measure(directions):
    """Run all structure measurements on the V matrices."""

    # Group by weight type and sort by layer
    import re
    groups = {}  # group_name -> [(layer_idx, name, Vt)]
    for name, ext in directions.items():
        group = ext['group']
        m = re.search(r'h\.(\d+)\.', name)
        layer = int(m.group(1)) if m else -1
        Vt = ext['Vt'].float()
        groups.setdefault(group, []).append((layer, name, Vt))

    for g in groups:
        groups[g].sort(key=lambda x: x[0])

    results = {}

    for group, layers in groups.items():
        if len(layers) < 2:
            continue

        print(f'\n{"="*60}')
        print(f'  {group}: {len(layers)} layers')
        print(f'{"="*60}')

        # ── 1. Adjacent layer cosine similarity ──
        adj_sims = []
        for i in range(len(layers) - 1):
            V_a = layers[i][2]   # (k, n)
            V_b = layers[i+1][2]
            # Compare corresponding singular vectors
            if V_a.shape == V_b.shape:
                # Cosine sim of each pair of singular vectors, then average
                sims = torch.nn.functional.cosine_similarity(V_a, V_b, dim=1)
                avg_sim = sims.abs().mean().item()  # abs because sign is arbitrary
                adj_sims.append(avg_sim)
                print(f'  L{layers[i][0]}→L{layers[i+1][0]}: '
                      f'mean |cos_sim| = {avg_sim:.4f}  '
                      f'(top-1: {sims[0].abs().item():.4f}, '
                      f'top-5 avg: {sims[:5].abs().mean().item():.4f})')

        if adj_sims:
            mean_sim = np.mean(adj_sims)
            print(f'\n  Average adjacent similarity: {mean_sim:.4f}')
            print(f'  Interpretation: {"HIGH redundancy — delta encoding will help" if mean_sim > 0.5 else "MODERATE redundancy" if mean_sim > 0.3 else "LOW redundancy — layers are independent"}')
            results[f'{group}_adj_sim'] = mean_sim

        # ── 2. Delta norms (Frobenius) ──
        raw_norms = []
        delta_norms = []
        for i in range(len(layers)):
            raw_norms.append(layers[i][2].norm().item())
            if i > 0 and layers[i][2].shape == layers[i-1][2].shape:
                delta = layers[i][2] - layers[i-1][2]
                delta_norms.append(delta.norm().item())

        if delta_norms:
            raw_mean = np.mean(raw_norms)
            delta_mean = np.mean(delta_norms)
            ratio = delta_mean / raw_mean
            print(f'\n  Raw V norm (avg): {raw_mean:.2f}')
            print(f'  Delta norm (avg): {delta_mean:.2f}')
            print(f'  Delta/Raw ratio:  {ratio:.4f}')
            print(f'  Interpretation: {"EXCELLENT — deltas are {:.0f}% of raw".format(ratio*100) if ratio < 0.5 else "MODERATE — deltas are {:.0f}% of raw".format(ratio*100) if ratio < 0.9 else "POOR — deltas are nearly as large as raw"}')
            results[f'{group}_delta_ratio'] = ratio

        # ── 3. PCA of stacked V matrices ──
        if all(layers[i][2].shape == layers[0][2].shape for i in range(len(layers))):
            stacked = torch.stack([l[2].reshape(-1) for l in layers])  # (n_layers, flattened)
            # Center
            stacked_centered = stacked - stacked.mean(dim=0)
            # SVD for PCA
            U, S, Vt_pca = torch.linalg.svd(stacked_centered, full_matrices=False)
            total_var = (S ** 2).sum().item()
            cum_var = torch.cumsum(S ** 2, dim=0) / total_var

            dims_90 = (cum_var >= 0.90).nonzero(as_tuple=True)[0][0].item() + 1
            dims_95 = (cum_var >= 0.95).nonzero(as_tuple=True)[0][0].item() + 1
            dims_99 = (cum_var >= 0.99).nonzero(as_tuple=True)[0][0].item() + 1

            print(f'\n  PCA of {len(layers)} stacked V matrices:')
            print(f'    Dims for 90% variance: {dims_90} / {len(layers)}')
            print(f'    Dims for 95% variance: {dims_95} / {len(layers)}')
            print(f'    Dims for 99% variance: {dims_99} / {len(layers)}')
            print(f'    Top singular values: {S[:5].tolist()}')
            results[f'{group}_pca_90'] = dims_90
            results[f'{group}_pca_95'] = dims_95
            results[f'{group}_pca_99'] = dims_99

        # ── 4. Transformation matrix consistency ──
        # T maps V[i] → V[i+1]. If T is consistent, one matrix generates all layers.
        if len(layers) >= 3 and all(layers[i][2].shape == layers[0][2].shape for i in range(len(layers))):
            transforms = []
            for i in range(len(layers) - 1):
                V_a = layers[i][2]
                V_b = layers[i+1][2]
                # Least squares: T @ V_a ≈ V_b → T ≈ V_b @ pinv(V_a)
                T = V_b @ torch.linalg.pinv(V_a)
                transforms.append(T)

            # How consistent are the T matrices?
            T_sims = []
            for i in range(len(transforms) - 1):
                sim = torch.nn.functional.cosine_similarity(
                    transforms[i].reshape(1, -1),
                    transforms[i+1].reshape(1, -1)
                ).item()
                T_sims.append(sim)

            mean_T_sim = np.mean(T_sims)
            print(f'\n  Transformation matrix consistency:')
            print(f'    Mean T[i]↔T[i+1] cosine sim: {mean_T_sim:.4f}')
            print(f'    Interpretation: {"CONSISTENT — one T generates all layers!" if mean_T_sim > 0.8 else "MODERATE — T varies but has structure" if mean_T_sim > 0.4 else "INCONSISTENT — each layer transition is unique"}')
            results[f'{group}_T_consistency'] = mean_T_sim

    # ── Summary ──
    print(f'\n{"="*60}')
    print(f'  SUMMARY')
    print(f'{"="*60}')
    for k, v in sorted(results.items()):
        print(f'  {k}: {v:.4f}')

    # Verdict
    adj_sims_all = [v for k, v in results.items() if 'adj_sim' in k]
    delta_ratios = [v for k, v in results.items() if 'delta_ratio' in k]
    t_cons = [v for k, v in results.items() if 'T_consistency' in k]

    print(f'\n  VERDICT:')
    if adj_sims_all:
        avg = np.mean(adj_sims_all)
        if avg > 0.5:
            print(f'  ✓ High inter-layer similarity ({avg:.2f}) — delta encoding is viable')
        else:
            print(f'  ✗ Low inter-layer similarity ({avg:.2f}) — delta encoding won\'t help much')

    if delta_ratios:
        avg = np.mean(delta_ratios)
        savings = (1 - avg) * 100
        print(f'  {"✓" if avg < 0.7 else "✗"} Delta encoding saves ~{savings:.0f}% of storage')

    if t_cons:
        avg = np.mean(t_cons)
        if avg > 0.7:
            print(f'  ✓ Transformation is consistent ({avg:.2f}) — unfold program is possible!')
        elif avg > 0.4:
            print(f'  ~ Transformation has moderate structure ({avg:.2f}) — partial unfold possible')
        else:
            print(f'  ✗ Transformations are inconsistent ({avg:.2f}) — no simple generator')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=str, default='.prism_cache/shakespeare',
                        help='Path to cached directions from prism_extract')
    args = parser.parse_args()

    directions = load_directions(args.cache)
    print(f'Loaded {len(directions)} weight matrices')
    results = measure(directions)

    out_path = os.path.join(args.cache, 'v_structure.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_path}')
