"""
prism_init.py — Prism spectral initialization for nanoGPT.

Extracts SVD spectra from pretrained GPT-2 (HuggingFace), compresses to
DCT coefficients per weight group, and applies Spectral Imprint + EigenTransfer
to a fresh nanoGPT model.

Usage (in train.py, after model creation, before torch.compile):

    from prism_init import apply_prism
    apply_prism(model)

Best config from CUDA sweep (April 2026):
    UV alignment 0.75, LR 1.5x, spike_skip 50, warmup 300

The pretrained spectra are extracted once and cached to disk.
"""
import os
import re
import json
import math
import gc
import numpy as np
import torch
import torch.nn as nn

# ── Spectral math ──

def dct_expand(coeffs, n):
    """Expand DCT coefficients to n spectral values."""
    k = len(coeffs)
    t = np.linspace(0, np.pi, n, endpoint=False)
    spectrum = np.zeros(n)
    for i in range(k):
        spectrum += coeffs[i] * np.cos((i + 0.5) * t)
    spectrum = np.log(1.0 + np.exp(np.clip(spectrum, -10, 10)))
    s_max = spectrum.max()
    if s_max > 0:
        spectrum = spectrum / s_max
    return spectrum


def spectral_target(W, sv0):
    """Impose the singular-value spectrum sv0 on W's CURRENT directions: U diag(sv0) Vt.

    The 'spectral' finetune anchor (train.py, resume path) rebuilds its mod-wheel
    target this way from the live weight each refresh: it holds the weight's spectral
    shape near sv0 while leaving U/V free to adapt to new data — so the mod wheel
    constrains the spectrum (the PRISM geometry) rather than pinning the whole weight
    (which is what the 'raw' L2-to-init anchor does). Column space is preserved."""
    U, s, Vt = torch.linalg.svd(W.float(), full_matrices=False)
    k = min(sv0.shape[0], s.shape[0])
    s_new = s.clone()
    s_new[:k] = sv0[:k].to(s.device, s.dtype)
    return ((U * s_new) @ Vt).to(W.dtype)


def blend_orthogonal(A, B, alpha):
    """Blend two orthogonal matrices, re-orthogonalized via SVD.

    Pairs columns by INDEX — the original EigenTransfer operation. Kept as the
    default so existing runs reproduce exactly."""
    blended = (1 - alpha) * A + alpha * B
    try:
        U, _, Vt = torch.linalg.svd(blended, full_matrices=False)
        return U @ Vt
    except torch._C._LinAlgError:
        return A


# ── Subspace-aware direction transfer (opt-in) ──
#
# The flat blend above pairs the student's i-th singular vector with the teacher's
# i-th and mixes them linearly. Two problems it doesn't address, and the tools for
# each:
#   • directions are paired by index, not by geometric proximity   → grassmann_interp
#   • one global strength for every layer/matrix                    → subspace_alignment
# All of this is gated behind non-default apply_prism args; the recipe path is
# untouched.

def _principal(A, B):
    """Principal cosines between the column spaces of two column-orthonormal
    matrices. Returns (P, cos, Q): A@P and B@Q are the paired principal
    directions, cos = cos(principal angles), descending."""
    M = A.transpose(-2, -1) @ B
    P, c, Qt = torch.linalg.svd(M, full_matrices=False)
    return P, torch.clamp(c, -1.0, 1.0), Qt.transpose(-2, -1)


def subspace_alignment(A, B, k=0, rows=False):
    """Mean cosine of the principal angles between the top-k subspaces of A and B
    (1.0 = identical subspace, 0.0 = orthogonal). A geometric similarity, used to
    set the transfer strength adaptively: transfer MORE where the student is far
    from the teacher, less where it already agrees."""
    if rows:
        A, B = A.transpose(-2, -1), B.transpose(-2, -1)
    if k and 0 < k < A.shape[1]:
        A, B = A[:, :k], B[:, :k]
    _, c, _ = _principal(A, B)
    return float(c.mean())


def grassmann_interp(A, B, alpha, rows=False):
    """Geodesic interpolation from orthonormal A toward B, fraction alpha∈[0,1],
    along the Grassmann/Stiefel manifold. Pairs principal directions FIRST, then
    rotates each pair through its own principal angle — so directions are matched
    by geometry, not by index. alpha=0 → A's subspace, alpha=1 → B's subspace.
    alpha may be a scalar or a per-direction vector (for top-k transfer)."""
    if rows:
        return grassmann_interp(A.transpose(-2, -1), B.transpose(-2, -1),
                                alpha, rows=False).transpose(-2, -1)
    try:
        P, cos, Q = _principal(A, B)
    except torch._C._LinAlgError:
        return A
    A_p = A @ P
    B_p = B @ Q
    theta = torch.arccos(cos)
    perp = B_p - A_p * cos
    perp = perp / torch.clamp(torch.linalg.norm(perp, dim=0, keepdim=True), min=1e-8)
    if not torch.is_tensor(alpha):
        alpha = torch.as_tensor(float(alpha), device=A.device, dtype=A.dtype)
    a = alpha.to(A.dtype)
    X = A_p * torch.cos(a * theta) + perp * torch.sin(a * theta)
    U, _, Vt = torch.linalg.svd(X, full_matrices=False)
    return U @ Vt


def align_directions(fresh, pre, alpha, mode='linear', topk=0, rows=False):
    """Transfer teacher directions into a fresh orthonormal frame.

    mode='linear'    — index-paired blend + re-orth (blend_orthogonal; the default)
    mode='grassmann' — principal-angle geodesic (directions matched by geometry)
    topk>0           — only the leading k singular directions are moved; the tail
                       stays fresh (the top-k structural transfer). The whole
                       frame is re-orthonormalized so the kept tail stays valid.
    rows=True        — operate on the row space (for Vt)."""
    if rows:
        return align_directions(fresh.transpose(-2, -1), pre.transpose(-2, -1),
                                alpha, mode, topk, rows=False).transpose(-2, -1)

    def core(f, p):
        if mode == 'grassmann':
            return grassmann_interp(f, p, alpha)
        return blend_orthogonal(f, p, float(alpha))

    r = fresh.shape[1]
    if topk and 0 < topk < r:
        head = core(fresh[:, :topk], pre[:, :topk])
        combined = torch.cat([head, fresh[:, topk:]], dim=1)
        try:
            U, _, Vt = torch.linalg.svd(combined, full_matrices=False)
            return U @ Vt
        except torch._C._LinAlgError:
            return fresh
    return core(fresh, pre)


def _layer_depth_frac(name, n_layer):
    """Fractional depth in [0,1] of a parameter from its 'h.<i>.' index, or None
    for non-block params (embeddings). Used to taper transfer with depth."""
    m = re.search(r'\bh\.(\d+)\.', name)
    if m is None or n_layer <= 1:
        return None
    return int(m.group(1)) / (n_layer - 1)


# ── Weight group classification (nanoGPT naming) ──

GROUPS = {
    'attention': [],   # c_attn (fused QKV)
    'attn_proj': [],   # attn.c_proj (residual-scaled)
    'ffn_up': [],      # mlp.c_fc
    'ffn_down': [],    # mlp.c_proj (residual-scaled)
    'embedding': [],   # wte, wpe
}


def classify_nanogpt_param(name):
    """Classify a nanoGPT parameter name into a spectrum group."""
    if 'ln' in name or 'bias' in name:
        return None
    if 'c_attn' in name:
        return 'attention'
    if 'attn' in name and 'c_proj' in name:
        return 'attn_proj'
    if 'c_fc' in name:
        return 'ffn_up'
    if 'mlp' in name and 'c_proj' in name:
        return 'ffn_down'
    if 'wte' in name or 'wpe' in name:
        return 'embedding'
    return None


# HuggingFace GPT-2 uses the same group structure but Conv1D (transposed)
def classify_hf_param(name):
    """Classify a HuggingFace GPT-2 parameter name."""
    if 'ln' in name or 'bias' in name or 'layernorm' in name:
        return None
    if 'c_attn' in name:
        return 'attention'
    if 'attn' in name and 'c_proj' in name:
        return 'attn_proj'
    if 'c_fc' in name:
        return 'ffn_up'
    if 'mlp' in name and 'c_proj' in name:
        return 'ffn_down'
    if 'wte' in name or 'wpe' in name:
        return 'embedding'
    return None


# ── Extraction ──

CACHE_DIR = os.path.join(os.path.dirname(__file__), '.prism_cache')
SPECTRA_CACHE = os.path.join(CACHE_DIR, 'spectra.json')
DIRS_CACHE = os.path.join(CACHE_DIR, 'directions.pt')


def extract_spectra(n_dct=8, force=False):
    """Extract group-averaged DCT spectra from pretrained GPT-2.

    Returns dict of group_name -> list of DCT coefficients.
    Cached to .prism_cache/spectra.json.
    """
    if os.path.exists(SPECTRA_CACHE) and not force:
        with open(SPECTRA_CACHE) as f:
            return json.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    print('[prism] Extracting spectra from pretrained GPT-2...')

    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    group_svs = {g: [] for g in GROUPS}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
            group = classify_hf_param(name)
            if group is None:
                continue
            # HF Conv1D stores weights as (in, out) — transpose for SVD
            W = param.data.float()
            if W.shape[0] < W.shape[1] and 'wte' not in name and 'wpe' not in name:
                W = W.T  # Conv1D → Linear convention
            if W.dim() > 2:
                W = W.reshape(W.shape[0], -1)
            s = torch.linalg.svdvals(W)
            s_norm = (s / s.max()).cpu().numpy()
            group_svs[group].append(s_norm)

    del model
    gc.collect()

    spectra = {}
    for group, sv_list in group_svs.items():
        if not sv_list:
            continue
        max_len = max(len(s) for s in sv_list)
        interp = [np.interp(np.linspace(0, 1, max_len),
                            np.linspace(0, 1, len(s)), s) for s in sv_list]
        avg = np.mean(interp, axis=0)
        clipped = np.clip(avg, 0.01, None)
        target = np.log(np.exp(clipped) - 1.0 + 1e-10)
        n = len(avg)
        t = np.linspace(0, np.pi, n, endpoint=False)
        basis = np.zeros((n, n_dct))
        for i in range(n_dct):
            basis[:, i] = np.cos((i + 0.5) * t)
        coeffs, _, _, _ = np.linalg.lstsq(basis, target, rcond=None)
        spectra[group] = coeffs.tolist()
        print(f'  {group}: {len(sv_list)} matrices, {n_dct} DCT coeffs')

    with open(SPECTRA_CACHE, 'w') as f:
        json.dump(spectra, f, indent=2)
    print(f'[prism] Spectra cached to {SPECTRA_CACHE}')
    return spectra


def extract_directions(force=False):
    """Extract per-layer U and Vt from pretrained GPT-2.

    Returns dict of param_name -> {U, Vt, group, shape}.
    Cached to .prism_cache/directions.pt (large: ~500MB).
    """
    if os.path.exists(DIRS_CACHE) and not force:
        return torch.load(DIRS_CACHE, map_location='cpu', weights_only=False)

    os.makedirs(CACHE_DIR, exist_ok=True)
    print('[prism] Extracting directions from pretrained GPT-2...')

    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    directions = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
            group = classify_hf_param(name)
            if group is None:
                continue
            W = param.data.float()
            if W.shape[0] < W.shape[1] and 'wte' not in name and 'wpe' not in name:
                W = W.T
            if W.dim() > 2:
                W = W.reshape(W.shape[0], -1)
            U, s, Vt = torch.linalg.svd(W, full_matrices=False)
            # Map HF name to nanoGPT name (they're the same except lm_head)
            nano_name = name.replace('transformer.', '')
            directions[nano_name] = {
                'U': U.cpu(), 'Vt': Vt.cpu(),
                'group': group, 'shape': list(W.shape),
            }

    del model
    gc.collect()
    torch.save(directions, DIRS_CACHE)
    print(f'[prism] Directions cached to {DIRS_CACHE}')
    return directions


# ── Application ──

def _parse_align_spec(spec):
    """Parse a per-group alignment spec 'attention:0.9,ffn_down:0.5' into a dict.
    Accepts a dict unchanged, or None. Groups not named fall back to the scalar
    align_strength."""
    if spec is None or isinstance(spec, dict):
        return spec
    out = {}
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        g, v = part.split(':')
        out[g.strip()] = float(v)
    return out


def apply_prism(model, align_strength=0.75, lam=1.0,
                spectra_path=None, directions_path=None, verbose=True,
                align_spec=None, align_mode='linear', align_topk=0,
                align_depth_gamma=0.0, per_layer_spectra_path=None,
                n_layer=None):
    """Apply Prism initialization to a nanoGPT model.

    Spectral Imprint: reshape singular values to match extracted spectrum.
    EigenTransfer: blend singular vectors toward extracted directions.

    Args:
        model: nanoGPT GPT model (after GPT(config), before torch.compile)
        align_strength: 0.0 = spectral only, 1.0 = full UV alignment
        lam: spectral blending (0 = flat, 1 = full shape)
        spectra_path: path to spectra.json (if None, extracts from HF GPT-2)
        directions_path: path to directions.pt (if None, extracts from HF GPT-2)
        verbose: print per-matrix info

    Opt-in transfer knobs (all default to the original behavior):
        align_spec: per-group alignment, dict or 'attention:0.9,ffn_down:0.5,…';
            groups unnamed fall back to align_strength.
        align_mode: 'linear' (index-paired blend, the default) | 'grassmann'
            (principal-angle geodesic) | 'subspace' (grassmann with the strength
            set adaptively from how far the student is from the teacher).
        align_topk: if >0, only the leading k singular directions are transferred;
            the tail stays fresh (top-k structural transfer).
        align_depth_gamma: taper alignment with layer depth — effective strength is
            base·(1 − γ·depth_frac), so later (more data-specific) layers get less
            teacher. 0 = uniform (default).
        per_layer_spectra_path: path to a per-matrix spectra json (name→coeffs); if
            present, each matrix uses its own spectrum instead of the group average.
        n_layer: block count, for depth tapering (read from model config if None).
    """
    align_spec = _parse_align_spec(align_spec)
    if n_layer is None:
        n_layer = getattr(getattr(model, 'config', None), 'n_layer', 0) or 0

    # Optional per-matrix spectra (name → DCT coeffs), overriding the group average.
    per_layer_spectra = {}
    if per_layer_spectra_path and os.path.exists(per_layer_spectra_path):
        with open(per_layer_spectra_path) as f:
            per_layer_spectra = json.load(f)
        if verbose:
            print(f'[prism] Loaded per-layer spectra from {per_layer_spectra_path} '
                  f'({len(per_layer_spectra)} matrices)')

    # Load spectra — from custom path or default HF GPT-2 extraction
    if spectra_path and os.path.exists(spectra_path):
        with open(spectra_path) as f:
            spectra = json.load(f)
        if verbose:
            print(f'[prism] Loaded spectra from {spectra_path}')
    else:
        spectra = extract_spectra()

    # Load directions — from custom path or default HF GPT-2 extraction
    want_dirs = align_strength > 0 or bool(align_spec)
    if want_dirs:
        if directions_path and os.path.exists(directions_path):
            directions = torch.load(directions_path, map_location='cpu', weights_only=False)
            if verbose:
                print(f'[prism] Loaded directions from {directions_path}')
        else:
            directions = extract_directions()
    else:
        directions = {}

    n_shaped = 0
    n_skipped = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() < 2:
                n_skipped += 1
                continue

            group = classify_nanogpt_param(name)
            if group is None:
                n_skipped += 1
                continue

            if group not in spectra:
                n_skipped += 1
                continue

            # Get DCT spectrum — per-matrix if available, else the group average
            coeffs = per_layer_spectra.get(name, spectra[group])

            # Effective alignment for this matrix: per-group override, then a
            # depth taper. Defaults leave this equal to align_strength.
            base = align_strength
            if align_spec and group in align_spec:
                base = align_spec[group]
            if align_depth_gamma:
                df = _layer_depth_frac(name, n_layer)
                if df is not None:
                    base = max(0.0, base * (1.0 - align_depth_gamma * df))

            W = param.data.float()
            orig_shape = W.shape

            # SVD the fresh random weight
            U_fresh, s_fresh, Vt_fresh = torch.linalg.svd(W, full_matrices=False)
            frob = torch.norm(W, 'fro').item()
            n = len(s_fresh)

            # Expand DCT spectrum to match rank
            target_spectrum = dct_expand(np.array(coeffs), n)
            target_t = torch.tensor(target_spectrum, dtype=s_fresh.dtype,
                                    device=s_fresh.device)

            # Blend with flat spectrum
            flat = torch.ones_like(s_fresh)
            shaped = torch.clamp(target_t, min=0.01)
            blended = flat + lam * (shaped - flat)
            blended = torch.clamp(blended, min=0.01)
            s_new = blended * (frob / torch.norm(blended).item())

            # Directional alignment
            U_use = U_fresh
            Vt_use = Vt_fresh
            eff_align = base

            # Find matching pretrained directions
            # nanoGPT name: h.0.attn.c_attn.weight → HF: h.0.attn.c_attn.weight
            if base > 0 and name in directions:
                ext = directions[name]
                Vt_pre = ext['Vt'].to(W.device)
                U_pre = ext['U'].to(W.device)
                # 'subspace' mode: scale strength by how far the student already is
                # from the teacher (transfer more into unaligned subspaces).
                if align_mode == 'subspace' and Vt_pre.shape == Vt_fresh.shape:
                    sa = subspace_alignment(Vt_fresh, Vt_pre, k=align_topk, rows=True)
                    eff_align = base * (1.0 - sa)
                core_mode = ('grassmann' if align_mode in ('grassmann', 'subspace')
                             else 'linear')
                if eff_align > 0:
                    if Vt_pre.shape == Vt_fresh.shape:
                        Vt_use = align_directions(Vt_fresh, Vt_pre, eff_align,
                                                  mode=core_mode, topk=align_topk,
                                                  rows=True)
                    if U_pre.shape == U_fresh.shape:
                        U_use = align_directions(U_fresh, U_pre, eff_align,
                                                 mode=core_mode, topk=align_topk,
                                                 rows=False)

            W_new = U_use @ torch.diag(s_new) @ Vt_use
            param.data = W_new.to(param.dtype)
            n_shaped += 1

            if verbose:
                align_str = f'{eff_align:.2f}' if name in directions else 'n/a'
                print(f'  [prism] {name:45s} {str(list(orig_shape)):>15s} '
                      f'{group:>10s} align={align_str}')

    if verbose:
        print(f'\n[prism] Shaped {n_shaped} matrices, skipped {n_skipped} params')
        extra = []
        if align_mode != 'linear':
            extra.append(f'mode={align_mode}')
        if align_topk:
            extra.append(f'topk={align_topk}')
        if align_depth_gamma:
            extra.append(f'depth_gamma={align_depth_gamma}')
        if align_spec:
            extra.append(f'spec={align_spec}')
        if per_layer_spectra:
            extra.append('per_layer_spectra')
        suffix = ('  [' + ', '.join(extra) + ']') if extra else ''
        print(f'[prism] Spectral Imprint (lam={lam}) + '
              f'EigenTransfer (align={align_strength}){suffix}')

    return n_shaped
