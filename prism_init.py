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


def blend_orthogonal(A, B, alpha):
    """Blend two orthogonal matrices, re-orthogonalized via SVD."""
    blended = (1 - alpha) * A + alpha * B
    try:
        U, _, Vt = torch.linalg.svd(blended, full_matrices=False)
        return U @ Vt
    except torch._C._LinAlgError:
        return A


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

def apply_prism(model, align_strength=0.75, lam=1.0, verbose=True):
    """Apply Prism initialization to a nanoGPT model.

    Spectral Imprint: reshape singular values to match pretrained spectrum.
    EigenTransfer: blend singular vectors toward pretrained directions.

    Args:
        model: nanoGPT GPT model (after GPT(config), before torch.compile)
        align_strength: 0.0 = spectral only, 1.0 = full UV alignment
        lam: spectral blending (0 = flat, 1 = full shape)
        verbose: print per-matrix info
    """
    spectra = extract_spectra()
    directions = extract_directions() if align_strength > 0 else {}

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

            # Get DCT spectrum for this group
            coeffs = spectra[group]
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

            # Find matching pretrained directions
            # nanoGPT name: h.0.attn.c_attn.weight → HF: h.0.attn.c_attn.weight
            if align_strength > 0 and name in directions:
                ext = directions[name]
                Vt_pre = ext['Vt'].to(W.device)
                U_pre = ext['U'].to(W.device)
                if Vt_pre.shape == Vt_fresh.shape:
                    Vt_use = blend_orthogonal(Vt_fresh, Vt_pre, align_strength)
                if U_pre.shape == U_fresh.shape:
                    U_use = blend_orthogonal(U_fresh, U_pre, align_strength)

            W_new = U_use @ torch.diag(s_new) @ Vt_use
            param.data = W_new.to(param.dtype)
            n_shaped += 1

            if verbose:
                align_str = f'{align_strength:.2f}' if name in directions else 'n/a'
                print(f'  [prism] {name:45s} {str(list(orig_shape)):>15s} '
                      f'{group:>10s} align={align_str}')

    if verbose:
        print(f'\n[prism] Shaped {n_shaped} matrices, skipped {n_skipped} params')
        print(f'[prism] Spectral Imprint (lam={lam}) + '
              f'EigenTransfer (align={align_strength})')

    return n_shaped
