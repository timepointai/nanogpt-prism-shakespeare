"""
prism_extract.py — Extract spectral fingerprint from a trained nanoGPT checkpoint.

Instead of always extracting from HuggingFace GPT-2, this extracts from
ANY trained nanoGPT model. Use the spectra from a converged model to
initialize a fresh model of the same architecture.

Usage:
    python prism_extract.py --ckpt out-shakespeare-char/ckpt.pt --out .prism_cache/shakespeare

This produces:
    shakespeare/spectra.json    (DCT coefficients per group)
    shakespeare/directions.pt   (per-layer U and Vt matrices)

Then in train.py:
    apply_prism(model, spectra_path='shakespeare/spectra.json',
                directions_path='shakespeare/directions.pt')
"""
import argparse
import os
import json
import gc
import numpy as np
import torch

from model import GPTConfig, GPT


# Weight group classification for nanoGPT
def classify_param(name):
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


def extract_from_checkpoint(ckpt_path, out_dir, n_dct=8):
    """Extract spectra and directions from a nanoGPT checkpoint."""
    os.makedirs(out_dir, exist_ok=True)

    print(f'Loading checkpoint: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_args = checkpoint['model_args']
    print(f'Model: n_layer={model_args["n_layer"]}, n_embd={model_args["n_embd"]}, '
          f'vocab={model_args["vocab_size"]}')

    # Reconstruct model and load weights
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'], strict=False)

    # Extract per-layer SVD
    group_svs = {}
    directions = {}

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
            group = classify_param(name)
            if group is None:
                continue

            W = param.data.float()
            U, s, Vt = torch.linalg.svd(W, full_matrices=False)
            s_norm = s / s.max() if s.max() > 0 else s

            group_svs.setdefault(group, []).append(s_norm.numpy())
            directions[name] = {
                'U': U, 'Vt': Vt,
                'group': group, 'shape': list(W.shape),
            }
            print(f'  {name:45s} {str(list(W.shape)):>15s} {group:>10s} '
                  f'top3=[{s_norm[0]:.3f},{s_norm[1]:.3f},{s_norm[2]:.3f}]')

    # Compress to DCT coefficients per group
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
        print(f'  {group}: {len(sv_list)} matrices → {n_dct} DCT coeffs')

    # Save
    spectra_path = os.path.join(out_dir, 'spectra.json')
    with open(spectra_path, 'w') as f:
        json.dump(spectra, f, indent=2)
    print(f'Saved spectra: {spectra_path}')

    dirs_path = os.path.join(out_dir, 'directions.pt')
    torch.save(directions, dirs_path)
    print(f'Saved directions: {dirs_path}')

    del model
    gc.collect()
    return spectra_path, dirs_path


def extract_from_pretrained_hf(model_name='gpt2', out_dir=None):
    """Extract from a HuggingFace pretrained model (original approach)."""
    if out_dir is None:
        out_dir = os.path.join('.prism_cache', model_name.replace('/', '_'))
    os.makedirs(out_dir, exist_ok=True)

    from transformers import GPT2LMHeadModel

    print(f'Loading HuggingFace model: {model_name}')
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)

    group_svs = {}
    directions = {}

    # HF param name → nanoGPT param name mapping
    def hf_to_nano(name):
        return name.replace('transformer.', '')

    with torch.no_grad():
        for name, param in hf_model.named_parameters():
            if param.dim() < 2:
                continue
            group = classify_param(name)
            if group is None:
                continue

            W = param.data.float()
            # HF Conv1D stores (in, out) — transpose to (out, in) for nn.Linear convention
            if W.shape[0] < W.shape[1] and 'wte' not in name and 'wpe' not in name:
                W = W.T

            U, s, Vt = torch.linalg.svd(W, full_matrices=False)
            s_norm = s / s.max() if s.max() > 0 else s

            group_svs.setdefault(group, []).append(s_norm.numpy())
            nano_name = hf_to_nano(name)
            directions[nano_name] = {
                'U': U, 'Vt': Vt,
                'group': group, 'shape': list(W.shape),
            }

    del hf_model
    gc.collect()

    # Compress to DCT
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
        basis = np.zeros((n, 8))
        for i in range(8):
            basis[:, i] = np.cos((i + 0.5) * t)
        coeffs, _, _, _ = np.linalg.lstsq(basis, target, rcond=None)
        spectra[group] = coeffs.tolist()

    spectra_path = os.path.join(out_dir, 'spectra.json')
    with open(spectra_path, 'w') as f:
        json.dump(spectra, f, indent=2)

    dirs_path = os.path.join(out_dir, 'directions.pt')
    torch.save(directions, dirs_path)

    print(f'Saved to {out_dir}/')
    return spectra_path, dirs_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, help='Path to nanoGPT checkpoint')
    parser.add_argument('--hf', type=str, help='HuggingFace model name (e.g. gpt2)')
    parser.add_argument('--out', type=str, default='.prism_cache/extracted')
    args = parser.parse_args()

    if args.ckpt:
        extract_from_checkpoint(args.ckpt, args.out)
    elif args.hf:
        extract_from_pretrained_hf(args.hf, args.out)
    else:
        print('Usage: python prism_extract.py --ckpt path/to/ckpt.pt --out .prism_cache/name')
        print('   or: python prism_extract.py --hf gpt2 --out .prism_cache/gpt2')
