"""
prism_selftest.py — fast, GPU-free validation of the new transfer math.

Run this once on any box with torch installed before trusting the grassmann /
top-k / subspace / per-layer / CKA paths:

    cd src && python prism_selftest.py

It checks mathematical invariants (orthonormality, CKA bounds, geodesic
endpoints), that the default apply_prism path still shapes a model, and that
every new mode runs end-to-end on a tiny GPT. No training, no dataset, seconds.
"""
import torch

from prism_init import (blend_orthogonal, grassmann_interp, subspace_alignment,
                        align_directions, apply_prism, _parse_align_spec)
from prism_cka import linear_cka
from model import GPTConfig, GPT

torch.manual_seed(0)
PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f'  {"✓" if cond else "✗ FAIL"}  {name}')


def orthonormal_cols(M, tol=1e-4):
    I = torch.eye(M.shape[1], device=M.device, dtype=M.dtype)
    return torch.allclose(M.transpose(-2, -1) @ M, I, atol=tol)


# ── CKA ──
print('linear_cka:')
X = torch.randn(512, 64)
check('CKA(X,X) == 1', abs(float(linear_cka(X, X)) - 1.0) < 1e-4)
Q, _ = torch.linalg.qr(torch.randn(64, 64))          # orthogonal transform
check('CKA invariant to orthogonal transform', abs(float(linear_cka(X, X @ Q)) - 1.0) < 1e-3)
check('CKA(X, noise) < 0.5', float(linear_cka(X, torch.randn(512, 64))) < 0.5)
check('CKA in [0,1]', 0.0 <= float(linear_cka(X, torch.randn(512, 32))) <= 1.0001)

# ── subspace_alignment ──
print('subspace_alignment:')
A = torch.linalg.qr(torch.randn(128, 32))[0]
B = torch.linalg.qr(torch.randn(128, 32))[0]
check('align(A,A) == 1', abs(subspace_alignment(A, A) - 1.0) < 1e-4)
check('align(random,random) < 0.7', subspace_alignment(A, B) < 0.7)
check('align in [0,1]', 0.0 <= subspace_alignment(A, B) <= 1.0001)

# ── grassmann_interp (columns) ──
print('grassmann_interp (column space):')
for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
    X_ = grassmann_interp(A, B, alpha)
    check(f'orthonormal @alpha={alpha}', orthonormal_cols(X_))
check('alpha=0 spans A', subspace_alignment(grassmann_interp(A, B, 0.0), A) > 0.999)
check('alpha=1 spans B', subspace_alignment(grassmann_interp(A, B, 1.0), B) > 0.999)

# ── grassmann_interp (rows, for Vt) ──
print('grassmann_interp (row space):')
Ar = torch.linalg.qr(torch.randn(64, 200))[0].transpose(0, 1)[:64]   # (64,200) rows orthonormal
Br = torch.linalg.qr(torch.randn(64, 200))[0].transpose(0, 1)[:64]
Xr = grassmann_interp(Ar, Br, 0.5, rows=True)
check('rows orthonormal', orthonormal_cols(Xr.transpose(-2, -1)))

# ── align_directions: top-k keeps orthonormality; linear default == blend ──
print('align_directions:')
tk = align_directions(A, B, 0.75, mode='grassmann', topk=8)
check('top-k output orthonormal', orthonormal_cols(tk))
lin = align_directions(A, B, 0.75, mode='linear', topk=0)
check('linear+topk0 == blend_orthogonal', torch.allclose(lin, blend_orthogonal(A, B, 0.75), atol=1e-5))

# ── _parse_align_spec ──
print('_parse_align_spec:')
d = _parse_align_spec('attention:0.9,ffn_down:0.5')
check('parses spec', d == {'attention': 0.9, 'ffn_down': 0.5})
check('None passthrough', _parse_align_spec(None) is None)

# ── apply_prism on a tiny GPT: default path + every new mode runs ──
print('apply_prism (tiny GPT, spectra from HF is skipped — pass explicit spectra):')
conf = GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=65, dropout=0.0)


def fresh():
    torch.manual_seed(1)
    return GPT(conf)


# minimal spectra + a fake directions file (shapes from a reference model) so the
# whole path — including the directional blend — runs offline, no HF download.
import json, tempfile, os
from prism_init import classify_nanogpt_param
spectra = {g: [0.0] * 8 for g in ['attention', 'attn_proj', 'ffn_up', 'ffn_down', 'embedding']}
tmp = tempfile.mkdtemp()
sp = os.path.join(tmp, 'spectra.json')
json.dump(spectra, open(sp, 'w'))

ref = fresh()
directions = {}
with torch.no_grad():
    for name, param in ref.named_parameters():
        if param.dim() < 2 or classify_nanogpt_param(name) is None:
            continue
        U, s, Vt = torch.linalg.svd(param.data.float(), full_matrices=False)
        directions[name] = {'U': U, 'Vt': Vt,
                            'group': classify_nanogpt_param(name),
                            'shape': list(param.shape)}
dp = os.path.join(tmp, 'directions.pt')
torch.save(directions, dp)

for kw in [dict(),
           dict(align_mode='grassmann'),
           dict(align_mode='subspace'),
           dict(align_topk=8),
           dict(align_depth_gamma=0.5),
           dict(align_spec='attention:0.9,ffn_down:0.4')]:
    try:
        m = fresh()
        n = apply_prism(m, align_strength=0.75, spectra_path=sp,
                        directions_path=dp, verbose=False, **kw)
        finite = all(torch.isfinite(p).all() for p in m.parameters())
        check(f'apply_prism runs {kw or "default"} (shaped {n}, finite={finite})',
              n > 0 and finite)
    except Exception as e:
        check(f'apply_prism runs {kw}: {type(e).__name__}: {e}', False)

print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
if FAIL:
    print('FAILED:', FAIL)
    raise SystemExit(1)
print('ALL SELFTESTS PASSED')
