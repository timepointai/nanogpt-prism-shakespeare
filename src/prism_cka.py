"""
prism_cka.py — representational-distance transfer for Prism (opt-in, experimental).

The Mod Wheel regularizes the student's *weights* toward the teacher's spectral
targets. This regularizes the student's *representations* instead: it pulls each
block's activations toward the teacher's activations on the same batch, measured
by linear CKA (Centered Kernel Alignment) — "compute the same relationships",
not "have the same matrices".

Wired behind train.py's --prism_cka* flags; off by default. It is differentiated
from any generic regularizer/prune because the signal is the teacher's own
activations, so it can carry structure that survives at large data-distance.
"""
import torch

from model import GPTConfig, GPT


def linear_cka(X, Y, eps=1e-6):
    """Linear CKA between two activation matrices.

    X: (N, d1), Y: (N, d2) — N paired samples (same rows), feature dims may differ.
    Returns a similarity in [0, 1]; 1.0 iff the two representations are identical
    up to an orthogonal transform + isotropic scaling. Differentiable in X."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic_xy = (X.transpose(0, 1) @ Y).pow(2).sum()
    hsic_xx = (X.transpose(0, 1) @ X).pow(2).sum()
    hsic_yy = (Y.transpose(0, 1) @ Y).pow(2).sum()
    return hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + eps)


class CKAMatcher:
    """Holds a frozen teacher, captures per-block activations from both teacher and
    student via forward hooks, and returns the mean (1 - CKA) across chosen blocks
    on a batch. The student's activations are captured by the main training
    forward; call loss(X) right after that forward, before X is reassigned."""

    def __init__(self, teacher_ckpt, device, layers=None, max_samples=2048):
        ck = torch.load(teacher_ckpt, map_location='cpu', weights_only=False)
        conf = GPTConfig(**ck['model_args'])
        teacher = GPT(conf)
        sd = {k.replace('_orig_mod.', ''): v for k, v in ck['model'].items()}
        teacher.load_state_dict(sd, strict=False)
        teacher.eval().to(device)
        for p in teacher.parameters():
            p.requires_grad_(False)

        self.teacher = teacher
        self.device = device
        self.max_samples = max_samples
        n = conf.n_layer
        self.layers = ([i for i in layers if 0 <= i < n] if layers
                       else list(range(n)))
        self._t_acts, self._s_acts = {}, {}
        for i in self.layers:
            teacher.transformer.h[i].register_forward_hook(
                self._capture(self._t_acts, i))

    def attach_student(self, student):
        """Register hooks on the student's blocks (call once, after init)."""
        for i in self.layers:
            student.transformer.h[i].register_forward_hook(
                self._capture(self._s_acts, i))

    def _capture(self, store, i):
        def hook(_mod, _inp, out):
            store[i] = out
        return hook

    def _flat(self, a):
        """(B, T, C) → (N, C), deterministically subsampled to max_samples rows.
        Same indices for teacher and student since both come from the same X."""
        x = a.reshape(-1, a.shape[-1])
        if self.max_samples and x.shape[0] > self.max_samples:
            idx = torch.linspace(0, x.shape[0] - 1, self.max_samples,
                                 device=x.device).long()
            x = x.index_select(0, idx)
        return x.float()

    def loss(self, X):
        """Mean (1 - CKA) over the chosen blocks for batch X, or None if no block
        activations were captured (e.g. hooks didn't fire under compile)."""
        self._t_acts.clear()
        with torch.no_grad():
            self.teacher(X)
        terms = []
        for i in self.layers:
            if i in self._s_acts and i in self._t_acts:
                s = self._flat(self._s_acts[i])
                t = self._flat(self._t_acts[i]).detach()
                if s.shape == t.shape:
                    terms.append(1.0 - linear_cka(s, t))
        if not terms:
            return None
        return torch.stack(terms).mean()
