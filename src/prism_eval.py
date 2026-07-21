"""
prism_eval.py — Standardized Prism benchmark.

Measures the Prism Score = baseline_steps / prism_steps to reach baseline's
best val loss on a held-out test set, across one or more seeds.

Every run writes a full artifact (loss curves + provenance) to results/ at the
repo root. Commit that file. A claim without a matching artifact is not a result.

Usage:
    python prism_eval.py                              # recipe, seeds 1337,1338,1339
    python prism_eval.py --seeds=1337                 # single seed (fast, not publishable)
    python prism_eval.py --method=spectral_only       # ablation: spectral shape only
    python prism_eval.py --teacher_steps=500          # cheaper teacher
    python prism_eval.py --report                     # print the last artifact

    # The clean attribution test — recipe at the baseline's schedule, so the
    # ONLY difference between the two arms is the spectral method (no LR confound):
    python prism_eval.py --method_lr=1e-3 --method_warmup=100
    # For the full 2x2, also get the baseline at the recipe's LR:
    python prism_eval.py --baseline_lr=5e-4 --baseline_warmup=50

    # The data-overlap sweep — does the advantage survive as teacher and student
    # stop sharing data? Difficulty-controlled (random blocks across the corpus),
    # wide + shallow probe (Prism acts in the first ~200-300 steps):
    python prism_eval.py --method_lr=1e-3 --method_warmup=20 --baseline_warmup=20 \
        --teacher_steps=500 --student_steps=100 --eval_every=10 --eval_iters=40 \
        --batch_size=32 --overlap=1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.0
    # overlap 1.0 = same-data; 0.0 = disjoint (the cross-data / structural test).

    # Transfer improvements (opt-in; all default to the committed recipe). Each
    # non-default knob is folded into the run key, so it gets its own artifact and
    # never false-resumes onto a plain-recipe result. See IMPROVEMENTS.md.
    python prism_eval.py --method_lr=1e-3 --method_warmup=100 \
        --align_mode=grassmann --align_topk=32     # geometry-paired top-k transfer
    python prism_eval.py --n_dct=16 --per_layer     # truer, per-matrix spectrum
    python prism_eval.py --cka=0.1 --cka_layers=2,4 # representational (CKA) transfer
    # The real structure-vs-content test: fresh blocks from a genuinely far corpus
    python prism_eval.py --overlap=0.0 --far_corpus=data/far.txt --method_lr=1e-3 --method_warmup=100

Recovery:
    The run is STEPWISE and RESUMABLE. Each expensive stage (per seed: teacher,
    baseline, method) is banked to .prism_runs/<run-key>/ the moment it finishes.
    Re-running the SAME command after a crash or a dropped Colab runtime skips
    every completed stage and picks up where it stopped — a finished 110-minute
    baseline is never recomputed. The run key is derived from the experiment
    config, so changing the experiment starts a fresh run; repeating it resumes.

    Nothing here defends against the VM's disk being wiped (a full Colab factory
    reset erases /content). Against that, only committing results/*.json helps.

Protocol:
    1. Prepare Shakespeare char-level dataset
    2. Split: Train (80%) / Teacher-Val (20%) / Test (original val)
    3. Per seed: train teacher on Train, extract fingerprint
    4. Per seed: train baseline on Train, eval on Test, record steps-to-best
    5. Per seed: train Prism student on Train, eval on Test, record steps-to-target
    6. Prism Score = baseline_steps / method_steps, reported as a median + range

Reading the score:
    1.0  = no benefit (same as baseline)
    None = the method never reached baseline quality
    A score is a LOWER BOUND when the method hits target at the first eval:
    resolution is --eval_every, so anything faster is unmeasurable.

Interpretation warning:
    The Prism Score is a RATIO. A weak baseline inflates it. Always read
    baseline_best alongside the score — a run whose baseline is worse than a
    previous run's baseline has produced a bigger score without a better method.

Protocol note (same-data transfer):
    The teacher and the student train on the SAME 80% split. This measures
    same-data spectral transfer. It does NOT establish that the transfer is
    structural rather than content — that requires a cross-data run where the
    teacher's fingerprint comes from a disjoint split.
"""
import argparse
import json
import os
import threading
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, 'results')
RUNS_DIR = os.path.join(REPO_ROOT, '.prism_runs')

# Generous: an A100 finishes a 5000-step run in minutes, but MPS/CPU take hours
# and a timeout here kills the run outright. Sized for the slow case.
TRAIN_TIMEOUT = 24 * 3600

# How often (seconds) to print a "still alive" heartbeat between eval points.
HEARTBEAT_SEC = 45


def log(msg, indent=0):
    """Timestamped stdout line, flushed so a streamed Colab cell stays live."""
    ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
    print(f'{ts}  {" " * indent}{msg}', flush=True)


def provenance():
    """Record what produced these numbers, so a reader can audit them."""

    def git(*a):
        try:
            return subprocess.run(['git', '-C', REPO_ROOT] + list(a),
                                  capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ''

    info = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': git('rev-parse', 'HEAD'),
        'git_dirty': bool(git('status', '--porcelain')),
        'python': sys.version.split()[0],
        'argv': sys.argv[1:],
    }
    try:
        import torch
        info['torch'] = torch.__version__
        info['device'] = (torch.cuda.get_device_name(0)
                          if torch.cuda.is_available() else 'cpu/mps')
    except Exception:
        info['torch'] = info['device'] = 'unknown'
    return info


# Fixed block partition for the overlap sweep — deterministic (independent of the
# training seed) so a given overlap always yields the same slices and can resume.
RANDOM_SLICE_SEED = 20260721
N_BLOCKS = 100


def _token_js(a, b):
    """Jensen–Shannon divergence (bits) between the token histograms of two token
    arrays — a distributional distance. Within one corpus, random slices share a
    distribution so this stays near 0; it only moves across domains. Recorded so
    the sweep's x-axis (overlap) can later be related to a real distance."""
    m = int(max(int(a.max()), int(b.max()))) + 1
    pa = np.bincount(a, minlength=m).astype(float); pa /= pa.sum()
    pb = np.bincount(b, minlength=m).astype(float); pb /= pb.sum()
    pm = 0.5 * (pa + pb)

    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    return round(0.5 * kl(pa, pm) + 0.5 * kl(pb, pm), 6)


def _encode_corpus(path, meta_pkl):
    """Char-encode an external text file with Shakespeare's vocab (stoi from
    meta.pkl), dropping out-of-vocab characters. Returns a uint16 token array — a
    genuinely different distribution in the SAME vocabulary, for the far arm."""
    import pickle
    with open(meta_pkl, 'rb') as f:
        stoi = pickle.load(f)['stoi']
    with open(path, 'r', errors='ignore') as f:
        text = f.read()
    toks = [stoi[c] for c in text if c in stoi]
    if not toks:
        raise RuntimeError(f'far_corpus {path}: no characters in the Shakespeare '
                           f'vocab — pick a text over the same alphabet.')
    return np.array(toks, dtype=np.uint16)


def setup(workdir=SRC_DIR, overlap=None, far_corpus=None, far_val=False):
    """Prepare dataset partitions. Returns (n_student_tokens, n_test_tokens, dist).

    overlap=None (default): teacher and student share the same 80% split —
    same-data transfer (the original protocol); dist is None.

    overlap in [0,1]: the teacher/student data-overlap DIAL, difficulty-controlled.
    The pool is cut into N_BLOCKS blocks; each arm gets a RANDOM half of them, so
    both arms span the whole corpus (same difficulty) at every overlap — the only
    thing that varies is the fraction of blocks they SHARE. The teacher's block
    set is fixed; the student swaps in `1-overlap` of fresh (non-teacher) blocks:
        overlap=1.0 → identical block set (same-data)
        overlap=0.0 → disjoint block sets (cross-data: nothing shared)
    This removes the slice-position/difficulty confound of a sliding window. The
    student is SCORED on the held-out Shakespeare val set by default; with
    far_corpus + far_val=True it is scored on a val mixture mirroring its own
    train mixture (held-out far text for the fresh fraction) — the accelerated-
    learning-of-the-new-domain test. dist carries the realized overlap plus the
    token-JS distance between the two arms' data."""
    os.chdir(workdir)

    if not os.path.exists('data/shakespeare_char/train.bin'):
        log('Preparing Shakespeare char dataset (first run only)…', 2)
        subprocess.run([sys.executable, 'data/shakespeare_char/prepare.py'],
                       capture_output=True, check=True)

    pool = np.array(np.memmap('data/shakespeare_char/train.bin',
                              dtype=np.uint16, mode='r'))
    test_data = np.array(np.memmap('data/shakespeare_char/val.bin',
                                    dtype=np.uint16, mode='r'))
    student_val = test_data
    val_mix = None
    dist = None

    if overlap is None:
        split = int(len(pool) * 0.80)
        teacher_train = pool[:split].astype(np.uint16)
        student_train = pool[:split].astype(np.uint16)
        teacher_val = pool[split:].astype(np.uint16)
    else:
        overlap = max(0.0, min(1.0, float(overlap)))
        blk = len(pool) // N_BLOCKS
        blocks = pool[:blk * N_BLOCKS].reshape(N_BLOCKS, blk)
        perm = np.random.default_rng(RANDOM_SLICE_SEED).permutation(N_BLOCKS)
        half = N_BLOCKS // 2
        teacher_idx, other_idx = perm[:half], perm[half:]
        n_shared = int(round(overlap * half))
        teacher_train = blocks[teacher_idx].reshape(-1).astype(np.uint16)
        shared = blocks[teacher_idx[:n_shared]].reshape(-1).astype(np.uint16)
        n_fresh = half - n_shared
        if far_corpus:
            # The fresh (non-shared) blocks come from a DIFFERENT corpus, so the
            # student's data is distributionally far from the teacher's — the real
            # structure-vs-content test (large token-JS), not just disjoint blocks
            # of the same Shakespeare.
            far = _encode_corpus(far_corpus, 'data/shakespeare_char/meta.pkl')
            far_val_toks = None
            if far_val:
                # Reserve a held-out tail of the far corpus BEFORE any tiling, so
                # the far val data is never seen in training even when the far
                # text is shorter than the fresh-block budget.
                n_res = min(len(far) // 10, len(test_data))
                far_val_toks = far[-n_res:].astype(np.uint16)
                far = far[:-n_res]
            need = n_fresh * blk
            if need and len(far) < need:
                far = np.tile(far, int(np.ceil(need / len(far))))
            fresh = far[:need].astype(np.uint16)
            if far_val_toks is not None:
                # Score the student on its OWN data mixture: Shakespeare val in
                # proportion to the shared blocks, held-out far text for the
                # rest. overlap=1.0 → pure Shakespeare val (identical to the base
                # protocol); overlap=0.0 → pure far-corpus val. This measures
                # accelerated learning OF the student's domain — a Shakespeare
                # val set would instead reward retaining the teacher's domain.
                n_sh_val = int(round((n_shared / half) * len(test_data)))
                n_fa_val = min(len(far_val_toks), len(test_data) - n_sh_val)
                student_val = np.concatenate(
                    [test_data[:n_sh_val].astype(np.uint16),
                     far_val_toks[:n_fa_val]]).astype(np.uint16)
                val_mix = {'shakespeare_tokens': int(n_sh_val),
                           'far_tokens': int(n_fa_val)}
        else:
            fresh = blocks[other_idx[:n_fresh]].reshape(-1).astype(np.uint16)
        student_train = np.concatenate([shared, fresh]).astype(np.uint16)
        teacher_val = test_data
        dist = {'overlap_requested': overlap,
                'overlap_realized': round(n_shared / half, 4),
                'shared_blocks': n_shared, 'blocks_per_arm': half,
                'far_corpus': os.path.basename(far_corpus) if far_corpus else None,
                'far_val': bool(far_corpus and far_val),
                'token_js': _token_js(teacher_train, student_train)}
        if val_mix:
            dist['val_mix'] = val_mix
        log(f'overlap={overlap:.2f}: {n_shared}/{half} blocks shared · '
            f'token-JS {dist["token_js"]:.4f} · '
            f'{"FAR corpus fresh blocks" if far_corpus else "random blocks across corpus"} '
            f'(difficulty controlled)', 2)

    for name, train, val in [('shakespeare_eval', student_train, student_val),
                              ('shakespeare_teacher', teacher_train, teacher_val)]:
        d = f'data/{name}'
        os.makedirs(d, exist_ok=True)
        train.astype(np.uint16).tofile(os.path.join(d, 'train.bin'))
        val.astype(np.uint16).tofile(os.path.join(d, 'val.bin'))
        shutil.copy('data/shakespeare_char/meta.pkl',
                     os.path.join(d, 'meta.pkl'))

    return len(student_train), len(student_val), dist


def parse_curve(stdout):
    """Extract {step: val_loss} from train.py's eval lines."""
    curve = {}
    for line in stdout.split('\n'):
        m = re.search(r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)', line)
        if m:
            curve[int(m.group(1))] = float(m.group(3))
    return curve


def default_device():
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'


def stream_train(cmd, label, max_step, timeout=TRAIN_TIMEOUT):
    """Run a training subprocess, echoing progress live so a long run stays
    auditable. Prints every eval point (with elapsed + ETA) and a throttled
    heartbeat from the iter log in between. Returns (returncode, full_stdout).

    Live output is the point: on flaky Colab, a silent 110-minute run is
    indistinguishable from a hung one. This makes the run narrate itself.
    """
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    out = []
    last_beat = 0.0
    try:
        for line in proc.stdout:
            out.append(line)
            now = time.time()
            el = now - t0

            m = re.search(r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)', line)
            if m:
                step, vloss = int(m.group(1)), float(m.group(3))
                eta = (el / step) * (max_step - step) if step else 0.0
                log(f'[{label}] eval @ step {step:>5}/{max_step}  '
                    f'val {vloss:.4f}   '
                    f'{el / 60:.1f}m elapsed · ~{eta / 60:.0f}m left', 4)
                last_beat = now
                continue

            h = re.search(r'iter (\d+): loss ([\d.]+), time ([\d.]+)ms', line)
            if h and now - last_beat >= HEARTBEAT_SEC:
                log(f'[{label}] … alive, iter {int(h.group(1)):>5}/{max_step}  '
                    f'({el / 60:.1f}m)', 4)
                last_beat = now

            if el > timeout:
                proc.kill()
                raise RuntimeError(f'{label}: exceeded {timeout}s wall clock, killed.')
    finally:
        rc = proc.wait()
    return rc, ''.join(out)


def train_teacher(steps, seed, eval_iters, device, label, cache_tag='eval',
                  batch_size=None, n_dct=8, per_layer=False):
    """Train teacher model and extract fingerprint. One teacher per (seed, data,
    steps). Cached by the presence of directions.pt, so a resumed run skips it.
    cache_tag separates teachers trained on different data (e.g. the same-data
    'eval' teacher vs. an 'rsweep' teacher on random blocks); the step count is in
    the path too, so changing teacher_steps can't silently reuse a stale one."""
    # n_dct / per_layer change the fingerprint, so they're in the cache path — a
    # 16-coeff extract must never silently reuse an 8-coeff one.
    sfx = ('' if n_dct == 8 else f'_dct{n_dct}') + ('_pl' if per_layer else '')
    cache = f'.prism_cache/{cache_tag}_teacher_s{seed}_t{steps}{sfx}'
    if os.path.exists(f'{cache}/directions.pt'):
        log(f'[resume] teacher (seed {seed}) already trained — using cached '
            f'fingerprint.', 2)
        return cache

    log(f'Training teacher (seed {seed}, {steps} steps)…', 2)
    rc, out = stream_train([
        sys.executable, '-u', 'train.py', 'config/train_shakespeare_char.py',
        '--dataset=shakespeare_teacher', f'--seed={seed}', f'--device={device}',
        f'--max_iters={steps}', f'--eval_interval={steps}',
        f'--eval_iters={eval_iters}', '--log_interval=100',
        f'--out_dir=out-eval-teacher-s{seed}',
        '--always_save_checkpoint=True',
        '--compile=False', '--prism_init=False', '--wandb_log=False',
    ] + ([f'--batch_size={batch_size}'] if batch_size else []),
        f'{label} teacher', steps)

    if rc != 0:
        raise RuntimeError(f'Teacher training failed (seed {seed}):\n{out[-2000:]}')

    curve = parse_curve(out)
    if curve:
        log(f'teacher val loss {list(curve.values())[-1]:.4f}. Extracting '
            f'fingerprint…', 2)
    e = subprocess.run([
        sys.executable, 'prism_extract.py',
        '--ckpt', f'out-eval-teacher-s{seed}/ckpt.pt',
        '--out', cache, f'--n_dct={n_dct}',
    ] + (['--per_layer'] if per_layer else []), capture_output=True, text=True, timeout=300)
    if e.returncode != 0:
        raise RuntimeError(f'Fingerprint extraction failed:\n{e.stderr[-2000:]}')
    log(f'fingerprint saved to {cache}.', 2)

    return cache


def run_training(name, extra_args, seed, steps, eval_every, eval_iters, device,
                 label, batch_size=None):
    """Run one training config. Raises on failure — never scores a partial run."""
    log(f'Running {name} (seed {seed}, {steps} steps, eval every {eval_every})…', 2)
    t0 = time.time()
    rc, out = stream_train(
        [sys.executable, '-u', 'train.py', 'config/train_shakespeare_char.py',
         '--dataset=shakespeare_eval', f'--seed={seed}', f'--device={device}',
         f'--max_iters={steps}', f'--eval_interval={eval_every}',
         f'--eval_iters={eval_iters}', '--log_interval=100',
         f'--out_dir=out-eval-{name}-s{seed}',
         '--wandb_log=False', '--compile=False']
        + ([f'--batch_size={batch_size}'] if batch_size else []) + extra_args,
        f'{label} {name}', steps)
    wall = time.time() - t0

    if rc != 0:
        raise RuntimeError(f'Training failed ({name}, seed {seed}):\n{out[-2000:]}')

    curve = parse_curve(out)
    if not curve:
        raise RuntimeError(f'No eval lines parsed ({name}, seed {seed}). '
                           f'stdout tail:\n{out[-1000:]}')
    if max(curve) < steps:
        raise RuntimeError(f'Run truncated ({name}, seed {seed}): last eval at '
                           f'step {max(curve)}, expected {steps}. Refusing to '
                           f'score a partial curve.')

    best = min(curve.values())
    best_step = min(curve, key=curve.get)
    at_end = curve[max(curve)]
    log(f'{name} done: best={best:.4f} @{best_step}, @{steps}={at_end:.4f}, '
        f'{wall / 60:.1f}m', 2)

    return {
        'curve': {str(k): v for k, v in sorted(curve.items())},
        'best': best,
        'best_step': best_step,
        'at_end': at_end,
        'wall_sec': round(wall, 1),
        'overfits': at_end > best * 1.05,
    }


def compute_score(baseline, method, eval_every):
    """Prism Score = baseline steps-to-best / method steps-to-baseline-quality."""
    target = baseline['best']
    curve = {int(k): v for k, v in method['curve'].items()}
    hit = next((s for s in sorted(curve) if curve[s] <= target), None)

    if hit is None:
        return {
            'prism_score': None,
            'hit_step': None,
            'reached_baseline_quality': False,
            'baseline_target': target,
            'left_censored': False,
            'note': 'method never reached baseline best loss',
        }

    if hit == 0:
        # The spectral init alone already meets the baseline's best, before any
        # training — the ratio is unbounded. Flag it rather than divide by zero.
        return {
            'prism_score': None,
            'hit_step': 0,
            'reached_baseline_quality': True,
            'reached_at_init': True,
            'baseline_target': target,
            'left_censored': True,
            'note': 'reached baseline best at initialization (step 0) — the '
                    'spectral init alone matches it; speedup unbounded',
        }

    return {
        'prism_score': baseline['best_step'] / hit,
        'hit_step': hit,
        'reached_baseline_quality': True,
        'baseline_target': target,
        # Hitting target at the first eval means the true crossing is somewhere
        # in (0, eval_every]; the score is a lower bound, not a measurement.
        'left_censored': hit == eval_every,
        'note': ('score is a LOWER BOUND: target reached at first eval, '
                 f'true crossing unresolved below step {eval_every}')
                if hit == eval_every else '',
    }


def method_args_for(method, cache, lr, warmup, knobs=None, seed=None):
    """Build the train.py flags for one method arm, plus any opt-in transfer knobs.
    knobs come straight from the CLI (align_mode/topk/…, mod overrides, CKA); they
    append to the base recipe so the defaults stay byte-for-byte identical."""
    common = [
        '--prism_init=True',
        f'--prism_spectra={cache}/spectra.json',
        f'--learning_rate={lr}', f'--warmup_iters={warmup}',
    ]
    dirs = [f'--prism_directions={cache}/directions.pt']
    base = {
        'recipe':        common + ['--prism_align=0.75'] + dirs + ['--prism_mod=0.01', '--prism_mod_decay=0.9999'],
        'marathon':      common + ['--prism_align=0.75'] + dirs + ['--prism_mod=0.01', '--prism_mod_decay=0.9999'],
        'sprint':        common + ['--prism_align=0.75'] + dirs + ['--prism_mod=0.005', '--prism_mod_decay=0.999'],
        'spectral_only': common + ['--prism_align=0.0', '--prism_mod=0.01', '--prism_mod_decay=0.9999'],
        'dirs_only':     common + ['--prism_align=0.75'] + dirs,
    }[method]

    k = knobs or {}
    extra = []
    if k.get('per_layer'):
        extra.append(f'--prism_per_layer_spectra={cache}/spectra_per_layer.json')
    if k.get('align_mode') is not None:
        extra.append(f'--prism_align_mode={k["align_mode"]}')
    if k.get('align_topk') is not None:
        extra.append(f'--prism_align_topk={k["align_topk"]}')
    if k.get('align_depth_gamma') is not None:
        extra.append(f'--prism_align_depth_gamma={k["align_depth_gamma"]}')
    if k.get('align_spec') is not None:
        extra.append(f'--prism_align_spec={k["align_spec"]}')
    if k.get('mod') is not None:
        extra.append(f'--prism_mod={k["mod"]}')
    if k.get('mod_decay') is not None:
        extra.append(f'--prism_mod_decay={k["mod_decay"]}')
    if k.get('mod_transition') is not None:
        extra.append(f'--prism_mod_transition={k["mod_transition"]}')
    if k.get('mod_sustain') is not None:
        extra.append(f'--prism_mod_sustain={k["mod_sustain"]}')
    if k.get('cka') is not None and seed is not None:
        extra.append(f'--prism_cka={k["cka"]}')
        extra.append(f'--prism_cka_teacher=out-eval-teacher-s{seed}/ckpt.pt')
        if k.get('cka_layers') is not None:
            extra.append(f'--prism_cka_layers={k["cka_layers"]}')
    return base + extra


def _stats(vals):
    if not vals:
        return None
    s = sorted(vals)
    return {'median': s[len(s) // 2], 'min': s[0], 'max': s[-1],
            'values': [round(v, 4) for v in vals]}


def _summ_group(group):
    scores = [r['score']['prism_score'] for r in group
              if r['score']['prism_score'] is not None]
    return {
        'n_seeds': len(group),
        'n_reached_baseline': len(scores),
        'prism_score': _stats(scores),
        'any_left_censored': any(r['score']['left_censored'] for r in group),
        'baseline_best': _stats([r['baseline']['best'] for r in group]),
        'method_best': _stats([r['method']['best'] for r in group]),
        'method_overfits_any': any(r['method']['overfits'] for r in group),
        'baseline_overfits_any': any(r['baseline']['overfits'] for r in group),
    }


def summarize(runs):
    """Flat summary for a single-condition run; a per-condition breakdown when the
    run sweeps a dimension (overlap or teacher_steps), so the artifact shows the
    advantage-vs-dimension curve directly."""
    if not runs:
        return None
    if any(r.get('overlap') is not None for r in runs):
        key, label, rev = 'overlap', 'by_overlap', True
    elif any(r.get('teacher_steps') is not None for r in runs):
        key, label, rev = 'teacher_steps', 'by_teacher_steps', False
    else:
        return _summ_group(runs)
    by = {}
    for r in runs:
        by.setdefault(r[key], []).append(r)
    return {label: {f'{k:g}': _summ_group(g)
                    for k, g in sorted(by.items(), reverse=rev)}}


# ---------------------------------------------------------------------------
# Resume machinery: a run key, a stale-lock guard, and a per-stage disk cache.
# ---------------------------------------------------------------------------

def run_key(config):
    """A stable, human-readable id for this experiment. Same command → same key
    → resumes; different experiment → different key → fresh run. Device is
    excluded on purpose so a run can resume on a different accelerator.

    The student schedule (LRs / warmups) is part of the experiment's identity: a
    recipe at LR 1e-3 must NOT resume onto a recipe-at-5e-4 result. Non-default
    schedule knobs are appended so those runs get their own dir."""
    seeds = '-'.join(str(s) for s in config['seeds'])
    key = (f"{config['method']}_t{config['teacher_steps']}"
           f"_s{config['student_steps']}_e{config['eval_every']}"
           f"_i{config['eval_iters']}_seeds{seeds}")
    if config.get('method_lr') != '5e-4' or config.get('method_warmup') != 50:
        key += f"_mlr{config['method_lr']}mw{config['method_warmup']}"
    if config.get('baseline_lr') or config.get('baseline_warmup') is not None:
        key += f"_blr{config.get('baseline_lr') or 'def'}bw{config.get('baseline_warmup')}"
    if config.get('overlaps'):
        key += "_ov" + '-'.join(f"{o:g}" for o in config['overlaps'])
    if config.get('teacher_sweep'):
        key += "_ts" + '-'.join(str(t) for t in config['teacher_sweep'])
    # New transfer knobs are part of the experiment's identity too — a grassmann /
    # top-k / per-layer / CKA run must not resume onto a plain-recipe result.
    if config.get('n_dct', 8) != 8:
        key += f"_dct{config['n_dct']}"
    mk = config.get('method_knobs') or {}
    for k in sorted(mk):
        v = mk[k]
        tag = re.sub(r'[^A-Za-z0-9.]+', '', str(v))
        key += f"_{k[:3]}{tag}"
    return key


# A lock is "live" only if its heartbeat is fresher than this. Timestamp-based,
# NOT pid-based: each cloud run is a fresh container where a pid from another
# container is meaningless (and low pids like 12 always exist), so pid-liveness
# gives false "already running" errors. A heartbeat works across containers.
LOCK_STALE_SEC = 300


def _lock_age(info):
    """Seconds since the lock's heartbeat, or None if it has none (old format)."""
    hb = info.get('heartbeat')
    if not hb:
        return None
    try:
        return (datetime.now(timezone.utc)
                - datetime.fromisoformat(hb)).total_seconds()
    except Exception:
        return None


def _write_lock(lock):
    with open(lock, 'w') as f:
        json.dump({'pid': os.getpid(),
                   'heartbeat': datetime.now(timezone.utc).isoformat(),
                   'started': datetime.now(timezone.utc).isoformat()}, f)


def _heartbeat_loop(lock):
    """Refresh the lock's timestamp so a concurrent run can tell we're alive.
    On a Volume, the parent's periodic commit is what makes these writes visible
    to other containers."""
    while True:
        time.sleep(60)
        try:
            info = json.load(open(lock))
            info['heartbeat'] = datetime.now(timezone.utc).isoformat()
            with open(lock, 'w') as f:
                json.dump(info, f)
        except Exception:
            return


def acquire_lock(run_dir):
    """Refuse to start only if another run's heartbeat is still fresh; take over
    any lock that's stale or heartbeat-less (a crashed/killed run, or the old
    pid-based format). Prevents two evals clobbering the same Volume while always
    allowing a clean resume after a death."""
    lock = os.path.join(run_dir, 'lock.json')
    if os.path.exists(lock):
        try:
            info = json.load(open(lock))
        except Exception:
            info = {}
        age = _lock_age(info)
        if age is not None and age < LOCK_STALE_SEC:
            raise SystemExit(
                f'\nAnother prism_eval is actively running for this config '
                f'(heartbeat {age:.0f}s ago < {LOCK_STALE_SEC}s).\n'
                f'If you are certain nothing is running, delete\n'
                f'  {lock}\nand run again.')
        why = 'no heartbeat' if age is None else f'heartbeat {age:.0f}s old'
        log(f'[resume] prior lock is stale ({why}) — taking over.', 2)
    _write_lock(lock)
    threading.Thread(target=_heartbeat_loop, args=(lock,), daemon=True).start()
    return lock


def cached_stage(run_dir, key, fn):
    """Run fn() once and bank its dict to disk; on a resumed run return the
    banked result instantly instead of recomputing. This is what turns a
    dropped runtime from 'redo everything' into 'redo the stage in flight'."""
    path = os.path.join(run_dir, key + '.json')
    if os.path.exists(path):
        try:
            r = json.load(open(path))
            log(f'[resume] stage "{key}" already complete — loaded from disk '
                f'(best {r.get("best", float("nan")):.4f}).', 2)
            return r
        except Exception:
            log(f'[resume] stage "{key}" cache unreadable — recomputing.', 2)
    r = fn()
    with open(path, 'w') as f:
        json.dump(r, f, indent=2)
    return r


def main():
    p = argparse.ArgumentParser(description='Prism Eval — standardized benchmark')
    p.add_argument('--teacher_steps', type=int, default=2000)
    p.add_argument('--student_steps', type=int, default=5000)
    p.add_argument('--eval_every', type=int, default=100)
    p.add_argument('--eval_iters', type=int, default=200,
                   help='val batches per eval; low values add noise (nanoGPT default 200)')
    p.add_argument('--seeds', type=str, default='1337,1338,1339',
                   help='comma-separated seeds; a single seed is not publishable')
    p.add_argument('--method', type=str, default='recipe',
                   choices=['recipe', 'spectral_only', 'dirs_only', 'sprint', 'marathon'])
    p.add_argument('--device', type=str, default=None,
                   help="cuda | mps | cpu (default: best available)")
    # Student-schedule knobs. The default recipe lowers the LR vs the baseline,
    # which confounds "is it Prism or the schedule". To test cleanly, match them:
    #   --method_lr=1e-3 --method_warmup=100   (recipe at the baseline's schedule)
    # Then baseline and method differ by NOTHING but the spectral flags.
    p.add_argument('--method_lr', type=str, default='5e-4',
                   help="student LR for the Prism arm (default 5e-4; set 1e-3 to match baseline)")
    p.add_argument('--method_warmup', type=int, default=50,
                   help="warmup iters for the Prism arm (default 50; set 100 to match baseline)")
    p.add_argument('--baseline_lr', type=str, default=None,
                   help="student LR for the baseline arm (default: config's 1e-3)")
    p.add_argument('--baseline_warmup', type=int, default=None,
                   help="warmup iters for the baseline arm (default: config's 100)")
    # Teacher/student data-overlap dial (a sweep). Constant per-arm data budget
    # (half the pool each), only the shared fraction varies: 1.0=same-data,
    # 0.0=disjoint (cross-data). A comma list sweeps them in one run.
    #   --overlap=1.0,0.75,0.5,0.25,0.0
    p.add_argument('--overlap', type=str, default=None,
                   help="comma-separated overlap fractions to sweep (1.0=same-data, 0.0=disjoint)")
    p.add_argument('--batch_size', type=int, default=None,
                   help="training batch size for all arms (default: config 64; smaller = faster/noisier probes)")
    # Teacher strength/size sweep: does a better teacher give a bigger head start?
    # Same-data; baseline is teacher-independent (computed once/seed), the recipe
    # is retrained per teacher size. e.g. --teacher_sweep=100,250,500,1000,2000
    p.add_argument('--teacher_sweep', type=str, default=None,
                   help="comma-separated teacher step counts to sweep (measures advantage vs teacher strength)")
    # A genuinely FAR student arm: at overlap 0.0 the disjoint blocks still come
    # from the SAME corpus (small token-JS), so it can't separate structure from
    # content. --far_corpus swaps the student's non-shared blocks for text from a
    # different corpus (char-encoded with Shakespeare's vocab) so token-JS is large
    # — the real structure-vs-content test. Report advantage against token-JS.
    p.add_argument('--far_corpus', type=str, default=None,
                   help="path to a .txt corpus for the far (large-distance) student arm")
    p.add_argument('--far_val', action='store_true',
                   help="score the student on a val mixture mirroring its own train "
                        "mixture (held-out far text for the fresh fraction) instead "
                        "of the Shakespeare val set — measures accelerated learning "
                        "OF the far domain rather than retention of the teacher's")
    # ── New transfer knobs (opt-in; recorded in the artifact and the run key) ──
    p.add_argument('--n_dct', type=int, default=None,
                   help="DCT coeffs for the spectral imprint (default 8); higher = truer spectrum")
    p.add_argument('--per_layer', action='store_true',
                   help="imprint each matrix's own spectrum instead of the group average")
    p.add_argument('--align_mode', type=str, default=None,
                   choices=['linear', 'grassmann', 'subspace'],
                   help="EigenTransfer direction pairing: linear (default) | grassmann | subspace")
    p.add_argument('--align_topk', type=int, default=None,
                   help="transfer only the leading k singular directions (default: all)")
    p.add_argument('--align_depth_gamma', type=float, default=None,
                   help="taper alignment with layer depth: base*(1-gamma*depth_frac)")
    p.add_argument('--align_spec', type=str, default=None,
                   help="per-group alignment, e.g. 'attention:0.9,ffn_down:0.5'")
    p.add_argument('--mod', type=str, default=None,
                   help="override Mod Wheel strength (default 0.01 for recipe)")
    p.add_argument('--mod_decay', type=str, default=None,
                   help="override Mod Wheel per-step decay (default 0.9999)")
    p.add_argument('--mod_transition', type=int, default=None,
                   help="Mod Wheel attack→sustain transition step (0 = single phase)")
    p.add_argument('--mod_sustain', type=str, default=None,
                   help="Mod Wheel sustain-phase strength (needs --mod_transition)")
    p.add_argument('--cka', type=str, default=None,
                   help="weight on the (1-CKA) representational-distance loss (needs teacher ckpt)")
    p.add_argument('--cka_layers', type=str, default=None,
                   help="comma block indices for the CKA match (default: all)")
    p.add_argument('--report', action='store_true', help='print the last artifact')
    args = p.parse_args()
    device = args.device or default_device()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    latest = os.path.join(RESULTS_DIR, 'latest.json')

    if args.report:
        if os.path.exists(latest):
            print_report(json.load(open(latest)))
        else:
            print('No artifact in results/. Run the eval first.')
        return

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    overlaps = ([float(x) for x in args.overlap.split(',') if x.strip() != '']
                if args.overlap else None)
    teacher_sweep = ([int(x) for x in args.teacher_sweep.split(',') if x.strip() != '']
                     if args.teacher_sweep else None)

    # Non-default transfer knobs — recorded in the artifact and folded into the run
    # key so a knob change starts a fresh run instead of false-resuming. n_dct also
    # governs teacher extraction, so it lives outside method_knobs (see below).
    n_dct = args.n_dct or 8
    method_knobs = {}
    for k, v, default in [
        ('per_layer', args.per_layer, False),
        ('align_mode', args.align_mode, None),
        ('align_topk', args.align_topk, None),
        ('align_depth_gamma', args.align_depth_gamma, None),
        ('align_spec', args.align_spec, None),
        ('mod', args.mod, None),
        ('mod_decay', args.mod_decay, None),
        ('mod_transition', args.mod_transition, None),
        ('mod_sustain', args.mod_sustain, None),
        ('cka', args.cka, None),
        ('cka_layers', args.cka_layers, None),
    ]:
        if v != default:
            method_knobs[k] = v
    if args.n_dct is not None:
        method_knobs['n_dct'] = args.n_dct
    if args.far_corpus:
        method_knobs['far_corpus'] = os.path.basename(args.far_corpus)
    if args.far_val:
        # Changes what BOTH arms are scored on — a far_val run must never resume
        # onto (or be compared against) a Shakespeare-val run.
        method_knobs['far_val'] = 1

    config = {
        'method': args.method,
        'teacher_steps': args.teacher_steps,
        'student_steps': args.student_steps,
        'eval_every': args.eval_every,
        'eval_iters': args.eval_iters,
        'seeds': seeds,
        'device': device,
        'overlaps': overlaps,
        'overlap_distances': {},
        'teacher_sweep': teacher_sweep,
        'batch_size': args.batch_size,
        'teacher_data_equals_student_data': overlaps is None,
        'method_lr': args.method_lr,
        'method_warmup': args.method_warmup,
        'baseline_lr': args.baseline_lr,
        'baseline_warmup': args.baseline_warmup,
        'schedule_matched': (args.method_lr == (args.baseline_lr or '1e-3')
                             and args.method_warmup == (args.baseline_warmup or 100)),
        'n_dct': n_dct,
        'method_knobs': method_knobs,
        'far_corpus': args.far_corpus,
        'far_val': args.far_val,
    }

    key = run_key(config)
    run_dir = os.path.join(RUNS_DIR, key)
    os.makedirs(run_dir, exist_ok=True)

    print('=' * 64)
    log('PRISM EVAL')
    print('=' * 64)

    # Reuse the run's original timestamp so every resume writes the SAME
    # results file instead of littering orphans; store it beside the stages.
    meta_path = os.path.join(run_dir, 'meta.json')
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        log(f'[resume] existing run for this config, started '
            f'{meta["started_utc"]}.', 2)
    else:
        meta = {'stamp': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
                'started_utc': datetime.now(timezone.utc).isoformat()}
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
    stamp = meta['stamp']
    path = os.path.join(RESULTS_DIR, f'{args.method}_{stamp}.json')

    # Report what is already banked, so a resume tells you exactly where it is.
    if teacher_sweep:
        total_stages = len(seeds) * (1 + 2 * len(teacher_sweep))  # 1 baseline + (teacher+recipe)/size
    else:
        total_stages = (len(overlaps) if overlaps else 1) * len(seeds) * 3
    done = [f for f in os.listdir(run_dir) if f.endswith('.json')
            and f not in ('meta.json', 'lock.json')]
    log(f'Run key: {key}', 2)
    log(f'Work dir: .prism_runs/{key}/  ({len(done)}/{total_stages} stages '
        f'already banked)', 2)

    lock = acquire_lock(run_dir)
    try:
        log(f'Seeds: {seeds} | Method: {args.method} | Device: {device}', 2)
        log(f'Schedule: method LR {args.method_lr}/warmup {args.method_warmup} · '
            f'baseline LR {args.baseline_lr or "1e-3(config)"}/warmup '
            f'{args.baseline_warmup if args.baseline_warmup is not None else "100(config)"}'
            + ('  [MATCHED — only the spectral flags differ]' if config['schedule_matched']
               else '  [recipe LR differs from baseline — confounded]'), 2)
        if overlaps is not None:
            log(f'Overlap sweep: {overlaps}  (teacher/student data-overlap dial; '
                f'1.0=same-data, 0.0=disjoint/cross-data)', 2)
        if teacher_sweep is not None:
            log(f'Teacher sweep: {teacher_sweep} steps  (does a stronger teacher '
                f'give a bigger head start? same-data, baseline reused per seed)', 2)
        if len(seeds) == 1:
            log('WARNING: single seed. Reports one sample, not a result.', 2)

        def build(runs, complete):
            return {
                'schema': 'prism-eval/1',
                'partial': not complete,
                'run_key': key,
                'seeds_requested': seeds,
                'seeds_done': [r['seed'] for r in runs],
                'provenance': provenance(),
                'config': config,
                'runs': runs,
                'summary': summarize(runs) if runs else None,
            }

        def persist(runs, complete):
            art = build(runs, complete)
            for f in (path, latest):
                with open(f, 'w') as fh:
                    json.dump(art, fh, indent=2)
            return art

        stage = [0]  # mutable counter for closure-free stage labelling
        bs = args.batch_size
        runs = []

        if teacher_sweep is not None:
            # Teacher strength/size sweep. Same-data; the baseline never sees the
            # teacher, so it is trained ONCE per seed and reused, while the recipe
            # is retrained per teacher size. Reveals advantage vs teacher strength.
            setup(overlap=None)
            for seed in seeds:
                stage[0] += 1
                log(f'[stage {stage[0]}/{total_stages}] baseline (seed {seed})', 2)
                base_extra = ['--prism_init=False']
                if args.baseline_lr:
                    base_extra.append(f'--learning_rate={args.baseline_lr}')
                if args.baseline_warmup is not None:
                    base_extra.append(f'--warmup_iters={args.baseline_warmup}')
                baseline = cached_stage(run_dir, f's{seed}_baseline', lambda:
                    run_training('baseline', base_extra, seed, args.student_steps,
                                 args.eval_every, args.eval_iters, device,
                                 f's{seed}', batch_size=bs))
                for ts in teacher_sweep:
                    print('-' * 64)
                    log(f'TEACHER {ts} steps · seed {seed}', 2)
                    lbl = f's{seed}_t{ts}'
                    stage[0] += 1
                    log(f'[stage {stage[0]}/{total_stages}] teacher {ts} (seed {seed})', 2)
                    cache = train_teacher(ts, seed, args.eval_iters, device, lbl,
                                          cache_tag='tsweep', batch_size=bs)
                    stage[0] += 1
                    log(f'[stage {stage[0]}/{total_stages}] {args.method} '
                        f'(seed {seed}, teacher {ts})', 2)
                    method = cached_stage(run_dir, f's{seed}_t{ts}_{args.method}', lambda:
                        run_training(args.method,
                                     method_args_for(args.method, cache,
                                                     args.method_lr, args.method_warmup,
                                                     knobs=method_knobs, seed=seed),
                                     seed, args.student_steps, args.eval_every,
                                     args.eval_iters, device, lbl, batch_size=bs))
                    runs.append({
                        'seed': seed,
                        'teacher_steps': ts,
                        'baseline': baseline,
                        'method': method,
                        'score': compute_score(baseline, method, args.eval_every),
                    })
                    persist(runs, complete=False)
                    log(f'  cell seed {seed} teacher {ts} banked ({len(runs)} total).', 2)
            artifact = persist(runs, complete=True)
            print_report(artifact)
            log(f'Artifact: results/{os.path.basename(path)} (also latest.json)', 2)
            log('COMMIT THIS FILE — it is the evidence for any claim you publish.', 2)
            return

        sweep = overlaps if overlaps is not None else [None]
        for overlap in sweep:
            n_train, n_test, dist = setup(overlap=overlap, far_corpus=args.far_corpus,
                                          far_val=args.far_val)
            if dist is not None:
                config['overlap_distances'][f'{overlap:g}'] = dist
            tag = 'eval' if overlap is None else 'rsweep'
            osfx = '' if overlap is None else f'_o{overlap:g}'
            bs = args.batch_size
            print('-' * 64)
            if overlap is None:
                log(f'Train: {n_train:,} tokens | Test: {n_test:,} tokens', 2)
            else:
                log(f'OVERLAP {overlap:g}  ·  {n_train:,} train tok/arm · Test {n_test:,}', 2)

            for seed in seeds:
                lbl = f's{seed}{osfx}'

                stage[0] += 1
                log(f'[stage {stage[0]}/{total_stages}] teacher (seed {seed})', 2)
                cache = train_teacher(args.teacher_steps, seed, args.eval_iters,
                                      device, lbl, cache_tag=tag, batch_size=bs,
                                      n_dct=n_dct, per_layer=args.per_layer)

                baseline_extra = ['--prism_init=False']
                if args.baseline_lr:
                    baseline_extra.append(f'--learning_rate={args.baseline_lr}')
                if args.baseline_warmup is not None:
                    baseline_extra.append(f'--warmup_iters={args.baseline_warmup}')

                stage[0] += 1
                log(f'[stage {stage[0]}/{total_stages}] baseline (seed {seed}{osfx})', 2)
                baseline = cached_stage(run_dir, f's{seed}{osfx}_baseline', lambda:
                    run_training('baseline', baseline_extra, seed,
                                 args.student_steps, args.eval_every,
                                 args.eval_iters, device, lbl, batch_size=bs))

                stage[0] += 1
                log(f'[stage {stage[0]}/{total_stages}] {args.method} (seed {seed}{osfx})', 2)
                method = cached_stage(run_dir, f's{seed}{osfx}_{args.method}', lambda:
                    run_training(args.method,
                                 method_args_for(args.method, cache,
                                                 args.method_lr, args.method_warmup,
                                                 knobs=method_knobs, seed=seed),
                                 seed, args.student_steps, args.eval_every,
                                 args.eval_iters, device, lbl, batch_size=bs))

                runs.append({
                    'seed': seed,
                    'overlap': overlap,
                    'baseline': baseline,
                    'method': method,
                    'score': compute_score(baseline, method, args.eval_every),
                })
                persist(runs, complete=False)
                log(f'  cell seed {seed}{osfx} banked ({len(runs)} total).', 2)

        artifact = persist(runs, complete=True)
        print_report(artifact)
        log(f'Artifact: results/{os.path.basename(path)} (also latest.json)', 2)
        log('COMMIT THIS FILE — it is the evidence for any claim you publish.', 2)
    finally:
        try:
            os.remove(lock)
        except Exception:
            pass


def print_report(a):
    s, c = a['summary'], a['config']
    if s and 'by_overlap' in s:
        return print_sweep_report(a)
    if s and 'by_teacher_steps' in s:
        return print_teacher_report(a)
    print()
    print('  ' + '-' * 56)
    print(f'    PRISM EVAL — {c["method"]}')
    print('  ' + '-' * 56)
    print(f'    Seeds:          {c["seeds"]}')
    print(f'    Teacher:        {c["teacher_steps"]} steps')
    print(f'    Student:        {c["student_steps"]} steps, eval every {c["eval_every"]}')
    print('  ' + '-' * 56)

    for r in a['runs']:
        sc = r['score']['prism_score']
        sc_s = f'{sc:.1f}x' if sc else 'never reached'
        cen = ' (lower bound)' if r['score']['left_censored'] else ''
        print(f'    seed {r["seed"]}: baseline {r["baseline"]["best"]:.4f} @{r["baseline"]["best_step"]:<5d}'
              f'  method {r["method"]["best"]:.4f} @{r["method"]["best_step"]:<5d}'
              f'  score {sc_s}{cen}')

    print('  ' + '-' * 56)
    ps = s['prism_score']
    if ps:
        print(f'    PRISM SCORE (median):  {ps["median"]:.1f}x   range {ps["min"]:.1f}-{ps["max"]:.1f}x'
              f'  (n={s["n_reached_baseline"]}/{s["n_seeds"]})')
    else:
        print(f'    PRISM SCORE: N/A — never reached baseline quality')
    bb, mb = s['baseline_best'], s['method_best']
    print(f'    Baseline best:  median {bb["median"]:.4f}  range {bb["min"]:.4f}-{bb["max"]:.4f}')
    print(f'    Method best:    median {mb["median"]:.4f}  range {mb["min"]:.4f}-{mb["max"]:.4f}')
    print(f'    Overfits:       method {"YES" if s["method_overfits_any"] else "no"}'
          f'  |  baseline {"YES" if s["baseline_overfits_any"] else "no"}')
    print('  ' + '-' * 56)
    if s['any_left_censored']:
        print('    NOTE: >=1 seed hit target at the first eval. Those scores are')
        print(f'          lower bounds — resolution floor is {c["eval_every"]} steps.')
    print('    The score is a ratio: read it against baseline_best. A weaker')
    print('    baseline raises the score without improving the method.')
    print(f'    Teacher and student share training data — this is same-data')
    print('    transfer and does not rule out content leakage.')
    print()


def print_sweep_report(a):
    c = a['config']
    by = a['summary']['by_overlap']
    print()
    print('  ' + '-' * 68)
    print(f'    PRISM OVERLAP SWEEP — {c["method"]}  '
          f'(teacher/student data overlap; 1.0=same-data, 0.0=disjoint)')
    print(f'    Seeds {c["seeds"]} · student {c["student_steps"]} steps, '
          f'eval every {c["eval_every"]}')
    dists = c.get('overlap_distances', {})
    print('  ' + '-' * 74)
    print(f'    {"overlap":>7} | {"tok-JS":>6} | {"baseline":>8} | {"recipe":>8} | '
          f'{"Δloss":>6} | {"score":>7} | b/m')
    print('  ' + '-' * 74)
    for ov, g in by.items():
        bb = g['baseline_best']['median'] if g['baseline_best'] else float('nan')
        mb = g['method_best']['median'] if g['method_best'] else float('nan')
        ps = g['prism_score']
        score = (f'{ps["median"]:.1f}x' + ('*' if g['any_left_censored'] else '')
                 if ps else 'never')
        js = dists.get(ov, {}).get('token_js', float('nan'))
        print(f'    {ov:>7} | {js:>6.4f} | {bb:>8.4f} | {mb:>8.4f} | {bb - mb:>6.3f} | '
              f'{score:>7} | '
              f'{"Y" if g["baseline_overfits_any"] else "n"}/'
              f'{"Y" if g["method_overfits_any"] else "n"}')
    print('  ' + '-' * 74)
    print('    Δloss = baseline_best − recipe_best (higher = recipe wins by more).')
    print('    score* = left-censored (lower bound). Read the trend down the')
    print('    overlap column: where the recipe advantage falls off as the')
    print('    teacher/student data stops overlapping is where content leakage,')
    print('    not structural transfer, was doing the work. Overlap 0.0 = the')
    print('    cross-data test: any advantage there is structural.')
    print()


def print_teacher_report(a):
    c = a['config']
    by = a['summary']['by_teacher_steps']
    print()
    print('  ' + '-' * 70)
    print(f'    PRISM TEACHER-STRENGTH SWEEP — {c["method"]}  '
          f'(does a stronger teacher help more?)')
    print(f'    Seeds {c["seeds"]} · student {c["student_steps"]} steps, '
          f'eval every {c["eval_every"]} · same-data')
    print('  ' + '-' * 70)
    print(f'    {"teacher":>8} | {"baseline":>8} | {"recipe":>8} | {"Δloss":>6} | '
          f'{"score":>7} | b/m overfit')
    print('  ' + '-' * 70)
    for ts, g in by.items():
        bb = g['baseline_best']['median'] if g['baseline_best'] else float('nan')
        mb = g['method_best']['median'] if g['method_best'] else float('nan')
        ps = g['prism_score']
        score = (f'{ps["median"]:.1f}x' + ('*' if g['any_left_censored'] else '')
                 if ps else 'never')
        print(f'    {ts + " st":>8} | {bb:>8.4f} | {mb:>8.4f} | {bb - mb:>6.3f} | '
              f'{score:>7} | '
              f'{"Y" if g["baseline_overfits_any"] else "n"}/'
              f'{"Y" if g["method_overfits_any"] else "n"}')
    print('  ' + '-' * 70)
    print('    Δloss = baseline_best − recipe_best. Read down the teacher column:')
    print('    if Δloss / score grow with teacher steps, a stronger teacher gives')
    print('    a bigger head start — a lever to push the advantage higher.')
    print('    score* = left-censored (lower bound).')
    print()


if __name__ == '__main__':
    main()
