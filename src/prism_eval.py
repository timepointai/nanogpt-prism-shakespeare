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

# Generous: an A100 finishes a 5000-step run in minutes, but MPS/CPU take hours
# and a timeout here kills the run outright. Sized for the slow case.
TRAIN_TIMEOUT = 24 * 3600


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


def setup(workdir=SRC_DIR):
    """Prepare dataset and partitions."""
    os.chdir(workdir)

    if not os.path.exists('data/shakespeare_char/train.bin'):
        subprocess.run([sys.executable, 'data/shakespeare_char/prepare.py'],
                       capture_output=True, check=True)

    train_all = np.array(np.memmap('data/shakespeare_char/train.bin',
                                    dtype=np.uint16, mode='r'))
    test_data = np.array(np.memmap('data/shakespeare_char/val.bin',
                                    dtype=np.uint16, mode='r'))

    split = int(len(train_all) * 0.80)
    train_data = train_all[:split].astype(np.uint16)
    teacher_val = train_all[split:].astype(np.uint16)

    # Both datasets share train.bin: teacher and student train on the same 80%.
    # Only the val stream differs (teacher tunes against the held-out 20%; the
    # student is scored on the original Shakespeare val, never trained on).
    for name, val in [('shakespeare_eval', test_data),
                       ('shakespeare_teacher', teacher_val)]:
        d = f'data/{name}'
        os.makedirs(d, exist_ok=True)
        train_data.tofile(os.path.join(d, 'train.bin'))
        val.tofile(os.path.join(d, 'val.bin'))
        shutil.copy('data/shakespeare_char/meta.pkl',
                     os.path.join(d, 'meta.pkl'))

    return len(train_data), len(test_data)


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


def train_teacher(steps, seed, eval_iters, device):
    """Train teacher model and extract fingerprint. One teacher per seed."""
    cache = f'.prism_cache/eval_teacher_s{seed}'
    if os.path.exists(f'{cache}/directions.pt'):
        print(f'  Teacher (seed {seed}) cached.')
        return cache

    print(f'  Training teacher (seed {seed}, {steps} steps)...')
    t0 = time.time()
    r = subprocess.run([
        sys.executable, 'train.py', 'config/train_shakespeare_char.py',
        '--dataset=shakespeare_teacher', f'--seed={seed}', f'--device={device}',
        f'--max_iters={steps}', f'--eval_interval={steps}',
        f'--eval_iters={eval_iters}', f'--log_interval={steps}',
        f'--out_dir=out-eval-teacher-s{seed}',
        '--always_save_checkpoint=True',
        '--compile=False', '--prism_init=False', '--wandb_log=False',
    ], capture_output=True, text=True, timeout=TRAIN_TIMEOUT)

    if r.returncode != 0:
        raise RuntimeError(f'Teacher training failed (seed {seed}):\n{r.stderr[-2000:]}')

    curve = parse_curve(r.stdout)
    if curve:
        print(f'  Teacher val loss: {list(curve.values())[-1]:.4f} '
              f'({time.time() - t0:.0f}s)')

    print(f'  Extracting fingerprint...')
    e = subprocess.run([
        sys.executable, 'prism_extract.py',
        '--ckpt', f'out-eval-teacher-s{seed}/ckpt.pt',
        '--out', cache,
    ], capture_output=True, text=True, timeout=300)
    if e.returncode != 0:
        raise RuntimeError(f'Fingerprint extraction failed:\n{e.stderr[-2000:]}')

    return cache


def run_training(name, extra_args, seed, steps, eval_every, eval_iters, device):
    """Run one training config. Raises on failure — never scores a partial run."""
    print(f'  Running {name} (seed {seed})...')
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, 'train.py', 'config/train_shakespeare_char.py',
         '--dataset=shakespeare_eval', f'--seed={seed}', f'--device={device}',
         f'--max_iters={steps}', f'--eval_interval={eval_every}',
         f'--eval_iters={eval_iters}', '--log_interval=500',
         f'--out_dir=out-eval-{name}-s{seed}',
         '--wandb_log=False', '--compile=False'] + extra_args,
        capture_output=True, text=True, timeout=TRAIN_TIMEOUT
    )
    wall = time.time() - t0

    if r.returncode != 0:
        raise RuntimeError(f'Training failed ({name}, seed {seed}):\n{r.stderr[-2000:]}')

    curve = parse_curve(r.stdout)
    if not curve:
        raise RuntimeError(f'No eval lines parsed ({name}, seed {seed}). '
                           f'stdout tail:\n{r.stdout[-1000:]}')
    if max(curve) < steps:
        raise RuntimeError(f'Run truncated ({name}, seed {seed}): last eval at '
                           f'step {max(curve)}, expected {steps}. Refusing to '
                           f'score a partial curve.')

    best = min(curve.values())
    best_step = min(curve, key=curve.get)
    at_end = curve[max(curve)]
    print(f'  {name}: best={best:.4f} @{best_step}, @{steps}={at_end:.4f}, {wall:.0f}s')

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


def method_args_for(method, cache):
    common = [
        '--prism_init=True',
        f'--prism_spectra={cache}/spectra.json',
        '--learning_rate=5e-4', '--warmup_iters=50',
    ]
    dirs = [f'--prism_directions={cache}/directions.pt']
    return {
        'recipe':        common + ['--prism_align=0.75'] + dirs + ['--prism_mod=0.01', '--prism_mod_decay=0.9999'],
        'marathon':      common + ['--prism_align=0.75'] + dirs + ['--prism_mod=0.01', '--prism_mod_decay=0.9999'],
        'sprint':        common + ['--prism_align=0.75'] + dirs + ['--prism_mod=0.005', '--prism_mod_decay=0.999'],
        'spectral_only': common + ['--prism_align=0.0', '--prism_mod=0.01', '--prism_mod_decay=0.9999'],
        'dirs_only':     common + ['--prism_align=0.75'] + dirs,
    }[method]


def summarize(runs):
    scores = [r['score']['prism_score'] for r in runs
              if r['score']['prism_score'] is not None]
    censored = any(r['score']['left_censored'] for r in runs)

    def stats(vals):
        if not vals:
            return None
        s = sorted(vals)
        return {'median': s[len(s) // 2], 'min': s[0], 'max': s[-1],
                'values': [round(v, 4) for v in vals]}

    return {
        'n_seeds': len(runs),
        'n_reached_baseline': len(scores),
        'prism_score': stats(scores),
        'any_left_censored': censored,
        'baseline_best': stats([r['baseline']['best'] for r in runs]),
        'method_best': stats([r['method']['best'] for r in runs]),
        'method_overfits_any': any(r['method']['overfits'] for r in runs),
        'baseline_overfits_any': any(r['baseline']['overfits'] for r in runs),
    }


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

    print('=' * 60)
    print('  PRISM EVAL')
    print('=' * 60)
    n_train, n_test = setup()
    print(f'  Train: {n_train:,} tokens | Test: {n_test:,} tokens')
    print(f'  Seeds: {seeds} | Method: {args.method} | Device: {device}')
    if len(seeds) == 1:
        print('  WARNING: single seed. Reports one sample, not a result.')

    config = {
        'method': args.method,
        'teacher_steps': args.teacher_steps,
        'student_steps': args.student_steps,
        'eval_every': args.eval_every,
        'eval_iters': args.eval_iters,
        'seeds': seeds,
        'device': device,
        'teacher_data_equals_student_data': True,
    }

    def build(runs, complete):
        return {
            'schema': 'prism-eval/1',
            'partial': not complete,
            'seeds_requested': seeds,
            'seeds_done': [r['seed'] for r in runs],
            'provenance': provenance(),
            'config': config,
            'runs': runs,
            'summary': summarize(runs) if runs else None,
        }

    # One timestamp for the whole run so every incremental write lands on the
    # same file — the free-tier GPU can drop at any hour, and a run that dies
    # after seed 1 must still leave seed 1 on disk (that is how the originals
    # were lost). latest.json always points at the newest state.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = os.path.join(RESULTS_DIR, f'{args.method}_{stamp}.json')

    def persist(runs, complete):
        art = build(runs, complete)
        for f in (path, latest):
            with open(f, 'w') as fh:
                json.dump(art, fh, indent=2)
        return art

    runs = []
    for i, seed in enumerate(seeds):
        print(f'\n--- seed {seed} ({i + 1}/{len(seeds)}) ---')
        cache = train_teacher(args.teacher_steps, seed, args.eval_iters, device)
        baseline = run_training('baseline', ['--prism_init=False'], seed,
                                args.student_steps, args.eval_every,
                                args.eval_iters, device)
        method = run_training(args.method, method_args_for(args.method, cache), seed,
                              args.student_steps, args.eval_every,
                              args.eval_iters, device)
        runs.append({
            'seed': seed,
            'baseline': baseline,
            'method': method,
            'score': compute_score(baseline, method, args.eval_every),
        })
        persist(runs, complete=(i + 1 == len(seeds)))
        print(f'  Seed {seed} done and saved. {len(runs)}/{len(seeds)} seeds in '
              f'results/{os.path.basename(path)}.')

    artifact = persist(runs, complete=True)
    print_report(artifact)
    print(f'\n  Artifact: results/{os.path.basename(path)} (also latest.json)')
    print(f'  COMMIT THIS FILE — it is the evidence for any claim you publish.')


def print_report(a):
    s, c = a['summary'], a['config']
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


if __name__ == '__main__':
    main()
