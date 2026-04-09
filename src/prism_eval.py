"""
prism_eval.py — Standardized Prism benchmark.

Produces one number: the Prism Score = baseline_steps / prism_steps
to reach baseline's best val loss on a held-out test set.

Anyone can run this to reproduce or compare init methods.

Usage:
    python prism_eval.py                          # full eval (baseline + recipe)
    python prism_eval.py --method=spectral_only   # test spectral shape only
    python prism_eval.py --teacher_steps=500      # cheaper teacher
    python prism_eval.py --report                 # just print cached results

Protocol:
    1. Prepare Shakespeare char-level dataset
    2. Split: Train (80%) / Teacher-Val (20%) / Test (original val)
    3. Train teacher on Train for --teacher_steps steps
    4. Extract spectral fingerprint from teacher
    5. Train baseline on Train, eval on Test, record steps-to-best
    6. Train Prism student on Train, eval on Test, record steps-to-target
    7. Prism Score = baseline_steps / prism_steps

The Prism Score measures marginal cost reduction. Higher = better.
    1.0  = no benefit (same as baseline)
    13.0 = 13x fewer steps to same quality (current recipe result)
    inf  = instant convergence (theoretical maximum, unreachable)
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np


def setup(workdir='.'):
    """Prepare dataset and partitions."""
    os.chdir(workdir)

    # Prepare Shakespeare if needed
    if not os.path.exists('data/shakespeare_char/train.bin'):
        subprocess.run(['python', 'data/shakespeare_char/prepare.py'],
                       capture_output=True, check=True)

    # Create strict partition
    train_all = np.array(np.memmap('data/shakespeare_char/train.bin',
                                    dtype=np.uint16, mode='r'))
    test_data = np.array(np.memmap('data/shakespeare_char/val.bin',
                                    dtype=np.uint16, mode='r'))

    split = int(len(train_all) * 0.80)
    train_data = train_all[:split].astype(np.uint16)
    teacher_val = train_all[split:].astype(np.uint16)

    for name, val in [('shakespeare_eval', test_data),
                       ('shakespeare_teacher', teacher_val)]:
        d = f'data/{name}'
        os.makedirs(d, exist_ok=True)
        train_data.tofile(os.path.join(d, 'train.bin'))
        val.tofile(os.path.join(d, 'val.bin'))
        shutil.copy('data/shakespeare_char/meta.pkl',
                     os.path.join(d, 'meta.pkl'))

    return len(train_data), len(test_data)


def train_teacher(steps=2000):
    """Train teacher model and extract fingerprint."""
    cache = '.prism_cache/eval_teacher'
    if os.path.exists(f'{cache}/directions.pt'):
        print(f'  Teacher cached.')
        return cache

    print(f'  Training teacher ({steps} steps)...')
    t0 = time.time()
    r = subprocess.run([
        'python', 'train.py', 'config/train_shakespeare_char.py',
        '--dataset=shakespeare_teacher',
        f'--max_iters={steps}', f'--eval_interval={steps}',
        '--eval_iters=50', f'--log_interval={steps}',
        '--out_dir=out-eval-teacher',
        '--always_save_checkpoint=True',
        '--compile=False', '--prism_init=False', '--wandb_log=False',
    ], capture_output=True, text=True, timeout=600)

    teacher_time = time.time() - t0

    # Show teacher quality
    for line in r.stdout.split('\n'):
        m = re.search(r'step \d+: train loss ([\d.]+), val loss ([\d.]+)', line)
        if m:
            print(f'  Teacher val loss: {m.group(2)} ({teacher_time:.0f}s)')

    print(f'  Extracting fingerprint...')
    subprocess.run([
        'python', 'prism_extract.py',
        '--ckpt', 'out-eval-teacher/ckpt.pt',
        '--out', cache,
    ], capture_output=True, timeout=120)

    return cache


def run_training(name, extra_args, steps=5000, eval_every=100):
    """Run a training config, return val loss dict."""
    print(f'  Running {name}...')
    t0 = time.time()
    r = subprocess.run(
        ['python', 'train.py', 'config/train_shakespeare_char.py',
         '--dataset=shakespeare_eval',
         f'--max_iters={steps}', f'--eval_interval={eval_every}',
         '--eval_iters=50', '--log_interval=500',
         f'--out_dir=out-eval-{name}',
         '--wandb_log=False', '--compile=False'] + extra_args,
        capture_output=True, text=True, timeout=1200
    )
    wall = time.time() - t0

    val = {}
    for line in r.stdout.split('\n'):
        m = re.search(r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)', line)
        if m:
            val[int(m.group(1))] = float(m.group(3))

    if r.returncode != 0:
        print(f'  ERROR: {r.stderr[-300:]}')

    best = min(val.values()) if val else 999
    best_s = min(val, key=val.get) if val else 0
    at_end = val.get(steps, val.get(max(val.keys()), 0)) if val else 0
    print(f'  {name}: best={best:.4f} @{best_s}, @{steps}={at_end:.4f}, {wall:.0f}s')

    return val, wall


def compute_score(baseline_val, method_val):
    """Compute Prism Score = baseline_steps / method_steps to reach baseline best."""
    bb = min(baseline_val.values())
    bs = min(baseline_val, key=baseline_val.get)

    hit = next((s for s in sorted(method_val.keys())
                if method_val[s] <= bb), None)

    if hit is None:
        return 0.0, bb, bs, None  # method never reached baseline quality

    score = bs / hit
    return score, bb, bs, hit


def main():
    parser = argparse.ArgumentParser(description='Prism Eval — standardized benchmark')
    parser.add_argument('--teacher_steps', type=int, default=2000)
    parser.add_argument('--student_steps', type=int, default=5000)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--method', type=str, default='recipe',
                        choices=['recipe', 'spectral_only', 'dirs_only',
                                 'sprint', 'marathon'])
    parser.add_argument('--report', action='store_true',
                        help='Just print cached results')
    args = parser.parse_args()

    results_path = 'prism_eval_results.json'

    if args.report:
        if os.path.exists(results_path):
            results = json.load(open(results_path))
            print_report(results)
        else:
            print('No cached results. Run eval first.')
        return

    # Setup
    print('='*60)
    print('  PRISM EVAL')
    print('='*60)
    n_train, n_test = setup()
    print(f'  Train: {n_train:,} tokens | Test: {n_test:,} tokens')

    # Teacher
    cache = train_teacher(args.teacher_steps)

    # Baseline
    baseline_val, baseline_wall = run_training(
        'baseline', ['--prism_init=False'],
        steps=args.student_steps, eval_every=args.eval_every
    )

    # Method
    if args.method == 'recipe':
        method_args = [
            '--prism_init=True', '--prism_align=0.75',
            f'--prism_spectra={cache}/spectra.json',
            f'--prism_directions={cache}/directions.pt',
            '--learning_rate=5e-4', '--warmup_iters=50',
            '--prism_mod=0.01', '--prism_mod_decay=0.9999',
        ]
    elif args.method == 'spectral_only':
        method_args = [
            '--prism_init=True', '--prism_align=0.0',
            f'--prism_spectra={cache}/spectra.json',
            '--learning_rate=5e-4', '--warmup_iters=50',
            '--prism_mod=0.01', '--prism_mod_decay=0.9999',
        ]
    elif args.method == 'dirs_only':
        method_args = [
            '--prism_init=True', '--prism_align=0.75',
            f'--prism_spectra={cache}/spectra.json',
            f'--prism_directions={cache}/directions.pt',
            '--learning_rate=5e-4', '--warmup_iters=50',
        ]
    elif args.method == 'sprint':
        method_args = [
            '--prism_init=True', '--prism_align=0.75',
            f'--prism_spectra={cache}/spectra.json',
            f'--prism_directions={cache}/directions.pt',
            '--learning_rate=5e-4', '--warmup_iters=50',
            '--prism_mod=0.005', '--prism_mod_decay=0.999',
        ]
    elif args.method == 'marathon':
        method_args = [
            '--prism_init=True', '--prism_align=0.75',
            f'--prism_spectra={cache}/spectra.json',
            f'--prism_directions={cache}/directions.pt',
            '--learning_rate=5e-4', '--warmup_iters=50',
            '--prism_mod=0.01', '--prism_mod_decay=0.9999',
        ]

    method_val, method_wall = run_training(
        args.method, method_args,
        steps=args.student_steps, eval_every=args.eval_every
    )

    # Score
    score, bb, bs, hit = compute_score(baseline_val, method_val)
    method_best = min(method_val.values())
    method_best_step = min(method_val, key=method_val.get)
    baseline_at_end = baseline_val.get(args.student_steps,
                                        baseline_val.get(max(baseline_val.keys()), 0))
    method_at_end = method_val.get(args.student_steps,
                                    method_val.get(max(method_val.keys()), 0))

    results = {
        'method': args.method,
        'teacher_steps': args.teacher_steps,
        'student_steps': args.student_steps,
        'prism_score': score,
        'baseline_best': bb,
        'baseline_best_step': bs,
        'baseline_at_end': baseline_at_end,
        'method_best': method_best,
        'method_best_step': method_best_step,
        'method_hit_step': hit,
        'method_at_end': method_at_end,
        'method_overfits': method_at_end > method_best * 1.05,
        'baseline_wall': baseline_wall,
        'method_wall': method_wall,
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print_report(results)


def print_report(r):
    print()
    print('  ┌─────────────────────────────────────────┐')
    print('  │          PRISM EVAL RESULTS              │')
    print('  ├─────────────────────────────────────────┤')
    print(f'  │  Method: {r["method"]:>30s} │')
    print(f'  │  Teacher: {r["teacher_steps"]:>5d} steps                   │')
    print('  ├─────────────────────────────────────────┤')
    print(f'  │  Baseline best:  {r["baseline_best"]:.4f} @ step {r["baseline_best_step"]:>5d}  │')
    print(f'  │  Method best:    {r["method_best"]:.4f} @ step {r["method_best_step"]:>5d}  │')
    print(f'  │  Method hit:     step {str(r["method_hit_step"] or "never"):>5s}              │')
    print(f'  │  Overfitting:    {"YES" if r["method_overfits"] else "no":>5s}              │')
    print('  ├─────────────────────────────────────────┤')
    score = r["prism_score"]
    if score > 0:
        print(f'  │                                         │')
        print(f'  │  PRISM SCORE:  {score:>6.1f}x                   │')
        print(f'  │                                         │')
    else:
        print(f'  │  PRISM SCORE:  N/A (never reached base) │')
    print('  └─────────────────────────────────────────┘')
    print()
    print(f'  Prism Score = baseline_steps_to_best / method_steps_to_best')
    print(f'  Higher = better. 1.0 = no benefit. 13.0 = current recipe.')
    print(f'  Theoretical max: the Kolmogorov limit of the init prior.')


if __name__ == '__main__':
    main()
