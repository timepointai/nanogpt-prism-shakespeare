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


def setup(workdir=SRC_DIR):
    """Prepare dataset and partitions."""
    os.chdir(workdir)

    if not os.path.exists('data/shakespeare_char/train.bin'):
        log('Preparing Shakespeare char dataset (first run only)…', 2)
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


def train_teacher(steps, seed, eval_iters, device, label):
    """Train teacher model and extract fingerprint. One teacher per seed.
    Cached by the presence of directions.pt, so a resumed run skips it."""
    cache = f'.prism_cache/eval_teacher_s{seed}'
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
    ], f'{label} teacher', steps)

    if rc != 0:
        raise RuntimeError(f'Teacher training failed (seed {seed}):\n{out[-2000:]}')

    curve = parse_curve(out)
    if curve:
        log(f'teacher val loss {list(curve.values())[-1]:.4f}. Extracting '
            f'fingerprint…', 2)
    e = subprocess.run([
        sys.executable, 'prism_extract.py',
        '--ckpt', f'out-eval-teacher-s{seed}/ckpt.pt',
        '--out', cache,
    ], capture_output=True, text=True, timeout=300)
    if e.returncode != 0:
        raise RuntimeError(f'Fingerprint extraction failed:\n{e.stderr[-2000:]}')
    log(f'fingerprint saved to {cache}.', 2)

    return cache


def run_training(name, extra_args, seed, steps, eval_every, eval_iters, device, label):
    """Run one training config. Raises on failure — never scores a partial run."""
    log(f'Running {name} (seed {seed}, {steps} steps, eval every {eval_every})…', 2)
    t0 = time.time()
    rc, out = stream_train(
        [sys.executable, '-u', 'train.py', 'config/train_shakespeare_char.py',
         '--dataset=shakespeare_eval', f'--seed={seed}', f'--device={device}',
         f'--max_iters={steps}', f'--eval_interval={eval_every}',
         f'--eval_iters={eval_iters}', '--log_interval=100',
         f'--out_dir=out-eval-{name}-s{seed}',
         '--wandb_log=False', '--compile=False'] + extra_args,
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


# ---------------------------------------------------------------------------
# Resume machinery: a run key, a stale-lock guard, and a per-stage disk cache.
# ---------------------------------------------------------------------------

def run_key(config):
    """A stable, human-readable id for this experiment. Same command → same key
    → resumes; different experiment → different key → fresh run. Device is
    excluded on purpose so a run can resume on a different accelerator."""
    seeds = '-'.join(str(s) for s in config['seeds'])
    return (f"{config['method']}_t{config['teacher_steps']}"
            f"_s{config['student_steps']}_e{config['eval_every']}"
            f"_i{config['eval_iters']}_seeds{seeds}")


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def acquire_lock(run_dir):
    """Refuse to start if a live prism_eval owns this run dir; take over a stale
    lock left by a crashed/killed run. Prevents two evals racing the same
    stages while still allowing a clean resume after a hard kill."""
    lock = os.path.join(run_dir, 'lock.json')
    if os.path.exists(lock):
        try:
            info = json.load(open(lock))
        except Exception:
            info = {}
        pid = info.get('pid')
        if pid and _pid_alive(pid):
            raise SystemExit(
                f'\nAnother prism_eval is already running for this config '
                f'(pid {pid}, started {info.get("started")}).\n'
                f'If that is wrong (e.g. the process was force-killed), delete\n'
                f'  {lock}\nand run again.')
        log(f'[resume] found a stale lock (pid {pid}) — previous run did not '
            f'exit cleanly. Taking over.', 2)
    with open(lock, 'w') as f:
        json.dump({'pid': os.getpid(),
                   'started': datetime.now(timezone.utc).isoformat()}, f)
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
    total_stages = len(seeds) * 3
    done = [f for f in os.listdir(run_dir) if f.endswith('.json')
            and f not in ('meta.json', 'lock.json')]
    log(f'Run key: {key}', 2)
    log(f'Work dir: .prism_runs/{key}/  ({len(done)}/{total_stages} stages '
        f'already banked)', 2)

    lock = acquire_lock(run_dir)
    try:
        n_train, n_test = setup()
        log(f'Train: {n_train:,} tokens | Test: {n_test:,} tokens', 2)
        log(f'Seeds: {seeds} | Method: {args.method} | Device: {device}', 2)
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

        def banner(seed, i):
            print('-' * 64)
            log(f'SEED {seed}  ({i + 1}/{len(seeds)})', 2)

        runs = []
        for i, seed in enumerate(seeds):
            banner(seed, i)
            lbl = f's{seed}'

            stage[0] += 1
            log(f'[stage {stage[0]}/{total_stages}] teacher', 2)
            cache = train_teacher(args.teacher_steps, seed, args.eval_iters,
                                  device, lbl)

            stage[0] += 1
            log(f'[stage {stage[0]}/{total_stages}] baseline', 2)
            baseline = cached_stage(run_dir, f's{seed}_baseline', lambda:
                run_training('baseline', ['--prism_init=False'], seed,
                             args.student_steps, args.eval_every,
                             args.eval_iters, device, lbl))

            stage[0] += 1
            log(f'[stage {stage[0]}/{total_stages}] {args.method}', 2)
            method = cached_stage(run_dir, f's{seed}_{args.method}', lambda:
                run_training(args.method, method_args_for(args.method, cache),
                             seed, args.student_steps, args.eval_every,
                             args.eval_iters, device, lbl))

            runs.append({
                'seed': seed,
                'baseline': baseline,
                'method': method,
                'score': compute_score(baseline, method, args.eval_every),
            })
            persist(runs, complete=(i + 1 == len(seeds)))
            log(f'seed {seed} banked → results/{os.path.basename(path)} '
                f'({len(runs)}/{len(seeds)} seeds).', 2)

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
