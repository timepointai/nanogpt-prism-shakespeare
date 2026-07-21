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
    # stop sharing data? Wide + shallow (Prism acts in the first ~200-300 steps):
    python prism_eval.py --method_lr=1e-3 --method_warmup=100 \
        --teacher_steps=1000 --student_steps=1000 --eval_every=50 \
        --overlap=1.0,0.75,0.5,0.25,0.0
    # overlap 1.0 = same-data; 0.0 = disjoint (the cross-data / structural test).

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


def setup(workdir=SRC_DIR, overlap=None):
    """Prepare dataset partitions.

    overlap=None (default): teacher and student share the same 80% split —
    same-data transfer (the original protocol).

    overlap in [0,1]: the teacher/student data-overlap DIAL. Each arm gets a
    fixed-size window of exactly HALF the train pool, so the data BUDGET is
    constant across the sweep — only the shared fraction changes, never the
    amount of data (that would be a second confound). The teacher always trains
    on pool[0:W]; the student window slides so its overlap with the teacher's is
    exactly `overlap`:
        overlap=1.0 → identical window (same-data, on half the pool)
        overlap=0.0 → disjoint windows (cross-data: nothing shared)
    The student is always SCORED on the original Shakespeare val set, which is
    held out of every window regardless of overlap."""
    os.chdir(workdir)

    if not os.path.exists('data/shakespeare_char/train.bin'):
        log('Preparing Shakespeare char dataset (first run only)…', 2)
        subprocess.run([sys.executable, 'data/shakespeare_char/prepare.py'],
                       capture_output=True, check=True)

    pool = np.array(np.memmap('data/shakespeare_char/train.bin',
                              dtype=np.uint16, mode='r'))
    test_data = np.array(np.memmap('data/shakespeare_char/val.bin',
                                    dtype=np.uint16, mode='r'))

    if overlap is None:
        split = int(len(pool) * 0.80)
        teacher_train = pool[:split].astype(np.uint16)
        student_train = pool[:split].astype(np.uint16)
        teacher_val = pool[split:].astype(np.uint16)
    else:
        overlap = max(0.0, min(1.0, float(overlap)))
        n = len(pool)
        w = n // 2                                   # constant per-arm budget
        start = int(round(w * (1.0 - overlap)))      # slide the student window
        teacher_train = pool[0:w].astype(np.uint16)
        student_train = pool[start:start + w].astype(np.uint16)
        teacher_val = test_data                      # teacher's own eval only
        log(f'overlap={overlap:.2f}: teacher pool[0:{w}] · student '
            f'pool[{start}:{start + w}] · shared {max(0, w - start):,} tok '
            f'({max(0, w - start) / w:.0%} of each window)', 2)

    for name, train, val in [('shakespeare_eval', student_train, test_data),
                              ('shakespeare_teacher', teacher_train, teacher_val)]:
        d = f'data/{name}'
        os.makedirs(d, exist_ok=True)
        train.astype(np.uint16).tofile(os.path.join(d, 'train.bin'))
        val.astype(np.uint16).tofile(os.path.join(d, 'val.bin'))
        shutil.copy('data/shakespeare_char/meta.pkl',
                     os.path.join(d, 'meta.pkl'))

    return len(student_train), len(test_data)


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


def train_teacher(steps, seed, eval_iters, device, label, cache_tag='eval'):
    """Train teacher model and extract fingerprint. One teacher per (seed, data,
    steps). Cached by the presence of directions.pt, so a resumed run skips it.
    cache_tag separates teachers trained on different data (e.g. the same-data
    'eval' teacher vs. a 'sweep' teacher on pool[0:W]); the step count is in the
    path too, so changing teacher_steps can't silently reuse a stale fingerprint."""
    cache = f'.prism_cache/{cache_tag}_teacher_s{seed}_t{steps}'
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


def method_args_for(method, cache, lr, warmup):
    common = [
        '--prism_init=True',
        f'--prism_spectra={cache}/spectra.json',
        f'--learning_rate={lr}', f'--warmup_iters={warmup}',
    ]
    dirs = [f'--prism_directions={cache}/directions.pt']
    return {
        'recipe':        common + ['--prism_align=0.75'] + dirs + ['--prism_mod=0.01', '--prism_mod_decay=0.9999'],
        'marathon':      common + ['--prism_align=0.75'] + dirs + ['--prism_mod=0.01', '--prism_mod_decay=0.9999'],
        'sprint':        common + ['--prism_align=0.75'] + dirs + ['--prism_mod=0.005', '--prism_mod_decay=0.999'],
        'spectral_only': common + ['--prism_align=0.0', '--prism_mod=0.01', '--prism_mod_decay=0.9999'],
        'dirs_only':     common + ['--prism_align=0.75'] + dirs,
    }[method]


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
    """Flat summary for a single-condition run; a per-overlap breakdown when the
    run is an overlap sweep (so the artifact shows the 7× vs. overlap curve)."""
    if not runs:
        return None
    if all(r.get('overlap') is None for r in runs):
        return _summ_group(runs)
    by = {}
    for r in runs:
        by.setdefault(r['overlap'], []).append(r)
    return {'by_overlap': {f'{o:g}': _summ_group(g)
                           for o, g in sorted(by.items(), reverse=True)}}


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

    config = {
        'method': args.method,
        'teacher_steps': args.teacher_steps,
        'student_steps': args.student_steps,
        'eval_every': args.eval_every,
        'eval_iters': args.eval_iters,
        'seeds': seeds,
        'device': device,
        'overlaps': overlaps,
        'teacher_data_equals_student_data': overlaps is None,
        'method_lr': args.method_lr,
        'method_warmup': args.method_warmup,
        'baseline_lr': args.baseline_lr,
        'baseline_warmup': args.baseline_warmup,
        'schedule_matched': (args.method_lr == (args.baseline_lr or '1e-3')
                             and args.method_warmup == (args.baseline_warmup or 100)),
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
        sweep = overlaps if overlaps is not None else [None]

        runs = []
        for overlap in sweep:
            n_train, n_test = setup(overlap=overlap)
            tag = 'eval' if overlap is None else 'sweep'
            osfx = '' if overlap is None else f'_o{overlap:g}'
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
                                      device, lbl, cache_tag=tag)

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
                                 args.eval_iters, device, lbl))

                stage[0] += 1
                log(f'[stage {stage[0]}/{total_stages}] {args.method} (seed {seed}{osfx})', 2)
                method = cached_stage(run_dir, f's{seed}{osfx}_{args.method}', lambda:
                    run_training(args.method,
                                 method_args_for(args.method, cache,
                                                 args.method_lr, args.method_warmup),
                                 seed, args.student_steps, args.eval_every,
                                 args.eval_iters, device, lbl))

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
    print('  ' + '-' * 68)
    print(f'    {"overlap":>7} | {"baseline":>8} | {"recipe":>8} | {"Δloss":>6} | '
          f'{"score":>7} | overfit b/m')
    print('  ' + '-' * 68)
    for ov, g in by.items():
        bb = g['baseline_best']['median'] if g['baseline_best'] else float('nan')
        mb = g['method_best']['median'] if g['method_best'] else float('nan')
        ps = g['prism_score']
        score = (f'{ps["median"]:.1f}x' + ('*' if g['any_left_censored'] else '')
                 if ps else 'n/a')
        print(f'    {ov:>7} | {bb:>8.4f} | {mb:>8.4f} | {bb - mb:>6.3f} | '
              f'{score:>7} | '
              f'{"Y" if g["baseline_overfits_any"] else "n"}/'
              f'{"Y" if g["method_overfits_any"] else "n"}')
    print('  ' + '-' * 68)
    print('    Δloss = baseline_best − recipe_best (higher = recipe wins by more).')
    print('    score* = left-censored (lower bound). Read the trend down the')
    print('    overlap column: where the recipe advantage falls off as the')
    print('    teacher/student data stops overlapping is where content leakage,')
    print('    not structural transfer, was doing the work. Overlap 0.0 = the')
    print('    cross-data test: any advantage there is structural.')
    print()


if __name__ == '__main__':
    main()
