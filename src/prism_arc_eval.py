"""
prism_arc_eval.py — PRISM × finetune, the "unified arc" base-interaction study (Round 0).

The question: PRISM's two proven results are the same mod-wheel operation with a
different target — pretraining transfers the SPECTRUM (structure), finetuning pins the
DIRECTIONS (content). Does that interaction actually help, or is "PRISM + finetuning"
just the two effects run in sequence with no synergy?

The sharpest, cheapest test: does a PRISM-pretrained base make a BETTER finetune-anchor
than a plain-pretrained base, at MATCHED old-domain quality? If PRISM's healthier,
non-overfit geometry is a better thing to anchor, base_prism should retain/adapt better
than base_plain even when both start the finetune at the same Shakespeare val loss.

Protocol (per seed):
  1. Train two Shakespeare bases to the SAME val target (--base_target_val, via
     train.py --stop_val_target — the matched-quality control):
       base_plain  — plain from scratch
       base_prism  — PRISM-accelerated (spectral init from a teacher + mod wheel)
     PRISM reaches the target in far fewer steps; both stop at ≈ the same A-val.
  2. Finetune each base on Sherlock with the IDENTICAL raw self-anchor (prism_mod=0.01,
     anchor_mode=raw), scored every step on both Sherlock (adaptation) and Shakespeare
     (retention, val2). Only the BASE differs — single variable.
  3. Compare forgetting + adaptation of the two finetunes at matched base quality.

Read:
  forget_ratio (plain/prism) > 1 with matched base_val  → PRISM base is a better anchor
    (genuine PRISM×finetune synergy — double down).
  forget_ratio ≈ 1                                       → the anchor is base-agnostic
    (honest negative; the value of "combining" is compounding + continual, not synergy).

Every run writes a prism-arc/1 artifact to results/. Stepwise-resumable; partial runs raise.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from prism_eval import (provenance, parse_curve, stream_train, acquire_lock, _stats,
                        log, default_device, RESULTS_DIR, RUNS_DIR)
from prism_finetune_eval import parse_retain, setup_ft, size_args, SHAKE, CONFIG

TEACHER = '.prism_cache/arc_teacher'


def train_teacher_arc(seed, a, device):
    """Plain Shakespeare teacher (gives the PRISM base its spectral fingerprint).
    Cached by the presence of directions.pt."""
    out = f'out-arc-teacher-s{seed}'
    cache = f'{TEACHER}_s{seed}'
    if os.path.exists(f'{cache}/directions.pt'):
        log(f'[resume] arc teacher (seed {seed}) fingerprint cached.', 2)
        return cache
    log(f'Training arc teacher (seed {seed}, {a.teacher_steps} steps)…', 2)
    rc, out_s = stream_train(
        [sys.executable, '-u', 'train.py', CONFIG, '--dataset=shakespeare_char',
         f'--seed={seed}', f'--device={device}', f'--max_iters={a.teacher_steps}',
         f'--eval_interval={a.teacher_steps}', f'--eval_iters={a.eval_iters}',
         '--log_interval=100', f'--out_dir={out}', '--always_save_checkpoint=True',
         '--compile=False', '--prism_init=False', '--wandb_log=False'] + size_args(a),
        f's{seed} teacher', a.teacher_steps)
    if rc != 0:
        raise RuntimeError(f'arc teacher failed (seed {seed}):\n{out_s[-2000:]}')
    e = subprocess.run([sys.executable, 'prism_extract.py', '--ckpt', f'{out}/ckpt.pt',
                        '--out', cache], capture_output=True, text=True, timeout=300)
    if e.returncode != 0:
        raise RuntimeError(f'fingerprint extraction failed:\n{e.stderr[-2000:]}')
    log(f'arc teacher fingerprint → {cache}.', 2)
    return cache


def train_base(method, seed, a, device, run_dir):
    """Train a Shakespeare base to the matched val target. method: 'plain' | 'prism'."""
    out = f'out-arc-{method}-s{seed}'
    ckpt = f'{out}/ckpt.pt'
    meta_f = os.path.join(run_dir, f's{seed}_{method}_base.json')
    if os.path.exists(ckpt) and os.path.exists(meta_f):
        r = json.load(open(meta_f))
        log(f'[resume] {method} base (seed {seed}) cached (val {r["val"]:.4f} @{r["steps"]}).', 2)
        return r

    cmd = [sys.executable, '-u', 'train.py', CONFIG, '--dataset=shakespeare_char',
           f'--seed={seed}', f'--device={device}', f'--max_iters={a.base_max_iters}',
           f'--eval_interval={a.base_eval_every}', f'--eval_iters={a.eval_iters}',
           '--log_interval=100', f'--out_dir={out}', '--always_save_checkpoint=True',
           '--compile=False', '--wandb_log=False',
           f'--stop_val_target={a.base_target_val}'] + size_args(a)
    if method == 'prism':
        cache = train_teacher_arc(seed, a, device)
        cmd += ['--prism_init=True', '--prism_align=0.75',
                f'--prism_spectra={cache}/spectra.json',
                f'--prism_directions={cache}/directions.pt',
                '--prism_mod=0.01', '--prism_mod_decay=0.9999',
                f'--learning_rate={a.prism_lr}', f'--warmup_iters={a.prism_warmup}']
    elif method == 'plain_fastlr':
        # attribution control: plain (NO spectral) but at PRISM's schedule (LR/warmup),
        # so prism-vs-plain_fastlr isolates the spectral geometry from the schedule.
        cmd += [f'--learning_rate={a.prism_lr}', f'--warmup_iters={a.prism_warmup}']
    # plain uses the config schedule (LR 1e-3, warmup 100); no prism flags.

    log(f'Training {method} base (seed {seed}) → val target {a.base_target_val}…', 2)
    rc, out_s = stream_train(cmd, f's{seed} {method} base', a.base_max_iters)
    if rc != 0:
        raise RuntimeError(f'{method} base failed (seed {seed}):\n{out_s[-2000:]}')
    curve = parse_curve(out_s)
    if not curve:
        raise RuntimeError(f'no eval lines for {method} base:\n{out_s[-1000:]}')
    stop_step = max(curve)
    val = curve[stop_step]
    r = {'method': method, 'ckpt': ckpt, 'val': val, 'steps': stop_step,
         'reached_target': val <= a.base_target_val,
         'curve': {str(k): v for k, v in sorted(curve.items())}}
    json.dump(r, open(meta_f, 'w'), indent=2)
    return r


def finetune(base, seed, a, device, run_dir):
    """Finetune a base on Sherlock with the raw self-anchor; dual-val (adapt + retain)."""
    method = base['method']
    out = f'out-arc-ft-{method}-s{seed}'
    meta_f = os.path.join(run_dir, f's{seed}_{method}_ft.json')
    if os.path.exists(meta_f):
        r = json.load(open(meta_f))
        log(f'[resume] {method} finetune (seed {seed}) cached '
            f'(forget {r["forgetting"]:+.3f}).', 2)
        return r

    bs = base['steps']
    last = bs + a.ft_steps
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    shutil.copy(base['ckpt'], f'{out}/ckpt.pt')          # fork the base

    cmd = [sys.executable, '-u', 'train.py', CONFIG, f'--seed={seed}',
           f'--device={device}', f'--eval_interval={a.eval_every}',
           f'--eval_iters={a.eval_iters}', '--log_interval=100', f'--out_dir={out}',
           '--compile=False', '--wandb_log=False', '--dataset=sherlock_ft',
           '--init_from=resume', f'--val2_dir={SHAKE}', f'--max_iters={last}',
           f'--warmup_iters={bs + a.ft_warmup}', f'--lr_decay_iters={last}',
           f'--learning_rate={a.ft_lr}', f'--min_lr={a.ft_min_lr}', '--decay_lr=True',
           '--prism_mod=0.01', '--prism_mod_decay=1.0', '--prism_anchor_mode=raw'] + size_args(a)

    log(f'Finetuning {method} base (seed {seed}) → step {last}…', 2)
    rc, out_s = stream_train(cmd, f's{seed} {method} ft', last)
    if rc != 0:
        raise RuntimeError(f'{method} finetune failed (seed {seed}):\n{out_s[-2000:]}')
    adapt = parse_curve(out_s)
    retain = parse_retain(out_s)
    if not adapt or max(adapt) < last:
        raise RuntimeError(f'{method} finetune (seed {seed}) truncated: '
                           f'last {max(adapt) if adapt else None}, expected {last}.')
    if bs not in retain or max(retain) < last:
        raise RuntimeError(f'{method} finetune (seed {seed}) missing retention val2 at '
                           f'base step {bs} or end. Got {sorted(retain)}.')
    r = {'method': method,
         'adapt_best': min(adapt.values()),
         'adapt_at_end': adapt[max(adapt)],
         'adapt_overfits': adapt[max(adapt)] > min(adapt.values()) * 1.05,
         'retain_at_base': retain[bs],
         'retain_at_end': retain[max(retain)],
         'forgetting': round(retain[max(retain)] - retain[bs], 4),
         'adapt_curve': {str(k): v for k, v in sorted(adapt.items())},
         'retain_curve': {str(k): v for k, v in sorted(retain.items())}}
    json.dump(r, open(meta_f, 'w'), indent=2)
    return r


def score_seed(by_method):
    """Compare the plain-base vs prism-base finetunes at matched base quality."""
    if 'plain' not in by_method or 'prism' not in by_method:
        return None
    p, q = by_method['plain'], by_method['prism']       # q = prism
    pb, qb = p['base'], q['base']
    base_val_gap = round(abs(qb['val'] - pb['val']), 4)
    fp, fq = p['ft']['forgetting'], q['ft']['forgetting']
    return {
        'base_val_plain': pb['val'], 'base_val_prism': qb['val'],
        'base_val_gap': base_val_gap,
        'base_matched': base_val_gap <= 0.03 and pb['reached_target'] and qb['reached_target'],
        'base_steps_plain': pb['steps'], 'base_steps_prism': qb['steps'],
        'base_speedup_prism': round(pb['steps'] / qb['steps'], 2) if qb['steps'] else None,
        'forgetting_plain': fp, 'forgetting_prism': fq,
        # >1 ⇒ the PRISM base forgets LESS ⇒ better anchor (synergy)
        'forget_ratio_plain_over_prism': round(fp / fq, 3) if abs(fq) > 1e-6 else None,
        'adapt_best_plain': p['ft']['adapt_best'], 'adapt_best_prism': q['ft']['adapt_best'],
        # <1 ⇒ the PRISM base adapts BETTER
        'adapt_ratio_prism_over_plain': round(q['ft']['adapt_best'] / p['ft']['adapt_best'], 4),
    }


def run_key(a):
    seeds = '-'.join(str(s) for s in a.seeds)
    return (f"arc_b{a.base_target_val}_ft{a.ft_steps}_lr{a.ft_lr}_bs{a.batch_size}"
            f"_bl{a.block_size}_seeds{seeds}" + (f"_{a.tag}" if a.tag else ''))


def main():
    p = argparse.ArgumentParser(description='PRISM×finetune base-interaction (unified arc)')
    p.add_argument('--far_corpus', default='data/far.txt')
    p.add_argument('--base_methods', default='plain,prism')
    p.add_argument('--base_target_val', type=float, default=1.85,
                   help='matched old-domain val the bases stop at (must be reachable by plain)')
    p.add_argument('--base_max_iters', type=int, default=3000,
                   help='cap if the target is never reached (then base_matched=false)')
    p.add_argument('--base_eval_every', type=int, default=25)
    p.add_argument('--teacher_steps', type=int, default=2000)
    p.add_argument('--prism_lr', default='5e-4')
    p.add_argument('--prism_warmup', type=int, default=50)
    p.add_argument('--ft_steps', type=int, default=1000)
    p.add_argument('--ft_warmup', type=int, default=20)
    p.add_argument('--ft_lr', default='3e-4')
    p.add_argument('--ft_min_lr', default='3e-5')
    p.add_argument('--eval_every', type=int, default=25)
    p.add_argument('--eval_iters', type=int, default=200)
    p.add_argument('--seeds', default='1337')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--n_layer', type=int, default=None)
    p.add_argument('--n_head', type=int, default=None)
    p.add_argument('--n_embd', type=int, default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--tag', default='')
    a = p.parse_args()
    a.seeds = [int(s) for s in a.seeds.split(',') if s.strip()]
    methods = [m for m in a.base_methods.split(',') if m.strip()]
    device = a.device or default_device()

    # (train.py now forces an eval at the resume step, so retention_at_base is always
    # captured — the finetune eval cadence no longer has to divide base_eval_every.)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    key = run_key(a)
    run_dir = os.path.join(RUNS_DIR, key)
    os.makedirs(run_dir, exist_ok=True)
    meta_path = os.path.join(run_dir, 'meta.json')
    if os.path.exists(meta_path):
        stamp = json.load(open(meta_path))['stamp']
        log('[resume] existing run for this config.', 2)
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        json.dump({'stamp': stamp, 'started_utc': datetime.now(timezone.utc).isoformat()},
                  open(meta_path, 'w'), indent=2)

    print('=' * 64)
    log('PRISM×FINETUNE ARC — base-interaction study')
    print('=' * 64)
    log(f'Run key: {key}', 2)
    log(f'Seeds {a.seeds} | base methods {methods} | matched val target '
        f'{a.base_target_val} | device {device}', 2)

    dist = setup_ft(a.far_corpus)
    lock = acquire_lock(run_dir)
    try:
        per_seed, metrics = [], []
        for seed in a.seeds:
            print('-' * 64)
            log(f'SEED {seed}', 2)
            by_method = {}
            for m in methods:
                base = train_base(m, seed, a, device, run_dir)
                ft = finetune(base, seed, a, device, run_dir)
                by_method[m] = {'base': base, 'ft': ft}
            entry = {'seed': seed, 'methods': by_method}
            sc = score_seed(by_method)
            if sc:
                entry['comparison'] = sc
                metrics.append(sc)
            per_seed.append(entry)
            _persist(stamp, key, a, dist, device, methods, per_seed, metrics, complete=False)

        art = _persist(stamp, key, a, dist, device, methods, per_seed, metrics, complete=True)
        _report(art)
        log(f'Artifact: results/arc_{stamp}.json (also arc_latest.json)', 2)
        log('COMMIT THIS FILE — it is the evidence for any claim you publish.', 2)
    finally:
        try:
            os.remove(lock)
        except Exception:
            pass


def _agg(vals):
    vals = [v for v in vals if v is not None]
    return _stats(vals) if vals else None


def _persist(stamp, key, a, dist, device, methods, per_seed, metrics, complete):
    summary = None
    if metrics:
        summary = {
            'n_seeds': len(metrics),
            'base_val_gap': _agg([m['base_val_gap'] for m in metrics]),
            'base_matched_all': all(m['base_matched'] for m in metrics),
            'base_speedup_prism': _agg([m['base_speedup_prism'] for m in metrics]),
            'forgetting_plain': _agg([m['forgetting_plain'] for m in metrics]),
            'forgetting_prism': _agg([m['forgetting_prism'] for m in metrics]),
            'forget_ratio_plain_over_prism': _agg([m['forget_ratio_plain_over_prism'] for m in metrics]),
            'adapt_ratio_prism_over_plain': _agg([m['adapt_ratio_prism_over_plain'] for m in metrics]),
        }
    art = {
        'schema': 'prism-arc/1',
        'partial': not complete,
        'run_key': key,
        'provenance': provenance(),
        'config': {
            'far_corpus': a.far_corpus, 'base_methods': methods,
            'base_target_val': a.base_target_val, 'base_max_iters': a.base_max_iters,
            'teacher_steps': a.teacher_steps, 'prism_lr': a.prism_lr,
            'prism_warmup': a.prism_warmup, 'ft_steps': a.ft_steps,
            'ft_lr': a.ft_lr, 'ft_min_lr': a.ft_min_lr, 'eval_every': a.eval_every,
            'eval_iters': a.eval_iters, 'seeds': a.seeds, 'batch_size': a.batch_size,
            'block_size': a.block_size, 'device': device,
            'anchor': 'raw self-anchor prism_mod=0.01 decay=1.0 (identical for both bases)',
            'control_note': 'the ONLY difference between the two finetunes is the base '
                '(plain vs PRISM), matched to equal Shakespeare val via stop_val_target.',
            'distances': dist,
        },
        'runs': per_seed,
        'per_seed_comparison': metrics,
        'summary': summary,
    }
    for f in (f'arc_{stamp}.json', 'arc_latest.json'):
        json.dump(art, open(os.path.join(RESULTS_DIR, f), 'w'), indent=2)
    return art


def _report(a):
    s, c = a['summary'], a['config']
    print()
    print('  ' + '-' * 70)
    print(f'    PRISM×FINETUNE ARC — base interaction (matched val {c["base_target_val"]})')
    print('  ' + '-' * 70)
    if not s:
        print('    (need both plain and prism bases to compare)')
        print('  ' + '-' * 70)
        return
    print(f'    Seeds {c["seeds"]} · base_val_gap {s["base_val_gap"]["median"]:.4f} '
          f'(matched: {s["base_matched_all"]}) · PRISM reached target '
          f'{s["base_speedup_prism"]["median"]:.1f}× faster')
    print('  ' + '-' * 70)
    print(f'    forgetting:  plain {s["forgetting_plain"]["median"]:+.3f}   '
          f'prism {s["forgetting_prism"]["median"]:+.3f}')
    fr = s['forget_ratio_plain_over_prism']
    print(f'    forget_ratio (plain/prism): '
          f'{("%.2f× — PRISM base forgets less" % fr["median"]) if fr else "N/A"}')
    print(f'    adapt_ratio  (prism/plain): {s["adapt_ratio_prism_over_plain"]["median"]:.3f}  '
          f'(<1 = PRISM base adapts better)')
    print('  ' + '-' * 70)
    if not s['base_matched_all']:
        print('    ⚠ bases NOT matched (val gap >0.03 or target unreached) — comparison')
        print('      confounded; adjust --base_target_val and re-run.')
    print('    forget_ratio ≫ 1 (or adapt_ratio ≪ 1) at matched base = PRISM base is a')
    print('    better anchor → genuine synergy. ≈ 1 = anchor is base-agnostic (honest')
    print('    negative; combining PRISM+finetune buys compounding, not synergy).')
    print()


if __name__ == '__main__':
    main()
