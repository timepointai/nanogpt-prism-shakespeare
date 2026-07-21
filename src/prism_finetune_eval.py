"""
prism_finetune_eval.py — Does the Mod Wheel let a trained model finetune WITHOUT
losing the advantage? (finetune-retention benchmark)

PRISM's from-scratch result says overfitting is DRIFTING OUT of a trained
transformer's converged spectral shape, and the Mod Wheel (a per-step pull back
toward a stored spectral target) prevents that drift. This benchmark tests the
sharp corollary: catastrophic forgetting during FINETUNING is the same drift, so
keeping the Mod Wheel engaged — self-anchored to the resumed base weights — should
let a model adapt to a NEW domain as well as a plain finetune while forgetting the
OLD domain far less, and without the new-domain overfitting U-turn.

Protocol (one base per seed; every arm forks the SAME base ckpt, byte-identical):
    base            scratch, no prism, OLD domain (Shakespeare), base_steps
    A · plain       resume(base), NEW domain (far corpus), prism_mod=0        [control]
    B · selfanchor  resume(base), NEW domain, prism_mod=0.01 decay=1.0        [technique]
    C · scratch     scratch, no prism, NEW domain, ft_steps                   [adapt ceiling]

A and B differ by exactly one flag (prism_mod). Both are scored on BOTH the new
domain (adaptation, val) and the old domain (retention, val2) every eval, so a
single run yields the adapt-vs-forget tradeoff. C bounds the best new-domain loss
reachable in ft_steps, so "B retained" can never be confused with "B learned
nothing."

HEADLINE METRIC (a ratio vs the control, never an absolute):
    forgetting_ratio = forgetting_plain / forgetting_mod   (>1 = wheel forgets less)
    where forgetting_* = retention_at_end - retention_at_base (old-domain val climb).

Guards (each voids the relevant comparison, recorded in the artifact):
    retention_floored   plain forgot <= 0.05 nats  -> domain too close, test VOID
    adaptation_censored B_best > 1.10 * A_best     -> wheel froze the model (fail)
    left_censored       best hit at first ft eval  -> floor, extend ft_steps

Usage:
    python prism_finetune_eval.py --far_corpus=data/far.txt \\
        --base_steps=2000 --ft_steps=1000 --eval_every=25 --seeds=1337,1338,1339 \\
        --batch_size=32 --block_size=256 --learning_rate=3e-4 --min_lr=3e-5 \\
        --arms=base,plain,selfanchor,scratch_ceiling

Every run writes a full artifact (schema prism-finetune/1) to results/. Commit it.
The run is stepwise-resumable: each base and each arm banks to .prism_runs/<key>/.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from prism_eval import (provenance, parse_curve, stream_train, acquire_lock,
                        _stats, _token_js, _encode_corpus, log, default_device,
                        RESULTS_DIR, RUNS_DIR, SRC_DIR)

SHAKE = 'data/shakespeare_char'      # OLD domain (retention). base trains here.
FT = 'data/sherlock_ft'              # NEW domain (adaptation). built from far_corpus.
CONFIG = 'config/train_shakespeare_char.py'


def parse_retain(stdout):
    """Extract {step: val2_loss} (old-domain / retention loss) from the eval lines."""
    curve = {}
    for line in stdout.split('\n'):
        m = re.search(r'step (\d+): train loss [\d.]+, val loss [\d.]+, '
                      r'val2 loss ([\d.]+)', line)
        if m:
            curve[int(m.group(1))] = float(m.group(2))
    return curve


def setup_ft(far_corpus):
    """Build the NEW-domain dataset (data/sherlock_ft) from far_corpus, encoded in
    Shakespeare's char vocab (so a resumed model's embedding matches). Holds out a
    tail for the adaptation val, mirroring the proven far_val split. Returns a
    distance dict for the artifact."""
    os.chdir(SRC_DIR)
    if not os.path.exists(f'{SHAKE}/train.bin'):
        log('Preparing Shakespeare char dataset (first run only)…', 2)
        subprocess.run([sys.executable, f'{SHAKE}/prepare.py'],
                       capture_output=True, check=True)

    far = _encode_corpus(far_corpus, f'{SHAKE}/meta.pkl')          # uint16, old vocab
    shake_train = np.array(np.memmap(f'{SHAKE}/train.bin', dtype=np.uint16, mode='r'))
    shake_val = np.array(np.memmap(f'{SHAKE}/val.bin', dtype=np.uint16, mode='r'))

    n_val = min(len(far) // 10, len(shake_val))
    ft_val, ft_train = far[-n_val:], far[:-n_val]

    os.makedirs(FT, exist_ok=True)
    ft_train.astype(np.uint16).tofile(f'{FT}/train.bin')
    ft_val.astype(np.uint16).tofile(f'{FT}/val.bin')
    shutil.copy(f'{SHAKE}/meta.pkl', f'{FT}/meta.pkl')

    dist = {
        'far_corpus': os.path.basename(far_corpus),
        'ft_train_tokens': int(len(ft_train)),
        'ft_val_tokens': int(len(ft_val)),
        'token_js_new_vs_old_train': _token_js(ft_train.astype(np.uint16),
                                               shake_train.astype(np.uint16)),
        'token_js_new_train_vs_old_val': _token_js(ft_train.astype(np.uint16),
                                                   shake_val.astype(np.uint16)),
    }
    log(f'NEW domain "{dist["far_corpus"]}": {len(ft_train):,} train / {len(ft_val):,} '
        f'val tok · token-JS vs old {dist["token_js_new_vs_old_train"]:.4f}', 2)
    return dist


def size_args(a):
    """Model-size overrides (for the CPU smoke); on resume train.py forces these
    from the ckpt, so they only bind the base + scratch_ceiling arms — pass them to
    every arm so all share one architecture/vocab."""
    out = [f'--batch_size={a.batch_size}', f'--block_size={a.block_size}']
    for k in ('n_layer', 'n_head', 'n_embd'):
        v = getattr(a, k)
        if v is not None:
            out.append(f'--{k}={v}')
    return out


def train_base(seed, a, device, run_dir):
    """Train the OLD-domain base once per seed. Idempotent: reuses the on-disk ckpt
    (the fork source) + banked meta when both survive a container recycle."""
    out = f'out-ft-base-s{seed}'
    ckpt = f'{out}/ckpt.pt'
    meta_f = os.path.join(run_dir, f's{seed}_base.json')
    if os.path.exists(ckpt) and os.path.exists(meta_f):
        r = json.load(open(meta_f))
        log(f'[resume] base (seed {seed}) already trained (val {r["val_end"]:.4f}).', 2)
        return r

    log(f'Training base (seed {seed}, {a.base_steps} steps, {SHAKE})…', 2)
    rc, stdout = stream_train(
        [sys.executable, '-u', 'train.py', CONFIG, f'--dataset=shakespeare_char',
         f'--seed={seed}', f'--device={device}', f'--max_iters={a.base_steps}',
         f'--eval_interval={a.base_steps}', f'--eval_iters={a.eval_iters}',
         '--log_interval=100', f'--out_dir={out}', '--always_save_checkpoint=True',
         '--compile=False', '--prism_init=False', '--wandb_log=False'] + size_args(a),
        f's{seed} base', a.base_steps)
    if rc != 0:
        raise RuntimeError(f'Base training failed (seed {seed}):\n{stdout[-2000:]}')
    curve = parse_curve(stdout)
    if not curve:
        raise RuntimeError(f'No eval lines for base (seed {seed}):\n{stdout[-1000:]}')
    r = {'ckpt': ckpt, 'val_best': min(curve.values()),
         'val_end': curve[max(curve)], 'curve': {str(k): v for k, v in sorted(curve.items())}}
    json.dump(r, open(meta_f, 'w'), indent=2)
    return r


def run_arm(arm, seed, base_ckpt, a, device, run_dir):
    """Finetune (or, for scratch_ceiling, train fresh) one arm on the NEW domain.
    Idempotent: a completed arm is banked; a re-run re-forks the base cleanly (never
    resumes a half-finetuned arm onto itself)."""
    out = f'out-ft-{arm}-s{seed}'
    meta_f = os.path.join(run_dir, f's{seed}_{arm}.json')
    if os.path.exists(meta_f):
        r = json.load(open(meta_f))
        log(f'[resume] arm "{arm}" (seed {seed}) already done '
            f'(adapt best {r["adapt_best"]:.4f}).', 2)
        return r

    resume = arm != 'scratch_ceiling'
    last_step = (a.base_steps + a.ft_steps) if resume else a.ft_steps
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)

    cmd = [sys.executable, '-u', 'train.py', CONFIG, f'--seed={seed}',
           f'--device={device}', f'--eval_interval={a.eval_every}',
           f'--eval_iters={a.eval_iters}', '--log_interval=100', f'--out_dir={out}',
           '--compile=False', '--wandb_log=False', '--dataset=sherlock_ft'] + size_args(a)

    if resume:
        shutil.copy(base_ckpt, f'{out}/ckpt.pt')          # fork the base
        cmd += ['--init_from=resume', f'--val2_dir={SHAKE}',
                f'--max_iters={a.base_steps + a.ft_steps}',
                f'--warmup_iters={a.base_steps + a.ft_warmup}',
                f'--lr_decay_iters={a.base_steps + a.ft_steps}',
                f'--learning_rate={a.learning_rate}', f'--min_lr={a.min_lr}',
                '--decay_lr=True']
        if arm == 'plain':
            pass                    # the control: mod wheel OFF (prism_mod defaults to 0.0)
        elif arm == 'selfanchor':
            cmd += ['--prism_mod=0.01', '--prism_mod_decay=1.0']   # the technique
        else:
            raise ValueError(f'unknown resume arm {arm}')
    else:   # scratch_ceiling: best NEW-domain loss reachable fresh in ft_steps
        cmd += ['--init_from=scratch', '--prism_init=False',
                f'--max_iters={a.ft_steps}', f'--warmup_iters={a.ft_warmup}',
                f'--lr_decay_iters={a.ft_steps}',
                f'--learning_rate={a.ceiling_lr}', f'--min_lr={a.min_lr}',
                '--decay_lr=True']

    log(f'Running arm "{arm}" (seed {seed}) → step {last_step}…', 2)
    rc, stdout = stream_train(cmd, f's{seed} {arm}', last_step)
    if rc != 0:
        raise RuntimeError(f'Arm {arm} failed (seed {seed}):\n{stdout[-2000:]}')

    adapt = parse_curve(stdout)                            # NEW-domain (val)
    if not adapt or max(adapt) < last_step:
        raise RuntimeError(f'Arm {arm} (seed {seed}) truncated: last eval '
                           f'{max(adapt) if adapt else None}, expected {last_step}. '
                           f'Refusing to score a partial curve.')
    adapt_best = min(adapt.values())
    r = {
        'arm': arm,
        'adapt_curve': {str(k): v for k, v in sorted(adapt.items())},
        'adapt_best': adapt_best,
        'adapt_best_step': min(adapt, key=adapt.get),
        'adapt_at_end': adapt[max(adapt)],
        'adapt_overfits': adapt[max(adapt)] > adapt_best * 1.05,
    }
    if resume:
        retain = parse_retain(stdout)                     # OLD-domain (val2)
        if a.base_steps not in retain or max(retain) < last_step:
            raise RuntimeError(f'Arm {arm} (seed {seed}) missing retention val2 at '
                               f'base step {a.base_steps} or end. Got {sorted(retain)}.')
        r['retain_curve'] = {str(k): v for k, v in sorted(retain.items())}
        r['retain_at_base'] = retain[a.base_steps]
        r['retain_at_end'] = retain[max(retain)]
    json.dump(r, open(meta_f, 'w'), indent=2)
    return r


def score_seed(arms):
    """Per-seed forgetting/adaptation metrics. Needs plain + selfanchor; ceiling
    optional."""
    plain, mod = arms['plain'], arms['selfanchor']
    ceil = arms.get('scratch_ceiling')
    rb = plain['retain_at_base']
    forget_plain = plain['retain_at_end'] - rb
    forget_mod = mod['retain_at_end'] - rb
    ratio = (forget_plain / forget_mod) if abs(forget_mod) > 1e-6 else None
    m = {
        'retention_at_base': rb,
        'retention_at_base_selfanchor': mod['retain_at_base'],   # sanity: ≈ rb
        'forgetting_plain': round(forget_plain, 4),
        'forgetting_mod': round(forget_mod, 4),
        'forgetting_ratio': round(ratio, 3) if ratio is not None else None,
        'retention_gap': round(forget_plain - forget_mod, 4),
        'adapt_best_plain': plain['adapt_best'],
        'adapt_best_mod': mod['adapt_best'],
        'adaptation_cost': round(mod['adapt_best'] / plain['adapt_best'], 4),
        'overfit_averted': bool(plain['adapt_overfits'] and not mod['adapt_overfits']),
        # guards
        'retention_floored': forget_plain <= 0.05,
        'adaptation_censored': mod['adapt_best'] > 1.10 * plain['adapt_best'],
    }
    if ceil:
        m['adapt_best_ceiling'] = ceil['adapt_best']
        m['adapt_gap_to_ceiling_mod'] = round(mod['adapt_best'] - ceil['adapt_best'], 4)
    return m


def run_key(a):
    seeds = '-'.join(str(s) for s in a.seeds)
    corpus = os.path.splitext(os.path.basename(a.far_corpus))[0]
    return (f"ft_{corpus}_b{a.base_steps}_f{a.ft_steps}_e{a.eval_every}"
            f"_lr{a.learning_rate}_bs{a.batch_size}_bl{a.block_size}_seeds{seeds}")


def main():
    p = argparse.ArgumentParser(description='PRISM finetune-retention benchmark')
    p.add_argument('--far_corpus', default='data/far.txt',
                   help='NEW-domain .txt (char-encoded in Shakespeare vocab)')
    p.add_argument('--base_steps', type=int, default=2000)
    p.add_argument('--ft_steps', type=int, default=1000)
    p.add_argument('--ft_warmup', type=int, default=20,
                   help='finetune warmup (added to base_steps for the resume arms)')
    p.add_argument('--eval_every', type=int, default=25)
    p.add_argument('--eval_iters', type=int, default=200)
    p.add_argument('--seeds', default='1337,1338,1339')
    p.add_argument('--arms', default='base,plain,selfanchor,scratch_ceiling',
                   help='comma list from base,plain,selfanchor,scratch_ceiling')
    p.add_argument('--learning_rate', default='3e-4', help='finetune LR (resume arms)')
    p.add_argument('--min_lr', default='3e-5')
    p.add_argument('--ceiling_lr', default='1e-3', help='LR for the scratch ceiling arm')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--n_layer', type=int, default=None)
    p.add_argument('--n_head', type=int, default=None)
    p.add_argument('--n_embd', type=int, default=None)
    p.add_argument('--device', default=None)
    a = p.parse_args()
    a.seeds = [int(s) for s in a.seeds.split(',') if s.strip()]
    arms = [x for x in a.arms.split(',') if x.strip()]
    device = a.device or default_device()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    key = run_key(a)
    run_dir = os.path.join(RUNS_DIR, key)
    os.makedirs(run_dir, exist_ok=True)

    # stable timestamp across resumes (so every resume writes the SAME artifact)
    meta_path = os.path.join(run_dir, 'meta.json')
    if os.path.exists(meta_path):
        stamp = json.load(open(meta_path))['stamp']
        log(f'[resume] existing run for this config.', 2)
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        json.dump({'stamp': stamp, 'started_utc': datetime.now(timezone.utc).isoformat()},
                  open(meta_path, 'w'), indent=2)

    print('=' * 64)
    log('PRISM FINETUNE-RETENTION EVAL')
    print('=' * 64)
    log(f'Run key: {key}', 2)
    log(f'Seeds {a.seeds} | arms {arms} | device {device}', 2)
    log(f'base {a.base_steps} steps ({SHAKE}) → finetune {a.ft_steps} steps '
        f'({a.far_corpus}) @ LR {a.learning_rate}', 2)

    dist = setup_ft(a.far_corpus)
    lock = acquire_lock(run_dir)
    try:
        per_seed, per_seed_metrics = [], []
        for seed in a.seeds:
            print('-' * 64)
            log(f'SEED {seed}', 2)
            arm_results = {}
            base = train_base(seed, a, device, run_dir)     # always needed to fork
            for arm in [x for x in arms if x != 'base']:
                arm_results[arm] = run_arm(arm, seed, base['ckpt'], a, device, run_dir)
            entry = {'seed': seed, 'base': base, 'arms': arm_results}
            if 'plain' in arm_results and 'selfanchor' in arm_results:
                entry['metrics'] = score_seed(arm_results)
                per_seed_metrics.append(entry['metrics'])
            per_seed.append(entry)
            _persist(stamp, key, a, dist, device, arms, per_seed, per_seed_metrics,
                     complete=False)

        art = _persist(stamp, key, a, dist, device, arms, per_seed, per_seed_metrics,
                       complete=True)
        _report(art)
        log(f'Artifact: results/finetune_{stamp}.json (also finetune_latest.json)', 2)
        log('COMMIT THIS FILE — it is the evidence for any claim you publish.', 2)
    finally:
        try:
            os.remove(lock)
        except Exception:
            pass


def _agg(metrics, kk):
    vals = [m[kk] for m in metrics if m.get(kk) is not None]
    return _stats(vals) if vals else None


def _persist(stamp, key, a, dist, device, arms, per_seed, metrics, complete):
    summary = None
    if metrics:
        summary = {
            'n_seeds': len(metrics),
            'forgetting_plain': _agg(metrics, 'forgetting_plain'),
            'forgetting_mod': _agg(metrics, 'forgetting_mod'),
            'forgetting_ratio': _agg(metrics, 'forgetting_ratio'),
            'retention_gap': _agg(metrics, 'retention_gap'),
            'adaptation_cost': _agg(metrics, 'adaptation_cost'),
            'adapt_best_plain': _agg(metrics, 'adapt_best_plain'),
            'adapt_best_mod': _agg(metrics, 'adapt_best_mod'),
            'overfit_averted_all': all(m['overfit_averted'] for m in metrics),
            'retention_floored_any': any(m['retention_floored'] for m in metrics),
            'adaptation_censored_any': any(m['adaptation_censored'] for m in metrics),
        }
    art = {
        'schema': 'prism-finetune/1',
        'partial': not complete,
        'run_key': key,
        'provenance': provenance(),
        'config': {
            'far_corpus': a.far_corpus,
            'base_steps': a.base_steps, 'ft_steps': a.ft_steps,
            'ft_warmup': a.ft_warmup, 'eval_every': a.eval_every,
            'eval_iters': a.eval_iters, 'seeds': a.seeds, 'arms': arms,
            'learning_rate': a.learning_rate, 'min_lr': a.min_lr,
            'ceiling_lr': a.ceiling_lr, 'batch_size': a.batch_size,
            'block_size': a.block_size, 'device': device,
            'prism_mod': 0.01, 'prism_mod_decay': 1.0,
            'prism_mod_decay_note': 'constant pull during finetune (decay=1.0): the '
                'canonical 0.9999 anneals a from-scratch reshape, which is not '
                'happening on resume; a constant pull removes the resume-iter clock.',
            'schedule_matched': True,   # plain vs selfanchor differ only in prism_mod
            'schedule_matched_note': 'plain (prism_mod=0) and selfanchor (prism_mod=0.01) '
                'share LR/warmup/steps/data — the ONLY difference is the mod wheel.',
            'distances': dist,
        },
        'runs': per_seed,
        'per_seed_metrics': metrics,
        'summary': summary,
    }
    for f in (f'finetune_{stamp}.json', 'finetune_latest.json'):
        json.dump(art, open(os.path.join(RESULTS_DIR, f), 'w'), indent=2)
    return art


def _report(a):
    s, c = a['summary'], a['config']
    print()
    print('  ' + '-' * 66)
    print(f'    PRISM FINETUNE-RETENTION — {c["far_corpus"]}  '
          f'(base {c["base_steps"]} → ft {c["ft_steps"]})')
    print('  ' + '-' * 66)
    if not s:
        print('    (no plain+selfanchor pair scored)')
        print('  ' + '-' * 66)
        return
    d = c['distances']
    print(f'    Seeds {c["seeds"]} · token-JS new-vs-old {d["token_js_new_vs_old_train"]:.4f}')
    for m in a['per_seed_metrics']:
        print(f'    forget plain {m["forgetting_plain"]:+.3f} · mod {m["forgetting_mod"]:+.3f}'
              f'  ratio {m["forgetting_ratio"]}  ·  adapt cost {m["adaptation_cost"]:.3f}'
              f'  overfit_averted {m["overfit_averted"]}')
    print('  ' + '-' * 66)
    fr = s['forgetting_ratio']
    print(f'    FORGETTING (old-domain val climb): plain median '
          f'{s["forgetting_plain"]["median"]:+.3f}  vs  mod {s["forgetting_mod"]["median"]:+.3f}')
    print(f'    FORGETTING RATIO (plain/mod): '
          f'{("median %.2fx  range %.2f-%.2f" % (fr["median"], fr["min"], fr["max"])) if fr else "N/A"}')
    print(f'    ADAPTATION cost (mod/plain new-domain best): '
          f'{s["adaptation_cost"]["median"]:.3f}  (≈1.0 = wheel did not freeze)')
    print(f'    overfit averted (all seeds): {s["overfit_averted_all"]}')
    print('  ' + '-' * 66)
    if s['retention_floored_any']:
        print('    ⚠ retention_floored: plain barely forgot (≤0.05) — domain too')
        print('      close, comparison VOID. Swap to a genuinely far corpus.')
    if s['adaptation_censored_any']:
        print('    ⚠ adaptation_censored: mod arm froze (new-domain best >1.10× plain)')
        print('      — retention was bought by not learning. A failure, not a win.')
    print('    forgetting_ratio > ~2 with adaptation_cost ≈ 1.0 = the wheel lets you')
    print('    finetune without losing the advantage. Ratio ≈ 1 = clean negative.')
    print()


if __name__ == '__main__':
    main()
