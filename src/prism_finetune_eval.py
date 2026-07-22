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

# The arms. Resume arms finetune a fork of the base; scratch_ceiling trains fresh.
# Round 1 used plain + selfanchor + scratch_ceiling. Round 2 adds the attribution
# ladder: a raw-anchor strength sweep, a low-LR frontier (the "mod wheel is just a
# smaller LR" null), the spectral anchor (hold base spectrum, free directions), and
# a shuffled-spectrum placebo — all at matched schedule so a win is attributable.
ARM_SPECS = {
    'plain':           dict(resume=True,  mod=0.0),                                   # control
    'selfanchor':      dict(resume=True,  mod=0.01,  mode='raw'),                     # R1 technique (== raw_mid)
    'raw_lo':          dict(resume=True,  mod=0.005, mode='raw'),
    'raw_mid':         dict(resume=True,  mod=0.01,  mode='raw'),
    'raw_hi':          dict(resume=True,  mod=0.02,  mode='raw'),
    'lowlr_a':         dict(resume=True,  mod=0.0,   lr='1.5e-4'),
    'lowlr_b':         dict(resume=True,  mod=0.0,   lr='1e-4'),
    'lowlr_c':         dict(resume=True,  mod=0.0,   lr='5e-5'),
    'spectral':        dict(resume=True,  mod=0.01,  mode='spectral', refresh=25),    # the PRISM claim
    'shuffled':        dict(resume=True,  mod=0.01,  mode='shuffled', refresh=25),    # placebo
    'scratch_ceiling': dict(resume=False),                                           # adapt ceiling
}


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

    spec = ARM_SPECS[arm]
    resume = spec['resume']
    last_step = (a.base_steps + a.ft_steps) if resume else a.ft_steps
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)

    cmd = [sys.executable, '-u', 'train.py', CONFIG, f'--seed={seed}',
           f'--device={device}', f'--eval_interval={a.eval_every}',
           f'--eval_iters={a.eval_iters}', '--log_interval=100', f'--out_dir={out}',
           '--compile=False', '--wandb_log=False', '--dataset=sherlock_ft'] + size_args(a)

    if resume:
        shutil.copy(base_ckpt, f'{out}/ckpt.pt')          # fork the base
        lr = spec.get('lr', a.learning_rate)
        cmd += ['--init_from=resume', f'--val2_dir={SHAKE}',
                f'--max_iters={a.base_steps + a.ft_steps}',
                f'--warmup_iters={a.base_steps + a.ft_warmup}',
                f'--lr_decay_iters={a.base_steps + a.ft_steps}',
                f'--learning_rate={lr}', f'--min_lr={a.min_lr}', '--decay_lr=True']
        if spec.get('mod', 0.0) > 0:   # engage the mod wheel (constant pull during ft)
            cmd += [f'--prism_mod={spec["mod"]}', '--prism_mod_decay=1.0',
                    f'--prism_anchor_mode={spec.get("mode", "raw")}']
            if spec.get('refresh'):
                cmd.append(f'--prism_anchor_refresh={spec["refresh"]}')
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
    """Per-seed forgetting/adaptation metrics for ANY set of resume arms. forgetting
    = old-domain (Shakespeare) val climb from the shared base; adapt = new-domain
    (Sherlock) best. Each anchor arm is compared to the plain control as a ratio.
    Keeps the Round-1 headline (plain vs the raw-0.01 anchor) when both are present."""
    resume = {k: v for k, v in arms.items() if 'retain_at_end' in v}
    if not resume:
        return None
    rb = next(iter(resume.values()))['retain_at_base']     # identical across arms (same fork)
    plain = resume.get('plain')
    per_arm = {}
    for k, v in resume.items():
        f = round(v['retain_at_end'] - rb, 4)
        rec = {'forgetting': f, 'adapt_best': v['adapt_best'],
               'adapt_at_end': v['adapt_at_end'], 'adapt_overfits': v['adapt_overfits']}
        if plain:
            pf = plain['retain_at_end'] - rb
            rec['forgetting_ratio_vs_plain'] = round(pf / f, 3) if abs(f) > 1e-6 else None
            rec['adaptation_cost_vs_plain'] = round(v['adapt_best'] / plain['adapt_best'], 4)
        per_arm[k] = rec
    m = {'retention_at_base': rb, 'arms': per_arm}
    if 'scratch_ceiling' in arms:
        m['adapt_best_ceiling'] = arms['scratch_ceiling']['adapt_best']
    # headline: plain vs the canonical raw-0.01 anchor (selfanchor or raw_mid)
    anchor = 'selfanchor' if 'selfanchor' in per_arm else \
             ('raw_mid' if 'raw_mid' in per_arm else None)
    if plain and anchor:
        fp, fm = per_arm['plain']['forgetting'], per_arm[anchor]['forgetting']
        m['forgetting_plain'] = fp
        m['forgetting_mod'] = fm
        m['forgetting_ratio'] = round(fp / fm, 3) if abs(fm) > 1e-6 else None
        m['adaptation_cost'] = per_arm[anchor]['adaptation_cost_vs_plain']
        m['overfit_averted'] = bool(per_arm['plain']['adapt_overfits']
                                    and not per_arm[anchor]['adapt_overfits'])
        m['retention_floored'] = fp <= 0.05
        m['adaptation_censored'] = per_arm[anchor]['adapt_best'] > 1.10 * per_arm['plain']['adapt_best']
    return m


def run_key(a):
    seeds = '-'.join(str(s) for s in a.seeds)
    corpus = os.path.splitext(os.path.basename(a.far_corpus))[0]
    # --tag disambiguates runs that share the schedule but differ in arm set (the
    # key does NOT hash arms, so a bigger arm set on the same schedule would collide
    # with / overwrite an earlier run's artifact without a distinct tag).
    return (f"ft_{corpus}_b{a.base_steps}_f{a.ft_steps}_e{a.eval_every}"
            f"_lr{a.learning_rate}_bs{a.batch_size}_bl{a.block_size}_seeds{seeds}"
            + (f"_{a.tag}" if a.tag else ''))


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
                   help='comma list; base + any of ' + ', '.join(ARM_SPECS)
                        + ' (Round 2 ladder: raw_lo/mid/hi, lowlr_a/b/c, spectral, shuffled)')
    p.add_argument('--learning_rate', default='3e-4', help='finetune LR (resume arms)')
    p.add_argument('--min_lr', default='3e-5')
    p.add_argument('--ceiling_lr', default='1e-3', help='LR for the scratch ceiling arm')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--n_layer', type=int, default=None)
    p.add_argument('--n_head', type=int, default=None)
    p.add_argument('--n_embd', type=int, default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--tag', default='',
                   help='disambiguator folded into the run key / artifact (use a '
                        'distinct tag when a new arm set shares a prior run schedule)')
    a = p.parse_args()
    a.seeds = [int(s) for s in a.seeds.split(',') if s.strip()]
    arms = [x for x in a.arms.split(',') if x.strip()]
    bad = [x for x in arms if x != 'base' and x not in ARM_SPECS]
    if bad:
        sys.exit(f'unknown arm(s): {bad}. known: base, ' + ', '.join(ARM_SPECS))
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
            m = score_seed(arm_results)
            if m:
                entry['metrics'] = m
                per_seed_metrics.append(m)
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


def _agg(vals):
    vals = [v for v in vals if v is not None]
    return _stats(vals) if vals else None


def _persist(stamp, key, a, dist, device, arms, per_seed, metrics, complete):
    summary = None
    if metrics:
        arm_names = sorted({name for m in metrics for name in m.get('arms', {})})
        arm_summ = {}
        for name in arm_names:
            got = [m['arms'][name] for m in metrics if name in m['arms']]
            arm_summ[name] = {
                'forgetting': _agg([g['forgetting'] for g in got]),
                'adapt_best': _agg([g['adapt_best'] for g in got]),
                'forgetting_ratio_vs_plain': _agg([g.get('forgetting_ratio_vs_plain') for g in got]),
                'adaptation_cost_vs_plain': _agg([g.get('adaptation_cost_vs_plain') for g in got]),
                'overfits_any': any(g['adapt_overfits'] for g in got),
            }
        summary = {
            'n_seeds': len(metrics),
            'retention_at_base': _agg([m['retention_at_base'] for m in metrics]),
            'adapt_best_ceiling': _agg([m.get('adapt_best_ceiling') for m in metrics]),
            'arms': arm_summ,
        }
        if any('forgetting_ratio' in m for m in metrics):   # headline pair present
            summary.update({
                'forgetting_plain': _agg([m.get('forgetting_plain') for m in metrics]),
                'forgetting_mod': _agg([m.get('forgetting_mod') for m in metrics]),
                'forgetting_ratio': _agg([m.get('forgetting_ratio') for m in metrics]),
                'adaptation_cost': _agg([m.get('adaptation_cost') for m in metrics]),
                'overfit_averted_all': all(m.get('overfit_averted') for m in metrics),
                'retention_floored_any': any(m.get('retention_floored') for m in metrics),
                'adaptation_censored_any': any(m.get('adaptation_censored') for m in metrics),
            })
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
            'prism_mod_decay': 1.0,
            'prism_mod_decay_note': 'constant pull during finetune (decay=1.0): the '
                'canonical 0.9999 anneals a from-scratch reshape, which is not '
                'happening on resume; a constant pull removes the resume-iter clock.',
            'arm_specs': {k: ARM_SPECS[k] for k in arms if k in ARM_SPECS},
            'schedule_matched': True,
            'schedule_matched_note': 'the plain control and every anchor arm share '
                'LR/warmup/steps/data and differ ONLY in the mod wheel (prism_mod / '
                'anchor_mode) — EXCEPT the lowlr_* frontier, which varies LR on '
                'purpose as the "mod wheel is just a smaller LR" null.',
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
    print('  ' + '-' * 74)
    print(f'    PRISM FINETUNE-RETENTION — {c["far_corpus"]}  '
          f'(base {c["base_steps"]} → ft {c["ft_steps"]})')
    print('  ' + '-' * 74)
    if not s:
        print('    (no resume arm scored)')
        print('  ' + '-' * 74)
        return
    d = c['distances']
    print(f'    Seeds {c["seeds"]} · token-JS new-vs-old {d["token_js_new_vs_old_train"]:.4f}'
          f' · retention_at_base {s["retention_at_base"]["median"]:.3f}'
          + (f' · scratch ceiling {s["adapt_best_ceiling"]["median"]:.3f}'
             if s.get('adapt_best_ceiling') else ''))
    print('  ' + '-' * 74)
    print(f'    {"arm":16} | {"forget↓":>8} | {"adapt↓":>7} | {"less-forget/plain":>17} | overfit')
    print('  ' + '-' * 74)
    # order: plain first, then anchors by forgetting (best retention first)
    items = sorted(s['arms'].items(),
                   key=lambda kv: (kv[0] != 'plain',
                                   kv[1]['forgetting']['median'] if kv[1]['forgetting'] else 9))
    for name, g in items:
        fo = g['forgetting']['median'] if g['forgetting'] else float('nan')
        ab = g['adapt_best']['median'] if g['adapt_best'] else float('nan')
        rr = g['forgetting_ratio_vs_plain']
        rr_s = f'{rr["median"]:.2f}x' if rr and rr['median'] is not None else '—'
        print(f'    {name:16} | {fo:>+8.3f} | {ab:>7.3f} | {rr_s:>17} | '
              f'{"Y" if g["overfits_any"] else "n"}')
    print('  ' + '-' * 74)
    print('    forget↓ = old-domain (Shakespeare) val climb from base (lower=better).')
    print('    adapt↓  = new-domain (Sherlock) best val (lower=better; beat the ceiling).')
    if 'forgetting_ratio' in s and s['forgetting_ratio']:
        fr = s['forgetting_ratio']
        print(f'    HEADLINE forgetting_ratio (plain/anchor): median {fr["median"]:.2f}x '
              f'range {fr["min"]:.2f}-{fr["max"]:.2f}  ·  adaptation_cost '
              f'{s["adaptation_cost"]["median"]:.3f}')
        if s.get('retention_floored_any'):
            print('    ⚠ retention_floored: plain barely forgot (≤0.05) — domain too close, VOID.')
        if s.get('adaptation_censored_any'):
            print('    ⚠ adaptation_censored: anchor froze (new-domain best >1.10× plain).')
    print('    ATTRIBUTION (Round 2): spectral must retain like raw AND adapt better,')
    print('    Pareto-dominating the raw_* and lowlr_* frontiers + beating shuffled,')
    print('    to earn "spectral"; else it is a generic proximal anchor.')
    print()


if __name__ == '__main__':
    main()
