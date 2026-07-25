"""
prism_prior_eval.py — Prior-Fused PRISM: does a T9-style fixed shared n-gram prior,
fused into the logits, stack with PRISM's spectral init for a much larger speedup?

The T9 idea: a tiny shared prior (~a dense n-gram table) already predicts held-out
same-modality text almost as well as the converged neural baseline (measured: a
context-3 char n-gram hits ~2.57 bits/char on Shakespeare val = the baseline's best).
Fuse it into the model (product of experts, final = model_logits + λ·log p_ngram) and
the model starts at baseline quality FOR FREE, then learns only the residual below the
n-gram floor — where PRISM's geometry helps.

Four arms, all on shakespeare_char, scored as steps to reach the plain baseline's best
val loss (the PRISM Score), plus the init loss (step 0) and the best loss:
    baseline     plain
    prism        spectral init + mod wheel (needs a teacher)
    prior        plain + fused n-gram prior (the T9 dictionary)
    prism_prior  both — the hybrid

Every run writes a prism-prior/1 artifact. Resumable; partial runs raise.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from prism_eval import (provenance, parse_curve, stream_train, run_training, compute_score,
                        acquire_lock, _stats, log, default_device, setup,
                        RESULTS_DIR, RUNS_DIR, SRC_DIR)

CONFIG = 'config/train_shakespeare_char.py'
ARMS = {  # name: (use_prism, use_prior)
    'baseline':    (False, False),
    'prism':       (True,  False),
    'prior':       (False, True),
    'prism_prior': (True,  True),
}


def build_prior(context, lambdas, out_base):
    tag = 'default' if not lambdas else 'l' + lambdas.replace('.', '').replace(',', '_')
    out = f'{out_base}/{tag}'
    tab = f'{out}/prior_c{context}.pt'
    if os.path.exists(tab):
        log(f'[resume] n-gram prior context {context} ({tag}) cached.', 2)
        return tab
    log(f'Building n-gram prior (context {context}, lambdas {lambdas or "default"})…', 2)
    cmd = [sys.executable, 'build_ngram_prior.py', '--dataset=shakespeare_char',
           f'--context={context}', f'--out={out}']
    if lambdas:
        cmd.append(f'--lambdas={lambdas}')
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f'build_ngram_prior failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}')
    for line in r.stdout.splitlines():
        if 'standalone' in line or 'saved' in line:
            log(line.strip(), 4)
    return tab


def train_teacher(seed, a, device, run_dir):
    out = f'out-prior-teacher-s{seed}'
    cache = f'.prism_cache/prior_teacher_s{seed}'
    if os.path.exists(f'{cache}/directions.pt'):
        log(f'[resume] teacher (seed {seed}) fingerprint cached.', 2)
        return cache
    log(f'Training teacher (seed {seed}, {a.teacher_steps} steps)…', 2)
    rc, out_s = stream_train(
        [sys.executable, '-u', 'train.py', CONFIG, '--dataset=shakespeare_char',
         f'--seed={seed}', f'--device={device}', f'--max_iters={a.teacher_steps}',
         f'--eval_interval={a.teacher_steps}', f'--eval_iters={a.eval_iters}',
         '--log_interval=100', f'--out_dir={out}', '--always_save_checkpoint=True',
         '--compile=False', '--prism_init=False', '--wandb_log=False',
         f'--batch_size={a.batch_size}', f'--block_size={a.block_size}'],
        f's{seed} teacher', a.teacher_steps)
    if rc != 0:
        raise RuntimeError(f'teacher failed (seed {seed}):\n{out_s[-2000:]}')
    e = subprocess.run([sys.executable, 'prism_extract.py', '--ckpt', f'{out}/ckpt.pt',
                        '--out', cache], capture_output=True, text=True, timeout=300)
    if e.returncode != 0:
        raise RuntimeError(f'extract failed:\n{e.stderr[-2000:]}')
    return cache


def run_arm(name, seed, a, device, run_dir, prior_tab, teacher_cache):
    use_prism, use_prior = ARMS[name]
    meta_f = os.path.join(run_dir, f's{seed}_{name}.json')
    if os.path.exists(meta_f):
        r = json.load(open(meta_f))
        log(f'[resume] arm "{name}" (seed {seed}) done (best {r["best"]:.4f}).', 2)
        return r
    extra = []
    if use_prism:
        extra += ['--prism_init=True', '--prism_align=0.75',
                  f'--prism_spectra={teacher_cache}/spectra.json',
                  f'--prism_directions={teacher_cache}/directions.pt',
                  '--prism_mod=0.01', '--prism_mod_decay=0.9999',
                  '--learning_rate=5e-4', '--warmup_iters=50']
    else:
        extra += ['--prism_init=False']
    if use_prior:
        extra += [f'--prior_table={prior_tab}', f'--prior_strength={a.prior_strength}']
        if a.gate_warmup > 0:
            extra += [f'--logit_gate_warmup={a.gate_warmup}']

    r = run_training(name, extra, seed, a.student_steps, a.eval_every, a.eval_iters,
                     device, f's{seed}', batch_size=a.batch_size)
    # init loss = the step-0 eval (free head start)
    curve = {int(k): v for k, v in r['curve'].items()}
    r['init_val'] = curve.get(0)
    json.dump(r, open(meta_f, 'w'), indent=2)
    return r


def run_key(a):
    seeds = '-'.join(str(s) for s in a.seeds)
    return (f"prior_c{a.context}_str{a.prior_strength}_g{a.gate_warmup}_s{a.student_steps}"
            f"_e{a.eval_every}_bs{a.batch_size}_seeds{seeds}"
            + (f"_{a.tag}" if a.tag else ''))


def main():
    p = argparse.ArgumentParser(description='Prior-Fused PRISM (T9 × PRISM)')
    p.add_argument('--context', type=int, default=3)
    p.add_argument('--prior_strength', type=str, default='1.0')
    p.add_argument('--prior_lambdas', default='',
                   help='interpolation weights orders 0..C (e.g. 0.0,0.03,0.12,0.85 = strong)')
    p.add_argument('--gate_warmup', type=int, default=0,
                   help='ramp the model logit contribution 0→1 over N steps (prior arms) so '
                        'the fused init = the pure prior; 0 = off')
    p.add_argument('--teacher_steps', type=int, default=2000)
    p.add_argument('--student_steps', type=int, default=1500)
    p.add_argument('--eval_every', type=int, default=25)
    p.add_argument('--eval_iters', type=int, default=100)
    p.add_argument('--seeds', default='1337')
    p.add_argument('--arms', default='baseline,prism,prior,prism_prior')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--device', default=None)
    p.add_argument('--tag', default='')
    a = p.parse_args()
    a.seeds = [int(s) for s in a.seeds.split(',') if s.strip()]
    arms = [x for x in a.arms.split(',') if x.strip()]
    device = a.device or default_device()
    os.chdir(SRC_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    key = run_key(a)
    run_dir = os.path.join(RUNS_DIR, key)
    os.makedirs(run_dir, exist_ok=True)
    meta_path = os.path.join(run_dir, 'meta.json')
    if os.path.exists(meta_path):
        stamp = json.load(open(meta_path))['stamp']
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        json.dump({'stamp': stamp}, open(meta_path, 'w'))

    print('=' * 64)
    log('PRIOR-FUSED PRISM (T9 × PRISM)')
    print('=' * 64)
    log(f'Run key: {key} | seeds {a.seeds} | arms {arms} | device {device}', 2)

    setup(overlap=None)                          # prepares shakespeare_char
    prior_tab = build_prior(a.context, a.prior_lambdas, '.prism_cache/ngram')
    lock = acquire_lock(run_dir)
    try:
        per_seed = []
        need_teacher = any(ARMS[x][0] for x in arms)
        for seed in a.seeds:
            print('-' * 64)
            log(f'SEED {seed}', 2)
            teacher_cache = train_teacher(seed, a, device, run_dir) if need_teacher else None
            results = {name: run_arm(name, seed, a, device, run_dir, prior_tab, teacher_cache)
                       for name in arms}
            base = results.get('baseline')
            entry = {'seed': seed, 'arms': {}}
            for name, r in results.items():
                sc = compute_score(base, r, a.eval_every) if base else None
                entry['arms'][name] = {
                    'init_val': r.get('init_val'), 'best': r['best'],
                    'best_step': r['best_step'], 'at_end': r['at_end'],
                    'score': sc}
            per_seed.append(entry)
            _persist(stamp, key, a, device, arms, per_seed, complete=False)
        art = _persist(stamp, key, a, device, arms, per_seed, complete=True)
        _report(art)
        log(f'Artifact: results/prior_{stamp}.json (also prior_latest.json)', 2)
        log('COMMIT THIS FILE — it is the evidence for any claim you publish.', 2)
    finally:
        try:
            os.remove(lock)
        except Exception:
            pass


def _persist(stamp, key, a, device, arms, per_seed, complete):
    # baseline best (median) = the target; per-arm speedup medians
    base_step = _stats([e['arms']['baseline']['best_step'] for e in per_seed
                        if 'baseline' in e['arms']]) if per_seed else None
    summ = {}
    for name in arms:
        got = [e['arms'][name] for e in per_seed if name in e['arms']]
        scores = [g['score']['prism_score'] for g in got
                  if g['score'] and g['score']['prism_score'] is not None]
        summ[name] = {
            'init_val': _stats([g['init_val'] for g in got if g['init_val'] is not None]),
            'best': _stats([g['best'] for g in got]),
            'prism_score': _stats(scores) if scores else None,
            'any_reached_at_init': any(g['score'] and g['score'].get('reached_at_init')
                                       for g in got),
            'any_left_censored': any(g['score'] and g['score'].get('left_censored')
                                     for g in got),
        }
    art = {
        'schema': 'prism-prior/1', 'partial': not complete, 'run_key': key,
        'provenance': provenance(),
        'config': {'context': a.context, 'prior_strength': a.prior_strength,
                   'teacher_steps': a.teacher_steps, 'student_steps': a.student_steps,
                   'eval_every': a.eval_every, 'eval_iters': a.eval_iters,
                   'seeds': a.seeds, 'arms': arms, 'batch_size': a.batch_size,
                   'block_size': a.block_size, 'device': device,
                   'note': 'scored as steps-to-baseline-best val loss (nats). loss is '
                           'nanoGPT cross-entropy in NATS; bits/char = nats/ln(2).'},
        'baseline_best_step': base_step,
        'runs': per_seed, 'summary': summ,
    }
    for f in (f'prior_{stamp}.json', 'prior_latest.json'):
        json.dump(art, open(os.path.join(RESULTS_DIR, f), 'w'), indent=2)
    return art


def _report(a):
    import math
    s, c = a['summary'], a['config']
    b2c = 1.0 / math.log(2)
    print()
    print('  ' + '-' * 72)
    print(f'    PRIOR-FUSED PRISM — context-{c["context"]} n-gram × PRISM  '
          f'(steps-to-baseline-best)')
    print('  ' + '-' * 72)
    bs = a.get('baseline_best_step')
    print(f'    seeds {c["seeds"]} · baseline best @ '
          f'{bs["median"] if bs else "?"} steps')
    print(f'    {"arm":12} | {"init loss":>9} {"(bits/ch)":>9} | {"best":>7} | {"speedup":>16}')
    print('  ' + '-' * 72)
    for name in c['arms']:
        g = s[name]
        iv = g['init_val']['median'] if g['init_val'] else float('nan')
        bb = g['best']['median'] if g['best'] else float('nan')
        if g['any_reached_at_init']:
            sp = 'reached at INIT (∞)'
        elif g['prism_score']:
            sp = f'{g["prism_score"]["median"]:.1f}×' + ('*' if g['any_left_censored'] else '')
        else:
            sp = 'never/—'
        print(f'    {name:12} | {iv:>9.3f} {iv * b2c:>9.3f} | {bb:>7.3f} | {sp:>16}')
    print('  ' + '-' * 72)
    print('    init loss = val at step 0 (the free head start). loss in nats; bits/char')
    print('    = nats/ln2. baseline best ≈ 1.78 nats ≈ 2.57 bits/char. "reached at init"')
    print('    = the fused model already predicts at baseline quality before any training.')
    print()


if __name__ == '__main__':
    main()
