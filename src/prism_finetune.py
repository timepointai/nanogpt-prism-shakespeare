"""
prism_finetune.py — Finetune a trained model on new data WITHOUT losing its
advantage. The "how" to the question "can we finetune without losing it?": keep
the Mod Wheel ON during the finetune, self-anchored to the model's own converged
geometry, so it learns the new data while it is held in the trained spectral shape
(the shape that, from scratch, is what stops it overfitting).

    python prism_finetune.py \
        --base_ckpt out-shakespeare-char/ckpt.pt \
        --new_data sherlock_ft \
        --retain_val shakespeare_char \
        --out_dir out-finetuned \
        -- --dropout=0.1

This resumes the checkpoint and finetunes on data/<new_data> with prism_mod on
(self-anchor, constant pull). Pass --retain_val <old dataset dir> to watch the OLD
domain's loss (val2) alongside the new one, so you can SEE the forgetting the mod
wheel is preventing. Re-run with --plain to see the same finetune with the wheel
OFF (the forgetting/overfitting control). Everything after `--` passes through to
train.py.

The teacher/base and the finetune must share architecture + vocabulary (resume
forces the model dims from the checkpoint). Measured on nanoGPT Shakespeare→Sherlock
by prism_finetune_eval.py — see RESULTS once that run is committed.
"""
import argparse
import os
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(
        description='Finetune a checkpoint with the Mod Wheel self-anchored (or off).')
    p.add_argument('--base_ckpt', required=True, help='a trained nanoGPT ckpt.pt to finetune')
    p.add_argument('--new_data', required=True, help='dataset dir under data/ to finetune on')
    p.add_argument('--retain_val', default='',
                   help='old-domain dataset dir (under data/) to score as val2 — shows forgetting')
    p.add_argument('--out_dir', default='out-finetuned')
    p.add_argument('--config', default='config/train_shakespeare_char.py')
    p.add_argument('--mod', default='0.01', help='mod-wheel pull strength')
    p.add_argument('--mod_decay', default='1.0', help='per-step decay (1.0 = constant pull)')
    p.add_argument('--lr', default='3e-4')
    p.add_argument('--ft_steps', type=int, default=1000)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--plain', action='store_true',
                   help='mod wheel OFF — the plain finetune (forgetting control)')
    args, passthrough = p.parse_known_args()
    if passthrough and passthrough[0] == '--':
        passthrough = passthrough[1:]

    if not os.path.exists(args.base_ckpt):
        sys.exit(f'base checkpoint not found: {args.base_ckpt}')

    # The Mod Wheel decays on the GLOBAL iter (which resumes at the base's step
    # count), so warmup / decay must be offset by the base's iter_num.
    import torch
    base_steps = torch.load(args.base_ckpt, map_location='cpu',
                            weights_only=False)['iter_num']
    print(f'[prism-finetune] base checkpoint trained {base_steps} steps; '
          f'finetuning {args.ft_steps} more '
          f'({"PLAIN — mod wheel OFF" if args.plain else "mod wheel ON, self-anchored"}).')

    os.makedirs(args.out_dir, exist_ok=True)
    shutil.copy(args.base_ckpt, os.path.join(args.out_dir, 'ckpt.pt'))    # fork

    cmd = [sys.executable, 'train.py', args.config, '--init_from=resume',
           f'--out_dir={args.out_dir}', f'--dataset={args.new_data}',
           f'--max_iters={base_steps + args.ft_steps}',
           f'--warmup_iters={base_steps + args.warmup}',
           f'--lr_decay_iters={base_steps + args.ft_steps}',
           f'--learning_rate={args.lr}', '--compile=False', '--wandb_log=False']
    if args.retain_val:
        rv = args.retain_val if args.retain_val.startswith('data/') else f'data/{args.retain_val}'
        cmd.append(f'--val2_dir={rv}')
    if args.plain:
        cmd.append('--prism_mod=0')
    else:
        cmd += [f'--prism_mod={args.mod}', f'--prism_mod_decay={args.mod_decay}']
    cmd += passthrough

    print('      ' + ' '.join(cmd[2:]))
    subprocess.run(cmd, check=True)
    print(f'\nDone → {args.out_dir}/. '
          + ('Re-run WITHOUT --plain to finetune with the mod wheel on.'
             if args.plain else
             'Re-run with --plain (and --retain_val) to see the plain finetune '
             'forget/overfit where this one did not.'))


if __name__ == '__main__':
    main()
