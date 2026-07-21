"""
prism_accelerate.py — Use a trained model's spectral geometry to accelerate
training a fresh one. This is the "apply PRISM" entry point: extract a fingerprint
from any nanoGPT checkpoint, then train a new model initialized + regularized by it.

    python prism_accelerate.py \
        --teacher_ckpt out-teacher/ckpt.pt \
        --config config/train_shakespeare_char.py \
        --out_dir out-accelerated \
        -- --max_iters=2000 --dataset=shakespeare_char

Everything after `--` is passed straight through to train.py, so any nanoGPT
config key works. What PRISM adds (the proven recipe) is applied for you:
initialization from the teacher's spectral geometry + the mod-wheel regularizer.

What to expect (measured on nanoGPT Shakespeare, 3 seeds — see RESULTS.md):
  · reaches the from-scratch baseline's best quality several times faster
  · a structural head start (~20%+ lower loss early) that does NOT require the
    new data to overlap the teacher's — it is geometry, not memorized content
  · it does not overfit where a from-scratch baseline does

The teacher and the new model must share the SAME architecture (the directional
transfer is dimension-specific). Cross-size transfer is not yet supported.
"""
import argparse
import os
import subprocess
import sys

# The proven recipe (config/prism_recipe.py). --matched_lr drops the LR override
# so the ONLY change from a from-scratch run is the spectral machinery.
RECIPE = ['--prism_init=True', '--prism_align=0.75',
          '--prism_mod=0.01', '--prism_mod_decay=0.9999',
          '--learning_rate=5e-4', '--warmup_iters=50']


def main():
    p = argparse.ArgumentParser(
        description='Accelerate training with a spectral fingerprint from a checkpoint.')
    p.add_argument('--teacher_ckpt', required=True,
                   help='path to a trained nanoGPT checkpoint (ckpt.pt)')
    p.add_argument('--config', default='config/train_shakespeare_char.py',
                   help='base train.py config for the new model')
    p.add_argument('--out_dir', default='out-prism',
                   help='where the accelerated run writes its checkpoints')
    p.add_argument('--fingerprint', default='.prism_cache/accelerate',
                   help='where to store the extracted spectral fingerprint')
    p.add_argument('--matched_lr', action='store_true',
                   help='keep the config learning rate instead of the tuned 5e-4 '
                        '(so only the spectral flags differ from a from-scratch run)')
    args, passthrough = p.parse_known_args()
    if passthrough and passthrough[0] == '--':      # drop the separator
        passthrough = passthrough[1:]

    if not os.path.exists(args.teacher_ckpt):
        sys.exit(f'teacher checkpoint not found: {args.teacher_ckpt}')

    # 1) Extract the spectral fingerprint (128-byte spectrum + directions).
    print(f'[1/2] extracting fingerprint  {args.teacher_ckpt} → {args.fingerprint}/')
    subprocess.run([sys.executable, 'prism_extract.py',
                    '--ckpt', args.teacher_ckpt, '--out', args.fingerprint],
                   check=True)

    # 2) Accelerated training, initialized + regularized by the fingerprint.
    recipe = [a for a in RECIPE if not a.startswith('--learning_rate')] \
        if args.matched_lr else list(RECIPE)
    cmd = [sys.executable, 'train.py', args.config,
           f'--out_dir={args.out_dir}',
           f'--prism_spectra={args.fingerprint}/spectra.json',
           f'--prism_directions={args.fingerprint}/directions.pt'] \
        + recipe + passthrough
    print(f'[2/2] accelerated training → {args.out_dir}/')
    print('      ' + ' '.join(cmd[2:]))
    subprocess.run(cmd, check=True)
    print(f'\nDone. Accelerated model in {args.out_dir}/. Compare it against a '
          f'from-scratch run (drop the --prism_* flags) to see the speedup on '
          f'your own data.')


if __name__ == '__main__':
    main()
