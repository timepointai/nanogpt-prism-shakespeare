# Prism Marathon: best quality, no overfitting
# Use for final training runs. Keeps improving through 5K+ steps.
#
# python train.py config/train_shakespeare_char.py config/prism_marathon.py

prism_init = True
prism_align = 0.75
prism_spectra = '.prism_cache/shakespeare/spectra.json'
prism_directions = '.prism_cache/shakespeare/directions.pt'

learning_rate = 5e-4
warmup_iters = 50

prism_mod = 0.01
prism_mod_decay = 0.9999
