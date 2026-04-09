# Prism Sprint: 4.8x faster to baseline quality
# Use when iterating fast. Stop training at step 500-600.
#
# python train.py config/train_shakespeare_char.py config/prism_sprint.py

prism_init = True
prism_align = 0.75
prism_spectra = '.prism_cache/shakespeare/spectra.json'
prism_directions = '.prism_cache/shakespeare/directions.pt'

learning_rate = 5e-4
warmup_iters = 50

prism_mod = 0.005
prism_mod_decay = 0.999
