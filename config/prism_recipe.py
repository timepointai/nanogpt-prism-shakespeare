# The Prism Recipe
#
# 13x faster to baseline quality. Best val loss baseline never reaches.
# Zero overfitting. Validated on held-out test data.
#
# Requires teacher spectra + directions:
#   python prism_extract.py --ckpt path/to/teacher.pt --out .prism_cache/teacher
#
# Usage:
#   python train.py config/train_shakespeare_char.py config/prism_recipe.py

prism_init = True
prism_align = 0.75
prism_spectra = '.prism_cache/teacher/spectra.json'
prism_directions = '.prism_cache/teacher/directions.pt'

learning_rate = 5e-4
warmup_iters = 50

prism_mod = 0.01
prism_mod_decay = 0.9999
