# config/train_gpt2_prism.py
# nanoGPT GPT-2 (124M) training with Prism spectral initialization.
#
# Based on config/train_gpt2.py with Prism-specific additions.
# Run: python train.py config/train_gpt2_prism.py

# --- Prism config ---
prism_init = True
prism_align = 0.75    # UV alignment strength (sweep winner)

# --- Model ---
# same as train_gpt2.py
n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024
bias = False
dropout = 0.0

# --- Training ---
# Start with nanoGPT defaults; may adjust after initial experiments
batch_size = 12
gradient_accumulation_steps = 5 * 8  # 5 per GPU × 8 GPUs = 40 total
max_iters = 600000
lr_decay_iters = 600000

# --- Optimizer ---
learning_rate = 6e-4     # nanoGPT default (NOT our 1.5x; test default first)
min_lr = 6e-5
warmup_iters = 2000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# --- Eval ---
eval_interval = 1000
eval_iters = 200
log_interval = 10

# --- Logging ---
wandb_log = True
wandb_project = 'prism-nanogpt'
wandb_run_name = 'prism-gpt2-124m'

# --- I/O ---
out_dir = 'out-prism-gpt2'
dataset = 'openwebtext'
init_from = 'scratch'

# --- System ---
compile = True
