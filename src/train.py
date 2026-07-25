"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import json
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from prism_init import spectral_target as _spectral_target

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# prism spectral initialization
prism_init = False # enable Prism (Spectral Imprint + EigenTransfer)
prism_align = 0.75 # UV alignment strength (0 = spectral only, 1 = full alignment)
prism_spectra = '' # path to spectra.json (empty = extract from HF GPT-2)
prism_directions = '' # path to directions.pt (empty = extract from HF GPT-2)
prism_mod = 0.0 # spectral modulation strength (0 = off, e.g. 0.01 = gentle pull)
prism_mod_decay = 0.999 # per-step decay of modulation strength
prism_mod_sustain = 0.0 # sustain phase strength (0 = use single-phase mod)
prism_mod_sustain_decay = 0.9999 # sustain phase decay
prism_mod_transition = 0 # step to switch from attack to sustain (0 = single phase)
prism_unfold = 0 # re-extract spectral targets every N steps (0 = fixed targets)
# prism finetune self-anchor mode (RESUME path only) — what the mod wheel pulls toward:
#   'raw'      the resumed base weights themselves (soft L2-to-init / EWC-lite)
#   'spectral' the base's singular-value SPECTRUM imposed on the CURRENT directions,
#              rebuilt every prism_anchor_refresh steps — holds the spectral shape,
#              frees U/V to adapt. The attribution test for PRISM's geometry thesis.
#   'shuffled' same as spectral but with the base spectrum permuted (placebo: same
#              spectral pressure, wrong spectrum-to-direction assignment)
prism_anchor_mode = 'raw'
prism_anchor_refresh = 25 # spectral/shuffled: rebuild the target every N steps
# prism direction-transfer knobs (opt-in; defaults reproduce the recipe exactly)
prism_align_spec = '' # per-group alignment, e.g. 'attention:0.9,ffn_down:0.5'
prism_align_mode = 'linear' # 'linear' | 'grassmann' | 'subspace'
prism_align_topk = 0 # transfer only the leading k singular directions (0 = all)
prism_align_depth_gamma = 0.0 # taper align with depth: base*(1-gamma*depth_frac)
prism_per_layer_spectra = '' # path to spectra_per_layer.json (empty = group avg)
# prism CKA representational regularizer (opt-in, experimental)
prism_cka = 0.0 # weight on the (1 - CKA) representational-distance loss (0 = off)
prism_cka_teacher = '' # path to a teacher ckpt.pt whose block activations to match
prism_cka_layers = '' # comma block indices to match (empty = all blocks)
prism_cka_samples = 2048 # max token rows subsampled per layer for the CKA estimate
# finetune dual-val: a second held-out val set scored alongside val (retention).
# Empty = single-val (unchanged). Used by the finetune benchmark to watch the
# OLD domain's loss (forgetting) while training on the NEW domain.
val2_dir = '' # path to a dataset dir containing val.bin (same vocab/meta as the model)
# matched-quality early stop: stop training the moment val loss reaches this target
# (0 = off). Lets bases trained by different methods (plain vs Prism) be compared at
# EQUAL old-domain quality — the control for the arc / base-interaction study.
stop_val_target = 0.0
# T9-style fixed shared n-gram prior fused into the logits (product of experts):
#   final_logits = model_logits + prior_strength * log p_ngram(next | last C chars)
# The model then learns only the RESIDUAL over the prior. prior_table = a .pt from
# build_ngram_prior.py (dense (V^C, V) log-prob table + context_len). 0 strength = off
# (byte-identical to a plain run).
prior_table = ''
prior_strength = 0.0
# logit gate: scale the MODEL's logit contribution by min(1, iter/warmup) so it ramps
# from 0 → 1. At init the fused output is the PURE prior (no random-logit noise), so a
# prior that already matches the baseline puts the model at baseline quality before any
# training. 0 = no gate (model contributes fully from step 0).
logit_gate_warmup = 0
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
seed = 1337 # RNG seed; vary across runs to measure seed variance
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    elif split == 'val2':
        # retention val: a held-out set from a DIFFERENT dataset dir (the old domain)
        data = np.memmap(os.path.join(val2_dir, 'val.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# Prism spectral initialization (after model creation, before compile)
if prism_init and init_from == 'scratch':
    from prism_init import apply_prism
    print(f"Applying Prism init (align={prism_align})...")
    apply_prism(model, align_strength=prism_align, lam=1.0,
                spectra_path=prism_spectra or None,
                directions_path=prism_directions or None,
                align_spec=prism_align_spec or None,
                align_mode=prism_align_mode,
                align_topk=prism_align_topk,
                align_depth_gamma=prism_align_depth_gamma,
                per_layer_spectra_path=prism_per_layer_spectra or None,
                n_layer=n_layer)
    print("Prism init complete.")
    # Capture spectral targets for modulation (before compile changes the model)
    if prism_mod > 0:
        prism_targets = {name: param.data.clone().cpu()
                         for name, param in model.named_parameters()
                         if param.dim() >= 2}
        print(f"[prism] Captured {len(prism_targets)} spectral targets for modulation "
              f"(strength={prism_mod}, decay={prism_mod_decay})")

# Prism self-anchor on RESUME — engage the mod wheel during FINETUNING. The scratch
# path above captures targets from a fresh Prism-shaped model; here we instead
# capture them from the RESUMED (already-trained) weights, so the mod wheel holds
# the model in its own converged geometry while it learns new data. This tests
# whether the no-drift property that prevents overfitting from scratch also
# prevents catastrophic forgetting during finetuning. Use prism_mod_decay=1.0 for
# a constant pull (the scratch decay anneals a reshape that isn't happening here).
elif prism_mod > 0 and init_from == 'resume':
    if prism_anchor_mode == 'raw':
        prism_targets = {name: param.data.clone().cpu()
                         for name, param in model.named_parameters()
                         if param.dim() >= 2}
    else:
        # spectral / shuffled: store the base spectrum per weight; the initial target
        # imposes it on the base's OWN directions (== base weight for 'spectral'; a
        # permuted-spectrum placebo for 'shuffled'). Refreshed during training below.
        prism_base_sv = {}
        prism_targets = {}
        _g = torch.Generator().manual_seed(seed)
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
            U, s, Vt = torch.linalg.svd(param.data.float(), full_matrices=False)
            sv0 = s.clone()
            if prism_anchor_mode == 'shuffled':
                sv0 = sv0[torch.randperm(sv0.shape[0], generator=_g)]
            prism_base_sv[name] = sv0.cpu()
            prism_targets[name] = ((U * sv0) @ Vt).to(param.dtype).cpu()
    print(f"[prism] Self-anchored {len(prism_targets)} targets from resumed ckpt "
          f"(mode={prism_anchor_mode}, strength={prism_mod}, decay={prism_mod_decay}"
          + (f", refresh={prism_anchor_refresh}" if prism_anchor_mode != 'raw' else '') + ')')

# Prism CKA representational regularizer (opt-in) — pull student block activations
# toward a frozen teacher's. Set up before compile so hooks fire on the raw model.
cka_matcher = None
if prism_cka > 0 and prism_cka_teacher:
    from prism_cka import CKAMatcher
    _cka_layers = [int(x) for x in prism_cka_layers.split(',') if x.strip() != ''] or None
    cka_matcher = CKAMatcher(prism_cka_teacher, device, layers=_cka_layers,
                             max_samples=prism_cka_samples)
    cka_matcher.attach_student(model)
    print(f"[prism] CKA regularizer on (weight={prism_cka}, "
          f"teacher={prism_cka_teacher}, blocks={cka_matcher.layers})")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# T9-style fixed n-gram prior fused into the logits (product of experts). Loaded once;
# gathered per position by context index. prior_strength=0 → disabled (plain loss path).
_prior_logtab = None
_prior_C = 0
_cur_gate = 1.0                 # model-logit gate (ramped in the loop if logit_gate_warmup>0)
if prior_table and prior_strength > 0:
    _pp = torch.load(prior_table, map_location=device, weights_only=False)
    _prior_logtab = _pp['table'].to(device)               # (V^C, V) log-probs
    _prior_C = int(_pp['context_len'])
    assert _pp['vocab_size'] == model_args['vocab_size'], 'prior vocab != model vocab'
    print(f"[prior] fused n-gram prior context={_prior_C} strength={prior_strength} "
          f"(standalone val {_pp.get('val_bits_per_char', float('nan')):.3f} bits/char)")


def _prior_logp(X):
    """(B,T,V) prior log-probs for predicting the next char at each position, gathered by
    the C-char context ending at each position. First C-1 positions get 0 (uniform)."""
    B, T = X.shape
    idx = torch.zeros(B, T, dtype=torch.long, device=X.device)
    V = model_args['vocab_size']
    for off in range(_prior_C - 1, -1, -1):               # oldest char first
        if off > 0:
            sh = torch.zeros_like(X)
            sh[:, off:] = X[:, :-off]
        else:
            sh = X
        idx = idx * V + sh
    lp = _prior_logtab[idx]                                # (B,T,V)
    if _prior_C > 1:
        lp[:, :_prior_C - 1, :] = 0.0                      # incomplete context → uniform
    return lp


def _loss(logits, X, Y, model_loss):
    """Fused cross-entropy when a prior is active, else the model's own loss."""
    if _prior_logtab is None:
        return model_loss
    fused = _cur_gate * logits + prior_strength * _prior_logp(X)
    return F.cross_entropy(fused.view(-1, fused.size(-1)), Y.view(-1), ignore_index=-1)


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in (['train', 'val', 'val2'] if val2_dir else ['train', 'val']):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = _loss(logits, X, Y, loss).item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # logit gate: ramp the model's contribution 0 → 1 (updated before the eval below, so
    # the step-0 eval reflects the pure prior). Only meaningful when a prior is fused.
    if logit_gate_warmup > 0:
        _cur_gate = min(1.0, iter_num / logit_gate_warmup)

    # evaluate the loss on train/val sets and write checkpoints. Also force an eval on
    # the first step (captures retention_at_base when resuming a finetune) and the last
    # step, so the eval cadence can be decoupled from the (arbitrary) resume/stop step.
    if (iter_num % eval_interval == 0 or local_iter_num == 0
            or iter_num == max_iters) and master_process:
        losses = estimate_loss()
        line = f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        if val2_dir:
            line += f", val2 loss {losses['val2']:.4f}"
        print(line)
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        # matched-quality early stop: bank the checkpoint the moment val reaches the
        # target and stop, so bases trained by different methods can be compared at
        # equal old-domain quality (the arc / base-interaction control).
        if stop_val_target > 0 and losses['val'] <= stop_val_target and iter_num > 0:
            ckpt = {'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'model_args': model_args, 'iter_num': iter_num,
                    'best_val_loss': float(min(best_val_loss, losses['val'])), 'config': config}
            torch.save(ckpt, os.path.join(out_dir, 'ckpt.pt'))
            print(f"[stop_val_target] val {losses['val']:.4f} <= {stop_val_target} "
                  f"at step {iter_num} — stopping")
            break
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = _loss(logits, X, Y, loss)   # fuse the T9 n-gram prior if active
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # representational transfer: pull student block activations toward the
            # teacher's on this same batch (linear-CKA distance). Uses the acts the
            # forward above just captured, so it must run before X is reassigned.
            if cka_matcher is not None:
                cka = cka_matcher.loss(X)
                if cka is not None:
                    loss = loss + (prism_cka * cka) / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # spectral modulation — pull weights toward spectral targets (the "mod wheel")
    # Two-phase ADSR: attack phase (strong, fast decay) → sustain phase (gentle, slow decay)
    if prism_mod > 0 and 'prism_targets' in dir():
        if prism_mod_transition > 0 and iter_num >= prism_mod_transition:
            # Sustain phase
            steps_in_sustain = iter_num - prism_mod_transition
            current_mod = prism_mod_sustain * (prism_mod_sustain_decay ** steps_in_sustain)
        else:
            # Attack phase
            current_mod = prism_mod * (prism_mod_decay ** iter_num)
        if current_mod > 1e-6:
            raw_model = model.module if ddp else model
            with torch.no_grad():
                for name, param in raw_model.named_parameters():
                    if name in prism_targets:
                        param.data.lerp_(prism_targets[name].to(param.device), current_mod)

    # spectral unfolding — re-extract targets from current model
    if prism_unfold > 0 and 'prism_targets' in dir() and iter_num > 0 and iter_num % prism_unfold == 0:
        raw_model = model.module if ddp else model
        with torch.no_grad():
            prism_targets = {name: param.data.clone().cpu()
                             for name, param in raw_model.named_parameters()
                             if param.dim() >= 2}
        if master_process:
            print(f"[prism] Unfolded: re-extracted {len(prism_targets)} spectral targets at step {iter_num}")

    # spectral-anchor refresh — rebuild each target as (current directions × base
    # spectrum), so the finetune's U/V stay free while the spectral shape is held to
    # the base's. This is what makes 'spectral' differ from the 'raw' fixed-weight pull.
    if (prism_anchor_mode in ('spectral', 'shuffled') and 'prism_base_sv' in dir()
            and prism_anchor_refresh > 0 and iter_num > 0
            and iter_num % prism_anchor_refresh == 0):
        raw_model = model.module if ddp else model
        with torch.no_grad():
            for name, param in raw_model.named_parameters():
                if name in prism_base_sv:
                    prism_targets[name] = _spectral_target(
                        param.data, prism_base_sv[name].to(param.device)).cpu()

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
