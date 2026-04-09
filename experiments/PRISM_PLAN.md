# Prism × nanoGPT Integration Plan

## Goal

Show that Prism (Spectral Imprint + EigenTransfer) accelerates GPT-2 124M
training from scratch on the nanoGPT benchmark, measured as lower val loss
at the same step count.

## What We Know (from CUDA sweep, April 2026)

- Prism achieves **2.74x faster convergence** at 750 steps on WikiText-2
  (batch 64, seq 1024, A100), using align=0.75, LR=1.5x, spike_skip=50
- The advantage is real at CUDA scale but needs LR tuning — 2x LR overfits,
  1.5x is the sweet spot
- Spectral shape alone gives ~1.4x; UV alignment adds another ~2x on top
- Results pending on whether 2.74x holds past 750 steps (validation running)

## Key Differences: Our Setup vs nanoGPT

| | Our CUDA experiments | nanoGPT default |
|---|---|---|
| Framework | HuggingFace GPT2LMHeadModel | Custom model.py (~330 lines) |
| Dataset | WikiText-2 (~2M tokens) | OpenWebText (~9B tokens) |
| Batch size | 64 (4×16) | ~491K tokens/iter (8 GPU) |
| Sequence length | 1024 | 1024 |
| Peak LR | 6.25e-5 (base), 9.38e-5 (Prism) | **6e-4** (10x higher!) |
| Warmup | 200-300 steps | 2000 steps |
| Total steps | 750-2000 | 600,000 |
| Vocab | 50257 | 50304 (padded) |
| Weight storage | HF Conv1D (transposed) | nn.Linear |
| Dropout | 0.0 | 0.0 |
| Weight decay | 0.01 | 0.1 |

**The LR difference is critical.** nanoGPT already uses 6e-4, which is 10x
our base LR. Our sweep found 1.5x over base is optimal. For nanoGPT, the
equivalent would be 9e-4 — but this might overfit given the already-high
base. We may need to keep nanoGPT's default LR and let Prism provide
advantage through better convergence at the same LR, not through LR boost.

## Integration Points

### 1. prism_init.py (DONE)

Self-contained module. Extracts spectra from HF GPT-2, caches to disk,
applies to nanoGPT model. Handles:
- Conv1D transpose (HF → nanoGPT weight convention)
- Vocab padding (50257 → 50304)
- Weight tying (wte = lm_head, init once)
- Group classification matching both naming conventions

### 2. train.py modification (~5 lines)

```python
# After model creation (~line 150), before torch.compile (~line 195):
if prism_init:
    from prism_init import apply_prism
    apply_prism(model, align_strength=prism_align, lam=1.0)
    # Re-apply residual scaling after Prism init
    for pn, p in model.named_parameters():
        if pn.endswith('c_proj.weight'):
            # Prism already set the spectral shape; just scale the norm
            with torch.no_grad():
                scale = 0.02 / math.sqrt(2 * config.n_layer)
                current_std = p.std().item()
                if current_std > 0:
                    p.mul_(scale / current_std)
```

### 3. config/train_gpt2_prism.py

New config file for Prism experiments. Based on train_gpt2.py with:
- `prism_init = True`
- `prism_align = 0.75`
- Potentially different LR/warmup (TBD based on initial experiments)
- wandb logging enabled for comparison curves

## Experiment Plan

### Phase 1: Smoke test (1 GPU, 1K steps)

Verify Prism init works with nanoGPT, no crashes, loss decreases.

```bash
python train.py config/train_gpt2_prism.py \
    --max_iters=1000 --eval_interval=100 \
    --wandb_log=True --wandb_project=prism-nanogpt
```

Compare val loss at steps 100, 250, 500, 1000 against baseline.

### Phase 2: Medium run (1 GPU, 10K steps)

Enough to see if the advantage holds through early training.

### Phase 3: Full benchmark (8 GPU, 600K steps)

Only if Phase 1-2 show signal. Match the canonical nanoGPT setup exactly.

### Phase 4: Config sweep on nanoGPT

If Phase 1-2 work, sweep alignment strength and LR on nanoGPT specifically.
The 6e-4 base LR changes the dynamics — we may find a different sweet spot.

## Open Questions

1. Should we keep nanoGPT's LR (6e-4) or adjust it?
2. Does the residual scaling interact badly with Prism's spectral shape?
3. Does the fused c_attn (768, 2304) need special handling vs separate Q/K/V?
4. Does OpenWebText (9B tokens, 33 epochs) change the overfitting dynamics
   vs WikiText-2 (2M tokens)?
5. Should we use nanoGPT's weight decay (0.1) or our sweep's default (0.01)?

## DO NOT

- PR to karpathy/nanoGPT
- Create branches on karpathy/nanoGPT
- Push to any remote that is not timepointai/nanogpt-prism-shakespeare
- Claim benchmark results without full 600K step reproduction
