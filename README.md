# Vision Transformer (ViT) from Scratch

PyTorch implementation of **"An Image is Worth 16x16 Words"** (Dosovitskiy et al., ICLR 2021).

Built from scratch for learning — every component is self-contained with paper equation references.

## Architecture

```
Input Image (32×32)
      │
      ▼
┌─────────────────┐
│  Patch Embedding │  32×32 → 64 patches (4×4 each) → Linear(48→256)
└────────┬────────┘
         │  prepend [CLS] token
         ▼
┌─────────────────┐
│ + Pos Embedding │  learnable 1D positional encoding
└────────┬────────┘
         │
         ▼  × 6
┌─────────────────────────────┐
│      Transformer Block       │
│  LayerNorm → MultiHeadAttn  │  8 heads, d_model=256
│  + residual                  │
│  LayerNorm → MLP(256→512)   │  GELU activation
│  + residual                  │
└────────┬────────────────────┘
         │  [CLS] token only
         ▼
┌─────────────────┐
│   LayerNorm     │
│   Linear(256→10)│  Classification head
└─────────────────┘
```

**ViT-Small config** tuned for CIFAR-10 (32×32):

| Param | Value |
|---|---|
| patch_size | 4 |
| d_model | 256 |
| n_heads | 8 |
| n_layers | 6 |
| mlp_dim | 512 |
| params | ~9M |

## Quick Start

```bash
# Install
git clone https://github.com/yamadan96/vit-from-scratch
cd vit-from-scratch
uv sync

# Train on CIFAR-10 (downloads automatically)
CHECKPOINT_DIR=./checkpoints uv run python -m src.train --epochs 100

# With Weights & Biases logging
WANDB_PROJECT=vit-cifar10 CHECKPOINT_DIR=./checkpoints uv run python -m src.train

# Launch Gradio demo
CHECKPOINT_DIR=./checkpoints uv run python app.py
```

## Project Structure

```
vit-from-scratch/
├── src/
│   ├── model.py      # ViT: PatchEmbedding / TransformerBlock / VisionTransformer
│   ├── train.py      # Training loop: AdamW + CosineAnnealing + AMP
│   └── predictor.py  # Singleton predictor for inference
└── app.py            # Gradio WebApp — top-3 CIFAR-10 classification
```

## Key Implementation Details

- **`PatchEmbedding`** — reshapes image into non-overlapping patches, projects to `d_model`
- **`TransformerBlock`** — Pre-Norm (LN before attention, as in the paper)
- **`[CLS] token`** — prepended learnable token; its final hidden state is used for classification
- **Positional encoding** — learned 1D embeddings (not sinusoidal)
- No pre-training; trained from scratch on CIFAR-10

## Training

| Setting | Value |
|---|---|
| Dataset | CIFAR-10 (50k train / 10k val) |
| Optimizer | AdamW (lr=1e-3, wd=0.1) |
| Scheduler | CosineAnnealingLR |
| Epochs | 100 |
| Mixed precision | ✅ (torch.cuda.amp) |

## References

- Dosovitskiy et al. (2021). [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). ICLR 2021.

## License

MIT
