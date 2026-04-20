"""
Vision Transformer (ViT) -- paper-faithful PyTorch implementation.

Reference: "An Image is Worth 16x16 Words: Transformers for Image Recognition
at Scale" (Dosovitskiy et al., ICLR 2021)

This module implements the core ViT architecture with educational comments
explaining the mathematical foundations from the original paper.
"""

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ViTConfig:
    """ViT-Small configuration tuned for CIFAR-10 (32x32 images).

    The original paper uses ViT-Base/16 (D=768, 12 heads, 12 layers)
    on 224x224 images. This is a scaled-down variant appropriate for
    CIFAR-10's 32x32 resolution.
    """

    image_size: int = 32
    patch_size: int = 4  # 32/4 = 8 patches per side -> N = 64 patches
    in_channels: int = 3
    num_classes: int = 10
    d_model: int = 256  # embedding dimension D
    n_heads: int = 8  # number of attention heads
    n_layers: int = 6  # number of Transformer encoder blocks
    mlp_dim: int = 512  # feed-forward hidden dimension (MLP ratio ~2x)
    dropout: float = 0.1


class PatchEmbedding(nn.Module):
    """Split an image into non-overlapping patches and linearly project each.

    From Section 3.1 of the paper:
        "We reshape the image x in R^(H x W x C) into a sequence of
         flattened 2D patches x_p in R^(N x (P^2 * C)), where (H, W) is
         the resolution of the original image, C is the number of channels,
         (P, P) is the resolution of each image patch, and
         N = HW / P^2 is the resulting number of patches."

    The linear projection E in R^((P^2*C) x D) maps each flattened patch
    to a D-dimensional embedding vector.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.n_patches = (config.image_size // config.patch_size) ** 2
        patch_dim = config.in_channels * config.patch_size**2

        # Linear projection E: R^(P^2*C) -> R^D
        # This is equivalent to a Conv2d with kernel_size=stride=patch_size,
        # but we use a Linear layer to stay faithful to the paper's notation.
        self.projection = nn.Linear(patch_dim, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image tensor to sequence of patch embeddings.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Patch embeddings [B, N, D] where N = (H/P)*(W/P)
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # Reshape: [B, C, H, W] -> [B, C, H/P, P, W/P, P]
        x = x.reshape(B, C, H // p, p, W // p, p)

        # Permute to group spatial patch indices together:
        # [B, C, H/P, P, W/P, P] -> [B, H/P, W/P, C, P, P]
        x = x.permute(0, 2, 4, 1, 3, 5)

        # Flatten patches: [B, H/P, W/P, C, P, P] -> [B, N, P^2*C]
        x = x.reshape(B, -1, C * p * p)

        # Project each flattened patch to D dimensions: [B, N, D]
        return self.projection(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention (MSA) as described in Section 3.1.

    From the paper:
        "Multihead self-attention (MSA) ... standard qkv self-attention
         (Vaswani et al., 2017)"

    Each head computes:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    where d_k = D / num_heads is the per-head dimension.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        if config.d_model % config.n_heads != 0:
            raise ValueError(
                f"d_model ({config.d_model}) must be divisible by "
                f"n_heads ({config.n_heads})"
            )

        # Combined QKV projection for efficiency: W_qkv in R^(D x 3D)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)

        # Scale factor: 1 / sqrt(d_k) for scaled dot-product attention
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-head self-attention.

        Args:
            x: Input tensor [B, N+1, D] (N patches + 1 CLS token)

        Returns:
            Attended output [B, N+1, D]
        """
        B, seq_len, _ = x.shape

        # Compute Q, K, V in one pass: [B, N+1, 3D]
        qkv = self.qkv(x)

        # Reshape to [B, N+1, 3, n_heads, head_dim] then permute
        qkv = qkv.reshape(B, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, n_heads, N+1, head_dim]
        q, k, v = qkv.unbind(0)  # Each: [B, n_heads, N+1, head_dim]

        # Scaled dot-product attention:
        # attn = softmax(Q K^T / sqrt(d_k))
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values: [B, n_heads, N+1, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads: [B, N+1, D]
        attn_output = attn_output.transpose(1, 2).reshape(B, seq_len, self.d_model)

        # Final linear projection and dropout
        return self.proj_dropout(self.proj(attn_output))


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block with Pre-Norm (Section 3.1).

    The paper uses Pre-LayerNorm (LN applied before each sub-layer):
        z'_l = MSA(LN(z_{l-1})) + z_{l-1}    (Eq. 2)
        z_l  = MLP(LN(z'_l))   + z'_l         (Eq. 3)

    This differs from the original Transformer (Vaswani et al.) which
    applies LN after the residual connection (Post-Norm).
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        # Pre-norm before self-attention
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)

        # Pre-norm before feed-forward
        self.norm2 = nn.LayerNorm(config.d_model)

        # MLP: two linear layers with GELU activation (Eq. 3)
        # "The MLP contains two layers with a GELU non-linearity."
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.mlp_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one Transformer encoder block.

        Args:
            x: Input tensor [B, N+1, D]

        Returns:
            Output tensor [B, N+1, D]
        """
        # Eq. 2: Multi-head self-attention with pre-norm and residual
        x = x + self.attn(self.norm1(x))

        # Eq. 3: Feed-forward network with pre-norm and residual
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) for image classification.

    Full architecture (Section 3.1):
        1. Split image into patches and linearly embed each patch
        2. Prepend a learnable [CLS] token to the sequence
        3. Add learnable 1D positional embeddings
        4. Pass through L Transformer encoder blocks
        5. Apply LayerNorm to [CLS] token output
        6. Classify via MLP head on [CLS] representation

    Equation from the paper:
        z_0 = [x_class; x_p^1 E; x_p^2 E; ...; x_p^N E] + E_pos   (Eq. 1)
        z_l = TransformerBlock(z_{l-1}),  l = 1...L                  (Eq. 2-3)
        y   = LN(z_L^0)                                              (Eq. 4)
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config

        # Patch embedding: image -> sequence of patch embeddings
        self.patch_embed = PatchEmbedding(config)

        # Learnable [CLS] token: x_class in R^D
        # "A learnable embedding is prepended to the sequence of embedded
        #  patches ... whose state at the output serves as the image
        #  representation y." (Section 3.1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))

        # Learnable 1D positional embeddings: E_pos in R^((N+1) x D)
        # "Position embeddings are added to the patch embeddings to retain
        #  positional information." (Section 3.1)
        n_positions = self.patch_embed.n_patches + 1  # N patches + 1 CLS
        self.pos_embed = nn.Parameter(torch.zeros(1, n_positions, config.d_model))

        self.embed_dropout = nn.Dropout(config.dropout)

        # Transformer encoder: L identical blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final LayerNorm applied to [CLS] output (Eq. 4)
        self.norm = nn.LayerNorm(config.d_model)

        # Classification head: linear projection from D to num_classes
        # "The classification head is implemented by a MLP with one hidden
        #  layer at pre-training time and by a single linear layer at
        #  fine-tuning time." (Section 3.1)
        # We use a single linear layer (fine-tuning style) for simplicity.
        self.head = nn.Linear(config.d_model, config.num_classes)

        self._init_weights()
        self._log_model_info()

    def _init_weights(self) -> None:
        """Initialize weights following ViT conventions.

        Positional embeddings and CLS token use truncated normal (std=0.02).
        Linear layers use truncated normal, biases initialized to zero.
        LayerNorm uses ones (weight) and zeros (bias).
        """
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _log_model_info(self) -> None:
        """Log model architecture summary."""
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "ViT initialized: %d layers, %d heads, D=%d, "
            "patches=%dx%d, params=%.2fM (trainable=%.2fM)",
            self.config.n_layers,
            self.config.n_heads,
            self.config.d_model,
            self.config.image_size // self.config.patch_size,
            self.config.image_size // self.config.patch_size,
            n_params / 1e6,
            n_trainable / 1e6,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Vision Transformer.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Class logits [B, num_classes]
        """
        B = x.shape[0]

        # Step 1: Patch embedding -- x_p^i E  [B, N, D]
        x = self.patch_embed(x)

        # Step 2: Prepend [CLS] token -- [x_class; x_p^1 E; ...; x_p^N E]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]

        # Step 3: Add positional embeddings (Eq. 1)
        x = self.embed_dropout(x + self.pos_embed)

        # Step 4: Pass through L Transformer encoder blocks (Eq. 2-3)
        for block in self.blocks:
            x = block(x)

        # Step 5: Extract [CLS] token and apply final LayerNorm (Eq. 4)
        cls_output = self.norm(x[:, 0])  # [B, D]

        # Step 6: Classification head
        return self.head(cls_output)  # [B, num_classes]
