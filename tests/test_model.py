"""Tests for the from-scratch ViT components (CPU only, no dataset download)."""

import pytest
import torch
import torch.nn.functional as F

from src.model import (
    MultiHeadSelfAttention,
    PatchEmbedding,
    TransformerBlock,
    VisionTransformer,
    ViTConfig,
)

# Parameter count of the default CIFAR-10 config, as documented in README
DEFAULT_CONFIG_PARAMS = 3_195_146


def _expected_param_count(cfg: ViTConfig) -> int:
    """Count parameters analytically from the config (weights + biases)."""
    d, m = cfg.d_model, cfg.mlp_dim
    n_patches = (cfg.image_size // cfg.patch_size) ** 2
    patch_dim = cfg.in_channels * cfg.patch_size**2

    patch_embed = patch_dim * d + d
    cls_and_pos = d + (n_patches + 1) * d
    block = (
        2 * d  # norm1
        + (d * 3 * d + 3 * d)  # qkv
        + (d * d + d)  # attention output projection
        + 2 * d  # norm2
        + (d * m + m)  # mlp fc1
        + (m * d + d)  # mlp fc2
    )
    final_norm = 2 * d
    head = d * cfg.num_classes + cfg.num_classes
    return patch_embed + cls_and_pos + cfg.n_layers * block + final_norm + head


def _tiny_config() -> ViTConfig:
    return ViTConfig(
        image_size=8, patch_size=4, d_model=16, n_heads=4, n_layers=2, mlp_dim=32
    )


class TestParameterCount:
    def test_default_config_matches_documented_count(self) -> None:
        model = VisionTransformer(ViTConfig())
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == DEFAULT_CONFIG_PARAMS

    def test_matches_analytic_formula_for_other_configs(self) -> None:
        cfg = _tiny_config()
        model = VisionTransformer(cfg)
        assert sum(p.numel() for p in model.parameters()) == _expected_param_count(cfg)


class TestShapes:
    def test_patch_embedding_shape(self) -> None:
        cfg = ViTConfig()
        out = PatchEmbedding(cfg)(torch.randn(2, 3, 32, 32))
        assert out.shape == (2, 64, 256)

    def test_attention_and_block_preserve_shape(self) -> None:
        cfg = ViTConfig()
        x = torch.randn(2, 65, cfg.d_model)
        assert MultiHeadSelfAttention(cfg)(x).shape == x.shape
        assert TransformerBlock(cfg)(x).shape == x.shape

    def test_model_outputs_logits_per_class(self) -> None:
        logits = VisionTransformer(ViTConfig())(torch.randn(2, 3, 32, 32))
        assert logits.shape == (2, 10)

    def test_attention_rejects_indivisible_heads(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadSelfAttention(ViTConfig(d_model=30, n_heads=8))


class TestMatchesReferenceOps:
    def test_patch_embedding_equals_strided_conv(self) -> None:
        """Linear on flattened patches == Conv2d(kernel=stride=patch_size)."""
        torch.manual_seed(0)
        cfg = ViTConfig()
        embed = PatchEmbedding(cfg)
        x = torch.randn(2, 3, 32, 32)

        weight = embed.projection.weight.reshape(
            cfg.d_model, cfg.in_channels, cfg.patch_size, cfg.patch_size
        )
        conv_out = F.conv2d(
            x, weight, embed.projection.bias, stride=cfg.patch_size
        )  # [B, D, H/P, W/P]
        expected = conv_out.flatten(2).transpose(1, 2)  # [B, N, D]

        torch.testing.assert_close(embed(x), expected, rtol=1e-5, atol=1e-5)

    def test_attention_equals_scaled_dot_product_attention(self) -> None:
        torch.manual_seed(0)
        cfg = ViTConfig(dropout=0.0)
        attn = MultiHeadSelfAttention(cfg).eval()
        x = torch.randn(2, 65, cfg.d_model)
        b, n, d, h = x.shape[0], x.shape[1], cfg.d_model, cfg.n_heads

        q, k, v = attn.qkv(x).reshape(b, n, 3, h, d // h).permute(2, 0, 3, 1, 4)
        ref = F.scaled_dot_product_attention(q, k, v)
        expected = attn.proj(ref.transpose(1, 2).reshape(b, n, d))

        torch.testing.assert_close(attn(x), expected, rtol=1e-5, atol=1e-5)


class TestTrainingSmoke:
    def test_loss_decreases_when_overfitting_one_batch(self) -> None:
        torch.manual_seed(0)
        cfg = _tiny_config()
        model = VisionTransformer(cfg)
        images = torch.randn(8, 3, cfg.image_size, cfg.image_size)
        labels = torch.randint(0, cfg.num_classes, (8,))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        model.train()
        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] * 0.5
