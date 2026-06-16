"""
Shared Transformer components adapted from FractalGen's ar.py for 3D octree nodes.

Key adaptations:
- 3D Rotary Position Embedding (RoPE) computed from node xyz coordinates
  instead of 2D grid positions
- DropPath (Stochastic Depth) for regularization
- SwiGLU FeedForward Network (same as FractalGen AR)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# DropPath (Stochastic Depth)
# ---------------------------------------------------------------------------

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False,
              scale_by_keep: bool = True) -> torch.Tensor:
    """Drop paths per sample in residual blocks (Stochastic Depth)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """DropPath as a nn.Module."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in FractalGen AR / Llama)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ---------------------------------------------------------------------------
# FeedForward (SwiGLU)
# ---------------------------------------------------------------------------

def find_multiple(n: int, k: int) -> int:
    """Round n up to the nearest multiple of k."""
    if n % k == 0:
        return n
    return n + k - (n % k)


class FeedForward(nn.Module):
    """SwiGLU FeedForward block (Llama-style).

    FFN(x) = w2(silu(w1(x)) * w3(x))
    Hidden dim = 2/3 * 4 * dim, rounded to multiple_of.
    """

    def __init__(self, dim: int, multiple_of: int = 256,
                 ffn_dim_multiplier: Optional[float] = None,
                 dropout: float = 0.0):
        super().__init__()
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # value
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # output
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(nn.functional.silu(self.w1(x)) * self.w3(x)))


# ---------------------------------------------------------------------------
# 3D Rotary Position Embedding
# ---------------------------------------------------------------------------

def precompute_freqs_cis_3d(
    xyz: torch.Tensor,  # (N, 3) — node coordinates at a given depth
    dim: int,
    base: float = 10000.0,
) -> torch.Tensor:
    """Compute 3D RoPE frequencies for arbitrary node positions.

    Splits the head dimension into 3 equal parts for x, y, z axes.
    Each axis encodes position using standard RoPE frequency bands.

    Args:
        xyz: (N, 3) float tensor of node coordinates in [0, 2^depth)
        dim: head dimension (dim // num_heads)
        base: RoPE base frequency

    Returns:
        freqs_cis: (N, dim // 2, 2) complex representation [cos, sin]
    """
    N = xyz.shape[0]
    third_dim = dim // 3
    # Use only even frequencies within each third
    n_elem = (third_dim // 2) * 2

    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=xyz.device).float() / n_elem))
    # freqs: (n_elem // 2,)

    # Compute outer product for each axis
    freqs_x = torch.outer(xyz[:, 0], freqs)  # (N, n_elem // 2)
    freqs_y = torch.outer(xyz[:, 1], freqs)
    freqs_z = torch.outer(xyz[:, 2], freqs)

    # Pad each to third_dim // 2 (handle odd dimensions)
    pad_len = third_dim // 2 - freqs_x.shape[1]
    if pad_len > 0:
        freqs_x = torch.cat([freqs_x, torch.zeros(N, pad_len, device=xyz.device)], dim=1)
        freqs_y = torch.cat([freqs_y, torch.zeros(N, pad_len, device=xyz.device)], dim=1)
        freqs_z = torch.cat([freqs_z, torch.zeros(N, pad_len, device=xyz.device)], dim=1)

    # Concatenate: [freqs_x | freqs_y | freqs_z] → (N, 3 * third_dim // 2) = (N, dim // 2)
    freqs_cat = torch.cat([freqs_x, freqs_y, freqs_z], dim=1)

    # Convert to complex representation [cos, sin]
    freqs_cis = torch.stack([torch.cos(freqs_cat), torch.sin(freqs_cat)], dim=-1)
    return freqs_cis  # (N, dim // 2, 2)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to a tensor.

    Args:
        x: (B, seq_len, n_head, head_dim)
        freqs_cis: (seq_len, head_dim // 2, 2) — [cos, sin] per position

    Returns:
        x with RoPE applied, same shape as input
    """
    B, seq_len, n_head, head_dim = x.shape
    # Reshape x to complex pairs
    xshaped = x.float().reshape(B, seq_len, n_head, head_dim // 2, 2)
    # Reshape freqs_cis for broadcasting
    freqs_cis = freqs_cis.view(1, seq_len, 1, head_dim // 2, 2)

    # Apply rotation: (a+bi) * (cos θ + i sin θ)
    x_out2 = torch.stack([
        xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
        xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)

    x_out2 = x_out2.flatten(3)  # (B, seq_len, n_head, head_dim)
    return x_out2.type_as(x)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-Head Self-Attention with optional KV-cache and 3D RoPE."""

    def __init__(self, dim: int, n_head: int, n_kv_head: Optional[int] = None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        total_kv_dim = (n_head + 2 * self.n_kv_head) * self.head_dim

        # Combined QKV projection
        self.wqkv = nn.Linear(dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.attn_dropout_p = attn_drop
        self.resid_dropout = nn.Dropout(proj_drop)

        # KV-cache (used during inference)
        self.kv_cache: Optional['KVCache'] = None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, dim)
            freqs_cis: (seq_len, head_dim//2, 2) for RoPE
            input_pos: (seq_len,) for KV-cache indexing
            mask: (seq_len, seq_len) attention mask, or None for causal

        Returns:
            (B, seq_len, dim)
        """
        B, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(B, seqlen, self.n_head, self.head_dim)
        xk = xk.view(B, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(B, seqlen, self.n_kv_head, self.head_dim)

        # Apply 3D RoPE
        if freqs_cis is not None:
            xq = apply_rotary_emb(xq, freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis)

        # Transpose to (B, n_head, seqlen, head_dim)
        xq, xk, xv = map(lambda t: t.transpose(1, 2), (xq, xk, xv))

        # KV-cache
        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv

        # GQA: expand KV heads to match Q heads
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        # Scaled dot-product attention
        is_causal = mask is None
        output = nn.functional.scaled_dot_product_attention(
            xq, keys, values,
            attn_mask=mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )

        output = output.transpose(1, 2).contiguous().view(B, seqlen, self.dim)
        output = self.resid_dropout(self.wo(output))
        return output


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------

class KVCache(nn.Module):
    """Key-Value cache for autoregressive inference."""

    def __init__(self, max_batch_size: int, max_seq_length: int,
                 n_head: int, head_dim: int):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape))
        self.register_buffer('v_cache', torch.zeros(cache_shape))

    def update(self, input_pos: torch.Tensor, k_val: torch.Tensor,
               v_val: torch.Tensor):
        """Write new K, V at input_pos into the cache."""
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val.to(k_out.dtype)
        v_out[:, :, input_pos] = v_val.to(k_out.dtype)
        return k_out, v_out


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: Attention + SwiGLU FFN.

    Layout:
        x = x + DropPath(Attention(RMSNorm(x)))
        x = x + DropPath(FFN(RMSNorm(x)))
    """

    def __init__(self, dim: int, n_head: int, n_kv_head: Optional[int] = None,
                 mlp_drop: float = 0.0, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, drop_path: float = 0.0,
                 norm_eps: float = 1e-6):
        super().__init__()
        self.attention = Attention(
            dim=dim, n_head=n_head, n_kv_head=n_kv_head,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.feed_forward = FeedForward(dim=dim, dropout=mlp_drop)
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention sub-block
        h = x + self.drop_path(
            self.attention(self.attention_norm(x), freqs_cis, input_pos, mask)
        )
        # FFN sub-block
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out
