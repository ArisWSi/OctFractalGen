"""
共享 Transformer 组件，从 FractalGen 的 ar.py 适配到 3D 八叉树节点。

关键适配:
- 3D Rotary Position Embedding (RoPE): 从节点 xyz 坐标动态计算，
  而非 2D 网格预计算
- DropPath (Stochastic Depth) 正则化
- SwiGLU FeedForward Network（与 FractalGen AR 一致）
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# DropPath（随机深度）
# ---------------------------------------------------------------------------

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False,
              scale_by_keep: bool = True) -> torch.Tensor:
    """在残差 block 中按样本随机丢弃路径（Stochastic Depth）。"""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """nn.Module 包装的 DropPath。"""

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
    """Root Mean Square Layer Normalization（FractalGen AR / Llama 风格）。

    参数:
        dim: 特征维度
        eps: 数值稳定性系数
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ---------------------------------------------------------------------------
# FeedForward（SwiGLU）
# ---------------------------------------------------------------------------

def find_multiple(n: int, k: int) -> int:
    """将 n 向上取整到 k 的最近倍数。"""
    if n % k == 0:
        return n
    return n + k - (n % k)


class FeedForward(nn.Module):
    """SwiGLU FeedForward block（Llama 风格）。

    FFN(x) = w2(silu(w1(x)) * w3(x))
    隐藏维度 = 2/3 * 4 * dim，向上取整到 multiple_of。
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
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # 输出
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.w2(nn.functional.silu(self.w1(x)) * self.w3(x)))


# ---------------------------------------------------------------------------
# 3D Rotary Position Embedding
# ---------------------------------------------------------------------------

def precompute_freqs_cis_3d(
    xyz: torch.Tensor,
    dim: int,
    base: float = 10000.0,
) -> torch.Tensor:
    """为任意节点位置计算 3D RoPE 频率。

    将 head 的 dim//2 个频率对分配到 x、y、z 三轴。
    每轴独立生成 RoPE 频率带，拼接后总长 == dim//2。

    参数:
        xyz: (N, 3) float 张量，节点坐标在 [0, 2^depth) 范围内
        dim: head 维度（embed_dim // num_heads）
        base: RoPE 基频

    返回:
        freqs_cis: (N, dim // 2, 2) 复数表示 [cos, sin]
    """
    N, device = xyz.shape[0], xyz.device
    dim_half = dim // 2

    # 将 dim_half 个频率对分配到 3 轴，余数优先分配给前几轴
    # 例: dim=64 → dim_half=32 → x:11, y:11, z:10
    base_per_axis = dim_half // 3
    remainder = dim_half % 3
    axis_sizes = [base_per_axis + (1 if i < remainder else 0) for i in range(3)]

    freqs_parts = []
    for i, d in enumerate(axis_sizes):
        if d == 0:
            continue
        # 每轴内仅使用偶数个频率元素
        n_elem = (d // 2) * 2
        if n_elem == 0:
            freqs_parts.append(torch.zeros(N, d, device=device))
            continue
        freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2,
                                             device=device).float() / n_elem))
        f = torch.outer(xyz[:, i], freqs)               # (N, n_elem // 2)
        # 补齐到 d
        pad_len = d - f.shape[1]
        if pad_len > 0:
            f = torch.cat([f, torch.zeros(N, pad_len, device=device)], dim=1)
        freqs_parts.append(f)

    freqs_cat = torch.cat(freqs_parts, dim=1)            # (N, dim // 2)
    freqs_cis = torch.stack([torch.cos(freqs_cat), torch.sin(freqs_cat)], dim=-1)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """对张量施加 rotary position embedding。

    参数:
        x: (B, seq_len, n_head, head_dim)
        freqs_cis: (seq_len, head_dim // 2, 2) — 每位置的 [cos, sin]

    返回:
        施加 RoPE 后的 x，形状不变
    """
    B, seq_len, n_head, head_dim = x.shape
    # 将 x 重塑为复数对
    xshaped = x.float().reshape(B, seq_len, n_head, head_dim // 2, 2)
    # 重塑 freqs_cis 以进行广播
    freqs_cis = freqs_cis.view(1, seq_len, 1, head_dim // 2, 2)

    # 施加旋转: (a+bi) * (cos θ + i sin θ)
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
    """多头自注意力，支持可选的 KV-cache 和 3D RoPE。

    参数:
        dim: 嵌入维度
        n_head: 查询头数
        n_kv_head: KV 头数（GQA，默认等于 n_head）
        attn_drop: 注意力 dropout 率
        proj_drop: 输出投影 dropout 率
    """

    def __init__(self, dim: int, n_head: int, n_kv_head: Optional[int] = None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        total_kv_dim = (n_head + 2 * self.n_kv_head) * self.head_dim

        # 合并 QKV 投影
        self.wqkv = nn.Linear(dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.attn_dropout_p = attn_drop
        self.resid_dropout = nn.Dropout(proj_drop)

        # KV-cache（推理时使用）
        self.kv_cache: Optional['KVCache'] = None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        参数:
            x: (B, seq_len, dim)
            freqs_cis: (seq_len, head_dim//2, 2) 用于 RoPE
            input_pos: (seq_len,) 用于 KV-cache 索引
            mask: (seq_len, seq_len) 注意力 mask，或 None 表示 causal

        返回:
            (B, seq_len, dim)
        """
        B, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split(
            [self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(B, seqlen, self.n_head, self.head_dim)
        xk = xk.view(B, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(B, seqlen, self.n_kv_head, self.head_dim)

        # 施加 3D RoPE
        if freqs_cis is not None:
            xq = apply_rotary_emb(xq, freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis)

        # 转置为 (B, n_head, seqlen, head_dim)
        xq, xk, xv = map(lambda t: t.transpose(1, 2), (xq, xk, xv))

        # KV-cache
        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv

        # GQA: 扩展 KV 头以匹配 Q 头
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        # 缩放点积注意力
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
    """自回归推理的键值缓存。

    参数:
        max_batch_size: 最大 batch 大小
        max_seq_length: 最大序列长度
        n_head: 注意力头数
        head_dim: 每头维度
    """

    def __init__(self, max_batch_size: int, max_seq_length: int,
                 n_head: int, head_dim: int):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape))
        self.register_buffer('v_cache', torch.zeros(cache_shape))

    def update(self, input_pos: torch.Tensor, k_val: torch.Tensor,
               v_val: torch.Tensor):
        """在 input_pos 位置写入新的 K、V 到缓存中。"""
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val.to(k_out.dtype)
        v_out[:, :, input_pos] = v_val.to(k_out.dtype)
        return k_out, v_out


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: Attention + SwiGLU FFN。

    布局:
        x = x + DropPath(Attention(RMSNorm(x)))
        x = x + DropPath(FFN(RMSNorm(x)))
    """

    def __init__(self, dim: int, n_head: int, n_kv_head: Optional[int] = None,
                 mlp_drop: float = 0.0, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, drop_path: float = 0.0,
                 norm_eps: float = 1e-5):
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
        # Attention 子 block
        h = x + self.drop_path(
            self.attention(self.attention_norm(x), freqs_cis, input_pos, mask)
        )
        # FFN 子 block
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out
