"""
单层八叉树 AR 生成器。

从 FractalGen 的 AR 类适配到 3D 八叉树节点。每个生成器
处理一个深度转换（d → d+1）：接收深度 d 的父节点，
预测其 8 个子节点在深度 d+1 处是否存在。

序列排序：所有父节点的子节点候选按 Morton（Z-order）码
全局排序，保证 3D 空间局部性。因果 mask 确保第 i 个 token
只能看到前面的 token。

训练：teacher-forcing（所有子节点可见，Morton 序因果 mask）
推理：按 Morton 序逐 token 自回归生成，使用 KV-Cache
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.model.transformer import (
    RMSNorm,
    TransformerBlock,
    KVCache,
    precompute_freqs_cis_3d,
    find_multiple,
)
from src.utils.octree_ops import child_xyz, morton_encode_3d


class OctreeAR(nn.Module):
    """单层八叉树深度转换的自回归生成器。

    接收深度 d 的 N 个父节点，预测其 8×N 个子节点在深度 d+1
    处的占用率，按 3D Morton 码排序。

    参数:
        embed_dim: 内部嵌入维度
        num_blocks: Transformer block 数量
        num_heads: 注意力头数
        cond_dim_in: 输入条件维度（来自上一层或类别嵌入）
        cond_dim_out: 输出条件维度（传给下一层）
        grad_checkpointing: 是否使用梯度检查点节省显存
        attn_drop: 注意力 dropout 率
        proj_drop: 投影/FFN dropout 率
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_blocks: int = 16,
        num_heads: int = 8,
        cond_dim_in: int = 512,
        cond_dim_out: int = 512,
        grad_checkpointing: bool = False,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.cond_dim_in = cond_dim_in
        self.cond_dim_out = cond_dim_out
        self.grad_checkpointing = grad_checkpointing

        # 嵌入层
        self.cond_proj = nn.Linear(cond_dim_in, embed_dim, bias=True)
        self.pos_emb = nn.Linear(3, embed_dim, bias=True)
        self.token_ln = RMSNorm(embed_dim, eps=1e-5)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, n_head=num_heads,
                attn_drop=attn_drop, proj_drop=proj_drop, mlp_drop=proj_drop,
            )
            for _ in range(num_blocks)
        ])
        self.norm = RMSNorm(embed_dim, eps=1e-5)

        # 输出头
        self.split_head = nn.Linear(embed_dim, 1, bias=True)       # 占用率 logit
        self.cond_head = nn.Linear(embed_dim, cond_dim_out, bias=True)  # 下一层条件

        # KV-Cache 状态（推理时延迟初始化）
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    # KV-Cache
    # ------------------------------------------------------------------

    def setup_caches(self, max_batch_size: int, max_seq_length: int):
        """初始化 KV-Cache，用于自回归推理。"""
        head_dim = self.embed_dim // self.num_heads
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for block in self.blocks:
            block.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.num_heads, head_dim)

    # ------------------------------------------------------------------
    # Morton 排序
    # ------------------------------------------------------------------

    def _morton_sort(
        self, children_xyz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 Np×8 个子节点的 Morton 排序索引。

        所有 batch 项共享相同的 Morton 排序（因为八叉树结构
        在 batch 内共享，坐标完全相同）。

        参数:
            children_xyz: (B, Np, 8, 3) 子节点坐标

        返回:
            sort_idx: (Np*8,) 将 parent-major 序重排为 Morton 序的索引
            unsort_idx: (Np*8,) 将 Morton 序恢复为 parent-major 序的索引
        """
        B, Np, _, _ = children_xyz.shape
        device = children_xyz.device

        # 使用第一个 batch 项的坐标来计算排序
        coords = children_xyz[0].view(Np * 8, 3)

        # 计算 Morton 码并排序
        codes = morton_encode_3d(coords[:, 0], coords[:, 1], coords[:, 2])
        sort_idx = torch.argsort(codes)                          # pm → Morton
        unsort_idx = torch.empty_like(sort_idx)
        unsort_idx[sort_idx] = torch.arange(Np * 8, device=device)  # Morton → pm

        return sort_idx, unsort_idx

    # ------------------------------------------------------------------
    # Token 构建
    # ------------------------------------------------------------------

    def _build_child_tokens_morton(
        self,
        parent_xyz: torch.Tensor,
        parent_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """构建 Morton 序排列的子节点 token。

        参数:
            parent_xyz: (B, Np, 3) 深度 d 的父节点坐标
            parent_cond: (B, Np, cond_dim_in) 每个父节点的条件

        返回:
            child_tokens: (B, Np*8, embed_dim) Morton 序排列的 token
            child_xyz_sorted: (B, Np*8, 3) Morton 序排列的坐标
            sort_idx: (Np*8,) parent-major → Morton 的索引
            unsort_idx: (Np*8,) Morton → parent-major 的索引
        """
        B, Np, _ = parent_xyz.shape

        # 计算所有 8 个子节点坐标：(B, Np, 8, 3)
        children_xyz = child_xyz(parent_xyz)

        # Morton 排序索引
        sort_idx, unsort_idx = self._morton_sort(children_xyz)

        # 父节点条件投影：(B, Np, embed_dim) → (B, Np, 1, embed_dim)
        cond_emb = self.cond_proj(parent_cond).unsqueeze(2)

        # 子节点位置嵌入：(B, Np, 8, embed_dim)
        pos_emb = self.pos_emb(children_xyz.float())

        # 组合并展平：(B, Np, 8, embed_dim) → (B, Np*8, embed_dim)
        child_tokens = (cond_emb + pos_emb).view(B, Np * 8, self.embed_dim)

        # 展平坐标用于排序
        children_xyz_flat = children_xyz.view(B, Np * 8, 3)

        # 按 Morton 序重排
        child_tokens = child_tokens[:, sort_idx, :]
        children_xyz_flat = children_xyz_flat[:, sort_idx, :]

        child_tokens = self.token_ln(child_tokens)

        return child_tokens, children_xyz_flat, sort_idx, unsort_idx

    # ------------------------------------------------------------------
    # RoPE 辅助函数
    # ------------------------------------------------------------------

    def _compute_3d_rope(self, xyz: torch.Tensor) -> torch.Tensor:
        """为展平序列计算 3D RoPE 频率。

        参数:
            xyz: (B, seq_len, 3) 节点坐标

        返回:
            freqs_cis: (B, seq_len, head_dim//2, 2)
        """
        B, seq_len, _ = xyz.shape
        flat_xyz = xyz.reshape(B * seq_len, 3)
        head_dim = self.embed_dim // self.num_heads
        freqs_cis = precompute_freqs_cis_3d(flat_xyz, head_dim)
        return freqs_cis.view(B, seq_len, head_dim // 2, 2)

    # ------------------------------------------------------------------
    # Transformer 前向
    # ------------------------------------------------------------------

    def _forward_transformer(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """对全部 Transformer block 做前向传播。"""
        if self.grad_checkpointing and self.training:
            for block in self.blocks:
                x = checkpoint(block, x, freqs_cis, input_pos, mask,
                               use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x, freqs_cis, input_pos, mask)
        return self.norm(x)

    # ------------------------------------------------------------------
    # 训练前向（teacher-forcing + Morton 序）
    # ------------------------------------------------------------------

    def forward(
        self,
        parent_xyz: torch.Tensor,
        parent_cond: torch.Tensor,
        child_gt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """训练前向传播。

        序列按 Morton 序排列，使用因果 mask（下三角）。
        模型在一次前向中看到所有子节点 token。

        参数:
            parent_xyz: (B, Np, 3) 父节点坐标
            parent_cond: (B, Np, cond_dim_in) 每父节点条件
            child_gt_mask: (B, Np, 8) ground truth 占用率（1=存在）

        返回:
            child_logits: (B, Np, 8) 占用率 logits（parent-major 序）
            child_cond: (B, Np*8, cond_dim_out) 每子节点特征（parent-major 序）
            loss: 标量 BCE 损失
        """
        B, Np, _ = parent_xyz.shape
        device = parent_xyz.device

        # 构建 Morton 序排列的 token
        child_tokens, child_xyz_sorted, sort_idx, unsort_idx = \
            self._build_child_tokens_morton(parent_xyz, parent_cond)

        # 在排序后的坐标上应用 3D RoPE
        freqs_cis = self._compute_3d_rope(child_xyz_sorted)

        # Transformer 前向（默认 causal mask）
        x = self._forward_transformer(child_tokens, freqs_cis)

        # 预测占用率（Morton 序）
        logits_morton = self.split_head(x).squeeze(-1)       # (B, Np*8)

        # 条件特征（Morton 序）
        cond_morton = self.cond_head(x)                       # (B, Np*8, cond_dim_out)

        # 恢复到 parent-major 序
        logits_pm = logits_morton[:, unsort_idx]              # (B, Np*8)
        cond_pm = cond_morton[:, unsort_idx, :]               # (B, Np*8, cond_dim_out)
        child_logits = logits_pm.view(B, Np, 8)               # (B, Np, 8)

        # 计算损失
        loss = torch.tensor(0.0, device=device)
        if child_gt_mask is not None:
            # 将 gt_mask 重排到 Morton 序再算 loss
            gt_flat = child_gt_mask.float().view(B, Np * 8)   # parent-major 序
            gt_morton = gt_flat[:, sort_idx]                   # Morton 序
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits_morton, gt_morton, reduction='mean'
            )

        return child_logits, cond_pm, loss

    # ------------------------------------------------------------------
    # 自回归采样（Morton 序）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        parent_xyz: torch.Tensor,
        parent_cond: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """按 Morton 序自回归生成子节点占用率。

        Morton 序保证空间中相邻的子节点在生成序列中也接近，
        从而提升空间一致性。

        参数:
            parent_xyz: (B, Np, 3) 父节点坐标
            parent_cond: (B, Np, cond_dim_in) 每父节点条件
            temperature: 采样温度

        返回:
            child_mask: (B, Np, 8) parent-major 序的占用率
            child_cond: (B, Np*8, cond_dim_out) parent-major 序的条件特征
        """
        B, Np, _ = parent_xyz.shape
        device = parent_xyz.device
        total_steps = Np * 8

        # 计算所有子节点坐标
        children_xyz = child_xyz(parent_xyz)              # (B, Np, 8, 3)
        children_xyz_flat = children_xyz.view(B, total_steps, 3)

        # Morton 排序索引
        sort_idx, unsort_idx = self._morton_sort(children_xyz)

        # 预计算每父节点的条件嵌入
        cond_emb = self.cond_proj(parent_cond)             # (B, Np, embed_dim)

        # 预计算 Morton 序的 (父节点索引, 八分圆索引) 列表
        # parent-major 序：[p0_c0, p0_c1, ..., p0_c7, p1_c0, ..., p{Np-1}_c7]
        parent_idx_pm = torch.arange(Np, device=device).repeat_interleave(8)
        octant_idx_pm = torch.arange(8, device=device).repeat(Np)
        # unsort_idx[k] = Morton 位置 k 对应的 parent-major 位置
        parent_morton = parent_idx_pm[unsort_idx]          # (Np*8,)
        octant_morton = octant_idx_pm[unsort_idx]          # (Np*8,)

        # 初始化 KV-Cache
        self.setup_caches(max_batch_size=B, max_seq_length=total_steps)

        # 存储（parent-major 序）
        child_mask = torch.zeros(B, Np, 8, device=device)
        all_cond = torch.zeros(B, total_steps, self.cond_dim_out, device=device)

        head_dim = self.embed_dim // self.num_heads

        # 按 Morton 序逐 step 生成
        for step in range(total_steps):
            p_idx = parent_morton[step].item()   # 当前 Morton 位置的父节点
            o_idx = octant_morton[step].item()   # 当前 Morton 位置的八分圆

            # 构建当前子节点的 token
            child_pos = children_xyz[:, p_idx, o_idx, :]  # (B, 3)
            child_token = (
                cond_emb[:, p_idx, :] +
                self.pos_emb(child_pos.float())
            ).unsqueeze(1)                                 # (B, 1, embed_dim)
            child_token = self.token_ln(child_token)

            # RoPE
            freqs_cis = precompute_freqs_cis_3d(
                child_pos.reshape(B, 3), head_dim
            ).unsqueeze(1)                                 # (B, 1, head_dim//2, 2)

            # 带 KV-Cache 的前向
            input_pos = torch.tensor([step], device=device)
            x = self._forward_transformer(
                child_token, freqs_cis, input_pos=input_pos)

            # 预测并采样
            logit = self.split_head(x).squeeze(-1).squeeze(-1)  # (B,)
            prob = torch.sigmoid(logit / temperature)
            child_mask[:, p_idx, o_idx] = (
                torch.rand(B, device=device) < prob
            ).float()

            # 存储条件特征（按 parent-major 位置）
            pm_pos = unsort_idx[step].item()
            cond_feat = self.cond_head(x).squeeze(1)       # (B, cond_dim_out)
            all_cond[:, pm_pos, :] = cond_feat

        # 清理 KV-Cache
        for block in self.blocks:
            block.attention.kv_cache = None

        return child_mask, all_cond


# ------------------------------------------------------------------
# 工厂函数
# ------------------------------------------------------------------

def octree_ar_tiny(**kwargs) -> OctreeAR:
    """微型 AR 生成器，用于快速迭代。"""
    return OctreeAR(embed_dim=128, num_blocks=4, num_heads=4,
                    cond_dim_in=256, cond_dim_out=128, **kwargs)


def octree_ar_base(**kwargs) -> OctreeAR:
    """基础 AR 生成器（深度 3→4，最粗层级）。"""
    return OctreeAR(embed_dim=512, num_blocks=16, num_heads=8,
                    cond_dim_in=512, cond_dim_out=512, **kwargs)


def octree_ar_light(**kwargs) -> OctreeAR:
    """轻量 AR 生成器（深度 4→5，较细层级）。"""
    return OctreeAR(embed_dim=256, num_blocks=8, num_heads=4,
                    cond_dim_in=512, cond_dim_out=256, **kwargs)
