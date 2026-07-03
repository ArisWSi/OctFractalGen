"""
单层八叉树分裂预测器 — 直接复用 OctGPT 的 OctFormer。

设计原则: 只引入"递归多模型"这一个变量，其余全部继承 OctGPT。
- OctFormer + OctreeAttention: 从 octgpt.models.octformer 导入
- AbsPosEmb + RotaryPosEmb: 从 octgpt.models.positional_embedding 导入
- 仅需一个轻量 ChildOctreeInfo（~30 行）替代 OctreeT 来适配合成子节点 token。
"""

import sys, os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── OctGPT 路径 ──────────────────────────────────────────────
_octgpt_root = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'extern', 'octgpt')
if _octgpt_root not in sys.path:
    sys.path.insert(0, _octgpt_root)

from models.octformer import OctFormer
from models.positional_embedding import AbsPosEmb, RMSNorm
from utils.utils import depth2batch, batch2depth, get_depth2batch_indices


# ══════════════════════════════════════════════════════════════════════════
# 轻量 OctreeT 替代: 为合成子节点 token 提供 OctFormer 所需接口
# ══════════════════════════════════════════════════════════════════════════

class ChildOctreeInfo:
    """OctreeT 兼容接口，适配从父坐标合成的子节点 token。

    OctFormer block 需要 octree 对象提供 xyz, batch_idx, depth_idx,
    patch_partition, patch_reverse 和 attention mask。
    本类对合成 token 提供这些，不依赖 ocnn Octree。

    核心简化: 单深度 → depth_idx 全零 → 无需 teacher-forcing mask。
    """

    def __init__(self, child_xyz, batch_ids, embed_dim, patch_size=1024,
                 dilation=4, use_swin=True):
        """
        child_xyz:  (N, 3) float  子节点 3D 坐标
        batch_ids: (N,)   long    每个 token 的 batch 归属 [0, B-1]
        """
        self.device = child_xyz.device
        self.xyz = child_xyz.float()
        self.batch_size = int(batch_ids.max().item()) + 1
        self.patch_size = patch_size
        self.dilation = dilation
        self.use_swin = use_swin
        self.block_num = patch_size * dilation

        # 计算 batch_idx 和 depth2batch indices
        self.nnum_t = torch.tensor(child_xyz.shape[0], device=self.device)
        self.nnum_a = (torch.ceil(self.nnum_t / self.block_num)
                       * self.block_num).int()

        # depth_idx: 全零（单深度，无 teacher-forcing 需求）
        self.depth_idx = torch.zeros(self.nnum_t, dtype=torch.long,
                                     device=self.device)
        self.depth_idx = self._pad(self.depth_idx, fill_value=0)

        # batch_idx: 展平 batch 标记，用于构建 attention mask
        self.batch_idx = self._pad(batch_ids.long(), fill_value=-1)

        # SWIN 偏移
        if self.use_swin:
            self.swin_nnum_pad = self.patch_size // 2
            self.swin_nnum_a = (torch.ceil(
                (self.nnum_t + self.swin_nnum_pad) / self.block_num
            ) * self.block_num).int()

        # depth2batch indices: 按 batch 重排
        _, self.indices = torch.sort(self.batch_idx[:self.nnum_t])

        # 预计算 attention mask
        self._build_masks()

    def _pad(self, data, fill_value=0):
        """补齐到 block_num 的整数倍。"""
        num = self.nnum_a - self.nnum_t
        if num <= 0:
            return data
        tail = data.new_full((num.item(),) + data.shape[1:], fill_value)
        return torch.cat([data, tail], dim=0)

    # ── patch partition / reverse（复制自 OctreeT）─────────────────

    def patch_partition(self, data, use_swin=False):
        """(N, C) → (num_patches, K, C)"""
        assert data.shape[0] == self.nnum_t, f"{data.shape[0]} != {self.nnum_t}"
        K = self.patch_size
        if use_swin:
            head = data.new_zeros(self.swin_nnum_pad, data.shape[-1])
            num = self.swin_nnum_a - self.nnum_t - self.swin_nnum_pad
            tail = data.new_zeros(num.item(), data.shape[-1])
            data = torch.cat([head, data, tail], dim=0)
        else:
            num = self.nnum_a - self.nnum_t
            tail = data.new_zeros(num.item(), data.shape[-1])
            data = torch.cat([data, tail], dim=0)
        return data.view(-1, K, data.shape[-1])

    def patch_reverse(self, data, use_swin=False):
        """(num_patches, K, C) → (N, C)"""
        K = self.patch_size
        data = data.reshape(-1, data.shape[-1])
        if use_swin:
            data = data[self.swin_nnum_pad:self.nnum_t + self.swin_nnum_pad]
        else:
            data = data[:self.nnum_t]
        return data

    # ── attention mask（复制自 OctreeT）───────────────────────────

    def _calc_mask(self, group, cond="neq"):
        """根据分组张量计算 attention mask。"""
        group = group.view(-1, self.patch_size)
        diff = group.unsqueeze(2) - group.unsqueeze(1)           # (P, K, K)
        invalid = torch.logical_or(
            group.unsqueeze(2) == -1, group.unsqueeze(1) == -1)
        if cond == "neq":
            mask_label = (diff != 0) | invalid
        elif cond == "le":
            mask_label = (diff < 0) | invalid
        else:
            raise ValueError(f"Unknown cond: {cond}")
        return torch.zeros_like(diff, dtype=torch.float).masked_fill(
            mask_label, -1e3)

    def _build_masks(self):
        """预计算 patch / dilation / SWIN mask。"""
        K, D = self.patch_size, self.dilation

        # batch mask（base）
        batch_patch = self.batch_idx.view(-1, K)
        batch_dilate = self.batch_idx.view(-1, D, K)
        batch_dilate = batch_dilate.transpose(1, 2).reshape(-1, K)

        self.patch_mask = self._calc_mask(self.batch_idx)
        self.dilate_mask = self._calc_mask(
            batch_dilate[:self.nnum_a, None].expand(-1, K))

        if self.use_swin:
            b_idx_swin = self._pad_swin(self.batch_idx[:self.nnum_t])
            self.swin_patch_mask = self._calc_mask(b_idx_swin)
            b_dil_swin = self._pad_swin(batch_dilate[:self.nnum_a])
            self.swin_dilate_mask = self._calc_mask(
                b_dil_swin[:, None].expand(-1, K))

    def _pad_swin(self, data):
        K = self.patch_size
        head = data.new_full((self.swin_nnum_pad,) + data.shape[1:], -1)
        num = self.swin_nnum_a - self.nnum_t - self.swin_nnum_pad
        tail = data.new_full((num.item(),) + data.shape[1:], -1)
        return torch.cat([head, data, tail], dim=0)


# ══════════════════════════════════════════════════════════════════════════
# OctreeLevelAR — 单层 OctFormer 分裂预测器
# ══════════════════════════════════════════════════════════════════════════

class OctreeLevelAR(nn.Module):
    """单层八叉树分裂预测器，复用 OctGPT OctFormer。

    参数:
        embed_dim:     特征维度
        num_blocks:    OctFormer block 数
        num_heads:     注意力头数
        cond_dim_in:   输入条件维度
        cond_dim_out:  输出条件维度
        patch_size:    patch 大小
        dilation:      膨胀率
        drop_rate:     dropout 率
        use_swin:      SWIN 偏移
        use_checkpoint: 梯度检查点
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_blocks: int = 12,
        num_heads: int = 8,
        cond_dim_in: int = 512,
        cond_dim_out: int = 512,
        patch_size: int = 1024,
        dilation: int = 4,
        drop_rate: float = 0.1,
        use_swin: bool = True,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.dilation = dilation
        self.use_swin = use_swin

        # 子节点嵌入（八分圆 + 条件）
        self.octant_emb = nn.Embedding(8, embed_dim)
        self.cond_proj = nn.Linear(cond_dim_in, embed_dim, bias=True)

        # OctGPT 位置编码（含 depth embedding，但 depth_idx 全零）
        self.pos_emb = AbsPosEmb(embed_dim)

        # OctGPT OctFormer（复用全部 attention + FFN + SWIN + dilation）
        self.octformer = OctFormer(
            channels=embed_dim, num_blocks=num_blocks,
            num_heads=num_heads, patch_size=patch_size,
            dilation=dilation, attn_drop=drop_rate,
            proj_drop=drop_rate, dropout=drop_rate,
            nempty=False, use_checkpoint=use_checkpoint,
            use_swin=use_swin, use_ctx=False,
        )
        self.ln = RMSNorm(embed_dim)

        # 输出头
        self.split_head = nn.Linear(embed_dim, 1, bias=True)
        self.cond_head = nn.Linear(embed_dim, cond_dim_out, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in [self.octant_emb, self.cond_proj,
                  self.split_head, self.cond_head]:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── 子节点坐标 ──────────────────────────────────────────────

    @staticmethod
    def _child_coords(parent_xyz, parent_batch):
        """父坐标 → 8N 子坐标, 子 batch, 子 octant。

        parent_xyz:   (N, 3)  父节点坐标
        parent_batch: (N,)    父节点 batch_id

        返回:
            child_xyz:   (8N, 3)
            child_batch: (8N,)
            child_octant:(8N,)  0-7
        """
        N = parent_xyz.shape[0]
        device = parent_xyz.device
        # 8 个偏移方向
        offsets = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
             [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
            device=device, dtype=parent_xyz.dtype)                        # (8, 3)
        child_xyz = (parent_xyz.unsqueeze(1) * 2 + offsets).view(N * 8, 3)
        child_batch = parent_batch.unsqueeze(1).expand(N, 8).reshape(N * 8)
        child_octant = torch.arange(8, device=device).unsqueeze(0).expand(
            N, 8).reshape(N * 8)
        return child_xyz, child_batch, child_octant

    # ── 训练前向 ────────────────────────────────────────────────

    def forward(
        self,
        parent_xyz: torch.Tensor,
        parent_batch: torch.Tensor,
        parent_cond: torch.Tensor,
        gt_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """训练 / 推理前向。

        参数:
            parent_xyz:   (N, 3) float
            parent_batch: (N,)   long   batch_id [0, B-1]
            parent_cond:  (N, cond_dim_in) 每节点条件
            gt_labels:    (N, 8) GT 占用 (0/1) — 训练时提供

        返回:
            logits:   (N, 8) occupancy logits
            cond_out: (N, cond_dim_out) 每父节点条件
            loss:     BCE 标量（gt_labels 为 None 则 = 0）
        """
        N = parent_xyz.shape[0]
        device = parent_xyz.device

        # 1. 构建 8N 子节点 token
        child_xyz, child_batch, child_octant = self._child_coords(
            parent_xyz, parent_batch)                                    # (8N, *)

        cond_8n = parent_cond.unsqueeze(1).expand(N, 8, -1).reshape(
            N * 8, -1)                                                   # (8N, cond_in)
        data = self.octant_emb(child_octant.long()) + self.cond_proj(cond_8n)

        # 2. 构建 ChildOctreeInfo + 位置编码
        child_info = ChildOctreeInfo(
            child_xyz, child_batch, self.embed_dim,
            self.patch_size, self.dilation, self.use_swin)

        data = data + self.pos_emb(data, child_info)

        # 3. OctFormer 前向（depth2batch → OctFormer → batch2depth）
        data = depth2batch(data, child_info.indices)
        data = self.octformer(data, child_info, context=None)
        data = batch2depth(data, child_info.indices)
        data = self.ln(data)                                            # (8N, C)

        # 4. 预测 + 聚合到 (N, 8)
        logits_8n = self.split_head(data).squeeze(-1)                   # (8N,)
        logits = logits_8n.view(N, 8)

        cond_8n = self.cond_head(data)                                   # (8N, cond_out)
        cond_out = cond_8n.view(N, 8, -1).mean(dim=1)                    # (N, cond_out)

        # 5. Loss
        loss = torch.tensor(0.0, device=device)
        if gt_labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, gt_labels.float())

        return logits, cond_out, loss

    # ── 采样 ────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        parent_xyz: torch.Tensor,
        parent_batch: torch.Tensor,
        parent_cond: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """推理: 单步前向 → sigmoid → threshold。

        返回:
            child_8way: (N, 8) {0,1}
            cond_out:   (N, cond_dim_out)
        """
        logits, cond_out, _ = self.forward(
            parent_xyz, parent_batch, parent_cond, gt_labels=None)
        probs = torch.sigmoid(logits / temperature)
        return (probs > 0.5).float(), cond_out
