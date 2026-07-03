"""
单层八叉树分裂预测器（OctFormer 风格）。

从 OctGPT 的 OctFormer 适配：每个 OctreeAR 管理一个深度转换
（d → d+1），使用 patch-wise + SWIN + dilation 注意力。

关键调整:
- 仅处理当前深度的 8N 个子节点 token（非全深度拼接）
- 双向注意力（同深度内 token 互相可见）
- Morton-order 排列保证 patch 内空间局部性
- 训练: teacher-forcing，全部 token 可见 → BCE loss
- 生成: 单步预测全部 token → threshold → split
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.model.transformer import (
    RMSNorm,
    PatchTransformerBlock,
    find_multiple,
)
from src.utils.octree_ops import child_xyz, morton_encode_3d


class OctreeAR(nn.Module):
    """单层八叉树分裂预测器，使用 OctFormer 风格 patch 注意力。

    接收深度 d 的 Np 个父节点，预测其 8×Np 个子节点在深度 d+1
    处的占用率，按 3D Morton 码排序。

    参数:
        embed_dim: 内部嵌入维度
        num_blocks: Transformer block 数量
        num_heads: 注意力头数
        cond_dim_in: 输入条件维度（来自上一层或类别嵌入）
        cond_dim_out: 输出条件维度（传给下一层）
        patch_size: patch 内 token 数（0 = 全注意力，>0 = OctFormer 风格）
        dilation: 膨胀率（跨 patch 连接）
        grad_checkpointing: 梯度检查点
        attn_drop: 注意力 dropout
        proj_drop: 投影/FFN dropout
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
        self.patch_size = patch_size
        self.dilation = dilation
        self.grad_checkpointing = grad_checkpointing

        # 嵌入
        self.cond_proj = nn.Linear(cond_dim_in, embed_dim, bias=True)
        self.pos_emb = nn.Linear(3, embed_dim, bias=True)
        self.token_ln = RMSNorm(embed_dim, eps=1e-5)

        # OctFormer 风格的 patch 注意力 blocks
        # 交替模式（仿 OctFormer）:
        #   i=0,4,8...:  patch dense
        #   i=1,5,9...:  patch dilation
        #   i=2,6,10...: SWIN dense
        #   i=3,7,11...: SWIN dilation
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block_dilation = 1 if (i % 2 == 0) else dilation
            use_swin = ((i // 2) % 2 == 1)
            blk = PatchTransformerBlock(
                dim=embed_dim,
                n_head=num_heads,
                patch_size=patch_size,
                dilation=block_dilation,
                mlp_drop=proj_drop,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            self.blocks.append(blk)

        self.norm = RMSNorm(embed_dim, eps=1e-5)

        # 输出头
        self.split_head = nn.Linear(embed_dim, 1, bias=True)          # logit
        self.cond_head = nn.Linear(embed_dim, cond_dim_out, bias=True)  # 条件

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, RMSNorm)):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    # Morton 排序
    # ------------------------------------------------------------------

    def _morton_sort(
        self, children_xyz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 Np×8 个子节点的 Morton 排序索引。

        参数:
            children_xyz: (B, Np, 8, 3) 子节点坐标

        返回:
            sort_idx: (Np*8,) parent-major → Morton
            unsort_idx: (Np*8,) Morton → parent-major
        """
        B, Np, _, _ = children_xyz.shape
        device = children_xyz.device
        coords = children_xyz[0].view(Np * 8, 3)
        codes = morton_encode_3d(coords[:, 0], coords[:, 1], coords[:, 2])
        sort_idx = torch.argsort(codes)
        unsort_idx = torch.empty_like(sort_idx)
        unsort_idx[sort_idx] = torch.arange(Np * 8, device=device)
        return sort_idx, unsort_idx

    # ------------------------------------------------------------------
    # Token 构建
    # ------------------------------------------------------------------

    def _build_child_tokens_morton(
        self,
        parent_xyz: torch.Tensor,
        parent_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """构建 Morton 序的子节点 token。

        参数:
            parent_xyz: (B, Np, 3)
            parent_cond: (B, Np, cond_dim_in)

        返回:
            child_tokens: (B, Np*8, embed_dim) Morton 序
            child_xyz_m: (B, Np*8, 3) Morton 序坐标
            sort_idx: (Np*8,) 排序索引
            unsort_idx: (Np*8,) 逆排序索引
        """
        B, Np, _ = parent_xyz.shape
        device = parent_xyz.device

        # 子节点坐标: (B, Np, 8, 3)
        children_xyz = child_xyz(parent_xyz)
        sort_idx, unsort_idx = self._morton_sort(children_xyz)

        # Parent-major 序
        child_xyz_pm = children_xyz.view(B, Np * 8, 3)
        child_cond_pm = parent_cond.unsqueeze(2).expand(
            B, Np, 8, parent_cond.shape[-1]).reshape(B, Np * 8, -1)

        # 重排到 Morton 序
        child_xyz_m = child_xyz_pm[:, sort_idx, :]        # (B, Np*8, 3)
        child_cond_m = child_cond_pm[:, sort_idx, :]       # (B, Np*8, cond_dim_in)

        # Token: pos_emb + cond_proj
        pos = self.pos_emb(child_xyz_m.float())
        cond = self.cond_proj(child_cond_m)
        child_tokens = self.token_ln(pos + cond)            # (B, Np*8, embed_dim)

        return child_tokens, child_xyz_m, sort_idx, unsort_idx

    # ------------------------------------------------------------------
    # Transformer 前向
    # ------------------------------------------------------------------

    def _forward_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """通过所有 patch attention block 做前向传播。"""
        if self.grad_checkpointing and self.training:
            for block in self.blocks:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x)
        return self.norm(x)

    # ------------------------------------------------------------------
    # 训练前向
    # ------------------------------------------------------------------

    def forward(
        self,
        parent_xyz: torch.Tensor,
        parent_cond: torch.Tensor,
        gt_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """训练前向：teacher-forcing，双向注意力。

        参数:
            parent_xyz: (B, Np, 3) 父节点坐标
            parent_cond: (B, Np, cond_dim_in) 父节点条件
            gt_labels: (B, Np, 8) ground-truth 占用率标签 {0, 1}

        返回:
            logits_pm: (B, Np*8) parent-major 序 logits
            cond_out: (B, Np, cond_dim_out) 每父节点条件
            loss: BCE 标量
        """
        B, Np, _ = parent_xyz.shape
        device = parent_xyz.device
        seq_len = Np * 8

        # 构建 Morton 序 token
        child_tokens, _, sort_idx, unsort_idx = (
            self._build_child_tokens_morton(parent_xyz, parent_cond))
        # child_tokens: (B, Np*8, embed_dim)

        # Patch attention 前向
        x = self._forward_transformer(child_tokens)     # (B, Np*8, embed_dim)

        # Split logits
        logits_morton = self.split_head(x).squeeze(-1)   # (B, Np*8)
        logits_pm = logits_morton[:, unsort_idx]         # 恢复 parent-major 序

        # 条件向量：每父节点上平均池化其 8 个子节点特征
        cond_morton = self.cond_head(x)                   # (B, Np*8, cond_dim_out)
        cond_pm = cond_morton[:, unsort_idx, :]            # 恢复 parent-major 序
        cond_out = cond_pm.view(B, Np, 8, -1).mean(dim=2)  # (B, Np, cond_dim_out)

        # Loss
        loss = torch.tensor(0.0, device=device)
        if gt_labels is not None:
            gt_flat = gt_labels.view(B, Np * 8).float()
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits_pm, gt_flat)

        return logits_pm, cond_out, loss

    # ------------------------------------------------------------------
    # 生成（单步预测）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        parent_xyz: torch.Tensor,
        parent_cond: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成子节点占用率：单步前向 + threshold。

        参数:
            parent_xyz: (B, Np, 3)
            parent_cond: (B, Np, cond_dim_in)
            temperature: sigmoid 温度

        返回:
            child_8way: (B, Np, 8) {0, 1} 子节点占用率
            cond_out: (B, Np, cond_dim_out) 条件向量
        """
        B, Np, _ = parent_xyz.shape

        child_tokens, _, sort_idx, unsort_idx = (
            self._build_child_tokens_morton(parent_xyz, parent_cond))

        x = self._forward_transformer(child_tokens)

        # Split logits
        logits_m = self.split_head(x).squeeze(-1) / temperature  # (B, Np*8)
        logits_pm = logits_m[:, unsort_idx]                        # parent-major

        # Sigmoid + threshold
        probs = torch.sigmoid(logits_pm)
        child_8way = (probs > 0.5).float().view(B, Np, 8)

        # 条件向量
        cond_m = self.cond_head(x)
        cond_pm = cond_m[:, unsort_idx, :]
        cond_out = cond_pm.view(B, Np, 8, -1).mean(dim=2)

        return child_8way, cond_out


# ------------------------------------------------------------------
# 工厂函数
# ------------------------------------------------------------------

def octree_ar_tiny(**kwargs) -> OctreeAR:
    return OctreeAR(embed_dim=256, num_blocks=8, num_heads=4, **kwargs)

def octree_ar_base(**kwargs) -> OctreeAR:
    return OctreeAR(embed_dim=512, num_blocks=16, num_heads=8, **kwargs)

def octree_ar_light(**kwargs) -> OctreeAR:
    return OctreeAR(embed_dim=128, num_blocks=4, num_heads=4, **kwargs)
