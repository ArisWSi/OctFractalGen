"""
单层八叉树分裂预测器（OctFormer 风格）。

每个 OctreeAR 管理一个深度转换（d → d+1），使用 patch-wise + SWIN +
dilation 注意力。直接对 N 个父节点 token 做 attention，输出 N 个 split
logit，不展开 8N 子节点（序列长度降 8 倍，FLOPs 降 ~64 倍）。

关键设计:
- 对父节点做 attention，每个父节点直接预测是否分裂
- 双向注意力（同深度内节点互相可见）
- Morton-order 排列保证 patch 内空间局部性
- 训练: teacher-forcing → BCE loss
- 生成: 单步预测 → threshold → split
- 支持稀疏八叉树：展平输入 (N, ...)，用 batch_id 索引条件
"""

import math
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.model.transformer import (
    RMSNorm,
    PatchTransformerBlock,
    find_multiple,
    precompute_freqs_cis_3d,
)
from src.utils.octree_ops import morton_encode_3d


class OctreeAR(nn.Module):
    """单层八叉树分裂预测器，使用 OctFormer 风格 patch 注意力。

    直接对 N 个父节点做 attention，预测每个父节点是否分裂（0/1）。
    不展开 8N 子节点，序列长度 = N（父节点数）。

    参数:
        embed_dim: 内部嵌入维度
        num_blocks: Transformer block 数量
        num_heads: 注意力头数
        cond_dim_in: 输入条件维度（来自上一层或类别嵌入）
        cond_dim_out: 输出条件维度（传给下一层）
        patch_size: patch 内 token 数
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
        buffer_size: int = 64,
        num_iters: int = 128,
        start_temperature: float = 1.0,
        remask_stage: float = 0.7,
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
        self.buffer_size = buffer_size
        self.num_iters = num_iters
        self.start_temperature = start_temperature
        self.remask_stage = remask_stage

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
        self.split_head = nn.Linear(embed_dim, 1, bias=True)          # split logit
        self.cond_head = nn.Linear(embed_dim, cond_dim_out, bias=True)  # 条件

        # MaskGIT: split token embedding（0=不分裂, 1=分裂）和 mask token
        self.split_emb = nn.Embedding(2, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.normal_(self.split_emb.weight, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

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
    # Morton 排序（父节点级别，不展开子节点）
    # ------------------------------------------------------------------

    def _morton_sort_parents(
        self, parent_xyz: torch.Tensor, batch_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """为父节点计算 Morton 排序（含 batch 分块）。

        参数:
            parent_xyz: (N, 3) 父节点坐标
            batch_ids: (N,) 父节点的 batch 归属

        返回:
            sort_idx: (N,) original → Morton
            unsort_idx: (N,) Morton → original
        """
        N = parent_xyz.shape[0]
        device = parent_xyz.device

        codes = morton_encode_3d(
            parent_xyz[:, 0], parent_xyz[:, 1], parent_xyz[:, 2])

        # 将 batch_id 编码到高位，确保 batch 分块
        codes = codes + batch_ids.long() * (codes.max() + 1)

        sort_idx = torch.argsort(codes)
        unsort_idx = torch.empty_like(sort_idx)
        unsort_idx[sort_idx] = torch.arange(N, device=device)

        return sort_idx, unsort_idx

    # ------------------------------------------------------------------
    # Token 构建（父节点级别，N 个 token 而非 8N）
    # ------------------------------------------------------------------

    def _build_parent_tokens(
        self,
        parent_xyz: torch.Tensor,
        parent_cond: torch.Tensor,
        batch_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """构建 Morton 序的父节点 token。

        参数:
            parent_xyz: (N, 3) 所有父节点坐标（展平）
            parent_cond: (N, cond_dim_in) 每父节点条件
            batch_ids: (N,) 父节点的 batch 归属

        返回:
            tokens: (1, N, embed_dim) Morton 序（B=1 用于 attention）
            xyz_m: (1, N, 3) Morton 序坐标
            sort_idx: (N,) original → Morton
            unsort_idx: (N,) Morton → original
        """
        N = parent_xyz.shape[0]
        sort_idx, unsort_idx = self._morton_sort_parents(parent_xyz, batch_ids)

        # 重排到 Morton 序
        xyz_m = parent_xyz[sort_idx, :]       # (N, 3)
        cond_m = parent_cond[sort_idx, :]     # (N, cond_dim_in)

        # Token: pos_emb + cond_proj
        pos = self.pos_emb(xyz_m.float())
        cond = self.cond_proj(cond_m)
        tokens = self.token_ln(pos + cond)     # (N, embed_dim)

        # 增加 batch 维度
        tokens = tokens.unsqueeze(0)           # (1, N, embed_dim)
        xyz_m = xyz_m.unsqueeze(0)             # (1, N, 3)

        return tokens, xyz_m, sort_idx, unsort_idx

    # ------------------------------------------------------------------
    # Transformer 前向
    # ------------------------------------------------------------------

    def _forward_transformer(self, x: torch.Tensor,
                             xyz: torch.Tensor = None) -> torch.Tensor:
        """通过所有 patch attention block 做前向传播。

        RoPE 频率在此一次性预计算，传给所有 block 复用（避免每 block 重复算）。
        """
        freqs_cis = None
        if xyz is not None:
            B, N, _ = x.shape
            hd = self.embed_dim // self.num_heads
            flat_xyz = xyz.reshape(B * N, 3)
            freqs_cis = precompute_freqs_cis_3d(flat_xyz, hd)

        if self.grad_checkpointing and self.training:
            for block in self.blocks:
                x = checkpoint(block, x, freqs_cis, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x, freqs_cis)
        return self.norm(x)

    # ------------------------------------------------------------------
    # 辅助: 构建 token + buffer
    # ------------------------------------------------------------------

    def _build_tokens_with_buffer(self, parent_xyz, parent_cond, batch_ids,
                                  token_features=None, mask=None,
                                  prefix_tokens=None, prefix_xyz=None):
        """构建 token 序列（含 buffer + 跨深度 prefix + Morton 排序）。

        序列布局: [buffer(B*buf) | prefix(N_prev) | tokens(N)]

        跨深度 token 交互: prefix_tokens 是上一层 AR 的 cond_out
        （每节点一个 cond_embed_dim 向量），经 cond_proj 投影到 embed_dim
        后参与本层 attention。这让本层 token 能 attend 到上一层节点
        的特征，而非只收到一个标量条件——对齐 OctGPT 的跨深度 token
        attention 能力。

        参数:
            parent_xyz: (N, 3)
            parent_cond: (N, cond_dim_in)
            batch_ids: (N,)
            token_features: (N, embed_dim) 已有的 token 特征（MaskGIT 用）
            mask: (N,) bool, True = 需要预测（用 mask_token 替换）
            prefix_tokens: (N_prev, cond_dim_in) 上一层 cond_out（跨深度交互）
            prefix_xyz: (N_prev, 3) 上一层节点坐标（用于 prefix 的 RoPE）

        返回:
            tokens: (1, B*buf + N_prev + N, embed_dim)
            xyz_m: (1, B*buf + N_prev + N, 3)
            sort_idx: (N,) 父节点排序索引（不含 buffer/prefix）
            unsort_idx: (N,)
        """
        N = parent_xyz.shape[0]
        B = int(batch_ids.max().item()) + 1
        device = parent_xyz.device

        # 父节点 token
        if token_features is None:
            tokens, xyz_m, sort_idx, unsort_idx = (
                self._build_parent_tokens(parent_xyz, parent_cond, batch_ids))
        else:
            sort_idx, unsort_idx = self._morton_sort_parents(parent_xyz, batch_ids)
            tokens = token_features[sort_idx].unsqueeze(0)
            xyz_m = parent_xyz[sort_idx].unsqueeze(0)

        # 替换被 mask 的 token
        if mask is not None:
            mask_m = mask[sort_idx]  # (N,)
            tokens[0, mask_m, :] = self.mask_token.to(tokens.dtype)

        # Buffer: 全局条件投影，每 batch item 重复 buffer_size 次
        buffer_cond = self.cond_proj(parent_cond[:B])  # (B, embed_dim)
        buffer = buffer_cond.unsqueeze(1).expand(
            B, self.buffer_size, -1).reshape(1, B * self.buffer_size, -1)
        buffer_xyz = torch.zeros(1, B * self.buffer_size, 3, device=device)

        # 跨深度 prefix: 上一层 cond_out → cond_proj → embed_dim
        if prefix_tokens is not None:
            prefix_emb = self.cond_proj(prefix_tokens).unsqueeze(0)  # (1, N_prev, dim)
            prefix_xyz_m = prefix_xyz.unsqueeze(0)                    # (1, N_prev, 3)
            tokens = torch.cat([buffer, prefix_emb, tokens], dim=1)
            xyz_m = torch.cat([buffer_xyz, prefix_xyz_m, xyz_m], dim=1)
        else:
            tokens = torch.cat([buffer, tokens], dim=1)
            xyz_m = torch.cat([buffer_xyz, xyz_m], dim=1)

        return tokens, xyz_m, sort_idx, unsort_idx

    # ------------------------------------------------------------------
    # 训练前向（random masking + buffer，与 OctGPT 对齐）
    # ------------------------------------------------------------------

    def forward(
        self,
        parent_xyz: torch.Tensor,
        parent_cond: torch.Tensor,
        gt_labels: Optional[torch.Tensor] = None,
        batch_ids: Optional[torch.Tensor] = None,
        prefix_tokens: Optional[torch.Tensor] = None,
        prefix_xyz: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """训练前向：random masking + buffer + 跨深度 prefix → 预测 split。

        与 OctGPT 对齐:
        1. 用 gt split 构建 token (split_emb)
        2. Random masking: 50-100% 的 token 被 mask_token 替换
        3. 加 buffer (全局条件) + prefix (上一层 token 特征)
        4. Attention → split_head → BCE loss（只在 masked 位置）

        参数:
            parent_xyz: (N, 3)
            parent_cond: (N, cond_dim_in)
            gt_labels: (N, 8) ground-truth 子节点占用率
            batch_ids: (N,)
            prefix_tokens: (N_prev, cond_dim_in) 上一层 cond_out（跨深度交互）
            prefix_xyz: (N_prev, 3) 上一层节点坐标

        返回:
            split_logits: (N,) split logits
            cond_out: (N, cond_dim_out) 传给下一层
            loss: BCE 标量（仅 masked 位置）
        """
        N = parent_xyz.shape[0]
        device = parent_xyz.device

        if batch_ids is None:
            batch_ids = torch.zeros(N, dtype=torch.long, device=device)

        # gt → binary split
        gt_split = (gt_labels.sum(dim=-1) > 0).long()  # (N,)

        # 构建 token: split_emb(gt)
        token_features = self.split_emb(gt_split)  # (N, embed_dim)

        # Random masking: 50-100%（用 CPU 随机避免 GPU 同步）
        mask_ratio = 0.5 + 0.5 * random.random()
        num_masked = max(int(N * mask_ratio), 1)
        orders = torch.randperm(N, device=device)
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        mask[orders[:num_masked]] = True

        # 构建含 buffer + prefix 的 token（masked 位置用 mask_token 替换）
        tokens, xyz_m, sort_idx, unsort_idx = self._build_tokens_with_buffer(
            parent_xyz, parent_cond, batch_ids,
            token_features=token_features, mask=mask,
            prefix_tokens=prefix_tokens, prefix_xyz=prefix_xyz)

        # Attention
        x = self._forward_transformer(tokens, xyz_m)  # (1, B*buf + N_prev + N, dim)
        B = int(batch_ids.max().item()) + 1
        prefix_len = prefix_tokens.shape[0] if prefix_tokens is not None else 0
        x = x[0, B * self.buffer_size + prefix_len:, :]  # (N_morton, dim)

        # Split logits
        split_logits_m = self.split_head(x).squeeze(-1)  # (N_morton,)
        split_logits = split_logits_m[unsort_idx]         # (N,)

        # 条件向量
        cond_m = self.cond_head(x)              # (N_morton, cond_out)
        cond_out = cond_m[unsort_idx, :]        # (N, cond_out)

        # Loss: 仅 masked 位置
        loss = torch.tensor(0.0, device=device)
        if gt_labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(
                split_logits[mask], gt_split[mask].float())

        return split_logits, cond_out, loss

    # ------------------------------------------------------------------
    # 生成（MaskGIT 迭代，与 OctGPT 对齐）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        parent_xyz: torch.Tensor,
        parent_cond: torch.Tensor,
        batch_ids: torch.Tensor,
        temperature: float = 1.0,
        prefix_tokens: Optional[torch.Tensor] = None,
        prefix_xyz: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MaskGIT 迭代生成 split（支持跨深度 prefix）。

        返回:
            child_8way: (N, 8) {0, 1}
            cond_out: (N, cond_dim_out)
        """
        N = parent_xyz.shape[0]
        device = parent_xyz.device

        # 初始: 全部 masked
        split_d = -1 * torch.ones(N, device=device).long()
        token_features = self.mask_token.expand(N, -1).clone()  # (N, embed_dim)
        mask_d = torch.ones(N, dtype=torch.bool, device=device)
        orders = torch.randperm(N, device=device)

        prefix_len = prefix_tokens.shape[0] if prefix_tokens is not None else 0

        for i in range(self.num_iters):
            # 构建 token（masked 位置用 mask_token + 跨深度 prefix）
            tokens, xyz_m, sort_idx, unsort_idx = self._build_tokens_with_buffer(
                parent_xyz, parent_cond, batch_ids,
                token_features=token_features, mask=mask_d,
                prefix_tokens=prefix_tokens, prefix_xyz=prefix_xyz)

            # 前向
            x = self._forward_transformer(tokens, xyz_m)
            B = int(batch_ids.max().item()) + 1
            x = x[0, B * self.buffer_size + prefix_len:, :]  # (N_morton, dim)

            # Split logits
            logits_m = self.split_head(x).squeeze(-1)  # (N_morton,)
            logits = logits_m[unsort_idx]               # (N,)

            # 余弦 mask 调度
            mask_ratio = math.cos(math.pi / 2. * (i + 1) / self.num_iters)
            mask_len = max(1, min(int(mask_d.sum().item()) - 1,
                                  int(N * mask_ratio)))
            mask_next = torch.zeros(N, dtype=torch.bool, device=device)
            mask_next[orders[:mask_len]] = True

            if i >= self.num_iters - 1:
                mask_to_pred = mask_d
            else:
                mask_to_pred = mask_d & ~mask_next
            mask_d = mask_next

            # 温度衰减
            temp = self.start_temperature * ((self.num_iters - i) / self.num_iters)

            # 在 mask_to_pred 位置采样
            probs = torch.sigmoid(logits[mask_to_pred] / temp)
            sampled = (torch.rand_like(probs) < probs).long()
            split_d[mask_to_pred] = sampled

            # 更新 token
            token_features[mask_to_pred] = self.split_emb(sampled)

        # split → 8-way
        child_8way = split_d.unsqueeze(-1).repeat(1, 8).float()  # (N, 8)

        # 条件向量（最后一次前向的特征）
        cond_m = self.cond_head(x)       # (N_morton, cond_out)
        cond_out = cond_m[unsort_idx, :]  # (N, cond_out)

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
