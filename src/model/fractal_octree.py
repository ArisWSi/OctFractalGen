"""
OctreeFractalGen: 递归多模型八叉树生成器。

每层使用 OctreeLevelAR（复用 OctGPT OctFormer）预测分裂。
终端 VQHead 预测 BSQ 几何编码。

全部使用展平张量 (nnum, ...) + batch_ids，兼容稀疏八叉树。
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.model.octree_ar import OctreeLevelAR
from src.utils.octree_ops import get_node_xyz, get_split_labels


# ══════════════════════════════════════════════════════════════════════════
# VQHead: 终端 BSQ 编码预测器（纯 MLP）
# ══════════════════════════════════════════════════════════════════════════

class VQHead(nn.Module):
    """在 depth_stop 处预测 BSQ 编码。

    对于 BSQ（D 组），编码是 D 个独立二值 → D×2 交叉熵。
    """

    def __init__(self, embed_dim: int = 128, cond_dim_in: int = 512,
                 vq_groups: int = 64):
        super().__init__()
        self.vq_groups = vq_groups
        self.vq_size = 2

        self.cond_proj = nn.Linear(cond_dim_in, embed_dim, bias=True)
        self.pos_emb = nn.Linear(3, embed_dim, bias=True)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, vq_groups * 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, xyz, cond, vq_targets=None):
        """xyz: (N, 3), cond: (N, cond_dim), vq_targets: (N, vq_groups)"""
        h = self.norm(self.pos_emb(xyz.float()) + self.cond_proj(cond))
        logits = self.mlp(h).view(-1, self.vq_groups, self.vq_size)
        loss = torch.tensor(0.0, device=xyz.device)
        if vq_targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, 2), vq_targets.reshape(-1).long(),
                reduction='mean')
        return logits, loss

    @torch.no_grad()
    def sample(self, xyz, cond, temperature=1.0):
        logits, _ = self.forward(xyz, cond)
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(
            probs.reshape(-1, 2), 1).squeeze(-1).reshape(-1, self.vq_groups)


# ══════════════════════════════════════════════════════════════════════════
# OctreeFractalGen — 递归多模型容器
# ══════════════════════════════════════════════════════════════════════════

class OctreeFractalGen(nn.Module):
    """递归多模型八叉树生成器。

    每 AR 层: OctreeLevelAR（OctFormer）→ split 预测
    终端层:   VQHead（纯 MLP）→ BSQ 编码 → VQ-VAE 解码

    fractal_levels 仅列出 AR generator 深度；VQHead 是递归终止后的额外终端层。
    """

    def __init__(self, config, vqvae_wrapper=None, fractal_level: int = 0):
        super().__init__()
        self.config = config
        self.fractal_level = fractal_level
        self.num_ar_levels = len(config.fractal_levels)
        self.is_terminal = (fractal_level >= self.num_ar_levels)
        self.is_ar = not self.is_terminal
        self.vqvae_wrapper = vqvae_wrapper

        self.current_depth = (config.fractal_levels[fractal_level]
                              if self.is_ar else config.depth_stop)

        # ── 类别嵌入（仅顶层）────────────────────────────────
        if fractal_level == 0:
            self.class_emb = nn.Embedding(config.num_classes,
                                          config.cond_embed_dim)
            self.label_drop_prob = config.label_drop_prob
            self.fake_latent = nn.Parameter(
                torch.zeros(1, config.cond_embed_dim))
            nn.init.normal_(self.class_emb.weight, std=0.02)
            nn.init.normal_(self.fake_latent, std=0.02)

        # ── AR generator / VQHead ────────────────────────────
        if self.is_ar:
            idx = fractal_level
            self.generator = OctreeLevelAR(
                embed_dim=config.embed_dims[idx],
                num_blocks=config.num_blocks[idx],
                num_heads=config.num_heads[idx],
                cond_dim_in=config.cond_embed_dim,
                cond_dim_out=config.cond_embed_dim,
                patch_size=config.patch_size,
                dilation=config.dilation,
                drop_rate=config.attn_drop,
                use_swin=True,
                use_checkpoint=config.grad_checkpointing,
            )
            self.next_fractal = OctreeFractalGen(
                config, vqvae_wrapper, fractal_level + 1)
        else:
            self.generator = None
            vq_groups = (vqvae_wrapper.get_vq_config()['vq_groups']
                         if vqvae_wrapper else 64)
            self.next_fractal = VQHead(
                embed_dim=config.embed_dims[-1],
                cond_dim_in=config.cond_embed_dim,
                vq_groups=vq_groups)

    # ── 条件辅助 ─────────────────────────────────────────────

    def _get_class_condition(self, octree, labels=None):
        B = octree.batch_size
        device = octree.device
        if labels is None:
            labels = torch.zeros(B, dtype=torch.long, device=device)
        emb = self.class_emb(labels)
        if self.training:
            drop = (torch.rand(B, device=device) < self.label_drop_prob).float()
            emb = drop.unsqueeze(-1) * self.fake_latent + (1 - drop).unsqueeze(-1) * emb
        return emb  # (B, C)

    def _cond_per_node(self, global_cond, octree, depth):
        """将 per-batch 条件 (B, C) 展开为 per-node (nnum, C)。"""
        batch_ids = octree.batch_id(depth)  # (nnum,)
        return global_cond[batch_ids]        # (nnum, C)

    # ── 训练前向 ─────────────────────────────────────────────

    def forward(self, octree, labels=None):
        if self.fractal_level != 0:
            raise RuntimeError("forward() 仅可在顶层模型上调用")
        class_cond = self._get_class_condition(octree, labels)
        return self._forward_level(octree, class_cond)

    def _forward_level(self, octree, global_cond):
        B, device = octree.batch_size, octree.device

        if self.is_ar:
            depth = self.current_depth
            nnum = octree.nnum[depth]
            if nnum == 0:
                return torch.tensor(0.0, device=device)

            parent_xyz, _ = get_node_xyz(octree, depth)              # (nnum, 3)
            parent_batch = octree.batch_id(depth)                     # (nnum,)
            parent_cond = self._cond_per_node(global_cond, octree, depth)

            gt_labels = get_split_labels(octree, depth)               # (nnum, 8)

            _, _, level_loss = self.generator(
                parent_xyz, parent_batch, parent_cond, gt_labels)

            return level_loss + self.next_fractal._forward_level(
                octree, global_cond)
        else:
            # 终端 VQHead
            if self.vqvae_wrapper is None:
                raise RuntimeError("终端层需要 vqvae_wrapper")

            final_depth = self.config.depth_stop
            nnum = octree.nnum[final_depth]
            if nnum == 0:
                return torch.tensor(0.0, device=device)

            leaf_xyz, _ = get_node_xyz(octree, final_depth)          # (nnum, 3)
            leaf_cond = self._cond_per_node(global_cond, octree, final_depth)
            vq_targets = self.vqvae_wrapper.extract_targets(octree)   # (nnum, vq_groups)

            _, loss = self.next_fractal(leaf_xyz, leaf_cond, vq_targets)
            return loss

    # ── 生成 ─────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, octree, labels=None, temperature=1.0, cfg_scale=1.0):
        if self.fractal_level != 0:
            raise RuntimeError("generate() 仅可在顶层模型上调用")
        B, device = octree.batch_size, octree.device
        if labels is None:
            labels = torch.zeros(B, dtype=torch.long, device=device)
        class_cond = self.class_emb(labels)
        uncond = self.fake_latent.expand(B, -1) if cfg_scale != 1.0 else None
        return self._generate_level(octree, class_cond, temperature,
                                    cfg_scale, uncond)

    @torch.no_grad()
    def _generate_level(self, octree, global_cond, temperature,
                        cfg_scale, uncond=None):
        B = octree.batch_size

        if self.is_ar:
            depth = self.current_depth
            nnum = octree.nnum[depth]
            if nnum == 0:
                return octree, None

            parent_xyz, _ = get_node_xyz(octree, depth)
            parent_batch = octree.batch_id(depth)
            parent_cond = self._cond_per_node(global_cond, octree, depth)

            child_8way, _ = self.generator.sample(
                parent_xyz, parent_batch, parent_cond, temperature)

            # 任一子节点被占据 → 分裂父节点
            split_label = child_8way.any(dim=-1).long()
            octree.octree_split(split_label, depth=depth)
            octree.octree_grow(depth + 1)

            return self.next_fractal._generate_level(
                octree, global_cond, temperature, cfg_scale, uncond)
        else:
            final_depth = self.config.depth_stop
            nnum = octree.nnum[final_depth]
            if nnum == 0:
                return octree, None

            leaf_xyz, _ = get_node_xyz(octree, final_depth)
            leaf_cond = self._cond_per_node(global_cond, octree, final_depth)
            vq_indices = self.next_fractal.sample(
                leaf_xyz, leaf_cond, temperature)
            return octree, vq_indices
