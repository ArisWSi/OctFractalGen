
"""FractalOctGPT: 多层串联的 OctGPT 宏观架构。

核心思想（用户指定）：
  - 微观（层内）：完全复用 OctGPT 的 OctFormer + OctreeT + teacher forcing
    + SWIN + MaskGIT，不自行实现残缺的 transformer。
  - 宏观（层间）：把 OctGPT 的"单模型处理多深度"改为"多层串联"，
    每层独立参数处理一个深度，通过 prefix tokens 传递跨深度信息。

与 OctreeFractalGen 的关系：
  接口完全对齐（forward/generate/config），可直接替换。
  内部用 OctGPTLayer 替代之前的 OctreeAR + VQHead。

接口契约:
  - forward(octree, labels=None) → scalar loss
  - generate(octree, labels, temperature, cfg_scale) → (octree, vq_indices)
  - model.config → ModelConfig
"""

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.model.octgpt_layer import OctGPTLayer
from src.utils.octree_ops import get_node_xyz, get_split_labels


class FractalOctGPT(nn.Module):
    """多层串联 OctGPT（FractalGen 宏观 + OctGPT 微观）。

    参数:
        config: ModelConfig dataclass
        vqvae_wrapper: VQVAEWrapper（终端 VQ 层用）
        fractal_level: 递归层级（顶层=0）
    """

    def __init__(self, config, vqvae_wrapper=None, fractal_level: int = 0):
        super().__init__()
        self.config = config
        self.model_config = config  # 别名，兼容 generate.py 访问
        self.vqvae_wrapper = vqvae_wrapper
        self.fractal_level = fractal_level

        # AR split 深度列表 + 终端 VQ 深度
        ar_depths = list(config.fractal_levels)  # e.g. [3, 4, 5]
        self.depth_stop = config.depth_stop      # e.g. 6
        self.full_depth = config.full_depth      # e.g. 3
        n_levels = len(ar_depths)

        # 从 config 读取 FractalOctGPT 架构参数
        num_blocks = config.num_blocks  # 逐层 block 数
        embed_dims = config.embed_dims   # 逐层维度
        num_heads = config.num_heads     # 逐层头数
        use_swin = getattr(config, 'use_swin', True)
        pos_emb_type = getattr(config, 'pos_emb_type', 'sin')
        # 每层维度/头数/blocks（VQ 层用最后一个 AR 层的配置）
        if fractal_level < n_levels:
            idx = fractal_level
            layer_dim = embed_dims[idx] if idx < len(embed_dims) else embed_dims[-1]
            layer_heads = num_heads[idx] if idx < len(num_heads) else num_heads[-1]
            layer_blocks = num_blocks[idx] if idx < len(num_blocks) else num_blocks[-1]
        else:
            layer_dim = embed_dims[-1]
            layer_heads = num_heads[-1]
            layer_blocks = num_blocks[-1]
        assert (layer_dim // layer_heads) % 6 == 0, \
            f"head_dim={layer_dim//layer_heads} 必须被 6 整除（RoPE 约束），" \
            f"dim={layer_dim}, heads={layer_heads}"

        # 当前层
        is_terminal = (fractal_level >= n_levels)
        if not is_terminal:
            # AR split 层
            self.is_ar = True
            self.current_depth = ar_depths[fractal_level]
            self.layer = OctGPTLayer(
                num_embed=layer_dim,
                num_heads=layer_heads,
                num_blocks=layer_blocks,
                patch_size=config.patch_size,
                dilation=config.dilation,
                buffer_size=config.buffer_size,
                is_vq=False,
                num_iters=(config.num_iters[fractal_level]
                           if fractal_level < len(config.num_iters) else 128),
                start_temperature=(
                    config.start_temperature[fractal_level]
                    if fractal_level < len(config.start_temperature) else 1.0),
                remask_stage=config.remask_stage,
                random_flip=config.random_flip,
                drop_rate=config.proj_drop,
                attn_drop=config.attn_drop,
                proj_drop=config.proj_drop,
                use_swin=use_swin,
                pos_emb_type=pos_emb_type,
                use_checkpoint=config.grad_checkpointing,
            )
            self.class_emb = nn.Embedding(
                max(config.num_classes, 1), layer_dim)
            self.next_fractal = FractalOctGPT(
                config, vqvae_wrapper, fractal_level + 1)
        else:
            # 终端 VQ 层
            self.is_ar = False
            self.current_depth = self.depth_stop
            self.layer = OctGPTLayer(
                num_embed=layer_dim,
                num_heads=layer_heads,
                num_blocks=layer_blocks,
                patch_size=config.patch_size,
                dilation=config.dilation,
                buffer_size=config.buffer_size,
                is_vq=True,
                vq_groups=_get_vq_groups(vqvae_wrapper),
                num_vq_embed=_get_num_vq_embed(vqvae_wrapper),
                num_iters=(config.num_iters[-1]
                           if len(config.num_iters) > n_levels else 256),
                start_temperature=(
                    config.start_temperature[-1]
                    if len(config.start_temperature) > n_levels else 0.5),
                remask_stage=config.remask_stage,
                random_flip=config.random_flip,
                drop_rate=config.proj_drop,
                attn_drop=config.attn_drop,
                proj_drop=config.proj_drop,
                use_swin=use_swin,
                pos_emb_type=pos_emb_type,
                use_checkpoint=config.grad_checkpointing,
            )
            self.class_emb = nn.Embedding(
                max(config.num_classes, 1), layer_dim)
            self.next_fractal = None

        # CFG: 无条件嵌入（用第一层维度）
        self.fake_latent = nn.Parameter(torch.zeros(1, embed_dims[0]))
        # 跨层 cond 投影：上一层 dim → 当前层 dim（维度不同时）
        if fractal_level > 0:
            prev_dim = embed_dims[fractal_level - 1] if fractal_level <= len(embed_dims) else embed_dims[-1]
            self.cond_proj_in = nn.Linear(prev_dim, layer_dim, bias=False) if prev_dim != layer_dim else nn.Identity()
        else:
            self.cond_proj_in = nn.Identity()
        nn.init.normal_(self.fake_latent, std=0.02)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _get_class_condition(self, octree, labels):
        """类别条件 (B, dim)。无条件时用 0。"""
        B = octree.batch_size
        device = octree.device
        dim = self.class_emb.embedding_dim
        if labels is None:
            labels = torch.zeros(B, dtype=torch.long, device=device)
        # label drop（训练时随机丢弃 → 无条件）
        if self.training and self.config.label_drop_prob > 0:
            drop = torch.rand(B, device=device) < self.config.label_drop_prob
            labels = torch.where(drop, torch.zeros_like(labels), labels)
        return self.class_emb(labels)  # (B, dim)

    @staticmethod
    def _make_per_node_cond(global_cond, batch_ids, nnum):
        """(B, dim) → (nnum, dim) 按节点 batch 归属展开。"""
        return global_cond[batch_ids.long()]

    # ------------------------------------------------------------------
    # 训练前向
    # ------------------------------------------------------------------

    def forward(self, octree, labels=None) -> torch.Tensor:
        """训练前向：逐层 MaskGIT，返回总 loss（标量）。"""
        if self.fractal_level != 0:
            raise RuntimeError("forward() 仅可在顶层模型上调用")
        class_cond = self._get_class_condition(octree, labels)
        return self._forward_level(octree, class_cond)

    def _forward_level(self, octree, global_cond) -> torch.Tensor:
        """递归单层前向。跨深度信息通过 cond（buffer）传递。"""
        B = octree.batch_size
        device = octree.device
        depth = self.current_depth
        # 投影 cond 到当前层维度
        global_cond = self.cond_proj_in(global_cond)

        nnum = octree.nnum[depth]
        if nnum == 0:
            return torch.tensor(0.0, device=device)

        if self.is_ar:
            gt_split_8way = get_split_labels(octree, depth)
            gt_split = (gt_split_8way.sum(dim=-1) > 0).long()
            loss, cond_out, diag = self.layer(
                octree, depth, global_cond, gt_split,
            )
            # cond_out 是 CLS token 输出 (B, dim)，直接传给下一层
            deeper = self.next_fractal._forward_level(octree, cond_out)
            return loss + deeper
        else:
            vq_targets = self.vqvae_wrapper.extract_targets(octree)
            loss, cond_out, diag = self.layer(
                octree, depth, global_cond, vq_targets,
                vqvae=self.vqvae_wrapper.vqvae,
            )
            return loss

    def _proj_cond(self, cond, target_dim, B, device):
        """条件维度投影（若 class_emb 维度与层维度不匹配）。"""
        # 简单方案：用当前层 class_emb 重新取
        labels = torch.zeros(B, dtype=torch.long, device=device)
        return self.class_emb(labels)

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, octree, labels=None, temperature=1.0, cfg_scale=1.0):
        """递归生成八叉树 + VQ 编码。

        返回:
            octree: 生成到 depth_stop 的八叉树
            vq_indices: (nnum_leaf, vq_groups) BSQ indices
        """
        if self.fractal_level != 0:
            raise RuntimeError("generate() 仅可在顶层模型上调用")
        B = octree.batch_size
        device = octree.device
        if labels is None:
            labels = torch.zeros(B, dtype=torch.long, device=device)
        class_cond = self._get_class_condition(octree, labels)
        return self._generate_level(octree, class_cond, temperature, cfg_scale)

    @torch.no_grad()
    def _generate_level(self, octree, global_cond, temperature, cfg_scale):
        B = octree.batch_size
        device = octree.device
        depth = self.current_depth
        # 投影 cond 到当前层维度
        global_cond = self.cond_proj_in(global_cond)

        nnum = octree.nnum[depth]
        if nnum == 0:
            return octree, None

        if self.is_ar:
            split_d, cond_out = self.layer.sample(
                octree, depth, global_cond,
                temperature=temperature,
            )
            octree.octree_split(split_d, depth=depth)
            octree.octree_grow(depth + 1)
            # cond_out 是 CLS token 输出 (B, dim)，直接传给下一层
            return self.next_fractal._generate_level(
                octree, cond_out, temperature, cfg_scale)
        else:
            vq_indices, cond_out = self.layer.sample(
                octree, depth, global_cond,
                temperature=temperature,
                vqvae=self.vqvae_wrapper.vqvae,
            )
            return octree, vq_indices


def _get_vq_groups(vqvae_wrapper):
    """从 wrapper 获取 BSQ 量化组数。"""
    if vqvae_wrapper is None:
        return 32
    if hasattr(vqvae_wrapper, 'vq_groups'):
        return vqvae_wrapper.vq_groups
    q = vqvae_wrapper.vqvae.quantizer
    if hasattr(q, 'embed_dim'):
        return q.embed_dim
    if hasattr(q, 'groups'):
        return q.groups
    return 32


def _get_num_vq_embed(vqvae_wrapper):
    """从 wrapper 获取 VQ-VAE 编码维度。"""
    if vqvae_wrapper is None:
        return 32
    return getattr(vqvae_wrapper.vqvae.quantizer, 'embed_dim', 32)
