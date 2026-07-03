"""
OctreeFractalGen: 递归多模型八叉树生成器。

从 FractalGen 的递归架构适配到 3D 八叉树。每层 AR generator
独立处理一个深度转换的 split 预测。终端 VQHead 预测 BSQ 几何编码。

架构（fractal_levels=(3,4), depth_stop=5）:
  Level 0（深度 3）: OctreeAR → split → 子节点在深度 4
  Level 1（深度 4）: OctreeAR → split → 子节点在深度 5
  Level 2（终端）:    VQHead @ depth_stop=5 → BSQ 编码 → VQ-VAE 解码

fractal_levels 仅列出 AR generator 的深度；VQHead 是递归终止时的
额外终端层，始终在 depth_stop 处操作。
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.model.ar_octree import OctreeAR
from src.utils.octree_ops import get_node_xyz, get_split_labels


# ---------------------------------------------------------------------------
# VQHead: 最终层 BSQ 编码预测器
# ---------------------------------------------------------------------------

class VQHead(nn.Module):
    """在最终深度预测 BSQ（二值球面量化）编码。

    每个 depth_stop 处的叶子节点需要一个 VQ 编码来描述其局部几何。
    对于 BSQ（D 组），编码是 D 个独立的二值 → D × 2 类交叉熵损失。

    参数:
        embed_dim: 内部特征维度
        cond_dim_in: 条件输入维度
        vq_groups: BSQ 量化组数（从 VQVAEWrapper 获取）
    """

    def __init__(self, embed_dim: int = 128, cond_dim_in: int = 512,
                 vq_groups: int = 64):
        super().__init__()
        self.vq_groups = vq_groups
        self.vq_size = 2  # BSQ: 每组二值化

        self.cond_proj = nn.Linear(cond_dim_in, embed_dim, bias=True)
        self.pos_emb = nn.Linear(3, embed_dim, bias=True)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)

        # 轻量 MLP: 位置 + 条件 → 每组的 VQ logits
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.vq_groups * self.vq_size),
        )
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

    def forward(
        self,
        parent_xyz: torch.Tensor,
        parent_cond: torch.Tensor,
        vq_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """为叶子节点预测 VQ 编码。

        参数:
            parent_xyz: (B, Np, 3) depth_stop-1 处的叶子节点坐标
            parent_cond: (B, Np, cond_dim_in) 每节点条件
            vq_targets: (nnum, vq_groups) ground-truth BSQ indices (每组 0/1)

        返回:
            logits: (B*Np, vq_groups, 2) VQ 分类 logits
            loss: 标量交叉熵损失
        """
        B, Np, _ = parent_xyz.shape
        device = parent_xyz.device
        total_nodes = B * Np

        pos_emb = self.pos_emb(parent_xyz.float())        # (B, Np, embed_dim)
        cond_emb = self.cond_proj(parent_cond)              # (B, Np, embed_dim)
        h = self.norm(pos_emb + cond_emb)                   # (B, Np, embed_dim)
        logits = self.mlp(h)                                # (B, Np, vq_groups*2)
        logits = logits.view(B, Np, self.vq_groups, self.vq_size)
        logits_flat = logits.reshape(total_nodes, self.vq_groups, self.vq_size)

        loss = torch.tensor(0.0, device=device)
        if vq_targets is not None:
            targets_flat = vq_targets.reshape(-1, self.vq_groups).long()
            # 每组独立的 2 类交叉熵
            loss = nn.functional.cross_entropy(
                logits_flat.reshape(-1, self.vq_size),
                targets_flat.reshape(-1),
                reduction='mean',
            )

        return logits_flat, loss

    @torch.no_grad()
    def sample(
        self,
        parent_xyz: torch.Tensor,
        parent_cond: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """在最终深度采样 VQ indices。

        返回:
            indices: (nnum, vq_groups) 预测的 BSQ indices {0, 1}
        """
        logits, _ = self.forward(parent_xyz, parent_cond)

        # 温度缩放
        logits = logits / temperature

        # 按组采样（2 类 categorical）
        probs = torch.softmax(logits, dim=-1)              # (nnum, vq_groups, 2)
        indices = torch.multinomial(
            probs.reshape(-1, self.vq_size), num_samples=1
        ).squeeze(-1)                                      # (nnum * vq_groups,)
        indices = indices.reshape(-1, self.vq_groups)      # (nnum, vq_groups)

        return indices


# ---------------------------------------------------------------------------
# OctreeFractalGen: 递归多模型容器
# ---------------------------------------------------------------------------

class OctreeFractalGen(nn.Module):
    """递归多模型八叉树生成器，配合 VQ-VAE 解码。

    中间层使用 OctreeAR 预测八叉树结构（split）。
    最终层使用 VQHead 预测 BSQ 几何编码，由冻结的预训练
    VQ-VAE 解码为 mesh。

    层级结构:
      fractal_levels = (3, 4, 5)  → 3 个 AR generator，在 depth 3, 4, 5 做 split
      depth_stop = 6              → 在第 0..2 层 AR 之后，终端 VQHead 在 depth 6
      递归: Level 0→1→2→Terminal(VQHead @ depth_stop)

    参数:
        config: ModelConfig
        vqvae_wrapper: VQVAEWrapper（冻结的 VQ-VAE，用于编码/解码）
        fractal_level: 内部递归计数器（顶层为 0）
    """

    def __init__(self, config, vqvae_wrapper=None, fractal_level: int = 0):
        super().__init__()
        self.config = config
        self.fractal_level = fractal_level
        self.num_ar_levels = len(config.fractal_levels)
        self.is_terminal = (fractal_level >= self.num_ar_levels)
        self.is_ar = not self.is_terminal
        self.vqvae_wrapper = vqvae_wrapper

        # current_depth: AR 层用 fractal_levels，终端用 depth_stop
        if self.is_ar:
            self.current_depth = config.fractal_levels[fractal_level]
        else:
            self.current_depth = config.depth_stop

        # ------------------------------------------------------------------
        # 类别嵌入（仅顶层）
        # ------------------------------------------------------------------
        if fractal_level == 0:
            self.num_classes = config.num_classes
            self.class_emb = nn.Embedding(
                config.num_classes, config.cond_embed_dim,
            )
            self.label_drop_prob = config.label_drop_prob
            self.fake_latent = nn.Parameter(torch.zeros(1, config.cond_embed_dim))
            nn.init.normal_(self.class_emb.weight, std=0.02)
            nn.init.normal_(self.fake_latent, std=0.02)

        # ------------------------------------------------------------------
        # 当前层生成器（仅 AR 层）
        # ------------------------------------------------------------------
        if self.is_ar:
            idx = fractal_level  # index into per-level capacity arrays
            self.generator = OctreeAR(
                embed_dim=config.embed_dims[idx],
                num_blocks=config.num_blocks[idx],
                num_heads=config.num_heads[idx],
                cond_dim_in=config.cond_embed_dim,
                cond_dim_out=config.cond_embed_dim,
                attn_drop=config.attn_drop,
                proj_drop=config.proj_drop,
                grad_checkpointing=config.grad_checkpointing,
            )
        else:
            self.generator = None

        # ------------------------------------------------------------------
        # 下一层（递归 AR 或终止 VQHead）
        # ------------------------------------------------------------------
        if self.is_ar:
            self.next_fractal = OctreeFractalGen(
                config, vqvae_wrapper, fractal_level + 1,
            )
        else:
            vq_groups = 64
            if vqvae_wrapper is not None:
                vq_groups = vqvae_wrapper.get_vq_config()['vq_groups']
            self.next_fractal = VQHead(
                embed_dim=config.embed_dims[-1],
                cond_dim_in=config.cond_embed_dim,
                vq_groups=vq_groups,
            )

    # ------------------------------------------------------------------
    # 条件辅助函数
    # ------------------------------------------------------------------

    def _get_class_condition(self, octree, labels=None) -> torch.Tensor:
        """获取类别条件嵌入，训练时随机 dropout（CFG 训练）。"""
        B = octree.batch_size
        device = octree.device
        if labels is None:
            labels = torch.zeros(B, dtype=torch.long, device=device)
        class_embedding = self.class_emb(labels)
        if self.training:
            drop_mask = (
                torch.rand(B, device=device) < self.label_drop_prob
            ).float().unsqueeze(-1)
            class_embedding = (
                drop_mask * self.fake_latent + (1 - drop_mask) * class_embedding
            )
        return class_embedding

    def _make_per_node_cond(self, global_cond, nnum, B):
        """将全局条件扩展到每节点。"""
        Np = nnum // B
        return global_cond.unsqueeze(1).expand(B, Np, -1)

    # ------------------------------------------------------------------
    # 训练前向传播
    # ------------------------------------------------------------------

    def forward(self, octree, labels=None) -> torch.Tensor:
        """顶层入口：计算类别条件并递归。

        参数:
            octree: ocnn.Octree（ground truth）
            labels: (B,) 类别标签（None → 无条件）

        返回:
            total_loss: 所有层级的损失之和
        """
        if self.fractal_level != 0:
            raise RuntimeError("forward() 仅可在顶层模型上调用")
        B = octree.batch_size
        class_cond = self._get_class_condition(octree, labels)
        return self._forward_level(octree, class_cond)

    def _forward_level(self, octree, global_cond) -> torch.Tensor:
        """通过一个递归层级做前向传播。

        AR 层: 预测 split（8-way 占用率的 BCE）。
        终端层: 预测 VQ 编码（BSQ indices 的交叉熵）。
        """
        B = octree.batch_size
        device = octree.device

        if self.is_ar:
            # --- AR 层: split 预测 ---
            depth = self.current_depth
            parent_xyz, _ = get_node_xyz(octree, depth)
            nnum = octree.nnum[depth]
            if nnum == 0:
                return torch.tensor(0.0, device=device)

            Np = nnum // B
            parent_xyz_3d = parent_xyz.view(B, Np, 3)
            parent_cond = self._make_per_node_cond(global_cond, nnum, B)

            gt_labels = get_split_labels(octree, depth).view(B, Np, 8)
            _, _, level_loss = self.generator(
                parent_xyz_3d, parent_cond, gt_labels)

            deeper_loss = self.next_fractal._forward_level(octree, global_cond)
            return level_loss + deeper_loss

        else:
            # --- 终端 VQHead 层: VQ 编码预测 ---
            if self.vqvae_wrapper is None:
                raise RuntimeError("终端层训练需要 vqvae_wrapper")

            final_depth = self.config.depth_stop
            leaf_xyz, _ = get_node_xyz(octree, final_depth)
            nnum_leaf = octree.nnum[final_depth]
            if nnum_leaf == 0:
                return torch.tensor(0.0, device=device)

            Np_final = nnum_leaf // B
            leaf_xyz_3d = leaf_xyz.view(B, Np_final, 3)
            leaf_cond = self._make_per_node_cond(global_cond, nnum_leaf, B)

            vq_targets = self.vqvae_wrapper.extract_targets(octree)
            _, level_loss = self.next_fractal(
                leaf_xyz_3d, leaf_cond, vq_targets)

            return level_loss

    # ------------------------------------------------------------------
    # 生成（推理）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, octree, labels=None, temperature=1.0, cfg_scale=1.0):
        """递归生成八叉树结构 + VQ 编码。

        返回:
            octree: 生成到 depth_stop 的八叉树
            vq_indices: (nnum_leaf, vq_groups) 预测的 BSQ indices
        """
        if self.fractal_level != 0:
            raise RuntimeError("generate() 仅可在顶层模型上调用")

        B = octree.batch_size
        device = octree.device
        if labels is None:
            labels = torch.zeros(B, dtype=torch.long, device=device)

        class_cond = self.class_emb(labels)
        uncond = self.fake_latent.expand(B, -1) if cfg_scale != 1.0 else None

        return self._generate_level(
            octree, class_cond, temperature, cfg_scale, uncond)

    @torch.no_grad()
    def _generate_level(self, octree, global_cond, temperature,
                        cfg_scale, uncond=None):
        """生成一个八叉树深度并递归。

        AR 层: 采样 split → 展开八叉树 → 递归。
        终端层: 在 depth_stop 处采样 VQ 编码。
        """
        B = octree.batch_size

        if self.is_ar:
            # --- AR 层: 采样 split, 展开八叉树 ---
            depth = self.current_depth
            parent_xyz, _ = get_node_xyz(octree, depth)
            nnum = octree.nnum[depth]
            if nnum == 0:
                return octree, None

            Np = nnum // B
            parent_xyz_3d = parent_xyz.view(B, Np, 3)
            parent_cond = self._make_per_node_cond(global_cond, nnum, B)

            child_8way, _ = self.generator.sample(
                parent_xyz_3d, parent_cond, temperature,
            )
            # 任一子节点被占据 → 分裂父节点
            split_label = child_8way.any(dim=-1).long().reshape(B * Np)
            octree.octree_split(split_label, depth=depth)
            octree.octree_grow(depth + 1)

            return self.next_fractal._generate_level(
                octree, global_cond, temperature, cfg_scale, uncond,
            )

        else:
            # --- 终端层: 在 depth_stop 处采样 VQ 编码 ---
            final_depth = self.config.depth_stop
            leaf_xyz, _ = get_node_xyz(octree, final_depth)
            nnum_leaf = octree.nnum[final_depth]
            if nnum_leaf == 0:
                return octree, None

            Np_leaf = nnum_leaf // B
            leaf_xyz_3d = leaf_xyz.view(B, Np_leaf, 3)
            leaf_cond = self._make_per_node_cond(global_cond, nnum_leaf, B)

            vq_indices = self.next_fractal.sample(
                leaf_xyz_3d, leaf_cond, temperature,
            )
            return octree, vq_indices


# ------------------------------------------------------------------
# 工厂函数（遵循 FractalGen 命名约定）
# ------------------------------------------------------------------

def octree_fractal_tiny(config=None, vqvae_wrapper=None):
    """微型模型，用于快速迭代。"""
    if config is None:
        from src.config import octree_fractal_tiny as make_cfg
        config = make_cfg().model
    return OctreeFractalGen(config, vqvae_wrapper, fractal_level=0)


def octree_fractal_base(config=None, vqvae_wrapper=None):
    """基础模型。"""
    if config is None:
        from src.config import octree_fractal_base as make_cfg
        config = make_cfg().model
    return OctreeFractalGen(config, vqvae_wrapper, fractal_level=0)


def octree_fractal_large(config=None, vqvae_wrapper=None):
    """大型模型。"""
    if config is None:
        from src.config import octree_fractal_large as make_cfg
        config = make_cfg().model
    return OctreeFractalGen(config, vqvae_wrapper, fractal_level=0)
