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
from src.model.transformer import (
    RMSNorm, PatchTransformerBlock, precompute_freqs_cis_3d,
)
from src.utils.octree_ops import get_node_xyz, get_split_labels


# ---------------------------------------------------------------------------
# VQHead: 最终层 BSQ 编码预测器
# ---------------------------------------------------------------------------

class VQHead(nn.Module):
    """在最终深度预测 BSQ 编码（OctFormer 风格 attention）。

    和 OctreeAR 一样用 PatchTransformerBlock 做节点间 attention，
    让叶子节点能利用邻居信息预测局部几何编码。
    这比纯 MLP 更接近 OctGPT 的做法——OctGPT 的 VQ 预测
    也是在 attention 特征上做的。

    参数:
        embed_dim: 内部嵌入维度
        cond_dim_in: 条件输入维度
        vq_groups: BSQ 量化组数
        num_blocks: attention block 数（轻量，4 层）
        patch_size: patch 注意力大小
        dilation: 膨胀率
        attn_drop, proj_drop: dropout
    """

    def __init__(self, embed_dim: int = 128, cond_dim_in: int = 512,
                 vq_groups: int = 64, num_blocks: int = 4,
                 patch_size: int = 1024, dilation: int = 4,
                 attn_drop: float = 0.1, proj_drop: float = 0.1,
                 buffer_size: int = 64, num_iters: int = 256,
                 start_temperature: float = 0.5, remask_stage: float = 0.7,
                 random_flip: float = 0.1):
        super().__init__()
        self.vq_groups = vq_groups
        self.vq_size = 2
        self.embed_dim = embed_dim
        self.n_head = max(2, embed_dim // 32)  # head_dim=32
        self.patch_size = patch_size
        self.dilation = dilation
        self.buffer_size = buffer_size
        self.num_iters = num_iters
        self.start_temperature = start_temperature
        self.remask_stage = remask_stage
        self.random_flip = random_flip

        # 嵌入
        self.cond_proj = nn.Linear(cond_dim_in, embed_dim, bias=True)
        self.pos_emb = nn.Linear(3, embed_dim, bias=True)
        self.token_ln = RMSNorm(embed_dim, eps=1e-5)

        # OctFormer 风格 patch attention blocks（轻量）
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block_dilation = 1 if (i % 2 == 0) else dilation
            blk = PatchTransformerBlock(
                dim=embed_dim,
                n_head=max(2, embed_dim // 32),  # head_dim=32
                patch_size=patch_size,
                dilation=block_dilation,
                mlp_drop=proj_drop,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            self.blocks.append(blk)
        self.norm = RMSNorm(embed_dim, eps=1e-5)

        # VQ 输出头
        self.vq_head = nn.Linear(embed_dim, vq_groups * self.vq_size, bias=True)

        # MaskGIT: mask token + vq_proj（BSQ code → embedding）
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.vq_proj = nn.Linear(vq_groups, embed_dim, bias=True)
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

    def _build_tokens(self, xyz, cond, batch_ids):
        """构建 Morton 序 token（与 OctreeAR 相同逻辑）。"""
        N = xyz.shape[0]
        sort_idx, unsort_idx = self._morton_sort(xyz, batch_ids)

        xyz_m = xyz[sort_idx, :]
        cond_m = cond[sort_idx, :]

        pos = self.pos_emb(xyz_m.float())
        c = self.cond_proj(cond_m)
        tokens = self.token_ln(pos + c)       # (N, embed_dim)
        tokens = tokens.unsqueeze(0)          # (1, N, embed_dim)
        xyz_m = xyz_m.unsqueeze(0)            # (1, N, 3)
        return tokens, xyz_m, sort_idx, unsort_idx

    def _morton_sort(self, xyz, batch_ids):
        """Morton 排序（含 batch 分块）。"""
        from src.utils.octree_ops import morton_encode_3d
        N = xyz.shape[0]
        device = xyz.device
        codes = morton_encode_3d(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        codes = codes + batch_ids.long() * (codes.max() + 1)
        sort_idx = torch.argsort(codes)
        unsort_idx = torch.empty_like(sort_idx)
        unsort_idx[sort_idx] = torch.arange(N, device=device)
        return sort_idx, unsort_idx

    def _forward_transformer(self, x, xyz=None):
        """RoPE 频率一次性预计算，传给所有 block 复用。"""
        freqs_cis = None
        if xyz is not None:
            B, N, _ = x.shape
            hd = self.embed_dim // self.n_head
            flat_xyz = xyz.reshape(B * N, 3)
            freqs_cis = precompute_freqs_cis_3d(flat_xyz, hd)
        for block in self.blocks:
            x = block(x, freqs_cis)
        return self.norm(x)

    def _build_tokens_with_buffer(self, xyz, cond, batch_ids,
                                  token_features=None, mask=None,
                                  prefix_tokens=None, prefix_xyz=None):
        """构建 token（含 buffer + 跨深度 prefix），与 OctreeAR 类似。

        序列布局: [buffer(B*buf) | prefix(N_prev) | tokens(N)]
        """
        N = xyz.shape[0]
        B = int(batch_ids.max().item()) + 1
        device = xyz.device

        if token_features is None:
            tokens, xyz_m, sort_idx, unsort_idx = self._build_tokens(
                xyz, cond, batch_ids)
        else:
            sort_idx, unsort_idx = self._morton_sort(xyz, batch_ids)
            tokens = token_features[sort_idx].unsqueeze(0)
            xyz_m = xyz[sort_idx].unsqueeze(0)

        if mask is not None:
            mask_m = mask[sort_idx]
            tokens[0, mask_m, :] = self.mask_token.to(tokens.dtype)

        buffer_cond = self.cond_proj(cond[:B])
        buffer = buffer_cond.unsqueeze(1).expand(
            B, self.buffer_size, -1).reshape(1, B * self.buffer_size, -1)
        buffer_xyz = torch.zeros(1, B * self.buffer_size, 3, device=device)

        if prefix_tokens is not None:
            prefix_emb = self.cond_proj(prefix_tokens).unsqueeze(0)
            prefix_xyz_m = prefix_xyz.unsqueeze(0)
            tokens = torch.cat([buffer, prefix_emb, tokens], dim=1)
            xyz_m = torch.cat([buffer_xyz, prefix_xyz_m, xyz_m], dim=1)
        else:
            tokens = torch.cat([buffer, tokens], dim=1)
            xyz_m = torch.cat([buffer_xyz, xyz_m], dim=1)
        return tokens, xyz_m, sort_idx, unsort_idx

    def forward(
        self,
        parent_xyz: torch.Tensor,
        parent_cond: torch.Tensor,
        vq_targets: Optional[torch.Tensor] = None,
        batch_ids: Optional[torch.Tensor] = None,
        prefix_tokens: Optional[torch.Tensor] = None,
        prefix_xyz: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """训练前向：random masking + buffer + 跨深度 prefix → VQ 预测。

        与 OctGPT 对齐:
        1. 用 vq_targets 构建 token (vq_proj)
        2. Random masking + random_flip 增强
        3. 加 buffer + prefix (上一层 token 特征)
        4. Attention → vq_head → CE loss（仅 masked 位置）
        """
        N = parent_xyz.shape[0]
        device = parent_xyz.device

        if batch_ids is None:
            batch_ids = torch.zeros(N, dtype=torch.long, device=device)

        # Random flip 增强（训练时）
        if self.training and self.random_flip > 0 and vq_targets is not None:
            flip = torch.rand_like(vq_targets.float()) < self.random_flip
            vq_targets_for_token = torch.where(flip, 1 - vq_targets, vq_targets)
        else:
            vq_targets_for_token = vq_targets

        # 构建 token: vq_proj(BSQ code → embedding)
        if vq_targets_for_token is not None:
            zq = vq_targets_for_token.float() * 2 - 1  # {0,1} → {-1,1}
            zq = zq * (1.0 / self.vq_groups ** 0.5)
            token_features = self.vq_proj(zq)  # (N, embed_dim)
        else:
            token_features = None

        # Random masking
        if self.training and vq_targets is not None:
            mask_ratio = 0.5 + 0.5 * torch.rand(1, device=device).item()
            num_masked = max(int(N * mask_ratio), 1)
            orders = torch.randperm(N, device=device)
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            mask[orders[:num_masked]] = True
        else:
            mask = None

        # 构建含 buffer + prefix 的 token
        tokens, xyz_m, sort_idx, unsort_idx = self._build_tokens_with_buffer(
            parent_xyz, parent_cond, batch_ids,
            token_features=token_features, mask=mask,
            prefix_tokens=prefix_tokens, prefix_xyz=prefix_xyz)

        # Attention
        x = self._forward_transformer(tokens, xyz_m)
        B = int(batch_ids.max().item()) + 1
        prefix_len = prefix_tokens.shape[0] if prefix_tokens is not None else 0
        x = x[0, B * self.buffer_size + prefix_len:, :]  # (N_morton, dim)
        x = x[unsort_idx, :]                 # (N, dim)

        logits = self.vq_head(x).view(N, self.vq_groups, self.vq_size)

        loss = torch.tensor(0.0, device=device)
        if vq_targets is not None and mask is not None:
            # CE loss 仅在 masked 位置
            targets_flat = vq_targets[mask].reshape(-1, self.vq_groups).long()
            logits_masked = logits[mask]
            loss = nn.functional.cross_entropy(
                logits_masked.reshape(-1, self.vq_size),
                targets_flat.reshape(-1),
                reduction='mean',
            )
        elif vq_targets is not None:
            targets_flat = vq_targets.reshape(-1, self.vq_groups).long()
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.vq_size),
                targets_flat.reshape(-1),
                reduction='mean',
            )

        return logits, loss

    @torch.no_grad()
    def sample(
        self,
        parent_xyz: torch.Tensor,
        parent_cond: torch.Tensor,
        temperature: float = 1.0,
        batch_ids: Optional[torch.Tensor] = None,
        prefix_tokens: Optional[torch.Tensor] = None,
        prefix_xyz: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MaskGIT 迭代生成 VQ indices（支持跨深度 prefix）。"""
        import math
        N = parent_xyz.shape[0]
        device = parent_xyz.device
        if batch_ids is None:
            batch_ids = torch.zeros(N, dtype=torch.long, device=device)

        # 初始: 全部 masked
        vq_indices_d = -1 * torch.ones(N, self.vq_groups, device=device).long()
        token_features = self.mask_token.expand(N, -1).clone()
        mask_d = torch.ones(N, dtype=torch.bool, device=device)
        orders = torch.randperm(N, device=device)

        prefix_len = prefix_tokens.shape[0] if prefix_tokens is not None else 0

        for i in range(self.num_iters):
            tokens, xyz_m, sort_idx, unsort_idx = self._build_tokens_with_buffer(
                parent_xyz, parent_cond, batch_ids,
                token_features=token_features, mask=mask_d,
                prefix_tokens=prefix_tokens, prefix_xyz=prefix_xyz)

            x = self._forward_transformer(tokens, xyz_m)
            B = int(batch_ids.max().item()) + 1
            x = x[0, B * self.buffer_size + prefix_len:, :]
            x = x[unsort_idx, :]

            logits = self.vq_head(x).view(N, self.vq_groups, self.vq_size)

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
            probs = torch.softmax(
                logits[mask_to_pred] / temp, dim=-1)  # (M, vq_groups, 2)
            sampled = torch.multinomial(
                probs.reshape(-1, self.vq_size), 1
            ).squeeze(-1).reshape(-1, self.vq_groups)
            vq_indices_d[mask_to_pred] = sampled

            # 更新 token
            zq = sampled.float() * 2 - 1
            zq = zq * (1.0 / self.vq_groups ** 0.5)
            token_features[mask_to_pred] = self.vq_proj(zq)

        return vq_indices_d


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
                patch_size=config.patch_size,
                dilation=config.dilation,
                attn_drop=config.attn_drop,
                proj_drop=config.proj_drop,
                grad_checkpointing=config.grad_checkpointing,
                buffer_size=config.buffer_size,
                num_iters=config.num_iters[idx],
                start_temperature=config.start_temperature[idx],
                remask_stage=config.remask_stage,
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
                num_blocks=4,
                patch_size=config.patch_size,
                dilation=config.dilation,
                attn_drop=config.attn_drop,
                proj_drop=config.proj_drop,
                buffer_size=config.buffer_size,
                num_iters=config.num_iters[-1],
                start_temperature=config.start_temperature[-1],
                remask_stage=config.remask_stage,
                random_flip=config.random_flip,
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

    def _make_per_node_cond(self, global_cond, batch_ids, nnum):
        """将全局条件按 batch 归属展开到每个节点（OctGPT 风格）。

        支持稀疏八叉树（每个 shape 的节点数不同）。

        参数:
            global_cond: (B, cond_dim) 全局条件
            batch_ids: (nnum,) 每个节点的 batch 索引
            nnum: 总节点数

        返回:
            per_node_cond: (nnum, cond_dim) 每节点条件
        """
        return global_cond[batch_ids]  # (nnum, cond_dim)

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

    def _forward_level(self, octree, global_cond,
                       prefix_tokens=None, prefix_xyz=None) -> torch.Tensor:
        """通过一个递归层级做前向传播。

        跨深度 token 交互: 每层 AR 的 cond_out + parent_xyz 作为
        下一层的 prefix，让下一层 token 能 attend 到上一层节点特征
        （而非只收到标量条件），对齐 OctGPT 的跨深度 attention。

        AR 层: OctFormer 并行预测 split（BCE）。
        终端层: VQHead 预测 VQ 编码（BSQ indices 的交叉熵）。
        """
        B = octree.batch_size
        device = octree.device

        if self.is_ar:
            # --- AR 层: split 预测 ---
            depth = self.current_depth
            parent_xyz, batch_ids = get_node_xyz(octree, depth)
            nnum = octree.nnum[depth]
            if nnum == 0:
                return torch.tensor(0.0, device=device)

            # 每节点条件（按 batch 归属展开）
            parent_cond = self._make_per_node_cond(
                global_cond, batch_ids, nnum)  # (nnum, cond_dim)

            # ground-truth split 标签: (nnum, 8)
            gt_labels = get_split_labels(octree, depth)

            # AR 前向（展平输入 + batch_ids + 跨深度 prefix）
            _, cond_out, level_loss = self.generator(
                parent_xyz, parent_cond, gt_labels, batch_ids,
                prefix_tokens=prefix_tokens, prefix_xyz=prefix_xyz)

            # cond_out + parent_xyz 传给下一层作为 prefix（跨深度 token 交互）
            deeper_loss = self.next_fractal._forward_level(
                octree, global_cond,
                prefix_tokens=cond_out, prefix_xyz=parent_xyz)
            return level_loss + deeper_loss

        else:
            # --- 终端 VQHead 层: VQ 编码预测 ---
            if self.vqvae_wrapper is None:
                raise RuntimeError("终端层训练需要 vqvae_wrapper")

            final_depth = self.config.depth_stop
            leaf_xyz, leaf_batch_ids = get_node_xyz(octree, final_depth)
            nnum_leaf = octree.nnum[final_depth]
            if nnum_leaf == 0:
                return torch.tensor(0.0, device=device)

            leaf_cond = self._make_per_node_cond(
                global_cond, leaf_batch_ids, nnum_leaf)

            vq_targets = self.vqvae_wrapper.extract_targets(octree)

            _, level_loss = self.next_fractal(
                leaf_xyz, leaf_cond, vq_targets, leaf_batch_ids,
                prefix_tokens=prefix_tokens, prefix_xyz=prefix_xyz)

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
                        cfg_scale, uncond=None,
                        prefix_tokens=None, prefix_xyz=None):
        """生成一个八叉树深度并递归（含跨深度 prefix）。"""
        B = octree.batch_size

        if self.is_ar:
            # --- AR 层: OctFormer 并行采样 split ---
            depth = self.current_depth
            parent_xyz, batch_ids = get_node_xyz(octree, depth)
            nnum = octree.nnum[depth]
            if nnum == 0:
                return octree, None

            parent_cond = self._make_per_node_cond(
                global_cond, batch_ids, nnum)

            # OctFormer 采样（展平输入 + batch_ids + 跨深度 prefix）
            child_8way, cond_out = self.generator.sample(
                parent_xyz, parent_cond, batch_ids, temperature,
                prefix_tokens=prefix_tokens, prefix_xyz=prefix_xyz,
            )

            # 任一子节点被占据 → 分裂父节点
            split_label = child_8way.any(dim=-1).long()
            octree.octree_split(split_label, depth=depth)
            octree.octree_grow(depth + 1)

            # cond_out + parent_xyz 传给下一层
            return self.next_fractal._generate_level(
                octree, global_cond, temperature, cfg_scale, uncond,
                prefix_tokens=cond_out, prefix_xyz=parent_xyz,
            )

        else:
            # --- 终端层: 在 depth_stop 处采样 VQ 编码 ---
            final_depth = self.config.depth_stop
            leaf_xyz, leaf_batch_ids = get_node_xyz(octree, final_depth)
            nnum_leaf = octree.nnum[final_depth]
            if nnum_leaf == 0:
                return octree, None

            leaf_cond = self._make_per_node_cond(
                global_cond, leaf_batch_ids, nnum_leaf)

            vq_indices = self.next_fractal.sample(
                leaf_xyz, leaf_cond, temperature, leaf_batch_ids,
                prefix_tokens=prefix_tokens, prefix_xyz=prefix_xyz)
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
