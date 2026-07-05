"""Coarse-Fine Grouped FractalOctGPT.

将原逐层 FractalOctGPT (d3->d4->d5->VQ) 改造为两组结构:

    Coarse group: [d3_split, d4_split]   (联合建模, 共享一个 OctFormer)
    Fine group:   [d5_split, d6_vq]      (联合建模, 共享另一个 OctFormer)
                   条件输入: coarse 输出的 full prefix hidden states

核心设计原则（对齐用户方案）:
  - 不重写 OctFormer / OctreeT / MaskGIT，最大程度复用 OctGPT 机制。
  - Coarse 和 Fine 各为一个独立的多深度 OctGPT 风格 layer（参考
    extern/octgpt/models/octgpt.py 的多深度拼接 forward）。
  - Coarse forward 返回最后一层 decoder hidden states (d3+d4 所有 token)。
  - 通过 Linear(coarse_dim, fine_dim) 投影后作为 Fine 的 prefix。
  - Fine 用 depth_list=[3,4,5,6] 构建 OctreeT，但在 depth 3/4 位置
    直接使用 prefix hidden states 作为 token embedding（替换 Fine 自身
    的 split_emb 输出）。OctreeT 原生 teacher-forcing mask 天然满足:
      * fine(d5/d6) 可 attend coarse(d3/d4) prefix  ✓
      * coarse(d3/d4) 不能 attend fine(d5/d6)      ✓ (teacher-forcing: 浅不能看深)
      * 同 batch 隔离由 OctreeT batch mask 保证    ✓
  - Fine loss 只在 d5/d6 计算，d3/d4 prefix 位置 loss_mask=False。
  - detach_prefix=False (默认): fine loss 可通过 prefix 反传更新 coarse,
    使 coarse hidden 主动学习服务于 fine 生成。

生成流程:
  1. coarse.sample(octree, depth 3..4) -> 生成 d3,d4 split, octree 扩展到 d4
  2. coarse.forward_tokens(GT gen tokens, return_hidden) -> prefix hidden
  3. fine.sample(octree, depth 5..6, prefix=prefix) -> 生成 d5 split + d6 VQ
  4. VQVAE decode -> mesh

接口契约:
  - forward(octree, labels=None) -> scalar loss
  - generate(octree, labels, temperature, cfg_scale) -> (octree, vq_indices)
  - model.config -> ModelConfig
"""

import copy
import math
from typing import Optional, Tuple

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

import ocnn
from ocnn.octree import Octree

import sys as _sys
_octgpt_root = __file__.replace('src/model/grouped_fractal_octgpt.py',
                                'extern/octgpt')
if _octgpt_root not in _sys.path:
    _sys.path.insert(0, _octgpt_root)

from models.octformer import OctFormer, OctreeT
from models.positional_embedding import RMSNorm, SinPosEmb
from utils.utils import (
    seq2octree, sample, depth2batch, batch2depth,
    get_batch_id, get_depth2batch_indices,
)

from src.utils.octree_ops import get_split_labels
from src.utils.train_metrics import compute_split_metrics, compute_vq_metrics


# ---------------------------------------------------------------------------
# 多深度 OctGPT 风格层 (支持 return_hidden + prefix 注入)
# ---------------------------------------------------------------------------


class GroupedOctGPTLayer(nn.Module):
    """多深度 OctGPT 风格层。

    与 extern/octgpt/models/octgpt.py 的 OctGPT.forward / generate 对齐,
    但只处理一个连续深度区间 [depth_low, depth_high], 且:
      - 最后一层为 split 层时: depth_high 的 token 也是 split (0/1)
      - 最后一层为 VQ 层时: depth_high 的 token 是 BSQ VQ indices

    支持:
      * return_hidden: 返回 decoder 最后一层 hidden states (含 buffer 去除)
      * prefix_emb / prefix_nnum: 在 depth_low 之前的浅深度位置注入外部
        hidden states 作为已知 context（不参与 loss, 不参与 token embedding）。
        OctreeT 的 depth_list 包含 prefix 对应的浅深度, teacher-forcing
        mask 天然保证 fine 可 attend prefix, prefix 不能 attend fine。

    参数:
        num_embed, num_heads, num_blocks: 同 OctGPT
        patch_size, dilation, buffer_size: 同 OctGPT
        is_vq_last: True=最后一深度是 VQ 层; False=全部为 split 层
        vq_groups, num_vq_embed: VQ 配置 (仅 is_vq_last=True)
        num_iters, start_temperature: per-depth MaskGIT 采样参数
        remask_stage, random_flip: 同 OctGPT
        drop_rate, attn_drop, proj_drop: dropout
        use_swin, use_checkpoint, pos_emb_type: 同 OctGPT
    """

    def __init__(
        self,
        num_embed: int = 576,
        num_heads: int = 8,
        num_blocks: int = 12,
        patch_size: int = 1024,
        dilation: int = 4,
        buffer_size: int = 64,
        is_vq_last: bool = False,
        vq_groups: int = 32,
        num_vq_embed: int = 32,
        num_iters=(64, 128),
        start_temperature=(1.0, 0.5),
        remask_stage: float = 0.7,
        random_flip: float = 0.1,
        drop_rate: float = 0.1,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        use_swin: bool = True,
        use_checkpoint: bool = False,
        pos_emb_type: str = "sin",
    ):
        super().__init__()
        assert num_blocks % 2 == 0, "num_blocks 必须为偶数"
        assert (num_embed // num_heads) % 6 == 0, \
            f"head_dim={num_embed//num_heads} 必须被 6 整除 (RoPE 约束), " \
            f"num_embed={num_embed}, num_heads={num_heads}"
        self.num_embed = num_embed
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.dilation = dilation
        self.buffer_size = buffer_size
        self.is_vq_last = is_vq_last
        self.vq_groups = vq_groups
        self.num_vq_embed = num_vq_embed
        self.num_iters = list(num_iters)
        self.start_temperature = list(start_temperature)
        self.remask_stage = remask_stage
        self.random_flip = random_flip
        self.use_swin = use_swin
        self.use_checkpoint = use_checkpoint

        if pos_emb_type == "abs":
            PosEmb = __import__('models.positional_embedding',
                                fromlist=['AbsPosEmb']).AbsPosEmb
        else:
            PosEmb = SinPosEmb
        self.pos_emb_type = pos_emb_type

        # 输出头 / embedding
        self.split_size = 2
        self.split_emb = nn.Embedding(self.split_size, num_embed)
        self.split_head = nn.Linear(num_embed, self.split_size)
        if is_vq_last:
            self.vq_size = 2
            self.vq_head = nn.Linear(num_embed, self.vq_size * vq_groups)
            self.vq_proj = nn.Linear(num_vq_embed, num_embed)
        else:
            self.vq_head = None
            self.vq_proj = None

        self.norm = RMSNorm(num_embed)
        self.mask_token = nn.Parameter(torch.zeros(1, num_embed))
        nn.init.normal_(self.mask_token, std=0.02)

        # Encoder / Decoder (对半分)
        self.encoder = OctFormer(
            channels=num_embed, num_heads=num_heads,
            num_blocks=num_blocks // 2, patch_size=patch_size,
            dilation=dilation, nempty=False,
            use_checkpoint=use_checkpoint, use_swin=use_swin,
            use_ctx=False, pos_emb=PosEmb, norm_layer=RMSNorm,
            attn_drop=attn_drop, proj_drop=proj_drop, dropout=drop_rate,
        )
        self.encoder_ln = RMSNorm(num_embed)
        self.decoder = OctFormer(
            channels=num_embed, num_heads=num_heads,
            num_blocks=num_blocks // 2, patch_size=patch_size,
            dilation=dilation, nempty=False,
            use_checkpoint=use_checkpoint, use_swin=use_swin,
            use_ctx=False, pos_emb=PosEmb, norm_layer=RMSNorm,
            attn_drop=attn_drop, proj_drop=proj_drop, dropout=drop_rate,
        )
        self.decoder_ln = RMSNorm(num_embed)

        self.mask_ratio_generator = stats.truncnorm(
            (0.5 - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        self.apply(self._init_weights)
        self._init_weights(self.mask_token)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data.fill_(1.0)

    # ------------------------------------------------------------------
    # MaskGIT 工具 (移植自 OctGPT)
    # ------------------------------------------------------------------

    def get_mask(self, seq_len, orders, mask_rate=None):
        if mask_rate is None:
            mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked = max(int(np.ceil(seq_len * mask_rate)), 1)
        return self.mask_by_order(num_masked, orders)

    def mask_by_order(self, mask_len, orders):
        mask = torch.zeros(orders.shape[0], device=orders.device).long()
        mask[orders[:mask_len]] = 1
        return mask

    def random_masking(self, x, mask, cond, octree=None, depth_list=None):
        """用 mask token 替换 masked 位置.

        对齐官方 OctGPT: none/category 条件下, mask token = cond[batch_id]
        (即 class_emb 按 batch 展开). 这样 mask token 与 buffer 用同一信号.
        若 batch_id 长度与 x 不匹配 (如 sample 中只处理单 depth), 回退到 mask_token.
        """
        if octree is not None and depth_list is not None:
            batch_id = get_batch_id(octree, depth_list)
            if batch_id.shape[0] == x.shape[0]:
                mask_tokens = cond[batch_id]
            else:
                # 长度不匹配 (sample 中 x 只是单 depth 的 token_emb_d)
                mask_tokens = self.mask_token.repeat(x.shape[0], 1)
        else:
            mask_tokens = self.mask_token.repeat(x.shape[0], 1)
        x = torch.where(mask.bool().unsqueeze(1), mask_tokens, x)
        return x

    def add_buffer(self, x, mask, cond):
        """序列前加条件 buffer (B*buffer_size 个 token)."""
        batch_size = cond.shape[0]
        buffer = cond.reshape(batch_size, 1, -1)
        buffer = buffer.repeat(1, self.buffer_size, 1).reshape(-1, self.num_embed)
        mask_buffer = torch.zeros(buffer.shape[0], device=x.device).bool()
        x = torch.cat([buffer, x], dim=0)
        mask = torch.cat([mask_buffer, mask], dim=0)
        return x, mask

    def forward_blocks(self, x, octree_t, blocks):
        x = depth2batch(x, octree_t.indices)
        x = blocks(x, octree_t, None)
        x = batch2depth(x, octree_t.indices)
        return x

    def forward_model(self, x, octree, depth_list, mask, cond):
        """OctFormer encoder-decoder (teacher forcing).

        depth_list: 该 OctreeT 覆盖的深度列表 (如 [3,4] 或 [3,4,5,6]).
        mask: (nnum_total,) bool, True=该位置被 mask (需预测).
              prefix 位置应为 False (已知 context, 不 mask).
        """
        depth_list = list(depth_list)

        # Encoder: 仅 unmasked token
        x_enc = x.clone()
        x_enc = x_enc[~mask]
        octree_t_enc = OctreeT(
            octree, x_enc.shape[0], self.patch_size, self.dilation,
            nempty=False, depth_list=depth_list, data_mask=mask,
            buffer_size=self.buffer_size, use_swin=self.use_swin)
        x_enc = self.forward_blocks(x_enc, octree_t_enc, self.encoder)
        x_enc = self.encoder_ln(x_enc)
        x[~mask] = x_enc

        # Decoder: 全部 token
        octree_t_dec = OctreeT(
            octree, x.shape[0], self.patch_size, self.dilation,
            nempty=False, depth_list=depth_list,
            buffer_size=self.buffer_size, use_swin=self.use_swin)
        x = self.forward_blocks(x, octree_t_dec, self.decoder)
        x = self.decoder_ln(x)
        return x

    # ------------------------------------------------------------------
    # 训练前向 (多深度, 对齐 OctGPT.forward)
    # ------------------------------------------------------------------

    def forward(self, octree, depth_low, depth_high, cond,
                targets_split=None, targets_vq=None, vqvae=None,
                prefix_emb=None, prefix_depths=None,
                return_hidden=False):
        """MaskGIT 训练前向, 处理 [depth_low, depth_high] 区间.

        参数:
            octree: GT octree (结构已到 depth_high, VQ 层到 depth_high)
            depth_low, depth_high: 该 layer 预测的深度区间 (含两端)
            cond: (B, num_embed) 全局条件 (类别嵌入)
            targets_split: dict {depth: (nnum_d,) long split 标签}
                           (VQ 之前的所有预测深度)
            targets_vq: (nnum_dh, vq_groups) long BSQ indices (仅 is_vq_last)
            vqvae: VQ 层用于 extract_code
            prefix_emb: (nnum_prefix, num_embed) 外部 prefix hidden states.
                        注入到 prefix_depths 对应的浅深度 token 位置,
                        替换 fine 模型自身的 split_emb 输出. None=无 prefix.
            prefix_depths: list[int] prefix 覆盖的深度 (如 [3,4]).
                           这些深度的 token 用 prefix_emb, 不参与 loss,
                           不被 mask (作为已知 context). 必须满足:
                           sum(octree.nnum[d] for d in prefix_depths)
                           == prefix_emb.shape[0].
            return_hidden: True=返回 decoder 最后一层 hidden states.

        返回:
            loss: 标量
            diag: dict (acc 等)
            hidden: (nnum_total_no_buffer, num_embed) 或 None
                    nnum_total_no_buffer = sum(nnum[d] for d in full_depth_list)
                    (含 prefix 位置, 不含 buffer)

        设计要点 (对齐用户方案):
          - 不额外 concat prefix token, 而是在 fine 模型原生 depth 序列
            的浅深度位置替换 embedding 为 coarse hidden.
          - full_depth_list = prefix_depths + depth_list (如 [3,4,5,6]),
            保持原生 OctGPT 序列结构.
          - OctreeT 原生 teacher-forcing mask 天然满足:
              * fine(d5/d6) 可 attend prefix(d3/d4)  ✓
              * prefix(d3/d4) 不能 attend fine(d5/d6)  ✓
              * 同 batch 隔离由 OctreeT batch mask 保证  ✓
          - prefix 位置 loss_mask=False, 不计算 loss.
        """
        device = octree.device
        depth_list = list(range(depth_low, depth_high + 1))
        prefix_depths = list(prefix_depths) if prefix_depths else []
        full_depth_list = prefix_depths + depth_list

        # 验证 prefix 长度对齐
        if prefix_emb is not None:
            nnum_prefix_expected = sum(octree.nnum[d] for d in prefix_depths)
            assert prefix_emb.shape[0] == nnum_prefix_expected, \
                f"prefix_emb 长度 {prefix_emb.shape[0]} != prefix_depths " \
                f"节点总数 {nnum_prefix_expected} (prefix_depths={prefix_depths})"

        # 构建 token embeddings (按 full_depth_list 顺序拼接)
        # prefix 深度: 用 prefix_emb; 预测深度: 用 split_emb / vq_proj
        token_embs = []
        nnum_per_depth = []
        prefix_offset = 0
        for d in full_depth_list:
            nnum_d = octree.nnum[d]
            nnum_per_depth.append(nnum_d)
            if d in prefix_depths and prefix_emb is not None:
                # prefix 位置: 用 coarse hidden (已投影到 num_embed)
                emb = prefix_emb[prefix_offset:prefix_offset + nnum_d]
                prefix_offset += nnum_d
            elif d < depth_high or not self.is_vq_last:
                # split 层 (预测深度)
                split = targets_split[d].long().clone()
                if self.random_flip > 0.0 and self.training:
                    flip = torch.rand_like(split.float()) < self.random_flip
                    split = torch.where(flip, 1 - split, split)
                emb = self.split_emb(split)  # (nnum_d, C)
            else:
                # VQ 层 (depth == depth_high, is_vq_last=True)
                with torch.no_grad():
                    if vqvae is not None:
                        vq_code = vqvae.extract_code(octree)
                        zq, indices, _ = vqvae.quantizer(vq_code)
                        targets_vq = indices.long().clone()
                    else:
                        zq = vqvae.quantizer.extract_code(targets_vq)
                    if self.random_flip > 0.0 and self.training:
                        flip = torch.rand_like(indices.float()) < self.random_flip
                        indices_flip = torch.where(flip, 1 - indices, indices)
                        zq = vqvae.quantizer.extract_code(indices_flip)
                emb = self.vq_proj(zq)  # (nnum_d, C)
            token_embs.append(emb)
        x_tokens = torch.cat(token_embs, dim=0)  # (nnum_total, C)
        seq_len = x_tokens.shape[0]

        # 构建 mask: prefix 位置 False (已知 context, 不 mask);
        #           预测位置 用 MaskGIT random masking.
        # 注意: masking 只在预测深度范围内随机 (对齐 OctGPT, 不 mask prefix).
        nnum_prefix = sum(nnum_per_depth[:len(prefix_depths)])
        nnum_pred = seq_len - nnum_prefix
        mask_prefix = torch.zeros(nnum_prefix, device=device).bool()
        mask_pred = self.get_mask(
            nnum_pred, torch.randperm(nnum_pred, device=device)).bool()
        mask_full = torch.cat([mask_prefix, mask_pred])

        # random_masking: 用 cond[batch_id] 替换 masked 位置 (对齐官方 OctGPT)
        x_tokens = self.random_masking(
            x_tokens, mask_full, cond, octree=octree, depth_list=full_depth_list)

        # add_buffer
        x, mask_full = self.add_buffer(x_tokens, mask_full, cond)

        # OctFormer encoder-decoder (用 full_depth_list, 保持原生序列结构)
        x_out = self.forward_model(x, octree, full_depth_list, mask_full, cond)
        # 去 buffer
        B = octree.batch_size
        x_node = x_out[B * self.buffer_size:]  # (nnum_total, C)

        # 计算 loss (只在预测深度, prefix 深度 loss_mask=False)
        # 对齐官方: split loss 是全 split masked token 拼接后一次 CE (reduction='mean'),
        #          而非 per-depth mean 相加. VQ loss 单独.
        diag = {}
        offset = nnum_prefix
        mask_pred_offset = 0  # 在 mask_pred 中的偏移

        # 先收集所有 split depth 的 masked logits/targets, VQ 单独处理
        split_logits_list = []
        split_targets_list = []
        split_mask_list = []
        vq_logits = None
        vq_targets = None
        vq_mask = None

        for i, d in enumerate(depth_list):
            idx_in_full = len(prefix_depths) + i
            nnum_d = nnum_per_depth[idx_in_full]
            x_d = x_node[offset:offset + nnum_d]
            mask_d = mask_pred[mask_pred_offset:mask_pred_offset + nnum_d]
            if d < depth_high or not self.is_vq_last:
                # split: 收集 masked logits/targets 用于统一 CE
                logits = self.split_head(x_d)
                targets_d = targets_split[d]
                split_logits_list.append(logits)
                split_targets_list.append(targets_d)
                split_mask_list.append(mask_d)
                with torch.no_grad():
                    if mask_d.any():
                        m = compute_split_metrics(logits[mask_d], targets_d[mask_d])
                        for k, v in m.items():
                            diag[f'd{d}_{k}'] = v
                    # per-depth loss 仅 logging
                    if mask_d.any():
                        diag[f'loss_d{d}'] = F.cross_entropy(
                            logits[mask_d], targets_d[mask_d]).item()
            else:
                # VQ
                vq_logits = self.vq_head(x_d)
                vq_targets = targets_vq
                vq_mask = mask_d
            offset += nnum_d
            mask_pred_offset += nnum_d

        # split loss: 全 split masked token 拼接后一次 CE (对齐官方)
        loss = torch.tensor(0.0, device=device)
        if split_logits_list:
            split_logits_all = torch.cat(split_logits_list, dim=0)
            split_targets_all = torch.cat(split_targets_list, dim=0)
            split_mask_all = torch.cat(split_mask_list, dim=0)
            if split_mask_all.any():
                loss_split = F.cross_entropy(
                    split_logits_all[split_mask_all],
                    split_targets_all[split_mask_all])
                loss = loss + loss_split
                diag['loss_split'] = loss_split.item()

        # VQ loss
        if vq_logits is not None and vq_mask is not None and vq_mask.any():
            if self.random_flip > 0.0 and self.training:
                loss_vq = F.cross_entropy(
                    vq_logits.reshape(-1, self.vq_size),
                    vq_targets.reshape(-1))
            else:
                loss_vq = F.cross_entropy(
                    vq_logits[vq_mask].reshape(-1, self.vq_size),
                    vq_targets[vq_mask].reshape(-1))
            loss = loss + loss_vq
            diag['loss_vq'] = loss_vq.item()
            with torch.no_grad():
                m = compute_vq_metrics(
                    vq_logits[vq_mask], vq_targets[vq_mask],
                    num_codes=self.vq_size)
                for k, v in m.items():
                    diag[f'vq_{k}'] = v

        hidden = x_node if return_hidden else None
        return loss, diag, hidden

    @staticmethod
    def _topk_acc(logits, targets, topk):
        topk = min(topk, logits.shape[-1] - 1)
        topk_idx = torch.topk(logits, topk, dim=-1).indices
        correct = topk_idx.eq(targets.unsqueeze(-1).expand_as(topk_idx))
        return correct.any(dim=-1).float().mean().item()

    def get_remask(self, logits, tokens, mask, remask_prob=0.2, topk=1):
        correct_topk = self._correct_topk(logits, tokens, topk)
        correct_by_group = correct_topk.any(dim=-1) if correct_topk.dim() > 1 else correct_topk
        remask = torch.zeros_like(mask).bool()
        if correct_by_group.dim() == 1:
            num_incorrect = (~correct_by_group).long()
            remask_scores = -logits[torch.arange(logits.shape[0]), tokens]
        else:
            num_incorrect = (~correct_by_group).sum(dim=-1)
            remask_scores = num_incorrect
        num_incorrect = num_incorrect.clone()
        num_incorrect[mask] = 0
        num_remask = int(num_incorrect.bool().sum() * remask_prob)
        if num_remask > 0:
            remask_indices = torch.topk(remask_scores, num_remask).indices
            remask[remask_indices] = True
        remask = remask & ~mask
        return remask

    @staticmethod
    def _correct_topk(logits, targets, topk):
        topk = min(topk, logits.shape[-1] - 1)
        topk_idx = torch.topk(logits, topk, dim=-1).indices
        return topk_idx.eq(targets.unsqueeze(-1).expand_as(topk_idx))

    # ------------------------------------------------------------------
    # 生成 (MaskGIT, 对齐 OctGPT.generate)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, octree, depth_low, depth_high, cond,
               num_iters=None, temperature=None, vqvae=None,
               prefix_emb=None, prefix_depths=None):
        """MaskGIT 迭代采样 [depth_low, depth_high].

        参数:
            octree: 初始 octree (结构到 depth_low, 即 nnum[depth_low] 存在)
            depth_low, depth_high: 预测的深度区间 (含两端)
            cond: (B, num_embed) 全局条件
            num_iters, temperature: per-depth MaskGIT 采样参数
            vqvae: VQ 层用于 extract_code
            prefix_emb: (nnum_prefix, num_embed) prefix hidden states.
                        None=无 prefix.
            prefix_depths: list[int] prefix 覆盖的深度 (如 [3,4]).
                           这些深度的 octree 节点必须已存在.

        返回:
            splits: dict {depth: (nnum_d,) long} (非 VQ 层)
            vq_indices: (nnum_dh, vq_groups) long 或 None (VQ 层)
            hidden: (nnum_total_no_buffer, C) 最后一次 forward 的 decoder 输出
                    (含 prefix 位置)

        设计要点 (对齐用户方案):
          - prefix 深度 (3,4) 的 token 用 prefix_emb, 作为已知 context.
          - 预测深度 (5,6) 用 MaskGIT 采样.
          - OctreeT 用 full_depth_list=[prefix_depths + depth_list],
            保持原生序列结构与 teacher-forcing mask.
          - 逐深度采样 (对齐 OctGPT.generate): 每个深度 d 的采样,
            OctreeT depth_list = prefix_depths + [depth_low..d].
        """
        device = octree.device
        depth_list = list(range(depth_low, depth_high + 1))
        num_iters = num_iters or self.num_iters
        temperature = temperature if temperature is not None else self.start_temperature
        prefix_depths = list(prefix_depths) if prefix_depths else []

        # prefix token embeddings (固定, 不采样)
        if prefix_emb is not None:
            nnum_prefix = prefix_emb.shape[0]
            token_embeddings = prefix_emb.clone()
            mask = torch.zeros(nnum_prefix, device=device).bool()
        else:
            nnum_prefix = 0
            token_embeddings = torch.empty((0, self.num_embed), device=device)
            mask = torch.empty((0,), device=device).bool()

        splits = {}
        vq_indices_final = None

        for di, d in enumerate(depth_list):
            nnum_d = octree.nnum[d]
            mask_d = torch.ones(nnum_d, device=device).bool()
            orders = torch.randperm(nnum_d, device=device)

            if d < depth_high or not self.is_vq_last:
                split_d = -torch.ones(nnum_d, device=device).long()
                token_emb_d = torch.zeros(nnum_d, self.num_embed, device=device)
            else:
                vq_indices_d = -torch.ones(
                    (nnum_d, self.vq_groups), device=device).long()
                token_emb_d = torch.zeros(nnum_d, self.num_embed, device=device)

            # 当前深度 d 的 OctreeT depth_list: prefix_depths + [depth_low..d]
            # (对齐 OctGPT.generate: 累积到当前深度)
            cur_depth_list = prefix_depths + list(range(depth_low, d + 1))

            # fully masked initial (对单 depth d 的 token, 用 [d] 取 batch_id)
            token_emb_d = self.random_masking(
                token_emb_d, mask_d, cond, octree=octree, depth_list=[d])

            n_iters_d = num_iters[di] if isinstance(num_iters, list) else num_iters
            temp_d = temperature[di] if isinstance(temperature, list) else temperature

            for i in range(n_iters_d):
                x = torch.cat([token_embeddings, token_emb_d], dim=0)
                mask_all = torch.cat([mask, mask_d])
                x, mask_all = self.add_buffer(x, mask_all, cond)
                x_out = self.forward_model(
                    x, octree, cur_depth_list, mask_all, cond)
                B = octree.batch_size
                x_node = x_out[B * self.buffer_size:]
                x_d = x_node[-nnum_d:]

                mask_ratio = np.cos(math.pi / 2. * (i + 1) / n_iters_d)
                mask_len = int(np.floor(nnum_d * mask_ratio))
                mask_len = max(1, min(int(mask_d.sum().item()) - 1, mask_len))
                mask_next = self.mask_by_order(mask_len, orders).bool()

                if i >= n_iters_d - 1:
                    mask_to_pred = mask_d.bool()
                else:
                    mask_to_pred = torch.logical_xor(mask_d.bool(), mask_next)
                mask_d = mask_next

                temp = temp_d * ((n_iters_d - i) / n_iters_d)

                if d < depth_high or not self.is_vq_last:
                    logits = self.split_head(x_d)
                    if i > n_iters_d * self.remask_stage:
                        remask = self.get_remask(
                            logits, split_d, mask_d, remask_prob=0.2)
                        mask_to_pred = mask_to_pred | remask
                    ix = sample(logits[mask_to_pred], temperature=temp)
                    split_d[mask_to_pred] = ix.long()
                    token_emb_d[mask_to_pred] = self.split_emb(ix)
                else:
                    logits = self.vq_head(x_d)
                    if i > n_iters_d * self.remask_stage:
                        logits_r = logits.reshape(-1, self.vq_groups, self.vq_size)
                        remask = self.get_remask(
                            logits_r, vq_indices_d, mask_d, topk=5, remask_prob=0.1)
                        mask_to_pred = mask_to_pred | remask
                    logits_p = logits[mask_to_pred].reshape(-1, self.vq_size)
                    ix = sample(logits_p, temperature=temp)
                    ix = ix.reshape(-1, self.vq_groups)
                    vq_indices_d[mask_to_pred] = ix.long()
                    if vqvae is not None:
                        zq = vqvae.quantizer.extract_code(ix)
                        token_emb_d[mask_to_pred] = self.vq_proj(zq).float()

            # 累积: 更新 token_embeddings, mask; 若 split 层则扩展 octree
            token_embeddings = torch.cat([token_embeddings, token_emb_d], dim=0)
            mask = torch.cat([mask, torch.zeros(nnum_d, device=device).bool()])

            if d < depth_high or not self.is_vq_last:
                splits[d] = split_d.long()
                # 扩展 octree 到 d+1
                octree = seq2octree(octree, split_d, d, d + 1)
            else:
                vq_indices_final = vq_indices_d

        # 最后一次 forward 获取 hidden
        # (对齐用户方案: 生成后用 gen tokens 再 forward 一次获得 prefix hidden)
        # full_depth_list 包含所有已生成深度
        final_depth_list = prefix_depths + depth_list
        mask_final = torch.zeros(token_embeddings.shape[0], device=device).bool()
        x, mask_final = self.add_buffer(token_embeddings, mask_final, cond)
        x_out = self.forward_model(
            x, octree, final_depth_list, mask_final, cond)
        B = octree.batch_size
        hidden = x_out[B * self.buffer_size:]  # (nnum_prefix + nnum_total, C)

        return splits, vq_indices_final, hidden


# ---------------------------------------------------------------------------
# CoarseFineOctGPT 顶层 wrapper
# ---------------------------------------------------------------------------


class CoarseFineOctGPT(nn.Module):
    """Coarse-Fine Grouped FractalOctGPT.

    结构:
        Coarse (GroupedOctGPTLayer): depth 3-4 split
            ↓ hidden states (d3+d4 所有 token) as prefix
        Fine (GroupedOctGPTLayer):   depth 5 split + depth 6 VQ
            conditioned on prefix

    参数:
        config: ModelConfig (含 coarse/fine 子配置或从 fractal_levels 推导)
        vqvae_wrapper: VQVAEWrapper
    """

    def __init__(self, config, vqvae_wrapper=None):
        super().__init__()
        self.config = config
        self.model_config = config
        self.vqvae_wrapper = vqvae_wrapper

        self.full_depth = config.full_depth  # 3
        self.depth_stop = config.depth_stop  # 6
        # coarse: depth 3-4, fine: depth 5 + VQ depth 6
        self.coarse_depths = (config.full_depth, config.full_depth + 1)  # (3,4)
        self.fine_depths = (config.full_depth + 2, config.depth_stop)    # (5,6)

        # 从 config 读取架构参数 (支持 coarse/fine 子段或回退到统一参数)
        def _get(cfg, key, idx, default):
            val = getattr(cfg, key, default)
            if isinstance(val, (list, tuple)):
                return val[idx] if idx < len(val) else val[-1]
            return val

        # Coarse 配置
        coarse_cfg = getattr(config, 'coarse', None)
        if coarse_cfg is not None:
            c_dim = coarse_cfg.get('dim', 576)
            c_heads = coarse_cfg.get('heads', 8)
            c_blocks = coarse_cfg.get('blocks', 12)
        else:
            c_dim = _get(config, 'embed_dims', 0, 576)
            c_heads = _get(config, 'num_heads', 0, 8)
            c_blocks = _get(config, 'num_blocks', 0, 12)

        # Fine 配置
        fine_cfg = getattr(config, 'fine', None)
        if fine_cfg is not None:
            f_dim = fine_cfg.get('dim', 768)
            f_heads = fine_cfg.get('heads', 8)
            f_blocks = fine_cfg.get('blocks', 12)
        else:
            f_dim = _get(config, 'embed_dims', 2, 768)
            f_heads = _get(config, 'num_heads', 2, 8)
            f_blocks = _get(config, 'num_blocks', 2, 12)

        use_swin = getattr(config, 'use_swin', True)
        pos_emb_type = getattr(config, 'pos_emb_type', 'sin')
        patch_size = getattr(config, 'patch_size', 1024)
        dilation = getattr(config, 'dilation', 4)
        buffer_size = getattr(config, 'buffer_size', 64)
        random_flip = getattr(config, 'random_flip', 0.1)
        remask_stage = getattr(config, 'remask_stage', 0.7)
        attn_drop = getattr(config, 'attn_drop', 0.1)
        proj_drop = getattr(config, 'proj_drop', 0.1)
        grad_ckpt = getattr(config, 'grad_checkpointing', False)

        num_iters = list(getattr(config, 'num_iters', (64, 128, 128, 256)))
        start_temp = list(getattr(config, 'start_temperature',
                                  (1.0, 1.2, 0.5, 0.5)))
        # coarse 用前 2 个, fine 用后 2 个
        coarse_num_iters = num_iters[:2] if len(num_iters) >= 2 else [64, 128]
        fine_num_iters = num_iters[2:4] if len(num_iters) >= 4 else [128, 256]
        coarse_temp = start_temp[:2] if len(start_temp) >= 2 else [1.0, 1.2]
        fine_temp = start_temp[2:4] if len(start_temp) >= 4 else [0.5, 0.5]

        # VQ 配置
        vq_groups = _get_vq_groups(vqvae_wrapper)
        num_vq_embed = _get_num_vq_embed(vqvae_wrapper)

        # Coarse layer (depth 3-4, 全 split)
        self.coarse = GroupedOctGPTLayer(
            num_embed=c_dim, num_heads=c_heads, num_blocks=c_blocks,
            patch_size=patch_size, dilation=dilation, buffer_size=buffer_size,
            is_vq_last=False,
            num_iters=coarse_num_iters, start_temperature=coarse_temp,
            remask_stage=remask_stage, random_flip=random_flip,
            drop_rate=proj_drop, attn_drop=attn_drop, proj_drop=proj_drop,
            use_swin=use_swin, use_checkpoint=grad_ckpt,
            pos_emb_type=pos_emb_type,
        )
        self.coarse_class_emb = nn.Embedding(
            max(getattr(config, 'num_classes', 1), 1), c_dim)

        # Fine layer (depth 5 split + depth 6 VQ)
        self.fine = GroupedOctGPTLayer(
            num_embed=f_dim, num_heads=f_heads, num_blocks=f_blocks,
            patch_size=patch_size, dilation=dilation, buffer_size=buffer_size,
            is_vq_last=True, vq_groups=vq_groups, num_vq_embed=num_vq_embed,
            num_iters=fine_num_iters, start_temperature=fine_temp,
            remask_stage=remask_stage, random_flip=random_flip,
            drop_rate=proj_drop, attn_drop=attn_drop, proj_drop=proj_drop,
            use_swin=use_swin, use_checkpoint=grad_ckpt,
            pos_emb_type=pos_emb_type,
        )
        self.fine_class_emb = nn.Embedding(
            max(getattr(config, 'num_classes', 1), 1), f_dim)

        # Prefix projection: coarse_dim -> fine_dim + LayerNorm (指南第4节)
        self.prefix_proj = nn.Linear(c_dim, f_dim, bias=False)
        self.prefix_norm = nn.LayerNorm(f_dim)
        self.detach_prefix = getattr(config, 'detach_prefix', False)

        # Plan B: finetune 配置
        ft_cfg = getattr(config, 'finetune', None) or {}
        self.ft_stage = ft_cfg.get('stage', 'none')  # none|adapter_only|partial_fine|full_fine_low_lr
        self.use_align_loss = ft_cfg.get('use_align_loss', False)
        self.align_loss_weight = ft_cfg.get('align_loss_weight', 0.1)
        if self.ft_stage != 'none':
            self.configure_finetune()

        # 诊断
        self._last_diag = {}

    # ------------------------------------------------------------------
    # 条件
    # ------------------------------------------------------------------

    def _get_class_cond(self, octree, labels, class_emb):
        B = octree.batch_size
        device = octree.device
        if labels is None:
            labels = torch.zeros(B, dtype=torch.long, device=device)
        if self.training and getattr(self.config, 'label_drop_prob', 0) > 0:
            drop = torch.rand(B, device=device) < self.config.label_drop_prob
            labels = torch.where(drop, torch.zeros_like(labels), labels)
        return class_emb(labels)

    def _apply_prefix_proj(self, coarse_hidden):
        """coarse hidden -> prefix embedding (proj + LayerNorm).

        对齐指南第4节: prefix = LayerNorm(Linear(coarse_hidden))
        """
        if self.detach_prefix:
            coarse_hidden = coarse_hidden.detach()
        return self.prefix_norm(self.prefix_proj(coarse_hidden))

    @staticmethod
    def _assert_prefix_alignment(octree, prefix_depths):
        """验证 prefix 与 fine 模型 depth 对齐 (指南 Milestone 3).

        coarse 和 fine 都基于同一 octree 的 nnum[d]/xyzb(d) 枚举节点,
        结构上对齐. 这里显式检查 prefix_depths 节点总数一致.
        """
        nnum_prefix = sum(octree.nnum[d] for d in prefix_depths)
        assert nnum_prefix > 0, f"prefix_depths {prefix_depths} 无节点"

    # ------------------------------------------------------------------
    # Plan B: finetune 配置
    # ------------------------------------------------------------------

    def configure_finetune(self):
        """根据 ft_stage 设置 requires_grad (指南第6节)."""
        ft_cfg = getattr(self.config, 'finetune', None) or {}
        stage = self.ft_stage

        if stage == 'adapter_only':
            # 冻结 fine, 训练 coarse + prefix
            self._set_requires_grad(self.fine, False)
            self._set_requires_grad(self.coarse, not ft_cfg.get('freeze_coarse', False))
            self._set_requires_grad(self.prefix_proj, True)
            self._set_requires_grad(self.prefix_norm, True)
        elif stage == 'partial_fine':
            # 冻结 fine 大部分, 解冻前几层 + norm
            self._set_requires_grad(self.fine, False)
            self._set_requires_grad(self.coarse, True)
            self._set_requires_grad(self.prefix_proj, True)
            self._set_requires_grad(self.prefix_norm, True)
            self._unfreeze_by_name(self.fine, ['norm', 'encoder_ln', 'decoder_ln',
                                               'blocks.0', 'blocks.1'])
        elif stage == 'full_fine_low_lr':
            self._set_requires_grad(self.fine, True)
            self._set_requires_grad(self.coarse, True)
            self._set_requires_grad(self.prefix_proj, True)
            self._set_requires_grad(self.prefix_norm, True)
        else:
            # none: 全部可训练
            pass

    @staticmethod
    def _set_requires_grad(module, flag):
        for p in module.parameters():
            p.requires_grad = flag

    @staticmethod
    def _unfreeze_by_name(module, patterns):
        for name, sub in module.named_modules():
            if any(p in name for p in patterns):
                for p in sub.parameters(recurse=False):
                    p.requires_grad = True

    def get_param_groups(self):
        """返回分组参数 (coarse/prefix/fine), 供 optimizer 使用不同 lr."""
        coarse_p = [p for p in self.coarse.parameters() if p.requires_grad]
        prefix_p = [p for p in self.prefix_proj.parameters() if p.requires_grad] + \
                   [p for p in self.prefix_norm.parameters() if p.requires_grad]
        fine_p = [p for p in self.fine.parameters() if p.requires_grad]
        groups = []
        if coarse_p:
            groups.append({'params': coarse_p, 'name': 'coarse'})
        if prefix_p:
            groups.append({'params': prefix_p, 'name': 'prefix'})
        if fine_p:
            groups.append({'params': fine_p, 'name': 'fine'})
        return groups

    def _compute_align_loss(self, prefix, octree, cond_f, prefix_depths):
        """Plan B align loss: prefix 应接近官方 fine 原生 depth 3/4 input embedding.

        关键: prefix 注入的是 OctFormer 输入的 token embedding 位置 (split_emb 输出),
        因此 align target 必须是 fine 原生的 input embedding (split_emb(split_targets)),
        而非 decoder 输出 hidden. 否则 input/hidden representation mismatch.

        原生 d3/d4 input embedding = fine.split_emb(split_targets) (未加 pos_emb,
        pos_emb 在 OctFormer 内部按 octree.xyz 计算).
        """
        with torch.no_grad():
            # 构建官方 fine 原生 d3/d4 的 input embedding
            ref_embs = []
            for d in prefix_depths:
                gt_split_8way = get_split_labels(octree, d)
                split = (gt_split_8way.sum(dim=-1) > 0).long()
                ref_embs.append(self.fine.split_emb(split))
            ref_emb_34 = torch.cat(ref_embs, dim=0).detach()
        # align: MSE + cosine
        loss_mse = F.mse_loss(prefix, ref_emb_34)
        cos = F.cosine_similarity(prefix, ref_emb_34, dim=-1).mean()
        loss_cos = 1.0 - cos
        return loss_mse + loss_cos, cos.item()

    # ------------------------------------------------------------------
    # 训练前向
    # ------------------------------------------------------------------

    def forward(self, octree, labels=None) -> torch.Tensor:
        """训练前向: coarse -> prefix -> fine, 返回总 loss (标量)."""
        device = octree.device

        # ---- Coarse ----
        cond_c = self._get_class_cond(octree, labels, self.coarse_class_emb)
        d_low_c, d_high_c = self.coarse_depths  # (3, 4)

        # coarse split targets
        targets_split_c = {}
        for d in range(d_low_c, d_high_c + 1):
            gt_split_8way = get_split_labels(octree, d)
            targets_split_c[d] = (gt_split_8way.sum(dim=-1) > 0).long()

        loss_c, diag_c, hidden_c = self.coarse(
            octree, d_low_c, d_high_c, cond_c,
            targets_split=targets_split_c, return_hidden=True)

        # ---- Prefix ----
        # hidden_c: (nnum_coarse_total, c_dim), 含 d3+d4 所有 token
        # nnum_coarse_total = nnum[3] + nnum[4]
        prefix_depths = list(self.coarse_depths)  # [3, 4]
        nnum_prefix = octree.nnum[d_low_c] + octree.nnum[d_high_c]
        # 位置对齐验证 (指南 Milestone 3)
        self._assert_prefix_alignment(octree, prefix_depths)
        assert hidden_c.shape[0] == nnum_prefix, \
            f"coarse hidden 长度 {hidden_c.shape[0]} != prefix_depths " \
            f"节点总数 {nnum_prefix}"
        prefix_emb = self._apply_prefix_proj(hidden_c)

        # ---- Fine ----
        cond_f = self._get_class_cond(octree, labels, self.fine_class_emb)
        d_low_f, d_high_f = self.fine_depths  # (5, 6)

        targets_split_f = {}
        for d in range(d_low_f, d_high_f):  # 5 (6 是 VQ)
            gt_split_8way = get_split_labels(octree, d)
            targets_split_f[d] = (gt_split_8way.sum(dim=-1) > 0).long()

        # VQ targets (由 fine layer 内部从 vqvae 提取)
        # prefix_depths=[3,4] 的 embedding 用 prefix_emb 替换, 不参与 loss
        loss_f, diag_f, hidden_f = self.fine(
            octree, d_low_f, d_high_f, cond_f,
            targets_split=targets_split_f, vqvae=self.vqvae_wrapper.vqvae,
            prefix_emb=prefix_emb, prefix_depths=prefix_depths,
            return_hidden=False)

        # ---- Plan B: align loss ----
        loss_align_val = 0.0
        prefix_ref_cos = 0.0
        if self.use_align_loss and self.training:
            loss_align, prefix_ref_cos = self._compute_align_loss(
                prefix_emb, octree, cond_f, prefix_depths)
            total_loss = loss_c + loss_f + self.align_loss_weight * loss_align
            loss_align_val = loss_align.item()
        else:
            total_loss = loss_c + loss_f

        # 汇总
        self._last_diag = {
            'loss_coarse': loss_c.item(),
            'loss_fine': loss_f.item(),
            # per-depth loss (指南第1.2节)
            'loss_d3_split': diag_c.get('loss_d3', 0.0),
            'loss_d4_split': diag_c.get('loss_d4', 0.0),
            'loss_d5_split': diag_f.get('loss_d5', 0.0),
            'loss_vq': diag_f.get('loss_vq', 0.0),
            # Plan B align
            'loss_align': loss_align_val,
            'prefix_ref_cosine': prefix_ref_cos,
            # coarse split 指标 (d3/d4)
            'd3_acc': diag_c.get('d3_acc', 0.0),
            'd3_pos_recall': diag_c.get('d3_pos_recall', 0.0),
            'd3_pos_f1': diag_c.get('d3_pos_f1', 0.0),
            'd3_target_pos_rate': diag_c.get('d3_target_pos_rate', 0.0),
            'd3_pred_pos_rate': diag_c.get('d3_pred_pos_rate', 0.0),
            'd4_acc': diag_c.get('d4_acc', 0.0),
            'd4_pos_recall': diag_c.get('d4_pos_recall', 0.0),
            'd4_pos_f1': diag_c.get('d4_pos_f1', 0.0),
            'd4_target_pos_rate': diag_c.get('d4_target_pos_rate', 0.0),
            'd4_pred_pos_rate': diag_c.get('d4_pred_pos_rate', 0.0),
            # fine split 指标 (d5)
            'd5_acc': diag_f.get('d5_acc', 0.0),
            'd5_pos_recall': diag_f.get('d5_pos_recall', 0.0),
            'd5_pos_f1': diag_f.get('d5_pos_f1', 0.0),
            'd5_target_pos_rate': diag_f.get('d5_target_pos_rate', 0.0),
            'd5_pred_pos_rate': diag_f.get('d5_pred_pos_rate', 0.0),
            # VQ 指标 (指南第3节)
            'vq_top1': diag_f.get('vq_top1', 0.0),
            'vq_top5': diag_f.get('vq_top5', 0.0),
            'vq_full_code_exact_rate': diag_f.get('vq_full_code_exact_rate', 0.0),
            'vq_hamming_per_node': diag_f.get('vq_hamming_per_node', 0.0),
            'vq_code_entropy': diag_f.get('vq_code_entropy', 0.0),
            'vq_unique_code_count': diag_f.get('vq_unique_code_count', 0),
            # 工程
            'prefix_len': nnum_prefix,
        }
        return total_loss

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, octree, labels=None, temperature=1.0, cfg_scale=1.0):
        """递归生成: coarse sample -> prefix -> fine sample -> VQ.

        返回:
            octree: 生成到 depth_stop 的 octree
            vq_indices: (nnum_leaf, vq_groups) BSQ indices
        """
        device = octree.device
        B = octree.batch_size

        # ---- Step 1: coarse sample (depth 3-4) ----
        cond_c = self._get_class_cond(octree, labels, self.coarse_class_emb)
        d_low_c, d_high_c = self.coarse_depths
        splits_c, _, hidden_c = self.coarse.sample(
            octree, d_low_c, d_high_c, cond_c,
            temperature=temperature, vqvae=None, prefix_emb=None)

        # 扩展 octree (coarse 生成的 split)
        for d in range(d_low_c, d_high_c + 1):
            split_d = splits_c[d]
            octree.octree_split(split_d, depth=d)
            octree.octree_grow(d + 1)

        # ---- Step 2: prefix from generated coarse tokens ----
        # hidden_c 已含 d3+d4 的 hidden states (来自生成后再次 forward)
        prefix_depths = list(self.coarse_depths)  # [3, 4]
        prefix_emb = self._apply_prefix_proj(hidden_c)
        # 注意: 此时 octree 已生长到 d_high_c+1 = 5, nnum[3]/nnum[4] 不变

        # ---- Step 3: fine sample (depth 5 split + depth 6 VQ) ----
        cond_f = self._get_class_cond(octree, labels, self.fine_class_emb)
        d_low_f, d_high_f = self.fine_depths
        splits_f, vq_indices, _ = self.fine.sample(
            octree, d_low_f, d_high_f, cond_f,
            temperature=temperature, vqvae=self.vqvae_wrapper.vqvae,
            prefix_emb=prefix_emb, prefix_depths=prefix_depths)

        # 应用 fine split (d5)
        for d in range(d_low_f, d_high_f):  # 5
            split_d = splits_f[d]
            octree.octree_split(split_d, depth=d)
            octree.octree_grow(d + 1)

        return octree, vq_indices

    def get_last_diag(self):
        return self._last_diag


# ---------------------------------------------------------------------------
# VQ 配置辅助
# ---------------------------------------------------------------------------


def _get_vq_groups(vqvae_wrapper):
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
    if vqvae_wrapper is None:
        return 32
    return getattr(vqvae_wrapper.vqvae.quantizer, 'embed_dim', 32)
