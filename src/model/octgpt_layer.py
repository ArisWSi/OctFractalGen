"""OctGPT-style 单层模型：复用 OctFormer + OctreeT + MaskGIT，处理单一深度。

这是 FractalOctGPT 的构建块。每层独立处理一个深度：
  - split 层：预测该深度节点的 split 信号（0/1）
  - VQ 层：预测叶子节点的 BSQ 编码（depth_stop 处）

与 OctGPT 的关系：
  OctGPT 用一个共享模型处理 depth_low..depth_high 多个深度（token 拼接）。
  这里把单深度逻辑抽取为独立层，参数不跨深度共享，由 FractalOctGPT 串联。

复用的 OctGPT 机制（经 OctreeT 实现）：
  - patch attention mask（跨 batch 隔离）
  - teacher forcing mask（浅→深单向 attention）
  - SWIN 窗口移位
  - encoder-decoder 分离（unmasked→encoder，all→decoder）
  - MaskGIT random masking 训练 + cosine 采样
"""

import copy
import math

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

import ocnn
from ocnn.octree import Octree

import sys as _sys
_octgpt_root = __file__.replace('src/model/octgpt_layer.py', 'extern/octgpt')
if _octgpt_root not in _sys.path:
    _sys.path.insert(0, _octgpt_root)

from models.octformer import OctFormer, OctreeT
from models.positional_embedding import RMSNorm, SinPosEmb, AbsPosEmb
from utils.utils import (
    seq2octree, sample, depth2batch, batch2depth,
    get_batch_id,
)


class OctGPTLayer(nn.Module):
    """单深度 OctGPT 风格层（OctFormer + OctreeT + MaskGIT）。

    参数:
        num_embed: 嵌入维度
        num_heads: 注意力头数
        num_blocks: OctFormer block 总数（encoder=decoder=num_blocks//2）
        patch_size: patch 注意力大小
        dilation: 膨胀率
        buffer_size: 条件 buffer 大小
        is_vq: True=VQ 层（预测 BSQ），False=split 层（预测 0/1）
        vq_groups: BSQ 量化组数（仅 is_vq=True）
        num_vq_embed: VQ-VAE 编码维度（仅 is_vq=True，用于 vq_proj）
        num_iters: MaskGIT 采样迭代数
        start_temperature: 采样起始温度
        remask_stage: remask 起始比例
        random_flip: 训练时随机翻转 token 的概率（标签增强）
        drop_rate, attn_drop, proj_drop: dropout
        use_swin: SWIN 窗口移位
        use_checkpoint: 梯度检查点
    """

    def __init__(
        self,
        num_embed: int = 512,
        num_heads: int = 8,
        num_blocks: int = 16,
        patch_size: int = 1024,
        dilation: int = 4,
        buffer_size: int = 64,
        is_vq: bool = False,
        vq_groups: int = 32,
        num_vq_embed: int = 32,
        num_iters: int = 128,
        start_temperature: float = 1.0,
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
        assert num_blocks % 2 == 0, "num_blocks 必须为偶数（encoder/decoder 对半）"
        # RotaryPosEmb 要求 head_dim 被 6 整除（OctGPT RoPE 的 3 轴分配约束）
        # 调用方需保证 num_embed // num_heads % 6 == 0
        assert (num_embed // num_heads) % 6 == 0, \
            f"head_dim={num_embed//num_heads} 必须被 6 整除（RoPE 约束），" \
            f"num_embed={num_embed}, num_heads={num_heads}"
        self.num_embed = num_embed
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.dilation = dilation
        self.buffer_size = buffer_size
        self.is_vq = is_vq
        self.vq_groups = vq_groups
        self.num_vq_embed = num_vq_embed
        self.num_iters = num_iters
        self.start_temperature = start_temperature
        self.remask_stage = remask_stage
        self.random_flip = random_flip
        self.use_swin = use_swin
        self.use_checkpoint = use_checkpoint
        # 位置嵌入选择（abs 对 num_embed%6≠0 有越界 bug，默认用 sin）
        if pos_emb_type == "abs":
            PosEmb = AbsPosEmb
        else:
            PosEmb = SinPosEmb
        self.pos_emb_type = pos_emb_type

        # 输出头
        if is_vq:
            self.vq_size = 2
            self.vq_head = nn.Linear(num_embed, self.vq_size * vq_groups)
            self.vq_proj = nn.Linear(num_vq_embed, num_embed)
            self.split_size = None
        else:
            self.split_size = 2  # 0/1
            self.split_emb = nn.Embedding(self.split_size, num_embed)
            self.split_head = nn.Linear(num_embed, self.split_size)
            self.vq_proj = None

        # 条件嵌入（类别/无条件 → buffer）
        # FractalOctGPT 传入 global_cond (B, num_embed) 作为 buffer
        self.norm = RMSNorm(num_embed)

        # MaskGIT: mask token（无条件模式下用 cond 作为 mask token）
        self.mask_token = nn.Parameter(torch.zeros(1, num_embed))
        nn.init.normal_(self.mask_token, std=0.02)

        # CLS token: 作为 buffer 第一个位置，聚合序列信息供下一层使用
        self.cls_token = nn.Parameter(torch.zeros(1, num_embed))
        nn.init.normal_(self.cls_token, std=0.02)

        # Encoder / Decoder（对半分，OctGPT 风格）
        self.encoder = OctFormer(
            channels=num_embed, num_heads=num_heads,
            num_blocks=num_blocks // 2, patch_size=patch_size,
            dilation=dilation, nempty=False,
            use_checkpoint=use_checkpoint, use_swin=use_swin,
            use_ctx=False, pos_emb=PosEmb,
            norm_layer=RMSNorm,
            attn_drop=attn_drop, proj_drop=proj_drop, dropout=drop_rate,
        )
        self.encoder_ln = RMSNorm(num_embed)
        self.decoder = OctFormer(
            channels=num_embed, num_heads=num_heads,
            num_blocks=num_blocks // 2, patch_size=patch_size,
            dilation=dilation, nempty=False,
            use_checkpoint=use_checkpoint, use_swin=use_swin,
            use_ctx=False, pos_emb=PosEmb,
            norm_layer=RMSNorm,
            attn_drop=attn_drop, proj_drop=proj_drop, dropout=drop_rate,
        )
        self.decoder_ln = RMSNorm(num_embed)

        # Mask ratio 生成器（截断正态）
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
    # MaskGIT 工具（移植自 OctGPT）
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

    def random_masking(self, x, mask, cond):
        """用 mask_token 替换被 mask 的位置。"""
        mask_tokens = self.mask_token.repeat(x.shape[0], 1)
        x = torch.where(mask.bool().unsqueeze(1), mask_tokens, x)
        return x

    def add_buffer(self, x, mask, cond):
        """在序列前加条件 buffer（每 batch 项 buffer_size 个）。

        布局: [CLS(1) | cond_buffer(buf-1) | tokens(N)] × B
        CLS token 占每 batch buffer 的第一个位置，用于聚合序列信息。
        """
        batch_size = cond.shape[0]
        # CLS token (可学习) + cond 填充剩余 buffer
        cls = self.cls_token.expand(batch_size, 1, -1)  # (B, 1, dim)
        cond_buf = cond.unsqueeze(1).repeat(1, self.buffer_size - 1, 1)  # (B, buf-1, dim)
        buffer = torch.cat([cls, cond_buf], dim=1).reshape(-1, self.num_embed)  # (B*buf, dim)
        mask_buffer = torch.zeros(buffer.shape[0], device=x.device).bool()
        x = torch.cat([buffer, x], dim=0)
        mask = torch.cat([mask_buffer, mask], dim=0)
        return x, mask

    def forward_blocks(self, x, octree_t, blocks):
        x = depth2batch(x, octree_t.indices)
        x = blocks(x, octree_t, None)  # context=None（无条件）
        x = batch2depth(x, octree_t.indices)
        return x

    def forward_model(self, x, octree, depth, mask, cond):
        """OctFormer encoder-decoder（teacher forcing）。

        Encoder 只处理 unmasked token → 干净上下文。
        Decoder 处理全部 token（masked 用 mask_token）→ 预测。

        注意: x 和 mask 在调用前已包含 buffer（由 add_buffer 处理）。
        """
        batch_size = octree.batch_size
        depth_list = [depth]

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
    # 训练前向
    # ------------------------------------------------------------------

    def forward(self, octree, depth, cond, targets,
                prefix_tokens=None, vqvae=None):
        """MaskGIT 训练前向（完全对齐 OctGPT 流程）。

        跨深度信息通过 cond（buffer）传递，不用 prefix token。
        这避免了 OctreeT 的 data_mask 与 prefix 长度不匹配问题。

        参数:
            octree: ocnn.Octree（GT 结构已生成到 depth）
            depth: 当前层处理的深度
            cond: (B, num_embed) 全局条件（类别 + 跨深度信息）
            targets: split 层→(nnum,) long split 标签；
                     VQ 层→(nnum, vq_groups) long BSQ indices
            prefix_tokens: 忽略（保留接口兼容）
            vqvae: VQ 层用于 extract_code（训练时实时编码 GT）

        返回:
            loss: 标量
            cond_out: (nnum, num_embed) 该层输出特征（供下一层 cond）
            diag: dict 诊断信息（acc 等）
        """
        device = octree.device
        nnum = octree.nnum[depth]

        # 构建 token embeddings
        if self.is_vq:
            with torch.no_grad():
                if vqvae is not None:
                    vq_code = vqvae.extract_code(octree)
                    zq, indices, _ = vqvae.quantizer(vq_code)
                    targets = indices.long().clone()
                else:
                    zq = vqvae.quantizer.extract_code(targets)
                if self.random_flip > 0.0 and self.training:
                    flip = torch.rand_like(indices.float()) < self.random_flip
                    indices_flip = torch.where(flip, 1 - indices, indices)
                    zq = vqvae.quantizer.extract_code(indices_flip)
            token_emb = self.vq_proj(zq)  # (nnum, num_embed)
        else:
            split = targets.long().clone()
            if self.random_flip > 0.0 and self.training:
                flip = torch.rand_like(split.float()) < self.random_flip
                split = torch.where(flip, 1 - split, split)
            token_emb = self.split_emb(split)  # (nnum, num_embed)

        # Random masking（对齐 OctGPT: mask 长度 = nnum，不含 buffer）
        orders = torch.randperm(nnum, device=device)
        mask = self.get_mask(nnum, orders).bool()
        token_emb = self.random_masking(token_emb, mask, cond)

        # add_buffer: [buffer(B*buf) | token_emb(nnum)]
        x, mask = self.add_buffer(token_emb, mask, cond)

        # OctFormer encoder-decoder
        x_out = self.forward_model(x, octree, depth, mask, cond)
        # 提取 CLS token 输出（每 batch buffer 的第一个位置）→ 下一层 cond
        B = octree.batch_size
        cls_out = x_out[::self.buffer_size][:B]  # (B, dim)
        # 去 buffer → 当前深度节点输出
        x_node = x_out[B * self.buffer_size:]
        cond_out = cls_out  # CLS 聚合特征，供下一层 cond

        # 计算 loss（仅 masked 位置）
        diag = {}
        if self.is_vq:
            logits = self.vq_head(x_node)
            logits_m = logits[mask_node := mask[octree.batch_size * self.buffer_size:]].reshape(-1, self.vq_size)
            targets_m = targets[mask_node].reshape(-1)
            loss = F.cross_entropy(logits_m, targets_m)
            with torch.no_grad():
                top1 = self._topk_acc(logits[mask_node].reshape(-1, self.vq_size),
                                      targets[mask_node].reshape(-1), 1)
                diag['vq_top1'] = top1
        else:
            logits = self.split_head(x_node)
            mask_node = mask[octree.batch_size * self.buffer_size:]
            logits_m = logits[mask_node]
            targets_m = targets[mask_node]
            loss = F.cross_entropy(logits_m, targets_m)
            with torch.no_grad():
                pred = logits[mask_node].argmax(dim=-1)
                diag['split_acc'] = (pred == targets[mask_node]).float().mean().item()

        return loss, cond_out, diag

    @staticmethod
    def _topk_acc(logits, targets, topk):
        topk = min(topk, logits.shape[-1] - 1)
        topk_idx = torch.topk(logits, topk, dim=-1).indices
        correct = topk_idx.eq(targets.unsqueeze(-1).expand_as(topk_idx))
        return correct.any(dim=-1).float().mean().item()

    def get_remask(self, logits, tokens, mask, remask_prob=0.2, topk=1):
        """移植自 OctGPT.get_remask。"""
        correct_topk = self._correct_topk(logits, tokens, topk)
        correct_by_group = correct_topk.any(dim=-1) if correct_topk.dim() > 1 else correct_topk
        remask = torch.zeros_like(mask).bool()
        if correct_by_group.dim() == 1:  # split
            num_incorrect = (~correct_by_group).long()
            remask_scores = -logits[torch.arange(logits.shape[0]), tokens]
        else:  # vq
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
    # 生成（MaskGIT 采样）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, octree, depth, cond, num_iters=None,
               temperature=None, vqvae=None,
               prefix_tokens=None):
        """MaskGIT 迭代采样（对齐 OctGPT，无 prefix）。

        返回:
            若 split 层: (split_d (nnum,) long, cond_out (nnum, num_embed))
            若 VQ 层: (vq_indices (nnum, vq_groups) long, cond_out)
        """
        device = octree.device
        nnum = octree.nnum[depth]
        num_iters = num_iters or self.num_iters
        temperature = temperature if temperature is not None else self.start_temperature

        token_emb = torch.zeros(nnum, self.num_embed, device=device)
        mask_d = torch.ones(nnum, device=device).bool()
        orders = torch.randperm(nnum, device=device)

        if self.is_vq:
            split_d = None
            vq_indices_d = -torch.ones(
                (nnum, self.vq_groups), device=device).long()
        else:
            split_d = -torch.ones(nnum, device=device).long()
            vq_indices_d = None

        for i in range(num_iters):
            # random_masking + add_buffer
            token_emb_masked = self.random_masking(token_emb, mask_d, cond)
            x, mask_all = self.add_buffer(token_emb_masked, mask_d, cond)
            x_out = self.forward_model(x, octree, depth, mask_all, cond)
            B = octree.batch_size
            cls_out = x_out[::self.buffer_size][:B]  # CLS 输出
            x_node = x_out[B * self.buffer_size:]

            mask_ratio = np.cos(math.pi / 2. * (i + 1) / num_iters)
            mask_len = int(np.floor(nnum * mask_ratio))
            mask_len = max(1, min(int(mask_d.sum().item()) - 1, mask_len))
            mask_next = self.mask_by_order(mask_len, orders).bool()

            if i >= num_iters - 1:
                mask_to_pred = mask_d.bool()
            else:
                mask_to_pred = torch.logical_xor(mask_d.bool(), mask_next)
            mask_d = mask_next

            temp = temperature * ((num_iters - i) / num_iters)

            if self.is_vq:
                logits = self.vq_head(x_node)
                if i > num_iters * self.remask_stage:
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
                    token_emb[mask_to_pred] = self.vq_proj(zq).float()
            else:
                logits = self.split_head(x_node)
                if i > num_iters * self.remask_stage:
                    remask = self.get_remask(
                        logits, split_d, mask_d, remask_prob=0.2)
                    mask_to_pred = mask_to_pred | remask
                ix = sample(logits[mask_to_pred], temperature=temp)
                split_d[mask_to_pred] = ix.long()
                token_emb[mask_to_pred] = self.split_emb(ix)

        cond_out = cls_out  # CLS 聚合特征
        if self.is_vq:
            return vq_indices_d, cond_out
        else:
            return split_d, cond_out
