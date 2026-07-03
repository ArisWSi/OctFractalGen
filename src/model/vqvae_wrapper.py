"""
OctGPT 预训练 VQ-VAE 的轻量封装。

将 VQ-VAE 的编码/解码管线封装起来，使递归生成器
不需要了解 Octree CNN 的内部细节。

数据流:
  训练: octree → extract_code → quantize → BSQ indices（作为target）
  推理: 预测的 indices → extract_code → decode → Neural MPU → mesh

参数:
    vqvae: OctGPT 的 VQVAE nn.Module（预训练、冻结）
    depth_stop: VQ codes 所在的最终八叉树深度
    full_depth: 初始八叉树深度
    vae_depth: VQ-VAE 使用的最大八叉树深度（通常为 8）
"""

import copy
from typing import Tuple

import torch
import torch.nn as nn


class VQVAEWrapper:
    """对 OctGPT 预训练 VQ-VAE 的轻量封装。

    VQ-VAE 在 OctreeFractalGen 训练期间保持冻结——它仅作为
    target 提供者（编码）和 mesh 重建器（解码）。

    参数:
        vqvae: OctGPT 的 VQVAE nn.Module（预训练、冻结）
        depth_stop: VQ codes 所在的最终八叉树深度
        full_depth: 初始八叉树深度
        vae_depth: VQ-VAE 的完整深度（用于解码时的八叉树扩展）
    """

    def __init__(self, vqvae: nn.Module, depth_stop: int = 5,
                 full_depth: int = 3, vae_depth: int = 8):
        self.vqvae = vqvae
        self.depth_stop = depth_stop
        self.full_depth = full_depth
        self.vae_depth = vae_depth

        # 从量化器类型推导 VQ 配置
        quantizer = vqvae.quantizer
        if hasattr(quantizer, 'embed_dim'):
            # BSQ 量化器: D = embedding_channels
            self.vq_groups = quantizer.embed_dim
        elif hasattr(quantizer, 'groups'):
            # 分组量化器
            self.vq_groups = quantizer.groups
        else:
            self.vq_groups = 64  # 回退默认值

        self.vq_size = 2  # BSQ: 每组二值化

    @torch.no_grad()
    def extract_targets(self, octree) -> torch.Tensor:
        """为所有叶子节点提取 ground-truth BSQ indices。

        运行冻结的编码器 + BSQ 量化器，产生离散 indices，
        用作生成器的训练 target。

        参数:
            octree: ocnn.Octree（ground truth, depth ≥ depth_stop）

        返回:
            indices: (nnum_at_depth_stop, vq_groups) long 张量，
                     值域 {0, 1}（BSQ 每组 1 bit）
        """
        # 编码器前向
        vq_code = self.vqvae.extract_code(octree)
        # 量化
        zq, indices, _ = self.vqvae.quantizer(vq_code)
        return indices.long()

    @torch.no_grad()
    def decode_to_mpu(self, indices: torch.Tensor, octree):
        """将预测的 VQ indices 转换为 Neural MPU 可调用对象。

        返回的函数可以在任意 3D 查询位置求值，产生 SDF 值，
        供 Marching Cubes 使用。

        参数:
            indices: (nnum_at_depth_stop, vq_groups) 预测的 BSQ indices
            octree: ocnn.Octree，结构已生成到 depth_stop

        返回:
            neural_mpu: 可调用对象 f(positions) → SDF values
        """
        from ognn.octreed import OctreeD

        # indices → 连续编码
        zq = self.vqvae.quantizer.extract_code(indices)

        # 将八叉树扩展到 VQ-VAE 的完整深度
        # 遵循 OctGPT 的做法：从 depth_stop 开始逐层补零 split
        octree_out = copy.deepcopy(octree)
        for d in range(self.depth_stop, self.vae_depth):
            split_zero = torch.zeros(
                octree_out.nnum[d], device=octree_out.device).long()
            octree_out.octree_split(split_zero, d)
            octree_out.octree_grow(d + 1)

        # 遵循 OctGPT 的 export_results 模式：用扩展后的八叉树
        # 同时作为 octree_in 和 octree_out
        doctree_out = OctreeD(octree_out)

        # 解码
        output = self.vqvae.decode_code(
            zq, self.depth_stop, doctree_out,
            copy.deepcopy(doctree_out), update_octree=True)

        return output['neural_mpu']

    def get_vq_config(self) -> dict:
        """返回 VQ 配置，供生成器 head 使用。"""
        return {
            'vq_groups': self.vq_groups,
            'vq_size': self.vq_size,
        }
