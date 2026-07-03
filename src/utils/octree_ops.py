"""
八叉树操作工具函数。

从 OctGPT 的 utils/utils.py 移植关键工具，尽量减少对 ocnn
内部 API 的依赖。ocnn 仅用于 Octree 数据结构；
张量操作均在此完成。

关键函数:
- octree2seq / seq2octree: 八叉树 children 与展平二值序列的互转
- get_node_xyz: 提取指定深度的节点 (x, y, z) 坐标
- morton_encode_3d: 计算 Morton (Z-order) 码用于空间排序
- child_xyz: 从父节点坐标计算子节点位置
- get_split_labels: 提取每父节点的 8-way 子节点占用标签
"""

import copy
from typing import List, Tuple

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Morton（Z-order）编码
# ---------------------------------------------------------------------------

def morton_encode_3d(x: torch.Tensor, y: torch.Tensor,
                     z: torch.Tensor) -> torch.Tensor:
    """为 3D 坐标计算 Morton (Z-order) 码。

    交错 x、y、z 的二进制位，产生保持空间局部性的 1D 排序。
    使用经典的位交错方法。

    参数:
        x, y, z: 同形状的整数张量，值在 [0, 2^depth) 范围内

    返回:
        Morton 码，int64 张量
    """
    def spread_bits(v):
        """将 v 的二进制位按因子 3 扩散。"""
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    x = x.long()
    y = y.long()
    z = z.long()

    return spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)


def morton_order_indices(xyz: torch.Tensor) -> torch.Tensor:
    """返回按 Morton 序排序的索引。

    参数:
        xyz: (N, 3) 整数坐标

    返回:
        indices: (N,) 排序后的索引张量
    """
    codes = morton_encode_3d(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    return torch.argsort(codes)


# ---------------------------------------------------------------------------
# 子节点坐标计算
# ---------------------------------------------------------------------------

def child_xyz(parent_xyz: torch.Tensor) -> torch.Tensor:
    """计算每个父节点的 8 个子节点坐标。

    八叉树中，深度 d 的每个节点在深度 d+1 有 8 个潜在子节点。
    子节点索引 0-7，其中 bit 0 = +x, bit 1 = +y, bit 2 = +z。

    子节点映射（cx, cy, cz），其中 cx = (index >> 0) & 1 等:
        0: (0,0,0)  1: (1,0,0)  2: (0,1,0)  3: (1,1,0)
        4: (0,0,1)  5: (1,0,1)  6: (0,1,1)  7: (1,1,1)

    参数:
        parent_xyz: (*, 3) 深度 d 的父节点坐标

    返回:
        children_xyz: (*, 8, 3) 深度 d+1 的子节点坐标
    """
    *batch_dims, _ = parent_xyz.shape
    device = parent_xyz.device

    # 8 个八分圆的子节点偏移: (8, 3)
    offsets = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], device=device, dtype=parent_xyz.dtype)

    # 父节点坐标加倍（深度 d 的每单位 = 深度 d+1 的 2 单位）
    parent_doubled = (parent_xyz * 2).unsqueeze(-2)  # (*, 1, 3)
    children = parent_doubled + offsets.view(
        *([1] * len(batch_dims)), 8, 3)
    return children


# ---------------------------------------------------------------------------
# 节点坐标提取
# ---------------------------------------------------------------------------

def get_node_xyz(octree, depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """获取指定深度的所有节点的 (x, y, z) 整数坐标。

    使用 ocnn 的 octree.xyzb()，返回 (x, y, z, batch_idx)。

    参数:
        octree: ocnn.Octree 对象
        depth: 查询的深度层级

    返回:
        xyz: (nnum, 3) 整数坐标张量，值在 [0, 2^depth) 范围内
        batch_idx: (nnum,) batch 索引张量
    """
    x, y, z, b = octree.xyzb(depth, nempty=False)
    xyz = torch.stack([x, y, z], dim=1).long()
    return xyz, b.long()


# ---------------------------------------------------------------------------
# octree2seq / seq2octree（从 OctGPT utils/utils.py 移植）
# ---------------------------------------------------------------------------

def octree2seq(octree, depth_low: int, depth_high: int,
               shift: bool = False) -> torch.Tensor:
    """从八叉树提取 ground-truth split 标签，展平为一维序列。

    深度 d 的每个节点有 children[d]，记录其第一个子节点的索引
    （无子节点则为 -1）。child[i] >= 0 表示子节点存在。

    参数:
        octree: ocnn.Octree 对象
        depth_low: 起始深度（含）
        depth_high: 结束深度（不含）
        shift: 若为 True，将标签从 {0,1} 缩放到 {-1,1}

    返回:
        seq: (total_nnum,) long 张量，二值 split 标签
    """
    seq = torch.cat([octree.children[d] for d in range(depth_low, depth_high)])
    seq = (seq >= 0).long()

    if shift:
        seq = 2 * seq - 1
    return seq


def seq2octree(octree, seq: torch.Tensor, depth_low: int, depth_high: int,
               threshold: float = 0.0):
    """从预测的 split 序列重建八叉树。

    序列 `seq` 包含 [depth_low, depth_high) 范围内所有深度
    的逐节点二值 split 标签。每个值指示对应节点是否应分裂
    （即有子节点）。

    octree_split(label, depth) 中 label=1 创建全部 8 个子节点；
    label=0 使该节点成为叶子。

    参数:
        octree: 起始 ocnn.Octree（浅拷贝，原地修改）
        seq: (total_nnum,) float 张量，split logits/概率
        depth_low: 起始深度（含）
        depth_high: 结束深度（不含）
        threshold: 值 > threshold → 分裂

    返回:
        octree_out: 新八叉树，结构到 depth_high
    """
    discrete_seq = (seq > threshold).long()

    octree_out = copy.deepcopy(octree)
    cur_nnum = 0
    for d in range(depth_low, depth_high):
        nnum_d = octree_out.nnum[d]
        label = discrete_seq[cur_nnum:cur_nnum + nnum_d].clone()
        cur_nnum += nnum_d
        if label.numel() == 0:
            label = torch.zeros(8, dtype=torch.long, device=octree.device)
        octree_out.octree_split(label, depth=d)
        octree_out.octree_grow(d + 1)
    return octree_out


# ---------------------------------------------------------------------------
# 逐深度 split 标签提取（用于递归模型的 target）
# ---------------------------------------------------------------------------

def get_split_labels(octree, depth: int) -> torch.Tensor:
    """获取单深度的逐父节点 8-way split 标签。

    将深度 d+1 的每个子节点映射到深度 d 的父节点，
    并确定其占据的八分圆。

    参数:
        octree: ocnn.Octree
        depth: 父节点深度 d

    返回:
        labels: (nnum_d, 8) float32 张量，labels[p, c] = 1.0
                表示父节点 p 的八分圆 c 在深度 d+1 处存在
    """
    nnum_d = octree.nnum[depth]
    device = octree.device
    labels = torch.zeros(nnum_d, 8, dtype=torch.float32, device=device)

    if depth + 1 > octree.depth:
        return labels

    nnum_next = octree.nnum[depth + 1]
    if nnum_next == 0:
        return labels

    # 获取两个深度的坐标
    cx, cy, cz, cb = octree.xyzb(depth + 1, nempty=False)
    px, py, pz, pb = octree.xyzb(depth, nempty=False)

    # 子节点 → 父节点 和 octant
    child_parent_x = cx // 2
    child_parent_y = cy // 2
    child_parent_z = cz // 2
    octant = (cx & 1) + 2 * (cy & 1) + 4 * (cz & 1)  # 0–7

    # ---- 向量化匹配: 1D 哈希键 + searchsorted ----
    S = 2 ** depth  # 坐标范围 [0, S-1]

    # 父节点键: (batch_id, x, y, z) → 1D integer
    p_key = (((pb.long() * S + px.long()) * S + py.long()) * S + pz.long())

    # 子节点键: (batch_id, x//2, y//2, z//2) → 对应父节点键
    c_key = (((cb.long() * S + child_parent_x.long()) * S
              + child_parent_y.long()) * S + child_parent_z.long())

    # 排序父键 → searchsorted 匹配
    p_sorted, p_idx = p_key.sort()          # (nnum_d,)
    c_sorted, c_idx = c_key.sort()           # (nnum_next,)

    insert = torch.searchsorted(p_sorted, c_sorted)  # (nnum_next,)
    valid = (insert < nnum_d)

    if valid.any():
        # 只处理位置合法的项
        v_child_sorted = c_sorted[valid]
        v_insert = insert[valid]
        match = (p_sorted[v_insert] == v_child_sorted)  # (num_valid,)

        if match.any():
            # 映射回原始索引
            orig_child = c_idx[valid][match]
            orig_parent = p_idx[v_insert[match]]
            labels[orig_parent, octant[orig_child].long()] = 1.0

    return labels


# ---------------------------------------------------------------------------
# 辅助函数: 节点排序
# ---------------------------------------------------------------------------

def get_morton_ordered_nodes(octree, depth: int
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """获取按 Morton (Z-order) 码排序的节点。

    参数:
        octree: ocnn.Octree
        depth: 查询的深度

    返回:
        xyz_sorted: (nnum, 3) Morton 序排列的坐标
        sort_idx: (nnum,) 重排到 Morton 序的索引
    """
    xyz, batch = get_node_xyz(octree, depth)
    sort_idx = morton_order_indices(xyz)
    return xyz[sort_idx], sort_idx
