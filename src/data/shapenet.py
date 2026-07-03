"""
ShapeNet 数据加载，用于八叉树 occupancy 生成。

从 OctGPT 的 datasets/shapenet.py 适配。精简为仅包含
点云 → 八叉树路径（无 SDF 加载、无图像/文本条件）。

管线:
    1. ReadFile: 从样本目录加载 pointcloud.npz
    2. TransformShape: 点云 → ocnn.Octree（通过 build_octree）
    3. collate_func: 通过 ocnn.dataset.CollateBatch 批量合并八叉树

用法:
    from src.data.shapenet import get_shapenet_dataset
    dataset, collate_fn = get_shapenet_dataset(data_config)
"""

import os
from typing import Dict, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# ReadFile: 从磁盘加载预处理的点云
# ---------------------------------------------------------------------------

class ReadFile:
    """从样本目录加载 pointcloud.npz。

    预期的目录结构:
        {location}/{sample_id}/
            pointcloud.npz — 包含 keys 'points' (N,3) 和 'normals' (N,3)

    sample_id 从 filelist 条目的最后两个路径分量推导。
    """

    def __init__(self, flags):
        self.flags = flags

    def __call__(self, filename: str) -> Dict:
        """从磁盘加载点云。

        参数:
            filename: 样本目录路径（来自 filelist）

        返回:
            含 'point_cloud' → {'points': (N,3), 'normals': (N,3)} 的 dict
        """
        output = {}
        filename_pc = os.path.join(filename, 'pointcloud.npz')
        raw = np.load(filename_pc)
        output['point_cloud'] = {
            'points': raw['points'].astype(np.float32),
            'normals': raw['normals'].astype(np.float32),
        }
        return output


# ---------------------------------------------------------------------------
# TransformShape: 点云 → 八叉树
# ---------------------------------------------------------------------------

class TransformShape:
    """将点云转换为 ocnn Octree。

    八叉树由 ocnn 的 Octree.build_octree() 从表面点云构建。
    这为每个深度提供了 ground truth 八叉树结构（哪些体素被占据）。

    关键参数:
        depth: 最大八叉树深度（例如 6）
        full_depth: 全分辨率的最小深度（例如 3）
        points_scale: 输入点云在 [-points_scale, points_scale] 范围内，
                      重缩放至 [-1, 1] 供 ocnn 使用
    """

    def __init__(self, flags):
        self.flags = flags
        self.depth = flags.depth
        self.full_depth = flags.full_depth
        self.points_scale = getattr(flags, 'points_scale', 1.0)
        self.max_points = getattr(flags, 'max_points', 120000)

    def points2octree(self, points, normals=None):
        """从点云构建 ocnn Octree。

        参数:
            points: (N, 3) float 张量
            normals: (N, 3) float 张量（可选）

        返回:
            ocnn.Octree 对象
        """
        import ocnn
        from ocnn.octree import Octree, Points

        if normals is not None:
            pts = Points(points=points.cpu(), normals=normals.cpu())
        else:
            pts = Points(points=points.cpu(),
                         normals=torch.zeros_like(points.cpu()))

        pts.clip(min=-1, max=1)
        octree = Octree(self.depth, self.full_depth)
        octree.build_octree(pts)
        return octree

    def process_points_cloud(self, sample: Dict) -> Dict:
        """从点云样本构建八叉树。

        参数:
            sample: 含 'point_cloud' → {'points': array, 'normals': array} 的 dict

        返回:
            含 'octree_gt': ocnn.Octree 的 dict
        """
        # 加载点云并缩放至 [-1, 1]
        points_np = sample['points'].astype(np.float32)
        normals_np = sample['normals'].astype(np.float32)

        points = torch.from_numpy(points_np).float()
        normals = torch.from_numpy(normals_np).float()

        # 缩放点云至 [-1, 1]
        points = points / self.points_scale

        # 若超过 max_points 则随机丢弃（避免 OOM）
        if self.max_points and points.shape[0] > self.max_points:
            rand_idx = np.random.choice(
                points.shape[0], size=self.max_points, replace=False)
            points = points[rand_idx]
            normals = normals[rand_idx]

        # 构建八叉树
        octree_gt = self.points2octree(points, normals)

        return {
            'octree_gt': octree_gt,
            'points': points,
            'normals': normals,
        }

    def __call__(self, sample: Dict, idx: int) -> Dict:
        """完整变换: 点云 → 八叉树。

        参数:
            sample: ReadFile 的输出
            idx: 样本索引（未使用）

        返回:
            含 'octree_gt' 的 dict
        """
        output = {}
        output.update(self.process_points_cloud(sample['point_cloud']))
        return output


# ---------------------------------------------------------------------------
# Collate 函数
# ---------------------------------------------------------------------------

def collate_func(batch):
    """将样本列表合并为一个 batch。

    使用 ocnn 的 CollateBatch 进行正确的八叉树合并。
    """
    import ocnn
    return ocnn.dataset.CollateBatch(merge_points=False)(batch)


# ---------------------------------------------------------------------------
# Dataset 工厂
# ---------------------------------------------------------------------------

def get_shapenet_dataset(flags) -> Tuple:
    """创建 ShapeNet 数据集和 collate 函数。

    参数:
        flags: 类 DataConfig 对象，包含:
            - location: 数据集根路径
            - filelist: 列出样本目录的文本文件路径
            - depth, full_depth, points_scale, max_points

    返回:
        (dataset, collate_func) 元组
    """
    # 延迟导入以避免模块级别的硬依赖
    try:
        from thsolver import Dataset
    except ImportError:
        # 回退: 使用简单的基于列表的 dataset
        return _simple_dataset(flags), collate_func

    transform = TransformShape(flags)
    read_file = ReadFile(flags)
    dataset = Dataset(flags.location, flags.filelist, transform, read_file)
    return dataset, collate_func


def _simple_dataset(flags):
    """不使用 thsolver.Dataset 的回退 dataset——直接读取 filelist。"""
    with open(flags.filelist, 'r') as f:
        filenames = [os.path.join(flags.location, line.strip().split()[0])
                     for line in f if line.strip()]

    return _SimpleShapeNetDataset(filenames, flags)


class _SimpleShapeNetDataset(torch.utils.data.Dataset):
    """不依赖 thsolver 的最小 ShapeNet 数据集。"""

    def __init__(self, filenames, flags):
        self.filenames = filenames
        self.transform = TransformShape(flags)
        self.read_file = ReadFile(flags)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        sample = self.read_file(self.filenames[idx])
        return self.transform(sample, idx)
