"""
从八叉树占用率提取 Mesh。

将八叉树叶子节点占用率转换为稠密体素网格，
然后运行 Marching Cubes 提取等值面 mesh。

与 OctGPT 不同（需要 VQ-VAE 解码器 + Neural MPU），
此模块直接对 occupancy 阈值为 mesh。

当使用 VQ-VAE 管线时，mesh 提取通过 VQVAEWrapper.decode_to_mpu()
进行——本模块的 marching_cubes 和 save_mesh 仍被共用。
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch


def octree_to_voxel(octree, depth: int) -> np.ndarray:
    """将八叉树叶子节点占用率转换为稠密二值体素网格。

    使用 ocnn 的 octree2voxel 将八叉树节点数据散射到
    指定深度的稠密 3D 网格中。

    参数:
        octree: ocnn.Octree 对象
        depth: 提取体素的深度

    返回:
        voxel: (D, D, D) numpy 数组，float 占用率值 [0, 1]
               其中 D = 2^depth
    """
    import ocnn

    voxel_size = 2 ** depth

    # 叶子节点占用率: 若节点存在则为 1，否则为 0
    batch_id = octree.batch_id(depth=depth, nempty=False)
    data = torch.ones(len(batch_id), 1, device=octree.device)

    # 散射到稠密网格
    # octree2voxel 返回 (B, D, D, D, C)，需 permute 到 (B, C, D, D, D)
    data_full = ocnn.nn.octree2voxel(
        data=data, octree=octree, depth=depth, nempty=False)
    data_full = data_full.permute(0, 4, 1, 2, 3).contiguous()

    # 返回第一个 batch 项
    voxel = data_full[0, 0].cpu().numpy()
    return voxel


def marching_cubes(
    voxel: np.ndarray,
    level: float = 0.5,
    bbmin: float = -0.9,
    bbmax: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    """使用 Marching Cubes 从体素网格提取 mesh 表面。

    参数:
        voxel: (D, D, D) 占用率网格
        level: 等值面阈值
        bbmin, bbmax: 输出顶点坐标范围 (体素索引 [0,D] -> [bbmin,bbmax])

    返回:
        vertices: (V, 3) 顶点位置 (已缩放到 [bbmin, bbmax])
        faces: (F, 3) 三角形索引
    """
    from skimage import measure

    try:
        verts, faces, _, _ = measure.marching_cubes(voxel, level=level)
        # 体素索引 [0, D] -> [bbmin, bbmax]
        D = voxel.shape[0]
        verts = verts * ((bbmax - bbmin) / D) + bbmin
        return verts, faces
    except (ValueError, RuntimeError):
        # 在该 level 未找到表面
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)


def voxel2mesh(voxel: np.ndarray, threshold: float = 0.5) -> 'trimesh.Trimesh':
    """直接体素→mesh 转换，使用方块状体素面片。

    比 Marching Cubes 更快的可视化替代方案。
    产生轴对齐的方块状 mesh。

    参数:
        voxel: (D, D, D) 占用率网格
        threshold: 占用率阈值

    返回:
        trimesh.Trimesh 对象
    """
    import trimesh

    D = voxel.shape[0]
    scale = 2.0 / D

    # 单位立方体的 6 个面 → 8 个顶点 → 12 个三角形
    top_verts = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    top_faces = np.array([[0, 1, 3], [1, 2, 3]])
    bottom_verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    bottom_faces = np.array([[1, 0, 3], [2, 1, 3]])
    left_verts = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]])
    left_faces = np.array([[0, 1, 3], [2, 0, 3]])
    right_verts = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    right_faces = np.array([[1, 0, 3], [0, 2, 3]])
    front_verts = np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]])
    front_faces = np.array([[1, 0, 3], [0, 2, 3]])
    back_verts = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]])
    back_faces = np.array([[0, 1, 3], [2, 0, 3]])

    # 为边界处理做填充
    D_pad = D + 2
    padded = np.zeros((D_pad, D_pad, D_pad))
    padded[1:D+1, 1:D+1, 1:D+1] = voxel

    verts = []
    faces = []
    curr_vert = 0
    a, b, c = np.where(padded > threshold)

    for i, j, k in zip(a, b, c):
        if padded[i, j, k+1] < threshold:      # +z
            verts.extend(scale * (top_verts + np.array([[i-1, j-1, k-1]])))
            faces.extend(top_faces + curr_vert); curr_vert += 4
        if padded[i, j, k-1] < threshold:      # -z
            verts.extend(scale * (bottom_verts + np.array([[i-1, j-1, k-1]])))
            faces.extend(bottom_faces + curr_vert); curr_vert += 4
        if padded[i-1, j, k] < threshold:      # -x
            verts.extend(scale * (left_verts + np.array([[i-1, j-1, k-1]])))
            faces.extend(left_faces + curr_vert); curr_vert += 4
        if padded[i+1, j, k] < threshold:      # +x
            verts.extend(scale * (right_verts + np.array([[i-1, j-1, k-1]])))
            faces.extend(right_faces + curr_vert); curr_vert += 4
        if padded[i, j+1, k] < threshold:      # +y
            verts.extend(scale * (front_verts + np.array([[i-1, j-1, k-1]])))
            faces.extend(front_faces + curr_vert); curr_vert += 4
        if padded[i, j-1, k] < threshold:      # -y
            verts.extend(scale * (back_verts + np.array([[i-1, j-1, k-1]])))
            faces.extend(back_faces + curr_vert); curr_vert += 4

    if len(verts) == 0:
        return trimesh.Trimesh()

    verts = np.array(verts) - 1.0   # 居中到原点
    faces = np.array(faces, dtype=np.int64)
    return trimesh.Trimesh(vertices=verts, faces=faces)


def save_mesh(verts: np.ndarray, faces: np.ndarray, filename: str,
              scale: float = 1.0):
    """保存 mesh 为 OBJ 文件。

    参数:
        verts: (V, 3) 顶点
        faces: (F, 3) 三角形索引
        filename: 输出路径 (.obj)
        scale: 均匀缩放系数
    """
    import trimesh

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    verts = verts * scale
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(filename)


def extract_mesh_from_octree(octree, depth: int, output_path: str,
                             method: str = 'marching_cubes',
                             level: float = 0.5, scale: float = 1.0):
    """端到端: 八叉树 → mesh 文件。

    参数:
        octree: ocnn.Octree 对象
        depth: 提取占用率的叶子深度
        output_path: .obj 保存路径
        method: 'marching_cubes' 光滑, 'voxel' 方块状
        level: 等值面阈值（仅 marching_cubes）
        scale: mesh 缩放系数
    """
    voxel = octree_to_voxel(octree, depth)

    if method == 'marching_cubes':
        verts, faces = marching_cubes(voxel, level=level)
    elif method == 'voxel':
        mesh = voxel2mesh(voxel, threshold=level)
        mesh.export(output_path)
        return
    else:
        raise ValueError(f"未知方法: {method}")

    save_mesh(verts, faces, output_path, scale=scale)
