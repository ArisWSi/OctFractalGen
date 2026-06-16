"""
Mesh extraction from octree occupancy.

Converts octree leaf occupancy to a dense voxel grid, then runs
Marching Cubes to extract an isosurface mesh.

Unlike OctGPT which requires VQ-VAE decoder + Neural MPU, this
directly thresholds occupancy → mesh.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch


def octree_to_voxel(octree, depth: int) -> np.ndarray:
    """Convert octree leaf occupancy to a dense binary voxel grid.

    Uses ocnn's octree2voxel to scatter octree node data into a
    dense 3D grid at the given depth.

    Args:
        octree: ocnn.Octree object
        depth: depth at which to extract voxels

    Returns:
        voxel: (D, D, D) numpy array of float occupancy values [0, 1]
               where D = 2^depth
    """
    import ocnn

    batch_size = octree.batch_size
    voxel_size = 2 ** depth

    # Get leaf occupancy: 1 if node exists, 0 otherwise
    # At depth `depth`, existing nodes are occupied
    batch_id = octree.batch_id(depth=depth, nempty=False)
    data = torch.ones(len(batch_id), 1, device=octree.device)

    # Scatter to dense grid
    data_full = ocnn.nn.octree2voxel(data=data, octree=octree, depth=depth, nempty=False)
    # data_full: (batch_size, 1, D, D, D)

    # Return first batch item
    voxel = data_full[0, 0].cpu().numpy()  # (D, D, D)
    return voxel


def marching_cubes(
    voxel: np.ndarray,
    level: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract mesh surface from voxel grid using Marching Cubes.

    Args:
        voxel: (D, D, D) occupancy grid
        level: isosurface threshold

    Returns:
        vertices: (V, 3) vertex positions
        faces: (F, 3) triangle indices
    """
    from skimage import measure

    try:
        verts, faces, _, _ = measure.marching_cubes(voxel, level=level)
        return verts, faces
    except (ValueError, RuntimeError):
        # No surface found at this level
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)


def voxel2mesh(
    voxel: np.ndarray,
    threshold: float = 0.5,
) -> 'trimesh.Trimesh':
    """Direct voxel-to-mesh conversion using blocky voxel faces.

    Faster alternative to marching cubes for visualization.
    Produces axis-aligned blocky meshes.

    Args:
        voxel: (D, D, D) occupancy grid
        threshold: occupancy threshold

    Returns:
        trimesh.Trimesh object
    """
    import trimesh

    D = voxel.shape[0]
    scale = 2.0 / D

    # Define 6 faces of a unit cube × 8 vertices × 12 triangles
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

    # Pad voxels for boundary handling
    D_pad = D + 2
    padded = np.zeros((D_pad, D_pad, D_pad))
    padded[1:D+1, 1:D+1, 1:D+1] = voxel

    verts = []
    faces = []
    curr_vert = 0
    a, b, c = np.where(padded > threshold)

    for i, j, k in zip(a, b, c):
        # +z (top)
        if padded[i, j, k+1] < threshold:
            verts.extend(scale * (top_verts + np.array([[i-1, j-1, k-1]])))
            faces.extend(top_faces + curr_vert)
            curr_vert += 4
        # -z (bottom)
        if padded[i, j, k-1] < threshold:
            verts.extend(scale * (bottom_verts + np.array([[i-1, j-1, k-1]])))
            faces.extend(bottom_faces + curr_vert)
            curr_vert += 4
        # -x (left)
        if padded[i-1, j, k] < threshold:
            verts.extend(scale * (left_verts + np.array([[i-1, j-1, k-1]])))
            faces.extend(left_faces + curr_vert)
            curr_vert += 4
        # +x (right)
        if padded[i+1, j, k] < threshold:
            verts.extend(scale * (right_verts + np.array([[i-1, j-1, k-1]])))
            faces.extend(right_faces + curr_vert)
            curr_vert += 4
        # +y (front)
        if padded[i, j+1, k] < threshold:
            verts.extend(scale * (front_verts + np.array([[i-1, j-1, k-1]])))
            faces.extend(front_faces + curr_vert)
            curr_vert += 4
        # -y (back)
        if padded[i, j-1, k] < threshold:
            verts.extend(scale * (back_verts + np.array([[i-1, j-1, k-1]])))
            faces.extend(back_faces + curr_vert)
            curr_vert += 4

    if len(verts) == 0:
        return trimesh.Trimesh()

    verts = np.array(verts) - 1.0  # Center at origin
    faces = np.array(faces, dtype=np.int64)
    return trimesh.Trimesh(vertices=verts, faces=faces)


def save_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    filename: str,
    scale: float = 1.0,
):
    """Save mesh to OBJ file.

    Args:
        verts: (V, 3) vertices
        faces: (F, 3) triangle indices
        filename: output path (.obj)
        scale: uniform scale factor
    """
    import trimesh

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    verts = verts * scale
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(filename)


def extract_mesh_from_octree(
    octree,
    depth: int,
    output_path: str,
    method: str = 'marching_cubes',
    level: float = 0.5,
    scale: float = 1.0,
):
    """End-to-end: octree → mesh file.

    Args:
        octree: ocnn.Octree object
        depth: leaf depth for occupancy extraction
        output_path: where to save the .obj
        method: 'marching_cubes' for smooth, 'voxel' for blocky
        level: isosurface threshold (marching_cubes only)
        scale: mesh scale factor
    """
    voxel = octree_to_voxel(octree, depth)

    if method == 'marching_cubes':
        verts, faces = marching_cubes(voxel, level=level)
    elif method == 'voxel':
        mesh = voxel2mesh(voxel, threshold=level)
        mesh.export(output_path)
        return
    else:
        raise ValueError(f"Unknown method: {method}")

    save_mesh(verts, faces, output_path, scale=scale)
