"""
ShapeNet data loading for occupancy-only octree generation.

Adapted from OctGPT's datasets/shapenet.py. Stripped down to only the
point cloud → octree path (no SDF loading, no image/text conditioning).

Pipeline:
    1. ReadFile: load pointcloud.npz from sample directory
    2. TransformShape: points → ocnn.Octree (via build_octree)
    3. collate_func: batch octrees via ocnn.dataset.CollateBatch

Usage:
    from src.data.shapenet import get_shapenet_dataset
    dataset, collate_fn = get_shapenet_dataset(data_config)
"""

import os
from typing import Dict, Tuple

import numpy as np
import torch


def _get_ocnn_available():
    """Check if ocnn is importable."""
    try:
        import ocnn  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# ReadFile: load preprocessed point cloud from disk
# ---------------------------------------------------------------------------

class ReadFile:
    """Load pointcloud.npz from a sample directory.

    Expected directory structure:
        {location}/{sample_id}/
            pointcloud.npz  — with keys 'points' (N,3) and 'normals' (N,3)

    The sample_id is derived from the last 2 path components of the filelist entry.
    """

    def __init__(self, flags):
        self.flags = flags

    def __call__(self, filename: str) -> Dict:
        """Load point cloud from disk.

        Args:
            filename: path to sample directory (from filelist)

        Returns:
            dict with key 'point_cloud' → {'points': (N,3), 'normals': (N,3)}
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
# TransformShape: point cloud → octree
# ---------------------------------------------------------------------------

class TransformShape:
    """Convert point cloud to ocnn Octree.

    The octree is built by ocnn's Octree.build_octree() from surface points.
    This gives us the ground truth octree structure (which voxels are occupied
    at each depth).

    Key parameters:
        depth: maximum octree depth (e.g., 6)
        full_depth: minimum depth with full resolution (e.g., 3)
        points_scale: input points are in [-points_scale, points_scale],
                      rescaled to [-1, 1] for ocnn
    """

    def __init__(self, flags):
        self.flags = flags
        self.depth = flags.depth
        self.full_depth = flags.full_depth
        self.points_scale = getattr(flags, 'points_scale', 1.0)
        self.max_points = getattr(flags, 'max_points', 120000)

    def points2octree(self, points, normals=None):
        """Build ocnn Octree from point cloud.

        Args:
            points: (N, 3) float tensor
            normals: (N, 3) float tensor (optional)

        Returns:
            ocnn.Octree object
        """
        import ocnn
        from ocnn.octree import Octree, Points

        if normals is not None:
            pts = Points(points=points.cpu(), normals=normals.cpu())
        else:
            pts = Points(points=points.cpu(), normals=torch.zeros_like(points.cpu()))

        pts.clip(min=-1, max=1)
        octree = Octree(self.depth, self.full_depth)
        octree.build_octree(pts)
        return octree

    def process_points_cloud(self, sample: Dict) -> Dict:
        """Build octree from point cloud sample.

        Args:
            sample: dict with 'point_cloud' → {'points': array, 'normals': array}

        Returns:
            dict with 'octree_gt': ocnn.Octree
        """
        # Load points and scale to [-1, 1]
        points_np = sample['points'].astype(np.float32)
        normals_np = sample['normals'].astype(np.float32)

        points = torch.from_numpy(points_np).float()
        normals = torch.from_numpy(normals_np).float()

        # Scale points to [-1, 1]
        points = points / self.points_scale
        # Normals are unaffected by uniform scaling

        # Randomly drop points if exceeding max_points (avoid OOM)
        if self.max_points and points.shape[0] > self.max_points:
            rand_idx = np.random.choice(
                points.shape[0], size=self.max_points, replace=False
            )
            points = points[rand_idx]
            normals = normals[rand_idx]

        # Build octree
        octree_gt = self.points2octree(points, normals)

        return {
            'octree_gt': octree_gt,
            'points': points,
            'normals': normals,
        }

    def __call__(self, sample: Dict, idx: int) -> Dict:
        """Full transform: point cloud → octree.

        Args:
            sample: output from ReadFile
            idx: sample index (unused)

        Returns:
            dict with 'octree_gt'
        """
        output = {}
        output.update(self.process_points_cloud(sample['point_cloud']))
        return output


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_func(batch):
    """Collate a list of samples into a batch.

    Uses ocnn's CollateBatch for proper octree merging.
    """
    import ocnn
    return ocnn.dataset.CollateBatch(merge_points=False)(batch)


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def get_shapenet_dataset(flags) -> Tuple:
    """Create ShapeNet dataset and collate function.

    Args:
        flags: DataConfig-like object with:
            - location: path to dataset root
            - filelist: path to text file listing sample dirs
            - depth, full_depth, points_scale, max_points

    Returns:
        (dataset, collate_func) tuple
    """
    # Lazy import to avoid hard dependency at module level
    try:
        from thsolver import Dataset
    except ImportError:
        # Fallback: use a simple list-based dataset
        return _simple_dataset(flags), collate_func

    transform = TransformShape(flags)
    read_file = ReadFile(flags)
    dataset = Dataset(flags.location, flags.filelist, transform, read_file)
    return dataset, collate_func


def _simple_dataset(flags):
    """Fallback dataset using plain list of file paths.

    Does NOT use thsolver.Dataset — reads filelist directly.
    """
    with open(flags.filelist, 'r') as f:
        filenames = [os.path.join(flags.location, line.strip().split()[0])
                     for line in f if line.strip()]

    return _SimpleShapeNetDataset(filenames, flags)


class _SimpleShapeNetDataset(torch.utils.data.Dataset):
    """Minimal ShapeNet dataset without thsolver dependency."""

    def __init__(self, filenames, flags):
        self.filenames = filenames
        self.transform = TransformShape(flags)
        self.read_file = ReadFile(flags)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        sample = self.read_file(self.filenames[idx])
        return self.transform(sample, idx)
