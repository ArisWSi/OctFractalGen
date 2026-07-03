#!/usr/bin/env python3
"""
ShapeNet data preprocessing: OBJ → pointcloud.npz (+ optional VQ targets).

Pipeline:
  1. Extract category zip from Git LFS repo
  2. For each object: load OBJ → sample surface points + normals → normalize
  3. Save as pointcloud.npz
  4. [Optional] Pre-compute VQ-VAE targets and save as vq_targets.pt

Usage:
    # Basic: OBJ → pointcloud.npz
    python scripts/preprocess_shapenet.py \\
        --category airplane \\
        --zip_root /root/autodl-tmp/ShapeNetCore \\
        --output_root /root/autodl-tmp/ShapeNet/processed \\
        --num_points 100000

    # With VQ target pre-computation
    python scripts/preprocess_shapenet.py \\
        --category airplane \\
        --zip_root /root/autodl-tmp/ShapeNetCore \\
        --output_root /root/autodl-tmp/ShapeNet/processed \\
        --vqvae_ckpt /root/autodl-tmp/OctGPT/vqvae_large_im5_uncond_bsq32.pth \\
        --compute_vq_targets

    # With filelist filter (only process objects in a specific split)
    python scripts/preprocess_shapenet.py \\
        --category airplane \\
        --filelist /root/autodl-tmp/OctGPT/ShapeNet/train_airplane.txt \\
        ...
"""

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from typing import List, Optional

import numpy as np
from tqdm import tqdm


# ShapeNet category name → ID mapping (13 common categories from OctGPT)
CATEGORY_NAME_TO_ID = {
    'airplane': '02691156',
    'bench': '02828884',
    'cabinet': '02933112',
    'car': '02958343',
    'chair': '03001627',
    'monitor': '03211117',
    'lamp': '03636649',
    'loudspeaker': '03691459',
    'rifle': '04090263',
    'sofa': '04256520',
    'table': '04379243',
    'telephone': '04401088',
    'vessel': '04530566',
}
CATEGORY_ID_TO_NAME = {v: k for k, v in CATEGORY_NAME_TO_ID.items()}


# ---------------------------------------------------------------------------
# OBJ → pointcloud.npz
# ---------------------------------------------------------------------------

def load_obj_and_sample(
    obj_path: str,
    num_points: int = 100000,
    add_noise: bool = False,
    noise_std: float = 0.005,
) -> dict:
    """Load an OBJ file and sample surface points + normals.

    Args:
        obj_path: path to model_normalized.obj
        num_points: target number of surface samples
        add_noise: whether to add Gaussian noise (default False for clean octree)
        noise_std: standard deviation of noise (scaled to [-1,1] space)

    Returns:
        dict with 'points': (N, 3) and 'normals': (N, 3) float32 arrays,
        or None if loading fails
    """
    try:
        import trimesh
    except ImportError:
        print("ERROR: trimesh is required. Install with: pip install trimesh")
        return None

    try:
        mesh = trimesh.load(obj_path, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()
    except Exception as e:
        print(f"  Warning: failed to load {obj_path}: {e}")
        return None

    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        print(f"  Warning: empty mesh {obj_path}")
        return None

    # Sample surface points
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)

    # Get normals for sampled points
    if hasattr(mesh, 'face_normals') and mesh.face_normals is not None:
        normals = mesh.face_normals[face_indices]
    else:
        # Fallback: compute normals from vertices
        normals = np.zeros_like(points)
        # Use approximate normals by computing gradient
        mesh.compute_vertex_normals()

    # Scale to [-1, 1] range (matching OctGPT's points_scale=1.0)
    centroid = points.mean(axis=0)
    points = points - centroid
    max_dist = np.linalg.norm(points, axis=1).max()
    if max_dist > 0:
        points = points / max_dist

    # Optional noise (disabled by default for clean octree building)
    if add_noise:
        noise = noise_std * np.random.randn(*points.shape).astype(np.float32)
        points = points + noise

    return {
        'points': points.astype(np.float32),
        'normals': normals.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# VQ target pre-computation (requires ocnn + VQ-VAE)
# ---------------------------------------------------------------------------

def compute_vq_targets_for_sample(
    pointcloud: dict,
    vqvae_wrapper,
    depth: int = 6,
    full_depth: int = 3,
    max_points: int = 120000,
) -> Optional[np.ndarray]:
    """Compute BSQ VQ targets for a single shape.

    Builds an ocnn Octree from the point cloud, then runs the frozen
    VQ-VAE encoder + quantizer to get binary BSQ indices.

    Args:
        pointcloud: dict with 'points' and 'normals' arrays
        vqvae_wrapper: VQVAEWrapper instance
        depth: max octree depth
        full_depth: starting octree depth
        max_points: max points to keep (to avoid OOM)

    Returns:
        vq_indices: (nnum_at_depth_stop, vq_groups) int64 array, or None
    """
    import torch

    try:
        import ocnn
        from ocnn.octree import Octree, Points
    except ImportError:
        print("  Warning: ocnn not available, skipping VQ targets")
        return None

    points = pointcloud['points']
    normals = pointcloud['normals']

    # Truncate to max_points
    if points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[idx]
        normals = normals[idx]

    # Build octree from points
    points_t = torch.from_numpy(points).float()
    normals_t = torch.from_numpy(normals).float()

    # Clip to [-1, 1]
    points_t = torch.clamp(points_t, -1.0, 1.0)

    p = Points(points=points_t, normals=normals_t)
    p.clip(min=-1, max=1)

    octree = Octree(depth, full_depth)
    octree.build_octree(p)

    # Run VQ-VAE encoder + quantizer
    device = next(vqvae_wrapper.vqvae.parameters()).device
    octree = octree.to(device)
    vq_indices = vqvae_wrapper.extract_targets(octree)

    return vq_indices.cpu().numpy()


# ---------------------------------------------------------------------------
# Main preprocessing
# ---------------------------------------------------------------------------

def parse_filelist(filelist_path: str) -> set:
    """Parse a filelist into a set of object hashes for filtering."""
    if not filelist_path or not os.path.exists(filelist_path):
        return None
    hashes = set()
    with open(filelist_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Format: category_id/object_hash
                parts = line.split('/')
                if len(parts) == 2:
                    hashes.add(parts[1])
    return hashes if hashes else None


def process_category(
    category_name: str,
    zip_root: str,
    output_root: str,
    num_points: int = 100000,
    filelist: Optional[str] = None,
    add_noise: bool = False,
    compute_vq_targets: bool = False,
    vqvae_ckpt: Optional[str] = None,
    octree_depth: int = 6,
    octree_full_depth: int = 3,
) -> dict:
    """Process one ShapeNet category: extract zip, convert OBJ → pointcloud.npz.

    Args:
        category_name: e.g. 'airplane'
        zip_root: directory containing LFS zip files
        output_root: where to write processed data
        num_points: surface samples per shape
        filelist: optional path to filelist for filtering
        add_noise: whether to add noise to points
        compute_vq_targets: pre-compute VQ targets
        vqvae_ckpt: path to VQ-VAE checkpoint
        octree_depth: max octree depth for VQ target computation
        octree_full_depth: starting octree depth

    Returns:
        dict with processing statistics
    """
    category_id = CATEGORY_NAME_TO_ID.get(category_name)
    if category_id is None:
        raise ValueError(f"Unknown category: {category_name}. "
                         f"Known: {list(CATEGORY_NAME_TO_ID.keys())}")

    zip_path = os.path.join(zip_root, f"{category_id}.zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Zip not found: {zip_path}. "
            f"Run 'git lfs pull' in {zip_root} first.")

    # Optional filelist filtering
    object_filter = parse_filelist(filelist)

    # Output directory structure: {output_root}/{category_id}/{object_hash}/
    category_output = os.path.join(output_root, category_id)
    os.makedirs(category_output, exist_ok=True)

    # Load VQ-VAE if needed
    vqvae_wrapper = None
    if compute_vq_targets:
        if vqvae_ckpt is None:
            print("Warning: --vqvae_ckpt not specified, skipping VQ targets")
            compute_vq_targets = False
        else:
            print(f"Loading VQ-VAE from {vqvae_ckpt} ...")
            vqvae_wrapper = _init_vqvae(vqvae_ckpt, octree_depth, octree_full_depth)

    # Extract zip to temp directory
    print(f"Extracting {zip_path} ({_format_size(os.path.getsize(zip_path))}) ...")
    tmp_dir = tempfile.mkdtemp(prefix=f"shapenet_{category_name}_")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get list of object directories
            all_names = zf.namelist()
            obj_dirs = set()
            for name in all_names:
                parts = name.split('/')
                if len(parts) >= 2 and parts[0] == category_id:
                    obj_dirs.add(parts[1])

            # Filter if filelist specified
            if object_filter is not None:
                obj_dirs = obj_dirs & object_filter
                print(f"  Filtered to {len(obj_dirs)} objects (from filelist)")

            obj_dirs = sorted(obj_dirs)
            print(f"Processing {len(obj_dirs)} objects ...")

            # Extract only needed files
            obj_files = []
            for obj_hash in obj_dirs:
                obj_files.append(f"{category_id}/{obj_hash}/models/model_normalized.obj")
            zf.extractall(tmp_dir, members=[
                n for n in all_names
                if any(n.startswith(f) for f in obj_files)
            ])

            # Process each object
            stats = {'total': len(obj_dirs), 'success': 0, 'failed': 0, 'skipped': 0}
            for obj_hash in tqdm(obj_dirs, desc=f"  {category_name}"):
                obj_dir = os.path.join(tmp_dir, category_id, obj_hash)
                obj_path = os.path.join(obj_dir, 'models', 'model_normalized.obj')

                if not os.path.exists(obj_path):
                    stats['failed'] += 1
                    continue

                # Output paths
                out_obj_dir = os.path.join(category_output, obj_hash)
                pc_path = os.path.join(out_obj_dir, 'pointcloud.npz')
                vq_path = os.path.join(out_obj_dir, 'vq_targets.npy')

                # Skip if already done
                if os.path.exists(pc_path):
                    stats['skipped'] += 1
                    continue

                # OBJ → pointcloud
                pointcloud = load_obj_and_sample(obj_path, num_points, add_noise)
                if pointcloud is None:
                    stats['failed'] += 1
                    continue

                os.makedirs(out_obj_dir, exist_ok=True)
                np.savez_compressed(pc_path, **pointcloud)

                # VQ targets
                if compute_vq_targets and vqvae_wrapper is not None:
                    vq_indices = compute_vq_targets_for_sample(
                        pointcloud, vqvae_wrapper, octree_depth, octree_full_depth)
                    if vq_indices is not None:
                        np.save(vq_path, vq_indices)

                stats['success'] += 1

    finally:
        # Clean up temp extraction
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return stats


def _init_vqvae(vqvae_ckpt: str, depth_stop: int = 5, full_depth: int = 3):
    """Initialize VQ-VAE wrapper for target pre-computation."""
    import torch
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'src'))
    from src.model.vqvae_wrapper import VQVAEWrapper

    # Add octgpt to path for VQVAE import
    octgpt_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'extern', 'octgpt')
    if octgpt_root not in sys.path:
        sys.path.insert(0, octgpt_root)
    from models.vae import VQVAE

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vqvae = VQVAE(
        in_channels=4,
        embedding_channels=32,
        embedding_sizes=128,
        quantizer_type='bsq',
        quantizer_group=4,
        feature='ND',
        n_node_type=7,
    )
    checkpoint = torch.load(vqvae_ckpt, map_location=device, weights_only=False)
    vqvae.load_state_dict(checkpoint)
    vqvae = vqvae.to(device)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    wrapper = VQVAEWrapper(vqvae, depth_stop, full_depth, vae_depth=8)
    return wrapper


def _format_size(bytes_val: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess ShapeNet data: OBJ → pointcloud.npz')
    parser.add_argument('--category', type=str, required=True,
                        help='Category name (e.g., airplane, chair, car)')
    parser.add_argument('--zip_root', type=str, required=True,
                        help='Directory containing ShapeNet LFS zip files')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--num_points', type=int, default=100000,
                        help='Surface samples per shape')
    parser.add_argument('--filelist', type=str, default=None,
                        help='Optional filelist to filter objects')
    parser.add_argument('--add_noise', action='store_true',
                        help='Add Gaussian noise to point samples')
    parser.add_argument('--compute_vq_targets', action='store_true',
                        help='Pre-compute VQ-VAE BSQ targets')
    parser.add_argument('--vqvae_ckpt', type=str, default=None,
                        help='VQ-VAE checkpoint path (for VQ targets)')
    parser.add_argument('--octree_depth', type=int, default=8,
                        help='Max octree depth for VQ targets (must match VQ-VAE, usually 8)')
    parser.add_argument('--octree_full_depth', type=int, default=3,
                        help='Starting octree depth (must match ModelConfig.full_depth)')
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"ShapeNet Preprocessing: {args.category}")
    print(f"{'='*60}")

    stats = process_category(
        category_name=args.category,
        zip_root=args.zip_root,
        output_root=args.output_root,
        num_points=args.num_points,
        filelist=args.filelist,
        add_noise=args.add_noise,
        compute_vq_targets=args.compute_vq_targets,
        vqvae_ckpt=args.vqvae_ckpt,
        octree_depth=args.octree_depth,
        octree_full_depth=args.octree_full_depth,
    )

    print(f"\n{'='*60}")
    print(f"Results: {stats['success']} success, {stats['failed']} failed, "
          f"{stats['skipped']} skipped (out of {stats['total']})")
    print(f"Output: {os.path.join(args.output_root, CATEGORY_NAME_TO_ID[args.category])}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
