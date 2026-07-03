#!/usr/bin/env python3
"""
ShapeNet data preprocessing: OBJ → pointcloud.npz (+ optional VQ targets).

Pipeline (two-phase):
  Phase 1 [parallel]: Extract OBJ → sample surface points → pointcloud.npz
  Phase 2 [sequential]: Load pointcloud → build Octree → VQ-VAE encode → vq_targets.npy

Usage:
    # Basic: OBJ → pointcloud.npz (8 workers by default)
    python scripts/preprocess_shapenet.py \\
        --category airplane \\
        --zip_root /root/autodl-tmp/ShapeNetCore \\
        --output_root /root/autodl-tmp/ShapeNet/processed

    # With more workers + VQ targets
    python scripts/preprocess_shapenet.py \\
        --category airplane \\
        --zip_root /root/autodl-tmp/ShapeNetCore \\
        --output_root /root/autodl-tmp/ShapeNet/processed \\
        --num_workers 16 \\
        --vqvae_ckpt /root/autodl-tmp/OctGPT/vqvae_large_im5_uncond_bsq32.pth \\
        --compute_vq_targets
"""

import argparse
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import zipfile
from typing import Optional

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


# ---------------------------------------------------------------------------
# Phase 1: OBJ → pointcloud.npz  (module-level for multiprocessing)
# ---------------------------------------------------------------------------

def _process_one_object(args: tuple) -> dict:
    """Worker function: process a single OBJ → pointcloud.npz.

    Must be at module level for multiprocessing pickle support.

    Args:
        args: (obj_path, output_dir, num_points, add_noise)

    Returns:
        {'status': 'success'|'failed'|'skipped', 'hash': str}
    """
    obj_path, output_dir, num_points, add_noise = args
    obj_hash = os.path.basename(os.path.dirname(os.path.dirname(obj_path)))
    pc_path = os.path.join(output_dir, obj_hash, 'pointcloud.npz')

    # Skip if already done
    if os.path.exists(pc_path):
        return {'status': 'skipped', 'hash': obj_hash}

    # Load OBJ and sample
    try:
        import trimesh
    except ImportError:
        return {'status': 'failed', 'hash': obj_hash, 'error': 'trimesh not available'}

    try:
        mesh = trimesh.load(obj_path, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()
    except Exception as e:
        return {'status': 'failed', 'hash': obj_hash, 'error': str(e)}

    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        return {'status': 'failed', 'hash': obj_hash, 'error': 'empty mesh'}

    try:
        points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    except Exception as e:
        return {'status': 'failed', 'hash': obj_hash, 'error': f'sampling: {e}'}

    # Normals
    if hasattr(mesh, 'face_normals') and mesh.face_normals is not None:
        normals = mesh.face_normals[face_indices]
    else:
        normals = np.zeros_like(points)

    # Normalize to [-1, 1]
    centroid = points.mean(axis=0)
    points = points - centroid
    max_dist = np.linalg.norm(points, axis=1).max()
    if max_dist > 0:
        points = points / max_dist

    # Optional noise
    if add_noise:
        noise = 0.005 * np.random.randn(*points.shape).astype(np.float32)
        points = points + noise

    # Save
    os.makedirs(os.path.dirname(pc_path), exist_ok=True)
    np.savez_compressed(
        pc_path,
        points=points.astype(np.float32),
        normals=normals.astype(np.float32),
    )
    return {'status': 'success', 'hash': obj_hash}


# ---------------------------------------------------------------------------
# Phase 2: pointcloud.npz → vq_targets.npy (GPU-bound, sequential)
# ---------------------------------------------------------------------------

def _compute_vq_targets_batch(
    output_root: str,
    category_id: str,
    obj_hashes: list,
    vqvae_ckpt: str,
    vae_name: str = "vqvae_large",
    depth_stop: int = 6,
    octree_depth: int = 8,
    octree_full_depth: int = 3,
    max_points: int = 120000,
) -> dict:
    """Compute VQ-VAE BSQ targets for all processed shapes (sequential GPU pass)."""
    import torch
    import ocnn
    from ocnn.octree import Octree, Points

    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
    from src.model.vqvae_wrapper import VQVAEWrapper, create_vqvae

    # Add octgpt to path
    octgpt_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'extern', 'octgpt')
    if octgpt_root not in sys.path:
        sys.path.insert(0, octgpt_root)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading VQ-VAE ({vae_name}) on {device} ...")

    vqvae = create_vqvae(
        vae_name, in_channels=4, embedding_channels=32,
        embedding_sizes=128, quantizer_type='bsq', quantizer_group=4,
        feature='ND', n_node_type=7,
    )
    ckpt = torch.load(vqvae_ckpt, map_location=device, weights_only=False)
    vqvae.load_state_dict(ckpt)
    vqvae = vqvae.to(device)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False
    wrapper = VQVAEWrapper(vqvae, depth_stop, octree_full_depth, vae_depth=8)

    stats = {'vq_success': 0, 'vq_failed': 0, 'vq_skipped': 0}
    for obj_hash in tqdm(obj_hashes, desc="  VQ targets"):
        out_dir = os.path.join(output_root, category_id, obj_hash)
        pc_path = os.path.join(out_dir, 'pointcloud.npz')
        vq_path = os.path.join(out_dir, 'vq_targets.npy')

        if not os.path.exists(pc_path):
            stats['vq_failed'] += 1
            continue
        if os.path.exists(vq_path):
            stats['vq_skipped'] += 1
            continue

        raw = np.load(pc_path)
        points = raw['points']
        normals = raw['normals']

        if points.shape[0] > max_points:
            idx = np.random.choice(points.shape[0], max_points, replace=False)
            points = points[idx]
            normals = normals[idx]

        pts_t = torch.from_numpy(points).float().clamp(-1.0, 1.0)
        nrm_t = torch.from_numpy(normals).float()

        p = Points(points=pts_t, normals=nrm_t)
        p.clip(min=-1, max=1)
        octree = Octree(octree_depth, octree_full_depth)
        octree.build_octree(p)
        octree = octree.to(device)

        try:
            vq_indices = wrapper.extract_targets(octree)
            np.save(vq_path, vq_indices.cpu().numpy())
            stats['vq_success'] += 1
        except Exception as e:
            print(f"  VQ error {obj_hash}: {e}")
            stats['vq_failed'] += 1

    return stats


# ---------------------------------------------------------------------------
# Main
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
                parts = line.split('/')
                if len(parts) == 2:
                    hashes.add(parts[1])
    return hashes if hashes else None


def _format_size(bytes_val: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


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
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count)')
    parser.add_argument('--filelist', type=str, default=None,
                        help='Optional filelist to filter objects')
    parser.add_argument('--add_noise', action='store_true',
                        help='Add Gaussian noise to point samples')
    parser.add_argument('--compute_vq_targets', action='store_true',
                        help='Pre-compute VQ-VAE BSQ targets (phase 2, GPU)')
    parser.add_argument('--vqvae_ckpt', type=str, default=None,
                        help='VQ-VAE checkpoint path')
    parser.add_argument('--vae_name', type=str, default='vqvae_large',
                        help='VQ-VAE variant (vqvae_big|vqvae_large|vqvae_huge)')
    parser.add_argument('--depth_stop', type=int, default=6,
                        help='VQ code depth (must match VQ-VAE code_depth)')
    parser.add_argument('--octree_depth', type=int, default=8,
                        help='Max octree depth for VQ-VAE (usually 8)')
    parser.add_argument('--octree_full_depth', type=int, default=3,
                        help='Starting octree depth')
    args = parser.parse_args()

    category_id = CATEGORY_NAME_TO_ID.get(args.category)
    if category_id is None:
        raise ValueError(f"Unknown category: {args.category}. "
                         f"Known: {list(CATEGORY_NAME_TO_ID.keys())}")

    zip_path = os.path.join(args.zip_root, f"{category_id}.zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Zip not found: {zip_path}. "
            f"Run 'git lfs pull' in {args.zip_root} first.")

    num_workers = args.num_workers or mp.cpu_count()
    category_output = os.path.join(args.output_root, category_id)
    os.makedirs(category_output, exist_ok=True)
    object_filter = parse_filelist(args.filelist)

    # ------------------------------------------------------------------
    # Extract zip
    # ------------------------------------------------------------------
    print(f"{'='*60}")
    print(f"ShapeNet Preprocessing: {args.category} ({category_id})")
    print(f"{'='*60}")

    print(f"Extracting {zip_path} ({_format_size(os.path.getsize(zip_path))}) ...")
    tmp_dir = tempfile.mkdtemp(prefix=f"shapenet_{args.category}_")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            all_names = zf.namelist()
            obj_dirs = set()
            for name in all_names:
                parts = name.split('/')
                if len(parts) >= 2 and parts[0] == category_id:
                    obj_dirs.add(parts[1])

            if object_filter is not None:
                obj_dirs = obj_dirs & object_filter
                print(f"  Filtered to {len(obj_dirs)} objects (from filelist)")

            obj_dirs = sorted(obj_dirs)
            print(f"  Total objects: {len(obj_dirs)}")

            # Extract only model_normalized.obj files
            obj_file_paths = [
                f"{category_id}/{h}/models/model_normalized.obj"
                for h in obj_dirs
            ]
            zf.extractall(tmp_dir, members=[
                n for n in all_names
                if any(n.startswith(f) for f in obj_file_paths)
            ])

            # ------------------------------------------------------------------
            # Phase 1: Parallel OBJ → pointcloud.npz
            # ------------------------------------------------------------------
            print(f"\nPhase 1: OBJ → pointcloud.npz ({num_workers} workers)")
            print(f"{'─'*40}")

            # Build task list
            tasks = []
            for obj_hash in obj_dirs:
                obj_path = os.path.join(
                    tmp_dir, category_id, obj_hash, 'models', 'model_normalized.obj')
                tasks.append((obj_path, category_output, args.num_points, args.add_noise))

            # Process in parallel
            stats = {'success': 0, 'failed': 0, 'skipped': 0}
            with mp.Pool(num_workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(_process_one_object, tasks),
                    total=len(tasks),
                    desc=f"  {args.category}",
                ):
                    if result['status'] == 'success':
                        stats['success'] += 1
                    elif result['status'] == 'failed':
                        stats['failed'] += 1
                    elif result['status'] == 'skipped':
                        stats['skipped'] += 1

            print(f"  Phase 1 done: {stats['success']} success, "
                  f"{stats['failed']} failed, {stats['skipped']} skipped")

            # ------------------------------------------------------------------
            # Phase 2: VQ targets (GPU, sequential)
            # ------------------------------------------------------------------
            if args.compute_vq_targets:
                if not args.vqvae_ckpt:
                    print("\nWarning: --vqvae_ckpt not specified, skipping VQ targets")
                else:
                    print(f"\nPhase 2: VQ target pre-computation (GPU)")
                    print(f"{'─'*40}")

                    # Only process successfully generated shapes
                    valid_hashes = [
                        h for h in obj_dirs
                        if os.path.exists(
                            os.path.join(category_output, h, 'pointcloud.npz'))
                    ]

                    vq_stats = _compute_vq_targets_batch(
                        output_root=args.output_root,
                        category_id=category_id,
                        obj_hashes=valid_hashes,
                        vqvae_ckpt=args.vqvae_ckpt,
                        vae_name=args.vae_name,
                        depth_stop=args.depth_stop,
                        octree_depth=args.octree_depth,
                        octree_full_depth=args.octree_full_depth,
                    )
                    stats.update(vq_stats)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"Results: {stats.get('success', 0)} success, "
          f"{stats.get('failed', 0)} failed, "
          f"{stats.get('skipped', 0)} skipped "
          f"(out of {len(obj_dirs)})")
    print(f"Output: {category_output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
