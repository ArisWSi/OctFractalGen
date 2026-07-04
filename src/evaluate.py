"""
Evaluation pipeline for OctreeFractalGen.

End-to-end evaluation following OctGPT's protocol:
  1. Load trained model + VQ-VAE
  2. Generate N shapes → extract meshes via Neural MPU + Marching Cubes
  3. Sample 2048 surface points per mesh
  4. Load reference point clouds from filelist
  5. Compute 1-NNA, COV, MMD (CD and EMD variants)
  6. Compute diversity histogram (per-sample min CD to training set)
  7. Save metrics.json + diversity.npy

Usage:
    python -m src.evaluate \\
        --checkpoint logs/best.pt \\
        --vqvae_ckpt saved_ckpt/vqvae.pt \\
        --ref_filelist data/ShapeNet/filelist/train_airplane.txt \\
        --ref_root data/ShapeNet/dataset_256 \\
        --num_samples 100 \\
        --output_dir eval/tiny_airplane/
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.metrics import (
    compute_all_metrics,
    compute_diversity,
    sample_points_from_mesh,
    load_reference_pointclouds,
)


# ---------------------------------------------------------------------------
# Generate meshes (reuses generate.py logic)
# ---------------------------------------------------------------------------

def _load_vqvae(ckpt_path: str, device: torch.device,
                embedding_channels: int = 32, vae_name: str = "vqvae_large"):
    """Load VQ-VAE from OctGPT checkpoint with correct architecture variant."""
    octgpt_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'extern', 'octgpt',
    )
    if octgpt_root not in sys.path:
        sys.path.insert(0, octgpt_root)
    from src.model.vqvae_wrapper import create_vqvae

    vqvae = create_vqvae(
        vae_name,
        in_channels=4,
        embedding_channels=embedding_channels,
        embedding_sizes=128,
        quantizer_type='bsq',
        quantizer_group=4,
        feature='ND',
        n_node_type=7,
    )
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    vqvae.load_state_dict(checkpoint)
    vqvae = vqvae.to(device)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False
    return vqvae


def load_model(checkpoint_path: str, device: torch.device,
               vqvae_ckpt_path: Optional[str] = None):
    """Load trained OctreeFractalGen model + VQ-VAE wrapper."""
    from src.config import Config
    from src.model.fractal_octree import OctreeFractalGen
    from src.model.vqvae_wrapper import VQVAEWrapper

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'config' not in checkpoint:
        raise ValueError("Checkpoint missing 'config'.")
    config = checkpoint['config']
    model_cfg = config.model if hasattr(config, 'model') else config

    # VQ-VAE
    vqvae_wrapper = None
    eff_path = vqvae_ckpt_path
    if eff_path is None and hasattr(config, 'vqvae'):
        eff_path = config.vqvae.ckpt_path
    if eff_path and os.path.exists(eff_path):
        print(f"Loading VQ-VAE from {eff_path} ...")
        embedding_channels = getattr(config.vqvae, 'embedding_channels', 32)
        vae_name = getattr(config.vqvae, 'vae_name', 'vqvae_large')
        vae_depth = getattr(config.vqvae, 'vae_depth', 8)
        vqvae = _load_vqvae(eff_path, device, embedding_channels, vae_name)
        vqvae_wrapper = VQVAEWrapper(
            vqvae, model_cfg.depth_stop, model_cfg.full_depth, vae_depth)
        print("VQ-VAE loaded.")

    # Model
    model = OctreeFractalGen(model_cfg, vqvae_wrapper=vqvae_wrapper, fractal_level=0)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loaded checkpoint, epoch {epoch}")
    return model, vqvae_wrapper, model_cfg


@torch.no_grad()
def generate_mesh_batch(
    model, vqvae_wrapper, device, batch_size: int,
    output_dir: str, start_idx: int,
    temperature: float = 1.0, resolution: int = 256,
):
    """Generate one batch of meshes. Returns list of (idx, mesh_path)."""
    import ocnn
    import trimesh
    from src.utils.mesh import marching_cubes, save_mesh

    config = model.config
    octree = ocnn.octree.init_octree(
        depth=config.depth_stop,
        full_depth=config.full_depth,
        batch_size=batch_size,
        device=device,
    )
    octree, vq_indices = model.generate(
        octree, labels=None, temperature=temperature, cfg_scale=1.0,
    )

    results = []
    for b in range(batch_size):
        idx = start_idx + b
        output_path = os.path.join(output_dir, f'{idx:04d}.obj')

        try:
            if vqvae_wrapper is not None:
                # Neural MPU → Marching Cubes
                neural_mpu = vqvae_wrapper.decode_to_mpu(vq_indices, octree)

                size = resolution
                coords = np.stack(np.meshgrid(
                    np.linspace(-0.9, 0.9, size),
                    np.linspace(-0.9, 0.9, size),
                    np.linspace(-0.9, 0.9, size),
                    indexing='ij',
                ), axis=-1).reshape(-1, 3)
                coords_t = torch.from_numpy(coords).float()

                # 获取设备，将查询点移到同一设备
                vae_device = next(vqvae_wrapper.vqvae.parameters()).device
                coords_t = coords_t.to(vae_device)

                sdf_values = []
                chunk_size = 64 ** 3
                for i in range(0, len(coords_t), chunk_size):
                    chunk = coords_t[i:i + chunk_size]
                    idx = torch.zeros(chunk.shape[0], 1, device=chunk.device)
                    pts = torch.cat([chunk, idx], dim=1)
                    sdf_chunk = neural_mpu(pts)
                    sdf_values.append(
                        sdf_chunk.cpu().numpy() if torch.is_tensor(sdf_chunk)
                        else np.array(sdf_chunk))
                sdf = np.concatenate(sdf_values, axis=0).reshape(size, size, size)

                verts, faces = marching_cubes(sdf, level=0.002)

                if len(verts) > 0 and len(faces) > 0:
                    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                    components = mesh.split(only_watertight=True)
                    if len(components) > 0:
                        mesh = trimesh.util.concatenate(components)
                        mesh.export(output_path)
                    else:
                        save_mesh(verts, faces, output_path, scale=1.0)
                else:
                    save_mesh(verts, faces, output_path, scale=1.0)
            else:
                # Fallback: direct voxel extraction
                from src.utils.mesh import extract_mesh_from_octree
                extract_mesh_from_octree(
                    octree, depth=config.depth_stop,
                    output_path=output_path, method='marching_cubes',
                )
            results.append((idx, output_path))
        except Exception as e:
            print(f"  Error generating mesh {idx}: {e}")

    return results


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate OctreeFractalGen on geometry metrics')
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument('--vqvae_ckpt', type=str, default=None,
                        help='VQ-VAE checkpoint path')
    parser.add_argument('--device', type=str, default='cuda')

    # Generation
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of shapes to generate for evaluation')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Generation batch size')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--resolution', type=int, default=256,
                        help='Marching Cubes grid resolution')
    parser.add_argument('--num_surface_points', type=int, default=2048,
                        help='Surface points sampled per mesh')

    # Reference data
    parser.add_argument('--ref_filelist', type=str, required=True,
                        help='Path to reference filelist (e.g., train_airplane.txt)')
    parser.add_argument('--ref_root', type=str, required=True,
                        help='Root directory of reference pointcloud.npz files')
    parser.add_argument('--ref_cache', type=str, default=None,
                        help='Cache path for reference point cloud tensor')

    # Output
    parser.add_argument('--output_dir', type=str, default='eval/',
                        help='Output directory for metrics and meshes')
    parser.add_argument('--keep_meshes', action='store_true',
                        help='Keep generated meshes after evaluation')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    mesh_dir = os.path.join(args.output_dir, 'meshes')
    os.makedirs(mesh_dir, exist_ok=True)

    start_time = time.time()

    # 1. Load model
    print(f"\n{'='*60}")
    print("Step 1/5: Loading model ...")
    model, vqvae_wrapper, model_cfg = load_model(
        args.checkpoint, device, args.vqvae_ckpt)
    print(f"  Model: full_depth={model_cfg.full_depth}, "
          f"depth_stop={model_cfg.depth_stop}")

    # 2. Generate meshes
    print(f"\n{'='*60}")
    print(f"Step 2/5: Generating {args.num_samples} meshes ...")
    mesh_paths = []
    gen_start = time.time()
    for start_idx in tqdm(range(0, args.num_samples, args.batch_size),
                          desc="Generating"):
        cur_bs = min(args.batch_size, args.num_samples - start_idx)
        results = generate_mesh_batch(
            model, vqvae_wrapper, device, cur_bs,
            mesh_dir, start_idx,
            temperature=args.temperature, resolution=args.resolution,
        )
        mesh_paths.extend(results)
    gen_time = time.time() - gen_start
    print(f"  Generated {len(mesh_paths)} meshes in {gen_time:.1f}s "
          f"({gen_time/len(mesh_paths):.1f}s per mesh)" if mesh_paths else "")

    if len(mesh_paths) == 0:
        print("ERROR: No meshes generated. Aborting evaluation.")
        return

    # 3. Sample surface points from generated meshes
    print(f"\n{'='*60}")
    print(f"Step 3/5: Sampling {args.num_surface_points} surface points per mesh ...")
    sample_pcs = []
    for idx, mesh_path in tqdm(mesh_paths, desc="Sampling"):
        try:
            pts = sample_points_from_mesh(mesh_path, args.num_surface_points)
            sample_pcs.append(pts)
        except Exception as e:
            print(f"  Warning: failed to sample {mesh_path}: {e}")

    sample_pcs = torch.from_numpy(np.stack(sample_pcs))
    print(f"  Sampled: {sample_pcs.shape}")

    # 4. Load reference point clouds
    print(f"\n{'='*60}")
    print(f"Step 4/5: Loading reference point clouds ...")
    ref_pcs = load_reference_pointclouds(
        args.ref_filelist, args.ref_root,
        num_points=args.num_surface_points,
        cache_path=args.ref_cache,
    )
    print(f"  References: {ref_pcs.shape}")

    # 5. Compute metrics
    print(f"\n{'='*60}")
    print(f"Step 5/5: Computing metrics ...")
    metrics = compute_all_metrics(
        sample_pcs, ref_pcs, batch_size=64, device=str(device))

    # Diversity
    diversity = compute_diversity(sample_pcs, ref_pcs)
    metrics['diversity_mean'] = float(diversity.mean())
    metrics['diversity_std'] = float(diversity.std())

    # Add metadata
    metrics['num_samples'] = len(sample_pcs)
    metrics['num_ref'] = ref_pcs.shape[0]
    metrics['num_points'] = args.num_surface_points
    metrics['generation_time_s'] = gen_time
    metrics['total_time_s'] = time.time() - start_time

    # Save
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Save diversity array
    diversity_path = os.path.join(args.output_dir, 'diversity.npy')
    np.save(diversity_path, diversity)
    print(f"Diversity saved to {diversity_path}")

    # Cleanup
    if not args.keep_meshes:
        import shutil
        shutil.rmtree(mesh_dir)
        print(f"Cleaned up mesh directory.")

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")
    print(f"{'='*60}")
    print(f"Total time: {metrics['total_time_s']:.1f}s")


if __name__ == '__main__':
    main()
