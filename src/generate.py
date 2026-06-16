"""
Inference script for Occupancy-Only Recursive Octree Generation.

Loads a trained checkpoint and generates 3D shapes via autoregressive
octree generation. Exports meshes using Marching Cubes on occupancy.

Usage:
    python -m src.generate --checkpoint logs/best.pt --output results/
    python -m src.generate --checkpoint logs/best.pt --num_samples 10 --temperature 0.8
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.model.fractal_octree import OctreeFractalGen
from src.utils.mesh import extract_mesh_from_octree


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained OctreeFractalGen from checkpoint.

    Args:
        checkpoint_path: path to .pt checkpoint
        device: torch device

    Returns:
        model: OctreeFractalGen in eval mode
        config: Config object from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config
    if 'config' in checkpoint:
        config = checkpoint['config']
        if hasattr(config, 'model'):
            model_cfg = config.model
        else:
            model_cfg = config
    else:
        raise ValueError("Checkpoint missing config. Use --model_cfg to specify manually.")

    # Create model
    model = OctreeFractalGen(model_cfg, fractal_level=0)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loaded checkpoint from epoch {epoch}")

    return model, model_cfg


def generate_one(
    model: OctreeFractalGen,
    device: torch.device,
    batch_size: int = 1,
    temperature: float = 1.0,
    cfg_scale: float = 1.0,
) -> tuple:
    """Generate a single batch of octrees.

    Args:
        model: trained OctreeFractalGen
        device: torch device
        batch_size: number of shapes to generate per batch
        temperature: sampling temperature
        cfg_scale: classifier-free guidance scale

    Returns:
        octree: generated ocnn.Octree
        occupancy: (B, N_leaf, 8) occupancy at final depth
    """
    import ocnn

    config = model.config

    # Initialize empty octree at full_depth
    octree = ocnn.octree.init_octree(
        depth=config.depth_stop,
        full_depth=config.full_depth,
        batch_size=batch_size,
        device=device,
    )

    # Generate
    octree, occupancy = model.generate(
        octree,
        labels=None,  # unconditional
        temperature=temperature,
        cfg_scale=cfg_scale,
    )

    return octree, occupancy


def main():
    parser = argparse.ArgumentParser(description='Generate shapes with OctreeFractalGen')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='results/',
                        help='Output directory for meshes')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of shapes to generate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for generation')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--cfg_scale', type=float, default=1.0,
                        help='CFG scale (>1 for stronger conditioning)')
    parser.add_argument('--mesh_method', type=str, default='marching_cubes',
                        choices=['marching_cubes', 'voxel'],
                        help='Mesh extraction method')
    parser.add_argument('--mesh_level', type=float, default=0.5,
                        help='Isosurface threshold')
    parser.add_argument('--mesh_scale', type=float, default=1.0,
                        help='Mesh scale factor')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, model_cfg = load_model(args.checkpoint, device)

    full_depth = model_cfg.full_depth
    depth_stop = model_cfg.depth_stop
    print(f"Model: full_depth={full_depth}, depth_stop={depth_stop}")

    # Generate
    print(f"Generating {args.num_samples} shapes...")
    for start_idx in tqdm(range(0, args.num_samples, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, args.num_samples)
        cur_batch_size = end_idx - start_idx

        octree, occupancy = generate_one(
            model, device,
            batch_size=cur_batch_size,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
        )

        # Export meshes for each batch item
        for b in range(cur_batch_size):
            sample_idx = start_idx + b
            output_path = os.path.join(args.output, f'{sample_idx:04d}.obj')

            # If we have multi-batch, we need to extract per-batch octree.
            # For now, extract from the first (only) batch item.
            try:
                extract_mesh_from_octree(
                    octree,
                    depth=depth_stop,
                    output_path=output_path,
                    method=args.mesh_method,
                    level=args.mesh_level,
                    scale=args.mesh_scale,
                )
                print(f"  Saved: {output_path}")
            except Exception as e:
                print(f"  Error saving {output_path}: {e}")

    print(f"\nDone! Results saved to {args.output}")


if __name__ == '__main__':
    main()
