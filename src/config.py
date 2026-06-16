"""Configuration for Occupancy-Only Recursive Octree Generation."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    """Architecture hyperparameters for OctreeFractalGen."""

    # Octree structure
    full_depth: int = 3  # initial octree depth (8 nodes at depth 3)
    depth_stop: int = 5  # final generation depth
    # fractal_levels: starting depth for each recursive level
    # e.g. (3, 4) means level 0 handles depth 3→4, level 1 handles depth 4→5
    fractal_levels: Tuple[int, ...] = (3, 4)

    # Per-level capacities (index 0 = coarsest, decreasing to finest)
    embed_dims: Tuple[int, ...] = (512, 256)
    num_blocks: Tuple[int, ...] = (16, 8)
    num_heads: Tuple[int, ...] = (8, 4)

    # Shared Transformer settings
    mlp_ratio: float = 4.0
    attn_drop: float = 0.1
    proj_drop: float = 0.1
    drop_path: float = 0.0
    rope_base: float = 10000.0

    # Class embedding (only Level 0)
    num_classes: int = 1  # 1 = unconditional (dummy class 0)
    label_drop_prob: float = 0.1  # for classifier-free guidance

    # Condition propagation
    num_spatial_neighbors: int = 7  # center + 6 face neighbors
    cond_embed_dim: int = 512  # dim of condition vectors passed between levels

    # Gradient checkpointing
    grad_checkpointing: bool = False


@dataclass
class DataConfig:
    """ShapeNet data loading configuration."""

    # Data paths
    location: str = "data/ShapeNet/dataset_256"
    filelist: str = "data/ShapeNet/airplane.txt"

    # Octree construction
    depth: int = 6  # max octree depth
    full_depth: int = 3  # initial depth (matching ModelConfig)
    points_scale: float = 1.0  # input points are in [-1, 1]

    # Data loading
    batch_size: int = 8
    num_workers: int = 4
    max_points: int = 120000
    distort: bool = False  # disable noise for clean octree


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # Schedule
    max_epoch: int = 200
    warmup_epochs: int = 5

    # Mixed precision
    use_amp: bool = True

    # Logging & checkpointing
    log_interval: int = 50  # steps between logging
    save_interval: int = 20  # epochs between checkpoints
    logdir: str = "logs/"

    # Hardware
    device: str = "cuda"
    seed: int = 42


@dataclass
class Config:
    """Top-level configuration aggregator."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# Predefined model variants (following FractalGen naming convention)

def octree_fractal_tiny(**overrides) -> Config:
    """Tiny model for fast iteration: 2 levels, small capacity."""
    return Config(
        model=ModelConfig(
            full_depth=3,
            depth_stop=5,
            fractal_levels=(3, 4),
            embed_dims=(256, 128),
            num_blocks=(8, 4),
            num_heads=(4, 4),
            cond_embed_dim=256,
            **overrides,
        )
    )


def octree_fractal_base(**overrides) -> Config:
    """Base model: 2 levels, moderate capacity."""
    return Config(
        model=ModelConfig(
            full_depth=3,
            depth_stop=5,
            fractal_levels=(3, 4),
            embed_dims=(512, 256),
            num_blocks=(16, 8),
            num_heads=(8, 4),
            cond_embed_dim=512,
            **overrides,
        )
    )


def octree_fractal_large(**overrides) -> Config:
    """Large model: 2 levels, high capacity."""
    return Config(
        model=ModelConfig(
            full_depth=3,
            depth_stop=5,
            fractal_levels=(3, 4),
            embed_dims=(768, 384),
            num_blocks=(24, 12),
            num_heads=(12, 6),
            cond_embed_dim=768,
            **overrides,
        )
    )
