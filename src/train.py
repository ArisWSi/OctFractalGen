"""
Training script for Occupancy-Only Recursive Octree Generation.

Adapts FractalGen's engine_fractalgen.py training pattern:
- AdamW optimizer with weight decay grouping (no decay for bias/norm)
- Cosine LR schedule with linear warmup
- AMP mixed precision
- Gradient clipping
- Single forward pass through recursive model → combined loss

Usage:
    python -m src.train --config octree_fractal_tiny
    python -m src.train --config octree_fractal_base --batch_size 4 --epochs 200
"""

import argparse
import math
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, ModelConfig, DataConfig, TrainConfig
from src.config import (
    octree_fractal_tiny,
    octree_fractal_base,
    octree_fractal_large,
)
from src.model.fractal_octree import OctreeFractalGen


# ---------------------------------------------------------------------------
# Optimizer utilities (adapted from FractalGen util/misc.py)
# ---------------------------------------------------------------------------

def add_weight_decay(model: nn.Module, weight_decay: float = 0.01,
                     skip_list: tuple = ()) -> list:
    """Split model parameters into decay and no-decay groups.

    Bias and normalization parameters get no weight decay.
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (len(param.shape) == 1 or  # bias
            name.endswith(".bias") or
            'norm' in name.lower() or
            'ln' in name.lower() or
            name in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay},
    ]


def create_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """Create AdamW optimizer with standard FractalGen settings."""
    param_groups = add_weight_decay(model, config.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.lr,
        betas=config.betas,
    )
    return optimizer


# ---------------------------------------------------------------------------
# LR Schedule (Cosine with Linear Warmup)
# ---------------------------------------------------------------------------

def cosine_scheduler(
    base_lr: float,
    warmup_epochs: int,
    max_epoch: int,
    steps_per_epoch: int,
    min_lr: float = 1e-6,
) -> np.ndarray:
    """Compute learning rate for each training step.

    Linear warmup for warmup_epochs, then cosine decay to min_lr.
    """
    total_steps = max_epoch * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    lr = np.zeros(total_steps)

    for step in range(total_steps):
        if step < warmup_steps:
            # Linear warmup
            lr[step] = base_lr * (step + 1) / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr[step] = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

    return lr


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    lr_schedule: np.ndarray,
    epoch: int,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    writer: SummaryWriter,
    config: TrainConfig,
    global_step: int = 0,
) -> int:
    """Train for one epoch.

    Returns:
        updated global_step
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Update LR
        lr_idx = global_step
        if lr_idx < len(lr_schedule):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[lr_idx]

        # Move octree to device
        if 'octree_gt' in batch:
            batch['octree_gt'] = batch['octree_gt'].to(device)
            octree = batch['octree_gt']
        elif 'octree_in' in batch:
            batch['octree_in'] = batch['octree_in'].to(device)
            octree = batch['octree_in']
        else:
            print(f"Warning: no octree in batch at step {global_step}")
            global_step += 1
            continue

        # Forward pass
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss = model(octree, labels=None)

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss at step {global_step}, skipping")
            global_step += 1
            continue

        # Backward pass
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        # Logging
        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss_val:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
        })

        if global_step % config.log_interval == 0 and writer is not None:
            writer.add_scalar('train/loss', loss_val, global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

        global_step += 1

    avg_loss = total_loss / max(num_batches, 1)
    if writer is not None:
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)

    print(f"Epoch {epoch} — Avg Loss: {avg_loss:.4f}")
    return global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
):
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc=f"Val {epoch}"):
        if 'octree_gt' in batch:
            octree = batch['octree_gt'].to(device)
        elif 'octree_in' in batch:
            octree = batch['octree_in'].to(device)
        else:
            continue

        loss = model(octree, labels=None)
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    print(f"Validation Epoch {epoch} — Avg Loss: {avg_loss:.4f}")

    if writer is not None:
        writer.add_scalar('val/loss', avg_loss, epoch)

    return avg_loss


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    scaler,
    config: Config,
    logdir: str,
    is_best: bool = False,
):
    """Save training checkpoint."""
    os.makedirs(logdir, exist_ok=True)

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'scaler': scaler.state_dict() if scaler is not None else None,
        'config': config,
    }

    filename = f'checkpoint_epoch{epoch:03d}.pt'
    path = os.path.join(logdir, filename)
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")

    if is_best:
        best_path = os.path.join(logdir, 'best.pt')
        torch.save(checkpoint, best_path)
        print(f"Saved best: {best_path}")


def load_checkpoint(model, optimizer, path: str, device: torch.device):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    print(f"Loaded checkpoint from {path} (epoch {epoch})")
    return epoch, global_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_config(name: str) -> Config:
    """Get config by name."""
    configs = {
        'octree_fractal_tiny': octree_fractal_tiny,
        'octree_fractal_base': octree_fractal_base,
        'octree_fractal_large': octree_fractal_large,
    }
    if name in configs:
        return configs[name]()
    raise ValueError(f"Unknown config: {name}. Options: {list(configs.keys())}")


def main():
    parser = argparse.ArgumentParser(description='Train OctreeFractalGen')
    parser.add_argument('--config', type=str, default='octree_fractal_tiny',
                        help='Model config name')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Override log directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')
    parser.add_argument('--data_location', type=str, default=None,
                        help='Path to ShapeNet dataset')
    parser.add_argument('--data_filelist', type=str, default=None,
                        help='Path to filelist.txt')
    parser.add_argument('--val_filelist', type=str, default=None,
                        help='Path to validation filelist')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    args = parser.parse_args()

    # Config
    config = get_config(args.config)
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.epochs is not None:
        config.train.max_epoch = args.epochs
    if args.lr is not None:
        config.train.lr = args.lr
    if args.logdir is not None:
        config.train.logdir = args.logdir
    if args.data_location is not None:
        config.data.location = args.data_location
    if args.data_filelist is not None:
        config.data.filelist = args.data_filelist

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Config: {config}")

    # Set seed
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)

    # Create model
    model = OctreeFractalGen(config.model, fractal_level=0)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Data loading
    from src.data.shapenet import get_shapenet_dataset

    train_dataset, train_collate = get_shapenet_dataset(config.data)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_collate,
        pin_memory=True,
    )
    print(f"Train dataset: {len(train_dataset)} samples")

    # Validation loader (use same filelist if no separate val list)
    val_loader = None
    if args.val_filelist:
        from dataclasses import replace
        val_data = replace(config.data, filelist=args.val_filelist)
        val_dataset, val_collate = get_shapenet_dataset(val_data)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=val_collate,
            pin_memory=True,
        )
        print(f"Val dataset: {len(val_dataset)} samples")

    # Optimizer & scheduler
    optimizer = create_optimizer(model, config.train)
    scaler = torch.cuda.amp.GradScaler() if config.train.use_amp else None

    lr_schedule = cosine_scheduler(
        config.train.lr,
        config.train.warmup_epochs,
        config.train.max_epoch,
        len(train_loader),
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=config.train.logdir)

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(
            model, optimizer, args.resume, device
        )

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(start_epoch, config.train.max_epoch):
        global_step = train_one_epoch(
            model, train_loader, optimizer, lr_schedule,
            epoch, device, scaler, writer, config.train,
            global_step=global_step,
        )

        # Validation
        if val_loader is not None and epoch % 5 == 0:
            val_loss = validate(model, val_loader, device, epoch, writer)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
        else:
            is_best = False

        # Save checkpoint
        if epoch % config.train.save_interval == 0 or epoch == config.train.max_epoch - 1:
            save_checkpoint(
                model, optimizer, epoch, global_step,
                scaler, config, config.train.logdir, is_best=is_best,
            )

    writer.close()
    print("Training complete.")


if __name__ == '__main__':
    main()
