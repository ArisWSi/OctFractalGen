"""
递归多模型八叉树生成（VQ-VAE 管线）的训练脚本。

遵循 FractalGen 的 engine_fractalgen.py 训练模式:
- AdamW 优化器，含 weight decay 分组（bias/norm 无衰减）
- 余弦 LR 调度 + 线性 warmup
- AMP 混合精度
- 梯度裁剪
- 递归模型单次前向 → 合并损失

用法:
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

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, ModelConfig, VQVAEConfig, DataConfig, TrainConfig
from src.config import (
    octree_fractal_tiny,
    octree_fractal_base,
    octree_fractal_large,
)
from src.model.fractal_octree import OctreeFractalGen
from src.model.vqvae_wrapper import VQVAEWrapper


# ---------------------------------------------------------------------------
# 优化器工具（从 FractalGen util/misc.py 适配）
# ---------------------------------------------------------------------------

def add_weight_decay(model: nn.Module, weight_decay: float = 0.01,
                     skip_list: tuple = ()) -> list:
    """将模型参数分为 decay 和 no-decay 两组。

    bias 和 norm 参数不使用 weight decay。
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (len(param.shape) == 1 or           # bias
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


def create_optimizer(model: nn.Module,
                     config: TrainConfig) -> torch.optim.Optimizer:
    """使用标准 FractalGen 设置创建 AdamW 优化器。"""
    param_groups = add_weight_decay(model, config.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups, lr=config.lr, betas=config.betas,
    )
    return optimizer


# ---------------------------------------------------------------------------
# LR 调度（余弦 + 线性 warmup）
# ---------------------------------------------------------------------------

def cosine_scheduler(
    base_lr: float,
    warmup_epochs: int,
    max_epoch: int,
    steps_per_epoch: int,
    min_lr: float = 1e-6,
) -> np.ndarray:
    """计算每个训练步的学习率。

    warmup_epochs 线性 warmup，然后余弦衰减至 min_lr。
    """
    total_steps = max_epoch * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    lr = np.zeros(total_steps)

    for step in range(total_steps):
        if step < warmup_steps:
            lr[step] = base_lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr[step] = (min_lr + 0.5 * (base_lr - min_lr) *
                        (1 + math.cos(math.pi * progress)))

    return lr


# ---------------------------------------------------------------------------
# VQ-VAE 加载
# ---------------------------------------------------------------------------

def _load_vqvae(vqvae_cfg, device: torch.device):
    """从 OctGPT checkpoint 加载预训练 VQ-VAE。

    先尝试从 extern/octgpt 导入，使用 config 中的参数。
    """
    import sys as _sys
    octgpt_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        'extern', 'octgpt',
    )
    if octgpt_root not in _sys.path:
        _sys.path.insert(0, octgpt_root)

    from models.vae import VQVAE

    vqvae = VQVAE(
        in_channels=vqvae_cfg.in_channels,
        embedding_channels=vqvae_cfg.embedding_channels,
        embedding_sizes=vqvae_cfg.embedding_sizes,
        quantizer_type=vqvae_cfg.quantizer_type,
        quantizer_group=vqvae_cfg.quantizer_group,
        feature=vqvae_cfg.feature,
        n_node_type=vqvae_cfg.n_node_type,
    )
    checkpoint = torch.load(vqvae_cfg.ckpt_path, map_location=device,
                            weights_only=False)
    vqvae.load_state_dict(checkpoint)
    vqvae = vqvae.to(device)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False
    return vqvae


# ---------------------------------------------------------------------------
# 训练循环
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
    """训练一个 epoch。

    返回:
        更新后的 global_step
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # 更新 LR
        lr_idx = global_step
        if lr_idx < len(lr_schedule):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[lr_idx]

        # 将八叉树移到设备
        if 'octree_gt' in batch:
            octree = batch['octree_gt'].to(device)
        elif 'octree_in' in batch:
            octree = batch['octree_in'].to(device)
        else:
            print(f"警告: step {global_step} 的 batch 中无八叉树")
            global_step += 1
            continue

        # 前向传播
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss = model(octree, labels=None)

        # 检查 NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"警告: step {global_step} 出现 NaN/Inf 损失，跳过")
            global_step += 1
            continue

        # 反向传播
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip)
            optimizer.step()

        # 日志
        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss_val:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
        })

        if global_step % config.log_interval == 0 and writer is not None:
            writer.add_scalar('train/loss', loss_val, global_step)
            writer.add_scalar('train/lr',
                              optimizer.param_groups[0]['lr'], global_step)

        global_step += 1

    avg_loss = total_loss / max(num_batches, 1)
    if writer is not None:
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)

    print(f"Epoch {epoch} — 平均损失: {avg_loss:.4f}")
    return global_step


@torch.no_grad()
def validate(model: nn.Module, dataloader, device: torch.device,
             epoch: int, writer: SummaryWriter):
    """计算验证损失。"""
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
    print(f"验证 Epoch {epoch} — 平均损失: {avg_loss:.4f}")

    if writer is not None:
        writer.add_scalar('val/loss', avg_loss, epoch)

    return avg_loss


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, global_step: int, scaler, config: Config,
                    logdir: str, is_best: bool = False):
    """保存训练 checkpoint。"""
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
    print(f"已保存 checkpoint: {path}")

    if is_best:
        best_path = os.path.join(logdir, 'best.pt')
        torch.save(checkpoint, best_path)
        print(f"已保存最佳: {best_path}")


def load_checkpoint(model, optimizer, path: str, device: torch.device):
    """加载训练 checkpoint。"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    print(f"从 {path} 加载 checkpoint（epoch {epoch}）")
    return epoch, global_step


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def get_config(name_or_path: str) -> Config:
    """按预设名称或 YAML 文件路径获取配置。"""
    # Try YAML file first
    if name_or_path.endswith(('.yaml', '.yml')):
        return load_config_from_yaml(name_or_path)

    # Try preset names
    configs = {
        'octree_fractal_tiny': octree_fractal_tiny,
        'octree_fractal_base': octree_fractal_base,
        'octree_fractal_large': octree_fractal_large,
    }
    if name_or_path in configs:
        return configs[name_or_path]()

    raise ValueError(
        f"Unknown config: {name_or_path}. "
        f"Use a preset ({list(configs.keys())}) or a .yaml file path.")


def load_config_from_yaml(yaml_path: str) -> Config:
    """从 YAML 文件加载配置。

    YAML 格式需匹配 Config dataclass 的层级结构:
        model: { full_depth, depth_stop, ... }
        vqvae: { ckpt_path, ... }
        data: { location, filelist, ... }
        train: { lr, max_epoch, ... }
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML configs. Install with: pip install pyyaml")

    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f)

    model_cfg = ModelConfig(**raw.get('model', {}))
    vqvae_cfg = VQVAEConfig(**raw.get('vqvae', {}))
    data_cfg = DataConfig(**raw.get('data', {}))
    train_cfg = TrainConfig(**raw.get('train', {}))

    config = Config(
        model=model_cfg,
        vqvae=vqvae_cfg,
        data=data_cfg,
        train=train_cfg,
    )
    print(f"Loaded config from {yaml_path}")
    return config


def main():
    parser = argparse.ArgumentParser(description='训练 OctreeFractalGen')
    parser.add_argument('--config', type=str, default='octree_fractal_tiny',
                        help='模型配置: 预设名称 (octree_fractal_tiny/base/large) 或 YAML 文件路径')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='覆盖 batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖最大 epoch 数')
    parser.add_argument('--lr', type=float, default=None,
                        help='覆盖学习率')
    parser.add_argument('--logdir', type=str, default=None,
                        help='覆盖日志目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复')
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备')
    parser.add_argument('--data_location', type=str, default=None,
                        help='ShapeNet 数据集路径')
    parser.add_argument('--data_filelist', type=str, default=None,
                        help='filelist.txt 路径')
    parser.add_argument('--val_filelist', type=str, default=None,
                        help='验证 filelist 路径')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader 工作进程数')
    parser.add_argument('--vqvae_ckpt', type=str, default=None,
                        help='VQ-VAE checkpoint 路径（覆盖 config）')
    args = parser.parse_args()

    # 配置
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
    if args.vqvae_ckpt is not None:
        config.vqvae.ckpt_path = args.vqvae_ckpt

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"配置: {config}")

    # 设置随机种子
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)

    # 加载 VQ-VAE（预训练、冻结）
    vqvae_wrapper = None
    if config.vqvae.ckpt_path:
        print(f"从 {config.vqvae.ckpt_path} 加载 VQ-VAE ...")
        vqvae = _load_vqvae(config.vqvae, device)
        vqvae_wrapper = VQVAEWrapper(
            vqvae, config.model.depth_stop, config.model.full_depth,
            config.vqvae.vae_depth)
        print("VQ-VAE 已加载并冻结。")

    # 创建模型
    model = OctreeFractalGen(
        config.model, vqvae_wrapper=vqvae_wrapper, fractal_level=0)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}")

    # 数据加载
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
    print(f"训练集: {len(train_dataset)} 样本")

    # 验证数据加载器（若无单独 val list 则使用相同 filelist）
    val_loader = None
    eff_val_filelist = args.val_filelist or config.data.val_filelist
    if eff_val_filelist:
        from dataclasses import replace
        val_data = replace(config.data, filelist=eff_val_filelist)
        val_dataset, val_collate = get_shapenet_dataset(val_data)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=val_collate,
            pin_memory=True,
        )
        print(f"验证集: {len(val_dataset)} 样本")

    # 优化器与调度器
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

    # 恢复训练
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(
            model, optimizer, args.resume, device)

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(start_epoch, config.train.max_epoch):
        global_step = train_one_epoch(
            model, train_loader, optimizer, lr_schedule,
            epoch, device, scaler, writer, config.train,
            global_step=global_step,
        )

        # 验证
        if val_loader is not None and epoch % 5 == 0:
            val_loss = validate(model, val_loader, device, epoch, writer)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
        else:
            is_best = False

        # 保存 checkpoint
        if (epoch % config.train.save_interval == 0 or
                epoch == config.train.max_epoch - 1):
            save_checkpoint(
                model, optimizer, epoch, global_step,
                scaler, config, config.train.logdir, is_best=is_best,
            )

    writer.close()
    print("训练完成。")


if __name__ == '__main__':
    main()
