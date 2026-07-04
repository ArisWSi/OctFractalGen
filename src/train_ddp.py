"""
DDP 多卡数据并行训练脚本（DistributedDataParallel）。

与 train.py 的区别:
- DDP: 每进程独立 GPU，forward 各自跑自己的 batch，仅同步梯度
  （Octree 作为 forward 输入不被 DDP 跟踪，天然兼容）
- DistributedSampler: 自动按 rank 切分数据，不重复不遗漏
- LR 线性放大: effective_bs = batch_size * world_size → lr *= world_size
- 仅 rank 0 做日志 / 保存 checkpoint / 验证

用法:
    torchrun --nproc_per_node=2 -m src.train_ddp \
        --config experiments/configs/fractal_base_airplane.yaml \
        --resume logs/fractal_base_airplane/checkpoint_epoch100.pt

等效 batch = batch_size(8) × 2 GPU = 16，lr 自动放大到 2e-4。
"""

import argparse
import logging
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# 模块级 logger（在 main() 中配置 handlers）
logger = logging.getLogger('train')
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, ModelConfig, VQVAEConfig, DataConfig, TrainConfig
from src.config import (
    octree_fractal_tiny,
    octree_fractal_base,
    octree_fractal_large,
)
from src.model.fractal_octree import OctreeFractalGen
from src.model.fractal_octgpt import FractalOctGPT
from src.model.vqvae_wrapper import VQVAEWrapper

# 复用 train.py 的工具函数
from src.train import (
    add_weight_decay,
    create_optimizer,
    cosine_scheduler,
    _load_vqvae,
    get_config,
    load_config_from_yaml,
)


def setup_ddp():
    """初始化 DDP 进程组，返回 (rank, world_size, local_rank)。"""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_one_epoch_ddp(
    model, dataloader, optimizer, lr_schedule, epoch, device,
    scaler, writer, config, rank, world_size, global_step=0,
):
    """DDP 训练一个 epoch。仅 rank 0 打印进度和写 TensorBoard。"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader

    for batch in pbar:
        lr_idx = global_step
        if lr_idx < len(lr_schedule):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[lr_idx]

        if 'octree_gt' in batch:
            octree = batch['octree_gt'].to(device)
        elif 'octree_in' in batch:
            octree = batch['octree_in'].to(device)
        else:
            global_step += 1
            continue

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss = model(octree, labels=None)

        if torch.isnan(loss) or torch.isinf(loss):
            if rank == 0:
                logger.warning(f"step {global_step} NaN/Inf，跳过")
            global_step += 1
            continue

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

        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1

        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss_val:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
            })
            if global_step % config.log_interval == 0 and writer is not None:
                writer.add_scalar('train/loss', loss_val, global_step)
                writer.add_scalar('train/lr',
                                  optimizer.param_groups[0]['lr'], global_step)
        global_step += 1

    # 汇总各 rank 的平均损失
    avg_loss = total_loss / max(num_batches, 1)
    avg_tensor = torch.tensor(avg_loss, device=device)
    dist.all_reduce(avg_tensor, op=dist.ReduceOp.AVG)
    avg_loss = avg_tensor.item()

    if rank == 0 and writer is not None:
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        logger.info(f"Epoch {epoch} — 平均损失: {avg_loss:.4f}")
    return global_step


@torch.no_grad()
def validate_ddp(model, dataloader, device, epoch, writer, rank):
    """验证（仅 rank 0 跑，避免 DDP 同步开销）。"""
    if rank != 0:
        return float('inf')
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
    logger.info(f"验证 Epoch {epoch} — 平均损失: {avg_loss:.4f}")
    if writer is not None:
        writer.add_scalar('val/loss', avg_loss, epoch)
    return avg_loss


def save_checkpoint_ddp(model, optimizer, epoch, global_step, scaler,
                        config, logdir, is_best=False):
    """保存 checkpoint（仅 rank 0 调用）。"""
    os.makedirs(logdir, exist_ok=True)
    raw = model.module if isinstance(model, DDP) else model
    checkpoint = {
        'model': raw.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'scaler': scaler.state_dict() if scaler is not None else None,
        'config': config,
    }
    path = os.path.join(logdir, f'checkpoint_epoch{epoch:03d}.pt')
    torch.save(checkpoint, path)
    logger.info(f"已保存 checkpoint: {path}")
    if is_best:
        best_path = os.path.join(logdir, 'best.pt')
        torch.save(checkpoint, best_path)
        logger.info(f"已保存最佳: {best_path}")


def load_checkpoint_ddp(model, optimizer, path, device):
    """加载 checkpoint（所有 rank 调用，加载后各自持有相同权重）。"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    raw = model.module if isinstance(model, DDP) else model
    raw.load_state_dict(checkpoint['model'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    logger.info(f"[rank {dist.get_rank()}] 从 {path} 加载（epoch {epoch}）")
    return epoch, global_step


def main():
    parser = argparse.ArgumentParser(description='DDP 训练 OctreeFractalGen')
    parser.add_argument('--config', type=str, default='octree_fractal_base')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='每卡 batch size（等效 = batch_size × world_size）')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None,
                        help='基础 LR（会自动 × world_size 线性放大）')
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--data_location', type=str, default=None)
    parser.add_argument('--data_filelist', type=str, default=None)
    parser.add_argument('--val_filelist', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--vqvae_ckpt', type=str, default=None)
    parser.add_argument('--no_lr_scale', action='store_true',
                        help='禁用 LR 线性放大（用原始 lr）')
    parser.add_argument('--model', type=str, default='fractal_octgpt',
                        choices=['fractal_octgpt', 'octree_fractal'],
                        help='模型架构: fractal_octgpt (复用OctFormer) 或 octree_fractal (旧架构)')
    args = parser.parse_args()

    # DDP 初始化
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')

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
    if args.val_filelist is not None:
        config.data.val_filelist = args.val_filelist
    if args.vqvae_ckpt is not None:
        config.vqvae.ckpt_path = args.vqvae_ckpt

    # 配置日志：rank 0 写文件 + stdout，其他 rank 只 stdout
    os.makedirs(config.train.logdir, exist_ok=True)
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(config.train.logdir, 'train.log'),
                                    mode='a', encoding='utf-8'),
                logging.StreamHandler(sys.stdout),
            ],
        )
    else:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s [rank%(rank)s] %(message)s')
    logger.info(f"=== 训练启动 ===")
    logger.info(f"配置: {config}")

    # LR 线性放大（等效 batch = batch_size × world_size）
    base_lr = config.train.lr
    if not args.no_lr_scale:
        config.train.lr = base_lr * world_size
    if rank == 0:
        logger.info(f"DDP: world_size={world_size}, per-GPU batch={config.data.batch_size}, "
              f"等效 batch={config.data.batch_size * world_size}")
        logger.info(f"LR: {base_lr:.2e} → {config.train.lr:.2e} (线性放大 ×{world_size})")
        logger.info(f"配置: {config}")

    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)

    # VQ-VAE（每进程各自加载冻结权重）
    vqvae_wrapper = None
    if config.vqvae.ckpt_path:
        if rank == 0:
            logger.info(f"从 {config.vqvae.ckpt_path} 加载 VQ-VAE ...")
        vqvae = _load_vqvae(config.vqvae, device)
        vqvae_wrapper = VQVAEWrapper(
            vqvae, config.model.depth_stop, config.model.full_depth,
            config.vqvae.vae_depth)

    # 模型
    if args.model == 'fractal_octgpt':
        model = FractalOctGPT(
            config.model, vqvae_wrapper=vqvae_wrapper, fractal_level=0)
    else:
        model = OctreeFractalGen(
            config.model, vqvae_wrapper=vqvae_wrapper, fractal_level=0)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        logger.info(f"模型参数量: {n_params:,}")

    # DDP 包装（find_unused_parameters: 容忍某些参数在某次 forward 中未使用，
    # 例如 VQHead 的 vq_proj 在 MaskGIT 训练时可能未被访问）
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True)

    # 数据：DistributedSampler 按 rank 切分
    from src.data.shapenet import get_shapenet_dataset
    train_dataset, train_collate = get_shapenet_dataset(config.data)
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=config.data.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, collate_fn=train_collate,
        pin_memory=True, drop_last=True)
    if rank == 0:
        logger.info(f"训练集: {len(train_dataset)} 样本 "
              f"(每卡 {len(train_dataset)//world_size})")

    # 验证：仅 rank 0 跑，用普通 sampler
    val_loader = None
    eff_val_filelist = args.val_filelist or config.data.val_filelist
    if rank == 0 and eff_val_filelist:
        from dataclasses import replace
        val_data = replace(config.data, filelist=eff_val_filelist)
        val_dataset, val_collate = get_shapenet_dataset(val_data)
        val_loader = DataLoader(
            val_dataset, batch_size=config.data.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=val_collate, pin_memory=True)
        logger.info(f"验证集: {len(val_dataset)} 样本")

    # 优化器与调度
    optimizer = create_optimizer(model, config.train)
    scaler = torch.cuda.amp.GradScaler() if config.train.use_amp else None
    lr_schedule = cosine_scheduler(
        config.train.lr, config.train.warmup_epochs,
        config.train.max_epoch, len(train_loader))

    # TensorBoard（仅 rank 0）
    writer = SummaryWriter(log_dir=config.train.logdir) if rank == 0 else None

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint_ddp(
            model, optimizer, args.resume, device)
        start_epoch += 1  # 从下一个 epoch 继续
        dist.barrier()

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(start_epoch, config.train.max_epoch):
        # 关键: 每个 epoch 设置 sampler 的 epoch，保证 shuffle 不同
        train_sampler.set_epoch(epoch)
        global_step = train_one_epoch_ddp(
            model, train_loader, optimizer, lr_schedule, epoch, device,
            scaler, writer, config.train, rank, world_size,
            global_step=global_step)

        # 验证
        is_best = False
        if val_loader is not None and epoch % 5 == 0:
            val_loss = validate_ddp(model, val_loader, device, epoch, writer, rank)
            if rank == 0:
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss

        # 保存（仅 rank 0）
        if rank == 0 and (epoch % config.train.save_interval == 0 or
                          epoch == config.train.max_epoch - 1):
            save_checkpoint_ddp(
                model, optimizer, epoch, global_step, scaler, config,
                config.train.logdir, is_best=is_best)

        dist.barrier()

    if writer is not None:
        writer.close()
    if rank == 0:
        logger.info("DDP 训练完成。")
    cleanup_ddp()


if __name__ == '__main__':
    main()
