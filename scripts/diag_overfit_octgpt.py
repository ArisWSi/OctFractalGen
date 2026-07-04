"""诊断实验 2b: FractalOctGPT 单样本过拟合。

验证新架构（复用 OctFormer）能否在单样本上收敛。
对比旧的 diag_overfit.py（残缺 PatchTransformer）。

用法:
  python -m scripts.diag_overfit_octgpt --max_steps 300
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_octgpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'extern', 'octgpt')
if _octgpt not in sys.path:
    sys.path.insert(0, _octgpt)

from src.config import Config, ModelConfig, VQVAEConfig, DataConfig, TrainConfig
from src.config import octree_fractal_base
from src.model.fractal_octgpt import FractalOctGPT
from src.model.vqvae_wrapper import VQVAEWrapper
from src.train import _load_vqvae
from src.data.shapenet import get_shapenet_dataset, collate_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=300)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_location', type=str,
                        default='/root/autodl-tmp/ShapeNet/processed')
    parser.add_argument('--data_filelist', type=str,
                        default='/root/autodl-tmp/OctGPT/ShapeNet/train_airplane.txt')
    parser.add_argument('--vqvae_ckpt', type=str,
                        default='/root/autodl-tmp/OctGPT/vqvae_large_im5_uncond_bsq32.pth')
    parser.add_argument('--log_interval', type=int, default=25)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42)

    config = octree_fractal_base()
    config.vqvae.ckpt_path = args.vqvae_ckpt
    config.data.location = args.data_location
    config.data.filelist = args.data_filelist
    config.data.batch_size = 1
    config.data.num_workers = 0

    print("加载 VQVAE ...")
    vqvae = _load_vqvae(config.vqvae, device)
    vqvae_wrapper = VQVAEWrapper(
        vqvae, config.model.depth_stop, config.model.full_depth,
        config.vqvae.vae_depth)

    print("构建 FractalOctGPT ...")
    model = FractalOctGPT(config.model, vqvae_wrapper=vqvae_wrapper,
                          fractal_level=0).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: {n_params:,}")

    dataset, collate = get_shapenet_dataset(config.data)
    sample = dataset[0]
    batch = collate_func([sample])
    octree_single = batch['octree_gt'].to(device)

    nnum_total = sum(octree_single.nnum[d] for d in range(9))
    print(f"单样本 octree: depth={octree_single.depth}, 总节点={nnum_total}")
    for d in range(config.model.full_depth, config.model.depth_stop + 1):
        print(f"  depth {d}: {octree_single.nnum[d]} 节点")

    from src.train import add_weight_decay
    param_groups = add_weight_decay(model, 0.01)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda')

    # Hook 收集每层 diag
    diags_collected = []
    orig_layer_forward = None

    def make_hook(layer_name, is_vq):
        def hook(module, args, kwargs):
            # layer.forward 返回后无法直接 hook，这里只在 forward 前记录
            return None
        return hook

    # 简单方案：monkey-patch layer.forward 收集 diag
    orig_forwards = {}
    for name, m in model.named_modules():
        if hasattr(m, 'vq_head') or hasattr(m, 'split_head'):
            orig_forwards[name] = m.forward
            def make_patched(orig, lname):
                def patched(self_layer, *a, **kw):
                    loss, cond_out, diag = orig(*a, **kw)
                    diag['layer'] = lname
                    diag['is_vq'] = hasattr(self_layer, 'vq_head')
                    diags_collected.append(diag)
                    return loss, cond_out, diag
                return patched
            m.forward = make_patched(m.forward, name).__get__(m, type(m))

    print(f"\n{'='*60}")
    print(f"实验 2b: FractalOctGPT 单样本过拟合（{args.max_steps} 步, lr={args.lr}）")
    print(f"{'='*60}\n")

    model.train()
    for step in range(args.max_steps):
        optimizer.zero_grad()
        diags_collected.clear()
        with torch.amp.autocast('cuda'):
            total_loss = model(octree_single, labels=None)

        if torch.isnan(total_loss):
            print(f"step {step}: NaN, 跳过")
            continue

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_interval == 0 or step == args.max_steps - 1:
            print(f"--- step {step:4d} | total_loss={total_loss.item():.4f} ---")
            for d in diags_collected:
                tag = 'vq' if d.get('is_vq') else 'split'
                if tag == 'vq':
                    print(f"  {d['layer']}: vq top1={d.get('vq_top1', 0):.3f}")
                else:
                    print(f"  {d['layer']}: split acc={d.get('split_acc', 0):.3f}")

    # 恢复
    for name, m in model.named_modules():
        if name in orig_forwards:
            m.forward = orig_forwards[name]

    print(f"\n{'='*60}")
    print(f"最终 total_loss = {total_loss.item():.4f}")
    print(f"\n判断标准（对比旧架构）:")
    print(f"  旧架构 300步: total=0.79, d6_vq top1=0.667")
    print(f"  若新架构 top1 > 0.80 → OctFormer 复用成功")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
