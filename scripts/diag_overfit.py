"""
诊断实验 2: 单样本过拟合。

验证模型架构能否表达单个 shape。
只取 1 个样本反复训练，看 loss 能否 → 0。

- 能 → 0：架构 OK，问题是数据/优化/泛化
- 不能 → 0：架构有 bug（信息流断裂）

同时记录每深度 split accuracy 和 VQ accuracy，定位瓶颈层。

用法:
  python -m scripts.diag_overfit --max_steps 500
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
from src.model.fractal_octree import OctreeFractalGen
from src.model.vqvae_wrapper import VQVAEWrapper
from src.model.ar_octree import OctreeAR
from src.model.fractal_octree import VQHead
from src.utils.octree_ops import get_node_xyz, get_split_labels
from src.train import _load_vqvae


def diag_forward(model, octree, labels=None):
    """带详细诊断的前向：返回每层 loss + accuracy。"""
    B = octree.batch_size
    class_cond = model._get_class_condition(octree, labels)
    return _diag_level(model, octree, class_cond, prefix_tokens=None, prefix_xyz=None)


def _diag_level(node, octree, global_cond, prefix_tokens, prefix_xyz):
    """递归诊断每层。"""
    device = octree.device
    results = {}

    if node.is_ar:
        depth = node.current_depth
        parent_xyz, batch_ids = get_node_xyz(octree, depth)
        nnum = octree.nnum[depth]
        if nnum == 0:
            return results

        parent_cond = node._make_per_node_cond(global_cond, batch_ids, nnum)
        gt_labels = get_split_labels(octree, depth)
        gt_split = (gt_labels.sum(dim=-1) > 0).long()

        # 调用 generator forward（带 prefix）
        split_logits, cond_out, level_loss = node.generator(
            parent_xyz, parent_cond, gt_labels, batch_ids,
            prefix_tokens=prefix_tokens, prefix_xyz=prefix_xyz)

        # 诊断: split accuracy
        with torch.no_grad():
            pred = (torch.sigmoid(split_logits) > 0.5).long()
            acc = (pred == gt_split).float().mean().item()
            split_rate_gt = gt_split.float().mean().item()
            split_rate_pred = pred.float().mean().item()

        results[f'd{depth}_split'] = {
            'loss': level_loss,
            'acc': acc,
            'gt_split_rate': split_rate_gt,
            'pred_split_rate': split_rate_pred,
            'nnum': nnum,
        }

        deeper = _diag_level(node.next_fractal, octree, global_cond,
                             prefix_tokens=cond_out, prefix_xyz=parent_xyz)
        results.update(deeper)
    else:
        # VQHead
        if node.vqvae_wrapper is None:
            return results
        final_depth = node.config.depth_stop
        leaf_xyz, leaf_batch_ids = get_node_xyz(octree, final_depth)
        nnum_leaf = octree.nnum[final_depth]
        if nnum_leaf == 0:
            return results

        leaf_cond = node._make_per_node_cond(
            global_cond, leaf_batch_ids, nnum_leaf)
        vq_targets = node.vqvae_wrapper.extract_targets(octree)

        logits, level_loss = node.next_fractal(
            leaf_xyz, leaf_cond, vq_targets, leaf_batch_ids,
            prefix_tokens=prefix_tokens, prefix_xyz=prefix_xyz)

        # 诊断: VQ accuracy
        with torch.no_grad():
            vq_pred = logits.argmax(dim=-1)  # (N, vq_groups)
            acc_top1 = (vq_pred == vq_targets).float().mean().item()
            # top5
            N, G, S = logits.shape
            top5 = torch.topk(logits, min(5, S), dim=-1).indices
            correct5 = top5.eq(vq_targets.unsqueeze(-1)).any(dim=-1).float().mean().item()

        results[f'd{final_depth}_vq'] = {
            'loss': level_loss,
            'acc_top1': acc_top1,
            'acc_top5': correct5,
            'nnum': nnum_leaf,
            'vq_groups': vq_targets.shape[1] if vq_targets.dim() > 1 else '?',
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='较高 LR 加速过拟合')
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

    # 配置（base）
    config = octree_fractal_base()
    config.vqvae.ckpt_path = args.vqvae_ckpt
    config.data.location = args.data_location
    config.data.filelist = args.data_filelist
    config.data.batch_size = 1
    config.data.num_workers = 0

    # VQVAE
    print("加载 VQVAE ...")
    vqvae = _load_vqvae(config.vqvae, device)
    vqvae_wrapper = VQVAEWrapper(
        vqvae, config.model.depth_stop, config.model.full_depth,
        config.vqvae.vae_depth)

    # 模型
    model = OctreeFractalGen(
        config.model, vqvae_wrapper=vqvae_wrapper, fractal_level=0).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: {n_params:,}")

    # 取 1 个样本
    from src.data.shapenet import get_shapenet_dataset, collate_func
    dataset, collate = get_shapenet_dataset(config.data)
    sample = dataset[0]
    # 用 collate_func 包装：触发 ocnn neigh buffer 构建
    batch = collate_func([sample])
    octree_single = batch['octree_gt'].to(device)

    nnum_total = sum(octree_single.nnum[d] for d in range(9))
    print(f"单样本 octree: depth={octree_single.depth}, 总节点={nnum_total}")
    for d in range(config.model.full_depth, config.model.depth_stop + 1):
        print(f"  depth {d}: {octree_single.nnum[d]} 节点")

    # 优化器
    from src.train import add_weight_decay
    param_groups = add_weight_decay(model, 0.01)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler()

    print(f"\n{'='*60}")
    print(f"实验 2: 单样本过拟合（{args.max_steps} 步, lr={args.lr}）")
    print(f"{'='*60}\n")

    model.train()
    for step in range(args.max_steps):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            diag = diag_forward(model, octree_single)

        total_loss = sum(v['loss'] for v in diag.values() if isinstance(v, dict))
        if torch.isnan(total_loss):
            print(f"step {step}: NaN, 跳过")
            continue

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_interval == 0 or step == args.max_steps - 1:
            tl = total_loss.item()
            print(f"--- step {step:4d} | total_loss={tl:.4f} ---")
            for key, val in diag.items():
                if not isinstance(val, dict):
                    continue
                if 'split' in key:
                    print(f"  {key}: loss={val['loss'].item():.4f} acc={val['acc']:.3f} "
                          f"(gt_split={val['gt_split_rate']:.2f} "
                          f"pred_split={val['pred_split_rate']:.2f} N={val['nnum']})")
                else:
                    print(f"  {key}: loss={val['loss'].item():.4f} "
                          f"top1={val['acc_top1']:.3f} top5={val['acc_top5']:.3f} "
                          f"N={val['nnum']} groups={val['vq_groups']}")

    final_total = sum(v['loss'].item() for v in diag.values() if isinstance(v, dict))
    print(f"\n{'='*60}")
    print(f"最终 total_loss = {final_total:.4f}")
    print(f"\n判断标准:")
    print(f"  - total_loss < 0.1 且 split acc > 0.95, vq top1 > 0.8")
    print(f"    → 架构 OK，之前效果差是优化/泛化/采样问题")
    print(f"  - total_loss 卡在 > 0.5")
    print(f"    → 架构有 bug，检查信息流（prefix/cond 传递）")
    print(f"  - 某 depth split acc 低")
    print(f"    → 该层是瓶颈")
    print(f"  - vq top1 低但 top5 高")
    print(f"    → VQ 预测接近正确但不够精确，可能容量不足")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
