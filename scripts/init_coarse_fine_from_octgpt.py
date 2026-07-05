"""Plan B: 加载官方 OctGPT 权重到 CoarseFineOctGPT 的 fine 模型.

用法:
  python -m scripts.init_coarse_fine_from_octgpt \
      --config experiments/configs/coarse_fine_airplane_planb.yaml \
      --official_ckpt /root/autodl-tmp/OctGPT/octgpt_airplane.pth \
      --save logs/coarse_fine_airplane_planb/init.pt
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_octgpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'extern', 'octgpt')
if _octgpt not in sys.path:
    sys.path.insert(0, _octgpt)

from src.config import ModelConfig, VQVAEConfig
from src.model.grouped_fractal_octgpt import CoarseFineOctGPT
from src.model.vqvae_wrapper import VQVAEWrapper
from src.train import _load_vqvae, get_config


def strip_prefix(state_dict, prefixes=('module.', 'model.', 'OctGPT.')):
    """去除常见前缀."""
    new_state = {}
    for k, v in state_dict.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
                    changed = True
        new_state[nk] = v
    return new_state


def load_official_to_fine(model, official_ckpt_path):
    """加载官方 OctGPT 权重到 fine 模型.

    官方 key: split_emb, encoder.*, decoder.*, split_head, vq_head, vq_proj, class_emb
    我们 fine key: fine.split_emb, fine.encoder.*, fine.decoder.*, ...

    class_emb 不加载 (我们的在 wrapper 层叫 fine_class_emb).
    """
    ckpt = torch.load(official_ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    state = strip_prefix(state)

    # 映射到 fine.* 前缀
    fine_state = {}
    for k, v in state.items():
        if k.startswith('class_emb'):  # wrapper 层管理, 跳过
            continue
        fine_state['fine.' + k] = v

    # 加载 (strict=False, 容忍 mask_token/norm 等差异)
    model_state = model.state_dict()
    # 只加载 shape 匹配的
    loaded = {}
    skipped_shape = []
    for k, v in fine_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            loaded[k] = v
        else:
            skipped_shape.append((k, v.shape if hasattr(v, 'shape') else None))

    model.load_state_dict(loaded, strict=False)

    # 统计
    fine_keys = [k for k in model_state if k.startswith('fine.')]
    loaded_keys = [k for k in loaded]
    print(f"[PlanB] 加载官方 OctGPT -> fine")
    print(f"  fine 总 key 数: {len(fine_keys)}")
    print(f"  成功加载: {len(loaded_keys)}")
    print(f"  loaded ratio: {len(loaded_keys)/max(len(fine_keys),1):.1%}")
    print(f"  跳过 (shape 不匹配或 key 不存在): {len(skipped_shape)}")
    if skipped_shape:
        print(f"  跳过的 key (前10):")
        for k, s in skipped_shape[:10]:
            print(f"    {k} {s}")
    return loaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--official_ckpt', type=str, required=True)
    parser.add_argument('--save', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    config = get_config(args.config)
    device = torch.device(args.device)

    print(f"加载 VQVAE ...")
    vqvae = _load_vqvae(config.vqvae, device)
    vqvae_wrapper = VQVAEWrapper(
        vqvae, config.model.depth_stop, config.model.full_depth,
        config.vqvae.vae_depth)

    print(f"构建 CoarseFineOctGPT (fine blocks={config.model.fine['blocks']}) ...")
    model = CoarseFineOctGPT(config.model, vqvae_wrapper=vqvae_wrapper).to(device)

    print(f"\n加载官方 OctGPT: {args.official_ckpt}")
    load_official_to_fine(model, args.official_ckpt)

    # 保存
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'config': config,
        'model_type': 'coarse_fine_octgpt',
        'epoch': 0,
        'global_step': 0,
        'note': 'Plan B init: official OctGPT loaded to fine',
    }
    torch.save(checkpoint, args.save)
    print(f"\n已保存初始化 checkpoint: {args.save}")

    # 打印可训练参数统计
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n参数统计:")
    print(f"  总参数: {n_total:,}")
    print(f"  可训练: {n_train:,} ({n_train/n_total:.1%})")
    for name in ['coarse', 'fine', 'prefix_proj', 'prefix_norm']:
        n = sum(p.numel() for p in getattr(model, name).parameters()
                if p.requires_grad)
        print(f"  {name} 可训练: {n:,}")


if __name__ == '__main__':
    main()
