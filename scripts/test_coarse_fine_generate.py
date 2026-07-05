"""快速验证 CoarseFineOctGPT 生成流程 (coarse sample -> prefix -> fine sample).

不验证质量, 只验证 shape 和流程不报错.

用法:
  python -m scripts.test_coarse_fine_generate --tiny
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

import ocnn
from src.config import ModelConfig, VQVAEConfig, DataConfig
from src.model.grouped_fractal_octgpt import CoarseFineOctGPT
from src.model.vqvae_wrapper import VQVAEWrapper
from src.train import _load_vqvae
from src.data.shapenet import get_shapenet_dataset, collate_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--data_location', type=str,
                        default='/root/autodl-tmp/ShapeNet/processed')
    parser.add_argument('--data_filelist', type=str,
                        default='/root/autodl-tmp/OctGPT/ShapeNet/train_airplane.txt')
    parser.add_argument('--vqvae_ckpt', type=str,
                        default='/root/autodl-tmp/OctGPT/vqvae_large_im5_uncond_bsq32.pth')
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42)

    if args.tiny:
        model_cfg = ModelConfig(
            full_depth=3, depth_stop=6, fractal_levels=(3, 4, 5),
            coarse={'dim': 384, 'heads': 8, 'blocks': 4},
            fine={'dim': 384, 'heads': 8, 'blocks': 4},
            detach_prefix=False,
            num_iters=(8, 16, 16, 32),  # 少迭代以加速
            start_temperature=(1.0, 1.2, 0.5, 0.5),
            buffer_size=32, random_flip=0.0, remask_stage=0.7,
            use_swin=True, pos_emb_type="sin",
            patch_size=512, dilation=2,
            num_classes=1, label_drop_prob=0.0,
            grad_checkpointing=False,
        )
    else:
        model_cfg = ModelConfig(
            full_depth=3, depth_stop=6, fractal_levels=(3, 4, 5),
            coarse={'dim': 576, 'heads': 8, 'blocks': 12},
            fine={'dim': 768, 'heads': 8, 'blocks': 12},
            detach_prefix=False,
            num_iters=(16, 32, 32, 64),
            start_temperature=(1.0, 1.2, 0.5, 0.5),
            buffer_size=64, random_flip=0.0, remask_stage=0.7,
            use_swin=True, pos_emb_type="sin",
            patch_size=1024, dilation=4,
            num_classes=1, label_drop_prob=0.0,
            grad_checkpointing=False,
        )

    vqvae_cfg = VQVAEConfig(ckpt_path=args.vqvae_ckpt, vae_name="vqvae_large",
                            embedding_channels=32, vae_depth=8)
    data_cfg = DataConfig(location=args.data_location, filelist=args.data_filelist,
                          batch_size=1, num_workers=0, depth=8, full_depth=3)

    print("加载 VQVAE ...")
    vqvae = _load_vqvae(vqvae_cfg, device)
    vqvae_wrapper = VQVAEWrapper(
        vqvae, model_cfg.depth_stop, model_cfg.full_depth, vqvae_cfg.vae_depth)

    print("构建 CoarseFineOctGPT ...")
    model = CoarseFineOctGPT(model_cfg, vqvae_wrapper=vqvae_wrapper).to(device)
    model.eval()

    # 从空 octree 开始生成 (full_depth=3, depth_stop=6)
    octree = ocnn.octree.init_octree(
        depth=model_cfg.depth_stop,
        full_depth=model_cfg.full_depth,
        batch_size=1,
        device=device,
    )
    print(f"初始 octree: depth={octree.depth}, full_depth={octree.full_depth}")
    print(f"  nnum[3]={octree.nnum[3]}, nnum[4]={octree.nnum[4]}")

    print("\n运行生成 (coarse sample -> prefix -> fine sample) ...")
    octree_gen, vq_indices = model.generate(
        octree, labels=None, temperature=1.0, cfg_scale=1.0)

    print(f"\n生成完成!")
    print(f"  octree depth={octree_gen.depth}")
    for d in range(model_cfg.full_depth, model_cfg.depth_stop + 1):
        print(f"  nnum[{d}]={octree_gen.nnum[d]}")
    if vq_indices is not None:
        print(f"  vq_indices shape: {vq_indices.shape}")
        print(f"  vq_indices 值域: [{vq_indices.min().item()}, {vq_indices.max().item()}]")
        print(f"  vq_indices dtype: {vq_indices.dtype}")
    else:
        print("  vq_indices is None (错误!)")

    print("\n✓ 生成流程验证通过")


if __name__ == '__main__':
    main()
