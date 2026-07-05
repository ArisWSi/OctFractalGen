"""严格 VQVAE 诊断: 区分 autoencoder upper bound 与 generation-path reconstruction.

Baseline A: VQVAE autoencoder reconstruction (官方 main_vae.py 路径)
    GT octree(depth8) → VQVAE encode → quantize → decode(GT octree, update_octree) → mesh
    回答: VQVAE 本身能否重建 GT shape?

Baseline B: GT-code generation-path reconstruction (官方 main_octgpt.py 路径)
    GT VQ codes + depth6 octree → split_zero 扩展到 depth8 → decode → mesh
    回答: 完美 GT codes 在 generation decode path 下能得到多好?

用法:
  python -m scripts.diag_vqvae_strict --resolution 128
"""

import argparse
import os
import sys

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_octgpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'extern', 'octgpt')
if _octgpt not in sys.path:
    sys.path.insert(0, _octgpt)

from ognn.octreed import OctreeD
from utils import utils as octgpt_utils

from src.config import DataConfig, VQVAEConfig
from src.model.vqvae_wrapper import VQVAEWrapper
from src.train import _load_vqvae
from src.data.shapenet import get_shapenet_dataset, collate_func
from src.utils.mesh import assert_mesh_scale


def export_autoencoder_recon(vqvae, vqvae_wrapper, octree_gt, output_path,
                             resolution=128, sdf_scale=0.9, points_scale=1.0):
    """Baseline A: VQVAE autoencoder reconstruction (官方 main_vae.py 路径).

    官方 VAE test: octree_in = GT (depth8), octree_out 初始化为只到 full_depth,
    然后 update_octree=True 让 decoder 自己预测 split 结构.
    这是 VQVAE 的完整 encode+decode, 不做 split_zero.
    """
    import ocnn
    # octree_in = GT depth8 (encode 用)
    doctree_in = OctreeD(octree_gt)
    # octree_out 初始化为只到 full_depth (对齐官方 main_vae.py: OctreeD(octree_in) 但实际只 grow 到 full_depth)
    # 官方 _init_octree_out: init_octree(full_depth, full_depth, ...) → OctreeD(octree_out, full_depth)
    octree_out = ocnn.octree.init_octree(
        depth=octree_gt.full_depth,
        full_depth=octree_gt.full_depth,
        batch_size=octree_gt.batch_size,
        device=octree_gt.device,
    )
    doctree_out = OctreeD(octree_out)
    # VQVAE 完整 forward (encode + decode, update_octree=True 让 decoder 预测 split)
    output = vqvae(octree_gt, doctree_out, update_octree=True)

    octgpt_utils.create_mesh(
        output['neural_mpu'], output_path,
        size=resolution, level=0.002, clean=True,
        bbmin=-sdf_scale, bbmax=sdf_scale,
        mesh_scale=points_scale, save_sdf=False)


def export_gt_code_gen_path(vqvae_wrapper, octree_gt, output_path,
                            resolution=128, sdf_scale=0.9, points_scale=1.0,
                            depth_stop=6, vae_depth=8):
    """Baseline B: GT-code generation-path reconstruction.

    用 GT VQ codes, 但 octree 只到 depth_stop, 然后 split_zero 扩展.
    对齐官方 OctGPT generation decode path.
    """
    import copy

    # 提取 GT VQ codes
    indices = vqvae_wrapper.extract_targets(octree_gt)
    zq = vqvae_wrapper.vqvae.quantizer.extract_code(indices)

    # 构建 depth_stop octree (从 GT 截断到 depth_stop)
    # GT octree 已有完整结构, 我们只需要到 depth_stop 的部分
    octree_d6 = copy.deepcopy(octree_gt)
    # GT octree depth=8, 我们需要重建一个只到 depth_stop=6 的 octree
    # 实际上 GT octree 已经包含 depth 0-8, decode_code 用 depth_stop 参数控制
    # 但 split_zero 扩展应该从 depth_stop 开始, 对 GT octree 的 depth 7/8 不应使用
    # 正确做法: 构建只到 depth_stop 的 octree, 再 split_zero 扩展
    octree_out = copy.deepcopy(octree_gt)
    # 截断: 重置 depth_stop 之后的节点 (通过重建)
    # 简化: 直接用 GT octree, 但从 depth_stop 开始 split_zero 覆盖
    for d in range(depth_stop, vae_depth):
        split_zero = torch.zeros(
            octree_out.nnum[d], device=octree_out.device).long()
        octree_out.octree_split(split_zero, d)
        octree_out.octree_grow(d + 1)

    doctree_out = OctreeD(octree_out)
    output = vqvae_wrapper.vqvae.decode_code(
        zq, depth_stop, doctree_out,
        copy.deepcopy(doctree_out), update_octree=True)

    octgpt_utils.create_mesh(
        output['neural_mpu'], output_path,
        size=resolution, level=0.002, clean=True,
        bbmin=-sdf_scale, bbmax=sdf_scale,
        mesh_scale=points_scale, save_sdf=False)


def mesh_stats(path):
    """统计 mesh 质量."""
    if not os.path.exists(path):
        return {'v': 0, 'f': 0, 'area': 0.0, 'empty': True}
    m = trimesh.load(path, force='mesh')
    v, f = len(m.vertices), len(m.faces)
    a = float(m.area) if v > 0 else 0.0
    return {'v': v, 'f': f, 'area': a, 'empty': v == 0 or f == 0,
            'bounds': m.bounds.tolist() if v > 0 else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_location', type=str,
                        default='/root/autodl-tmp/ShapeNet/processed')
    parser.add_argument('--data_filelist', type=str,
                        default='/root/autodl-tmp/OctGPT/ShapeNet/train_airplane.txt')
    parser.add_argument('--vqvae_ckpt', type=str,
                        default='/root/autodl-tmp/OctGPT/vqvae_large_im5_uncond_bsq32.pth')
    parser.add_argument('--num_samples', type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs('logs/vqvae_strict_diag', exist_ok=True)

    vc = VQVAEConfig(ckpt_path=args.vqvae_ckpt, vae_name='vqvae_large',
                     embedding_channels=32, vae_depth=8)
    dc = DataConfig(location=args.data_location, filelist=args.data_filelist,
                    batch_size=1, num_workers=0, depth=8, full_depth=3)

    print("加载 VQVAE ...")
    vqvae = _load_vqvae(vc, device)
    vw = VQVAEWrapper(vqvae, 6, 3, 8)

    ds, coll = get_shapenet_dataset(dc)

    print(f"\n{'='*70}")
    print(f"严格 VQVAE 诊断 ({args.num_samples} 样本, resolution={args.resolution})")
    print(f"{'='*70}\n")

    for i in range(min(args.num_samples, len(ds))):
        batch = coll([ds[i]])
        octree_gt = batch['octree_gt'].to(device)
        print(f"=== sample {i} ===")
        for d in range(3, 9):
            print(f"  nnum[{d}] = {octree_gt.nnum[d]}")

        # Baseline A: autoencoder reconstruction
        path_a = f'logs/vqvae_strict_diag/autoencoder_{i:02d}.obj'
        try:
            export_autoencoder_recon(vqvae, vw, octree_gt, path_a,
                                     resolution=args.resolution)
            assert_mesh_scale(path_a)
            sa = mesh_stats(path_a)
            print(f"  [A] autoencoder: v={sa['v']} f={sa['f']} area={sa['area']:.4f}")
        except Exception as e:
            print(f"  [A] autoencoder FAILED: {e}")
            sa = {'v': 0, 'f': 0, 'area': 0.0}

        # Baseline B: GT-code generation-path
        path_b = f'logs/vqvae_strict_diag/gtcode_genpath_{i:02d}.obj'
        try:
            export_gt_code_gen_path(vw, octree_gt, path_b,
                                    resolution=args.resolution)
            assert_mesh_scale(path_b)
            sb = mesh_stats(path_b)
            print(f"  [B] gtcode_genpath: v={sb['v']} f={sb['f']} area={sb['area']:.4f}")
        except Exception as e:
            print(f"  [B] gtcode_genpath FAILED: {e}")
            import traceback; traceback.print_exc()
            sb = {'v': 0, 'f': 0, 'area': 0.0}

        print(f"  对比: autoencoder v={sa['v']} vs genpath v={sb['v']}")
        print()

    print(f"{'='*70}")
    print(f"结论判断:")
    print(f"  若 [A] autoencoder 明显好于 [B] genpath:")
    print(f"    → VQVAE 本身不差, generation decode path 是瓶颈")
    print(f"  若 [A] 和 [B] 都差:")
    print(f"    → VQVAE checkpoint 或 decoder 有问题")
    print(f"  若 [A] 和 [B] 接近:")
    print(f"    → generation decode path 对 GT codes 影响不大")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
