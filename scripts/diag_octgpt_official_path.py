"""
用 OctGPT 官方代码路径做 GT VQVAE 重建，对比我们的诊断脚本。

直接调用 OctGPT 的 export_results 逻辑：
  octree → extract_code → quantize → decode_code → create_mesh
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'extern', 'octgpt'))

from src.config import VQVAEConfig, DataConfig
from src.model.vqvae_wrapper import VQVAEWrapper, create_vqvae
from src.data.shapenet import get_shapenet_dataset, collate_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--output', type=str, default='logs/diag_octgpt_path')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--resolution', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    # 用和官方完全一样的方式加载 VQVAE
    from models.vae import VQVAE
    vqvae_cfg = VQVAEConfig(
        ckpt_path='/root/autodl-tmp/OctGPT/vqvae_large_im5_uncond_bsq32.pth',
        vae_name='vqvae_large',
        vae_depth=8,
    )
    vqvae = create_vqvae(
        vqvae_cfg.vae_name,
        in_channels=vqvae_cfg.in_channels,
        embedding_channels=vqvae_cfg.embedding_channels,
        embedding_sizes=vqvae_cfg.embedding_sizes,
        quantizer_type=vqvae_cfg.quantizer_type,
        quantizer_group=vqvae_cfg.quantizer_group,
        feature=vqvae_cfg.feature,
        n_node_type=vqvae_cfg.n_node_type,
    )
    ckpt = torch.load(vqvae_cfg.ckpt_path, map_location=device, weights_only=False)
    vqvae.load_state_dict(ckpt)
    vqvae = vqvae.to(device).eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    # 数据
    data_cfg = DataConfig(
        location='/root/autodl-tmp/ShapeNet/processed',
        filelist='/root/autodl-tmp/OctGPT/ShapeNet/train_airplane.txt',
        depth=8, full_depth=3, batch_size=1, num_workers=0, distort=False,
    )
    dataset, collate = get_shapenet_dataset(data_cfg)

    # 导入 OctGPT 官方工具（确保 extern/octgpt 在 sys.path 最前）
    _octgpt_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'extern', 'octgpt')
    if _octgpt_path not in sys.path:
        sys.path.insert(0, _octgpt_path)
    # 强制重新导入 octgpt 的 utils（而非 stdlib）
    for mod_name in list(sys.modules.keys()):
        if mod_name in ('utils', 'utils.utils'):
            del sys.modules[mod_name]
    from utils.utils import create_mesh as octgpt_create_mesh
    from utils.utils import calc_field_values
    from ognn.octreed import OctreeD
    import copy

    depth_stop = 6
    vae_depth = 8

    print(f"\n{'='*60}")
    print(f"OctGPT 官方代码路径 GT 重建（{args.num_samples} 样本）")
    print(f"{'='*60}\n")

    for i in range(args.num_samples):
        sample = dataset[i]
        batch = collate_func([sample])
        octree = batch['octree_gt'].to(device)

        nnum_d6 = octree.nnum[depth_stop]
        print(f"[样本 {i}] depth6={nnum_d6} nodes")

        # 官方路径: extract_code → quantize → decode
        with torch.no_grad():
            vq_code = vqvae.extract_code(octree)
            zq, indices, _ = vqvae.quantizer(vq_code)
            print(f"  codes shape: {indices.shape}")

            # split_zero 扩展 (和官方一致)
            octree_out = copy.deepcopy(octree)
            for d in range(depth_stop, vae_depth):
                split_zero = torch.zeros(
                    octree_out.nnum[d], device=octree_out.device).long()
                octree_out.octree_split(split_zero, d)
                octree_out.octree_grow(d + 1)
            doctree_out = OctreeD(octree_out)

            output = vqvae.decode_code(
                zq, depth_stop, doctree_out,
                copy.deepcopy(doctree_out), update_octree=True)
            neural_mpu = output['neural_mpu']

        # 官方 create_mesh
        out_path = os.path.join(args.output, f's{i}_octgpt_path.obj')
        octgpt_create_mesh(
            neural_mpu, out_path,
            size=args.resolution,
            level=0.002, clean=True,
            bbmin=-0.9, bbmax=0.9,
            mesh_scale=1.0,  # points_scale=1.0 for uncond
            save_sdf=False)

        import trimesh
        m = trimesh.load(out_path)
        n_v = len(m.vertices) if hasattr(m, 'vertices') else 0
        n_f = len(m.faces) if hasattr(m, 'faces') else 0
        area = m.area if hasattr(m, 'area') else 0
        print(f"  官方路径: {n_v} verts, {n_f} faces, area={area:.2f} → {out_path}")
        print()

    print(f"{'='*60}")
    print(f"对比: 我们的诊断脚本 logs/diag_subnode/s*_b_splitzero.obj")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
