"""
诊断实验 1: VQVAE 重建上界。

验证冻结的预训练 VQVAE 能否从 GT octree 重建出合理的 mesh。
这是整个管线的性能天花板——如果 VQVAE 重建就很差，
生成模型再完美也无法突破。

流程:
  GT 点云 → octree(depth=8) → VQVAE.encode → BSQ indices
  → VQVAE.decode → Neural MPU → Marching Cubes → mesh

对比:
  - voxel mesh: 直接从 GT octree 的占用率提取（无 VQVAE，纯结构）
  - vqvae mesh: 经过 encode→quantize→decode 的重建

用法:
  python -m scripts.diag_vqvae_recon --num_samples 3
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# OctGPT 路径（VQVAE 依赖）
_octgpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'extern', 'octgpt')
if _octgpt not in sys.path:
    sys.path.insert(0, _octgpt)

from src.config import VQVAEConfig, DataConfig
from src.model.vqvae_wrapper import VQVAEWrapper, create_vqvae, get_vqvae_code_depth
from src.data.shapenet import get_shapenet_dataset, collate_func
from src.utils.mesh import octree_to_voxel, marching_cubes, save_mesh


def load_vqvae(vqvae_cfg, device):
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
    code_depth = get_vqvae_code_depth(vqvae_cfg.vae_name, vqvae_cfg.vae_depth)
    print(f"VQVAE({vqvae_cfg.vae_name}) loaded, code_depth={code_depth}")
    return vqvae


def export_vqvae_mesh(vqvae_wrapper, octree, indices, output_path, resolution=256):
    """VQVAE decode → Neural MPU → Marching Cubes."""
    import trimesh
    from src.utils.mesh import marching_cubes, save_mesh

    neural_mpu = vqvae_wrapper.decode_to_mpu(indices, octree)

    size = resolution
    coords = np.stack(np.meshgrid(
        np.linspace(-0.9, 0.9, size),
        np.linspace(-0.9, 0.9, size),
        np.linspace(-0.9, 0.9, size),
        indexing='ij',
    ), axis=-1).reshape(-1, 3)
    coords_t = torch.from_numpy(coords).float()
    device = next(vqvae_wrapper.vqvae.parameters()).device
    coords_t = coords_t.to(device)

    sdf_values = []
    chunk = 64 ** 3
    for i in range(0, len(coords_t), chunk):
        c = coords_t[i:i + chunk]
        idx = torch.zeros(c.shape[0], 1, device=c.device)
        pts = torch.cat([c, idx], dim=1)
        s = neural_mpu(pts)
        sdf_values.append(s.cpu().numpy() if torch.is_tensor(s) else np.array(s))
    sdf = np.concatenate(sdf_values, axis=0).reshape(size, size, size)

    verts, faces = marching_cubes(sdf, level=0.002)
    if len(verts) > 0 and len(faces) > 0:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        comps = mesh.split(only_watertight=True)
        if comps:
            mesh = trimesh.util.concatenate(comps)
            mesh.export(output_path)
            return len(mesh.vertices), len(mesh.faces)
    save_mesh(verts, faces, output_path, scale=1.0)
    return len(verts), len(faces)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--output', type=str, default='logs/diag_vqvae_recon')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--vae_name', type=str, default='vqvae_large')
    parser.add_argument('--vae_ckpt', type=str,
                        default='/root/autodl-tmp/OctGPT/vqvae_large_im5_uncond_bsq32.pth')
    parser.add_argument('--data_location', type=str,
                        default='/root/autodl-tmp/ShapeNet/processed')
    parser.add_argument('--data_filelist', type=str,
                        default='/root/autodl-tmp/OctGPT/ShapeNet/train_airplane.txt')
    parser.add_argument('--depth_stop', type=int, default=6,
                        help='VQ codes 所在深度（= code_depth）')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    # VQVAE
    vqvae_cfg = VQVAEConfig(
        ckpt_path=args.vae_ckpt,
        vae_name=args.vae_name,
        vae_depth=8,
    )
    vqvae = load_vqvae(vqvae_cfg, device)
    vqvae_wrapper = VQVAEWrapper(
        vqvae, depth_stop=args.depth_stop, full_depth=3, vae_depth=8)

    # 数据
    data_cfg = DataConfig(
        location=args.data_location,
        filelist=args.data_filelist,
        depth=8, full_depth=3, batch_size=1, num_workers=0, distort=False,
    )
    dataset, collate = get_shapenet_dataset(data_cfg)

    print(f"\n{'='*60}")
    print(f"实验 1: VQVAE 重建上界（{args.num_samples} 样本）")
    print(f"  VQVAE: {args.vae_name}, depth_stop={args.depth_stop}")
    print(f"  code_depth = vae_depth(8) - delta_depth(2) = 6")
    print(f"{'='*60}\n")

    for i in range(args.num_samples):
        sample = dataset[i]
        # 用 collate_func 包装：触发 ocnn neigh buffer 构建（训练路径也是这样）
        batch = collate_func([sample])
        octree = batch['octree_gt'].to(device)

        # 统计 GT 八叉树节点数
        nnum_total = sum(octree.nnum[d] for d in range(9))
        nnum_leaf = octree.nnum[args.depth_stop]
        print(f"[样本 {i}] GT octree: depth={octree.depth}, "
              f"总节点={nnum_total}, depth{args.depth_stop}节点={nnum_leaf}")

        # --- (a) Voxel mesh: 直接从 GT octree 结构提取（无 VQVAE）---
        voxel_path = os.path.join(args.output, f'sample{i}_a_voxel_gt.obj')
        try:
            voxel = octree_to_voxel(octree, depth=args.depth_stop)
            verts, faces = marching_cubes(voxel, level=0.5)
            save_mesh(verts, faces, voxel_path, scale=1.0)
            import trimesh
            m = trimesh.load(voxel_path)
            if hasattr(m, 'faces'):
                print(f"  (a) Voxel GT:   {len(m.vertices)} verts, {len(m.faces)} faces → {voxel_path}")
            else:
                print(f"  (a) Voxel GT:   空表面（无占用体素），voxel max={voxel.max():.2f}")
        except Exception as e:
            print(f"  (a) Voxel GT 失败: {e}")

        # --- (b) VQVAE 重建: encode → quantize → decode ---
        vqvae_path = os.path.join(args.output, f'sample{i}_b_vqvae_recon.obj')
        try:
            indices = vqvae_wrapper.extract_targets(octree)
            print(f"  (b) VQ indices shape: {indices.shape}, "
                  f"bit密度: {indices.numel()}/{nnum_leaf} = "
                  f"{indices.numel()/max(nnum_leaf,1):.1f} bits/node")

            n_verts, n_faces = export_vqvae_mesh(
                vqvae_wrapper, octree, indices, vqvae_path,
                resolution=args.resolution)
            print(f"  (b) VQVAE recon: {n_verts} verts, {n_faces} faces → {vqvae_path}")
        except Exception as e:
            import traceback
            print(f"  (b) VQVAE 重建失败: {e}")
            traceback.print_exc()

        # --- (c) 更高 depth_stop 对比（如果支持）---
        # 尝试 depth_stop=7 看信息密度提升效果
        if args.depth_stop < 7:
            print(f"\n  [对比] 尝试 depth_stop=7（8× 信息密度）...")
            try:
                vqvae_wrapper_7 = VQVAEWrapper(
                    vqvae, depth_stop=7, full_depth=3, vae_depth=8)
                indices_7 = vqvae_wrapper_7.extract_targets(octree)
                nnum_leaf_7 = octree.nnum[7]
                print(f"  (c) depth7: {indices_7.shape}, "
                      f"{indices_7.numel()}/{nnum_leaf_7} = "
                      f"{indices_7.numel()/max(nnum_leaf_7,1):.1f} bits/node")

                path7 = os.path.join(args.output, f'sample{i}_c_vqvae_depth7.obj')
                nv, nf = export_vqvae_mesh(
                    vqvae_wrapper_7, octree, indices_7, path7,
                    resolution=args.resolution)
                print(f"  (c) VQVAE depth7: {nv} verts, {nf} faces → {path7}")
            except Exception as e:
                print(f"  (c) depth_stop=7 失败（预期，VQVAE code_depth=6）: {e}")

        print()

    print(f"{'='*60}")
    print(f"完成！结果在 {args.output}/")
    print(f"\n判断标准:")
    print(f"  - (b) 顶点数 > 5000 且形状可辨 → VQVAE 上限 OK，问题在生成模型")
    print(f"  - (b) 顶点数 < 1000 或形状畸形 → VQVAE 是瓶颈，需换 VQVAE/提 depth_stop")
    print(f"  - (c) depth7 比 (b) depth6 明显更好 → 提 depth_stop 是出路")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
