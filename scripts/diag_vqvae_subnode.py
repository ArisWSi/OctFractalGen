"""
诊断: depth6 子节点结构对 VQVAE 重建的影响。

对比 3 种重建方式:
  (a) Voxel GT:   直接从 GT octree depth6 提取体素（无 VQVAE，结构上界）
  (b) split_zero: 当前做法 — depth6 codes + 全0 split 扩展到 depth8 → decode
  (c) GT struct:  depth6 codes + 保留 GT depth7/8 子节点结构 → decode

如果 (c) >> (b)，说明 split_zero 扩展丢失了子节点结构信息，是重建瓶颈。
如果 (c) ≈ (b)，说明 VQVAE decoder 不依赖子节点结构，瓶颈在 codebook。

用法:
  python -m scripts.diag_vqvae_subnode --num_samples 3
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
    return vqvae


def decode_with_structure(vqvae, indices, octree, depth_stop, vae_depth,
                          use_gt_subtree, device):
    """VQVAE decode，可选择用 GT 子树结构或 split_zero 扩展。

    use_gt_subtree=True:  保留 GT octree 的 depth7/8 子节点结构
    use_gt_subtree=False: 用全0 split 扩展（当前做法）
    """
    import copy
    from ognn.octreed import OctreeD

    zq = vqvae.quantizer.extract_code(indices)

    if use_gt_subtree:
        # 用 GT octree 的完整结构（已有 depth7/8）
        octree_out = copy.deepcopy(octree)
        # 确保 octree 长到 vae_depth
        for d in range(octree_out.depth + 1, vae_depth + 1):
            split_zero = torch.zeros(
                octree_out.nnum[d - 1] if d - 1 <= octree_out.depth else 0,
                device=octree_out.device).long()
            if octree_out.nnum[d - 1] > 0:
                octree_out.octree_split(split_zero, d - 1)
                octree_out.octree_grow(d)
    else:
        # 当前做法: split_zero 扩展
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
    return output['neural_mpu']


def mpu_to_mesh(neural_mpu, output_path, resolution=256, device='cuda'):
    """Neural MPU → Marching Cubes → OBJ。"""
    import trimesh
    size = resolution
    coords = np.stack(np.meshgrid(
        np.linspace(-0.9, 0.9, size),
        np.linspace(-0.9, 0.9, size),
        np.linspace(-0.9, 0.9, size),
        indexing='ij',
    ), axis=-1).reshape(-1, 3)
    coords_t = torch.from_numpy(coords).float().to(device)
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
        return len(mesh.vertices), len(mesh.faces), mesh.area if hasattr(mesh, 'area') else 0
    save_mesh(verts, faces, output_path, scale=1.0)
    return len(verts), len(faces), 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--output', type=str, default='logs/diag_subnode')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--depth_stop', type=int, default=6)
    parser.add_argument('--vae_ckpt', type=str,
                        default='/root/autodl-tmp/OctGPT/vqvae_large_im5_uncond_bsq32.pth')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    vqvae_cfg = VQVAEConfig(
        ckpt_path=args.vae_ckpt,
        vae_name='vqvae_large',
        vae_depth=8,
    )
    vqvae = load_vqvae(vqvae_cfg, device)
    vqvae_wrapper = VQVAEWrapper(
        vqvae, depth_stop=args.depth_stop, full_depth=3, vae_depth=8)

    data_cfg = DataConfig(
        location='/root/autodl-tmp/ShapeNet/processed',
        filelist='/root/autodl-tmp/OctGPT/ShapeNet/train_airplane.txt',
        depth=8, full_depth=3, batch_size=1, num_workers=0, distort=False,
    )
    dataset, collate = get_shapenet_dataset(data_cfg)

    print(f"\n{'='*70}")
    print(f"诊断: depth6 子节点结构对 VQVAE 重建的影响（{args.num_samples} 样本）")
    print(f"  depth_stop={args.depth_stop}, vae_depth=8")
    print(f"{'='*70}\n")

    for i in range(args.num_samples):
        sample = dataset[i]
        batch = collate_func([sample])
        octree = batch['octree_gt'].to(device)

        nnum_d6 = octree.nnum[args.depth_stop]
        nnum_d7 = octree.nnum[args.depth_stop + 1]
        nnum_d8 = octree.nnum[args.depth_stop + 2]
        print(f"[样本 {i}] depth6={nnum_d6} nodes, "
              f"depth7(GT)={nnum_d7} nodes, depth8(GT)={nnum_d8} nodes")

        # --- (a) Voxel GT at depth6 ---
        voxel_path = os.path.join(args.output, f's{i}_a_voxel_d6.obj')
        try:
            voxel = octree_to_voxel(octree, depth=args.depth_stop)
            verts, faces = marching_cubes(voxel, level=0.5)
            save_mesh(verts, faces, voxel_path, scale=1.0)
            import trimesh
            m = trimesh.load(voxel_path)
            n_v = len(m.vertices) if hasattr(m, 'vertices') else 0
            n_f = len(m.faces) if hasattr(m, 'faces') else 0
            print(f"  (a) Voxel GT d6:   {n_v} verts, {n_f} faces → {voxel_path}")
        except Exception as e:
            print(f"  (a) Voxel GT 失败: {e}")

        # 提取 GT VQ codes
        indices = vqvae_wrapper.extract_targets(octree)
        print(f"  VQ codes: {indices.shape}")

        # --- (b) split_zero 扩展（当前做法）---
        path_b = os.path.join(args.output, f's{i}_b_splitzero.obj')
        try:
            mpu_b = decode_with_structure(
                vqvae, indices, octree, args.depth_stop, 8,
                use_gt_subtree=False, device=device)
            nv, nf, area = mpu_to_mesh(mpu_b, path_b, args.resolution, device)
            print(f"  (b) split_zero:    {nv} verts, {nf} faces, area={area:.2f} → {path_b}")
        except Exception as e:
            import traceback
            print(f"  (b) split_zero 失败: {e}")
            traceback.print_exc()

        # --- (c) GT 子节点结构 ---
        path_c = os.path.join(args.output, f's{i}_c_gtstruct.obj')
        try:
            mpu_c = decode_with_structure(
                vqvae, indices, octree, args.depth_stop, 8,
                use_gt_subtree=True, device=device)
            nv, nf, area = mpu_to_mesh(mpu_c, path_c, args.resolution, device)
            print(f"  (c) GT struct:     {nv} verts, {nf} faces, area={area:.2f} → {path_c}")
        except Exception as e:
            import traceback
            print(f"  (c) GT struct 失败: {e}")
            traceback.print_exc()

        print()

    print(f"{'='*70}")
    print(f"判断标准:")
    print(f"  (c) >> (b): split_zero 是瓶颈，子节点结构信息重要")
    print(f"  (c) ≈ (b): VQVAE decoder 不依赖子节点结构，瓶颈在 codebook")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
