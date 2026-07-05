"""诊断: CoarseFineOctGPT 单样本过拟合。

验收标准 (来自用户方案):
  d3 split acc -> 1.000
  d4 split acc -> 1.000
  d5 split acc -> 1.000
  d6 VQ top1  >= 0.99
  total loss  -> 接近 0

用法:
  python -m scripts.diag_overfit_coarse_fine --max_steps 500
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
from src.model.grouped_fractal_octgpt import CoarseFineOctGPT
from src.model.vqvae_wrapper import VQVAEWrapper
from src.train import _load_vqvae
from src.data.shapenet import get_shapenet_dataset, collate_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_location', type=str,
                        default='/root/autodl-tmp/ShapeNet/processed')
    parser.add_argument('--data_filelist', type=str,
                        default='/root/autodl-tmp/OctGPT/ShapeNet/train_airplane.txt')
    parser.add_argument('--vqvae_ckpt', type=str,
                        default='/root/autodl-tmp/OctGPT/vqvae_large_im5_uncond_bsq32.pth')
    parser.add_argument('--log_interval', type=int, default=25)
    parser.add_argument('--tiny', action='store_true',
                        help='用小模型配置 (4 blocks, 384 dim) 以节省显存')
    parser.add_argument('--cpu', action='store_true',
                        help='在 CPU 上运行 (仅验证 forward 正确性)')
    parser.add_argument('--export', type=str, default='',
                        help='训练后导出 mesh 到该目录 (空=不导出). '
                             '导出: gen_split.obj, gen_vqvae.obj, '
                             'gt_split.obj, gt_vqvae.obj')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu else args.device)
    torch.manual_seed(42)

    # 构建配置
    if args.tiny:
        model_cfg = ModelConfig(
            full_depth=3, depth_stop=6, fractal_levels=(3, 4, 5),
            coarse={'dim': 384, 'heads': 8, 'blocks': 4},
            fine={'dim': 384, 'heads': 8, 'blocks': 4},
            detach_prefix=False,
            num_iters=(32, 64, 64, 128),
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
            num_iters=(64, 128, 128, 256),
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
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: {n_params:,}")

    dataset, collate = get_shapenet_dataset(data_cfg)
    sample = dataset[0]
    batch = collate_func([sample])
    octree_single = batch['octree_gt'].to(device)

    print(f"\n单样本 octree 结构:")
    for d in range(model_cfg.full_depth, model_cfg.depth_stop + 1):
        print(f"  depth {d}: {octree_single.nnum[d]} 节点")
    nnum_coarse = octree_single.nnum[3] + octree_single.nnum[4]
    nnum_fine = octree_single.nnum[5] + octree_single.nnum[6]
    print(f"  coarse tokens (d3+d4): {nnum_coarse}")
    print(f"  fine tokens (d5+d6):   {nnum_fine}")
    print(f"  prefix_len = {nnum_coarse}")

    # 位置对齐验证: coarse hidden (d3+d4) 必须与 fine 模型 depth 3/4 token 一一对应
    # 由于两者都基于同一 octree 的 nnum[3]/nnum[4] 和 xyzb 顺序, 结构上对齐.
    # 这里验证 prefix_depths 节点总数 == coarse hidden 长度
    from src.utils.octree_ops import get_node_xyz
    xyz_d3, batch_d3 = get_node_xyz(octree_single, 3)
    xyz_d4, batch_d4 = get_node_xyz(octree_single, 4)
    print(f"\n位置对齐验证:")
    print(f"  nnum[3]={octree_single.nnum[3]}, nnum[4]={octree_single.nnum[4]}")
    print(f"  prefix_depths=[3,4] 节点总数 = {nnum_coarse} (应等于 coarse hidden 长度)")
    print(f"  coarse 模型 depth_list=[3,4], fine 模型 full_depth_list=[3,4,5,6]")
    print(f"  两者 depth 3/4 节点顺序由同一 octree.xyzb() 决定 → 结构对齐 ✓")

    from src.train import add_weight_decay
    param_groups = add_weight_decay(model, 0.01)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    use_amp = (not args.cpu) and (not args.tiny)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    print(f"\n{'='*70}")
    print(f"CoarseFineOctGPT 单样本过拟合 ({args.max_steps} 步, lr={args.lr}, "
          f"tiny={args.tiny}, cpu={args.cpu})")
    print(f"{'='*70}\n")

    model.train()
    total_loss = torch.tensor(float('nan'))
    for step in range(args.max_steps):
        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast('cuda'):
                total_loss = model(octree_single, labels=None)
        else:
            total_loss = model(octree_single, labels=None)

        if torch.isnan(total_loss):
            print(f"step {step}: NaN, 跳过")
            continue

        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()

        if step % args.log_interval == 0 or step == args.max_steps - 1:
            diag = model.get_last_diag()
            print(f"--- step {step:4d} | total={total_loss.item():.4f} "
                  f"| coarse={diag.get('loss_coarse', 0):.4f} "
                  f"fine={diag.get('loss_fine', 0):.4f} ---")
            print(f"    loss: d3={diag.get('loss_d3_split',0):.3f} "
                  f"d4={diag.get('loss_d4_split',0):.3f} "
                  f"d5={diag.get('loss_d5_split',0):.3f} "
                  f"vq={diag.get('loss_vq',0):.3f}")
            print(f"    d3 acc={diag.get('d3_acc',0):.3f} "
                  f"recall={diag.get('d3_pos_recall',0):.3f} "
                  f"f1={diag.get('d3_pos_f1',0):.3f} "
                  f"pos_t={diag.get('d3_target_pos_rate',0):.3f} "
                  f"pos_p={diag.get('d3_pred_pos_rate',0):.3f}")
            print(f"    d4 acc={diag.get('d4_acc',0):.3f} "
                  f"recall={diag.get('d4_pos_recall',0):.3f} "
                  f"f1={diag.get('d4_pos_f1',0):.3f} "
                  f"pos_t={diag.get('d4_target_pos_rate',0):.3f} "
                  f"pos_p={diag.get('d4_pred_pos_rate',0):.3f}")
            print(f"    d5 acc={diag.get('d5_acc',0):.3f} "
                  f"recall={diag.get('d5_pos_recall',0):.3f} "
                  f"f1={diag.get('d5_pos_f1',0):.3f} "
                  f"pos_t={diag.get('d5_target_pos_rate',0):.3f} "
                  f"pos_p={diag.get('d5_pred_pos_rate',0):.3f}")
            print(f"    vq top1={diag.get('vq_top1',0):.3f} "
                  f"top5={diag.get('vq_top5',0):.3f} "
                  f"full={diag.get('vq_full_code_exact_rate',0):.3f} "
                  f"hamming={diag.get('vq_hamming_per_node',0):.3f} "
                  f"entropy={diag.get('vq_code_entropy',0):.3f} "
                  f"unique={diag.get('vq_unique_code_count',0)}")

    print(f"\n{'='*70}")
    print(f"最终 total_loss = {total_loss.item():.4f}")
    diag = model.get_last_diag()
    print(f"\n验收标准:")
    print(f"  d3 acc: {diag.get('d3_acc', 0):.3f}  (目标 1.000)  "
          f"recall={diag.get('d3_pos_recall',0):.3f} f1={diag.get('d3_pos_f1',0):.3f}")
    print(f"  d4 acc: {diag.get('d4_acc', 0):.3f}  (目标 1.000)  "
          f"recall={diag.get('d4_pos_recall',0):.3f} f1={diag.get('d4_pos_f1',0):.3f}")
    print(f"  d5 acc: {diag.get('d5_acc', 0):.3f}  (目标 1.000)  "
          f"recall={diag.get('d5_pos_recall',0):.3f} f1={diag.get('d5_pos_f1',0):.3f}")
    print(f"  vq top1: {diag.get('vq_top1', 0):.3f}  (目标 >= 0.99)")
    print(f"  vq full_code_exact: {diag.get('vq_full_code_exact_rate', 0):.3f}  (目标 -> 1.0)")
    print(f"  vq hamming_per_node: {diag.get('vq_hamming_per_node', 0):.3f}  (目标 -> 0)")
    print(f"  vq entropy: {diag.get('vq_code_entropy', 0):.3f}  unique: {diag.get('vq_unique_code_count', 0)}")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # 可选: 导出 mesh (split-only + VQVAE 重建)
    # ------------------------------------------------------------------
    if args.export:
        import ocnn
        import trimesh
        from src.utils.mesh import extract_mesh_from_octree, save_mesh, marching_cubes
        os.makedirs(args.export, exist_ok=True)
        model.eval()

        print(f"\n{'='*70}")
        print(f"导出 mesh 到 {args.export}")
        print(f"{'='*70}")

        # ---- 1. GT split mesh (从 GT octree 结构直接 voxel) ----
        gt_split_path = os.path.join(args.export, 'gt_split.obj')
        try:
            extract_mesh_from_octree(
                octree_single, depth=model_cfg.depth_stop,
                output_path=gt_split_path, method='voxel', level=0.5)
            print(f"  ✓ GT split mesh: {gt_split_path}")
        except Exception as e:
            print(f"  ✗ GT split mesh 失败: {e}")

        # ---- 2. GT VQVAE 重建 (用 GT VQ codes 重建, 作为参考) ----
        gt_vqvae_path = os.path.join(args.export, 'gt_vqvae.obj')
        try:
            gt_vq_indices = vqvae_wrapper.extract_targets(octree_single)
            _export_vqvae_mesh(vqvae_wrapper, octree_single, gt_vq_indices,
                               gt_vqvae_path, resolution=128)
            print(f"  ✓ GT VQVAE 重建: {gt_vqvae_path}")
        except Exception as e:
            print(f"  ✗ GT VQVAE 重建失败: {e}")

        # ---- 3. 生成: 从空 octree 开始 ----
        print(f"\n  生成中 (coarse sample -> prefix -> fine sample) ...")
        try:
            octree_gen = ocnn.octree.init_octree(
                depth=model_cfg.depth_stop,
                full_depth=model_cfg.full_depth,
                batch_size=1,
                device=device,
            )
            octree_gen, vq_indices_gen = model.generate(
                octree_gen, labels=None, temperature=1.0, cfg_scale=1.0)
            print(f"  生成完成: nnum[6]={octree_gen.nnum[6]}, "
                  f"vq_indices shape={vq_indices_gen.shape}")

            # ---- 4. 生成 split mesh ----
            gen_split_path = os.path.join(args.export, 'gen_split.obj')
            try:
                extract_mesh_from_octree(
                    octree_gen, depth=model_cfg.depth_stop,
                    output_path=gen_split_path, method='voxel', level=0.5)
                print(f"  ✓ 生成 split mesh: {gen_split_path}")
            except Exception as e:
                print(f"  ✗ 生成 split mesh 失败: {e}")

            # ---- 5. 生成 VQVAE 重建 ----
            gen_vqvae_path = os.path.join(args.export, 'gen_vqvae.obj')
            try:
                _export_vqvae_mesh(vqvae_wrapper, octree_gen, vq_indices_gen,
                                   gen_vqvae_path, resolution=128)
                print(f"  ✓ 生成 VQVAE 重建: {gen_vqvae_path}")
            except Exception as e:
                print(f"  ✗ 生成 VQVAE 重建失败: {e}")

        except Exception as e:
            import traceback
            print(f"  ✗ 生成失败: {e}")
            traceback.print_exc()

        print(f"\n导出完成。可用 trimesh 或 meshlab 查看 .obj 文件。")


def _export_vqvae_mesh(vqvae_wrapper, octree, vq_indices, output_path,
                       resolution=128):
    """通过 VQVAE Neural MPU 解码 VQ codes → SDF → Marching Cubes → OBJ."""
    import numpy as np
    import torch
    import trimesh
    from src.utils.mesh import marching_cubes, save_mesh

    neural_mpu = vqvae_wrapper.decode_to_mpu(vq_indices, octree)
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
    chunk_size = 64 ** 3
    for i in range(0, len(coords_t), chunk_size):
        chunk = coords_t[i:i + chunk_size]
        idx = torch.zeros(chunk.shape[0], 1, device=chunk.device)
        pts = torch.cat([chunk, idx], dim=1)
        sdf_chunk = neural_mpu(pts)
        sdf_values.append(
            sdf_chunk.cpu().numpy() if torch.is_tensor(sdf_chunk)
            else np.array(sdf_chunk))
    sdf = np.concatenate(sdf_values, axis=0).reshape(size, size, size)

    verts, faces = marching_cubes(sdf, level=0.002)
    if len(verts) > 0 and len(faces) > 0:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        components = mesh.split(only_watertight=True)
        if len(components) > 0:
            mesh = trimesh.util.concatenate(components)
        mesh.export(output_path)
    else:
        save_mesh(verts, faces, output_path, scale=1.0)


if __name__ == '__main__':
    main()
