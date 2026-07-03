"""
递归多模型八叉树生成（VQ-VAE 管线）的推理脚本。

加载训练好的 checkpoint，通过自回归八叉树生成配合
VQ-VAE 几何解码来生成 3D 形状。

用法:
    python -m src.generate --checkpoint logs/best.pt --vqvae_ckpt saved_ckpt/vqvae.pt
    python -m src.generate --checkpoint logs/best.pt --num_samples 10 --temperature 0.8
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.model.fractal_octree import OctreeFractalGen
from src.model.vqvae_wrapper import VQVAEWrapper
from src.utils.mesh import marching_cubes, save_mesh


def _load_vqvae(ckpt_path: str, device: torch.device,
                embedding_channels: int = 32):
    """从 OctGPT checkpoint 加载 VQ-VAE。

    参数:
        ckpt_path: VQ-VAE .pt checkpoint 路径
        device: torch 设备
        embedding_channels: BSQ 嵌入维度（需与 checkpoint 匹配）
    """
    import sys as _sys
    octgpt_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        'extern', 'octgpt',
    )
    if octgpt_root not in _sys.path:
        _sys.path.insert(0, octgpt_root)
    from models.vae import VQVAE

    vqvae = VQVAE(
        in_channels=4,
        embedding_channels=embedding_channels,
        embedding_sizes=128,
        quantizer_type='bsq',
        quantizer_group=4,
        feature='ND',
        n_node_type=7,
    )
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    vqvae.load_state_dict(checkpoint)
    vqvae = vqvae.to(device)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False
    return vqvae


def load_model(checkpoint_path: str, device: torch.device,
               vqvae_ckpt_path: Optional[str] = None):
    """从 checkpoint 加载训练好的模型和 VQ-VAE。

    参数:
        checkpoint_path: OctreeFractalGen .pt checkpoint 路径
        device: torch 设备
        vqvae_ckpt_path: VQ-VAE .pt checkpoint 路径（可选，可从 config 读取）

    返回:
        model: 评估模式下的 OctreeFractalGen
        vqvae_wrapper: 用于 mesh 解码的 VQVAEWrapper（或 None）
        model_cfg: ModelConfig
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'config' not in checkpoint:
        raise ValueError("Checkpoint 缺少 config。")
    config = checkpoint['config']
    model_cfg = config.model if hasattr(config, 'model') else config

    # VQ-VAE 加载
    vqvae_wrapper = None
    eff_path = vqvae_ckpt_path
    if eff_path is None and hasattr(config, 'vqvae'):
        eff_path = config.vqvae.ckpt_path
    if eff_path:
        print(f"从 {eff_path} 加载 VQ-VAE ...")
        embedding_channels = getattr(config.vqvae, 'embedding_channels', 32)
        vae_depth = getattr(config.vqvae, 'vae_depth', 8)
        vqvae = _load_vqvae(eff_path, device, embedding_channels)
        vqvae_wrapper = VQVAEWrapper(
            vqvae, model_cfg.depth_stop, model_cfg.full_depth, vae_depth)
        print("VQ-VAE 已加载。")

    # 模型加载
    model = OctreeFractalGen(
        model_cfg, vqvae_wrapper=vqvae_wrapper, fractal_level=0)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    print(f"已加载 checkpoint，epoch {epoch}")
    return model, vqvae_wrapper, model_cfg


@torch.no_grad()
def generate_one(model, device, batch_size=1, temperature=1.0, cfg_scale=1.0):
    """生成一个 batch: 八叉树 + VQ indices。"""
    import ocnn

    config = model.config
    octree = ocnn.octree.init_octree(
        depth=config.depth_stop,
        full_depth=config.full_depth,
        batch_size=batch_size,
        device=device,
    )
    octree, vq_indices = model.generate(
        octree, labels=None,
        temperature=temperature, cfg_scale=cfg_scale,
    )
    return octree, vq_indices


def export_mesh_vqvae(vqvae_wrapper, octree, vq_indices, output_path,
                      resolution=256):
    """通过 VQ-VAE 解码 VQ 编码并用 Marching Cubes 提取 mesh。

    遵循 OctGPT 的 create_mesh 流程:
    Neural MPU 在稠密网格上求值 → Marching Cubes → OBJ 导出。
    """
    import trimesh

    neural_mpu = vqvae_wrapper.decode_to_mpu(vq_indices, octree)

    # 在稠密网格上求值 Neural MPU
    size = resolution
    coords = np.stack(np.meshgrid(
        np.linspace(-0.9, 0.9, size),
        np.linspace(-0.9, 0.9, size),
        np.linspace(-0.9, 0.9, size),
        indexing='ij',
    ), axis=-1).reshape(-1, 3)
    coords_t = torch.from_numpy(coords).float()

    # 分块求值以避免 OOM
    sdf_values = []
    chunk_size = 64 ** 3
    for i in range(0, len(coords_t), chunk_size):
        chunk = coords_t[i:i + chunk_size]
        sdf_chunk = neural_mpu(chunk)
        sdf_values.append(
            sdf_chunk.cpu().numpy() if torch.is_tensor(sdf_chunk)
            else np.array(sdf_chunk))
    sdf = np.concatenate(sdf_values, axis=0).reshape(size, size, size)

    # Marching Cubes 提取等值面
    verts, faces = marching_cubes(sdf, level=0.002)

    # 清理: 仅保留水密连通分量（遵循 OctGPT 的 clean=True）
    if len(verts) > 0 and len(faces) > 0:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        components = mesh.split(only_watertight=True)
        if len(components) > 0:
            mesh = trimesh.util.concatenate(components)
            mesh.export(output_path)
        else:
            save_mesh(verts, faces, output_path, scale=1.0)
    else:
        save_mesh(verts, faces, output_path, scale=1.0)


def main():
    parser = argparse.ArgumentParser(
        description='使用 OctreeFractalGen 生成形状')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型 checkpoint 路径')
    parser.add_argument('--vqvae_ckpt', type=str, default=None,
                        help='VQ-VAE checkpoint 路径（覆盖 config 中的设置）')
    parser.add_argument('--output', type=str, default='results/',
                        help='mesh 输出目录')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--cfg_scale', type=float, default=1.0)
    parser.add_argument('--resolution', type=int, default=256,
                        help='Marching Cubes 网格分辨率')
    parser.add_argument('--method', type=str, default='vqvae',
                        choices=['vqvae', 'voxel'],
                        help='Mesh 提取: vqvae (Neural MPU) 或 voxel (直接八叉树)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # 加载
    print(f"从 {args.checkpoint} 加载模型 ...")
    model, vqvae_wrapper, model_cfg = load_model(
        args.checkpoint, device, args.vqvae_ckpt,
    )
    print(f"模型: full_depth={model_cfg.full_depth}, "
          f"depth_stop={model_cfg.depth_stop}")

    # 生成
    print(f"生成 {args.num_samples} 个形状 ...")
    for start_idx in tqdm(range(0, args.num_samples, args.batch_size)):
        cur_bs = min(args.batch_size, args.num_samples - start_idx)

        octree, vq_indices = generate_one(
            model, device, batch_size=cur_bs,
            temperature=args.temperature, cfg_scale=args.cfg_scale,
        )

        for b in range(cur_bs):
            idx = start_idx + b
            output_path = os.path.join(args.output, f'{idx:04d}.obj')

            try:
                if args.method == 'vqvae' and vqvae_wrapper is not None:
                    export_mesh_vqvae(
                        vqvae_wrapper, octree, vq_indices,
                        output_path, resolution=args.resolution,
                    )
                else:
                    from src.utils.mesh import extract_mesh_from_octree
                    extract_mesh_from_octree(
                        octree, depth=model_cfg.depth_stop,
                        output_path=output_path, method='marching_cubes',
                    )
                print(f"  已保存: {output_path}")
            except Exception as e:
                print(f"  保存 {output_path} 出错: {e}")

    print(f"\n完成！结果已保存到 {args.output}")


if __name__ == '__main__':
    main()
