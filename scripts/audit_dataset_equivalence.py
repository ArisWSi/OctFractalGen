"""数据等价性审计: 比较我们的 dataset 与官方 OctGPT dataset 的 octree_gt.

用法:
  python -m scripts.audit_dataset_equivalence
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_octgpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'extern', 'octgpt')
if _octgpt not in sys.path:
    sys.path.insert(0, _octgpt)

from src.config import DataConfig
from src.data.shapenet import get_shapenet_dataset, collate_func


def main():
    data_cfg = DataConfig(
        location='/root/autodl-tmp/ShapeNet/processed',
        filelist='/root/autodl-tmp/OctGPT/ShapeNet/train_airplane.txt',
        batch_size=1, num_workers=0, depth=8, full_depth=3,
        points_scale=1.0, distort=False,
    )

    ds, coll = get_shapenet_dataset(data_cfg)
    print(f"dataset size: {len(ds)}")
    print(f"points_scale: {data_cfg.points_scale}")
    print(f"depth: {data_cfg.depth}, full_depth: {data_cfg.full_depth}")
    print(f"distort: {data_cfg.distort}")
    print()

    # 检查前 3 个样本
    for i in range(min(3, len(ds))):
        sample = ds[i]
        batch = coll([sample])
        octree = batch['octree_gt']
        print(f"=== sample {i} ===")
        print(f"  octree depth: {octree.depth}, full_depth: {octree.full_depth}")
        print(f"  batch_size: {octree.batch_size}")
        for d in range(data_cfg.full_depth, data_cfg.depth + 1):
            print(f"  nnum[{d}] = {octree.nnum[d]}")
        # 检查 keys 是否存在
        for d in range(data_cfg.full_depth, data_cfg.depth + 1):
            keys = getattr(octree, 'keys', None)
            if keys is not None and d < len(keys):
                print(f"  keys[{d}] shape: {keys[d].shape}, first: {keys[d][:4]}")
        print()

    # 比较两个样本的 octree 是否一致 (确定性)
    s0 = coll([ds[0]])['octree_gt']
    s0_again = coll([ds[0]])['octree_gt']
    print("=== 确定性检查 (同样本两次加载) ===")
    consistent = True
    for d in range(data_cfg.full_depth, data_cfg.depth + 1):
        if s0.nnum[d] != s0_again.nnum[d]:
            print(f"  ✗ nnum[{d}] 不一致: {s0.nnum[d]} vs {s0_again.nnum[d]}")
            consistent = False
        else:
            print(f"  ✓ nnum[{d}] = {s0.nnum[d]}")
    print(f"  确定性: {'✓' if consistent else '✗'}")


if __name__ == '__main__':
    main()
