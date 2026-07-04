#!/bin/bash
# 启动远端版本（OctGPT OctFormer 直接复用）训练
# 用法: bash run_remote.sh [GPU_ID] [CONFIG]

GPU=${1:-0}
CONFIG=${2:-experiments/configs/fractal_base_airplane_remote.yaml}

cd /root/OctFractalGen
source activate octgpt

CUDA_VISIBLE_DEVICES=$GPU python -c "
import sys, os
sys.path.insert(0, 'extern/octgpt')
# 临时替换 import：用远端版本
import src.model.fractal_octree_remote as fo
sys.modules['src.model.fractal_octree'] = fo

from src.train import main
sys.argv = ['train', '--config', '$CONFIG', '--device', 'cuda']
main()
"
