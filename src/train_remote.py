#!/usr/bin/env python3
"""远端版本训练入口（直接复用 OctGPT OctFormer）。

用法: python -m src.train_remote --config <yaml> --device cuda
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 在 import train 之前替换模块
import src.model.fractal_octree_remote
sys.modules['src.model.fractal_octree'] = sys.modules['src.model.fractal_octree_remote']

from src.train import main

if __name__ == '__main__':
    main()
