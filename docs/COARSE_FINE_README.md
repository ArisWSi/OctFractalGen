# CoarseFineOctGPT — Coarse-Fine Grouped FractalOctGPT

将逐层 FractalOctGPT 改造为 coarse-fine 分组结构，用 coarse hidden 替换 fine OctGPT 中 depth 3/4 的原生 embedding，验证 full prefix 能否缩小与 OctGPT 的生成质量差距。

## 架构

```
Coarse OctGPT: depth_list = [3, 4]   (联合建模 d3/d4 split)
        ↓ hidden injection (proj + LayerNorm)
Fine OctGPT:   depth_list = [3, 4, 5, 6]  (d3/d4 embedding 被 prefix 替换)
                                           (只预测 d5 split + d6 VQ)
```

核心设计（不重写 OctFormer/OctreeT/MaskGIT）：
- fine 模型构建原生 `[d3,d4,d5,d6]` 序列，token 数不增加
- d3/d4 的 embedding 被 `LayerNorm(Linear(coarse_hidden))` 替换
- OctreeT 原生 teacher-forcing mask (`cond="le"`) 天然满足：fine(d5/d6) 可 attend prefix(d3/d4)，prefix 不能 attend fine
- d3/d4 不参与 fine loss，只在 coarse 模型计算 loss

## 环境假设

- Python 3.10+, PyTorch 2.5+, CUDA 12.4
- `ocnn`, `ognn` (OctGPT 依赖)
- conda 环境 `octgpt`（含所有依赖）

```bash
source activate octgpt
```

## 数据路径

```text
/root/autodl-tmp/ShapeNet/processed              # ShapeNet octree 数据
/root/autodl-tmp/OctGPT/ShapeNet/train_airplane.txt
/root/autodl-tmp/OctGPT/ShapeNet/val_airplane.txt
/root/autodl-tmp/OctGPT/vqvae_large_im5_uncond_bsq32.pth   # VQVAE checkpoint
```

## 文件清单

| 文件 | 说明 |
|------|------|
| `src/model/grouped_fractal_octgpt.py` | `GroupedOctGPTLayer` + `CoarseFineOctGPT` wrapper |
| `src/model/octgpt_layer.py` | 单深度 OctGPT 层（未改动，fractal_octgpt 用） |
| `src/config.py` | ModelConfig 增加 `coarse`/`fine`/`detach_prefix` 字段 |
| `src/train_ddp.py` | DDP 训练入口，注册 `--model coarse_fine_octgpt`，分层日志 |
| `src/generate.py` | 生成入口，`--model_type coarse_fine_octgpt` |
| `src/utils/mesh.py` | octree→voxel→mesh 导出（已修复 permute bug） |
| `experiments/configs/coarse_fine_octgpt_airplane.yaml` | airplane 配置 |
| `scripts/diag_overfit_coarse_fine.py` | 单样本过拟合诊断 + mesh 导出 |
| `scripts/test_coarse_fine_generate.py` | 生成流程验证 |

## 单样本过拟合诊断

完整训练前必须先通过单样本过拟合（指南 Milestone 4）：

```bash
cd /root/OctFractalGen && source activate octgpt

# 完整模型
python -m scripts.diag_overfit_coarse_fine --max_steps 500 --log_interval 25

# 带 mesh 导出
python -m scripts.diag_overfit_coarse_fine --max_steps 500 --log_interval 25 \
    --export logs/coarse_fine_overfit

# 小模型快速验证（显存不足时）
python -m scripts.diag_overfit_coarse_fine --max_steps 300 --tiny
```

验收标准：
```
d3 split acc = 1.000
d4 split acc = 1.000
d5 split acc = 1.000
d6 VQ top1  >= 0.99
total loss  -> 接近 0
```

导出的 mesh（`--export` 指定目录）：
- `gt_split.obj` — GT 八叉树 split 结构（体素方块）
- `gt_vqvae.obj` — GT VQ codes 经 VQVAE 重建
- `gen_split.obj` — 模型生成的 split 结构
- `gen_vqvae.obj` — 模型生成 VQ codes 经 VQVAE 重建

## 训练

DDP 双卡训练：

```bash
cd /root/OctFractalGen && source activate octgpt

torchrun --nproc_per_node=2 -m src.train_ddp \
    --config experiments/configs/coarse_fine_octgpt_airplane.yaml \
    --model coarse_fine_octgpt \
    --batch_size 4
```

checkpoint 保存：
- `latest.pt` — 最新
- `best.pt` — 验证最优
- `checkpoint_epoch{XXX:03d}.pt` — 每 20 epoch 快照

训练日志（TensorBoard + train.log）记录分层指标：
```
train/loss, train/loss_coarse, train/loss_fine
train/coarse_split_acc_d3, train/coarse_split_acc_d4
train/fine_split_acc_d5, train/fine_vq_top1, train/fine_vq_top5
```

## 生成

```bash
cd /root/OctFractalGen && source activate octgpt

python -m src.generate \
    --checkpoint logs/coarse_fine_octgpt_airplane/best.pt \
    --model_type coarse_fine_octgpt \
    --output logs/coarse_fine_octgpt_airplane/generate \
    --num_samples 16 \
    --temperature 1.0
```

生成流程：
1. coarse sample → 生成 d3/d4 split，扩展 octree
2. coarse forward(gen tokens, return_hidden) → prefix hidden
3. `prefix = LayerNorm(Linear(coarse_hidden))`
4. fine sample → d3/d4 embedding 用 prefix 替换，生成 d5 split + d6 VQ
5. VQVAE decode → Neural MPU → Marching Cubes → OBJ

## 常见错误

| 症状 | 排查 |
|------|------|
| 单样本 split 过拟合但 VQ top1 上不去 | coarse hidden 与 fine d3/d4 顺序未对齐 |
| fine loss 异常 | fine d3/d4 仍参与 loss（应 `loss_depths=[5,6]`） |
| validation 正常但 sample 崩坏 | 生成时用了 GT hidden 而非 generated tokens 的 hidden |
| split mesh 空白 | `octree2voxel` 缺 `permute(0,4,1,2,3)` |
| OOM | 用 `--tiny` 或减小 batch_size/blocks |

## 与 OctGPT baseline 对比

| 模型 | 结构 | cond | 目标 |
|------|------|------|------|
| OctGPT official | d3-d4-d5-VQ 单模型 | same sequence | upper reference |
| 当前 FractalOctGPT | d3→d4→d5→VQ 逐层 | CLS/mean pool | baseline |
| CoarseFineOctGPT | d3-d4 → d5-VQ 分组 | full prefix | 主实验 |

关键指标：train/val loss, vq_top1, sample mesh vertices, surface area, 视觉质量。
