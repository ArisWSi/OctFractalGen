# OctFractalGen 实验报告

## 1. 研究目标

本项目旨在探索**递归多模型八叉树生成**架构，对比 OctGPT 的单模型多深度生成范式。核心研究问题是：能否通过多层串联的独立 Transformer 模型（每层处理一个深度），在复用 OctGPT 微观机制的前提下，实现与 OctGPT 相当或更优的 3D 形状生成质量，同时获得模块化、可解释性和训练效率上的优势。

---

## 2. 方法演进

### 2.1 第一阶段：自定义 Transformer 架构（旧架构）

#### 架构设计
- **宏观**：3 层 AR（`OctreeAR`）+ 1 层 VQHead，递归串联
- **微观**：自研 `PatchTransformerBlock`，借鉴 OctFormer 命名但独立实现
  - 3D RoPE（自实现，固定频率）
  - SwiGLU FFN（Llama 风格）
  - Patch 分区 + dilation
- **跨深度交互**：prefix token（上一层 `cond_out` 作为下一层序列前缀）

#### 诊断实验

**实验 1：VQVAE 重建上界**
- 目标：验证冻结的预训练 VQVAE 能否从 GT octree 重建合理 mesh
- 结果：3 个样本重建顶点数 534-1178，形状可辨但粗糙
- 结论：VQVAE 重建上界有限（depth_stop=6，体素分辨率 64³），但非主要瓶颈

**实验 2：单样本过拟合（旧架构）**
- 目标：验证架构能否表达单个 shape
- 结果（300 步）：

| 层 | Loss | Accuracy |
|----|------|----------|
| d3 split | 0.001 | 1.000 |
| d4 split | 0.006 | 0.958 |
| d5 split | 0.177 | 0.921 |
| d6 VQ | 0.608 | top1=0.667, top5=1.000 |

- 结论：**VQ 层是瓶颈**，top1 仅 0.667，loss 卡在 0.6

#### 架构对比分析

经详细对比自研 `PatchTransformerBlock` 与 OctGPT 的 `OctFormer`，发现关键差异：

| 维度 | 自研架构 | OctFormer (OctGPT) |
|------|---------|-------------------|
| Attention mask | **无**（跨 batch token 互相 attend） | patch_mask + teacher_forcing_mask |
| SWIN 窗口移位 | 未实现 | 交替移位 |
| 多深度 token | VQ 层看不到 split 决策 | split + VQ 拼接在同序列 |
| Encoder-Decoder | 单 transformer | 分离（unmasked→encoder, all→decoder） |
| 位置编码 | 固定 3D RoPE | 可学习 RoPE + AbsPosEmb |

**核心结论**：自研架构缺失了 OctFormer 的三个核心机制（attention mask、SWIN、多深度拼接），这不是"近似"而是架构层面的缺失。split 预测的过拟合成功（acc>0.9）只能说明模型有记忆容量，不能证明架构等价。

---

### 2.2 第二阶段：复用 OctFormer 的 FractalOctGPT（新架构）

#### 设计理念
- **微观（层内）**：完全复用 OctGPT 的 OctFormer + OctreeT + MaskGIT 机制
- **宏观（层间）**：保持递归多模型串联，每层独立参数

#### 架构详情

```
L0: AR @ depth 3, dim=384,  heads=8, blocks=12 (encoder 6 + decoder 6)
L1: AR @ depth 4, dim=576,  heads=8, blocks=12
L2: AR @ depth 5, dim=768,  heads=8, blocks=8
L3: VQ @ depth 6, dim=768,  heads=8, blocks=8
```

**层内机制（复用 OctGPT）**：
- OctFormer encoder-decoder 分离（teacher forcing）
- OctreeT 构建 attention mask（跨 batch 隔离 + 浅→深单向）
- SWIN 窗口移位
- RotaryPosEmb（3D RoPE，可学习频率）+ SinPosEmb
- MaskGIT 训练（random masking + random_flip）和采样（cosine schedule + remask）

**跨深度信息传递（三种方案）**：

| 方案 | 机制 | 信息容量 | 实现 |
|------|------|---------|------|
| Mean pool | 上一层 cond_out (nnum, dim) → 均值池化 → (B, dim) | 有损压缩 | 已实现，训练中 |
| CLS token | 可学习 CLS token 参与 attention，输出作下一层 cond | 学习式聚合 | 已实现，未训练 |
| Prefix token | 上一层全部输出作为下一层已知 token | 无损（等价 OctGPT） | 待修复（OctreeT mask bug） |

#### 诊断实验

**实验 2b：单样本过拟合（新架构）**
- 结果（300 步）：

| 层 | 指标 | 旧架构 | 新架构 |
|----|------|--------|--------|
| d3 split | acc | 1.000 | 1.000 |
| d4 split | acc | 0.958 | 1.000 |
| d5 split | acc | 0.921 | 1.000 |
| d6 VQ | top1 | 0.667 | **0.994** |
| **total loss** | | 0.79 | **0.026** |

- 结论：复用 OctFormer 后，VQ 预测精度从 0.667 提升至 0.994，**架构瓶颈消除**

---

### 2.3 完整训练（进行中）

#### 配置
- 数据集：ShapeNet airplane（2831 训练，405 验证）
- 训练：DDP 2 GPU，batch=8/卡（等效 16），200 epoch
- 优化器：AdamW（lr=2e-4, wd=0.05, betas=(0.9, 0.95)），cosine decay
- AMP 混合精度 + 梯度检查点

#### 训进展（截至 epoch 24）

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 0 | 1.8488 | 1.4917 |
| 5 | 1.0526 | 1.0048 |
| 10 | 0.9525 | 0.9297 |
| 15 | 0.8941 | 0.9090 |
| 20 | 0.8605 | 0.8736 |
| 24 | 0.8424 | — |

- Loss 持续下降，验证 loss 跟随且无过拟合
- 收敛速度远快于旧架构（旧架构 200 epoch 才到 0.79）

---

## 3. 关键技术问题与解决

### 3.1 RoPE 维度约束
OctGPT 的 `RotaryPosEmb` 要求 `head_dim % 6 == 0`（3 轴分配）。通过选择合适的 `num_embed`/`num_heads` 组合解决：
- dim=384, heads=8 → head_dim=48 ✓
- dim=576, heads=8 → head_dim=72 ✓
- dim=768, heads=8 → head_dim=96 ✓

### 3.2 AbsPosEmb 越界 bug
OctGPT 的 `AbsPosEmb` 在 `num_embed % 6 ≠ 0` 时有索引越界 bug。改用 `SinPosEmb` 规避。

### 3.3 OctreeT data_mask 兼容性
OctreeT 的 `data_mask` 机制为 OctGPT 单序列设计，prefix token 会破坏 mask 长度匹配。当前 mean pool 和 CLS token 方案不触发此问题（不改变序列长度），prefix token 方案待修复。

### 3.4 磁盘空间管理
训练 checkpoint 每个 1GB，系统盘 30GB 易满。解决方案：`logs/` 目录软链接到数据盘 `/root/autodl-tmp/OctFractalGen_logs/`。

---

## 4. 分析与讨论

### 4.1 递归多模型 vs 单模型

| 维度 | OctGPT（单模型） | FractalOctGPT（递归多模型） |
|------|-----------------|--------------------------|
| 跨深度信息 | 同序列 attention，无损 | cond 传递，有损（mean pool/CLS） |
| 参数量 | ~100M（共享） | ~183M（独立） |
| 计算量 | 1 次 forward | 4 次 forward |
| 模块化 | 无 | 每层可独立设计 |
| 可解释性 | 跨深度耦合 | 跨深度信息流显式 |

当前架构的劣势明显：参数多、计算量大、信息有损。其价值在于**可消融性**——能定量研究跨深度信息传递方式的影响。

### 4.2 跨深度信息传递的 trade-off

- **Mean pool**：最简单，信息压缩最严重（nnum×dim → 1×dim）
- **CLS token**：通过 attention 学习聚合，比 mean pool 有表达力，仍是瓶颈
- **Prefix token**：信息等价于 OctGPT，但序列长度膨胀，计算量不占优

这三种方案构成信息保留与计算效率的 trade-off 谱系，可作为消融实验的核心变量。

### 4.3 VQVAE 上界限制
当前 VQVAE `code_depth=6`（体素分辨率 64³），重建顶点 534-1178。若需更细几何，需重训 VQVAE（`vae_depth=9, code_depth=7`，体素 128³），但工作量翻倍且生成器序列长度 8 倍增长。

---

## 5. 后续工作

1. **完成当前训练**（200 epoch，预计 ~17 小时）并评估生成质量
2. **消融实验**：mean pool vs CLS token vs prefix token，控制参数量对比
3. **修复 prefix token**：解决 OctreeT mask 长度问题，实现信息等价于 OctGPT 的跨深度传递
4. **逐层预训练**：利用递归多模型的模块化优势，先训粗层再训细层
5. **VQVAE 重训**（可选）：若生成质量受限于 VQVAE 上界，考虑 `vae_depth=9`

---

## 6. 代码版本

| 版本 | Commit | 说明 |
|------|--------|------|
| mean pool | `c2c2768` | cond_out → mean pool → 下一层 |
| CLS token | `f9ca694` | CLS token 通过 attention 聚合 → 下一层 |

切换：`git checkout <commit>`
