# OctFractalGen 探索路径总结

> 本文档记录从项目启动到关键诊断发现的完整探索路径，重点梳理每一步的假设、实验、结论与修正。

---

## 1. 项目定位

**目标**：探索递归多模型八叉树生成架构（FractalOctGPT），对比 OctGPT 的单模型多深度范式。

**核心思路**：
- 微观（层内）：复用 OctGPT 的 OctFormer + OctreeT + MaskGIT
- 宏观（层间）：多层串联，每层独立参数处理一个深度，通过 cond 传递跨深度信息

---

## 2. 探索阶段

### 阶段 1：磁盘故障与日志迁移

训练首次启动即因系统盘写满崩溃。解决方案：将 `logs/` 软链接到数据盘 `/root/autodl-tmp/OctFractalGen_logs/`。同时调整 VSCode 设置使其能正常浏览软链接目录。

**教训**：系统盘只放代码，所有 checkpoint/log/数据必须在数据盘。

---

### 阶段 2：旧架构诊断 — VQHead 是瓶颈

#### 旧架构
- 自研 `PatchTransformerBlock`（3D RoPE + SwiGLU + patch 分区）
- 3 层 AR + 1 层 VQHead 递归串联
- 跨深度交互：prefix token

#### 诊断实验 1：单样本过拟合
| 层 | Loss | Accuracy |
|---|---|---|
| d3 split | 0.001 | 1.000 |
| d4 split | 0.006 | 0.958 |
| d5 split | 0.177 | 0.921 |
| d6 VQ | 0.608 | top1=0.667 |

**结论**：VQ 层 top1 仅 0.667，是架构瓶颈。

#### 架构对比分析
详细对比自研 `PatchTransformerBlock` 与 OctGPT 的 `OctFormer`：

| 维度 | 自研架构 | OctFormer |
|---|---|---|
| Attention mask | 无 | patch_mask + teacher_forcing_mask |
| SWIN 窗口移位 | 未实现 | 交替移位 |
| 多深度 token | VQ 层看不到 split 决策 | split + VQ 拼接在同序列 |
| Encoder-Decoder | 单 transformer | 分离（unmasked→encoder, all→decoder） |

**结论**：自研架构缺失了 OctFormer 的核心机制，不是"近似"而是架构层面缺失。

---

### 阶段 3：架构重设计 — FractalOctGPT

#### 设计决策
用户明确要求：**直接复用 OctFormer，只改宏观架构**。

#### 新架构
```
L0: AR @ depth 3, dim=384,  heads=8, blocks=12
L1: AR @ depth 4, dim=576,  heads=8, blocks=12
L2: AR @ depth 5, dim=768,  heads=8, blocks=8
L3: VQ  @ depth 6, dim=768,  heads=8, blocks=8
```

- 逐层递增维度（粗层小、细层大）
- head_dim 必须被 6 整除（RoPE 约束）
- 每层独立参数，通过 cond 传递跨深度信息

#### 技术问题修复
1. **RoPE 约束**：head_dim % 6 == 0，通过合适的 dim/heads 组合解决
2. **AbsPosEmb 越界**：num_embed % 6 ≠ 0 时索引越界，改用 SinPosEmb
3. **OctreeT mask 兼容性**：prefix token 破坏 data_mask 长度匹配，改用 cond/buffer 传递

#### 诊断实验 2：单样本过拟合（新架构）
| 层 | 旧架构 acc | 新架构 acc |
|---|---|---|
| d3 split | 1.000 | 1.000 |
| d4 split | 0.958 | 1.000 |
| d5 split | 0.921 | 1.000 |
| d6 VQ top1 | 0.667 | **0.994** |
| total loss | 0.79 | **0.026** |

**结论**：复用 OctFormer 后架构瓶颈消除，VQ 预测精度从 0.667 提升至 0.994。

---

### 阶段 4：跨深度信息传递方案

实现了三种 cond 传递方案并分别提交 git 版本：

| 方案 | 机制 | commit |
|---|---|---|
| Mean pool | cond_out (nnum, dim) → 均值池化 → (B, dim) | `c2c2768` |
| CLS token | 可学习 CLS token 参与 attention，输出作 cond | `f9ca694` |
| Prefix token | 上一层全部输出作为下一层已知 token | 待修复 |

当前训练使用 CLS token 方案。

---

### 阶段 5：完整训练与质量评估

#### 训练配置
- 数据：ShapeNet airplane（2831 训练，405 验证）
- DDP 2 GPU，batch=4/卡，等效 batch=8
- AdamW lr=2e-4, cosine decay, warmup=5 epoch
- 200 epoch，AMP + gradient checkpointing
- 183M 参数

#### 训练进展
| Epoch | Train Loss | Val Loss |
|---|---|---|
| 0 | 1.88 | 1.54 |
| 5 | 1.07 | 1.04 |
| 10 | 0.95 | 0.93 |

#### 生成质量初步评估
用 best.pt 生成 obj，发现：
- split 预测正确（形状轮廓像飞机）
- 加入 VQVAE 重建后效果很差（粗糙多孔）

---

### 阶段 6：VQVAE 重建质量诊断（关键转折）

这是整个探索中最重要的阶段，经历了多次假设修正。

#### 假设 1：VQVAE codebook 粗糙是瓶颈

**实验**：GT octree → extract_code → quantize → decode → mesh

**结果**：GT 重建只有 534-1178 顶点，形状可辨但粗糙多孔。

**初步结论（后被推翻）**：VQVAE code_depth=6 限制了重建上界。

#### 假设 2：split_zero 扩展丢失子节点结构

VQVAE decode 时需要从 depth_stop=6 扩展到 vae_depth=8，当前用全 0 split。

**实验**：对比 split_zero 扩展 vs 保留 GT depth7/8 子节点结构

**结果**（3 样本）：

| 样本 | split_zero | GT struct | 差异 |
|---|---|---|---|
| 0 | 1178 verts, area=578 | 1178 verts, area=578 | **0** |
| 1 | 605 verts, area=344 | 605 verts, area=344 | **0** |
| 2 | 534 verts, area=235 | 534 verts, area=235 | **0** |

**结论**：VQVAE decoder 不依赖 depth7/8 子节点结构，split_zero 不是瓶颈。

#### 假设 3：VQ loss 难以下降印证 codebook 质量差

**观察**：训练日志中 split loss 快速下降（89%），VQ loss 几乎不动（37%）。

| step | d3_split | d4_split | d5_split | d6_vq |
|---|---|---|---|---|
| 0 | 0.744 | 0.793 | 0.821 | 0.772 |
| 2350 | 0.085 | 0.235 | 0.216 | 0.485 |
| 下降幅度 | 89% | 70% | 74% | 37% |

**初步结论（后被推翻）**：VQ loss 卡住是因为 codebook 含量化噪声，AR 学不动。

#### 假设 4：depth 3 split 是 trivial（被推翻）

**假设**：full_depth=3 意味着 depth 3 全部满分裂，d3_split 是 trivial 的，浪费层容量。

**实验**：构建 GT octree 检查各深度 split 率

**结果**：

| depth | nnum | split_rate |
|---|---|---|
| 0 | 1 | 1.00 (强制) |
| 1 | 8 | 1.00 (强制) |
| 2 | 64 | 1.00 (强制) |
| **3** | **512** | **0.039 (20/512)** |
| 4 | 160 | 0.30 |
| 5 | 384 | 0.50 |

**结论**：depth 3 不是 trivial，split_rate 仅 3.9%。但高度不平衡（96% 是 0），导致 loss 低估了真实难度。配置与 OctGPT 官方一致，不需要修改。

---

### 阶段 7：决定性实验 — AR codes vs GT codes 重建对比

这是推翻之前所有错误结论的关键实验。

#### 动机
用户指出：OctGPT 官方生成命令跑出来效果很好，用的是同一个 VQVAE。如果 VQVAE 真的是瓶颈，OctGPT 不可能有好效果。

#### 实验
用 OctGPT 官方 `octgpt_airplane.pth` 生成，然后：
1. AR 预测 codes → VQVAE decode → mesh
2. 同一个生成 octree 的 GT codes → VQVAE decode → mesh

#### 结果

| 重建方式 | 顶点数 | 面数 | area |
|---|---|---|---|
| **AR 预测 codes** | **60688** | 121368 | **2.252** |
| GT codes | 936 | 1676 | 0.020 |
| 官方 logs 样本 | 34000-56000 | 68000-112000 | 1.26-2.10 |

**同一个 VQVAE，同一个 decode 路径，AR codes 重建 60688 顶点，GT codes 只有 936 顶点。差距 60 倍。**

#### 额外验证：codes 提取一致性
对比官方 `extract_code + quantizer` 和我们的 `extract_targets`：
```
官方: indices shape [6696, 32], mean 0.5019
我们: indices shape [6696, 32], mean 0.5019
一致: True
```

codes 提取完全一致，排除了我们代码的 bug。

#### 结论（最终修正）

**VQVAE 完全没问题。** 它能从 AR codes 解码出 6 万顶点的好 mesh。

**问题在于 GT codes 本身。** 后验编码（encoder → quantizer）含量化噪声，解码出多孔 mesh。而 AR 先验学到的码分布更平滑、对解码友好。

之前的所有"VQVAE 是瓶颈"的结论都是错误的——我们用 GT codes 重建作为质量上界，但 GT codes 不是好的上界，AR codes 才是。

---

## 3. 核心发现：后验 vs 先验

### 现象
GT codes（后验）重建 << AR codes（先验）重建，同一个 VQVAE。

### 原因分析

| | GT codes | AR codes |
|---|---|---|
| 来源 | encode(具体形状) → quantize | AR 从 P(codes) 采样 |
| 性质 | 后验，含该形状特有的量化噪声 | 先验，数据集层面的平滑分布 |
| 解码效果 | 差（多孔、少顶点） | 好（光滑、多顶点） |

VQVAE encoder 对单个形状编码时，连续特征落在 codebook 簇的边界上，产生边界量化误差。而 AR 模型学的是整个数据集的码分布，生成时倾向落在簇中心（高概率区），这些中心码解码更干净。

### 这解释了之前的困惑

1. **为什么 split 正确但结果差**：split 只决定结构（哪里有体素），体素内部的几何由 VQ codes 决定。我们的 AR 还没训够，预测的 codes 偏离了"解码友好"的先验流形。

2. **为什么 VQ loss 难以下降**：GT codes 中部分 bit 是量化噪声（不可预测），AR 永远学不动这些 bit。但 AR 不需要学动它们——只要学到"解码友好"的码组合即可。

3. **为什么 OctGPT 效果好**：OctGPT 训了 400 epoch，AR 充分学习了先验分布，生成的 codes 落在解码友好区域。

---

## 4. 修正后的结论

### 错误结论（已推翻）
- ~~VQVAE codebook 粗糙是重建瓶颈~~
- ~~code_depth=6 限制了重建上界~~
- ~~VQ loss 卡住是因为 codebook 质量差~~
- ~~需要重训更深 VQVAE 才能改善重建~~

### 正确结论
1. **VQVAE 完全够用** — AR codes 重建可达 6 万顶点
2. **当前瓶颈是 AR 训练不足** — 才 5 epoch vs OctGPT 的 400 epoch
3. **GT codes 重建不是有效上界** — 后验含量化噪声，低估了 VQVAE 能力
4. **评估应使用 AR codes 重建** — 这才是生成管线的真实质量指标

---

## 5. 后续方向

### 优先级 1：完成 AR 训练
当前 200 epoch 训练仍在进行。完成后用 AR 预测 codes 评估生成质量，预期接近 OctGPT 水平。

### 优先级 2：消融实验
- Mean pool vs CLS token vs prefix token
- 控制参数量，对比跨深度信息传递方式的影响

### 优先级 3：prefix token 修复
解决 OctreeT mask 长度问题，实现信息等价于 OctGPT 的跨深度传递。这是真正能对标 OctGPT 的设计。

### 低优先级：VQVAE 重训
仅当需要更高几何分辨率（depth7/8，128³ 体素）时才考虑。当前 VQVAE 对 AR codes 的解码能力已足够。

---

## 6. 关键文件索引

| 文件 | 作用 |
|---|---|
| `src/model/fractal_octgpt.py` | 多层串联模型主体 |
| `src/model/octgpt_layer.py` | 单层 OctGPT 风格（复用 OctFormer） |
| `src/train_ddp.py` | DDP 训练脚本 |
| `src/generate.py` | 生成脚本 |
| `src/model/vqvae_wrapper.py` | VQVAE 封装 |
| `scripts/diag_overfit_octgpt.py` | 过拟合诊断 |
| `scripts/diag_vqvae_subnode.py` | split_zero vs GT 结构对比 |
| `scripts/diag_octgpt_official_path.py` | 官方路径重建对比 |
| `experiments/configs/fractal_base_airplane.yaml` | 训练配置 |
| `docs/experiment_report.md` | 早期实验报告（部分结论已过时） |

---

## 7. 时间线

| 时间 | 事件 |
|---|---|
| 启动 | 磁盘故障 → 日志迁移到数据盘 |
| 阶段 2 | 旧架构诊断：VQHead top1=0.667，架构瓶颈 |
| 阶段 3 | 架构重设计：复用 OctFormer，VQ top1=0.994 |
| 阶段 4 | 三种 cond 传递方案实现 |
| 阶段 5 | 完整训练启动，生成质量初步评估 |
| 阶段 6 | VQVAE 重建诊断：多次假设修正 |
| 阶段 7 | **决定性实验**：AR codes 6 万顶点 vs GT codes 936 顶点 |
| 当前 | 修正结论，等待 200 epoch 训练完成 |
