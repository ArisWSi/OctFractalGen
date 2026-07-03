# 3D 分形自回归模型 (3D Fractal Autoregressive Model)

## TL;DR — 现状与目标

### 我们要证明什么

将 FractalGen 的**递归多模型范式**引入 3D 八叉树生成，替代 OctGPT 的迭代单模型范式。

**核心主张**: 递归多模型比迭代单模型**更高效**——在相同生成质量下大幅降低计算成本，并能扩展到 OctGPT 无法触及的八叉树深度。

**预期证据链**:

| 实验阶段 | 对比 | 预期结论 |
|---------|------|---------|
| **depth=6 公平对比**（相同参数量、相同 VQ-VAE） | vs OctGPT | 质量持平（1-NNA ≈ 50%），速度 2-3× 快，显存减半 |
| **消融实验** | AR vs MAR, 空间邻居数, 容量分配 | 哪些设计选择关键 |
| **扩展实验** | depth=7,8（OctGPT 不可运行） | 我们的方法能到更高几何精度 |

**我们不在 depth=6 就宣称质量超越**——OctGPT 的跨深度 attention 提供了更强的层间信息流。速度+扩展性是主攻方向，质量是防守底线。

### 实验框架

```
实验定义:  config/{exp_name}.yaml     ← 所有超参数集中管理
数据准备:  scripts/preprocess.py       ← ShapeNet OBJ → pointcloud.npz
训练:     src/train.py --config {cfg}  ← 自动保存配置快照 + TensorBoard
生成:     src/generate.py --ckpt {ckpt}
评测:     src/evaluate.py --ckpt {ckpt}
         ├── 采样 2048 表面点 → pairwise CD/EMD 矩阵
         ├── 1-NNA, COV, MMD (几何指标)
         └── 多样性直方图 (mode collapse 检测)
```

评测指标完全对齐 OctGPT 论文：1-NNA（分类准确率，接近 50% 为佳）、COV（覆盖率）、MMD（最小匹配距离）、Diversity（多样性）。

### 当前进展

| 项目 | 状态 |
|------|------|
| 架构代码 (`transformer`, `ar_octree`, `fractal_octree`) | ✅ 完成 |
| VQ-VAE 接口 (`vqvae_wrapper`) | ✅ 完成 |
| 训练/推理脚本 (`train.py`, `generate.py`) | ✅ 完成 |
| 数据管线 | ❌ ShapeNet 55 类 LFS zip 已下载，OBJ 未预处理为 pointcloud.npz |
| 评测管线 (`evaluate.py`, `metrics.py`) | ❌ 待开发 |
| 实验配置 (`configs/`) | ❌ 待建立 |
| 训练运行 | ❌ 尚未开始 |

**阻塞项**: 数据预处理（OBJ → pointcloud.npz）+ 评测脚本编写。

**远端环境**: RTX 3090 24GB, PyTorch 2.5.1+cu124, ocnn 2.3.2。磁盘 50G（剩余 ~9G）。

### 执行路线

```
P0: scripts/preprocess.py  →  打通数据管线
P0: src/utils/metrics.py   →  移植 CD/EMD/1-NNA/COV/MMD
P0: src/evaluate.py        →  评测入口
P0: configs/fractal_tiny_airplane.yaml → 首个实验配置
P1: 跑通 airplane tiny 训练 + 评测
P1: configs/octgpt_baseline.yaml → 复现 baseline
P2: depth=6 公平对比实验
P3: 消融 + 扩展实验
```

---

## 项目定位

在抽象层面，FractalGen（2D 四叉树 Transformer）和 OctGPT（3D 八叉树 Transformer）都是**多叉树 Transformer**——通过层级树结构将自注意力的 $O(N^2)$ 复杂度降到 $O(N \log N)$。但它们的**实现范式**存在本质差异：

| 维度 | FractalGen | OctGPT |
|---|---|---|
| 层级处理 | **递归多模型**：每层独立 Generator，递归嵌套 | **迭代单模型**：一个 OctFormer 处理所有深度 |
| 层间信息传递 | **显式条件向量**（5 个空间邻居） | **隐式跨深度 attention**（teacher-forcing mask） |
| 层内生成 | 因果 AR（逐 token + KV-Cache）或 MAR（masked） | 仅 masked prediction（cosine schedule） |
| 因果性保证 | 天然保证（递归结构） | 需要精心设计的 attention mask |
| 模型分配 | 粗尺度重模型，细尺度轻模型 | 所有尺度共用一个模型容量 |
| 序列长度 | 每层仅含当前尺度 token | 所有尺度 token 拼接 |

**本项目核心问题：将 FractalGen 的递归多模型范式引入 3D 八叉树生成，替代 OctGPT 的单模型迭代范式，探索其在表达能力、计算效率、生成质量上的优劣。**

同时探索多个派生研究方向。

---

## 核心参考论文

### 1. Fractal Generative Models (FractalGen)
- **论文**: "Fractal Generative Models", Li et al., arXiv:2502.17437, 2025
- **代码**: `extern/fractalgen/`
- **本质**（周弈帆博客解读）：一种多叉树视觉 Transformer，不是"新生成范式"而是"新 Transformer 架构"
- **关键架构**:
  - `FractalGen` (`models/fractalgen.py`): 递归模型，含 generator (AR/MAR) + next_fractal (下层 FractalGen 或 PixelLoss)
  - **AR** (`models/ar.py`): 因果自回归 + 2D RoPE + KV-Cache
  - **MAR** (`models/mar.py`): 随机 mask 训练 + cosine schedule 解码
  - **PixelLoss** (`models/pixelloss.py`): 最终层，256-entry codebook 逐像素预测 RGB
  - 多尺度: 256→16→4→1 (4层) 或 64→4→1 (3层)
  - 层间传递 5 个空间条件: middle, top, right, bottom, left
  - **复杂度**: $O(N \log N)$，每元素只与 $O(\log N)$ 个节点交互

### 2. OctGPT: Octree-based Multiscale Autoregressive Models for 3D Shape Generation
- **论文**: "OctGPT", Wei et al., SIGGRAPH 2025, arXiv:2504.09975
- **代码**: `extern/octgpt/`
- **核心思想**: 八叉树 + Masked Autoregressive 建模
- **关键架构**:
  - VQ-VAE: Encoder(3D CNN on octree) → Binary Spherical Quantizer → Decoder(Dual Octree GNN + MPU)
  - OctGPT (`models/octgpt.py`): 深度循环生成 split + VQ token
  - OctFormer (`models/octformer.py`): 八叉树 Transformer，patch-wise attn + SWIN + teacher-forcing mask
  - 生成: init octree(depth=3) → 逐深度 masked prediction → VQ-VAE decoder → mesh
  - 支持无条件/类别/图像/文本四种条件生成

---

## 研究方向

### 方向 1（主线）：递归多模型八叉树生成 (Recursive Multi-Model Octree Generation)

**核心问题**：将 FractalGen 的递归范式引入 3D，每层用独立的 Generator，对比 OctGPT 的单模型迭代范式。

**架构设计**：
```
3D_FractalGen(full_depth=2, depth_stop=6):
  Level 0 (depth 2→3):
    Seq = 1 根节点 → 预测 8 个子节点的 split
    Generator: 大 AR/MAR Transformer (e.g., 32 blocks, 16 heads)
    Condition: class_embedding (或 unconditional)
    → 输出条件向量传给 Level 1

  Level 1 (depth 3→4):
    Seq = ≤8 节点 → 预测 ≤64 个子节点的 split
    Generator: 中等 AR/MAR Transformer (e.g., 16 blocks, 8 heads)
    Condition: Level 0 输出的 per-node 特征

  Level 2 (depth 4→5):
    Seq = ≤64 节点 → 预测 ≤512 个子节点的 split
    Generator: 轻量 AR/MAR Transformer (e.g., 8 blocks, 4 heads)

  Level 3 (depth 5→6, final):
    Seq = ≤512 节点 → 几何特征预测
    GeoHead: MLP → occupancy 或 VQ codebook matching
    等价于 2D FractalGen 的 PixelLoss
```

**关键研究问题**：
- 递归多模型是否比迭代单模型生成质量更好？在什么尺度下差异显著？
- 内存效率：各层峰值内存 vs OctGPT 的全深度拼接
- 训练稳定性：多层联合训练 vs 逐层训练

**实验设计**：
- Baseline: OctGPT unconditional airplane (control 组)
- 实验组: 3D FractalGen 递归版，使用相同 VQ-VAE backbone
- 控制变量: 总参数量尽量接近
- 指标: Chamfer Distance, F-Score, 生成速度, GPU memory

### 方向 2：八叉树上的因果自回归生成 (Causal AR with Morton-order KV-Cache)

**核心问题**：OctGPT 只支持 masked prediction（类 BERT），不支持真正的因果自回归。引入 FractalGen AR Generator 的逐 token 生成 + KV-Cache 到 3D 八叉树。

**关键设计**：
- **Morton (Z-order) 编码**定义八叉树节点的因果顺序：同一深度内按 Z-order，保证空间局部性
- **3D KV-Cache**：不同于 2D grid，八叉树各深度节点数不同，需要 depth-aware cache 管理
- **3D RoPE**：将 `precompute_freqs_cis_2d` 拓展到 xyz 三维

```python
def precompute_freqs_cis_3d(grid_size, n_elem, base=10000):
    third_dim = n_elem // 3
    freqs = 1.0 / (base ** (torch.arange(0, third_dim, 2)[:third_dim//2].float() / third_dim))
    t = torch.arange(grid_size)
    freqs = torch.outer(t, freqs)
    freqs_grid = torch.concat([
        freqs[:, None, None, :].expand(-1, grid_size, grid_size, -1),  # x
        freqs[None, :, None, :].expand(grid_size, -1, grid_size, -1),  # y
        freqs[None, None, :, :].expand(grid_size, grid_size, -1, -1),  # z
    ], dim=-1)
    ...
```

**研究问题**：
- Morton-order causal AR vs masked prediction：哪个生成质量更好？
- KV-Cache 在稀疏八叉树上的加速比
- AR 的 teacher-forcing vs MAR 的 random masking 训练稳定性

### 方向 3：层自适应架构分配 (Level-Adaptive Architecture)

**核心问题**：粗尺度（全局结构）和细尺度（局部细节）需要不同容量的模型吗？

**实验设计**：
- **Uniform**：所有层相同 Transformer（对照，类似 OctGPT 但递归）
- **Decreasing**：深度 ↑ → blocks ↓, heads ↓（FractalGen 默认策略）
- **Increasing**：深度 ↑ → blocks ↑, heads ↑（细尺度细节多，需要更大容量？）
- **Learned**：用 NAS 搜索最优配置

**研究问题**：
- 粗尺度全局结构的建模难度 vs 细尺度局部细节的建模难度
- 不同形状类别（airplane 细长 vs car 方正）是否需要不同的容量分配？

### 方向 4：跨层空间条件机制 (Cross-Level Spatial Conditioning)

**核心问题**：FractalGen 2D 用 5 个邻居条件，3D 八叉树需要几个？能否替代 OctGPT 的跨深度 attention？

**方案**：
| 方案 | 邻居数 | 说明 |
|---|---|---|
| 7-neighbor | 7 | 6 个面邻居 + 自身 (center) |
| 19-neighbor | 19 | 面+边邻居 + 自身 |
| 27-neighbor | 27 | 完整 3×3×3 邻域 |
| Attention-based | 可变 | 学习哪些邻居重要 |

**研究问题**：
- 简单 7-neighbor 条件 vs OctGPT 的 teacher-forcing cross-depth attention
- 空间条件能否解决八叉树节点间的边界一致性问题？（类比 FractalGen 的 "临近图块生成" trick）
- 稀疏八叉树中邻居可能不存在——如何处理 missing neighbors？

### 方向 5：占用率直出简化管线 (Occupancy-Only Pipeline)

**核心问题**：不依赖 VQ-VAE，直接预测八叉树叶子节点的 binary occupancy，用 Marching Cubes 提取 mesh。

**架构**：
- 最终层 GeoHead 输出：`sigmoid(x) → occupancy_probability`
- 损失：Binary Cross-Entropy
- 推理：threshold → occupancy bits → marching cubes → mesh

**优点**：
- 无需训练/加载 VQ-VAE，代码量大幅减少
- 训练快，适合快速迭代验证架构想法
- 可直接与 occupancy-based baseline 比较

**缺点**：
- 几何精度不如 VQ-VAE + SDF decoder

**定位**：方向 1 的快速验证版，成熟后再接入 VQ-VAE。

### 方向 6：稀疏八叉树的自回归建模 (Sparse Octree AR)

**核心问题**：2D 图像是 dense grid（所有 patch 都存在），3D 形状是 sparse octree（大部分空间为空）。FractalGen 假设固定序列长度，如何适配可变长度的稀疏八叉树？

**方案**：
- **Padding**：补到最大节点数（浪费但简单）
- **Dynamic batching**：类似 OctGPT，将各深度节点分别 batch
- **Depth-conditional sequence length**：每层序列长度由上一层 split 结果决定
- **Learnable "empty" embedding**：空节点用可学习的 embedding 替代

**研究问题**：
- 稀疏八叉树的节点数方差很大（airplane 机翼 vs 机身），如何保持训练稳定？
- Dense assumption 对生成质量的影响

### 方向 7：混合生成策略 (Hybrid AR+MAR Generation)

**核心问题**：不同深度适合不同的生成策略。浅层（节点少）适合 AR（因果逐 token），深层（节点多）适合 MAR（并行 masked prediction）。

**设计**：
```
3D_FractalGen:
  Level 0 (depth 2→3, 1 node):    AR  (因果, KV-Cache)
  Level 1 (depth 3→4, ≤8 nodes):  AR  (因果, KV-Cache)  
  Level 2 (depth 4→5, ≤64 nodes): MAR (masked, cosine schedule)
  Level 3 (depth 5→6, ≤512):      MAR (masked, cosine schedule)
```

**研究问题**：
- AR→MAR 的切换时机：序列长度阈值
- 混合策略 vs 纯 AR vs 纯 MAR 的质量/速度 tradeoff

### 方向 8：渐进式训练 (Progressive Training)

**核心问题**：直接训练深度 6 的八叉树内存开销大。借鉴 ProGAN，逐层渐进训练。

**方法**：
1. 先训练 Level 0 (depth 2→3)，冻结
2. 加 Level 1 (depth 3→4)，冻结 Level 0，训练 Level 1
3. 逐层添加直到目标深度
4. 最后全部解冻联合微调

**优点**：训练快，内存友好，可训练更深八叉树
**风险**：逐层贪婪训练可能导致次优解

### 方向 9：多条件 3D 生成 (Multi-Condition Generation)

**核心问题**：将 FractalGen 的 Classifier-Free Guidance (CFG) 拓展到 3D，支持更多条件类型。

- **Class-conditioned**：类别嵌入注入 Level 0，CFG 引导
- **Text-conditioned**：CLIP text embedding → cross-attention 注入各层
- **Image-conditioned**：DINOv2/ViT image embedding → cross-attention
- **Shape completion**：部分八叉树作为条件，生成缺失部分
- **Sketch-conditioned**：2D 草图 → 3D 形状

### 方向 10：理论分析 (Theoretical Analysis)

**复杂度对比**（假设深度范围 [D_low, D_high]，节点数 N_d = O(8^d)）：

| 方法 | 每层复杂度 | 总复杂度 |
|---|---|---|
| OctGPT (单模型) | $O((\sum N_d)^2)$ per forward | $O(64^{D_{high}})$ |
| OctGPT (teacher-forcing) | $O(N_d \cdot \sum_{i \leq d} N_i)$ per depth | $O(8^{2D_{high}})$ (worst) |
| Fractal-3D (递归) | $O(N_d \log N_d)$ per level | $O(\sum 8^d \cdot 3d) = O(D \cdot 8^D)$ |

**内存对比**：
- OctGPT: 所有深度 tokens 同时在 GPU → peak at deepest
- Fractal-3D: 逐层 forward/backward → peak at each level separately

---

## 实验计划

### Phase 1: 最简可行验证
- **方向 5（Occupancy-Only）** + **方向 1（递归架构，2层）**
- ShapeNet airplane, depth 2→4, 仅 predict occupancy
- 手工 Marching Cubes 转 mesh
- 目标：证明递归八叉树生成可行，代码框架跑通

### Phase 2: 主线对比实验
- **方向 1（完整递归多模型）** + VQ-VAE backbone
- Baseline: OctGPT unconditional airplane
- depth 2→6, 3-4 层递归
- 指标: CD, F-Score, 速度, 内存

### Phase 3: 派生方向探索
- **方向 2**: AR + Morton KV-Cache
- **方向 4**: 空间条件消融实验
- **方向 7**: 混合 AR/MAR

### Phase 4: 规模化
- **方向 8**: 渐进训练 depth 2→7
- **方向 9**: 多条件生成
- 扩展到更多类别 (car, chair, table) 或 Objaverse

---

## 代码参考索引

**FractalGen**:
- `extern/fractalgen/models/fractalgen.py` — 递归模型定义，level 间如何连接
- `extern/fractalgen/models/ar.py` — AR generator, patchify, 2D RoPE, KV-Cache, sampling
- `extern/fractalgen/models/mar.py` — MAR generator, random masking, cosine schedule
- `extern/fractalgen/models/pixelloss.py` — 最终层, codebook 机制
- `extern/fractalgen/engine_fractalgen.py` — 训练/评估循环

**OctGPT**:
- `extern/octgpt/models/octgpt.py` — 八叉树 masked AR 生成流程
- `extern/octgpt/models/octformer.py` — OctFormer, OctreeT, teacher-forcing mask
- `extern/octgpt/utils/utils.py` — `octree2seq`, `seq2octree`, `depth2batch`, `batch2depth`
- `extern/octgpt/models/vae.py` — VQ-VAE (Encoder, Decoder, BSQ)
- `extern/octgpt/configs/ShapeNet/shapenet_uncond.yaml` — ShapeNet 训练配置
- `extern/octgpt/datasets/shapenet.py` — ShapeNet 数据加载

---

## 环境与依赖

- Python 3.10+, PyTorch 2.5+, CUDA 12.4
- `ocnn` (Octree CNN), `ognn` (Octree GNN) — OctGPT 依赖
- 参考 `extern/octgpt/requirements.txt` 和 `extern/fractalgen/environment.yaml`
