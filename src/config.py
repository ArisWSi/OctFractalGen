"""递归多模型八叉树生成（VQ-VAE 管线）的配置。"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    """OctreeFractalGen 的架构超参数。"""

    # 八叉树结构
    full_depth: int = 3          # 初始八叉树深度
    depth_stop: int = 6          # VQHead 预测 VQ codes 的最终深度（= VQVAE code_depth）
    # fractal_levels: AR generator 所在的深度列表
    # 例如 (3, 4, 5) 表示 depth 3, 4, 5 各一个 AR generator
    # VQHead 在 depth_stop 作为额外的终端层（不计入此列表）
    # 约束: depth_stop == fractal_levels[-1] + 1
    fractal_levels: Tuple[int, ...] = (3, 4, 5)

    # 每层容量（索引 0 = 最粗，向最细递减）
    embed_dims: Tuple[int, ...] = (512, 384, 256)
    num_blocks: Tuple[int, ...] = (16, 12, 8)
    num_heads: Tuple[int, ...] = (8, 6, 4)

    # 共享 Transformer 设置
    mlp_ratio: float = 4.0
    attn_drop: float = 0.1
    proj_drop: float = 0.1
    drop_path: float = 0.0

    # Patch 注意力（OctFormer 风格）
    patch_size: int = 1024         # patch 内 token 数（0 = 全注意力）
    dilation: int = 4              # 膨胀率（跨 patch 连接）

    # 类别嵌入（仅 Level 0）
    num_classes: int = 1         # 1 = 无条件（虚拟类别 0）
    label_drop_prob: float = 0.1  # CFG 的类别 dropout 概率

    # 条件传播
    num_spatial_neighbors: int = 7   # 中心 + 6 个面邻居
    cond_embed_dim: int = 512        # 层间传递的条件向量维度

    # 梯度检查点
    grad_checkpointing: bool = False

    # ── FractalOctGPT 架构参数（复用 OctFormer）─────────────────
    # 统一维度/头数（所有层共享，满足 RoPE 约束 head_dim % 6 == 0）
    # OctGPT 原版: num_embed=768, num_heads=8 → head_dim=96, 96%6==0
    octgpt_embed_dim: int = 768
    octgpt_num_heads: int = 8
    use_swin: bool = True              # SWIN 窗口移位
    pos_emb_type: str = "sin"          # 'sin' | 'abs'（abs 对 num_embed%6≠0 有 bug）

    # ── MaskGIT 生成参数（与 OctGPT 对齐）──────────────────────
    buffer_size: int = 64           # 每层条件 buffer token 数（per batch item）
    random_flip: float = 0.1        # BSQ bit 翻转增强（训练时）
    remask_stage: float = 0.7       # 开始 remask 的进度比例
    # 每 depth 的迭代次数和起始温度（fractal_levels + VQHead = 4 层）
    num_iters: Tuple[int, ...] = (64, 128, 128, 256)
    start_temperature: Tuple[float, ...] = (1.0, 1.2, 0.5, 0.5)

    # ── CoarseFineOctGPT 专用 ─────────────────────────────────
    # coarse/fine 子配置 (dict): {dim, heads, blocks}
    coarse: dict = None
    fine: dict = None
    detach_prefix: bool = False     # True=prefix 不反传到 coarse

    # ── Plan B: 官方 OctGPT 初始化 fine ──────────────────────
    official_octgpt_ckpt: str = ""  # 官方 OctGPT checkpoint 路径
    finetune: dict = None           # {stage, freeze_fine, freeze_coarse, ...}
    lr_coarse: float = 0.0          # 分组学习率 (0=用 train.lr)
    lr_prefix: float = 0.0
    lr_fine: float = 0.0


@dataclass
class VQVAEConfig:
    """预训练 VQ-VAE 的配置。"""

    # 预训练 VQ-VAE checkpoint 路径
    ckpt_path: str = ""

    # 模型变体（必须与 checkpoint 匹配，控制 encoder/decoder 架构）
    vae_name: str = "vqvae_large"   # 'vqvae_big' | 'vqvae_large' | 'vqvae_huge'

    # VQ 架构（必须与 checkpoint 匹配）
    embedding_channels: int = 32     # 注意：OctGPT 默认用 32，非 64
    embedding_sizes: int = 128
    quantizer_type: str = "bsq"
    quantizer_group: int = 4
    feature: str = "ND"
    in_channels: int = 4
    n_node_type: int = 7
    vae_depth: int = 8              # VQ-VAE 的最大八叉树深度


@dataclass
class DataConfig:
    """ShapeNet 数据加载配置。"""

    # 数据路径
    location: str = "data/ShapeNet/dataset_256"
    filelist: str = "data/ShapeNet/filelist/train_airplane.txt"
    val_filelist: str = ""            # 验证集文件列表（空则跳过验证）

    # 八叉树构建
    depth: int = 8                  # 最大八叉树深度（需 ≥ VQVAE 的 vae_depth）
    full_depth: int = 3             # 初始深度（与 ModelConfig 匹配）
    points_scale: float = 1.0       # 输入点云在 [-1, 1] 范围内

    # 数据加载
    batch_size: int = 8
    num_workers: int = 4
    max_points: int = 120000
    distort: bool = False           # 禁用噪声以得到干净的八叉树


@dataclass
class TrainConfig:
    """训练超参数，遵循 FractalGen 的配方。"""

    # 优化器（FractalGen 的 AdamW 设置: wd=0.05, β=(0.9, 0.95)）
    lr: float = 1e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 3.0           # 遵循 FractalGen

    # 学习率调度（FractalGen: 10% warmup + cosine）
    max_epoch: int = 200
    warmup_epochs: int = 20          # 200 epoch 的 10%

    # 混合精度
    use_amp: bool = True

    # 日志与检查点
    log_interval: int = 50           # 日志间隔（步数）
    save_interval: int = 20          # checkpoint 间隔（epoch）
    logdir: str = "logs/"

    # 硬件
    device: str = "cuda"
    seed: int = 42


@dataclass
class Config:
    """顶层配置聚合器。"""

    model: ModelConfig = field(default_factory=ModelConfig)
    vqvae: VQVAEConfig = field(default_factory=VQVAEConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ------------------------------------------------------------------
# 预定义模型变体（遵循 FractalGen 命名约定）
# ------------------------------------------------------------------

def octree_fractal_tiny(**overrides) -> Config:
    """微型模型，用于快速迭代: 3 层 AR，小容量。"""
    return Config(
        model=ModelConfig(
            full_depth=3, depth_stop=6, fractal_levels=(3, 4, 5),
            embed_dims=(256, 192, 128), num_blocks=(8, 6, 4),
            num_heads=(4, 4, 4),
            cond_embed_dim=256,
            **overrides,
        )
    )


def octree_fractal_base(**overrides) -> Config:
    """基础模型: 3 层 AR，中等容量。"""
    return Config(
        model=ModelConfig(
            full_depth=3, depth_stop=6, fractal_levels=(3, 4, 5),
            embed_dims=(512, 384, 256), num_blocks=(16, 12, 8),
            num_heads=(8, 6, 4),
            cond_embed_dim=512,
            **overrides,
        )
    )


def octree_fractal_large(**overrides) -> Config:
    """大型模型: 3 层 AR，高容量。"""
    return Config(
        model=ModelConfig(
            full_depth=3, depth_stop=6, fractal_levels=(3, 4, 5),
            embed_dims=(768, 512, 384), num_blocks=(24, 16, 12),
            num_heads=(12, 8, 6),
            cond_embed_dim=768,
            **overrides,
        )
    )
