"""
OctreeFractalGen: Recursive Multi-Model Octree Generator.

Adapts FractalGen's recursive architecture for 3D octrees.
Each level handles one octree depth transition independently
(reads parent nodes from ground-truth octree at training time).
The recursive structure provides:
  - Hierarchical model capacity (coarse → fine decreasing params)
  - Independent autoregressive generation per level
  - Direct occupancy output (no VQ-VAE)

Phase 1 design (Direction 5):
  - 2 recursive levels: full_depth→full_depth+1, full_depth+1→full_depth+2
  - Each level: AR Transformer → child occupancy prediction (BCE loss)
  - Final level: GeoHead MLP → occupancy at finest depth
  - Global condition: class embedding repeated per node
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.model.ar_octree import OctreeAR
from src.utils.octree_ops import get_node_xyz, get_split_labels


# ---------------------------------------------------------------------------
# GeoHead: lightweight final-level occupancy predictor
# ---------------------------------------------------------------------------

class GeoHead(nn.Module):
    """Predict binary occupancy at the finest octree depth.

    Lightweight MLP: position + condition → occupancy logit.
    No VQ-VAE, no codebook — direct BCE on occupancy.
    """

    def __init__(self, embed_dim: int = 128, cond_dim_in: int = 512):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim_in, embed_dim, bias=True)
        self.pos_emb = nn.Linear(3, embed_dim, bias=True)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        parent_xyz: torch.Tensor,        # (B, Np, 3)
        parent_cond: torch.Tensor,       # (B, Np, cond_dim_in)
        child_gt_mask: Optional[torch.Tensor] = None,  # (B, Np, 8)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict occupancy for 8 children per parent.

        Returns:
            logits: (B, Np, 8)
            loss: scalar BCE loss
        """
        B, Np, _ = parent_xyz.shape
        device = parent_xyz.device

        # 8 child positions per parent
        offsets = torch.tensor([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], device=device, dtype=torch.float32)

        children_xyz = (parent_xyz.float() * 2).unsqueeze(2) + offsets.view(1, 1, 8, 3)
        # (B, Np, 8, 3)

        pos_emb = self.pos_emb(children_xyz)                    # (B, Np, 8, embed_dim)
        cond_emb = self.cond_proj(parent_cond).unsqueeze(2)      # (B, Np, 1, embed_dim)
        h = self.norm(pos_emb + cond_emb)                        # (B, Np, 8, embed_dim)
        logits = self.mlp(h).squeeze(-1)                         # (B, Np, 8)

        loss = torch.tensor(0.0, device=device)
        if child_gt_mask is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits.reshape(-1), child_gt_mask.float().reshape(-1),
                reduction='mean',
            )

        return logits, loss

    @torch.no_grad()
    def sample(
        self,
        parent_xyz: torch.Tensor,
        parent_cond: torch.Tensor,
        temperature: float = 1.0,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Sample occupancy at final depth.

        Returns:
            mask: (B, Np, 8) binary occupancy
        """
        logits, _ = self.forward(parent_xyz, parent_cond)
        probs = torch.sigmoid(logits / temperature)
        return (probs > threshold).float()


# ---------------------------------------------------------------------------
# OctreeFractalGen: recursive multi-model container
# ---------------------------------------------------------------------------

class OctreeFractalGen(nn.Module):
    """Recursive multi-model octree generator.

    Architecture (2-level example, full_depth=3, depth_stop=5):
        Level 0 (depth 3→4):
            OctreeAR: 512-dim, 16 blocks, 8 heads
            Predicts which of ≤64 children (8 parents × 8) exist

        Level 1 / GeoHead (depth 4→5):
            GeoHead: 128-dim MLP
            Predicts occupancy of ≤512 children (≤64 parents × 8)

    Each level receives the same global class-condition repeated per node.
    Cross-level conditioning (spatial neighbors, per-node features) is
    deferred to Direction 4.
    """

    def __init__(self, config, fractal_level: int = 0):
        super().__init__()
        self.config = config
        self.fractal_level = fractal_level
        self.num_levels = len(config.fractal_levels)
        self.current_depth = config.fractal_levels[fractal_level]

        # ------------------------------------------------------------------
        # Class embedding (top level only)
        # ------------------------------------------------------------------
        if fractal_level == 0:
            self.num_classes = config.num_classes
            # +1 slot for CFG "fake" class
            self.class_emb = nn.Embedding(config.num_classes + 1,
                                          config.cond_embed_dim)
            self.label_drop_prob = config.label_drop_prob
            self.fake_latent = nn.Parameter(torch.zeros(1, config.cond_embed_dim))
            nn.init.normal_(self.class_emb.weight, std=0.02)
            nn.init.normal_(self.fake_latent, std=0.02)

        # ------------------------------------------------------------------
        # Current level generator
        # ------------------------------------------------------------------
        is_final = (fractal_level == self.num_levels - 1)

        if not is_final:
            self.generator = OctreeAR(
                embed_dim=config.embed_dims[fractal_level],
                num_blocks=config.num_blocks[fractal_level],
                num_heads=config.num_heads[fractal_level],
                cond_dim_in=config.cond_embed_dim,
                cond_dim_out=config.cond_embed_dim,
                attn_drop=config.attn_drop,
                proj_drop=config.proj_drop,
                grad_checkpointing=config.grad_checkpointing,
            )

        # ------------------------------------------------------------------
        # Next level (recursive or terminal)
        # ------------------------------------------------------------------
        if not is_final:
            self.next_fractal = OctreeFractalGen(config, fractal_level + 1)
        else:
            self.next_fractal = GeoHead(
                embed_dim=config.embed_dims[-1],
                cond_dim_in=config.cond_embed_dim,
            )

    # ------------------------------------------------------------------
    # Condition helper
    # ------------------------------------------------------------------

    def _get_class_condition(self, octree, labels: Optional[torch.Tensor] = None
                             ) -> torch.Tensor:
        """Get per-node class condition for the top-level nodes.

        Returns:
            cond: (B, Np, cond_embed_dim)
        """
        B = octree.batch_size
        device = octree.device

        if labels is None:
            labels = torch.zeros(B, dtype=torch.long, device=device)

        class_embedding = self.class_emb(labels)  # (B, cond_dim)

        if self.training:
            drop_mask = (
                torch.rand(B, device=device) < self.label_drop_prob
            ).float().unsqueeze(-1)
            class_embedding = (
                drop_mask * self.fake_latent + (1 - drop_mask) * class_embedding
            )

        return class_embedding  # (B, cond_dim)

    def _make_per_node_cond(self, global_cond: torch.Tensor, nnum: int, B: int
                            ) -> torch.Tensor:
        """Expand global condition to per-node.

        Args:
            global_cond: (B, cond_dim)
            nnum: total number of nodes at this depth
            B: batch size

        Returns:
            per_node_cond: (B, nnum//B, cond_dim)
        """
        Np = nnum // B
        return global_cond.unsqueeze(1).expand(B, Np, -1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, octree, labels: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """Top-level entry: compute class condition and recurse.

        Args:
            octree: ocnn.Octree (ground truth for training)
            labels: (B,) class labels (None → unconditional)

        Returns:
            total_loss: sum of BCE losses from all levels
        """
        if self.fractal_level != 0:
            raise RuntimeError(
                "Forward should only be called on the top-level OctreeFractalGen"
            )

        B = octree.batch_size
        device = octree.device

        # Global class condition
        class_cond = self._get_class_condition(octree, labels)  # (B, cond_dim)

        # Recurse through all levels
        total_loss = self._forward_level(octree, class_cond)
        return total_loss

    def _forward_level(self, octree, global_cond: torch.Tensor) -> torch.Tensor:
        """Forward through one recursive level.

        Each level:
        1. Reads parent nodes at its depth from octree
        2. Expands global condition to per-node
        3. Predicts child occupancy via its generator
        4. Computes BCE loss against ground truth
        5. Recurses to next level

        Args:
            octree: ocnn.Octree (ground truth)
            global_cond: (B, cond_dim) class condition

        Returns:
            scalar loss
        """
        depth = self.config.fractal_levels[self.fractal_level]
        B = octree.batch_size
        device = octree.device

        # Get parent nodes at this depth
        parent_xyz, _ = get_node_xyz(octree, depth)     # (total_nnum, 3)
        nnum = octree.nnum[depth]

        if nnum == 0:
            # No nodes at this depth (empty octree) → skip
            return torch.tensor(0.0, device=device)

        Np = nnum // B
        parent_xyz_3d = parent_xyz.view(B, Np, 3)        # (B, Np, 3)
        parent_cond = self._make_per_node_cond(global_cond, nnum, B)  # (B, Np, cond_dim)

        # Ground truth child labels
        gt_labels = get_split_labels(octree, depth)       # (nnum, 8)
        gt_labels = gt_labels.view(B, Np, 8)              # (B, Np, 8)

        # Forward through current level's generator
        if self.fractal_level < self.num_levels - 1:
            # AR Transformer level
            _, _, level_loss = self.generator(parent_xyz_3d, parent_cond, gt_labels)
        else:
            # GeoHead (final level)
            _, level_loss = self.next_fractal(parent_xyz_3d, parent_cond, gt_labels)

        # Recurse to next level with same global condition
        deeper_loss = torch.tensor(0.0, device=device)
        if isinstance(self.next_fractal, OctreeFractalGen):
            deeper_loss = self.next_fractal._forward_level(octree, global_cond)
        # Note: if self.next_fractal is GeoHead, it was already computed above
        # (when self.fractal_level == self.num_levels - 1)

        return level_loss + deeper_loss

    # ------------------------------------------------------------------
    # Generation (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        octree,
        labels: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
    ):
        """Recursively generate the octree from scratch.

        Args:
            octree: ocnn.Octree initialized at full_depth (via init_octree)
            labels: (B,) class labels
            temperature: sampling temperature
            cfg_scale: CFG scale (1.0 = no guidance)

        Returns:
            octree: generated octree with structure up to depth_stop
        """
        if self.fractal_level != 0:
            raise RuntimeError("generate() should only be called on top level")

        B = octree.batch_size
        device = octree.device

        if labels is None:
            labels = torch.zeros(B, dtype=torch.long, device=device)

        # Class condition
        class_cond = self.class_emb(labels)  # (B, cond_dim)

        # CFG unconditional branch
        uncond = None
        if cfg_scale != 1.0:
            uncond = self.fake_latent.expand(B, -1)  # (B, cond_dim)

        return self._generate_level(octree, class_cond, temperature, cfg_scale, uncond)

    @torch.no_grad()
    def _generate_level(
        self,
        octree,
        global_cond: torch.Tensor,
        temperature: float,
        cfg_scale: float,
        uncond: Optional[torch.Tensor] = None,
    ):
        """Generate one octree depth and recurse."""
        depth = self.config.fractal_levels[self.fractal_level]
        B = octree.batch_size
        device = octree.device

        # Get parent nodes at current depth
        parent_xyz, _ = get_node_xyz(octree, depth)
        nnum = octree.nnum[depth]

        if nnum == 0:
            return octree, None

        Np = nnum // B
        parent_xyz_3d = parent_xyz.view(B, Np, 3)
        parent_cond = self._make_per_node_cond(global_cond, nnum, B)

        # CFG: blend conditional and unconditional logits
        if cfg_scale != 1.0 and uncond is not None:
            parent_cond_uncond = self._make_per_node_cond(uncond, nnum, B)
        else:
            parent_cond_uncond = None

        if self.fractal_level < self.num_levels - 1:
            # AR Transformer level: autoregressive sample (8-way per parent)
            child_8way, _ = self.generator.sample(
                parent_xyz_3d, parent_cond, temperature
            )  # (B, Np, 8) binary

            # Collapse 8-way → per-node binary: split if ANY child exists
            split_label = child_8way.any(dim=-1).long()  # (B, Np)
            split_label = split_label.reshape(B * Np)     # (nnum,)
            octree.octree_split(split_label, depth=depth)
            octree.octree_grow(depth + 1)

            # Recurse
            if isinstance(self.next_fractal, OctreeFractalGen):
                return self.next_fractal._generate_level(
                    octree, global_cond, temperature, cfg_scale, uncond,
                )
            else:
                # Last AR level → GeoHead
                next_depth = self.config.fractal_levels[-1]
                next_xyz, _ = get_node_xyz(octree, next_depth)
                nnum_next = octree.nnum[next_depth]
                Np_next = nnum_next // B
                final_xyz = next_xyz.view(B, Np_next, 3)

                occupancy = self.next_fractal.sample(
                    final_xyz,
                    self._make_per_node_cond(global_cond, nnum_next, B),
                    temperature,
                )
                return octree, occupancy
        else:
            # GeoHead level: sample occupancy
            child_mask = self.next_fractal.sample(
                parent_xyz_3d, parent_cond, temperature
            )
            return octree, child_mask


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def octree_fractal_tiny(config=None):
    if config is None:
        from src.config import octree_fractal_tiny as make_cfg
        config = make_cfg().model
    return OctreeFractalGen(config, fractal_level=0)


def octree_fractal_base(config=None):
    if config is None:
        from src.config import octree_fractal_base as make_cfg
        config = make_cfg().model
    return OctreeFractalGen(config, fractal_level=0)


def octree_fractal_large(config=None):
    if config is None:
        from src.config import octree_fractal_large as make_cfg
        config = make_cfg().model
    return OctreeFractalGen(config, fractal_level=0)
