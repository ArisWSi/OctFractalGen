"""
AR Generator for one octree level.

Adapts FractalGen's AR class for 3D octree nodes. Each generator
handles one depth transition (d -> d+1): takes parent nodes at depth d,
predicts which of their 8 children exist at depth d+1.

Sequence ordering: all child candidates across all parents are flattened
into a single sequence sorted by Morton (Z-order) code for 3D spatial
locality. The causal mask ensures child i can only see children j < i.

Training: teacher-forcing (all children visible, causal mask on Morton order)
Inference: autoregressive in Morton order with KV-Cache
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.model.transformer import (
    RMSNorm,
    TransformerBlock,
    KVCache,
    precompute_freqs_cis_3d,
    find_multiple,
)
from src.utils.octree_ops import child_xyz, morton_encode_3d


class OctreeAR(nn.Module):
    """Autoregressive generator for one octree depth level.

    Takes N parent nodes at depth d and predicts occupancy for their
    8xN children at depth d+1, ordered by 3D Morton (Z-order) code.

    Args:
        embed_dim: internal embedding dimension
        num_blocks: number of Transformer blocks
        num_heads: number of attention heads
        cond_dim_in: dimension of incoming condition
        cond_dim_out: dimension of outgoing condition
        grad_checkpointing: use gradient checkpointing
        attn_drop: attention dropout rate
        proj_drop: projection/FFN dropout rate
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_blocks: int = 16,
        num_heads: int = 8,
        cond_dim_in: int = 512,
        cond_dim_out: int = 512,
        grad_checkpointing: bool = False,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.cond_dim_in = cond_dim_in
        self.cond_dim_out = cond_dim_out
        self.grad_checkpointing = grad_checkpointing

        # Embeddings
        self.cond_proj = nn.Linear(cond_dim_in, embed_dim, bias=True)
        self.pos_emb = nn.Linear(3, embed_dim, bias=True)
        self.token_ln = RMSNorm(embed_dim, eps=1e-6)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, n_head=num_heads,
                attn_drop=attn_drop, proj_drop=proj_drop, mlp_drop=proj_drop,
            )
            for _ in range(num_blocks)
        ])
        self.norm = RMSNorm(embed_dim, eps=1e-6)

        # Output heads
        self.split_head = nn.Linear(embed_dim, 1, bias=True)      # occupancy logit
        self.cond_head = nn.Linear(embed_dim, cond_dim_out, bias=True)  # next-level condition

        # KV-cache state (lazy init during inference)
        self.max_batch_size = -1
        self.max_seq_length = -1

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

    # ------------------------------------------------------------------
    # KV-Cache
    # ------------------------------------------------------------------

    def setup_caches(self, max_batch_size: int, max_seq_length: int):
        head_dim = self.embed_dim // self.num_heads
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for block in self.blocks:
            block.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.num_heads, head_dim)

    # ------------------------------------------------------------------
    # Morton ordering
    # ------------------------------------------------------------------

    def _morton_sort(
        self, children_xyz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Morton sort indices for Np*8 children.

        All batch items share the same Morton order (coordinates are
        identical across batch since octree structure is shared).

        Args:
            children_xyz: (B, Np, 8, 3) child coordinates

        Returns:
            sort_idx: (Np*8,) indices to reorder into Morton order
            unsort_idx: (Np*8,) indices to restore parent-major order
        """
        B, Np, _, _ = children_xyz.shape
        device = children_xyz.device

        # Use first batch item's coordinates for ordering
        coords = children_xyz[0].view(Np * 8, 3)  # (Np*8, 3)

        # Compute Morton codes
        codes = morton_encode_3d(coords[:, 0], coords[:, 1], coords[:, 2])

        # Sort indices
        sort_idx = torch.argsort(codes)
        unsort_idx = torch.empty_like(sort_idx)
        unsort_idx[sort_idx] = torch.arange(Np * 8, device=device)

        return sort_idx, unsort_idx

    # ------------------------------------------------------------------
    # Token construction
    # ------------------------------------------------------------------

    def _build_child_tokens_morton(
        self,
        parent_xyz: torch.Tensor,
        parent_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build child tokens, sorted by Morton (Z-order) code.

        Args:
            parent_xyz: (B, Np, 3) parent coordinates at depth d
            parent_cond: (B, Np, cond_dim_in) condition per parent

        Returns:
            child_tokens: (B, Np*8, embed_dim) sorted in Morton order
            child_xyz_sorted: (B, Np*8, 3) coordinates in Morton order
            unsort_idx: (Np*8,) to restore parent-major order
        """
        B, Np, _ = parent_xyz.shape

        # All 8 child coordinates per parent: (B, Np, 8, 3)
        children_xyz = child_xyz(parent_xyz)

        # Morton sort indices
        sort_idx, unsort_idx = self._morton_sort(children_xyz)

        # Project parent condition: (B, Np, embed_dim) -> (B, Np, 1, embed_dim)
        cond_emb = self.cond_proj(parent_cond).unsqueeze(2)  # (B, Np, 1, embed_dim)

        # Position embedding from child coordinates: (B, Np, 8, embed_dim)
        pos_emb = self.pos_emb(children_xyz.float())

        # Combine and flatten: (B, Np, 8, embed_dim) -> (B, Np*8, embed_dim)
        child_tokens = (cond_emb + pos_emb).view(B, Np * 8, self.embed_dim)

        # Flatten coordinates for sorting
        children_xyz_flat = children_xyz.view(B, Np * 8, 3)

        # Sort by Morton order
        child_tokens = child_tokens[:, sort_idx, :]
        children_xyz_flat = children_xyz_flat[:, sort_idx, :]

        child_tokens = self.token_ln(child_tokens)

        return child_tokens, children_xyz_flat, unsort_idx

    # ------------------------------------------------------------------
    # RoPE helper
    # ------------------------------------------------------------------

    def _compute_3d_rope(self, xyz: torch.Tensor) -> torch.Tensor:
        """Compute 3D RoPE frequencies for a flat sequence.

        Args:
            xyz: (B, seq_len, 3) node coordinates

        Returns:
            freqs_cis: (B, seq_len, head_dim//2, 2)
        """
        B, seq_len, _ = xyz.shape
        flat_xyz = xyz.reshape(B * seq_len, 3)
        head_dim = self.embed_dim // self.num_heads
        freqs_cis = precompute_freqs_cis_3d(flat_xyz, head_dim)
        return freqs_cis.view(B, seq_len, head_dim // 2, 2)

    # ------------------------------------------------------------------
    # Transformer forward
    # ------------------------------------------------------------------

    def _forward_transformer(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply all Transformer blocks."""
        if self.grad_checkpointing and self.training:
            for block in self.blocks:
                x = checkpoint(block, x, freqs_cis, input_pos, mask,
                               use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x, freqs_cis, input_pos, mask)
        return self.norm(x)

    # ------------------------------------------------------------------
    # Training forward (teacher-forcing with Morton ordering)
    # ------------------------------------------------------------------

    def forward(
        self,
        parent_xyz: torch.Tensor,
        parent_cond: torch.Tensor,
        child_gt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass.

        Sequence: Morton-ordered children, causal mask. The model sees
        all children with causal attention (Z-order locality).

        Args:
            parent_xyz: (B, Np, 3) parent node coordinates
            parent_cond: (B, Np, cond_dim_in) per-parent condition
            child_gt_mask: (B, Np, 8) ground truth occupancy (1=exists)

        Returns:
            child_logits: (B, Np, 8) occupancy logits (parent-major order)
            child_cond: (B, Np*8, cond_dim_out) per-child features (parent-major)
            loss: scalar BCE loss
        """
        B, Np, _ = parent_xyz.shape
        device = parent_xyz.device

        # Build Morton-sorted tokens
        child_tokens, child_xyz_sorted, unsort_idx = \
            self._build_child_tokens_morton(parent_xyz, parent_cond)

        # 3D RoPE on sorted coordinates
        freqs_cis = self._compute_3d_rope(child_xyz_sorted)

        # Transformer forward (causal mask by default)
        x = self._forward_transformer(child_tokens, freqs_cis)

        # Predict occupancy (B, Np*8, 1) in Morton order
        logits_morton = self.split_head(x).squeeze(-1)  # (B, Np*8)

        # Condition features in Morton order
        cond_morton = self.cond_head(x)  # (B, Np*8, cond_dim_out)

        # Unsort back to parent-major order
        logits_pm = logits_morton[:, unsort_idx]       # (B, Np*8)
        cond_pm = cond_morton[:, unsort_idx, :]         # (B, Np*8, cond_dim_out)
        child_logits = logits_pm.view(B, Np, 8)         # (B, Np, 8)

        # Compute loss
        loss = torch.tensor(0.0, device=device)
        if child_gt_mask is not None:
            # Reorder gt_mask to Morton order for loss computation
            gt_flat = child_gt_mask.float().view(B, Np * 8)  # parent-major
            gt_morton = gt_flat[:, sort_idx]                   # Morton order
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits_morton, gt_morton, reduction='mean'
            )

        return child_logits, cond_pm, loss

    # ------------------------------------------------------------------
    # Autoregressive sampling (Morton order)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        parent_xyz: torch.Tensor,
        parent_cond: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively generate child occupancy in Morton order.

        Morton ordering ensures that spatially adjacent children
        are generated close together, improving coherence.

        Args:
            parent_xyz: (B, Np, 3)
            parent_cond: (B, Np, cond_dim_in)
            temperature: sampling temperature

        Returns:
            child_mask: (B, Np, 8) occupancy in parent-major layout
            child_cond: (B, Np*8, cond_dim_out) in parent-major layout
        """
        B, Np, _ = parent_xyz.shape
        device = parent_xyz.device
        total_steps = Np * 8  # sequence length

        # Compute all child positions
        children_xyz = child_xyz(parent_xyz)          # (B, Np, 8, 3)
        children_xyz_flat = children_xyz.view(B, total_steps, 3)

        # Morton sort indices
        sort_idx, unsort_idx = self._morton_sort(children_xyz)

        # Pre-compute condition embeddings per parent: (B, Np, embed_dim)
        cond_emb = self.cond_proj(parent_cond)

        # Precompute Morton-ordered list of (parent_idx, octant_idx)
        # parent-major order: [p0_c0, p0_c1, ..., p0_c7, p1_c0, ..., p{Np-1}_c7]
        parent_idx_pm = torch.arange(Np, device=device).repeat_interleave(8)
        octant_idx_pm = torch.arange(8, device=device).repeat(Np)

        # Morton order: unsort_idx maps Morton position -> parent-major position
        # unsort_idx[k] = parent-major index of the child at Morton position k
        parent_morton = parent_idx_pm[unsort_idx]   # (Np*8,) parent idx per Morton pos
        octant_morton = octant_idx_pm[unsort_idx]   # (Np*8,) octant idx per Morton pos

        # Setup KV-cache
        self.setup_caches(max_batch_size=B, max_seq_length=total_steps)

        # Storage (parent-major)
        child_mask = torch.zeros(B, Np, 8, device=device)
        all_cond = torch.zeros(B, total_steps, self.cond_dim_out, device=device)

        head_dim = self.embed_dim // self.num_heads

        # Generate step by step in Morton order
        for step in range(total_steps):
            p_idx = parent_morton[step].item()  # parent index in Morton order
            o_idx = octant_morton[step].item()  # octant index in Morton order

            # Build token for this child
            child_pos = children_xyz[:, p_idx, o_idx, :]  # (B, 3)
            child_token = (
                cond_emb[:, p_idx, :] +
                self.pos_emb(child_pos.float())
            ).unsqueeze(1)  # (B, 1, embed_dim)
            child_token = self.token_ln(child_token)

            # RoPE
            freqs_cis = precompute_freqs_cis_3d(
                child_pos.reshape(B, 3), head_dim
            ).unsqueeze(1)  # (B, 1, head_dim//2, 2)

            # Forward with KV-cache
            input_pos = torch.tensor([step], device=device)
            x = self._forward_transformer(
                child_token, freqs_cis, input_pos=input_pos)

            # Predict occupancy
            logit = self.split_head(x).squeeze(-1).squeeze(-1)  # (B,)
            prob = torch.sigmoid(logit / temperature)
            child_mask[:, p_idx, o_idx] = torch.rand(B, device=device) < prob
            child_mask[:, p_idx, o_idx] = child_mask[:, p_idx, o_idx].float()

            # Store condition features at the correct parent-major position
            pm_pos = unsort_idx[step].item()  # Morton step -> parent-major index
            cond_feat = self.cond_head(x).squeeze(1)  # (B, cond_dim_out)
            all_cond[:, pm_pos, :] = cond_feat

        # Clean up KV-cache
        for block in self.blocks:
            block.attention.kv_cache = None

        return child_mask, all_cond


# ------------------------------------------------------------------
# Factory functions
# ------------------------------------------------------------------

def octree_ar_tiny(**kwargs) -> OctreeAR:
    return OctreeAR(embed_dim=128, num_blocks=4, num_heads=4,
                    cond_dim_in=256, cond_dim_out=128, **kwargs)


def octree_ar_base(**kwargs) -> OctreeAR:
    return OctreeAR(embed_dim=512, num_blocks=16, num_heads=8,
                    cond_dim_in=512, cond_dim_out=512, **kwargs)


def octree_ar_light(**kwargs) -> OctreeAR:
    return OctreeAR(embed_dim=256, num_blocks=8, num_heads=4,
                    cond_dim_in=512, cond_dim_out=256, **kwargs)
