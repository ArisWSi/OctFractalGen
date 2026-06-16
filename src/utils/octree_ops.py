"""
Octree operations for occupancy-only recursive generation.

Ports key utilities from OctGPT's utils/utils.py, adapted to minimize
dependency on ocnn's internal APIs. We use ocnn only for the Octree
data structure; tensor manipulation is done here.

Key functions:
- octree2seq / seq2octree: convert octree children to flat binary sequence
- get_node_xyz: extract (x, y, z) coordinates for nodes at a depth
- morton_code: compute Morton (Z-order) code for spatial ordering
- child_xyz: compute child node positions from parent
"""

import copy
from typing import List, Tuple

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Morton (Z-order) encoding
# ---------------------------------------------------------------------------

def morton_encode_3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Compute Morton (Z-order) code for 3D coordinates.

    Interleaves bits from x, y, z to produce a 1D ordering that preserves
    spatial locality. Uses the classic bit-interleaving approach.

    Args:
        x, y, z: integer tensors of the same shape, values in [0, 2^depth)

    Returns:
        Morton codes as int64 tensor
    """
    def spread_bits(v):
        """Spread bits of v apart by factor of 3."""
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    # Ensure inputs are int64 for bit operations
    x = x.long()
    y = y.long()
    z = z.long()

    return spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)


def morton_order_indices(xyz: torch.Tensor) -> torch.Tensor:
    """Return indices that sort coordinates by Morton order.

    Args:
        xyz: (N, 3) integer coordinates

    Returns:
        indices: (N,) tensor of sorted indices
    """
    codes = morton_encode_3d(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    return torch.argsort(codes)


# ---------------------------------------------------------------------------
# Child coordinate computation
# ---------------------------------------------------------------------------

def child_xyz(parent_xyz: torch.Tensor) -> torch.Tensor:
    """Compute coordinates of 8 children for each parent node.

    In an octree, each node at depth d has 8 potential children at depth d+1.
    Children are indexed 0-7, where bit 0 = +x, bit 1 = +y, bit 2 = +z.

    Child mapping (cx, cy, cz) where cx = (index >> 0) & 1, etc.:
        0: (0,0,0)  1: (1,0,0)  2: (0,1,0)  3: (1,1,0)
        4: (0,0,1)  5: (1,0,1)  6: (0,1,1)  7: (1,1,1)

    Args:
        parent_xyz: (*, 3) parent coordinates at depth d

    Returns:
        children_xyz: (*, 8, 3) child coordinates at depth d+1
    """
    *batch_dims, _ = parent_xyz.shape
    device = parent_xyz.device

    # Child offsets for 8 octants: (8, 3)
    offsets = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], device=device, dtype=parent_xyz.dtype)

    # Parent coords doubled (since each unit at depth d = 2 units at depth d+1)
    parent_doubled = (parent_xyz * 2).unsqueeze(-2)  # (*, 1, 3)
    children = parent_doubled + offsets.view(*([1] * len(batch_dims)), 8, 3)
    return children


# ---------------------------------------------------------------------------
# Node coordinate extraction
# ---------------------------------------------------------------------------

def get_node_xyz(octree, depth: int) -> torch.Tensor:
    """Get (x, y, z) integer coordinates of all nodes at a given depth.

    Uses ocnn's octree.xyzb() which returns (x, y, z, batch_idx).

    Args:
        octree: ocnn.Octree object
        depth: depth level to query

    Returns:
        xyz: (nnum, 3) tensor of integer coordinates in [0, 2^depth)
        batch_idx: (nnum,) tensor of batch indices
    """
    x, y, z, b = octree.xyzb(depth, nempty=False)
    xyz = torch.stack([x, y, z], dim=1).long()
    return xyz, b.long()


# ---------------------------------------------------------------------------
# octree2seq / seq2octree (ported from OctGPT utils/utils.py)
# ---------------------------------------------------------------------------

def octree2seq(octree, depth_low: int, depth_high: int,
               shift: bool = False) -> torch.Tensor:
    """Extract ground-truth split labels from octree as a flat sequence.

    Each node at depth d has children[d] which records the index of its
    first child (or -1 if it has none). child[i] >= 0 means child i exists.

    Args:
        octree: ocnn.Octree object
        depth_low: start depth (inclusive)
        depth_high: end depth (inclusive)
        shift: if True, scale labels from {0,1} to {-1,1}

    Returns:
        seq: (total_nnum,) long tensor of binary split labels
    """
    seq = torch.cat([octree.children[d] for d in range(depth_low, depth_high)])
    seq = (seq >= 0).long()

    if shift:
        seq = 2 * seq - 1
    return seq


def seq2octree(octree, seq: torch.Tensor, depth_low: int, depth_high: int,
               threshold: float = 0.0):
    """Reconstruct octree from predicted split sequence.

    The sequence `seq` contains per-node binary split labels across all
    depths in [depth_low, depth_high). Each value indicates whether the
    corresponding node should be split (i.e., has children).

    octree_split(label, depth) with label=1 creates all 8 children;
    label=0 makes the node a leaf.

    Args:
        octree: starting ocnn.Octree (modified in-place)
        seq: (total_nnum,) float tensor of split logits/probabilities
        depth_low: start depth (inclusive)
        depth_high: end depth (exclusive)
        threshold: values > threshold → split

    Returns:
        octree_out: new octree with structure up to depth_high
    """
    discrete_seq = (seq > threshold).long()

    octree_out = copy.deepcopy(octree)
    cur_nnum = 0
    for d in range(depth_low, depth_high):
        nnum_d = octree_out.nnum[d]
        label = discrete_seq[cur_nnum:cur_nnum + nnum_d].clone()
        cur_nnum += nnum_d
        if label.numel() == 0:
            label = torch.zeros(8, dtype=torch.long, device=octree.device)
        octree_out.octree_split(label, depth=d)
        octree_out.octree_grow(d + 1)
    return octree_out


# ---------------------------------------------------------------------------
# Per-depth split extraction (for recursive model targets)
# ---------------------------------------------------------------------------

def get_split_labels(octree, depth: int) -> torch.Tensor:
    """Get per-parent 8-way split labels at a single depth.

    Maps each child node at depth d+1 to its parent at depth d and
    determines which of the 8 octants it occupies.

    Args:
        octree: ocnn.Octree
        depth: parent depth d

    Returns:
        labels: (nnum_d, 8) float32 tensor, labels[p, c] = 1.0
                if octant c of parent p exists at depth d+1
    """
    nnum_d = octree.nnum[depth]
    device = octree.device
    labels = torch.zeros(nnum_d, 8, dtype=torch.float32, device=device)

    if depth + 1 > octree.depth:
        return labels

    nnum_next = octree.nnum[depth + 1]
    if nnum_next == 0:
        return labels

    # Get coordinates at both depths
    cx, cy, cz, cb = octree.xyzb(depth + 1, nempty=False)
    px, py, pz, pb = octree.xyzb(depth, nempty=False)

    # Parent coordinate = child coordinate // 2
    child_parent_x = cx // 2
    child_parent_y = cy // 2
    child_parent_z = cz // 2
    oct_x = cx % 2
    oct_y = cy % 2
    oct_z = cz % 2
    octant = oct_x + 2 * oct_y + 4 * oct_z  # 0-7

    # Build parent lookup: (batch, x, y, z) → parent_index
    parent_lookup = {}
    for p in range(nnum_d):
        key = (int(pb[p]), int(px[p]), int(py[p]), int(pz[p]))
        parent_lookup[key] = p

    # Match children to parents
    for c in range(nnum_next):
        pkey = (int(cb[c]), int(child_parent_x[c]),
                int(child_parent_y[c]), int(child_parent_z[c]))
        if pkey in parent_lookup:
            p = parent_lookup[pkey]
            labels[p, int(octant[c])] = 1.0

    return labels


# ---------------------------------------------------------------------------
# Helper: node ordering
# ---------------------------------------------------------------------------

def get_morton_ordered_nodes(octree, depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get nodes at a depth sorted by Morton (Z-order) code.

    Args:
        octree: ocnn.Octree
        depth: depth to query

    Returns:
        xyz_sorted: (nnum, 3) coordinates in Morton order
        sort_idx: (nnum,) indices for reordering to Morton order
    """
    xyz, batch = get_node_xyz(octree, depth)
    sort_idx = morton_order_indices(xyz)
    return xyz[sort_idx], sort_idx


def get_parent_children_xyz(octree, depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get parent xyz and all child xyz at consecutive depths.

    Args:
        octree: ocnn.Octree
        depth: parent depth

    Returns:
        parent_xyz: (nnum_d, 3) parent coordinates at depth
        all_children_xyz: (nnum_d, 8, 3) coordinates of all 8 children per parent
    """
    xyz, _ = get_node_xyz(octree, depth)
    all_children = child_xyz(xyz)  # (nnum_d, 8, 3)
    return xyz, all_children
