"""
3D shape generation evaluation metrics.

Ported from OctGPT's metrics/evaluation_metrics.py and adapted for our
OctreeFractalGen pipeline. Computes standard geometry-level metrics
by comparing point clouds sampled from generated meshes against
reference point clouds.

Metrics:
  - 1-NNA (1-Nearest Neighbor Accuracy): classification-based metric.
    Closer to 50% means generated shapes are indistinguishable from real.
  - COV (Coverage): fraction of reference shapes matched to at least one
    generated shape. Higher is better.
  - MMD (Minimum Matching Distance): average distance from each generated
    shape to its nearest reference neighbor. Lower is better.
  - Diversity: histogram of per-sample minimum CD to training set. Detects
    mode collapse.

Usage:
    from src.utils.metrics import compute_all_metrics
    results = compute_all_metrics(sample_pcs, ref_pcs, batch_size=64)
"""

from typing import Dict, Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Chamfer Distance (CD)
# ---------------------------------------------------------------------------

def dist_chamfer(x: torch.Tensor, y: torch.Tensor):
    """Batched Chamfer distance between two point clouds.

    Adapted from AtlasNet / OctGPT implementation.

    Args:
        x: (B, N, D) first point cloud batch
        y: (B, M, D) second point cloud batch

    Returns:
        dist_left:  (B, N) distance from each point in x to nearest in y
        dist_right: (B, M) distance from each point in y to nearest in x
    """
    B, N, D = x.shape
    M = y.shape[1]

    # Compute pairwise distances via expand + broadcast
    xx = torch.bmm(x, x.transpose(2, 1))        # (B, N, N)
    yy = torch.bmm(y, y.transpose(2, 1))        # (B, M, M)
    zz = torch.bmm(x, y.transpose(2, 1))        # (B, N, M)

    # Extract diagonal for ||x_i||^2 and ||y_j||^2 terms
    diag_x = torch.arange(0, N, device=x.device).long()
    diag_y = torch.arange(0, M, device=y.device).long()
    rx = xx[:, diag_x, diag_x].unsqueeze(1).expand_as(xx)   # (B, N, N)
    ry = yy[:, diag_y, diag_y].unsqueeze(1).expand_as(yy)   # (B, M, M)

    # ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2<x_i, y_j>
    P = rx.transpose(2, 1) + ry - 2 * zz                    # (B, N, M)

    dist_left = P.min(dim=2)[0]    # (B, N) — nearest y for each x
    dist_right = P.min(dim=1)[0]   # (B, M) — nearest x for each y

    return dist_left, dist_right


# ---------------------------------------------------------------------------
# Earth Mover's Distance (EMD) — approximate via Hungarian algorithm
# ---------------------------------------------------------------------------

def emd_approx(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Approximate Earth Mover's Distance via linear sum assignment.

    Requires x and y to have the same number of points.

    Args:
        x: (B, N, D) first point cloud
        y: (B, N, D) second point cloud

    Returns:
        emd: (B,) EMD values
    """
    B, N, D = x.shape
    assert y.shape[1] == N, "EMD requires equal number of points"

    # Pairwise L2 distance: (B, N, N)
    x_exp = x.reshape(B, N, 1, D)
    y_exp = y.reshape(B, 1, N, D)
    dist = (x_exp - y_exp).norm(dim=-1, keepdim=False)  # (B, N, N)

    dist_np = dist.cpu().detach().numpy()
    emd_vals = []
    for i in range(B):
        d_i = dist_np[i]
        r_idx, c_idx = linear_sum_assignment(d_i)
        emd_vals.append(d_i[r_idx, c_idx].mean())

    emd = np.stack(emd_vals).reshape(-1)
    return torch.from_numpy(emd).to(x)


# ---------------------------------------------------------------------------
# Pairwise distance matrix (sample × reference)
# ---------------------------------------------------------------------------

def pairwise_cd_emd(
    sample_pcs: torch.Tensor,
    ref_pcs: torch.Tensor,
    batch_size: int = 64,
) -> tuple:
    """Compute pairwise CD and EMD between all sample-ref pairs.

    For each generated shape, compute distance to every reference shape.

    Args:
        sample_pcs: (N_sample, num_points, 3) generated point clouds
        ref_pcs:    (N_ref, num_points, 3) reference point clouds
        batch_size: ref batch size for memory efficiency

    Returns:
        all_cd:  (N_sample, N_ref) pairwise Chamfer distances
        all_emd: (N_sample, N_ref) pairwise EMD values
    """
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]

    all_cd = []
    all_emd = []

    for s_idx in tqdm(range(N_sample), desc="Pairwise CD/EMD"):
        sample_i = sample_pcs[s_idx:s_idx + 1]  # (1, N, 3)

        cd_list = []
        emd_list = []
        for r_start in range(0, N_ref, batch_size):
            r_end = min(N_ref, r_start + batch_size)
            ref_batch = ref_pcs[r_start:r_end]           # (B, N, 3)
            B_ref = ref_batch.size(0)

            # Expand single sample to match ref batch
            sample_exp = sample_i.expand(B_ref, -1, -1).contiguous()

            dl, dr = dist_chamfer(sample_exp, ref_batch)
            cd = (dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1)
            cd_list.append(cd)

            emd_batch = emd_approx(sample_exp, ref_batch)
            emd_list.append(emd_batch.view(1, -1))

        all_cd.append(torch.cat(cd_list, dim=1))     # (1, N_ref)
        all_emd.append(torch.cat(emd_list, dim=1))

    all_cd = torch.cat(all_cd, dim=0)    # (N_sample, N_ref)
    all_emd = torch.cat(all_emd, dim=0)
    return all_cd, all_emd


# ---------------------------------------------------------------------------
# 1-NNA (1-Nearest Neighbor Accuracy)
# ---------------------------------------------------------------------------

def _knn(
    M_xx: torch.Tensor,
    M_xy: torch.Tensor,
    M_yy: torch.Tensor,
    k: int = 1,
    sqrt: bool = False,
) -> Dict[str, float]:
    """K-nearest neighbor classification between two distributions.

    Uses precomputed pairwise distance matrices.

    Args:
        M_xx: (N, N) pairwise distances within set X (reference)
        M_xy: (N, M) cross distances X ↔ Y (reference ↔ sample)
        M_yy: (M, M) pairwise distances within set Y (sample)
        k: number of neighbors
        sqrt: whether to sqrt distances before comparison

    Returns:
        dict with 'acc' (1-NN accuracy), 'precision', 'recall'
    """
    n0 = M_xx.size(0)  # N_ref
    n1 = M_yy.size(0)  # N_sample
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(M_xx)

    # Build the full distance matrix
    M_top = torch.cat((M_xx, M_xy), dim=1)           # (N_ref, N_ref+N_sample)
    M_bot = torch.cat((M_xy.t(), M_yy), dim=1)       # (N_sample, N_ref+N_sample)
    M = torch.cat((M_top, M_bot), dim=0)              # (total, total)

    if sqrt:
        M = M.abs().sqrt()

    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(M_xx))).topk(
        k, dim=0, largest=False)

    count = torch.zeros(n0 + n1).to(M_xx)
    for i in range(k):
        count = count + label.index_select(0, idx[i])

    pred = (count >= (float(k) / 2) * torch.ones(n0 + n1).to(M_xx)).float()

    tp = (pred * label).sum()
    fp = (pred * (1 - label)).sum()
    fn = ((1 - pred) * label).sum()
    tn = ((1 - pred) * (1 - label)).sum()

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    acc = (tp + tn) / (n0 + n1)

    return {
        'acc': acc.item(),
        'precision': precision.item(),
        'recall': recall.item(),
    }


# ---------------------------------------------------------------------------
# COV & MMD (Coverage & Minimum Matching Distance)
# ---------------------------------------------------------------------------

def _lgan_mmd_cov(all_dist: torch.Tensor) -> Dict[str, float]:
    """Compute MMD and Coverage from pairwise distance matrix.

    Args:
        all_dist: (N_sample, N_ref) pairwise distances

    Returns:
        dict with 'lgan_mmd', 'lgan_cov', 'lgan_mmd_smp'
    """
    N_sample, N_ref = all_dist.shape

    # MMD: mean of minimum distances
    min_val_from_sample, min_idx = torch.min(all_dist, dim=1)  # (N_sample,)
    min_val_from_ref, _ = torch.min(all_dist, dim=0)           # (N_ref,)

    mmd = min_val_from_ref.mean()
    mmd_smp = min_val_from_sample.mean()

    # Coverage: fraction of reference shapes matched
    cov = float(min_idx.unique().numel()) / float(N_ref)

    return {
        'lgan_mmd': mmd.item(),
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp.item(),
    }


def compute_cov_mmd(
    sample_pcs: torch.Tensor,
    ref_pcs: torch.Tensor,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Compute COV and MMD metrics (CD and EMD variants).

    Args:
        sample_pcs: (N_sample, N, 3) generated point clouds
        ref_pcs:    (N_ref, N, 3) reference point clouds
        batch_size: batch size for pairwise computation

    Returns:
        dict with keys: COV-CD, MMD-CD, COV-EMD, MMD-EMD
    """
    M_rs_cd, M_rs_emd = pairwise_cd_emd(ref_pcs, sample_pcs, batch_size)

    results = {}
    res_cd = _lgan_mmd_cov(M_rs_cd.t())
    results.update({f"COV-CD" if 'cov' in k else f"MMD-CD": v
                    for k, v in res_cd.items() if 'mmd' in k.lower() or 'cov' in k.lower()})
    res_emd = _lgan_mmd_cov(M_rs_emd.t())
    results.update({f"COV-EMD" if 'cov' in k else f"MMD-EMD": v
                    for k, v in res_emd.items() if 'mmd' in k.lower() or 'cov' in k.lower()})

    return results


def compute_1_nna(
    sample_pcs: torch.Tensor,
    ref_pcs: torch.Tensor,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Compute 1-NNA metrics (CD and EMD variants).

    Closer to 50% accuracy means generated shapes are indistinguishable
    from real shapes.

    Args:
        sample_pcs: (N_sample, N, 3)
        ref_pcs:    (N_ref, N, 3)
        batch_size: batch size

    Returns:
        dict with keys: 1-NNA-CD-acc, 1-NNA-EMD-acc
    """
    # Cross distances: ref ↔ sample
    M_rs_cd, M_rs_emd = pairwise_cd_emd(ref_pcs, sample_pcs, batch_size)

    # Within-set distances
    # For ref ↔ ref: re-use pairwise function
    M_rr_cd, M_rr_emd = pairwise_cd_emd(ref_pcs, ref_pcs, batch_size)
    M_ss_cd, M_ss_emd = pairwise_cd_emd(sample_pcs, sample_pcs, batch_size)

    results = {}
    one_nn_cd = _knn(M_rr_cd, M_rs_cd, M_ss_cd, k=1, sqrt=False)
    results.update({f"1-NNA-CD-{k}": v for k, v in one_nn_cd.items()})
    one_nn_emd = _knn(M_rr_emd, M_rs_emd, M_ss_emd, k=1, sqrt=False)
    results.update({f"1-NNA-EMD-{k}": v for k, v in one_nn_emd.items()})

    return results


# ---------------------------------------------------------------------------
# Top-level metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    sample_pcs: torch.Tensor,
    ref_pcs: torch.Tensor,
    batch_size: int = 64,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        sample_pcs: (N_sample, N_points, 3) generated point clouds
        ref_pcs:    (N_ref, N_points, 3) reference point clouds
        batch_size: batch size for pairwise computation
        device: torch device (auto-detect if None)
        verbose: print progress

    Returns:
        dict with 1-NNA-CD-acc, 1-NNA-EMD-acc, COV-CD, COV-EMD,
             MMD-CD, MMD-EMD
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sample_pcs = sample_pcs.to(device)
    ref_pcs = ref_pcs.to(device)

    if verbose:
        print(f"Computing metrics: {sample_pcs.shape[0]} samples "
              f"vs {ref_pcs.shape[0]} references on {device}")

    results = {}
    results.update(compute_1_nna(sample_pcs, ref_pcs, batch_size))
    results.update(compute_cov_mmd(sample_pcs, ref_pcs, batch_size))

    if verbose:
        print("\n--- Evaluation Results ---")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return results


# ---------------------------------------------------------------------------
# Utility: point cloud sampling from mesh
# ---------------------------------------------------------------------------

def sample_points_from_mesh(
    mesh_path: str,
    num_points: int = 2048,
    scale_to_unit: bool = True,
) -> np.ndarray:
    """Sample surface points from a mesh file.

    Args:
        mesh_path: path to .obj/.ply file
        num_points: number of points to sample
        scale_to_unit: normalize to unit sphere

    Returns:
        points: (num_points, 3) numpy array
    """
    import trimesh

    mesh = trimesh.load(mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    points, _ = trimesh.sample.sample_surface(mesh, num_points)

    if scale_to_unit:
        centroid = points.mean(axis=0)
        points = points - centroid
        max_dist = np.linalg.norm(points, axis=1).max()
        points = points / max_dist

    return points.astype(np.float32)


def load_reference_pointclouds(
    filelist_path: str,
    data_root: str,
    num_points: int = 2048,
    cache_path: Optional[str] = None,
) -> torch.Tensor:
    """Load reference point clouds from a filelist.

    Each line in the filelist is `category_id/object_hash`. Loads
    pointcloud.npz from `{data_root}/{category_id}/{object_hash}/`.

    Args:
        filelist_path: path to filelist (e.g., train_airplane.txt)
        data_root: root directory of processed data
        num_points: points to keep per shape
        cache_path: if provided, cache the stacked tensor here

    Returns:
        ref_pcs: (N, num_points, 3) tensor
    """
    import os

    # Try loading from cache
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path)

    with open(filelist_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    all_points = []
    for line in tqdm(lines, desc="Loading reference PCs"):
        pc_path = os.path.join(data_root, line, "pointcloud.npz")
        if not os.path.exists(pc_path):
            print(f"Warning: missing {pc_path}")
            continue
        raw = np.load(pc_path)
        pts = raw['points'].astype(np.float32)

        # Random sample to fixed size
        if pts.shape[0] > num_points:
            idx = np.random.choice(pts.shape[0], num_points, replace=False)
            pts = pts[idx]
        elif pts.shape[0] < num_points:
            # Pad with resampling
            idx = np.random.choice(pts.shape[0], num_points, replace=True)
            pts = pts[idx]

        # Normalize to unit sphere
        centroid = pts.mean(axis=0)
        pts = pts - centroid
        max_dist = np.linalg.norm(pts, axis=1).max()
        if max_dist > 0:
            pts = pts / max_dist

        all_points.append(pts)

    ref_pcs = torch.from_numpy(np.stack(all_points))

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(ref_pcs, cache_path)

    return ref_pcs


def compute_diversity(
    sample_pcs: torch.Tensor,
    ref_pcs: torch.Tensor,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute per-sample minimum CD to reference set (diversity check).

    Args:
        sample_pcs: (N_sample, N, 3)
        ref_pcs:    (N_ref, N, 3)
        batch_size: batch size

    Returns:
        min_cd: (N_sample,) minimum CD for each generated shape
    """
    min_cd_list = []
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]

    for s_idx in range(N_sample):
        sample_i = sample_pcs[s_idx:s_idx + 1]

        cd_vals = []
        for r_start in range(0, N_ref, batch_size):
            r_end = min(N_ref, r_start + batch_size)
            ref_batch = ref_pcs[r_start:r_end]
            B_ref = ref_batch.size(0)
            sample_exp = sample_i.expand(B_ref, -1, -1).contiguous()

            dl, dr = dist_chamfer(sample_exp, ref_batch)
            cd = (dl.mean(dim=1) + dr.mean(dim=1))
            cd_vals.append(cd)

        all_cd_for_sample = torch.cat(cd_vals)
        min_cd_list.append(all_cd_for_sample.min().item())

    return np.array(min_cd_list)
