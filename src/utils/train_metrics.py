"""CoarseFineOctGPT 训练指标计算工具.

实现指南要求的分层指标:
  - split: acc / positive precision / recall / F1 / pos_rate
  - VQ: top1 / top5 / full-code exact rate / hamming / entropy / perplexity
  - DDP 聚合用 sum/count (避免先算 F1 再平均)

所有函数返回 dict, key 不带前缀 (由调用方加 train/ val/ 前缀).
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Split (二分类 0/1) 指标
# ---------------------------------------------------------------------------

def compute_split_metrics(logits, targets, valid_mask=None):
    """二分类 split 指标 (指南第2节).

    Args:
        logits: (N, 2) split logits
        targets: (N,) long {0,1}
        valid_mask: (N,) bool, True=参与计算 (通常=masked 位置)

    Returns:
        dict: acc, pos_precision, pos_recall, pos_f1,
              target_pos_rate, pred_pos_rate, tp, fp, fn, count
    """
    if valid_mask is not None:
        logits = logits[valid_mask]
        targets = targets[valid_mask]
    if logits.shape[0] == 0:
        return _empty_split_metrics()

    preds = logits.argmax(dim=-1)  # (N,)
    targets = targets.long()

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    total = tp + fp + fn + tn

    acc = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    target_pos_rate = (tp + fn) / max(total, 1)
    pred_pos_rate = (tp + fp) / max(total, 1)

    return {
        'acc': acc,
        'pos_precision': precision,
        'pos_recall': recall,
        'pos_f1': f1,
        'target_pos_rate': target_pos_rate,
        'pred_pos_rate': pred_pos_rate,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'count': total,
    }


def _empty_split_metrics():
    return {
        'acc': 0.0, 'pos_precision': 0.0, 'pos_recall': 0.0,
        'pos_f1': 0.0, 'target_pos_rate': 0.0, 'pred_pos_rate': 0.0,
        'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'count': 0,
    }


# ---------------------------------------------------------------------------
# VQ (per-dim 二分类, 32 维) 指标
# ---------------------------------------------------------------------------

def compute_vq_metrics(logits, targets, valid_mask=None, num_codes=2):
    """VQ per-dim 指标 (指南第3节).

    Args:
        logits: (N, vq_groups * vq_size) 或 (N, vq_groups, vq_size)
        targets: (N, vq_groups) long
        valid_mask: (N,) bool
        num_codes: 每组 code 数 (BSQ=2)

    Returns:
        dict: top1, top5, full_code_exact_rate,
              hamming_per_node, hamming_per_dim,
              code_entropy, code_perplexity, unique_code_count,
              token_count
    """
    if valid_mask is not None:
        logits = logits[valid_mask]
        targets = targets[valid_mask]
    if logits.shape[0] == 0:
        return _empty_vq_metrics()

    N = logits.shape[0]
    vq_groups = targets.shape[1]
    vq_size = num_codes

    # reshape logits -> (N, vq_groups, vq_size)
    logits = logits.reshape(N, vq_groups, vq_size)

    # top-k (per-dim)
    max_k = min(5, vq_size - 1) if vq_size > 1 else 1
    _, pred_topk = logits.topk(max_k, dim=-1)  # (N, G, max_k)
    correct = pred_topk.eq(targets.unsqueeze(-1).expand_as(pred_topk))
    top1 = correct[:, :, 0].float().mean().item()
    top5 = correct[:, :max_k].any(dim=-1).float().mean().item() if max_k >= 5 else top1

    # full-code exact rate & hamming
    pred_argmax = logits.argmax(dim=-1)  # (N, G)
    correct_dim = pred_argmax.eq(targets)  # (N, G)
    full_exact = correct_dim.all(dim=-1).float().mean().item()
    hamming_per_node = (~correct_dim).float().sum(dim=-1).mean().item()
    hamming_per_dim = (~correct_dim).float().mean().item()

    # code entropy / perplexity / unique (跨所有 dim 的 bit 分布)
    flat = targets.reshape(-1).long()
    hist = torch.bincount(flat, minlength=vq_size).float()
    prob = hist / hist.sum().clamp_min(1)
    entropy = -(prob * (prob + 1e-12).log()).sum().item()
    perplexity = float(torch.exp(torch.tensor(entropy)))
    unique_count = int((hist > 0).sum().item())

    return {
        'top1': top1,
        'top5': top5,
        'full_code_exact_rate': full_exact,
        'hamming_per_node': hamming_per_node,
        'hamming_per_dim': hamming_per_dim,
        'code_entropy': entropy,
        'code_perplexity': perplexity,
        'unique_code_count': unique_count,
        'token_count': N,
    }


def _empty_vq_metrics():
    return {
        'top1': 0.0, 'top5': 0.0, 'full_code_exact_rate': 0.0,
        'hamming_per_node': 0.0, 'hamming_per_dim': 0.0,
        'code_entropy': 0.0, 'code_perplexity': 0.0,
        'unique_code_count': 0, 'token_count': 0,
    }


# ---------------------------------------------------------------------------
# 工程健康指标
# ---------------------------------------------------------------------------

def compute_grad_norm(model, prefix_filter=None):
    """计算梯度范数, 可按参数名前缀分组 (指南第10节).

    Args:
        model: nn.Module
        prefix_filter: dict {prefix_name: param_prefix}, 如
            {'coarse': 'coarse.', 'fine': 'fine.', 'prefix_proj': 'prefix_'}

    Returns:
        dict: {name: grad_norm}, total 在 'total' key
    """
    norms = {}
    if prefix_filter is None:
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.detach().data.norm(2).item() ** 2
        norms['total'] = total ** 0.5
        return norms

    # 分组
    group_sq = {name: 0.0 for name in prefix_filter}
    group_sq['total'] = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        n = p.grad.detach().data.norm(2).item() ** 2
        group_sq['total'] += n
        for gname, gprefix in prefix_filter.items():
            if name.startswith(gprefix):
                group_sq[gname] += n
                break
    for k in group_sq:
        norms[k] = group_sq[k] ** 0.5
    return norms


# ---------------------------------------------------------------------------
# DDP 聚合
# ---------------------------------------------------------------------------

def ddp_all_reduce_dict(metrics_dict, device):
    """对 dict 中的标量值做 DDP all_reduce (SUM).

    用于聚合 TP/FP/FN/count 等. 调用方需自行用聚合后的值计算 F1.
    """
    if not torch.distributed.is_initialized():
        return metrics_dict
    out = {}
    for k, v in metrics_dict.items():
        if isinstance(v, (int, float)):
            t = torch.tensor(float(v), device=device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            out[k] = t.item()
        else:
            out[k] = v
    return out


def aggregate_split_metrics_from_counts(tp, fp, fn, tn, count):
    """从聚合后的 TP/FP/FN/TN/count 计算 acc/precision/recall/F1."""
    total = tp + fp + fn + tn
    acc = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        'acc': acc, 'pos_precision': precision, 'pos_recall': recall,
        'pos_f1': f1,
        'target_pos_rate': (tp + fn) / max(total, 1),
        'pred_pos_rate': (tp + fp) / max(total, 1),
    }
