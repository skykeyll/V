from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from sklearn import metrics


def sigmoid_probs(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.size(-1) == 1:
        return torch.sigmoid(logits.squeeze(-1))
    if logits.ndim == 1:
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=-1)[..., 1]


def focal_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    logits = logits.float().view(-1)
    targets = targets.float().view(-1)
    probs = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    modulating = (1 - p_t).pow(gamma)
    return (alpha_factor * modulating * ce_loss).mean()


def tversky_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
) -> torch.Tensor:
    logits = logits.float().view(-1)
    targets = targets.float().view(-1)
    probs = torch.sigmoid(logits)
    true_positive = (probs * targets).sum()
    false_positive = (probs * (1.0 - targets)).sum()
    false_negative = ((1.0 - probs) * targets).sum()
    tversky_index = (true_positive + smooth) / (
        true_positive + alpha * false_positive + beta * false_negative + smooth
    )
    return 1.0 - tversky_index


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor],
    focal_alpha: float,
    focal_gamma: float,
    focal_weight: float,
    loss_mode: str = "bce_focal",
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    tversky_weight: float = 0.3,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    logits = logits.view(-1)
    targets = targets.float().view(-1)
    if label_smoothing > 0:
        targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing

    if loss_mode == "bce":
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    if loss_mode == "focal":
        return focal_binary_loss(logits, targets, alpha=focal_alpha, gamma=focal_gamma)

    if loss_mode == "bce_focal":
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        focal = focal_binary_loss(logits, targets, alpha=focal_alpha, gamma=focal_gamma)
        return bce + focal_weight * focal

    if loss_mode == "bce_tversky":
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        tversky = tversky_binary_loss(logits, targets, alpha=tversky_alpha, beta=tversky_beta)
        return bce + tversky_weight * tversky

    raise ValueError(f"Unsupported loss_mode: {loss_mode}")


def find_best_threshold(y_true: List[int], y_score: List[float]) -> float:
    best_threshold = 0.5
    best_mcc = -1.0
    for threshold in [i / 100 for i in range(5, 96)]:
        y_pred = [1 if p >= threshold else 0 for p in y_score]
        mcc = metrics.matthews_corrcoef(y_true, y_pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold
    return best_threshold


def compute_metrics(y_true: List[int], y_score: List[float], threshold: float = None) -> Dict[str, float]:
    if threshold is None:
        threshold = find_best_threshold(y_true, y_score)
    y_pred = [1 if p >= threshold else 0 for p in y_score]
    precisions, recalls, _ = metrics.precision_recall_curve(y_true, y_score)
    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_true, y_pred, zero_division=0),
        "f1": metrics.f1_score(y_true, y_pred, zero_division=0),
        "mcc": metrics.matthews_corrcoef(y_true, y_pred),
        "auroc": metrics.roc_auc_score(y_true, y_score),
        "auprc": metrics.auc(recalls, precisions),
        "threshold": threshold,
    }
