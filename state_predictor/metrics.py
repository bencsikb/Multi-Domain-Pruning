import numpy as np
import torch
from typing import Union, Tuple, Any, List

from utils.common_utils import denormalize, normalize


def calculate_metrics(gt, pred, margin=0.05) -> dict:
    """
    Calculate standard regression metrics from sklearn and customized metrics for evaluating SPN.
    :param gt: ground truth - tensor
    :param pred: predicted values - tensor
    :param margin: for calculating accuracy with a margin - denormalized value
    :return: list of the calculated metrics
            - margin_accuracy: proportion of the denormalized predictions falling into the margin
            - negsign_recall: proportion of the negative gts predicted with a negative sign as well
                              (in dperf the negative sign means performance improvement)
            - mean_absoute_error: with sklearn on denormalized data
            - max_error: with sklearn on denormalized data
            - mean_squared_error: with sklearn on denormalized data
            - r2_score: with sklearn on denormalized data
    """
    from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score


    metrics = {}

    gt, pred = denormalize(gt, (0.0, 1.0)), denormalize(pred, (0.0, 1.0))

    # Handcrafted metric: margin accuracy
    margin_bool = torch.abs(gt - pred) <= margin
    margin_accuracy = margin_bool.float().mean()
    metrics["margin_accuracy"] = margin_accuracy.item()

    # Convert to numpy for sklearn metrics
    gt, pred = gt.cpu().detach().numpy(), pred.cpu().detach().numpy()

    metrics["mae"] = mean_absolute_error(gt, pred)
    metrics["max_error"] = max_error(gt, pred)
    metrics["mse"] = mean_squared_error(gt, pred)
    metrics["r2"] = r2_score(gt, pred)

    return metrics


def calculate_transformer_metrics(gt, pred, margin=0.05) -> dict:
    """
    Calculates regression metrics for variable-length sequence data.
    Args:
        gt:       (B, T, 2) - ground truth
        pred:     (B, T, 2) - predictions
        attention_mask: (B, T) - 1 for valid, 0 for padded
        margin:   allowed absolute error for margin_accuracy (on denormalized scale)
    """
    from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score

    # Flatten and filter using attention mask
    B, T, D = gt.shape
    gt = gt.reshape(-1, D)
    pred = pred.reshape(-1, D)

    # Denormalize if needed
    valid_gt = denormalize(gt, (0.0, 1.0))
    valid_pred = denormalize(pred, (0.0, 1.0))

    # Margin accuracy
    margin_bool = torch.abs(valid_gt - valid_pred) <= margin
    margin_accuracy = margin_bool.float().mean().item()

    # Convert to NumPy
    valid_gt_np = valid_gt.cpu().detach().numpy()
    valid_pred_np = valid_pred.cpu().detach().numpy()

    return {
        "margin_accuracy": margin_accuracy,
        "mae": mean_absolute_error(valid_gt_np, valid_pred_np),
        "max_error_spars": max_error(valid_gt_np[:, 0], valid_pred_np[:, 0]),
        "max_error_dmap": max_error(valid_gt_np[:, 1], valid_pred_np[:, 1]),
        "mse": mean_squared_error(valid_gt_np, valid_pred_np),
        "r2": r2_score(valid_gt_np, valid_pred_np),
    }