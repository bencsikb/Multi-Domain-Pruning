import numpy as np
import torch
from typing import Union, Tuple, Any, List

def denormalize(
    values: Union[float, np.ndarray, torch.Tensor], 
    value_range: Tuple[float],
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Denormalize input values from the normalized range [-1, 1] back to [min_val, max_val].

    Args:
        values (float | np.ndarray | torch.Tensor): The input value(s) to be denormalized.
        value_range: Tuple[float]:  The minimum and maximum value in the original range.

    Returns:
        float | np.ndarray | torch.Tensor: The denormalized value(s), preserving the input type.

    Raises:
        TypeError: If `values` is not a float, np.ndarray, or torch.Tensor.
    """
    min_val, max_val = value_range

    def scale(value: Any) -> Any:
        return ((value + 1) * (max_val - min_val)) / 2 + min_val

    if isinstance(values, torch.Tensor):
        denorm_values = scale(values)
        # denorm_values = torch.clamp(denorm_values, min_val, max_val)
    elif isinstance(values, np.ndarray):
        denorm_values = scale(values.astype(float))  # Ensure float dtype
        # denorm_values = np.clip(denorm_values, min_val, max_val)
    elif isinstance(values, (float, int)):
        denorm_values = scale(float(values))
        # denorm_values = max(min(denorm_values, max_val), min_val)
    else:
        raise TypeError(f"Unsupported input type: {type(values)}. Function expects float, np.ndarray, or torch.Tensor.")

    return denorm_values


def normalize(
    values: Union[int, float, np.ndarray, torch.Tensor], 
    value_range: Tuple[float],
    norm_range: Tuple[float] = (-1, 1)
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Normalize input values to a specified range [lower_bound, upper_bound].

    Args:
        values (int | float | np.ndarray | torch.Tensor): The input value(s) to be normalized.
        value_range: Tuple[float]:  The minimum and maximum value in the original range.
        norm_range: Tuple[float]: The lower and upper bound of the target range. Defaults to (-1, 1).

    Returns:
        float | np.ndarray | torch.Tensor: The normalized value(s), scaled to [lower_bound, upper_bound].
    
    Raises:
        TypeError: If `values` is not an int, float, np.ndarray, or torch.Tensor.
    """

    min_val, max_val = value_range
    norm_min, norm_max = norm_range

    # Avoid division by zero if min_val == max_val
    if max_val == min_val:
        raise ValueError("Normalization range cannot have min_val equal to max_val.")

    def scale(value: Any) -> Any:
        return ((value - min_val) / (max_val - min_val)) * (norm_max - norm_min) + norm_min

    if isinstance(values, torch.Tensor):
        normalized_values = scale(values)
    elif isinstance(values, np.ndarray):
        normalized_values = scale(values.astype(float))  # Ensure float dtype
    elif isinstance(values, (float, int)):  
        normalized_values = scale(float(values))
    else:
        raise TypeError(f"Unsupported input type: {type(values)}. Function expects int, float, np.ndarray, or torch.Tensor.")

    return normalized_values


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