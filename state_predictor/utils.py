import torch
from typing import Union, List, Any



def denormalize(
    values: Union[float, List[float], torch.Tensor], 
    min_val: float, 
    max_val: float
) -> Union[float, List[float], torch.Tensor]:
    """
    Denormalize input values from the normalized range [-1, 1] back to [min_val, max_val].

    Args:
        values (float | list[float] | torch.Tensor): The input value(s) to be denormalized.
        min_val (float): The minimum value of the original range.
        max_val (float): The maximum value of the original range.

    Returns:
        float | list[float] | torch.Tensor: The denormalized value(s), preserving the input type.

    Raises:
        TypeError: If `values` is not a float, list of floats, or torch.Tensor.
    """

    def scale(value: Any) -> Any:
        return ((value + 1) * (max_val - min_val)) / 2 + min_val

    if isinstance(values, torch.Tensor):
        denorm_values = scale(values)  
        denorm_values = torch.clamp(denorm_values, min_val, max_val)
    elif isinstance(values, list):
        denorm_values = [scale(float(v)) for v in values]
        denorm_values = [max(min(v, max_val), min_val) for v in denorm_values]  
    elif isinstance(values, float):
        denorm_values = scale(float(values))
        denorm_values = max(min(denorm_values, max_val), min_val)  
    else:
        raise TypeError(f"Unsupported input type: {type(values)}. Function denormalize() expects float, list, or torch.Tensor.")

    return denorm_values



def normalize(
    values: Union[int, float, List[float], torch.Tensor], 
    min_val: float, 
    max_val: float, 
    lower_bound: float = -1, 
    upper_bound: float = 1
) -> Union[float, List[float], torch.Tensor]:
    """
    Normalize input values to a specified range [lower_bound, upper_bound].

    Args:
        values (float | list[float] | torch.Tensor): The input value(s) to be normalized.
        min_val (float): The minimum value in the original range.
        max_val (float): The maximum value in the original range.
        lower_bound (float, optional): The lower bound of the target range. Defaults to -1.
        upper_bound (float, optional): The upper bound of the target range. Defaults to 1.

    Returns:
        float | list[float] | torch.Tensor: The normalized value(s), scaled to [lower_bound, upper_bound].
    
    Raises:
        TypeError: If `values` is not an int, float, list of floats, or torch.Tensor.
    """

    #TODO Avoid division by zero if min_val == max_val 

    def scale(value: Any) -> Any:
        return ((value - min_val) / (max_val - min_val)) * (upper_bound - lower_bound) + lower_bound

    # Apply scaling based on input type
    if isinstance(values, torch.Tensor):
        normalized_values = scale(values)
    elif isinstance(values, List):
        normalized_values = [scale(float(v)) for v in values]
    elif isinstance(values, (float, int)):  
        normalized_values = scale(float(values))
    else:
        raise TypeError(f"Unsupported input type: {type(values)}. Function normalize() expects int, float, list, or torch.Tensor.")

    return normalized_values