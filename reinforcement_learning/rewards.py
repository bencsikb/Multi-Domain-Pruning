import torch
def reward_function_proposed( spars, Tspars, dmap, Tdmap, spars_coeff, dmap_coeff, beta=5):
    """
    Compute reward based on error and sparsity deviation from target thresholds.

    Args:
        dmap (Tensor): mAP deterioration (error).
        Tdmap (float): Target/acceptable max deterioration.
        spars (Tensor): Sparsity (e.g. fraction of pruned params).
        Tspars (float): Desired sparsity.
        dmap_coeff (float): Weight of the error term.
        spars_coeff (float): Weight of the sparsity term.
        device: Torch device.
        beta (float): Reward scaling factor.

    Returns:
        Tensor: Reward values (negative for penalty).
    """
    term1 = torch.clamp((dmap - Tdmap) / (1 - Tdmap), min=0)
    term2 = torch.clamp(1 - spars / Tspars, min=0)
    reward = -beta * (dmap_coeff * term1 + spars_coeff * term2)
    return reward


def reward_function_purl(spars, Tspars, dmap, Tmap, map_before, beta=5):
    """
    Compute reward for PURL method based on post-pruning mAP and sparsity.

    Args:
        dmap (Tensor): mAP deterioration (1 - mAP_after / mAP_before).
        Tmap (float): Target final mAP.
        spars (Tensor): Final sparsity (fraction pruned).
        Tspars (float): Target sparsity.
        map_before (float): Baseline mAP before pruning.
        device: Torch device.
        beta (float): Reward scaling factor.

    Returns:
        Tensor: Reward values (negative penalty).
    """
    map_after = (1 - dmap) * map_before
    term1 = torch.clamp(1 - map_after / Tmap, min=0)
    term2 = torch.clamp(1 - spars / Tspars, min=0)
    reward = -beta * (term1 + term2)
    return reward


def reward_function_amc(dmap, spars, init_params=63980766):
    """
    Compute reward for AMC (AutoML for Model Compression) style pruning.

    Args:
        dmap (Tensor): mAP deterioration.
        spars (Tensor): Fraction of model pruned.
        device: Torch device.
        init_params (int): Total number of initial parameters in the model.

    Returns:
        Tensor: Reward value.
    """
    epsilon = 1e-10
    reward = -dmap * torch.log((spars + epsilon) * init_params)
    return reward
