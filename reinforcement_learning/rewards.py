import torch

from typing import List 

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



class PartialTargetReward:
    def __init__(
        self,
        prunable_layers: List[torch.nn.Module],
        T_spars_total: float,
        Tdmap: torch.Tensor,
        beta: float,
        spars_coeff: float,
        dmap_coeff: float,
        device=None
    ):
        self.Tdmap = Tdmap
        self.beta = beta
        self.dmap_coeff = dmap_coeff
        self.spars_coeff = spars_coeff

        param_counts = [sum(p.numel() for p in l.parameters()) for _, l in prunable_layers]
        total = sum(param_counts)
        cum_ratio = torch.tensor(
            [sum(param_counts[:i+1]) / total for i in range(len(param_counts))],
            dtype=torch.float32, device=device
        )
        self.layer_Tspars = T_spars_total * cum_ratio

    def get_reward(self, layer_idx, spars, dmap, eps=1e-12):
        Ts = torch.clamp(self.layer_Tspars[layer_idx], min=eps).to(dmap.device)
        Td = self.Tdmap
        term1 = torch.clamp((dmap - Td) / (1 - Td + eps), min=0)
        term2 = torch.clamp(1 - spars / Ts, min=0)
        return -self.beta * (self.dmap_coeff * term1 + self.spars_coeff * term2)
    

def sigmoid_gate(dmap, Td, tau=0.02):
    # ~1 if dmap << Td, ~0 if dmap >> Td (soft priority gating)
    return torch.sigmoid((Td - dmap) / tau)

def _sat01(x):  # clamp to [0,1]
    return torch.clamp(x, 0.0, 1.0)

class SigmoidGateReward:  # still hard-gated per spec
    def __init__(self, prunable_layers, T_spars_total, Tdmap, beta, dmap_coeff, spars_coeff,
                 device=None, k_exp: float = 8.0, margin: float = 1.0):
        """
        k_exp : steepness of penalty above Td (bigger => harsher just above Td).
        margin: separation margin; ensures any 'above Td' reward < any 'below Td' reward.
                (effective min penalty above Td is margin*dmap_coeff*beta)
        """
        self.Tdmap = Tdmap if isinstance(Tdmap, torch.Tensor) else torch.tensor(Tdmap, dtype=torch.float32, device=device)
        self.beta = beta
        self.dmap_coeff = dmap_coeff
        self.spars_coeff = spars_coeff
        self.k_exp = k_exp
        self.margin = margin

        # per-layer Tspars scaled by parameter counts
        param_counts = [sum(p.numel() for p in l.parameters()) for _, l in prunable_layers]
        total = sum(param_counts)
        cum_ratio = torch.tensor([sum(param_counts[:i+1]) / total for i in range(len(param_counts))],
                                 dtype=torch.float32, device=device)
        self.layer_Tspars = T_spars_total * cum_ratio

    def get_reward(self, layer_idx: int, spars: torch.Tensor, dmap: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        device = dmap.device
        Ts = torch.clamp(self.layer_Tspars[layer_idx], min=eps).to(device)
        Td = self.Tdmap.to(device)

        # --- Gate by accuracy (hard priority)
        below = (dmap <= Td).to(dmap.dtype)          # 1 if at/below Td, else 0
        above = 1.0 - below

        # --- Penalty when dmap > Td (sparsity ignored, steep rise)
        # Use relative-to-Td normalization so even tiny exceedances hurt (especially when Td is small).
        over_rel = torch.clamp((dmap - Td) / (Td + eps), min=0.0)   # 0 at Td, 1 at 2*Td, ...
        pen_d = self.dmap_coeff * (self.margin + torch.expm1(self.k_exp * over_rel))  # ≥ margin*dmap_coeff

        # --- Gain when dmap <= Td (monotone in spars, saturates at Ts)
        gain_s = self.spars_coeff * _sat01(spars / (Ts + eps))      # 0..1; plateau at/above Ts

        # --- Final reward (maximize)
        #   - Below Td:  +beta * spars_gain
        #   - Above Td:  -beta * penalty (independent of spars)
        reward = self.beta * (below * gain_s) - self.beta * (above * pen_d)
        return reward