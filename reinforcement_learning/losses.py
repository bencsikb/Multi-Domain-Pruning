import torch
import torch.nn as nn
from typing import List, Optional


def get_discounted_reward(
    rewards: List[torch.Tensor],
    gamma: float
) -> torch.Tensor:
    """
    Compute the discounted cumulative rewards.

    Args:
        rewards (List[torch.Tensor]): List of rewards (each a 0D tensor or scalar tensor).
        values (torch.Tensor): Value function predictions (not used here directly).
        gamma (float): Discount factor in [0, 1].

    Returns:
        torch.Tensor: Normalized discounted cumulative rewards.
    """
    disc_rewards = []
    val = torch.zeros_like(rewards[0])  # ensure tensor type

    for i in reversed(range(len(rewards))):
        val = rewards[i] + gamma * val
        disc_rewards.insert(0, val)

    disc_rewards = torch.stack(disc_rewards)
    #out = (disc_rewards - disc_rewards.mean()) / (disc_rewards.std() + 1e-8)
    return disc_rewards


def get_advantage(
    rewards: List[torch.Tensor],
    values: List[torch.Tensor],
    gamma: float = 0.99
) -> torch.Tensor:
    """
    Compute TD(0) advantage:
        A_t = r_t + gamma * V_{t+1} - V_t
    """

    rewards = torch.stack(rewards)        # [T, B, 1]
    values = torch.stack(values)          # [T, B, 1]

    # Shift values to get V_{t+1}
    next_values = torch.zeros_like(values)
    next_values[:-1] = values[1:]
    next_values[-1] = 0.0  # terminal state

    td_target = rewards + gamma * next_values
    advantage = td_target - values

    # Detach critic from actor update
    advantage = advantage.detach()

    # Normalize (important!)
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    return advantage


class CriticLoss(nn.Module):
    """
    Computes MSE loss between discounted rewards and value predictions.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        rewards: List[torch.Tensor],
        values: List[torch.Tensor],
        gamma: float = 0.99
    ) -> torch.Tensor:
        """
        Args:
            rewards (List[torch.Tensor]): List of reward scalars.
            values (torch.Tensor): Estimated value function.
            gamma (float): Discount factor.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        rewards = torch.stack(rewards)   # [T, B, 1]
        values = torch.stack(values)     # [T, B, 1]

        next_values = torch.zeros_like(values)
        next_values[:-1] = values[1:]
        next_values[-1] = 0.0

        td_target = rewards + gamma * next_values

        loss = torch.mean((values - td_target.detach()) ** 2)
        return loss


class ActorLoss(nn.Module):
    """
    Computes policy loss using advantage-weighted log-probabilities and entropy regularization.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        rewards: List[torch.Tensor],
        values: List[torch.Tensor],
        policies: torch.Tensor,
        log_probs: List[torch.Tensor],
        entropies: Optional[List[torch.Tensor]] = None,
        # policy_masks: Optional[List[torch.Tensor]] = None,
        ent_coef: float = 0.5,
        gamma: float = 0.99
    ) -> torch.Tensor:
        """
        Args:
            rewards (List[torch.Tensor]): List of rewards.
            values (torch.Tensor): Value estimates.
            policies (torch.Tensor): Action probabilities (not used in this version).
            log_probs (torch.Tensor): Log probabilities of actions taken.
            entropies (List[torch.Tensor], optional): List of entropy values.
            ent_coef (float): Entropy regularization coefficient.
            gamma (float): Discount factor.

        Returns:
            torch.Tensor: Scalar actor loss value.
        """
        advantage = get_advantage(rewards, values, gamma)

        logp = torch.stack(log_probs)

        loss = -(logp * advantage).mean()

        if entropies is not None and len(entropies) > 0:
            loss -= ent_coef * torch.stack(entropies).mean()

        return loss


class ActorPPOLoss(nn.Module):
    """
    Computes Proximal Policy Optimization (PPO) clipped surrogate actor loss with entropy.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        rewards: List[torch.Tensor],
        values: torch.Tensor,
        policies: torch.Tensor,
        log_probs: torch.Tensor,
        log_probs_prev: torch.Tensor,
        entropies: Optional[List[torch.Tensor]] = None,
        policy_masks: Optional[List[torch.Tensor]] = None,
        ent_coef: float = 0.5,
        gamma: float = 0.99,
        eps: float = 0.2
    ) -> torch.Tensor:
        """
        Args:
            rewards (List[torch.Tensor]): Rewards received.
            values (List[torch.Tensor]): Value predictions.
            policies (torch.Tensor): Policy outputs (not used in loss directly).
            log_probs (torch.Tensor): Log probabilities under current policy.
            log_probs_prev (torch.Tensor): Log probabilities under old policy.
            entropies (List[torch.Tensor], optional): List of entropy values.
            ent_coef (float): Entropy coefficient.
            gamma (float): Discount factor.
            eps (float): Clipping epsilon for PPO.

        Returns:
            torch.Tensor: Scalar PPO loss.
        """
        advantage = get_advantage(rewards, values, gamma)
        ratio = torch.exp(log_probs - log_probs_prev)

        # PPO clipped surrogate objective
        p1 = ratio * advantage
        p2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
        surrogate = torch.min(p1, p2)

        if policy_masks is not None:
            surrogate = torch.stack(policy_masks) * surrogate

        loss = -surrogate.mean()

        if entropies is not None and len(entropies) > 0:
            loss -= ent_coef * torch.stack(entropies).mean()

        return loss
