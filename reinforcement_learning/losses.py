import torch
import torch.nn as nn
from typing import List, Optional


def get_discounted_reward(
    rewards: List[torch.Tensor],
    values: torch.Tensor,
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
    val = 0.0  # Assumed terminal value

    for i in reversed(range(len(rewards))):
        val = rewards[i] + gamma * val
        disc_rewards.insert(0, val)

    disc_rewards = torch.tensor(disc_rewards, dtype=torch.float32, device=rewards[0].device)
    out = disc_rewards - disc_rewards.mean()
    out /= disc_rewards.std() + 1e-8  # Add epsilon to avoid division by zero

    return out


def get_advantage(
    rewards: List[torch.Tensor],
    values: torch.Tensor,
    gamma: float = 0.99
) -> torch.Tensor:
    """
    Compute the advantage function.

    Args:
        rewards (List[torch.Tensor]): List of reward scalars.
        values (torch.Tensor): Estimated value function for each state.
        gamma (float): Discount factor.

    Returns:
        torch.Tensor: Advantage values.
    """
    disc_rewards = get_discounted_reward(rewards, values, gamma)
    advantage = disc_rewards - values
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
        values: torch.Tensor,
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
        advantage = get_advantage(rewards, values, gamma)
        loss = advantage.pow(2).mean()
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
        values: torch.Tensor,
        policies: torch.Tensor,
        log_probs: torch.Tensor,
        entropies: Optional[List[torch.Tensor]] = [],
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

        if entropies:
            ent = ent_coef * torch.mean(torch.stack(entropies))
            loss = - (log_probs * advantage + ent).mean()
        else:
            loss = - (log_probs * advantage).mean()

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
        entropies: Optional[List[torch.Tensor]] = [],
        ent_coef: float = 0.5,
        gamma: float = 0.99,
        eps: float = 0.2
    ) -> torch.Tensor:
        """
        Args:
            rewards (List[torch.Tensor]): Rewards received.
            values (torch.Tensor): Value predictions.
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
        p1 = ratio * advantage.detach()
        p2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage.detach()
        surrogate = torch.min(p1, p2)

        if entropies:
            ent = ent_coef * torch.mean(torch.stack(entropies))
            loss = - (surrogate + ent).mean()
        else:
            loss = - surrogate.mean()

        return loss
