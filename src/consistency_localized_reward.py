"""Consistency-localized intrinsic reward model.

This module is a compact PyTorch scaffold for extending a SWIFT-style
hidden-state reward head with two auxiliary signals:

1. PRISM-style consistency across LLM-guided style/domain rewrites.
2. Token-level hallucination localization with negative tail rewards.

The expected input hidden states are already extracted from a frozen LLM and
flattened or selected into shape [batch, time, hidden_dim].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class RewardConfig:
    hidden_dim: int
    projection_dim: int = 256
    consistency_margin: float = 0.2
    negative_tail_margin: float = 0.5
    eps: float = 1e-8

    final_weight: float = 1.0
    consistency_weight: float = 1.0
    score_consistency_weight: float = 0.1
    hallucination_weight: float = 1.0
    mil_weight: float = 0.25
    token_reward_weight: float = 0.5
    tail_weight: float = 0.5


class ConsistencyLocalizedReward(nn.Module):
    """SWIFT-style reward head plus consistency and localization losses."""

    def __init__(self, config: RewardConfig) -> None:
        super().__init__()
        self.config = config
        self.token_reward_head = nn.Linear(config.hidden_dim, 2)
        self.hallucination_head = nn.Linear(config.hidden_dim, 1)
        self.progress_head = nn.Linear(config.hidden_dim, 1)
        self.projector = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.projection_dim),
        )

    def forward(self, hidden_states: Tensor, mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Compute scalar and token-level reward outputs.

        Args:
            hidden_states: Tensor with shape [batch, time, hidden_dim].
            mask: Optional tensor with shape [batch, time], where 1 marks valid
                generated tokens.
        """
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, time, hidden_dim]")

        batch, time, _ = hidden_states.shape
        if mask is None:
            mask = hidden_states.new_ones(batch, time)
        mask = mask.to(dtype=hidden_states.dtype, device=hidden_states.device)

        token_head = self.token_reward_head(hidden_states)
        gate_logits = token_head[..., 0]
        token_rewards = token_head[..., 1]
        gates = torch.sigmoid(gate_logits) * mask

        denom = gates.sum(dim=1).clamp_min(self.config.eps)
        scores = (gates * token_rewards).sum(dim=1) / denom

        pooled = masked_mean(hidden_states, mask)
        representations = F.normalize(self.projector(pooled), dim=-1)

        return {
            "scores": scores,
            "token_rewards": token_rewards,
            "gate_logits": gate_logits,
            "gates": gates,
            "hallucination_logits": self.hallucination_head(hidden_states).squeeze(-1),
            "progress": self.progress_head(hidden_states).squeeze(-1),
            "representations": representations,
            "mask": mask,
        }

    def loss(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Compute all available losses.

        Expected optional batch fields:
            correctness: [batch] binary final correctness label.
            semantic_ids: [batch] id shared by rewrites of the same trajectory.
            style_ids: [batch] id for rewrite style/domain.
            hallucination_onset: [batch] first hallucinated token, or -1.
            path_hallucinated: [batch] binary path-level hallucination label.
            token_advantage: [batch, time] optional progress target.
        """
        losses: Dict[str, Tensor] = {}
        total = outputs["scores"].new_zeros(())

        if "correctness" in batch:
            target = batch["correctness"].to(outputs["scores"]).view_as(outputs["scores"])
            final_loss = F.binary_cross_entropy_with_logits(outputs["scores"], target)
            losses["final"] = final_loss
            total = total + self.config.final_weight * final_loss

        if "semantic_ids" in batch and "style_ids" in batch:
            consistency = prism_style_consistency_loss(
                outputs["representations"],
                outputs["scores"],
                batch["semantic_ids"],
                batch["style_ids"],
                margin=self.config.consistency_margin,
                score_weight=self.config.score_consistency_weight,
            )
            losses.update({f"consistency_{k}": v for k, v in consistency.items()})
            total = total + self.config.consistency_weight * consistency["total"]

        if "hallucination_onset" in batch:
            loc_losses = hallucination_localization_losses(
                outputs["hallucination_logits"],
                outputs["token_rewards"],
                outputs["mask"],
                batch["hallucination_onset"],
                token_advantage=batch.get("token_advantage"),
                negative_tail_margin=self.config.negative_tail_margin,
            )
            losses.update({f"localization_{k}": v for k, v in loc_losses.items()})
            total = total + self.config.hallucination_weight * loc_losses["token_bce"]
            total = total + self.config.token_reward_weight * loc_losses["token_reward"]
            total = total + self.config.tail_weight * loc_losses["tail_margin"]

        if "path_hallucinated" in batch:
            mil_loss = path_level_hallucination_mil(
                outputs["hallucination_logits"],
                outputs["mask"],
                batch["path_hallucinated"],
            )
            losses["hallucination_mil"] = mil_loss
            total = total + self.config.mil_weight * mil_loss

        losses["total"] = total
        return losses


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(dtype=values.dtype, device=values.device)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (values * mask.unsqueeze(-1)).sum(dim=1) / denom


def prism_style_consistency_loss(
    representations: Tensor,
    scores: Tensor,
    semantic_ids: Tensor,
    style_ids: Tensor,
    margin: float,
    score_weight: float,
) -> Dict[str, Tensor]:
    """PRISM-like consistency loss over augmented reasoning trajectories."""
    device = representations.device
    semantic_ids = semantic_ids.to(device)
    style_ids = style_ids.to(device)

    sim = representations @ representations.T
    semantic_eq = semantic_ids[:, None] == semantic_ids[None, :]
    style_eq = style_ids[:, None] == style_ids[None, :]
    upper = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)

    positive_pairs = upper & semantic_eq & ~style_eq
    negative_pairs = upper & ~semantic_eq & style_eq

    zero = sim.new_zeros(())
    if positive_pairs.any():
        positive = (1.0 - sim[positive_pairs]).mean()
        score_consistency = (scores[:, None] - scores[None, :]).pow(2)[positive_pairs].mean()
    else:
        positive = zero
        score_consistency = zero

    if negative_pairs.any():
        negative = F.relu(sim[negative_pairs] - margin).mean()
    else:
        negative = zero

    total = positive + negative + score_weight * score_consistency
    return {
        "positive": positive,
        "negative": negative,
        "score": score_consistency,
        "total": total,
    }


def hallucination_localization_losses(
    hallucination_logits: Tensor,
    token_rewards: Tensor,
    mask: Tensor,
    onset: Tensor,
    token_advantage: Optional[Tensor],
    negative_tail_margin: float,
) -> Dict[str, Tensor]:
    """Token-level onset supervision and negative-tail reward shaping.

    onset is -1 for trajectories without a known hallucination.
    """
    device = hallucination_logits.device
    mask_bool = mask.to(device=device).bool()
    onset = onset.to(device=device).long()
    positions = torch.arange(hallucination_logits.shape[1], device=device)[None, :]

    has_onset = onset >= 0
    tail = has_onset[:, None] & (positions >= onset[:, None]) & mask_bool
    hallucination_target = tail.to(dtype=hallucination_logits.dtype)

    token_bce_raw = F.binary_cross_entropy_with_logits(
        hallucination_logits,
        hallucination_target,
        reduction="none",
    )
    token_bce = token_bce_raw[mask_bool].mean() if mask_bool.any() else token_bce_raw.mean()

    if token_advantage is None:
        reward_target = torch.zeros_like(token_rewards)
        known_reward = tail
    else:
        reward_target = token_advantage.to(device=device, dtype=token_rewards.dtype).clone()
        known_reward = mask_bool
    reward_target = reward_target.masked_fill(tail, -negative_tail_margin)

    if known_reward.any():
        token_reward = F.mse_loss(token_rewards[known_reward], reward_target[known_reward])
    else:
        token_reward = token_rewards.new_zeros(())

    if tail.any():
        tail_margin = F.relu(token_rewards[tail] + negative_tail_margin).pow(2).mean()
    else:
        tail_margin = token_rewards.new_zeros(())

    return {
        "token_bce": token_bce,
        "token_reward": token_reward,
        "tail_margin": tail_margin,
    }


def path_level_hallucination_mil(
    hallucination_logits: Tensor,
    mask: Tensor,
    path_hallucinated: Tensor,
) -> Tensor:
    """Weak path-level hallucination loss using noisy-or over token probabilities."""
    probs = torch.sigmoid(hallucination_logits).clamp(1e-6, 1.0 - 1e-6)
    mask = mask.to(device=probs.device, dtype=probs.dtype)
    probs = probs * mask
    path_prob = 1.0 - torch.prod(1.0 - probs, dim=1)
    target = path_hallucinated.to(device=probs.device, dtype=probs.dtype).view_as(path_prob)
    return F.binary_cross_entropy(path_prob, target)


@torch.no_grad()
def select_best_of_n(scores: Tensor, group_ids: Tensor) -> Dict[int, int]:
    """Return the index of the best candidate for each query/group id."""
    best: Dict[int, int] = {}
    best_score: Dict[int, float] = {}
    for idx, group in enumerate(group_ids.detach().cpu().tolist()):
        score = float(scores[idx].detach().cpu())
        if group not in best_score or score > best_score[group]:
            best[group] = idx
            best_score[group] = score
    return best


__all__ = [
    "ConsistencyLocalizedReward",
    "RewardConfig",
    "hallucination_localization_losses",
    "path_level_hallucination_mil",
    "prism_style_consistency_loss",
    "select_best_of_n",
]
