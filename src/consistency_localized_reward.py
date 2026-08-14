"""Consistency-localized intrinsic reward model.

This module implements the first self-contained CLIR framework for this repo.
It uses SWIFT as an architectural reference, not as a dependency:

1. SWIFT-style token reward and gate heads over frozen LLM hidden states.
2. PRISM-style consistency over LLM-guided style/domain rewrites.
3. Hallucination onset localization with negative-tail token rewards.
4. DPCL-inspired key/complete dual-prior localization for grounded reasoning.

Inputs are pre-extracted hidden states with shape [batch, time, hidden_dim].
For real all-layer features, ``hidden_dim`` is the raw concatenated width while
``model_dim`` is the compact width used by CLIR's quadratic modules. Optional
condition states are encoded by the same feature encoder as trajectory states,
then fused through token-level attention before reward and localization heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class RewardConfig:
    hidden_dim: int
    model_variant: str = "clir"
    encoder_type: str = "identity"
    model_dim: Optional[int] = None
    num_feature_layers: int = 1
    per_layer_dim: Optional[int] = None
    layer_encoder_dim: int = 256
    layer_encoder_blocks: int = 2
    layer_encoder_heads: int = 8
    layer_pool_queries: int = 4
    encoder_dropout: float = 0.0
    projection_dim: int = 256
    consistency_margin: float = 0.2
    negative_tail_margin: float = 0.5
    pseudo_onset_threshold: float = 0.5
    prior_fusion_alpha: float = 0.5
    condition_attention_temperature: float = 1.0
    progress_score_weight: float = 0.5
    eps: float = 1e-8

    final_weight: float = 1.0
    consistency_weight: float = 1.0
    score_consistency_weight: float = 0.1
    hallucination_weight: float = 1.0
    mil_weight: float = 0.25
    token_reward_weight: float = 0.5
    tail_weight: float = 0.5
    pseudo_tail_weight: float = 0.1
    progress_weight: float = 0.25
    prior_weight: float = 0.25
    key_prior_weight: float = 1.0
    complete_prior_weight: float = 1.0
    prior_distill_weight: float = 0.25
    gate_prior_weight: float = 0.25
    reconstruction_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.model_variant not in {"strict_swift", "encoded_swift", "clir"}:
            raise ValueError("model_variant must be one of: strict_swift, encoded_swift, clir")
        if self.encoder_type not in {"identity", "flat_linear", "layer_transformer"}:
            raise ValueError("encoder_type must be one of: identity, flat_linear, layer_transformer")

        if self.model_dim is None:
            self.model_dim = (
                self.hidden_dim
                if self.encoder_type == "identity"
                else min(self.hidden_dim, 768)
            )
        if self.model_dim <= 0:
            raise ValueError("model_dim must be positive")

        if self.num_feature_layers <= 0:
            raise ValueError("num_feature_layers must be positive")
        if self.per_layer_dim is None:
            if self.hidden_dim % self.num_feature_layers != 0:
                raise ValueError("hidden_dim must be divisible by num_feature_layers")
            self.per_layer_dim = self.hidden_dim // self.num_feature_layers
        if self.hidden_dim != self.num_feature_layers * self.per_layer_dim:
            raise ValueError("hidden_dim must equal num_feature_layers * per_layer_dim")
        if (
            self.encoder_type == "layer_transformer"
            and self.num_feature_layers == 1
            and self.hidden_dim >= 65_536
        ):
            raise ValueError(
                "large layer_transformer inputs require an explicit multi-layer contract; "
                "set num_feature_layers and per_layer_dim from the extraction manifest"
            )

        if self.encoder_type == "identity" and self.model_dim != self.hidden_dim:
            raise ValueError("identity encoder requires model_dim == hidden_dim")
        if self.model_variant == "strict_swift" and self.encoder_type != "identity":
            raise ValueError("strict_swift is the unencoded baseline and requires encoder_type='identity'")
        if self.model_variant == "encoded_swift" and self.encoder_type == "identity":
            raise ValueError("encoded_swift requires a non-identity encoder")
        if (
            self.model_variant == "clir"
            and self.encoder_type == "identity"
            and self.hidden_dim >= 65_536
        ):
            raise ValueError(
                "identity CLIR would apply quadratic modules at the raw feature width; "
                "use encoder_type='layer_transformer' and a compact model_dim"
            )
        if self.layer_encoder_dim <= 0 or self.layer_encoder_blocks <= 0:
            raise ValueError("layer encoder dimensions and block count must be positive")
        if self.layer_encoder_heads <= 0:
            raise ValueError("layer_encoder_heads must be positive")
        if self.layer_encoder_dim % self.layer_encoder_heads != 0:
            raise ValueError("layer_encoder_dim must be divisible by layer_encoder_heads")
        if self.layer_pool_queries <= 0:
            raise ValueError("layer_pool_queries must be positive")
        if not 0.0 <= self.encoder_dropout < 1.0:
            raise ValueError("encoder_dropout must be in [0, 1)")


class IdentityFeatureEncoder(nn.Module):
    """Pass through already compact token features."""

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        return hidden_states, None


class FlatLinearFeatureEncoder(nn.Module):
    """Unstructured full-width projection used as a development ablation."""

    def __init__(self, input_dim: int, model_dim: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, model_dim)
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        encoded = self.projection(self.input_norm(hidden_states))
        return self.output_norm(F.gelu(encoded)), None


class LayerAxisFeatureEncoder(nn.Module):
    """Compress every token while preserving and modeling its full layer axis.

    The raw concatenation is reshaped to ``[layers, per_layer_dim]``. A shared
    low-rank projection, learned depth positions, a small Transformer over the
    layer axis, and learned pooling queries produce one compact token vector.
    Pool attention is returned for diagnostics as ``[B, T, K, L]``.
    """

    # Some CUDA LayerNorm kernels silently overflow their 32-bit indexing
    # above 2**32 input elements.  Staying below 2**31 leaves a conservative
    # safety margin while preserving the exact operation in ordinary batches.
    max_normalization_elements = 2**31 - 1

    def __init__(self, config: RewardConfig) -> None:
        super().__init__()
        self.input_dim = config.hidden_dim
        self.num_feature_layers = config.num_feature_layers
        self.per_layer_dim = int(config.per_layer_dim)
        self.layer_encoder_dim = config.layer_encoder_dim
        self.layer_pool_queries = config.layer_pool_queries

        self.input_norm = nn.LayerNorm(self.per_layer_dim)
        self.input_projection = nn.Linear(self.per_layer_dim, self.layer_encoder_dim)
        self.layer_positions = nn.Parameter(
            torch.empty(1, self.num_feature_layers, self.layer_encoder_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.layer_encoder_dim,
            nhead=config.layer_encoder_heads,
            dim_feedforward=self.layer_encoder_dim * 4,
            dropout=config.encoder_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layer_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.layer_encoder_blocks,
            enable_nested_tensor=False,
        )
        self.pool_queries = nn.Parameter(
            torch.empty(1, self.layer_pool_queries, self.layer_encoder_dim)
        )
        self.layer_pool = nn.MultiheadAttention(
            embed_dim=self.layer_encoder_dim,
            num_heads=config.layer_encoder_heads,
            dropout=config.encoder_dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(
            self.layer_pool_queries * self.layer_encoder_dim,
            int(config.model_dim),
        )
        self.output_norm = nn.LayerNorm(int(config.model_dim))
        nn.init.normal_(self.layer_positions, mean=0.0, std=0.02)
        nn.init.normal_(self.pool_queries, mean=0.0, std=0.02)

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.input_dim:
            raise ValueError(
                f"layer encoder expects [batch, time, {self.input_dim}], "
                f"got {tuple(hidden_states.shape)}"
            )
        batch, time, _ = hidden_states.shape
        layer_states = hidden_states.reshape(
            batch * time,
            self.num_feature_layers,
            self.per_layer_dim,
        )
        elements_per_row = self.num_feature_layers * self.per_layer_dim
        rows_per_chunk = max(1, self.max_normalization_elements // elements_per_row)
        if layer_states.shape[0] <= rows_per_chunk:
            layer_states = self.input_projection(self.input_norm(layer_states))
        else:
            layer_states = torch.cat(
                [
                    self.input_projection(self.input_norm(chunk))
                    for chunk in layer_states.split(rows_per_chunk, dim=0)
                ],
                dim=0,
            )
        layer_states = self.layer_transformer(layer_states + self.layer_positions)

        queries = self.pool_queries.expand(batch * time, -1, -1)
        pooled, attention = self.layer_pool(
            queries,
            layer_states,
            layer_states,
            need_weights=True,
            average_attn_weights=True,
        )
        encoded = pooled.flatten(start_dim=1)
        encoded = self.output_norm(F.gelu(self.output_projection(encoded)))
        return (
            encoded.reshape(batch, time, -1),
            attention.reshape(
                batch,
                time,
                self.layer_pool_queries,
                self.num_feature_layers,
            ),
        )


def build_feature_encoder(config: RewardConfig) -> nn.Module:
    if config.encoder_type == "identity":
        return IdentityFeatureEncoder()
    if config.encoder_type == "flat_linear":
        return FlatLinearFeatureEncoder(config.hidden_dim, int(config.model_dim))
    return LayerAxisFeatureEncoder(config)


def _validate_raw_features(hidden_states: Tensor, config: RewardConfig, name: str) -> None:
    if hidden_states.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, time, hidden_dim]")
    if hidden_states.shape[-1] != config.hidden_dim:
        raise ValueError(
            f"{name} hidden_dim mismatch: got {hidden_states.shape[-1]}, expected {config.hidden_dim}"
        )


def _swift_outputs(token_features: Tensor, mask: Tensor, token_head: nn.Linear, eps: float) -> Dict[str, Tensor]:
    token_output = token_head(token_features)
    gate_logits = token_output[..., 0]
    token_rewards = token_output[..., 1]
    gates = torch.sigmoid(gate_logits) * mask
    scores = (gates * token_rewards).sum(dim=1) / gates.sum(dim=1).clamp_min(eps)
    return {
        "scores": scores,
        "token_rewards": token_rewards,
        "token_values": token_rewards,
        "token_scores": scores,
        "gate_logits": gate_logits,
        "gates": gates,
        "token_features": token_features,
        "mask": mask,
    }


def masked_binary_cross_entropy_with_logits(
    logits: Tensor,
    targets: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Binary cross entropy that never turns missing labels into negatives."""

    targets = targets.to(device=logits.device, dtype=logits.dtype).view_as(logits)
    if mask is None:
        return F.binary_cross_entropy_with_logits(logits.float(), targets.float())
    valid = mask.to(device=logits.device).bool().view_as(logits)
    if not valid.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        logits[valid].float(),
        targets[valid].float(),
    )


class SwiftRewardBase(nn.Module):
    """Shared correctness-only training contract for strict/encoded SWIFT."""

    config: RewardConfig

    def training_step(
        self,
        batch: Dict[str, Tensor],
        prior_phase: str = "joint",
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        del prior_phase
        outputs = self.forward(
            batch["hidden_states"],
            mask=batch.get("mask"),
            condition_states=batch.get("condition_states"),
            condition_mask=batch.get("condition_mask"),
            condition_embedding=batch.get("condition_embedding"),
            condition_embedding_mask=batch.get("condition_embedding_mask"),
        )
        total = outputs["scores"].new_zeros(())
        losses: Dict[str, Tensor] = {}
        if "correctness" in batch:
            target = batch["correctness"].to(outputs["scores"]).view_as(outputs["scores"])
            final = masked_binary_cross_entropy_with_logits(
                outputs["scores"],
                target,
                batch.get("correctness_mask"),
            )
            losses["final"] = final
            total = total + self.config.final_weight * final
        losses["total"] = total
        return outputs, losses


class StrictSwiftReward(SwiftRewardBase):
    """Official SWIFT-style full-width Linear(D, 2) reward/gate baseline."""

    def __init__(self, config: RewardConfig) -> None:
        super().__init__()
        if config.model_variant != "strict_swift":
            raise ValueError("StrictSwiftReward requires model_variant='strict_swift'")
        self.config = config
        self.token_reward_head = nn.Linear(config.hidden_dim, 2)

    def forward(
        self,
        hidden_states: Tensor,
        mask: Optional[Tensor] = None,
        condition_states: Optional[Tensor] = None,
        condition_mask: Optional[Tensor] = None,
        condition_embedding: Optional[Tensor] = None,
        condition_embedding_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        del condition_states, condition_mask, condition_embedding, condition_embedding_mask
        _validate_raw_features(hidden_states, self.config, "hidden_states")
        if mask is None:
            mask = hidden_states.new_ones(hidden_states.shape[:2])
        mask = mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return _swift_outputs(hidden_states, mask, self.token_reward_head, self.config.eps)


class EncodedSwiftReward(SwiftRewardBase):
    """SWIFT reward/gate head after the same encoder family used by CLIR."""

    def __init__(self, config: RewardConfig) -> None:
        super().__init__()
        if config.model_variant != "encoded_swift":
            raise ValueError("EncodedSwiftReward requires model_variant='encoded_swift'")
        self.config = config
        self.input_encoder = build_feature_encoder(config)
        self.token_reward_head = nn.Linear(int(config.model_dim), 2)

    def forward(
        self,
        hidden_states: Tensor,
        mask: Optional[Tensor] = None,
        condition_states: Optional[Tensor] = None,
        condition_mask: Optional[Tensor] = None,
        condition_embedding: Optional[Tensor] = None,
        condition_embedding_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        del condition_states, condition_mask, condition_embedding, condition_embedding_mask
        _validate_raw_features(hidden_states, self.config, "hidden_states")
        token_features, layer_attention = self.input_encoder(hidden_states)
        if mask is None:
            mask = token_features.new_ones(token_features.shape[:2])
        mask = mask.to(device=token_features.device, dtype=token_features.dtype)
        outputs = _swift_outputs(token_features, mask, self.token_reward_head, self.config.eps)
        if layer_attention is not None:
            outputs["trajectory_layer_attention"] = layer_attention
        return outputs


class ConsistencyLocalizedReward(nn.Module):
    """SWIFT-style reward head plus consistency and localization objectives."""

    def __init__(self, config: RewardConfig) -> None:
        super().__init__()
        if config.model_variant != "clir":
            raise ValueError("ConsistencyLocalizedReward requires model_variant='clir'")
        self.config = config
        self.input_encoder = build_feature_encoder(config)
        model_dim = int(config.model_dim)
        self.condition_query = nn.Linear(model_dim, model_dim, bias=False)
        self.condition_key = nn.Linear(model_dim, model_dim, bias=False)
        self.condition_value = nn.Linear(model_dim, model_dim, bias=False)
        self.condition_fusion = nn.Sequential(
            nn.LayerNorm(model_dim * 4 + 1),
            nn.Linear(model_dim * 4 + 1, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.feature_norm = nn.LayerNorm(model_dim)
        self.token_reward_head = nn.Linear(model_dim, 2)
        self.hallucination_head = nn.Linear(model_dim, 1)
        self.progress_head = nn.Linear(model_dim, 1)
        self.final_score_head = nn.Linear(model_dim, 1)
        self.key_prior_head = nn.Linear(model_dim, 1)
        self.complete_prior_head = nn.Linear(model_dim, 1)
        self.complete_reconstructor = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.projector = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, config.projection_dim),
        )

    def forward(
        self,
        hidden_states: Tensor,
        mask: Optional[Tensor] = None,
        condition_states: Optional[Tensor] = None,
        condition_mask: Optional[Tensor] = None,
        condition_embedding: Optional[Tensor] = None,
        condition_embedding_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute scalar reward, token rewards, hallucination, and priors."""
        _validate_raw_features(hidden_states, self.config, "hidden_states")
        hidden_states, trajectory_layer_attention = self.input_encoder(hidden_states)
        batch, time, _ = hidden_states.shape

        condition_layer_parts = []
        if condition_states is not None:
            _validate_raw_features(condition_states, self.config, "condition_states")
            condition_states, condition_layer_attention = self.input_encoder(condition_states)
            if condition_layer_attention is not None:
                condition_layer_parts.append(condition_layer_attention)
        if condition_embedding is not None:
            if condition_embedding.ndim != 2 or condition_embedding.shape[-1] != self.config.hidden_dim:
                raise ValueError(
                    f"condition_embedding must have shape [batch, {self.config.hidden_dim}]"
                )
            encoded_embedding, embedding_layer_attention = self.input_encoder(
                condition_embedding.unsqueeze(1)
            )
            condition_embedding = encoded_embedding[:, 0]
            if embedding_layer_attention is not None:
                condition_layer_parts.append(embedding_layer_attention)

        if mask is None:
            mask = hidden_states.new_ones(batch, time)
        mask = mask.to(dtype=hidden_states.dtype, device=hidden_states.device)

        token_features, condition_relevance, condition_attention = self._condition_token_features(
            hidden_states,
            mask,
            condition_states=condition_states,
            condition_mask=condition_mask,
            condition_embedding=condition_embedding,
            condition_embedding_mask=condition_embedding_mask,
        )

        token_head = self.token_reward_head(token_features)
        gate_logits = token_head[..., 0]
        token_rewards = token_head[..., 1]
        gates = torch.sigmoid(gate_logits) * mask

        hallucination_logits = self.hallucination_head(token_features).squeeze(-1)
        progress = self.progress_head(token_features).squeeze(-1)
        token_values = token_rewards + self.config.progress_score_weight * progress

        denom = gates.sum(dim=1).clamp_min(self.config.eps)
        token_scores = (gates * token_values).sum(dim=1) / denom

        pooled = masked_mean(token_features, mask)
        score_residual = self.final_score_head(pooled).squeeze(-1)
        scores = token_scores + score_residual
        representations = F.normalize(self.projector(pooled), dim=-1)

        key_prior_logits = self.key_prior_head(token_features).squeeze(-1)
        complete_prior_logits = self.complete_prior_head(token_features).squeeze(-1)
        key_prior = masked_softmax(key_prior_logits, mask)
        complete_prior = masked_softmax(complete_prior_logits, mask)
        fused_prior = normalize_attention(
            self.config.prior_fusion_alpha * key_prior
            + (1.0 - self.config.prior_fusion_alpha) * complete_prior,
            mask,
            eps=self.config.eps,
        )

        complete_context = (complete_prior.unsqueeze(-1) * token_features).sum(dim=1)
        complete_reconstruction = self.complete_reconstructor(complete_context)

        outputs = {
            "scores": scores,
            "token_rewards": token_rewards,
            "token_values": token_values,
            "token_scores": token_scores,
            "score_residual": score_residual,
            "gate_logits": gate_logits,
            "gates": gates,
            "hallucination_logits": hallucination_logits,
            "progress": progress,
            "condition_relevance": condition_relevance,
            "condition_attention": condition_attention,
            "representations": representations,
            "key_prior_logits": key_prior_logits,
            "complete_prior_logits": complete_prior_logits,
            "key_prior": key_prior,
            "complete_prior": complete_prior,
            "fused_prior": fused_prior,
            "complete_reconstruction": complete_reconstruction,
            "pooled_features": pooled,
            "token_features": token_features,
            "mask": mask,
        }
        if trajectory_layer_attention is not None:
            outputs["trajectory_layer_attention"] = trajectory_layer_attention
        if condition_layer_parts:
            outputs["condition_layer_attention"] = torch.cat(condition_layer_parts, dim=1)
        return outputs

    def training_step(
        self,
        batch: Dict[str, Tensor],
        prior_phase: str = "joint",
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Run forward and compute all configured losses for a batch."""
        outputs = self.forward(
            batch["hidden_states"],
            mask=batch.get("mask"),
            condition_states=batch.get("condition_states"),
            condition_mask=batch.get("condition_mask"),
            condition_embedding=batch.get("condition_embedding"),
            condition_embedding_mask=batch.get("condition_embedding_mask"),
        )
        losses = self.loss(outputs, batch, prior_phase=prior_phase)
        return outputs, losses

    def loss(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        prior_phase: str = "joint",
    ) -> Dict[str, Tensor]:
        """Compute every available CLIR loss.

        Optional batch fields:
            correctness: [batch] binary final correctness.
            semantic_ids/style_ids: ids for PRISM-style consistency.
            consistency_mask: [batch] marks rows with valid consistency ids.
            hallucination_onset: [batch] first hallucinated token, -1 if none.
            onset_label_mask: [batch] marks rows with explicit onset/non-onset labels.
            path_hallucinated/path_label_mask: weak path-level hallucination labels.
            token_advantage/token_advantage_mask: token progress/advantage targets.
            progress_targets/progress_mask: direct progress-head supervision.
            key_prior_target/complete_prior_target: DPCL-style prior maps.
            complete_reconstruction_target: independently generated fixed
                evidence/answer vector with width model_dim. It must not be
                pooled from the same candidate trajectory.
        """
        losses: Dict[str, Tensor] = {}
        total = outputs["scores"].new_zeros(())

        if "correctness" in batch:
            target = batch["correctness"].to(outputs["scores"]).view_as(outputs["scores"])
            final_loss = masked_binary_cross_entropy_with_logits(
                outputs["scores"],
                target,
                batch.get("correctness_mask"),
            )
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
                metadata_mask=batch.get("consistency_mask"),
            )
            losses.update({f"consistency_{k}": v for k, v in consistency.items()})
            total = total + self.config.consistency_weight * consistency["total"]

        if "hallucination_onset" in batch:
            loc_losses = hallucination_localization_losses(
                outputs["hallucination_logits"],
                outputs["token_values"],
                outputs["mask"],
                batch["hallucination_onset"],
                onset_label_mask=batch.get("onset_label_mask"),
                token_advantage=batch.get("token_advantage"),
                token_advantage_mask=batch.get("token_advantage_mask"),
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
                label_mask=batch.get("path_label_mask"),
            )
            losses["hallucination_mil"] = mil_loss
            total = total + self.config.mil_weight * mil_loss

            pseudo_tail = pseudo_onset_tail_loss(
                outputs["hallucination_logits"],
                outputs["token_values"],
                outputs["mask"],
                batch["path_hallucinated"],
                label_mask=batch.get("path_label_mask"),
                onset_label_mask=batch.get("onset_label_mask"),
                threshold=self.config.pseudo_onset_threshold,
                negative_tail_margin=self.config.negative_tail_margin,
            )
            losses["pseudo_tail"] = pseudo_tail
            total = total + self.config.pseudo_tail_weight * pseudo_tail

        if "progress_targets" in batch:
            progress = token_regression_loss(
                outputs["progress"],
                batch["progress_targets"],
                outputs["mask"],
                target_mask=batch.get("progress_mask"),
            )
            losses["progress"] = progress
            total = total + self.config.progress_weight * progress

        prior = dual_prior_losses(
            outputs,
            batch,
            key_weight=self.config.key_prior_weight,
            complete_weight=self.config.complete_prior_weight,
            distill_weight=self.config.prior_distill_weight,
            gate_weight=self.config.gate_prior_weight,
            reconstruction_weight=self.config.reconstruction_weight,
            phase=prior_phase,
        )
        losses.update({f"prior_{k}": v for k, v in prior.items()})
        total = total + self.config.prior_weight * prior["total"]

        losses["total"] = total
        return losses

    def _condition_token_features(
        self,
        hidden_states: Tensor,
        mask: Tensor,
        condition_states: Optional[Tensor],
        condition_mask: Optional[Tensor],
        condition_embedding: Optional[Tensor],
        condition_embedding_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        condition_parts = []
        mask_parts = []
        if condition_states is not None:
            if condition_states.ndim != 3:
                raise ValueError("condition_states must have shape [batch, time, hidden_dim]")
            if condition_states.shape[-1] != hidden_states.shape[-1]:
                raise ValueError("condition_states hidden_dim must match hidden_states")
            condition_parts.append(condition_states.to(hidden_states))
            if condition_mask is None:
                condition_mask = condition_states.new_ones(condition_states.shape[:2])
            mask_parts.append(condition_mask.to(device=hidden_states.device, dtype=hidden_states.dtype))

        if condition_embedding is not None:
            condition_embedding = condition_embedding.to(hidden_states)
            if condition_embedding.shape != hidden_states.shape[:1] + hidden_states.shape[2:]:
                raise ValueError("condition embedding must have shape [batch, hidden_dim]")
            condition_parts.append(condition_embedding.unsqueeze(1))
            if condition_embedding_mask is None:
                condition_embedding_mask = condition_embedding.new_ones(condition_embedding.shape[0])
            mask_parts.append(condition_embedding_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)[:, None])

        if not condition_parts:
            relevance = hidden_states.new_zeros(hidden_states.shape[:2])
            attention = hidden_states.new_zeros(hidden_states.shape[0], hidden_states.shape[1], 0)
            return hidden_states, relevance, attention

        condition_tokens = torch.cat(condition_parts, dim=1)
        combined_condition_mask = torch.cat(mask_parts, dim=1)
        condition_bool = combined_condition_mask.to(device=hidden_states.device).bool()
        has_condition = condition_bool.any(dim=1)

        queries = F.normalize(self.condition_query(hidden_states), dim=-1)
        keys = F.normalize(self.condition_key(condition_tokens), dim=-1)
        values = self.condition_value(condition_tokens)

        temperature = max(self.config.condition_attention_temperature, self.config.eps)
        attention_logits = torch.matmul(queries, keys.transpose(1, 2)) / temperature
        attention_logits = attention_logits.masked_fill(~condition_bool[:, None, :], -1e4)
        attention = torch.softmax(attention_logits, dim=-1) * condition_bool[:, None, :].to(hidden_states.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)

        context = torch.matmul(attention, values)
        matched_keys = torch.matmul(attention, keys)
        relevance = (queries * matched_keys).sum(dim=-1) * mask * has_condition[:, None].to(hidden_states.dtype)

        fusion_input = torch.cat(
            [
                hidden_states,
                context,
                hidden_states * context,
                hidden_states - context,
                relevance.unsqueeze(-1),
            ],
            dim=-1,
        )
        delta = self.condition_fusion(fusion_input)
        active_rows = has_condition[:, None, None].to(hidden_states.dtype)
        conditioned = self.feature_norm(hidden_states + delta * mask.unsqueeze(-1) * active_rows)
        token_features = torch.where(has_condition[:, None, None], conditioned, hidden_states)
        attention = attention * has_condition[:, None, None].to(hidden_states.dtype)
        return token_features, relevance, attention


def build_reward_model(config: RewardConfig) -> nn.Module:
    """Build the explicit baseline/CLIR variant recorded in ``config``."""
    if config.model_variant == "strict_swift":
        return StrictSwiftReward(config)
    if config.model_variant == "encoded_swift":
        return EncodedSwiftReward(config)
    return ConsistencyLocalizedReward(config)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(dtype=values.dtype, device=values.device)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (values * mask.unsqueeze(-1)).sum(dim=1) / denom


def normalize_attention(attention: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    mask = mask.to(device=attention.device, dtype=attention.dtype)
    attention = attention.clamp_min(0.0) * mask
    denom = attention.sum(dim=1, keepdim=True).clamp_min(eps)
    return attention / denom


def masked_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    mask_bool = mask.to(device=logits.device).bool()
    masked_logits = logits.masked_fill(~mask_bool, -1e4)
    probs = torch.softmax(masked_logits, dim=1) * mask_bool.to(dtype=logits.dtype)
    return normalize_attention(probs, mask_bool.to(dtype=logits.dtype))


def masked_bce_with_logits(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    target_mask: Optional[Tensor] = None,
) -> Tensor:
    mask_bool = mask.to(device=logits.device).bool()
    if target_mask is not None:
        mask_bool = mask_bool & target_mask.to(device=logits.device).bool()
    if not mask_bool.any():
        return logits.new_zeros(())
    target = target.to(device=logits.device, dtype=logits.dtype)
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return raw[mask_bool].mean()


def token_regression_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    target_mask: Optional[Tensor] = None,
) -> Tensor:
    mask_bool = mask.to(device=prediction.device).bool()
    if target_mask is not None:
        mask_bool = mask_bool & target_mask.to(device=prediction.device).bool()
    if not mask_bool.any():
        return prediction.new_zeros(())
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    return F.mse_loss(prediction[mask_bool], target[mask_bool])


def prism_style_consistency_loss(
    representations: Tensor,
    scores: Tensor,
    semantic_ids: Tensor,
    style_ids: Tensor,
    margin: float,
    score_weight: float,
    metadata_mask: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """PRISM-like consistency over augmented reasoning trajectories."""
    device = representations.device
    if metadata_mask is None:
        metadata_mask = torch.ones(representations.shape[0], dtype=torch.bool, device=device)
    else:
        metadata_mask = metadata_mask.to(device=device).bool()

    if metadata_mask.sum() < 2:
        zero = scores.new_zeros(())
        return {"positive": zero, "negative": zero, "score": zero, "total": zero}

    representations = representations[metadata_mask]
    scores = scores[metadata_mask]
    semantic_ids = semantic_ids.to(device=device)[metadata_mask]
    style_ids = style_ids.to(device=device)[metadata_mask]

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
    return {"positive": positive, "negative": negative, "score": score_consistency, "total": total}


def hallucination_localization_losses(
    hallucination_logits: Tensor,
    token_rewards: Tensor,
    mask: Tensor,
    onset: Tensor,
    onset_label_mask: Optional[Tensor],
    token_advantage: Optional[Tensor],
    token_advantage_mask: Optional[Tensor],
    negative_tail_margin: float,
) -> Dict[str, Tensor]:
    """Token-level onset supervision and negative-tail reward shaping.

    onset is -1 for known non-hallucinated trajectories. onset_label_mask marks
    rows with explicit onset/non-onset labels; unlabeled rows are ignored.
    """
    device = hallucination_logits.device
    mask_bool = mask.to(device=device).bool()
    onset = onset.to(device=device).long()
    if onset_label_mask is None:
        onset_label_mask = torch.ones(onset.shape, dtype=torch.bool, device=device)
    else:
        onset_label_mask = onset_label_mask.to(device=device).bool()

    positions = torch.arange(hallucination_logits.shape[1], device=device)[None, :]
    lengths = mask_bool.long().sum(dim=1)
    invalid_onset = onset_label_mask & (onset >= lengths) & (onset >= 0)
    if invalid_onset.any():
        bad = torch.nonzero(invalid_onset, as_tuple=False).flatten().detach().cpu().tolist()
        raise ValueError(f"hallucination_onset is outside the valid token range for rows {bad}")

    has_onset = onset_label_mask[:, None] & (onset[:, None] >= 0)
    tail = has_onset & (positions >= onset[:, None]) & mask_bool
    supervised_tokens = onset_label_mask[:, None] & mask_bool
    hallucination_target = tail.to(dtype=hallucination_logits.dtype)
    zero = hallucination_logits.new_zeros(())

    if supervised_tokens.any():
        token_bce_raw = F.binary_cross_entropy_with_logits(
            hallucination_logits,
            hallucination_target,
            reduction="none",
        )
        token_bce = token_bce_raw[supervised_tokens].mean()
    else:
        token_bce = zero

    if token_advantage is None:
        reward_target = torch.zeros_like(token_rewards)
        known_reward = tail
    else:
        reward_target = token_advantage.to(device=device, dtype=token_rewards.dtype).clone()
        if token_advantage_mask is None:
            known_reward = mask_bool
        else:
            known_reward = token_advantage_mask.to(device=device).bool() & mask_bool
    reward_target = reward_target.masked_fill(tail, -negative_tail_margin)
    reward_mask = known_reward | tail

    if reward_mask.any():
        token_reward = F.mse_loss(token_rewards[reward_mask], reward_target[reward_mask])
    else:
        token_reward = zero

    if tail.any():
        tail_margin = F.relu(token_rewards[tail] + negative_tail_margin).pow(2).mean()
    else:
        tail_margin = zero

    return {"token_bce": token_bce, "token_reward": token_reward, "tail_margin": tail_margin}


def path_level_hallucination_mil(
    hallucination_logits: Tensor,
    mask: Tensor,
    path_hallucinated: Tensor,
    label_mask: Optional[Tensor] = None,
) -> Tensor:
    """Weak path-level hallucination loss using log-space noisy-or.

    Let ``q = product_t (1 - sigmoid(logit_t))`` be the probability that no
    valid token hallucinates. Directly forming ``1 - q`` rounds to exactly one
    on realistic sequence lengths and destroys the negative-path gradient.
    We instead keep ``log(q)`` as a sum of log-sigmoid values and evaluate the
    positive/negative BCE branches from that stable representation.
    """
    device = hallucination_logits.device
    if label_mask is None:
        label_mask = torch.ones(path_hallucinated.shape, dtype=torch.bool, device=device)
    else:
        label_mask = label_mask.to(device=device).bool()
    if label_mask.sum() == 0:
        return hallucination_logits.new_zeros(())

    logits = hallucination_logits[label_mask].float()
    row_mask = mask.to(device=device)[label_mask].bool()
    if not row_mask.any(dim=1).all():
        raise ValueError("Every path-level hallucination label requires at least one valid token")
    labels = path_hallucinated.to(device=device)[label_mask].bool()
    log_no_hallucination = (
        F.logsigmoid(-logits) * row_mask.to(dtype=logits.dtype)
    ).sum(dim=1)
    positive_loss = -_log1mexp(log_no_hallucination)
    negative_loss = -log_no_hallucination
    return torch.where(labels, positive_loss, negative_loss).mean()


def _log1mexp(log_x: Tensor) -> Tensor:
    """Compute ``log(1 - exp(log_x))`` stably for ``log_x <= 0``."""

    if torch.any(log_x > 0):
        raise ValueError("_log1mexp requires values <= 0")
    log_half = -0.6931471805599453
    # ``torch.where`` evaluates both branches.  Mask each branch's input first
    # so the unselected expression cannot create an infinite derivative and
    # poison the selected gradient through 0 * inf.
    low_input = torch.where(
        log_x < log_half,
        log_x,
        torch.full_like(log_x, -1.0),
    )
    high_input = torch.where(
        log_x < log_half,
        torch.full_like(log_x, -1.0),
        log_x,
    )
    return torch.where(
        log_x < log_half,
        torch.log1p(-torch.exp(low_input)),
        torch.log(-torch.expm1(high_input)),
    )


def pseudo_onset_tail_loss(
    hallucination_logits: Tensor,
    token_rewards: Tensor,
    mask: Tensor,
    path_hallucinated: Tensor,
    label_mask: Optional[Tensor],
    onset_label_mask: Optional[Tensor],
    threshold: float,
    negative_tail_margin: float,
) -> Tensor:
    """Apply a weak negative-tail loss after a predicted onset for positive paths."""
    device = hallucination_logits.device
    row_mask = mask.to(device=device).bool()
    if label_mask is None:
        label_mask = torch.ones(path_hallucinated.shape, dtype=torch.bool, device=device)
    else:
        label_mask = label_mask.to(device=device).bool()
    if onset_label_mask is None:
        onset_label_mask = torch.zeros(path_hallucinated.shape, dtype=torch.bool, device=device)
    else:
        onset_label_mask = onset_label_mask.to(device=device).bool()

    path_positive = path_hallucinated.to(device=device).bool()
    weak_rows = label_mask & path_positive & ~onset_label_mask
    if weak_rows.sum() == 0:
        return token_rewards.new_zeros(())

    probs = torch.sigmoid(hallucination_logits)
    crosses = (probs >= threshold) & row_mask
    has_cross = crosses.any(dim=1)
    pseudo_onset = torch.argmax(crosses.to(dtype=torch.long), dim=1)
    valid_rows = weak_rows & has_cross
    if valid_rows.sum() == 0:
        return token_rewards.new_zeros(())

    positions = torch.arange(token_rewards.shape[1], device=device)[None, :]
    tail = valid_rows[:, None] & (positions >= pseudo_onset[:, None]) & row_mask
    if not tail.any():
        return token_rewards.new_zeros(())
    return F.relu(token_rewards[tail] + negative_tail_margin).pow(2).mean()


def dual_prior_losses(
    outputs: Dict[str, Tensor],
    batch: Dict[str, Tensor],
    key_weight: float,
    complete_weight: float,
    distill_weight: float,
    gate_weight: float,
    reconstruction_weight: float,
    phase: str = "joint",
) -> Dict[str, Tensor]:
    """DPCL-inspired key/complete prior collaboration losses."""
    mask = outputs["mask"]
    zero = outputs["scores"].new_zeros(())
    has_key = "key_prior_target" in batch
    has_complete = "complete_prior_target" in batch
    if phase not in {"joint", "key", "complete"}:
        raise ValueError("prior_phase must be one of: joint, key, complete")

    key = masked_bce_with_logits(
        outputs["key_prior_logits"],
        batch["key_prior_target"],
        mask,
        target_mask=batch.get("key_prior_mask"),
    ) if has_key and phase in {"joint", "key"} else zero

    complete = masked_bce_with_logits(
        outputs["complete_prior_logits"],
        batch["complete_prior_target"],
        mask,
        target_mask=batch.get("complete_prior_mask"),
    ) if has_complete and phase in {"joint", "complete"} else zero

    shared_prior_mask = mask
    if has_key and "key_prior_mask" in batch:
        shared_prior_mask = shared_prior_mask * batch["key_prior_mask"].to(device=mask.device, dtype=mask.dtype)
    if has_complete and "complete_prior_mask" in batch:
        shared_prior_mask = shared_prior_mask * batch["complete_prior_mask"].to(device=mask.device, dtype=mask.dtype)

    distill = zero
    if has_key and has_complete:
        if phase in {"joint", "key"}:
            distill = distill + attention_mse(
                outputs["key_prior"],
                outputs["complete_prior"].detach(),
                shared_prior_mask,
            )
        if phase in {"joint", "complete"}:
            distill = distill + attention_mse(
                outputs["complete_prior"],
                outputs["key_prior"].detach(),
                shared_prior_mask,
            )

    gate_attention = normalize_attention(outputs["gates"], mask)
    fused_prior = outputs["fused_prior"].detach()
    gate_prior = attention_mse(gate_attention, fused_prior, shared_prior_mask) if has_key and has_complete else zero

    reconstruction = zero
    if "complete_reconstruction_target" in batch and phase in {"joint", "complete"}:
        target = batch["complete_reconstruction_target"].to(outputs["complete_reconstruction"])
        if target.shape != outputs["complete_reconstruction"].shape:
            raise ValueError(
                "complete_reconstruction_target must be an externally generated fixed vector "
                f"with shape {tuple(outputs['complete_reconstruction'].shape)}, got {tuple(target.shape)}"
            )
        if "complete_reconstruction_target_mask" in batch:
            reconstruction_mask = batch["complete_reconstruction_target_mask"].to(device=target.device).bool()
            if reconstruction_mask.any():
                reconstruction = F.mse_loss(
                    outputs["complete_reconstruction"][reconstruction_mask],
                    target[reconstruction_mask],
                )
        else:
            reconstruction = F.mse_loss(outputs["complete_reconstruction"], target)

    total = (
        key_weight * key
        + complete_weight * complete
        + distill_weight * distill
        + gate_weight * gate_prior
        + reconstruction_weight * reconstruction
    )
    return {
        "key": key,
        "complete": complete,
        "distill": distill,
        "gate": gate_prior,
        "reconstruction": reconstruction,
        "total": total,
    }


def attention_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    mask_bool = mask.to(device=prediction.device).bool()
    if not mask_bool.any():
        return prediction.new_zeros(())
    return F.mse_loss(prediction[mask_bool], target.to(prediction)[mask_bool])


@torch.no_grad()
def infer_pseudo_onsets(
    hallucination_logits: Tensor,
    mask: Tensor,
    threshold: float = 0.5,
) -> Tensor:
    """Return first token with hallucination probability above threshold, else -1."""
    probs = torch.sigmoid(hallucination_logits)
    mask_bool = mask.to(device=probs.device).bool()
    crosses = (probs >= threshold) & mask_bool
    has_cross = crosses.any(dim=1)
    onset = torch.argmax(crosses.to(dtype=torch.long), dim=1)
    return torch.where(has_cross, onset, torch.full_like(onset, -1))


@torch.no_grad()
def path_hallucination_probability(hallucination_logits: Tensor, mask: Tensor) -> Tensor:
    """Noisy-or path probability, accumulated through log survival values."""

    logits = hallucination_logits.float()
    mask = mask.to(device=logits.device).bool()
    log_no_hallucination = (
        F.logsigmoid(-logits) * mask.to(dtype=logits.dtype)
    ).sum(dim=1)
    return -torch.expm1(log_no_hallucination)


@torch.no_grad()
def path_no_hallucination_log_probability(
    hallucination_logits: Tensor,
    mask: Tensor,
) -> Tensor:
    """Return log P(no hallucination), which remains informative for long paths."""

    logits = hallucination_logits.float()
    mask = mask.to(device=logits.device).bool()
    return (F.logsigmoid(-logits) * mask.to(dtype=logits.dtype)).sum(dim=1)


@torch.no_grad()
def select_best_of_n(scores: Tensor, group_ids: Tensor) -> Dict[int, int]:
    """Return the row index of the best candidate for each query/group id."""
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
    "EncodedSwiftReward",
    "FlatLinearFeatureEncoder",
    "LayerAxisFeatureEncoder",
    "RewardConfig",
    "StrictSwiftReward",
    "build_feature_encoder",
    "build_reward_model",
    "count_trainable_parameters",
    "dual_prior_losses",
    "hallucination_localization_losses",
    "infer_pseudo_onsets",
    "masked_mean",
    "path_hallucination_probability",
    "path_no_hallucination_log_probability",
    "path_level_hallucination_mil",
    "prism_style_consistency_loss",
    "pseudo_onset_tail_loss",
    "select_best_of_n",
]
