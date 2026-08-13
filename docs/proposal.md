# Consistency-Localized Intrinsic Rewards for Robust Best-of-N Reasoning

## Motivation

This is an ICLR 2027 project direction. The SWIFT baseline learns a lightweight reward model directly from LLM hidden states. For a generated trajectory, it computes token-level rewards and gates, then aggregates them into a scalar reward for Best-of-N selection. CLIR uses this as an architectural reference, but the code in this repository is self-contained rather than a wrapper around the SWIFT repository.

The SWIFT-style reward is efficient, but it can still learn shortcuts: prompt style, domain wording, response length, reasoning format, or superficial context overlap.

Two observations motivate CLIR:

1. Different questions and domains may induce different hidden-state styles even when the underlying reasoning quality is the same. We can use LLM-guided rewrites to expose these spurious style factors and train the reward representation to ignore them.
2. In grounded generation, a trajectory may be useful until the first unsupported claim. After that point, downstream reasoning is contaminated. The reward should become negative from the hallucination onset onward, not merely penalize the final answer.

CLIR combines these into a hidden-state reward model with three auxiliary training signals: style/domain consistency, hallucination localization, and dual-prior evidence localization.

## Paper Anchors

- **SWIFT**: token-level linear reward and gating over hidden states for efficient Best-of-N sampling.
- **PRISM**: LLM-guided discovery of spurious correlations plus a contrastive-style projection objective that keeps same-label/different-spurious views close and pushes different-label/same-spurious views apart.
- **MLLM4WTAL / DPCL**: weak temporal localization with key priors, complete priors, and interactive distillation to address incomplete and over-complete localization.

CLIR adapts the second and third ideas from vision/video into LLM reasoning trajectories.

## Problem Setup

For a query `x`, optional context `c`, and generated trajectory `y = (y_1, ..., y_T)`, the frozen generator emits token hidden states:

```text
H = (h_1, ..., h_T), h_t in R^D
```

where `D` may be a concatenation of selected layer hidden states, logits, or another accessible internal representation.

### Frozen first-stage representation

For the first real pipeline pilot, the representation is no longer left open: the generator is
`microsoft/Phi-3.5-mini-instruct` on GSM8K, and `D` concatenates the embedding output and every
Transformer-block output with no additional final normalization. The pilot uses the exact token IDs
returned during generation; a later teacher-forced pass receives
`prompt_token_ids + output_token_ids` directly and slices the trajectory at the saved prompt length.
The query condition is acquired by a separate prompt-only forward once per query and shared by all
of that query's candidates; this avoids sequence-length-dependent GPU numerical drift in deep prompt
states. The pipeline must not reconstruct model input from heuristic reasoning-step splits. This keeps
every token-localized CLIR target indexed to the same generated sequence. The full frozen protocol,
checkpoint/data revisions, generation settings, and terminal-token policy are recorded in
`docs/pilot_protocol.md` and `configs/phi35_gsm8k_pilot_v1.json`.

### Full-layer encoder and controlled baselines

For Phi-3.5-mini, the raw per-token width is `D_raw = 33 * 3072 = 101376`. CLIR cannot apply
dense `D_raw -> D_raw` condition attention or fusion because that makes parameter complexity
quadratic in the raw concatenation. The primary encoder therefore reshapes every token to
`[33, 3072]`, applies one shared `3072 -> 256` layer projection, models the 33 layer vectors with
a two-block layer-axis Transformer, pools them with four learned attention queries, and projects
the result to `D_model = 768`. No layer is discarded, and all CLIR quadratic modules operate only
after this compression. The returned layer-pooling attention is retained as a diagnostic.

The experiment must separate three model variants on identical frozen features:

- `strict_swift`: raw all-layer features followed directly by `Linear(D_raw, 2)`;
- `encoded_swift`: the shared full-layer encoder followed only by the SWIFT gate/reward head;
- `clir`: the same encoder followed by condition fusion and all CLIR objectives.

The first contrast measures the encoder's effect; only the second contrast isolates the increment
from CLIR. A flat `D_raw -> D_model` projection is retained only as an encoder ablation.

The reward model outputs:

```text
g_t = sigmoid(w_g h_t + b_g)       token gate
r_t = w_r h_t + b_r                token reward
R(H) = sum_t g_t r_t / max(sum_t g_t, eps)
```

This recovers the SWIFT-style scalar reward. CLIR adds:

```text
z(H)       trajectory representation used for consistency
p_t        hallucination probability at token t
a_t        local progress / advantage estimate at token t
A_key,t    key support prior
A_comp,t   complete support prior
```

## LLM-Guided Augmentation

For each original sample, generate `K` semantics-preserving views:

```text
A(x, c, y) = { (x^k, c^k, y^k, s^k) }_{k=1..K}
```

where `s^k` is the style/domain attribute used for the rewrite.

Useful rewrite axes:

- Prompt style: direct, verbose, exam-style, conversational.
- Reasoning style: short derivation, detailed chain, equation-first, explanation-first.
- Context style: reorder paragraphs, paraphrase evidence, add irrelevant but non-conflicting distractors.
- Domain shell: rename entities or change surface setting while preserving the same formal structure.
- Length: compress or expand the trajectory without changing the answer.

The generator should preserve the answer and the support relation. Rewrites that alter the answer or evidence are filtered by a verifier.

## Consistency Objective

Inspired by PRISM, treat rewrite style/domain as a spurious attribute. Let `z_i` be the normalized projected representation of trajectory `i`, `u_i` its semantic group id, and `s_i` its style id.

Same semantics across different styles should be close:

```text
L_pos = mean_{u_i = u_j, s_i != s_j} (1 - cos(z_i, z_j))
```

Different semantics under the same style should be separated:

```text
L_neg = mean_{u_i != u_j, s_i = s_j} relu(cos(z_i, z_j) - m)
```

The scalar rewards should also be stable across style rewrites:

```text
L_score = mean_{u_i = u_j, s_i != s_j} (R_i - R_j)^2
```

The consistency loss is:

```text
L_cons = L_pos + L_neg + beta * L_score
```

This should reduce cases where the reward model prefers a trajectory because it is longer, more formal, or closer to a domain seen during training.

## Hallucination Localization and Negative Tail Rewards

For query+context tasks, define a token-level hallucination onset:

```text
tau = first token index where the trajectory makes an unsupported or contradicted claim
```

If no hallucination occurs, `tau = None`.

The target hallucination indicator is:

```text
h_t = 1[t >= tau] if tau exists, else 0
```

The hallucination head is trained with token-level BCE when onset labels are available:

```text
L_hall_tok = BCEWithLogits(p_t, h_t)
```

The token reward is shaped as:

```text
r_t target = a_t^*          for t < tau
r_t target = -gamma         for t >= tau
```

where `a_t^*` is a progress/advantage target. It can be produced from answer progress, evidence entailment, or a verifier score comparing `(query, context, prefix)` before and after token `t`.

A simple tail-margin loss is:

```text
L_tail = mean_{t >= tau} relu(r_t + gamma)^2
```

This enforces `r_t <= -gamma` after hallucination onset.

## Weak Localization When Onset Labels Are Missing

If only a path-level hallucination label `H_path` is available, use a multiple-instance objective:

```text
P_path = 1 - product_t (1 - sigmoid(p_t))
L_hall_mil = BCE(P_path, H_path)
```

Then estimate a pseudo-onset:

```text
tau_hat = min { t : sigmoid(p_t) > delta }
```

and apply the negative-tail reward on the pseudo tail with a lower weight.

This mirrors weak temporal localization: the path label says whether a bad event happened, while the model must infer where it began.

## Dual-Prior Localization View

A useful extension is to train two attention maps:

- **Key support prior**: tokens/windows most directly matched to query and retrieved context.
- **Complete support prior**: broader tokens/windows needed to reconstruct the full supported answer.

The key prior may be too narrow, while the complete prior may include irrelevant context. Following the dual-prior idea, alternate training can align them:

```text
L_key = L_relevance + lambda_1 * MSE(A_key, stopgrad(A_complete))
L_complete = L_reconstruction + lambda_2 * MSE(A_complete, stopgrad(A_key))
```

The SWIFT-style gate `g_t` can then be regularized toward the fused prior. This encourages the reward model to focus on tokens that are both locally evidential and globally necessary.

The current repository implements a guarded proxy of this idea in `src/consistency_localized_reward.py`:

- key/complete prior heads are always predicted for inspection;
- key/complete supervised losses are used only when external prior targets are present;
- mutual stop-gradient distillation and gate-prior regularization are evaluated only on tokens where both prior branches have label coverage, while preserving their probability mass from the full trajectory attention distribution;
- complete-prior reconstruction is enabled only when an independently generated, fixed
  `complete_reconstruction_target` of width `D_model` is present;
- `train_clir.py` supports `joint`, `key`, `complete`, and epoch-level `alternate` prior optimization.

This avoids the degenerate self-reconstruction solution where a uniform complete prior can trivially reconstruct the average trajectory feature. Stage 1 has no such real target, so this loss is exactly zero rather than being trained on a self-derived proxy. A full paper implementation should generate the target independently from masked supported-answer descriptions or verifier-derived evidence summaries, freeze it, and map it to `D_model` before training the reward model.

One open modeling choice is how much the progress head should contribute to the final token value. The current code uses `token_values = token_rewards + progress_score_weight * progress`, then applies hallucination-tail shaping to `token_values`. This keeps final scoring aligned with the localized penalty, but real annotations should test whether reward and progress need stronger separation.

## Full Objective

The training objective is:

```text
L = L_final
  + lambda_cons * L_cons
  + lambda_hall * (L_hall_tok or L_hall_mil)
  + lambda_tail * L_tail
  + lambda_adv * L_adv
  + lambda_prior * L_dual_prior
```

where `L_final` is the original SWIFT correctness BCE or a pairwise preference loss.

## Training Pipeline

1. Generate `N` trajectories per query with the task LLM and collect hidden states.
2. Label final correctness using answer checkers or verifier models.
3. Generate `K` LLM-guided rewrites for each trajectory and collect hidden states under teacher forcing or regeneration.
4. Filter rewrites that change the final answer or evidence relation.
5. Annotate hallucination labels:
   - strong: onset token/span labels from a verifier or human annotation;
   - weak: path-level hallucination indicator.
6. Train CLIR with final correctness, consistency, and localization losses.
7. Select Best-of-N trajectories by `R(H)`.

## Current Code Framework

The initial implementation in this repository is self-contained:

- `src/consistency_localized_reward.py`: SWIFT-style reward/gate backbone, token-level query/context attention fusion, PRISM consistency, hallucination localization, MIL, pseudo-onset tail loss, and guarded dual-prior localization.
- `src/clir_data.py`: JSONL dataset, collate utilities, and semantic-group batch sampler for pre-extracted hidden states.
- `train_clir.py`: single-trajectory BCE plus CLIR auxiliary losses, semantic-group batching, and dual-prior phase scheduling.
- `score_clir.py`: reward scoring, hallucination probability, pseudo-onset inference, prior diagnostics, and Best-of-N selection.
- `examples/create_toy_clir_data.py`: synthetic hidden-state data for smoke testing.
- `tests/test_clir_smoke.py`: forward/loss and dataset smoke tests.

## Evaluation Plan

Primary metrics:

- Reward-selected Best-of-N task accuracy. For each query, use the first `k` candidates in generation
  order, select the candidate with maximum reward, then average selected-candidate correctness over
  queries. This is not classical pass@k.
- The Phi/GSM8K pilot predeclares BoN@16 as its primary task metric and reports
  `k = 1, 2, 4, 8, 16`; final SWIFT parity expands to BoN@64 and also reports `32, 64`.
- BoN@1, random selection, SWIFT, and the oracle candidate-pool ceiling on the identical query groups.
- Hallucination rate among selected trajectories.
- Worst-augmentation accuracy across style/domain rewrites.
- Score variance across rewrites of the same trajectory.
- Token onset localization F1 / mAP when labels exist.

Key ablations:

- SWIFT only.
- SWIFT + style consistency.
- SWIFT + score consistency only.
- SWIFT + hallucination-tail loss.
- Weak-only localization vs explicit onset labels.
- Hidden states vs logits-only inputs.
- Number and type of LLM rewrites.
- Negative reward from onset vs only final-answer penalty.

## Expected Contribution

CLIR reframes intrinsic hidden-state reward modeling as a robust representation learning problem. The main novelty is not a larger reward model, but better supervision:

1. reward invariance across LLM-guided style/domain shifts;
2. token-level localization of where grounded reasoning becomes invalid;
3. negative-tail reward shaping so late hallucinations cannot be hidden by earlier correct reasoning.

This makes the method a natural ICLR submission direction: efficient test-time scaling with explicit robustness and faithfulness constraints.
