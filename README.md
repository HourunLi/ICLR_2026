# ICLR 2026: Consistency-Localized Intrinsic Rewards

This repository develops an extension of **SWIFT** ("Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling") for reasoning and retrieval-augmented generation settings where reward models must be robust to domain shift and sensitive to hallucination onset.

Working name: **CLIR** - Consistency-Localized Intrinsic Rewards.

## Core Idea

SWIFT shows that a lightweight linear reward head over LLM hidden states can rank reasoning trajectories efficiently. CLIR keeps that efficient hidden-state reward backbone, but adds two missing pieces:

1. **LLM-guided consistency under domain/style shift.** For each query or reasoning trajectory, an LLM produces semantics-preserving rewrites that vary prompt style, length, context order, surface domain, and reasoning format. The reward representation is trained to be invariant across these augmentations while still separating correct and incorrect trajectories.
2. **Hallucination-aware token localization.** For query+context inputs, the model predicts token-level progress and hallucination onset. Once a hallucination is detected, all later generated tokens receive negative reward. Optional onset labels can supervise the localization head directly; otherwise weak path-level hallucination labels can train it with multiple-instance localization.

The result is a reward model that is still small enough for Best-of-N sampling, but less likely to reward spurious prompt style cues or late-stage hallucinated reasoning.

## Repository Layout

- `docs/proposal.md` - ICLR-style method draft with notation, losses, training plan, and ablations.
- `src/consistency_localized_reward.py` - PyTorch module implementing the core reward, consistency, and localization losses.

## SWIFT-Based Implementation

A SWIFT-compatible implementation patch has been generated as a Codex output artifact: `outputs/swift_clir.patch`. It applies cleanly to `aster2024/SWIFT` and adds:

- `CLIRRewardModel` on top of SWIFT's token reward/gating interface.
- PRISM-style consistency loss over LLM rewrite metadata.
- Token-level hallucination onset, path-level MIL, and negative-tail reward losses.
- `--use_clir` training/scoring flags in SWIFT's `train/extract_train.py` and `eval/get_rewards.py`.

Apply it to a SWIFT checkout with:

```bash
git clone https://github.com/aster2024/SWIFT.git
cd SWIFT
git apply /path/to/outputs/swift_clir.patch
```

After applying the patch, train and score with `--use_clir` to enable the CLIR reward head and auxiliary consistency/localization losses.

## Baseline

The direct baseline is SWIFT:

```text
Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling
```

CLIR can be evaluated as incremental additions:

1. SWIFT token reward/gating head.
2. SWIFT + LLM-guided style/domain consistency.
3. SWIFT + hallucination-tail negative reward.
4. Full CLIR with consistency, token progress, hallucination localization, and optional localization labels.

## First Experiments

Recommended initial setup:

- Reasoning: MATH, GSM8K, AQuA-RAT, HellaSwag, CoinFlip.
- Grounded QA / hallucination: retrieval-augmented QA datasets with answerable and unanswerable contexts.
- Generation model: one open LLM where hidden states are accessible.
- Metrics: Best-of-N accuracy, hallucination rate after selection, onset localization F1, score consistency across augmentations, and worst-augmentation accuracy.

The most important ablations are the consistency projection, number/type of rewrites, token-level hallucination supervision, weak-only hallucination supervision, and whether negative reward begins exactly at the hallucination onset or only at the final answer.
