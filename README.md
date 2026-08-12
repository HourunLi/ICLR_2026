# ICLR 2027: Consistency-Localized Intrinsic Rewards

最后更新：2026-08-12

本仓库是 ICLR 2027 方向的初步代码框架。我们参考 **SWIFT** (*Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling*) 的 hidden-state token reward / gate 设计，但不依赖、不调用 SWIFT 仓库代码；CLIR 的模型、数据读取、训练和打分脚本都在本仓库内自包含实现。

当前工作名：**CLIR**，即 **Consistency-Localized Intrinsic Rewards**。

## 代码目标

CLIR 在 SWIFT-style hidden-state reward backbone 上加入三类监督：

1. **Domain/style consistency learning**  
   参考 PRISM：对同一个语义问题构造不同风格、长度、题干形式、context 排列或 domain 表达的 LLM rewrite / augmentation。模型需要让这些 augmentation 的 projected reward representation 保持一致，从而削弱题干风格、长度、表面 domain 等 spurious cues。

2. **Hallucination-aware localized reward**  
   对 `query + context + trajectory` 输入，模型输出整体 reward、token-level progress / advantage 和 hallucination probability。如果某个 token 被定位为 hallucination onset，那么从该点之后的 token reward 会被推向负值。若有 onset 标签，做 token-level 定位监督；若只有 path-level hallucination 标签，则做弱监督 MIL，并生成 pseudo onset 做低权重 negative-tail shaping。

3. **Dual-prior localization**  
   参考 Weakly Supervised Temporal Action Localization via Dual-Prior Collaborative Learning：模型学习 key support prior 和 complete support prior。前者偏向最关键证据 token，后者偏向完整支持答案的 token span，再通过互蒸馏和 gate-prior regularization 约束 reward gate。

最终目标是在 Best-of-N sampling 中得到一个更稳的 lightweight reward model：既能筛出更好的 reasoning trajectory，又不容易奖励风格偏差或后半段幻觉。

## 当前进展

- 已完成 CLIR 方法设计草案，见 `docs/proposal.md`。
- 已实现自包含 PyTorch 模型框架：`src/consistency_localized_reward.py`。
- 已实现 JSONL 数据读取和 batch collate：`src/clir_data.py`。
- 已实现训练入口：`train_clir.py`。
- 已实现打分和 Best-of-N 选择入口：`score_clir.py`。
- 已实现 toy 数据生成脚本：`examples/create_toy_clir_data.py`。
- 已实现 smoke tests：`tests/test_clir_smoke.py`。
- 当前模型包含：
  - SWIFT-style token reward / gate aggregation；
  - token-level query/context cross-attention conditional fusion；
  - condition relevance 与 condition attention 输出，供定位诊断使用；
  - gate-weighted `token_values` + trajectory residual 的最终 score logit；
  - PRISM-style semantic/style consistency loss；
  - 按 `semantic_id` 分组的 batch sampler，默认让同语义不同风格 augmentation 落入同一 mini-batch；
  - hallucination onset token BCE；
  - path-level hallucination MIL；
  - pseudo-onset negative-tail reward；
  - token progress / advantage regression；
  - key prior / complete prior heads；
  - 有外部 prior labels 时启用 dual-prior interactive distillation，且只在 key/complete 共同有监督覆盖的 token 上对齐；
  - 有外部 `complete_reconstruction_target` 时启用 complete-prior reconstruction；
  - reward gate 与 fused prior 的 regularization；
  - 打分脚本输出 key/complete prior、gate attention、condition relevance 和 gate-prior alignment。

本次同步修复了前一轮代码审查指出的关键问题：

- 条件化不再是给所有 token 加同一个 query/context 偏置，而是每个 generated token 对 condition tokens 做 attention，再把 token-context match 特征送入后续 heads。
- dual-prior loss 不再无标签常开；没有 `key_prior_target` / `complete_prior_target` / `complete_reconstruction_target` 时，对应 loss 为 0。
- dual-prior 的 distillation / gate-prior regularization 只在 key 与 complete prior 都有标签的 token 上计算，避免半标注样本把 prior 拉向无意义一致。
- dual-prior 支持 `joint`、`key`、`complete` 和 epoch-level `alternate` 训练模式，默认用 `alternate` 贴近 DPCL 的交替优化思路。
- PRISM consistency 不再依赖普通随机 shuffle 碰运气；训练默认启用 `SemanticGroupBatchSampler`。
- hallucination onset 越界会直接报错，不再静默当成无幻觉样本。
- `query_id` 只用于 Best-of-N candidate 分组，不再默认 fallback 到 `semantic_id`，避免候选分组和 augmentation 分组混在一起。

## 遇到的问题

- 目前输入假设是已经抽取好的 frozen LLM hidden states；还没有实现从具体 LLM 自动抽 hidden states 的脚本。
- LLM rewrite augmentation 的生成、过滤和 `semantic_id` / `style_id` / `domain_id` 元数据构造还没有自动化。
- hallucination onset / path-level label / token advantage / prior target 还需要 verifier 或 LLM judge 数据流水线生成。
- `complete_reconstruction_target` 现在被设计成外部 CSR-style target embedding；仓库不会再用 pooled hidden state 做自指重构，但真实 target 的生成方式还需要后续实现。
- 当前训练目标是单 trajectory BCE + auxiliary losses；后续还需要加入 pairwise preference、DPO、InfoNCA / NCA 风格目标，方便更直接对齐 SWIFT baseline。

本次复查已经在装有 `torch`（2.13, CPU）/ `pytest` 的环境里实测：`pytest tests/test_clir_smoke.py` 7/7 通过；`create_toy_clir_data.py -> train_clir.py -> score_clir.py` 端到端跑通，无崩溃、无 NaN。复查中发现以下遗留问题，尚未修复：

- **`SemanticGroupBatchSampler.__len__()` 会低估实际 batch 数**：当很多语义组的大小和 `batch_size` 不能整除时（例如 5 个大小为 2 的语义组、`batch_size=5`），`__iter__()` 会因为"不拆散同组 chunk"而提前 flush 出偏小的 batch，实测会产出 3 个 batch，但 `__len__()` 按 `ceil(N/batch_size)` 算出来是 2（已用脚本复现：`batches=[[0,1,2,3],[4,5,6,7],[8,9]]` vs `len(sampler)=2`）。当前 `train_clir.py` 没有用到 `len(loader)`，所以暂不影响训练本身；但如果之后加进度条或按 `len(loader)*epochs` 算 LR 总步数，会读到错误的步数。
- **两个新的 `RewardConfig` 字段没有接到 CLI**：`condition_attention_temperature`、`progress_score_weight` 现在只能用 dataclass 默认值，`train_clir.py` 的 `make_config()` 没有像其它 loss 权重一样暴露对应的 `--xxx` 参数。
- **`dual_prior_losses` 里 `distill` / `gate_prior` 的重新归一化在标签部分覆盖时会失真**：这两项用 `shared_prior_mask`（key 与 complete 标签都覆盖的 token）对 `key_prior`/`complete_prior`/`gates`/`fused_prior` 重新做 `normalize_attention`。标签是全覆盖（目前 toy 数据和 smoke test 都是）时没问题；但如果以后接入只标注部分 token 的真实弱监督数据，重新归一化会把"标注子集内的相对权重"当成比较对象，而不是这些 token 在完整轨迹注意力里的真实占比，可能夸大或掩盖 key/complete 之间的真实分歧。
- **`token_values` 把 `token_rewards` 和 `progress` 合并后再做幻觉相关监督**：`hallucination_localization_losses` / `pseudo_onset_tail_loss` 现在监督的是 `token_values = token_rewards + progress_score_weight * progress`，而 `progress` 同时还被 `progress_targets` 单独回归。当 `progress_targets` 和 `token_advantage` 来自同一份数据（目前 toy 数据就是这样）时，`token_rewards` 与 `progress` 之间没有显式的分工约束，只是训练时被动地推出一个隐式拆分，值得在接入真实数据后重新评估要不要拆开监督。

## 未来解决方向

- 增加 hidden-state extraction 脚本：
  - 输入 query、context、trajectory；
  - 输出 generated-token hidden states；
  - 可选择保存 query/context condition hidden states。
- 补充 augmentation 生成脚本：
  - 改写题干风格；
  - 改写 reasoning trajectory 长度；
  - 改写 context 顺序和表达；
  - 生成不同 domain surface forms；
  - 输出 `semantic_id`、`style_id`、`domain_id` 元数据。
- 补充 hallucination 标注流水线：
  - 使用 verifier / LLM judge 标注 path-level hallucination；
  - 进一步定位 hallucination onset token；
  - 生成 `hallucination_onset`、`path_hallucinated`、`token_advantage`、`progress_targets`。
- 补充 dual-prior target 生成：
  - key support prior：最关键证据 token/span；
  - complete support prior：完整支持答案的 token/span；
  - 用 query/context entailment 或 verifier score 生成 weak prior。
- 加入 pairwise/listwise 训练 objective，并做 SWIFT 对齐实验。
- 在带 GPU 和完整依赖的环境中跑 end-to-end toy training、small-scale training 和 Best-of-N evaluation。
- 修 `SemanticGroupBatchSampler.__len__()` 的低估问题，让它和 `__iter__()` 实际产出的 batch 数一致。
- 把 `condition_attention_temperature`、`progress_score_weight` 补成 `train_clir.py` 的 CLI 参数。
- 评估 `dual_prior_losses` 在部分覆盖标签下的归一化方式是否需要改成"按完整轨迹 mask 归一化、只在 shared 子集上取 MSE"，而不是先按 shared 子集重新归一化。
- real 数据接入后，重新评估 `token_values`（`token_rewards + progress_score_weight * progress`）是否需要拆分监督，或者给 `token_rewards`/`progress` 设计不重叠的目标。

## 运行代码

### 1. 准备环境

```bash
pip install torch numpy pytest
```

真实实验还需要安装用于抽取 hidden states 的 LLM 依赖，例如 `transformers`、`accelerate` 等。

### 2. 生成 toy 数据

```bash
python examples/create_toy_clir_data.py \
  --output_jsonl examples/toy_clir.jsonl \
  --feature_dir examples/features \
  --hidden_dim 8
```

### 3. 训练 CLIR reward model

```bash
python train_clir.py \
  --train_jsonl examples/toy_clir.jsonl \
  --output_model outputs/clir_toy.pt \
  --hidden_dim 8 \
  --projection_dim 4 \
  --batch_size 4 \
  --epochs 3 \
  --lr 1e-3 \
  --group_by_semantic_id \
  --prior_phase_mode alternate
```

说明：

- `--group_by_semantic_id` 默认开启，用于保证同语义不同风格 rewrite 更容易进入同一 batch，从而触发 PRISM consistency。
- `--prior_phase_mode alternate` 默认开启，奇数 epoch 训练 key-prior phase，偶数 epoch 训练 complete-prior phase；也可设为 `joint`、`key` 或 `complete`。
- `query_id` 用于 Best-of-N candidate 分组；`semantic_id` 用于 rewrite/augmentation consistency 分组；`style_id` 或 `domain_id` 用于 spurious attribute 分组。

### 4. 用 CLIR reward model 打分

```bash
python score_clir.py \
  --input_jsonl examples/toy_clir.jsonl \
  --model outputs/clir_toy.pt \
  --output_jsonl outputs/clir_toy_scores.jsonl \
  --batch_size 4
```

输出 JSONL 会增加：

- `clir_score`
- `clir_path_hallucination_prob`
- `clir_pseudo_onset`
- `clir_mean_gate`
- `clir_prior_gate_alignment`
- `clir_condition_relevance`
- `clir_gate_attention`
- `clir_key_prior`
- `clir_complete_prior`
- `clir_selected_best_of_n`

### 5. 运行 smoke test

```bash
pytest tests/test_clir_smoke.py
```

已在装有 `torch`（2.13, CPU）/ `pytest` 的环境中实测通过（7/7）；同时跑通了 `create_toy_clir_data.py -> train_clir.py -> score_clir.py` 端到端流程，无崩溃、无 NaN。如果所在环境没有 `torch` / `pytest`，至少应先跑一遍语法检查：

```bash
python -m py_compile \
  src/consistency_localized_reward.py \
  src/clir_data.py \
  train_clir.py \
  score_clir.py \
  examples/create_toy_clir_data.py \
  tests/test_clir_smoke.py
```

## Baseline

直接 baseline 是：

```text
Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling
```

CLIR 推荐按以下顺序与 SWIFT 做增量比较：

1. SWIFT token reward / gating head。
2. SWIFT-style reward backbone in this repo。
3. CLIR + LLM-guided consistency。
4. CLIR + hallucination localization。
5. CLIR + dual-prior localization。
6. Full CLIR。

## 维护规则

之后每次改代码，都同步维护这个 README：

- `代码目标`：如果研究目标或实现接口变了，要更新。
- `当前进展`：新增模块、脚本、测试、实验结果后要更新。
- `遇到的问题`：记录当前阻塞点、环境问题、设计风险。
- `未来解决方向`：把下一步任务按优先级写清楚。
- `运行代码`：任何命令、参数、路径、依赖变化都要同步更新。
