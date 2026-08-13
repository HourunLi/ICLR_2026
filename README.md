# ICLR 2027: Consistency-Localized Intrinsic Rewards

最后更新：2026-08-13

本仓库是 ICLR 2027 方向的初步代码框架。我们参考 **SWIFT** (*Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling*) 的 hidden-state token reward / gate 设计，但不依赖、不调用 SWIFT 仓库代码；CLIR 的模型、数据读取、训练和打分脚本都在本仓库内自包含实现。

当前工作名：**CLIR**，即 **Consistency-Localized Intrinsic Rewards**。

> **给新接手的人（人类或 AI）**：先看 `docs/handoff.md`。那份文档专门写给零上下文的人看，包含完整的架构讲解、数据 schema、bug 修复历史和下一步优先级排序；这份 README 更偏向 changelog，细节粒度不如那边。`docs/proposal.md` 是研究方法设计文档。

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
- 已冻结第一阶段真实实验协议：`docs/pilot_protocol.md`，机器可读配置为
  `configs/phi35_gsm8k_pilot_v1.json`。
- 已实现保留原始 prompt/output token IDs 的 GSM8K/vLLM 生成入口：
  `scripts/generate_gsm8k_rollouts.py`。
- 已实现 exact-token-ID teacher forcing、embedding + 全部 Transformer block 拼接的
  hidden-state 提取入口：`scripts/extract_hidden_states.py`。
- 真实 manifest 一旦带有 `output_token_ids`，数据加载器会在任何 pad/trim 之前严格检查
  `len(output_token_ids) == hidden_states.shape[0] == token-label length`。
- 已实现 smoke tests：`tests/test_clir_smoke.py` 和 `tests/test_clir_real_data.py`。
- 已添加可安装的真实生成/提取依赖；由于 SWIFT 官方的 NumPy pin 与 vLLM 冲突，采用
  `numpy==1.26.4` + `vllm==0.5.3.post1` 的解析后组合。
- Stage 1 单题真实对齐 gate 已实际通过：生成 16 个候选，并对前两条候选完成 exact-ID
  全层提取；两条 trajectory 为 `[162,101376]` / `[250,101376]`，共享的 prompt-only
  condition 为 `[113,101376]`，全部 finite、bfloat16。
- 已实现 checker 单位后缀归一化与 SWIFT parity 审计入口：
  `scripts/audit_swift_checker_parity.py`。17 个 query / 272 条候选上，CLIR v2 与固定
  SWIFT checker 一致 260 条（95.59%）；12 个差异全是官方 checker 的单位/尾随文本
  假阴性。
- 已把原始输入宽度 `input_dim` 与内部 `model_dim` 分离，并实现全层感知的
  `layer_transformer` 编码器：每个 token 的 `[33,3072]` 先经共享层投影、层轴 Transformer
  和 learned-query pooling，再进入 768 维 CLIR 模块。全部 33 层仍被使用。
- 已显式实现 `strict_swift`、`encoded_swift`、`clir` 三种模型，checkpoint 和打分输出都会
  记录变体。前者保持原始 `Linear(101376,2)`；后两者共享同规格编码器，避免把编码器收益
  误算成 CLIR 收益。
- 已在现有两条真实轨迹上完成三种模型的 correctness-only forward/backward gate：参数量
  分别为 202,754 / 3,435,266 / 9,547,273，全部 scores、loss、gradients finite；CLIR 最大
  参数矩阵为 `[768,3073]`，峰值 allocated 显存约 1.28 GB。该 gate 只属于工程证据。
- **Stage A 工程门已完成**：冻结 query-level split manifest
  `configs/splits/gsm8k_phi35_v1.json`（manifest SHA256
  `cb7f23e3da36c253d3fe7d3a33675db2e43eafb05685bf55d65511e35f553186`），包含
  train-primary 6000、validation 500、reserve 973、development-32 32、pilot-test 500、
  final-test 1319 个 query。生成和抽取现在按 query 原子分片、checksum 验证、成功 marker
  与断点续跑；`scripts/merge_query_shards.py` 只按冻结顺序合并完整 shard。
- development-32 已真实采集 32×8=256 条候选并提取全部 33 层：237 correct / 19
  incorrect，7 个 mixed-label query、25 个 all-correct query；统一特征规格为
  33×3072=101376、bfloat16，总特征字节 14,739,981,408。生成和抽取都通过“全完成不改写”
  与“移走一个 marker 后只重建该 query”的恢复验收。该结果属于 pipeline pilot。
- `train_clir.py` 现在支持显式、query-disjoint 的 train/validation manifest，保存模型、
  optimizer、completed epoch、Python/NumPy/Torch/CUDA RNG、data hash 与逐组件 metrics/counts
  的 full-state checkpoint；测试验证了 1 epoch 中断后恢复到 2 epochs 与连续训练参数逐
  tensor 完全相同。`evaluate_clir.py` 独立报告 reward BoN、random、oracle、pool coverage
  和 query bootstrap CI。
- development-32 的 24-query train / 8-query engineering validation 上，strict SWIFT、
  encoded SWIFT、CLIR 都在 101376 维真实输入上完成 1 epoch 训练、打分和 BoN@1/2/4/8
  闭环。验证池 7/8 query 的候选全对、1/8 为 7/8 对，过于简单，不能据此比较方法效果。
- **Stage 1 small-scale real correctness-only 已完成**：冻结的 512×8 train 和 128×16
  validation 共抽取 343.86 GiB 全 33 层 BF16 特征；三种模型各以 seeds 42/43/44 训练
  5 epochs。BoN@16 三 seed 为 strict SWIFT `88.28±2.07%`、encoded SWIFT
  `88.54±1.63%`、CLIR `89.32±2.51%`，random expected 为 `86.52%`、oracle 为
  `96.09%`。CLIR 相对 encoded 平均仅 `+0.78±2.07` 个百分点且一个 seed 为负，配对 CI
  未建立稳定增益。完整口径见 `docs/stage1_results.md`；`pilot_test` 尚未生成或检查。
- 新增 `summarize_clir.py`：严格校验多 seed evaluation 的 query/k/baseline 一致性，报告
  跨 seed 样本标准差，并在主指标上计算逐 query 配对 bootstrap CI。
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
- `SemanticGroupBatchSampler.__len__()` 现在和 `__iter__()` 复用同一套 batch 构造逻辑，避免不拆 semantic chunk 时低估 batch 数。
- hallucination onset 越界会直接报错，不再静默当成无幻觉样本。
- `query_id` 只用于 Best-of-N candidate 分组，不再默认 fallback 到 `semantic_id`，避免候选分组和 augmentation 分组混在一起。
- `condition_attention_temperature` 和 `progress_score_weight` 已接入 `train_clir.py` CLI。
- dual-prior 在部分 token 有标签时不再对子集重新归一化，而是比较完整轨迹 attention 分布在已标注 token 上的原始概率质量。

## 遇到的问题

- Stage 1 small-scale validation 已有跨 seed 结果，但 incremental 证据不稳定：
  encoded→CLIR 的 BoN@16 为 `+2.34/+1.56/-1.56` 个百分点，strict→encoded 也发生方向
  翻转。当前只能报告弱排序信号，不能表述为 CLIR 已优于共享编码器 baseline。
- validation 的 128 个 query 中只有 39 个 mixed pools；84 个 all-correct 与 5 个
  all-wrong pools 在 BoN@16 上不能区分选择器，导致配对 CI 较宽。下一轮应只按冻结顺序
  扩展 query prefix，不能根据答案 cherry-pick。
- 全层 feature 实测宽度为 101,376，每 token BF16 为 202,752 bytes；本轮 640 个 query
  的 feature payload 已达 343.86 GiB。Stage 1B 扩大 train/validation 前必须继续冻结容量
  上限并保留 query-atomic resume/checksum，不应以静默删层规避存储问题。
- vLLM `candidate.text` 在当前版本会比按原始 IDs decode 的文本多一个前导空格；272/272
  条都只差这个空格。`output_token_ids` 和由它 decode 的 `response` 仍是事实来源。
- SWIFT 官方 `evaluate_math` 未覆盖 `bolts`、`downloads` 等单位或某些尾随文本；共享审计
  272 条中产生 12 个假阴性。主标签使用冻结的 `clir_gsm8k_numeric_v2`，同时保留官方
  checker parity 数字，避免把修正后的标签冒充官方原样口径。
- LLM rewrite augmentation 的生成、过滤和 `semantic_id` / `style_id` / `domain_id` 元数据构造还没有自动化。
- hallucination onset / path-level label / token advantage / prior target 还需要 verifier 或 LLM judge 数据流水线生成。
- `complete_reconstruction_target` 现在被设计成外部 CSR-style target embedding；仓库不会再用 pooled hidden state 做自指重构，但真实 target 的生成方式还需要后续实现。
- `token_values = token_rewards + progress_score_weight * progress` 目前仍把 reward head 和 progress head 合并后做 hallucination tail shaping；toy 数据里 `progress_targets` 和 `token_advantage` 相同，真实数据接入后需要重新评估两个 head 的分工和量纲。
- 当前训练目标是单 trajectory BCE + auxiliary losses；后续还需要加入 pairwise preference、DPO、InfoNCA / NCA 风格目标，方便更直接对齐 SWIFT baseline。

当前 `SWIFT` 环境是 `torch==2.3.1+cu121`；测试数量以最新完整 `pytest` 输出为准；
`create_toy_clir_data.py -> train_clir.py（含 --condition_attention_temperature /
--progress_score_weight）-> score_clir.py` 端到端跑通，checkpoint 里也确认存了新增配置。
另外针对 `SemanticGroupBatchSampler` 做过 300 组随机场景、共 1200 次采样的压力测试，
确认 `__len__()` 与 `__iter__()` 实际 batch 数一致，样本覆盖和 `drop_last` 语义正确。
复查中发现的问题已同步处理：

- **已修复并复核：`SemanticGroupBatchSampler.__len__()` 低估实际 batch 数**。`__len__()` 与 `__iter__()` 复用同一套 `_build_batches()` 逻辑，`chunks.sort(key=len, reverse=True)` 让贪心装箱结果与是否 shuffle 无关，因此计数稳定；随机化压力测试没有发现反例。
- **已修复并复核：两个新的 `RewardConfig` 字段没有接到 CLI**。`condition_attention_temperature`、`progress_score_weight` 已加入 `train_clir.py` 参数并传入 `make_config()`，端到端跑过一遍确认写进了 checkpoint。
- **已修复并复核：`dual_prior_losses` 在标签部分覆盖时对子集重新归一化会失真**。现在 distill / gate-prior 保留完整轨迹 attention 分布的概率质量，只在 key 与 complete 都有标签覆盖的 token 上取 MSE；手工验算过 `test_dual_prior_partial_mask_preserves_full_attention_mass` 里的数值，和实现结果一致。
- **仍需观察，本轮未改动：`token_values` 把 `token_rewards` 和 `progress` 合并后再做幻觉相关监督**。`hallucination_localization_losses` / `pseudo_onset_tail_loss` 现在监督的是 `token_values = token_rewards + progress_score_weight * progress`，而 `progress` 同时还被 `progress_targets` 单独回归。当 `progress_targets` 和 `token_advantage` 来自同一份数据（目前 toy 数据就是这样）时，`token_rewards` 与 `progress` 之间没有显式的分工约束，真实数据接入后需要重新评估要不要拆开监督。这是设计取舍问题，不是逻辑 bug。

本轮复查没有发现新的逻辑 bug。

## 未来解决方向

- 先做 Stage 1B 诊断：在现有 39 个 mixed validation pools 上分析逐 query 分歧、长度偏差、
  score margin 与 layer attention，不利用 `pilot_test` 调参。
- 冻结新的 ordered-prefix 扩展配置以增加 mixed validation query；任何 class weighting、
  objective、epoch 或解码策略变化都必须版本化，不能覆盖 `stage1-small-scale-v1`。
- 仅在 validation 的 random 增益和 encoded→CLIR 增量跨 seed 稳定后锁定配置，再一次性
  进入 `pilot_test`；否则将当前结果保留为弱信号/负增量证据。
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
- real 数据接入后，重新评估 `token_values`（`token_rewards + progress_score_weight * progress`）是否需要拆分监督，或者给 `token_rewards`/`progress` 设计不重叠的目标。

## 运行代码

### 1. 准备环境

```bash
pip install -r requirements.txt
```

`requirements.txt` 现在包含真实数据阶段需要的 `transformers`、`datasets`、
`accelerate` 和 `vllm`。SWIFT 官方文件同时固定 `numpy==2.2.6` 和
`vllm==0.5.3.post1`，但后者要求 `numpy==1.26.4`；本仓库使用可解析、也与本地 SWIFT
复现报告一致的 `numpy==1.26.4`。大型外部 reward-model 微调才需要的
`cuml/cupy/peft/wandb` 等仍不安装。

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
  --prior_phase_mode alternate \
  --condition_attention_temperature 1.0 \
  --progress_score_weight 0.5
```

说明：

- `--group_by_semantic_id` 默认开启，用于保证同语义不同风格 rewrite 更容易进入同一 batch，从而触发 PRISM consistency。
- `--prior_phase_mode alternate` 默认开启，奇数 epoch 训练 key-prior phase，偶数 epoch 训练 complete-prior phase；也可设为 `joint`、`key` 或 `complete`。
- `--condition_attention_temperature` 控制 generated-token 到 query/context condition tokens 的 attention sharpness。
- `--progress_score_weight` 控制 progress head 进入最终 `token_values` 的权重。
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
python -m pytest -q tests/test_clir_smoke.py tests/test_clir_real_data.py
```

完整测试覆盖 toy/model、真实数据契约、模型变体、split/shard/merge、checkpoint resume
与 query evaluator。它们证明管线逻辑，不证明 Phi/CLIR 的研究效果。

```bash
python -m py_compile \
  src/consistency_localized_reward.py \
  src/clir_real_data.py \
  src/clir_data.py \
  train_clir.py \
  score_clir.py \
  scripts/generate_gsm8k_rollouts.py \
  scripts/extract_hidden_states.py \
  scripts/audit_swift_checker_parity.py \
  examples/create_toy_clir_data.py \
  tests/test_clir_smoke.py \
  tests/test_clir_real_data.py
```

### 6. 复跑第一条真实对齐 gate

完整协议和命令见 `docs/pilot_protocol.md` 与 `docs/runbook_zh.md`。第一轮必须保持
`--max-queries 1`。本仓库已有验收 artifact；复跑时使用新输出目录或显式
`--overwrite`，不能跳过该 gate 直接生成大规模数据。

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
- `docs/handoff.md`：如果改动大到影响架构、数据 schema、已知问题清单或优先级排序，同步更新这份交接文档，避免它过时后误导下一个接手的人。
