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
  - query/context conditional feature fusion；
  - PRISM-style semantic/style consistency loss；
  - hallucination onset token BCE；
  - path-level hallucination MIL；
  - pseudo-onset negative-tail reward；
  - token progress / advantage regression；
  - key prior / complete prior heads；
  - dual-prior interactive distillation；
  - reward gate 与 fused prior 的 regularization。

## 遇到的问题

- 当前 Codex 环境没有安装 `torch` / `pytest`，所以本地只能做语法级检查，真实 forward、training smoke test 需要在完整 PyTorch 环境里跑。
- 目前输入假设是已经抽取好的 frozen LLM hidden states；还没有实现从具体 LLM 自动抽 hidden states 的脚本。
- LLM rewrite augmentation 的生成、过滤和 `semantic_id` / `style_id` / `domain_id` 元数据构造还没有自动化。
- hallucination onset / path-level label / token advantage / prior target 还需要 verifier 或 LLM judge 数据流水线生成。
- 当前训练目标是单 trajectory BCE + auxiliary losses；后续还需要加入 pairwise preference、DPO、InfoNCA / NCA 风格目标，方便更直接对齐 SWIFT baseline。

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
  --lr 1e-3
```

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
- `clir_selected_best_of_n`

### 5. 运行 smoke test

```bash
pytest tests/test_clir_smoke.py
```

当前 Codex 环境缺少 `torch` / `pytest`，因此这一步需要在完整 PyTorch 环境中执行。

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
