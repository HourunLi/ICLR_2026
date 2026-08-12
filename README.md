# ICLR 2026: Consistency-Localized Intrinsic Rewards

最后更新：2026-08-12

本仓库围绕 **SWIFT** baseline（*Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling*）做扩展，目标是在保持 hidden-state intrinsic reward 高效性的同时，增强 reward model 对 domain/style shift 的鲁棒性，并让它能定位 hallucination onset。

当前工作名：**CLIR**，即 **Consistency-Localized Intrinsic Rewards**。

## 代码目标

CLIR 在 SWIFT 的 token reward / gating reward 框架上加入两类监督：

1. **Domain/style consistency learning**  
   对同一个语义问题构造不同风格、长度、题干形式、context 排列或 domain 表达的 LLM rewrite / augmentation。模型需要让这些 augmentation 的 hidden-state reward representation 保持一致，从而削弱题干风格、长度、表面 domain 等 spurious cues。

2. **Hallucination-aware localized reward**  
   对 `query + context + trajectory` 输入，模型不仅输出整体 reward，还输出 token-level progress / advantage 和 hallucination gate。如果某个 token 被定位为 hallucination onset，那么从该点之后的 token reward 会被推向负值。若有 hallucination onset 标签，可以做额外定位监督；若只有 path-level hallucination 标签，则用弱监督 MIL 形式训练。

最终目标是在 Best-of-N sampling 中得到一个更稳的 lightweight reward model：既能筛出更好的 reasoning trajectory，又不容易奖励风格偏差或后半段幻觉。

## 当前进展

- 已完成 CLIR 方法设计草案，见 `docs/proposal.md`。
- 已完成一个独立 PyTorch scaffold，见 `src/consistency_localized_reward.py`。
- 已基于 `aster2024/SWIFT` 写出可应用的实现 patch：`outputs/swift_clir.patch`。
- patch 内容包括：
  - `CLIRRewardModel`，兼容 SWIFT 原有 scalar reward 接口；
  - PRISM-style consistency loss；
  - hallucination onset localization loss；
  - path-level hallucination MIL loss；
  - hallucination tail negative reward loss；
  - `train/extract_train.py` 的 `--use_clir` 训练入口；
  - `eval/get_rewards.py` 的 `--use_clir` scoring 入口；
  - `script/train_clir_example.sh` 示例脚本；
  - `tests/test_clir_smoke.py` smoke test。
- 已验证 `outputs/swift_clir.patch` 可以 cleanly apply 到全新的 SWIFT checkout。
- 已用 `python -m py_compile` 检查新增/修改 Python 文件的语法。

## 遇到的问题

- 当前环境没有安装 `torch` / `pytest`，所以目前只完成了语法检查和 `git apply --check`，还没有跑真实 forward / training smoke test。
- 本地 shell 无法直接 push 到 `HourunLi/ICLR_2026`，因此 SWIFT patch 文件本体目前作为 Codex output artifact 保存，还没有正式进入远端仓库路径。
- CLIR auxiliary losses 目前主要接入了 SWIFT 的 CE / Hinge 训练分支；DPO、InfoNCA、NCA 等训练分支还需要进一步接入。
- consistency learning 依赖 batch 中存在同一个 `semantic_id` 的不同 augmentation；后续需要数据构造脚本稳定生成 `semantic_id` / `style_id` / `domain_id`。
- hallucination-aware reward 依赖 token-level 或 path-level hallucination metadata；目前模型侧接口已经预留，但自动标注、验证器和数据流水线还未完成。

## 未来解决方向

- 将 `outputs/swift_clir.patch` 正式纳入仓库，例如放到 `patches/swift_clir.patch`，或直接维护一个 SWIFT fork / branch。
- 补充 augmentation 生成脚本：
  - 改写题干风格；
  - 改写 reasoning trajectory 长度；
  - 改写 context 顺序和表达；
  - 生成不同 domain surface forms；
  - 输出 `semantic_id`、`style_id`、`domain_id` 元数据。
- 补充 hallucination 标注流水线：
  - 使用 verifier / LLM judge 标注 path-level hallucination；
  - 进一步定位 hallucination onset token；
  - 生成 `hallucinated`、`hallucination_start`、`progress_targets` 等字段。
- 把 CLIR auxiliary losses 接入 SWIFT 的更多训练 objective，包括 DPO、InfoNCA 和 NCA。
- 在带 GPU 和完整依赖的环境中跑 end-to-end smoke test、small-scale training 和 Best-of-N evaluation。
- 做 ablation：
  - SWIFT；
  - SWIFT + consistency；
  - SWIFT + hallucination localization；
  - SWIFT + negative tail reward；
  - full CLIR。
- 增加评估指标：
  - Best-of-N accuracy；
  - hallucination rate after selection；
  - onset localization F1；
  - reward consistency across augmentations；
  - worst-augmentation accuracy。

## 运行代码

### 1. 应用 SWIFT patch

在本次 Codex 工作区中，patch 路径为：

```bash
/Users/hourunli/Documents/Codex/2026-08-12/github-plugin-github-openai-curated-remote/outputs/swift_clir.patch
```

应用到 SWIFT：

```bash
git clone https://github.com/aster2024/SWIFT.git
cd SWIFT
git apply /Users/hourunli/Documents/Codex/2026-08-12/github-plugin-github-openai-curated-remote/outputs/swift_clir.patch
```

如果后续 patch 已经放进本仓库 `patches/swift_clir.patch`，则可以改用：

```bash
git clone https://github.com/aster2024/SWIFT.git
cd SWIFT
git apply ../ICLR_2026/patches/swift_clir.patch
```

### 2. 语法和 patch 检查

```bash
python -m py_compile utils.py train/extract_train.py eval/get_rewards.py tests/test_clir_smoke.py
git apply --check /Users/hourunli/Documents/Codex/2026-08-12/github-plugin-github-openai-curated-remote/outputs/swift_clir.patch
```

### 3. 训练 CLIR reward model

示例命令：

```bash
python train/extract_train.py \
  --dataset math \
  --model_name meta-llama/Meta-Llama-3-8B \
  --max_samples 6000 \
  --methods ce \
  --reward_batch_size 16 \
  --memory_efficient \
  --use_clir \
  --clir_projection_dim 256 \
  --clir_consistency_weight 0.1 \
  --clir_hallucination_weight 0.5 \
  --clir_mil_weight 0.2 \
  --clir_tail_weight 0.1 \
  --output_model_file model/math/clir_reward_model_ce-llama3-8b-6000.pt
```

### 4. 用 CLIR reward model 打分

```bash
python eval/get_rewards.py \
  --model_name meta-llama/Meta-Llama-3-8B \
  --dataset math \
  --reward_model_load model/math/clir_reward_model_ce-llama3-8b-6000.pt \
  --use_clir \
  --output_file data/math/clir_extracted_rewards_ce-llama3-8b-6000.json
```

### 5. 运行 smoke test

完整依赖安装后运行：

```bash
pytest tests/test_clir_smoke.py
```

当前环境缺少 `torch` / `pytest`，因此这一步需要在完整 SWIFT 训练环境中执行。

## Baseline

直接 baseline 是：

```text
Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling
```

CLIR 推荐按以下顺序与 SWIFT 做增量比较：

1. SWIFT token reward / gating head。
2. SWIFT + LLM-guided consistency。
3. SWIFT + hallucination localization。
4. SWIFT + hallucination-tail negative reward。
5. Full CLIR。

## 维护规则

之后每次改代码，都同步维护这个 README：

- `代码目标`：如果研究目标或实现接口变了，要更新。
- `当前进展`：新增模块、脚本、测试、实验结果后要更新。
- `遇到的问题`：记录当前阻塞点、环境问题、设计风险。
- `未来解决方向`：把下一步任务按优先级写清楚。
- `运行代码`：任何命令、参数、路径、依赖变化都要同步更新。
