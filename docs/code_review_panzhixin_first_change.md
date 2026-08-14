# 代码审查报告：`panzhixin` 分支第一次改动

- 审查日期：2026-08-13
- 审查范围：`panzhixin` 分支相对 `main` 的全部改动
  - 提交 `bf66282`（真实数据管线，约 +2042 行）
  - 审查时的未提交工作区改动（编码器 / model variant 重构，约 +485 行，diff 776 行）
- 审查基准：`/tmp/clir_review_snapshot/`（含 `HEAD.txt`、`uncommitted.diff`、`committed.diff`、`checksums.txt`）
- 审查目标（按需求）：**能否真正跑起来实验和训练，代码逻辑是否正确**。命名/风格问题不在范围内；崩溃、卡死、OOM、静默算错、扩规模阻塞在范围内。

> 说明：审查期间工作区被另一个 agent 并发修改（`score_clir.py`、`tests/test_clir_model_variants.py`、`scripts/gate_reward_architecture.py` 都是审查开始后出现的）。为保证结论自洽，全部结论基于上述冻结快照。落地修复前请先确认工作区已停止并发编辑。

---

## 0. 总体结论

改动方向是对的，真实数据路径确实能跑通。所有严重问题都不是"跑不起来"，而是**跑得起来但结果被静默污染**——尤其是 rollout 候选序号那一条，它直接决定试点的主指标 BoN@k 是否有效。

按需要处理的优先级：

| 优先级 | 问题数 | 性质 |
| --- | --- | --- |
| P0 | 4 | 静默污染实验结果，修复都只有几行 |
| P1 | 5 | 现在能跑，一接标签或一改配置就错 |
| P2 | 7 | 不影响正确性，影响能否跑完全量 |

---

## 1. 已实测通过的部分（不是只看代码）

- `tests/` 全部 26 个测试通过。
- 三个 variant 在**真实 101376 维全层特征**（`33 × 3072`，bf16）上 `train_clir.py` → `score_clir.py` 端到端跑通：
  - `strict_swift` 202,754 参数
  - `encoded_swift` 3,435,266 参数
  - `clir` 9,547,273 参数
- 显存充足：`clir`、`B=16`、`T=1024` 前反向峰值 **33.37 GiB / 80 GiB**（L20Z），单步 0.15 s。8 卡全空闲。
- `LayerAxisFeatureEncoder` 沿**层轴**（33）而非时间轴做 attention，因此时间维 padding 不会污染任何输出。逐个核对过 `masked_mean` / `masked_softmax` / `gates` / `complete_context` 的 mask 使用，均正确。
- 特征 I/O 在页缓存命中时很快：48.3 MiB 单文件 0.02 s。
- 未提交的编码器改动，正是 `README.md` / `docs/handoff.md` 里标注"真实训练前必须先实现"的低维全层编码器，实现方向正确。

---

## 2. P0：会静默污染实验结果

### P0-1 `candidate_index` 是 likelihood 排名，不是生成顺序

- 位置：`scripts/generate_gsm8k_rollouts.py:186`
- 根因：vLLM 0.5.3.post1 在 `vllm/outputs.py:111-118` 中，当 `len(seqs) > 1` 时按 `seq.get_cumulative_logprob()` **降序排序**后再构造 `CompletionOutput`。脚本用 `for candidate_index, candidate in enumerate(request_output.outputs)` 取序号，把 vLLM 已经保留在 `CompletionOutput.index` 里的原始序号丢掉了。

在已有的 17 题 × 16 候选（272 条）真实 rollout 上实测：

| 证据 | 数值 | 无偏时的期望 |
| --- | --- | --- |
| `candidate_index` 与输出长度的一致对数 | **0.672** | 0.5 |
| `cand-000` 平均输出长度 | 266.1 token | 与其他候选相同 |
| 最后一个候选平均输出长度 | 414.6 token | 同上 |
| `cand-000` 正确率 | **0.82** | 与其他候选相同 |
| `cand-015` 正确率 | **0.59** | 同上 |
| `oracle@1 = oracle@2 = oracle@4` | 0.824 | 应随 k 单调上升 |
| `oracle@16` | 0.941 | — |

原因是累积 logprob 没做长度归一化，天然偏爱短答案，而 GSM8K 上短答案正确率更高。

- 影响：`configs/phi35_gsm8k_pilot_v1.json` 中 `candidate_subset: first_k_in_generation_order` 失效。BoN@1 基线被抬高约 11 个点，random 基线不再随机，BoN@k 曲线被人为压平——**这是试点的主指标**。
- 修复（一行）：

```python
# 现在
for candidate_index, candidate in enumerate(request_output.outputs):
# 改为
for candidate in request_output.outputs:
    candidate_index = candidate.index
```

- **已生成的 272 条 rollout 无法事后修复**（原始序号已被丢弃），必须重新生成。建议同时把 `candidate.cumulative_logprob` 也落盘，便于事后核验顺序。

### P0-2 缺 `correctness` 的行被静默当成负样本

- 位置：`src/clir_data.py:270`（`add_optional_float(output, batch, "correctness")` 未传 `mask_key`）；`src/clir_real_data.py:290` 的 `validate_extracted_row` 不检查 `correctness` 是否存在。
- 实测：把两行 manifest 的第二行 `correctness` 字段删掉 → loader 照常接受 → `batch['correctness'] == [1.0, 0.0]`，且 batch 中**不存在** `correctness_mask` 键 → 该行作为 hard negative 参与训练。
- 影响：`correctness` 是当前真实数据上**唯一在训的标签**（见 P1-4），一旦提取阶段有任何行漏写该字段，就会被当成"答错"训进去，且没有任何告警。
- 修复：
  1. `add_optional_float(output, batch, "correctness", mask_key="correctness_mask")`，并在 `SwiftRewardBase.training_step` 的 final loss 里按 mask 归约；
  2. 在 `validate_extracted_row` 中要求 `correctness` 存在且取值在 `{0, 1}`（真实管线的行都应该有）。

### P0-3 token 级标签的别名绕过严格长度校验

- 位置：`src/clir_real_data.py:31` 的 `TOKEN_LABEL_FIELDS` 与 `src/clir_data.py:190-195` 的 `sequence_aliases` 不一致。
- 缺失的别名：`key_prior`、`complete_prior`、`progress`、`progress_target`。这四个别名 `extract_metadata` 接受，而且 `docs/handoff.md` 已经对外公布。
- 实测（同一个长度 10 的数组，真实行 `T=162`）：

| 写法 | 结果 |
| --- | --- |
| `key_prior_target` | 被严格 gate **拒绝**（正确行为） |
| `key_prior` | **被接受**，静默 zero-pad 到 162，`key_prior_mask` 全 True，`supervised_positions=162` |

即 152 个伪造的 0 被当作真实负样本监督。`docs/pilot_protocol.md` 明确禁止静默 pad/trim。

- 影响：这个洞正好开在下一个 milestone（dual-prior / onset / progress 标签）要写入的字段上。
- 修复：让 `TOKEN_LABEL_FIELDS` 和 `sequence_aliases` 共用同一份定义（例如在 `clir_data.py` 里导出别名表，`clir_real_data.py` 展开引用），保证新增别名自动纳入长度校验。

### P0-4 GSM8K 正确性判定会把对的判错

- 位置：`src/clir_real_data.py:163`（`_numeric_value`）、`src/clir_real_data.py:146`（`_last_numeric_expression`）

| 输入 | 当前判定 | 应为 | 原因 |
| --- | --- | --- | --- |
| `\boxed{60\%}` | 1 | 1 | — |
| `\boxed{60%}` | **0** | 1 | 未转义百分号被当作"除以 100"，解析为 3/5 |
| `\boxed{36\text{ cm}^2}` | **0** | 1 | `_last_numeric_expression` 取"结束位置最晚"的数值表达式，选中指数 `2` |

`_last_numeric_expression` 的 `max(candidates, key=lambda item: (item[0], item[1]))` 是按结束位置取最后一个；其 lookbehind `(?<![A-Za-z])` 没有排除 `^`。

- 当前是**潜伏 bug**：272 条真实 rollout 中 16 条含百分号的答案全部用了转义 `\%`，0 条答案含 `^`。但 `correctness` 是唯一在训的标签，一旦采样到未转义百分号或带单位指数的答案就会产生错标。
- 修复要点：
  1. 百分号当单位剥离；只有当 reference 也是分数/小数形式时才做 /100，参考答案是纯数就不要除；
  2. 在数字匹配中排除前一个字符是 `^` 的情况。
- **注意不要**改成"取第一个数值表达式"：我验证过那会把现在判对的 `\boxed{Profit Difference: $125 - $96 = $29}` 这类行搞坏。

---

## 3. P1：现在能跑，一接标签 / 一改配置就出问题

### P1-1 noisy-or path 概率在真实长度下饱和，梯度精确为 0

- 位置：`src/consistency_localized_reward.py:860`（`path_level_hallucination_mil`），饱和点在第 879 行的 `path_prob = 1 - prod(1 - probs)`。
- 实测：

| T | `path_prob` | label=0 的 MIL loss | grad_norm |
| --- | --- | --- | --- |
| 162 | 1.000000 | 100.00 | **0.000e+00** |
| 250 | 1.000000 | 100.00 | 0.000e+00 |
| 1024 | 1.000000 | 100.00 | 0.000e+00 |

`probs` 被 `clamp(1e-6, 1-1e-6)`，几百个 token 连乘后 `1 - prod(...)` 在 fp32 下精确等于 1.0，梯度整条链路归零。`score_clir.py` 写出的 `clir_path_hallucination_prob` 也因此恒为 1.0，作为诊断量毫无信息。

- 修复：在 log 空间计算，`log(1 - p_path) = Σ logsigmoid(-logits) · mask`。实测同一批数据下 loss 112.29、**grad_norm 6.364**，梯度恢复。（label=1 时梯度接近 0 是正常的——预测已经饱和在正确一侧。）

### P1-2 `protocol_sha256` 覆盖整个配置文件

- 位置：`scripts/extract_hidden_states.py:122`
- 实测：仅修改 `evaluation.formal_k`，hash 就从 `3c49bd02…` 变成 `ff981805…`，导致所有已生成的 rollout 被 provenance gate 拒绝、无法再提取特征，且没有任何绕过开关。
- 生成/提取阶段实际只依赖 `model` / `dataset` / `prompt` / `generation` / `hidden_states` 几个子树；`evaluation.*` 是下游分析参数。
- 修复：对上述子集做 canonical JSON 哈希；或提供 `--allow_protocol_hash_drift` 并把两个 hash 一起记录进 manifest。

### P1-3 编码器配置没有和 manifest 交叉校验

- 位置：`src/consistency_localized_reward.py:66-99`（`RewardConfig.__post_init__`）、`train_clir.py` 的新增 CLI 默认值。

两个静默陷阱：

1. `--encoder_type layer_transformer` 但不给 `--num_feature_layers/--per_layer_dim` → 默认 `(1, 101376)`，33 层被坍缩成 1 层。实测：静默通过、参数 34.9M、`layer_positions` 形状 `(1, 1, 256)`、无任何告警。**跑得动，但完全不是设计中的模型**，层注意力诊断全部无意义。
2. 不给 `--model_dim` → `model_dim` 默认等于 `hidden_dim = 101376` → CLIR 会尝试构造 `Linear(101376, 101376)` 量级的模块（三个这样的矩阵 fp32 约 123 GB），直接分配失败，且没有提前校验。`README.md` 提到过这个风险，代码里没有 guard。

- 现成的修复素材：并发新增的 `scripts/gate_reward_architecture.py:165-167` 已经有正确的检查：

```python
expected_dim = args.num_feature_layers * args.per_layer_dim
if input_dim != expected_dim:
    raise ValueError(f"feature width is {input_dim}, but layer contract gives {expected_dim}")
```

- 修复：把该检查搬进 `RewardConfig.__post_init__`；`train_clir.py` 直接从 manifest 的 `feature_metadata.layer_count` / `per_layer_hidden_size`（真实值 33 / 3072）读取默认值，而不是让用户在命令行重复写。同时对 `model_dim` 设一个上界告警（例如 `model_dim > 8192` 时报错要求显式确认）。

### P1-4 验证集按行切分，同一 query 的候选跨 split

- 位置：`train_clir.py:131-141`（`split_indices` 用 `torch.randperm` 对**行**打散）
- 实测：8 候选/题、`val_fraction=0.2` 时，**7/7 个 val query 全部在 train 中出现过**。
- 直接违反 `docs/pilot_protocol.md` 第 2 节"同一 query 的所有候选不得跨 split"。验证集指标会被同题泄漏抬高。
- 修复：先对 `query_id` 去重划分，再展开到行。

### P1-5 真实 manifest 上只有 `final` 一个 loss 在训

- 实测真实 2 行 manifest 上的 loss 明细：

| loss 项 | 值 |
| --- | --- |
| `final` | 0.3289 |
| `prior_key` / `prior_complete` / `prior_distill` / `prior_gate` / `prior_reconstruction` | 全部**精确等于 0.0** |
| consistency / localization | 未进入分支 |

原因：真实数据里没有 `semantic_id` / `style_id` / `hallucination_onset` / dual-prior 标签，也没有 `complete_reconstruction_target`。

- 影响：现在跑 `--model_variant clir` 得到的实际是"`encoded_swift` + 一批从未收到梯度的头"，`--prior_phase_mode alternate` 在交替两个什么都不做的阶段。**在标签管线做出来之前，`clir` vs `swift` 的对比不具备任何区分度，只能测出编码器的差异。**
- 这是已知排期问题（`docs/handoff.md` 有列），不算遗漏，但结论口径必须写清楚，避免把编码器收益当成 CLIR 收益。

---

## 4. P2：扩规模前的工程缺口

1. **`condition_cache` 从不清理** — `scripts/extract_hidden_states.py:146`。每 query 缓存一个 `[C, 101376]` bf16 张量约 23 MB；6000 题约 138 GB 常驻内存（本机 1024 GB，能撑但没必要）。加 LRU 上限即可。
2. **没有分片 / 续跑 / skip-existing** — 提取脚本的输出 JSONL 只在**最后一次性写出**，中途崩溃全部白跑；`_assert_target_available` 又拒绝覆盖，重跑还得手动清理。建议：`--shard_index/--num_shards`、逐行追加 manifest、已存在特征文件则跳过。
3. **DataLoader 没有并行与 pin_memory** — `train_clir.py:144`。每步需读 1.55–3.09 GiB（`B=8–16`, `T=1024`），单进程同步读盘。加 `num_workers` / `pin_memory` / `persistent_workers`。
4. **共享 condition 被重复处理** — 同一 query 的所有候选共用一份 prompt-only condition，但当前每个候选都重新读盘、重新 upcast 到 fp32、重新过一遍 encoder，batch 内还被复制 B 份。按 query 去重可直接省下可观时间和显存。
5. **每次 `__getitem__` 都做全量 isfinite 扫描** — `validate_extracted_row` 的 finite 检查在训练循环里反复跑，量级接近一次热加载。建议提取阶段校验一次并在 manifest 里打标，训练时用 `--skip_feature_finite_check` 跳过。
6. **checkpoint 只在最后一个 epoch 后保存，无 resume，无梯度裁剪** — `train_clir.py:249`。全量数据下单 epoch 约 3.5 小时量级，中断即全丢。至少加每 epoch 存盘 + `--resume` + `clip_grad_norm_`。
7. **仓库卫生与评测脚本缺口**
   - 没有 `.gitignore`，`run_artifacts/` 已有 104 MB 未跟踪文件，`git add -A` 会把特征文件提交进仓库。
   - 协议要求的 BoN@1/2/4/8/16（试点）、/32/64（正式）、random / BoN@1 / oracle 基线、paired bootstrap **全部还没有脚本**；`score_clir.py` 只写了一个 best-of-all 布尔 `reward_selected_best_of_n`。这是"跑实验"的必需件。
   - 存储：2.25 TB 外推 vs `/prodcpfs` 剩 48 TB（该挂载点已 100% used，共享盘），能放但没余量，建议先跑分片试点再决定全量。

### 附：一处口径问题（不影响计算）

`scripts/extract_hidden_states.py:193` 把 `apply_final_norm: False` 写进每个特征文件，但事实相反。实测 per-layer RMS：

| layer | 0 | 15 | 30 | 31 | 32 |
| --- | --- | --- | --- | --- | --- |
| RMS | 0.032 | 1.264 | 9.827 | 11.251 | **1.408** |

HuggingFace 返回的第 33 个 hidden state 是**已经过 final RMSNorm** 的，block 31 的原始输出从未被采集。该字段目前没有任何代码读取，不影响数值，但会让 33 层的 layer-attention 诊断不可比，论文/文档口径需要修正。

---

## 5. 明确排除的"伪问题"

以下几条听起来严重，但实测站不住，不要花时间：

| 说法 | 实测结论 |
| --- | --- |
| `B=16, T=1024` 会 OOM | 峰值 33.37 GiB / 80 GiB，不会 |
| `lr 1e-3` 会让 `strict_swift` 崩掉（因梯度过小） | 因果链不成立，AdamW 对梯度尺度不变；实测 grad norm 约 440。（2 行数据上几步就饱和是数据量问题，正式跑仍建议扫 lr） |
| condition cross-attention 数学上无法定位 | 随机初始化下接近均匀是 init 假象；q/k 对齐后 matched/unmatched 相差约 14×，可学 |
| `LayerNorm(101376)` 导致 `flat_linear` vs `layer_transformer` 的层方差失衡，从而决定消融结论 | 层间方差差异是 Phi 残差流本身的性质，LayerNorm 保持相对方差 |
| vLLM 采样不可复现 | 已设 `LLM(seed=42)`，同一条命令可复现（换分片方式不保证） |
| `score_clir.py` 输出 JSON 体积 / 内存会爆 | 量级远低于预估 |

（对抗验证共产出 25 条候选问题，14 条被实测推翻，11 条存活并写入本报告。）

---

## 6. 建议修复顺序

1. **P0-1**（候选序号）——决定要不要重新生成数据，先做。
2. **P0-2 / P0-3 / P0-4**（correctness mask、别名校验、判分器）——都是几行，且是唯一在训标签的正确性保障。
3. **P1-3 / P1-4**（配置交叉校验、按 query 切分）——决定你跑出来的是不是设计中的模型、验证集是否干净。
4. **P1-1 / P1-2**（noisy-or log 空间、protocol hash 范围）。
5. **P2**（分片/续跑、DataLoader、condition 去重、checkpoint、`.gitignore`、BoN 评测脚本）。
6. **P1-5** 是排期问题：标签管线到位之前，不要把 `clir` vs `swift` 的数字当作 CLIR 有效性的结论。
