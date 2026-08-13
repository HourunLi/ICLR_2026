# CLIR / SWIFT 第一阶段真实实验协议

协议版本：`phi35-gsm8k-pilot-v1`
冻结日期：2026-08-13
证据等级：`pipeline pilot`（工程证据，不是正式研究结论）

本文件记录项目所有者与 Codex 在正式实验开始前已经确认的选择。机器可读副本位于
`configs/phi35_gsm8k_pilot_v1.json`。协议若有修改，必须同时更新这两个文件，并在
结果中记录实际使用的协议文件 hash。

## 1. 目标和与 SWIFT 的关系

- 第一阶段使用 `microsoft/Phi-3.5-mini-instruct` 在 GSM8K 上打通真实候选生成、
  correctness、全层 hidden-state 提取、严格 SWIFT baseline 和 Best-of-N 评估。
- Phi-3.5-mini 不是 SWIFT 官方主实验模型，因此这一阶段是 SWIFT extension / CLIR
  pipeline pilot，不能表述为严格复现论文主表。
- SWIFT 官方仓库对照固定到 commit
  `41f7c9f7e13734267450870f977e5dd7d62ac23e`。
- 候选、token 特征、query split 和评估预算必须在 SWIFT 与 CLIR 之间完全共享。

## 2. 固定的数据、模型和生成口径

### 数据

- 数据集：`openai/gsm8k`，subset `main`。
- 数据 revision：`740312add88f781978c0658806c59bc2815b9866`。
- 所有 split 以原始 query 为单位；同一 query 的所有候选和后续 rewrites 不得跨 split。
- SWIFT parity 报告使用官方代码口径的 GSM8K test 前 500 题；最终实验另报告完整
  GSM8K test，二者不能混写。
- 首个端到端验收只运行极小 development slice。它只用于验证数据契约、吞吐和存储，
  不用于选择模型或报告效果。

### 模型

- 生成器：`microsoft/Phi-3.5-mini-instruct`。
- 模型与 tokenizer revision：
  `2fe192450127e6a83f7441aef6e3ca586c338b77`。
- 模型 forward 使用 bfloat16。reward 特征的落盘 dtype 必须写入 extraction manifest；
  训练/打分加载器保持 BF16，并在 CUDA autocast 下消费，避免 101376 维特征在 CPU 重复
  扩展为 FP32。只有通过 extraction marker/checksum/finite 验收的 manifest 才可显式跳过
  每个 epoch 的重复 finite 全量扫描。

### Prompt 与 sampling

GSM8K prompt 与 SWIFT 官方脚本保持一致：

```text
Solve the following math problem step-by-step.
Simplify your answer as much as possible. Present your final answer as \boxed{Your Answer}.
{question}
```

- temperature：`1.0`
- top-p：`0.9`
- max new tokens：`1024`
- max model length：`4096`
- seed：`42`
- 训练 query 每题生成 8 个候选。
- pilot 评估每题生成 16 个候选，报告 `BoN@1/2/4/8/16`，主指标为 `BoN@16`。
- 最终 SWIFT parity 扩展到 64 个候选，报告 `BoN@1/2/4/8/16/32/64`，主指标为
  `BoN@64`。

## 3. 原始 token-ID 数据契约

生成阶段必须为每个候选保存：

- `prompt_token_ids`：实际传给生成器的、已经套用 chat template 的 prompt IDs；
- `output_token_ids`：生成后端实际返回的候选 IDs；
- 原始 response 文本、reference、解析结果、correctness、finish/stop reason；
- 模型/tokenizer revision、chat template hash、sampling 参数和软件版本。

hidden states 可以在生成后 teacher-force 重新提取，但输入必须直接使用：

```text
prompt_token_ids + output_token_ids
```

禁止使用以下路径重建特征输入：

```text
response -> split_steps -> join -> tokenize
```

`steps` 只能作为从原始 token 序列派生的标注或展示元数据，不能成为 token 序列的事实
来源。所有 token-level 标签以 `output_token_ids` 的零基索引为唯一坐标。

主实验的 terminal policy 是 `as_returned`：不人为补造 EOS/EOT；若生成后端返回的 token
IDs 本身包含终止 token，则原样保留。SWIFT 官方 chat-template EOT 行为仅作为单独的
parity diagnostic，不与主比较混写。

## 4. Hidden-state 定义

- 对精确的 `prompt_token_ids + output_token_ids` 做 causal teacher-forced forward。
- 设置 `output_hidden_states=True`、`use_cache=False`，不启用梯度。
- 使用 `outputs.hidden_states` 的全部元素：embedding 输出以及每个 Transformer block
  的输出。
- 不额外施加 final layer normalization。
- 对每个位置沿 feature 维拼接全部层。
- trajectory feature 严格切片为 output token 区间，形状为 `[T, D]`，其中
  `T = len(output_token_ids)`。
- acquisition 阶段对每个 query 的精确 prompt IDs 单独做一次 prompt-only forward，并缓存
  全部 prompt token 的 `[C, D]` condition features。同一 query 的全部候选必须共享这一份
  canonical condition；不能从不同总长度的 `prompt + output` forward 分别切出 prompt states，
  因为 GPU kernel 的序列长度相关数值路径会让深层前缀状态产生不可忽略的浮点漂移。
- CLIR 后续可以使用整段条件序列，最后一个 generation-boundary token 作为条件表示的方案
  保留为模型消融，而不是在提取阶段丢弃其余 prompt states。
- “最后 4 层”不用于 pilot 主路径，只保留为后续效率/层选择消融。

对每条真实样本，进入训练前必须满足：

```text
len(output_token_ids)
= trajectory_hidden_states.shape[0]
= 每一个已提供 token-level label 的长度
```

加载器不得通过静默 pad/trim 把正式数据的错位隐藏掉。

### Reward 输入编码与三种强制对照

原始特征仍完整保留 33 组状态，即每个 token 的输入为
`D_raw = 33 × 3072 = 101376`。但 CLIR 内部不得把 `D_raw` 当作 attention、fusion 或
reconstruction 的工作宽度，因为 `Linear(D_raw,D_raw)` 会产生约 102.8B 参数。主编码器
在任何平方复杂度操作之前执行：

1. 将每个 token reshape 为 `[33,3072]`，不丢弃任何层；
2. 对每层使用共享的 `3072 → 256` 投影并加入可学习层位置；
3. 使用 2 层、8 头的 layer-axis Transformer 建模层间关系；
4. 使用 4 个可学习 query 对 33 层做 attention pooling；
5. 将结果映射为 `model_dim=768`，后续 CLIR 模块只在 768 维工作。

必须在完全相同的 candidates、features、correctness 和 split 上训练三种显式变体：

- `strict_swift`：原始 101376 维直接接官方形式的 `Linear(D_raw,2)`，不使用编码器；
- `encoded_swift`：共享上述编码器，只接 SWIFT gate/reward head；
- `clir`：共享同一个编码器，再接 condition fusion 与 CLIR auxiliary heads。

这样 `strict_swift → encoded_swift` 测量编码器影响，`encoded_swift → clir` 才测量 CLIR
方法模块的增量。`flat_linear` 只允许作为开发/消融变体，不能替代主编码器。

`complete_reconstruction_target` 不是 correctness 标量，也不是当前候选 trajectory 自己的
均值。它必须是独立生成、固定不回传梯度的完整证据/答案摘要向量，宽度等于
`model_dim=768`。第一阶段真实数据还没有这种外部目标，所以字段缺失时 reconstruction
loss 必须严格等于 0；禁止用当前候选自身池化特征伪造 target。这个缺失不阻塞
strict/encoded SWIFT 或 correctness-only CLIR baseline。

## 5. Correctness 与 Best-of-N

- correctness checker 使用原始 response，而不是重建的 steps。
- 保存 raw answer、parsed answer、reference、checker 结果和失败原因。
- pilot 阶段先使用确定性的 GSM8K 数值/boxed answer checker
  `clir_gsm8k_numeric_v2`。保留完整 boxed 内容供审计；若 boxed 内容因右侧单位而不能直接
  解析成数值（如 `160 minutes`），按 SWIFT 官方 evaluator 的单位归一化语义取其中最后
  一个数值表达式后比较。必须人工审计 positive、negative、parse failure 的分层样本；
  正式报告前要与固定 commit 中的 `generate.generate_utils.evaluate_math` 在共享样本上核对
  一致率。

对 query `q` 的前 `k` 个生成候选，定义：

```text
BoN@k(s) = mean_q correctness[q, argmax_{i < k} score[q, i]]
```

这是 reward-selected Best-of-k accuracy，不是经典 pass@k。候选集合按生成顺序取前 `k`
个。除 SWIFT 和 CLIR 外必须报告 random selection、BoN@1 和 oracle ceiling。正式结果按
query 做 paired bootstrap confidence interval，并跨多个训练 seed 报告均值和标准差。

## 6. 阶段门与尚未冻结的事项

Stage 1 的真实数据与严格对齐提取、以及 reward architecture gate 均已通过。扩大数据前
仍必须保持以下验收条件：

1. 原始 token IDs decode 后能恢复保存的 response；
2. `[T,D]` 的 `T` 与 output IDs 严格一致；
3. tensor 全部 finite，层数、hidden dimension 和 dtype 与 manifest 一致；
4. condition features 在同一 query 的不同候选之间一致；
5. correctness checker 经过人工小样本审计；
6. 生成/提取时间和真实落盘字节数已经测量。

### Stage A query split 与恢复门（已通过）

- split manifest：`configs/splits/gsm8k_phi35_v1.json`，SHA256
  `cb7f23e3da36c253d3fe7d3a33675db2e43eafb05685bf55d65511e35f553186`。
- 6000 train-primary / 500 validation / 973 reserve；development-32 是 train-primary 的
  工程子集；500 pilot-test 是 1319 final-test 的子集。
- query 是生成/抽取的最小原子恢复单元；每个 marker 包含 payload size+SHA256，合并按冻结
  membership 顺序并拒绝缺失、损坏、候选数/索引不连续。
- 32-query 真实运行中，生成和抽取均验证了全完成恢复不改 marker；移走单个 marker 后，
  所属 shard 精确跳过 3 个健康 query、只重建 1 个。抽取重建后总 manifest SHA 不变。

### 2026-08-13 gate 验收记录

Stage 1 单题 gate 已通过，但这只是工程验收，不是 SWIFT/CLIR 效果结论：

- test index 0 生成 16 个候选；`prompt_token_ids` 为 113 个，候选长度范围为
  156--274，首两条为 162/250；终止 token 按后端原样保存。
- vLLM 的 `candidate.text` 比按 IDs decode 的文本统一多一个前导空格（272/272 条复现），
  正文完全一致；事实来源保持为保存的 IDs 和由其 decode 的 `response`。
- 首两条候选提取结果分别为 `[162,101376]`、`[250,101376]`，canonical condition 为
  `[113,101376]`；`101376 = 33 * 3072`，对应 embedding + 32 blocks。全部 tensor finite，
  落盘 dtype 为 bfloat16，condition 文件由两条候选共享。
- 两条 trajectory 加一份 condition 的实际总字节数为 `106,450,149`；纯 tensor 理论成本
  为每 token `202,752` bytes。首题全部 16 条按实测长度估算约 727.47 MB（含一份
  condition）。按同一题平均长度外推，6000 train queries × 8 candidates 约 2.25 TB；
  该外推只用于设计分片/流式读取，不能替代扩大 development slice 后的正式容量测量。
- checker 审计覆盖 17 个 query、272 条候选：v2 checker 为 196 positive、75 numeric
  negative、1 non-numeric negative，无生成态 parse failure；parse-failure 分支另有测试。
  对固定 SWIFT commit 的 `evaluate_math` 一致率为 260/272（95.59%）。12 个分歧全部是
  SWIFT checker 对单位或尾随文本的假阴性，v2 没有相对 SWIFT 的假阴性；正式报告必须
  同时披露该 parity diagnostic。
- 模型已缓存时，单题 16 candidates 的 vLLM 初始化加生成约半分钟，2 条全层提取约
  20--25 秒；首次约 7.6 GB 权重下载耗时约 5 分钟。吞吐需在分片提取实现后重新测量。
- 同一两条真实 trajectory/condition 上，三种 reward 变体均完成 correctness-only
  forward/backward，scores、loss 和 gradients 全部 finite。`strict_swift`、
  `encoded_swift`、`clir` 的可训练参数分别为 202,754、3,435,266、9,547,273；CLIR 最大
  参数矩阵为 `[768,3073]`，不含原始宽度平方矩阵。L20Z 上三者峰值 allocated 显存分别约
  0.22/0.98/1.28 GB。该结果只证明架构和数据链路可运行，不证明 reward 有效。

尚未冻结、不能由实现者自行假定的研究选择：

- CLIR hallucination judge、onset 标注和 dual-prior target 的生成协议；
- full CLIR 中 progress 与 token reward 是否继续共享 tail-shaping 目标。
