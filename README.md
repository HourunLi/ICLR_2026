# CLIR: Consistency-Localized Intrinsic Rewards

最后更新：2026-08-18

CLIR 是一个面向 ICLR 2027 的研究代码库：在自包含的 SWIFT-style hidden-state reward
backbone 上研究一致性学习、幻觉定位和 dual-prior localization，目标是在 Best-of-N
推理中获得更可靠的轨迹排序。仓库借鉴 SWIFT 的研究设定，但不调用 SWIFT 仓库代码。

新接手者先读 [docs/handoff.md](docs/handoff.md)；方法设计见
[docs/proposal.md](docs/proposal.md)。

## 当前结论边界

- Stage 1 correctness-only 小规模实验已完成。BoN@16 三 seed 均值为 strict SWIFT
  `88.28%`、encoded SWIFT `88.54%`、CLIR `89.32%`；encoded→CLIR 只有 `+0.78`
  个百分点，方向不稳定，不能宣称稳定增益。详见
  [docs/stage1_results.md](docs/stage1_results.md)。
- Stage 1B v1 是 pre-audit 诊断产物；v2 在 0 个完成 epoch 时中止；v3 尚未正式执行，
  已被第四轮审查后的 v4 取代。
- Stage 1B v4 outcome-only 3×3 已在提交 `b1c4fae` 上完整执行。冻结的历史 summary 记录为
  1/9 通过，但第四轮审查追加的 §13 已确认：训练健康门误用了最后一个 epoch 的 batch-average BCE，
  而不是 final checkpoint 在完整 train split 上的 BCE。对现有 checkpoint 的只读重算为 4/9
  （strict SWIFT 3/3、encoded SWIFT 1/3、CLIR 0/3）。因此原 1/9 分类已知不准确，v4 仍只允许作为
  diagnostic-only 证据，不得形成主效果结论。
- train-only semantics rewrite 工程 pilot v1 已在 clean commit `bc393cf` 上通过：4 条源
  trajectory 生成 12 个视图，12 个正 consistency pair 和 18 个负 pair，全部 tensor
  payload 校验通过。该 pilot 只证明数据/特征流水线闭环，不是真实 LLM semantic
  rewrite，`formal_mechanism_claim_allowed=false`。
- 真实 rewrite 的任务/feature model 保持 Phi-3.5-mini。最初的本地 Qwen-2.5-3B 选择冻结在
  [configs/semantic_rewrite_models_v1.json](configs/semantic_rewrite_models_v1.json)；按失败触发条件切换的
  3.23B Falcon3 fallback 冻结在
  [configs/semantic_rewrite_models_v2.json](configs/semantic_rewrite_models_v2.json)；模型规模诊断使用的
  Qwen-2.5-7B 冻结在
  [configs/semantic_rewrite_models_v3.json](configs/semantic_rewrite_models_v3.json)。三版都让 generator 与
  pilot verifier 共用 checkpoint，只允许作为 self-verification，不能支持独立 verifier 或正式机制结论。
- 新增 exact-token online hidden-state 路径。训练/打分可只持久化 prompt/question/response、精确
  `prompt_token_ids/output_token_ids` 和 provenance，再由冻结 Phi 在 batch 内重现全部 33 层；旧的
  precomputed feature 路径保持兼容。一个真实 Phi 样本与历史 trajectory/condition payload 均达到
  bitwise equality，证明接口等价；尚未做大规模吞吐比较。
- 第四轮审查 §13 的两个当前代码问题已经修复：健康门现在来自每个 epoch checkpoint 的完整 train
  split no-grad evaluation；shuffle 顺序由显式 `(seed, epoch)` sampler 决定，DataLoader worker RNG 与
  reward-model RNG 分离。覆盖冻结真实 loader 参数（4 workers、pin memory、persistent workers）的
  中断/续训回归达到模型、优化器和 metrics bit-exact。历史 v4 artifact 保持不变。
- 真实本地 4-source × 2-style gate 已执行到 v7。runtime-pinned Qwen v4 接受 1/8；结构化逐行生成的
  Qwen v5 为 0/8（6 条主要因数字 occurrence 漂移，2 条无法解析）；保持 v5 prompt/门禁不变、只换
  Falcon3-3B 的 v6 仍为 0/8（8 条均无法遵守标签协议）。只把 Qwen checkpoint 扩至 7.61B 的 v7
  接受 1/8：3 条标签输出不完整，另 4 条可解析输出改变数字 occurrence，其中 2 条还改变数字集合；
  0 次错误轨迹 repair。v7 复用的是已暴露 source，只是 model-scale diagnostic，机器报告明确
  `scale_authorization_eligible=false`、`scale_allowed=false`。失败 shard/audit 原样保留，32-source
  扩展和 rewrite 训练数据发布尚未授权。
- semantics augmentation 的主方向已由外部 Qwen rewrite 改为 **on-policy equivalence mining**：对同一
  原始 problem prompt 下由 Phi 正式 decoding policy 产生的原生 trajectories 做便宜预筛，再由独立
  relation verifier 判断 reasoning equivalence。Phi self-rewrite 保留为并行 Route B；Qwen rewrite
  降为后续 off-policy control。该变更避免把“Phi 作者 vs 外部作者”与目标表面变化完全混淆，但尚未
  产生训练效果证据。
- 独立 verifier 候选 `mistralai/Mistral-Small-24B-Instruct-2501@9527884...` 已在 64 条、双人工标注并
  裁决的模型盲 selection set 上完成 pipeline-pilot 评估：64/64 可解析，decision accuracy `87.5%`，
  auto-accept precision `87.5%`（28 TP / 4 FP）。当前只授权为 **Silver pilot gate**，用于宽门禁首轮；
  不得称为 Gold verifier 或论文级自动标注质量。
- `development_32` 的 256 条 Phi 原生轨迹经“同题、同 correctness、同归一化答案、文本不同”预筛，
  得到 727 个 correct 候选对和 8 个 incorrect 候选对。它们尚未经过 relation verifier，不等于
  reasoning-equivalent 数据。Pilot-0 因此先做 correct-only；incorrect 等价组后续定向补采样，不阻塞
  首轮训练。
- Route A Pilot-0 v1 已在高推理强度复审后**部分解冻并完成 v1a 修复训练门**：原生
  candidate/source/token/hash 血缘继续有效，旧 Mistral operational verifier 的 121 个 Silver accept、
  31 组/62 行发布及其训练资格仍撤销。31 对新 claim-level 双标的 status agreement 为 `30/31`
  （`96.77%`，Cohen's kappa `0.8394`）；唯一分歧经裁决为 reject，最终 `27 accept / 4 reject`。
  新 manifest 不复制 trajectory，只在原 4096 行 correctness-mixed train 中给 54 行添加 consistency
  metadata；query-disjoint 诊断验证集为 32 query/512 行。
- seed 42、1 epoch 的 matched pipeline pilot 已完成。consistency=1 相对 consistency=0 的完整 train
  checkpoint BCE 为 `0.2182 vs 0.2389`，validation BCE 为 `0.4600 vs 0.5074`；within-query pairwise
  accuracy 为 `0.5871 vs 0.5702`，BoN@16 同为 `0.875`。32 个验证 query 只有 10 个 mixed，不能据此
  宣称排序增益。表示诊断显示它主要把 different-semantic/same-style projected cosine 从 `0.9926`
  拉到 `0.7305`，而 same-semantic pair 从 `0.9989` 降到 `0.9827`；因此当前信号是防止全局表示塌缩，
  不是把已饱和的正对继续贴近。完整边界见
  [docs/on_policy_pilot0_reaudit_v1.md](docs/on_policy_pilot0_reaudit_v1.md) 和
  [configs/on_policy_pilot0_v1a/training_result_v1.json](configs/on_policy_pilot0_v1a/training_result_v1.json)。
- Stage 1B base train/validation 的辅助监督仍全部为 0；只有 v1a 派生 mixed train 的 54 行具有真实
  consistency metadata。hallucination、progress、dual-prior 和 reconstruction 覆盖仍为 0。仓库不会从
  correctness 伪造这些标签，也不会用全零向量冒充缺失监督。
- `pilot_test` 尚未生成、读取或用于任何选择。

## Stage 1B v4

机器协议为 [configs/stage1b_validation_v4.json](configs/stage1b_validation_v4.json)，人类可读
说明为 [docs/stage1b_v4_protocol.md](docs/stage1b_v4_protocol.md)。v4 保留 v3 的冻结数据、
模型矩阵和预算，只修正审查发现的实验有效性问题：

- 协议要求 final checkpoint 在完整 train split 上的 correctness BCE 比常数类别先验至少改善 1%；
  当前代码已按该口径发布 `clir-training-health-v3`，冻结 v4 artifact 仍保留历史误测结果；
- validation reward score population std 必须至少为 0.1；
- query 内 correct-vs-incorrect pairwise 排序准确率必须至少为 0.60，tie 计 0.5；
- final 失败不再封死健康 epoch snapshot；snapshot 只用于注明 post-hoc 的诊断；
- 失败 cell 只有在持久化证据与同一协议哈希、同一阈值匹配时才可省略；不完整汇总明确标为
  diagnostic-only，不能支持正式主结论；未知或未运行 cell 仍是错误；
- resume 已记录源 checkpoint 路径/SHA256、起始 epoch 和每段 resolved device，并拒绝跨设备续训；
  当前显式 epoch sampler 已在冻结 loader 参数下通过 bit-exact 中断恢复回归。

第四轮审查最初条目的逐项处理见
[docs/code_review_panzhixin_fourth_change_resolution.md](docs/code_review_panzhixin_fourth_change_resolution.md)；
随后追加的 §13 中，健康门和 persistent-worker resume 两项已由当前代码与回归覆盖；历史 v4 不回写。

正式运行产物位于 `run_artifacts/stage1b_v4`。冻结的 `summary.json` SHA256 为
`0f1e49ad72f71c4b2f51ad904f92f0cd02593e77d41d10caf34fe2e00a3b095c`；它按有缺陷的门禁实现记录
9 个已尝试 cell、1 个纳入 cell、8 个训练健康失败和 0 个未知 cell。不要把该 1/9 当作当前正确的
健康分类，也不要原地回写历史 artifact；修复后必须发布新协议/新输出。其 included encoded SWIFT
cell 在 k=`1/2/4/8/16` 的 BoN accuracy 为 `0.884/0.894/0.902/0.916/0.912`，但单 seed 数字只作
诊断，不可解释为稳定 baseline 结果，更不可用于判断 CLIR 机制。

rewrite pilot 的冻结结果见
[docs/semantic_rewrite_pilot_v1.md](docs/semantic_rewrite_pilot_v1.md)；它已完成
prepare/extract/audit。Qwen/Falcon v1-v7 是保留不覆盖的 off-policy 失败诊断，不再是当前主路线，也
不得通过降低旧保护阈值把失败输出纳入训练。当前设计与 verifier 契约见
[docs/semantic_rewrite_v8_reasoning_equivalent.md](docs/semantic_rewrite_v8_reasoning_equivalent.md)。Route A v1a
的盲第二标注、裁决、4096 行 mixed manifest 和 matched training 已完成；旧 62 行 all-positive
manifest 继续禁止训练。下一机制模块恢复为
[hallucination localization Pilot v1](docs/hallucination_localization_pilot_v1.md)，先构造 64-row
selection/annotation protocol，不把 consistency、localization 和 dual-prior 同时混训。Phi self-rewrite
Route B 保留为并行后续路线。NLL 仍只记录为 distribution diagnostic，首轮不设硬门；rewrite loss
代码本轮未改，旧 tiny sweep 的 0.5/1.0 不冻结为默认。

## 代码结构

| 路径 | 职责 |
|---|---|
| `src/consistency_localized_reward.py` | 三种 reward 变体、loss 与模型构造 |
| `src/clir_data.py` | JSONL/feature 数据集、严格 token 对齐与 collate |
| `src/clir_hidden_states.py` | exact-token online 全层 feature 重现与 provenance 门 |
| `src/clir_real_data.py` | 真实数据协议、checker、hash 与评估契约 |
| `src/clir_supervision.py` | 外部机制监督的身份绑定、校验与覆盖审计 |
| `src/clir_rewrite.py` | 可逆 rewrite pilot 的身份、变换、answer gate 与 pair 审计 |
| `src/clir_llm_rewrite.py` | 真实 LLM rewrite 的协议、确定性 fidelity gate、lineage 与 coverage 审计 |
| `src/clir_reasoning_rewrite.py` | reasoning-equivalence verifier 契约、fail-closed JSON 与分层 acceptance |
| `src/clir_on_policy_reaudit.py` | Route A v1a compact/expanded 定向、盲 item/lineage 与 claim-label validation |
| `train_clir.py` | 可恢复训练、健康门、checkpoint 与 run provenance |
| `score_clir.py` | reward 打分和 score-distribution 健康证据 |
| `evaluate_clir.py` | ordered-prefix BoN、baseline、bootstrap 与排序健康门 |
| `summarize_clir.py` | 多 seed 汇总、配对比较与显式失败矩阵 |
| `scripts/run_stage1b_validation.py` | Stage 1B v3/v4 的唯一正式 launcher |
| `scripts/run_semantic_rewrite_pilot.py` | train-only semantics rewrite 工程 pilot |
| `scripts/run_llm_semantic_rewrite.py` | 本地 LLM preflight/rewrite/audit 与 source-atomic resume |
| `tests/` | 模型、数据、恢复、门禁、评估和 launcher 回归测试 |

Best-of-N 选择只在 `evaluate_clir.py` 中进行；`score_clir.py` 只发布逐候选分数和 provenance，
避免维护两套选择逻辑。

## 关键数据契约

- JSONL 的规范事实是人类可审计文本、精确 `prompt_token_ids/output_token_ids` 和模型/tokenizer/
  protocol provenance；绝不能在训练时从 prompt/output 文本重新 tokenize 来决定 feature token 轴。
- hidden state 使用 embedding 加全部 32 个 Transformer block，共 33 层；真实宽度固定为
  `33 × 3072 = 101376`，不得静默删层或降采样。
- 真实样本必须满足
  `len(output_token_ids) == trajectory length == every token-label length`。
- condition 必须来自独立 prompt-only forward；precomputed 模式每个 query 保存一次并由候选共享，
  online 模式按 batch 对相同精确 prompt IDs 去重重现。不能从不同总序列长度的 forward 中切片替代。
- `query_id` 只定义 Best-of-N pool；`semantic_id` 只定义 rewrite/consistency group。
- correctness、checkpoint、manifest、协议和 feature payload 均由版本/hash 绑定；历史 artifact
  不原地回写。
- 归档 Stage 1B v1 scored 文件缺少当前 scoring provenance，不能直接交给当前 evaluator；若确实
  需要重新诊断，必须重新打分，不能放宽当前契约。

## Hidden-state 数据源

`train_clir.py` 和 `score_clir.py` 现在显式支持两种互斥模式：

- `--hidden_state_source precomputed`：读取历史 `.pt` payload。用于复现已冻结的 Stage 1/1B，避免
  每个 epoch 重跑 Phi。
- `--hidden_state_source online`：数据集只返回精确 token IDs 和监督；冻结 Phi 只加载一次，每条
  trajectory 做不带 padding 的 `prompt+output` teacher-forced forward，condition 做独立
  prompt-only forward并在当前 batch 内按精确 prompt IDs 去重。全宽 tensor 只在 batch 生命周期内
  存在，不写入磁盘。

这不是把 schema 简化成三个裸字符串。裸文本重新 tokenize 会破坏既有 token 标签和 hidden-state
对齐；prompt/question/response 用于人工审计，精确 token IDs 才是模型输入事实。Phi BF16 单条
feature 的理论字节数是 `T × 101376 × 2`；最大 `T=4096` 时约 0.77 GiB，condition 另计。现有
Stage 1B 镜像的 12,096 行 payload 为 725.8 GB，因此新 rewrite 数据默认应优先评估 online 模式。

代价是每个 epoch 都重跑冻结 Phi。数据源模式属于实验协议的一部分，同一正式比较中的变体和 seed
必须一致，不能让一个变体读预计算 feature、另一个变体在线抽取。真实单样本门只验证了 bitwise
等价和工程可行性，不是规模化吞吐结论。

## 本地验证

使用项目指定的 SWIFT 环境，不要调用系统 `pytest`：

```bash
P=/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python
"$P" -m pytest -q
"$P" -m pip check
```

快速查看各入口：

```bash
P=/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python
"$P" train_clir.py --help
"$P" score_clir.py --help
"$P" evaluate_clir.py --help
"$P" summarize_clir.py --help
```

toy 数据只验证工程闭环，不能作为研究证据。完整命令见
[docs/runbook_zh.md](docs/runbook_zh.md)。

## v4 只读预检与命令审计

下面两条命令不启动训练：

```bash
P=/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python
"$P" scripts/run_stage1b_validation.py --stage preflight
"$P" scripts/run_stage1b_validation.py --stage train --device cuda
```

v4 已完成一次冻结执行；不要在同一协议下改阈值、补跑失败 cell 或覆盖产物。只读 preflight 和
命令审计仍可使用；任何新的正式比较必须发布新协议和新输出目录。

## 文档索引

- [docs/handoff.md](docs/handoff.md)：当前事实、约束和下一步选择
- [docs/proposal.md](docs/proposal.md)：CLIR 方法设计
- [docs/runbook_zh.md](docs/runbook_zh.md)：最短可执行手册
- [docs/clir_supervision_protocol.md](docs/clir_supervision_protocol.md)：真实机制监督接入契约
- [docs/semantic_rewrite_pilot_v1.md](docs/semantic_rewrite_pilot_v1.md)：当前 rewrite 工程门
- [docs/semantic_rewrite_v8_reasoning_equivalent.md](docs/semantic_rewrite_v8_reasoning_equivalent.md)：
  on-policy mining / Phi self-rewrite 双路线与 reasoning-equivalence 契约
- [configs/verifier_selection_v1](configs/verifier_selection_v1)：64-pair 双标注、裁决后的 verifier
  selection package
- [configs/semantic_rewrite_models_v1.json](configs/semantic_rewrite_models_v1.json)、
  [configs/semantic_rewrite_models_v2.json](configs/semantic_rewrite_models_v2.json) 与
  [configs/semantic_rewrite_models_v3.json](configs/semantic_rewrite_models_v3.json)：Qwen-3B、Falcon3-3B
  和 Qwen-7B 的 generator/verifier 冻结选型
- [configs/semantic_rewrite_llm_v7.json](configs/semantic_rewrite_llm_v7.json)：当前 Qwen-7B、失败的
  model-scale diagnostic；v1–v7 均为不可覆盖证据，尚未授权扩规模
- [docs/stage1b_v4_protocol.md](docs/stage1b_v4_protocol.md)：当前 Stage 1B 协议
- `docs/stage1b*_protocol.md` 与旧 config：冻结历史，不得原地修改
- `docs/code_review_panzhixin_*_change.md`：审查证据，不是当前运行说明

## 维护原则

代码行为、机器协议和测试是运行事实来源；README/handoff 只保留当前状态，不复制完整历史。
改变数据 schema、损失、门禁、指标或结论边界时，必须同步更新对应测试和版本化协议。正式协议一旦
冻结，只能由新版本取代，不能原地改变阈值或实验语义。
