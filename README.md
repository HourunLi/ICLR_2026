# CLIR: Consistency-Localized Intrinsic Rewards

最后更新：2026-08-15

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
- Stage 1B v4 已准备但尚未运行。它仍只有 correctness 标签，因此只是
  outcome-only 容量/优化对照，不能验证或否证 CLIR 的三个机制。
- 当前冻结 train/validation 中，consistency、hallucination、progress、dual-prior 和
  reconstruction 的真实监督覆盖全部为 0。仓库不会从 correctness 伪造这些标签，也不会用
  全零向量冒充缺失监督。
- `pilot_test` 尚未生成、读取或用于任何选择。

## Stage 1B v4

机器协议为 [configs/stage1b_validation_v4.json](configs/stage1b_validation_v4.json)，人类可读
说明为 [docs/stage1b_v4_protocol.md](docs/stage1b_v4_protocol.md)。v4 保留 v3 的冻结数据、
模型矩阵和预算，只修正审查发现的实验有效性问题：

- train BCE 必须比常数类别先验至少改善 1%，方向错误或改善不足都失败；
- validation reward score population std 必须至少为 0.1；
- query 内 correct-vs-incorrect pairwise 排序准确率必须至少为 0.60，tie 计 0.5；
- final 失败不再封死健康 epoch snapshot；snapshot 只用于注明 post-hoc 的诊断；
- 失败 cell 只有在持久化证据与同一协议哈希、同一阈值匹配时才可省略；不完整汇总明确标为
  diagnostic-only，不能支持正式主结论；未知或未运行 cell 仍是错误；
- resume 记录源 checkpoint 路径/SHA256、起始 epoch 和每段 resolved device，并拒绝跨设备续训。

第四轮审查的逐项处理见
[docs/code_review_panzhixin_fourth_change_resolution.md](docs/code_review_panzhixin_fourth_change_resolution.md)。

## 代码结构

| 路径 | 职责 |
|---|---|
| `src/consistency_localized_reward.py` | 三种 reward 变体、loss 与模型构造 |
| `src/clir_data.py` | JSONL/feature 数据集、严格 token 对齐与 collate |
| `src/clir_real_data.py` | 真实数据协议、checker、hash 与评估契约 |
| `src/clir_supervision.py` | 外部机制监督的身份绑定、校验与覆盖审计 |
| `train_clir.py` | 可恢复训练、健康门、checkpoint 与 run provenance |
| `score_clir.py` | reward 打分和 score-distribution 健康证据 |
| `evaluate_clir.py` | ordered-prefix BoN、baseline、bootstrap 与排序健康门 |
| `summarize_clir.py` | 多 seed 汇总、配对比较与显式失败矩阵 |
| `scripts/run_stage1b_validation.py` | Stage 1B v3/v4 的唯一正式 launcher |
| `tests/` | 模型、数据、恢复、门禁、评估和 launcher 回归测试 |

Best-of-N 选择只在 `evaluate_clir.py` 中进行；`score_clir.py` 只发布逐候选分数和 provenance，
避免维护两套选择逻辑。

## 关键数据契约

- hidden state 使用 embedding 加全部 32 个 Transformer block，共 33 层；真实宽度固定为
  `33 × 3072 = 101376`，不得静默删层或降采样。
- 真实样本必须满足
  `len(output_token_ids) == trajectory length == every token-label length`。
- condition 是每个 query 一次独立 prompt-only forward，候选共享；不能从不同总序列长度的
  forward 中切片替代。
- `query_id` 只定义 Best-of-N pool；`semantic_id` 只定义 rewrite/consistency group。
- correctness、checkpoint、manifest、协议和 feature payload 均由版本/hash 绑定；历史 artifact
  不原地回写。
- 归档 Stage 1B v1 scored 文件缺少当前 scoring provenance，不能直接交给当前 evaluator；若确实
  需要重新诊断，必须重新打分，不能放宽当前契约。

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

正式执行必须满足 clean commit、完整矩阵原子 preflight 和用户明确授权。未经授权不要运行
`--stage preflight --execute`，也不要启动任一 GPU cell。

## 文档索引

- [docs/handoff.md](docs/handoff.md)：当前事实、约束和下一步选择
- [docs/proposal.md](docs/proposal.md)：CLIR 方法设计
- [docs/runbook_zh.md](docs/runbook_zh.md)：最短可执行手册
- [docs/clir_supervision_protocol.md](docs/clir_supervision_protocol.md)：真实机制监督接入契约
- [docs/stage1b_v4_protocol.md](docs/stage1b_v4_protocol.md)：当前 Stage 1B 协议
- `docs/stage1b*_protocol.md` 与旧 config：冻结历史，不得原地修改
- `docs/code_review_panzhixin_*_change.md`：审查证据，不是当前运行说明

## 维护原则

代码行为、机器协议和测试是运行事实来源；README/handoff 只保留当前状态，不复制完整历史。
改变数据 schema、损失、门禁、指标或结论边界时，必须同步更新对应测试和版本化协议。正式协议一旦
冻结，只能由新版本取代，不能原地改变阈值或实验语义。
