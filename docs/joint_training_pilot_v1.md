# Joint Training Pilot v1

最后更新：2026-08-20

## 1. 结论

第一轮联合训练的工程闭环已经成功，但完整联合组没有通过冻结的 seed-42 扩展门槛，因此不扩跑 seeds 43/44，
也不能宣称 CLIR 联合训练有效。

这里必须区分两件事：

- **接入成功**：J0、JP、JALL 都完成 5 epochs；无 OOM、NaN 或运行时失败；每个 epoch 的 row、pair、token
  监督计数与协议完全一致；训练前真实 feature 梯度路由审计通过。
- **效果未通过**：JALL 保住了冻结的 BoN@16 回退保护线，也明显学到了 consistency 训练关系，但 sparse
  hallucination 的两个 AP 都低于位置基线，且 key prior AP 相对 JP 回退 `.118`，超过允许的 `.05`。

本轮证据等级仍是 `small-scale real integration pilot`。它只回答当前 single-stream 方案在 seed 42 是否值得扩跑，
不回答正式机制有效性。

## 2. 冻结设计

统一训练集是 3968 rows / 496 queries：

| 行类型 | 数量 | 监督 |
|---|---:|---|
| correctness-only | 3866 | outcome BCE |
| correctness + consistency | 54 | outcome BCE；组成 27 个 on-policy equivalence pairs |
| correctness + mechanism | 48 | outcome BCE、S1 sparse token BCE、原始 dual prior |

Consistency 行与 mechanism 行零重叠。另有 query-disjoint mechanism dev 16 rows，以及固定且与训练、mechanism
dev 都 query-disjoint 的 ranking validation 500 queries × 16 candidates。

三格使用相同初始化、seed 42、数据、semantic-group batch 顺序、batch 4、5 epochs、BF16、LR `1e-4`：

| Cell | 启用的外层目标 |
|---|---|
| J0 | correctness |
| JP | correctness + 项目原始 dual prior |
| JALL | correctness + Route-A consistency + S1 sparse hallucination + 项目原始 dual prior |

原始 dual prior 没有被替换：仍然是 direct key/complete BCE、双向 stop-gradient mutual MSE（`.25`），以及
shared-gradient reward gate 对 detached `0.5/0.5` fused prior 的 MSE（`10`）。所有 tail、MIL、token reward、
progress 和 reconstruction 都显式为 0。

## 3. 结果

| Cell | BoN@16 | Pairwise acc. | H span AP | H claim AP | Key AP | Complete AP | Consistency gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| J0 | `.920` | `.667` | `.169` | `.421` | `.098` | `.277` | `.00001` |
| JP | `.918` | `.700` | `.192` | `.172` | `.432` | `.946` | `.023` |
| JALL | `.912` | `.681` | `.272` | `.289` | `.314` | `.931` | `.789` |

冻结 gate 的判断是：

- ranking 保护线通过：JALL−J0 为 `-.008`，query bootstrap 95% CI `[-.026,.010]`；JALL−JP 为
  `-.006`，CI `[-.026,.012]`，都没有超过允许的 `.02` point regression；
- consistency 训练关系 geometry 通过：JALL gap `.789 > 0`，但这是训练 relation，不能当 held-out 语义证据；
- complete prior 通过：`.931` 高于位置基线，且相对 JP 只下降 `.015`；
- key prior 的绝对位置门通过，但保护门失败：JALL `.314` 相对 JP `.432` 下降 `.118 > .05`；
- sparse hallucination 失败：span AP `.272 < .393` position baseline，claim AP `.289 < .422` position
  baseline。

所以机器状态是 `completed_seed42_expansion_gates_failed`，而不是训练失败，也不是效果通过。

## 4. 失败诊断

### 4.1 Sparse labels 不是唯一解释

Standalone S1 使用相同的 48/16 mechanism split、相同 5 epochs 和 unweighted explicit token BCE，最终
span AP `.416`、claim AP `.464`，都超过位置基线。它的 H BCE 五个 epoch 为
`.501/.426/.389/.476/.392`；JALL 的 active H BCE 为 `.656/.625/.541/.588/.536`。

因此，当前证据支持“同一 sparse 监督在联合环境里没有保持 standalone 的拟合和泛化表现”，不支持直接说
“标签本身无效”。

### 4.2 原始 prior 确实在训练，但 JALL 学得更差

JP 的 active key BCE 从 `.290` 降到 `.229`，complete BCE 从 `.559` 降到 `.331`；JALL 分别从
`.294` 降到 `.249`、从 `.590` 降到 `.380`。这说明 direct、mutual 和 gate 路径没有被漏掉，但增加 H+C 后，
prior 的训练拟合及 held-out key AP 都变差。

### 4.3 Consistency 产生了很强的训练内几何变化

JALL 的 active negative consistency loss 从 `.785` 降到 `.448`，最终 relation cosine gap 为 `.789`。这说明
该目标不是“没起作用”，但五个 epoch 后的变化比旧 1-epoch pilot 强很多；又因为没有 held-out equivalence
relation，目前只能将它视为一个可能的共享表示干扰源，不能视为语义泛化收益。

### 4.4 当前矩阵无法判定是谁造成的

相对 JP，JALL 同时增加了 H 和 consistency。因而以下三种机制与现有结果都相容：

1. consistency 单独足以伤害 key/H 表示；
2. sparse H 单独足以伤害 key prior；
3. 两者单独都可共存，但同时启用才发生交互冲突。

此外，standalone H 的 48 rows 每 epoch 组成 12 个普通 batch；联合 single-stream 中，这些 singleton semantic
groups 每 epoch 分散在 45–48 个 active batches，并穿插在 3968-row correctness stream 中。每行仍只出现一次，
但 auxiliary batch packing、optimizer-step 上下文以及相邻 correctness-only 更新数与 standalone 不同。这是一个
真实的 mixing confound，尚不能和梯度冲突区分。

## 5. 下一步：只补两个 drop-one cell

在任何调权重、改 sampler 或切 multistream 之前，最小可识别实验是：

| 新 cell | Prior | H | Consistency | 回答的问题 |
|---|---:|---:|---:|---|
| JPH | 1 | 1 | 0 | H 与原始 prior 单独能否共存 |
| JPC | 1 | 0 | 1 | consistency 与原始 prior 单独能否共存 |

JP 和 JALL 直接作为已冻结对照。新两格继续使用 seed 42、相同初始化、manifest、batch order、single stream、
batch 4、5 epochs、所有原 loss 权重与 final-epoch selection；不自动调 loss，不更换原始 mutual/gate，不访问
pilot/final test。

解释规则预先写死：

- JPC 复现 key drop：consistency 在 prior 条件下已足以造成该问题；
- JPH 复现 key drop：H 在 prior 条件下已足以造成该问题；
- 两者单独都不复现、JALL 才复现：是 H×consistency interaction；
- JPH 的 H AP 恢复：consistency 是 JALL localization 失败的重要嫌疑；
- JPH 的 H AP 仍失败：下一步才比较当前 packing 与 supervision-aware packing，并保持每行每 epoch 一次，
  不能直接跳到扫 loss weight。

这一 drop-one 只是失败归因，不是新方法效果实验。用户已批准执行，冻结协议为
`configs/joint_training_drop_one_v1/training_protocol_v1.json`；它不修改或覆盖本轮任何 artifact。

## 6. 可复核 artifact

- 冻结协议：`configs/joint_training_pilot_v1/training_protocol_v1.json`
- 机器结果：`configs/joint_training_pilot_v1/training_result_v1.json`
- 失败诊断：`configs/joint_training_pilot_v1/failure_diagnostic_v1.json`
- 梯度审计：`run_artifacts/joint_training_pilot_v1/audits/gradient_routing_v1.json`
- 三格训练目录：`run_artifacts/joint_training_pilot_v1/seed42_v1/seed_42/`

当前没有读取 `pilot_test` 或 `final_test`，也没有基于结果修改冻结 checkpoint 或历史门槛。
