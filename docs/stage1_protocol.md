# CLIR Stage 1 small-scale correctness-only 协议

版本：`clir-stage1-small-scale-v1`

证据等级：`small-scale real`

本阶段的目标是在加入 rewrite、hallucination onset 或 dual-prior target 之前，判断真实
correctness-only reward 是否能在 held-out query 上提供可解释的 Best-of-N 选择信号。机器
配置为 `configs/stage1_small_scale_v1.json`；任何规模、seed、epoch、模型或评估口径变更都
必须产生新版本，不能覆盖本文件。

配置中的 protocol 与 split manifest 哈希均采用项目统一的 `canonical_json_sha256` 口径，
不使用受空白和换行影响的原始文件字节哈希。

## 数据与停止规则

- train：冻结 `train_primary` membership 的前 512 query，每题 8 candidates。
- validation：冻结 `validation` membership 的前 128 query，每题 16 candidates。
- 同一 query 的所有候选只进入一个 split；development-32 不作为 validation。
- generation 完成后、extraction 开始前检查标签覆盖，不查看 reward-model 结果：train 至少
  128 correct 和 128 incorrect candidates，validation 至少 32 个 mixed-label query。
- 若覆盖不足，停止抽取并冻结一个只扩大 ordered prefix 的 v2 配置；禁止根据题目或答案
  cherry-pick query。
- 预计全层 BF16 特征约 354GB，硬停止上限 500GiB。超过上限时停止并审计，不静默删层。

## 模型与训练

在完全相同的 candidates、features、labels 和 split 上运行：

1. `strict_swift`：raw 101376 维 `Linear(D,2)`；
2. `encoded_swift`：共享 33-layer encoder + SWIFT head；
3. `clir`：相同 encoder + correctness-only CLIR backbone。

训练 seeds 固定为 42/43/44；5 epochs、batch size 2、AdamW、LR 1e-4、无 class
reweighting、无 early stopping，统一使用最终 epoch。当前数据没有任何 CLIR auxiliary target，
对应 loss 与 applicable count 必须为 0。

## 评估和归因

validation 报告 BoN@1/2/4/8/16、random expected/seeded、oracle、query bootstrap 95% CI，
并跨三个 seed 报告均值和标准差。

- strict SWIFT → encoded SWIFT：只归因于 full-layer encoder；
- encoded SWIFT → CLIR：才是 correctness-only CLIR backbone 的增量。

在 validation 配置和结论锁定前，不生成、不检查 `pilot_test`。本阶段若没有稳定超过 random
或其他预先同意的简单基线，只能报告负结果/诊断，不能继续把辅助模块的收益当作已建立。
