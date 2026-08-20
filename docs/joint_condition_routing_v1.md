# JPH Hallucination-to-Condition Gradient Routing v1

最后更新：2026-08-20

状态：`completed_seed42_condition_route_not_supported`

## 1. 目的与边界

JPH 的无更新梯度审计不支持全 shared encoder 的 blanket gradient surgery，但 H↔prior-total
在 `condition_query/key/value + condition_fusion + feature_norm` 子空间的初始化/JP 状态都出现负向信号。
后续 packing 负结果又说明，对 per-row H/prior 改组批只会压缩 effective auxiliary updates，不是合适
修复。

因此本格只审计一个窄的梯度路由：

- H 的 conditioned forward 值必须完全不变；
- H BCE 不得更新 `condition_query/key/value` 和 `condition_fusion`；
- H BCE 对共享 input encoder、`feature_norm` 和 hallucination head 的梯度必须保留；
- correctness、key/complete direct target、双向 stop-gradient mutual `.25` 和 shared-gradient gate `10`
  的 loss 与梯度必须与原 JPH 一致；
- 不修改 sampler、row exposure、loss weight 或 prior 架构；不训练 checkpoint。

证据等级是 `small-scale real` engineering audit，不是效果证据。

## 2. 实现方式

这里没有对整个 conditioned feature 直接 `.detach()`。那样虽然会阻止 condition 参数梯度，但也会误伤
H 对 input encoder 和共享 representation 的梯度。

当 `hallucination_condition_stop_gradient=true` 且 batch 有 hallucination supervision 时，模型额外计算一条
数值相同的 H feature path：

- query/key/value 线性层和 `condition_fusion` 使用当前参数的 detached weight/bias 做计算；
- detached weight 仍允许梯度对输入求导，所以 trajectory/condition input encoder 的 H 梯度不变；
- 外层 `feature_norm` 不 detach，H 仍可更新它；
- 其他 head 继续使用原 token feature graph，所以 correctness 和 prior 的路径不变；
- 无 H supervision 的训练 batch 不构建该额外 graph；no-grad 打分也不增加额外计算。

默认值为 `false`，所有历史 protocol/checkpoint 保持原行为。

## 3. 冻结审计覆盖

审计从 clean commit `a5bf692` 运行，覆盖：

- 冻结 mechanism-train 的全部 48 rows，按 manifest order 组成 12 个 4-row controlled batches；
- seed-42 deterministic initialization 和冻结 JP epoch-5 checkpoint 两个状态；
- 每个 batch 都同时计算 baseline/routed 的 forward、loss 和梯度；
- forward 比较 `scores/hallucination_logits/token_features/key_prior/complete_prior/gates`；
- gradient 比较 H 以及 final/key/complete/mutual/gate 六个目标；
- 参数 checksum 前后一致、`.grad` buffer 不存在，没有 optimizer step。

预注册的 gradient difference 容差是 max-abs/relative-L2 均 `1e-5`，forward/loss 和 blocked-gradient
容差为严格 `0`。

## 4. 结果

24/24 controlled batches 全部通过，而且观测差异比容差更强：

| 检查 | Initialization | JP epoch 5 |
|---|---:|---:|
| max forward absolute difference | `0` | `0` |
| max objective-loss absolute difference | `0` | `0` |
| baseline H→blocked minimum L2 | `.804` | `.367` |
| routed H→blocked maximum L2 | `0` | `0` |
| H nonblocked max relative/max-abs gradient difference | `0 / 0` | `0 / 0` |
| final/direct/mutual/gate max gradient difference | `0` | `0` |
| parameter update / `.grad` buffer | `no / absent` | `no / absent` |

H 仍保留的三条路由在每个 batch 都非零。其中 minimum L2：

| H route | Initialization | JP epoch 5 |
|---|---:|---:|
| input encoder | `5.739` | `.865` |
| `feature_norm` | `.0488` | `.1218` |
| hallucination head | `1.772` | `4.713` |

原 dual-prior 路由也完整保留：

- mutual 对 key 和 complete head 的梯度在每个 batch 都非零；
- gate 对 reward head 的梯度在每个 batch 都非零；
- gate 对 detached fused-prior target 背后 key/complete head 的梯度始终为 0。

因此工程结论可以锁定：该实现只截断 H BCE 对四个目标 condition module 的参数梯度，
没有改 forward 函数值、其他 H 梯度或项目原 correctness/dual-prior 路由。这仍不说明它能改善
localization 或 ranking。

## 5. 已获批准并冻结的训练格

用户已批准唯一的 seed-42 `JPH + H-condition-stopgrad` 单格，协议冻结为：

- control 为冻结 JPH，candidate 唯一变量为 `hallucination_condition_stop_gradient=true`；
- 恢复原 ordinary `SemanticGroupBatchSampler`，禁用 packing；
- 同一 3968-row manifest、seed 42、5 epochs、batch 4、LR `1e-4`、BF16、final checkpoint；
- final/H/prior 外层权重仍 `1/1/1`，direct `1/1`、mutual `.25`、gate `10`；C/tail/progress/
  reconstruction 仍关闭；
- 效果门同时要求：H span/claim AP 严格超过冻结位置基线；key AP 相比 JPH 至少恢复 `.05` 且距
  JP 不超过 `.05`；complete AP 相比 JPH 最多下降 `.05`；BoN@16 相比 JP 最多下降 `.02`；
- 不自动扩 seeds、调 loss、做 gradient surgery 或改 sampler/stream。

这些门在训练前写入 `training_protocol_v1.json`。本格仍只是一个 seed 的 small-scale real diagnostic；
即使全门通过，也只授权讨论后续复验，不构成正式效果结论。

## 6. 真实训练结果

训练从 clean commit `df33e7b` 完成 seed 42 × 5 epochs，5 个 epoch 都是 3968 rows / 992
batches，H/prior active batches 精确为 `[48,47,48,45,45]`，与原 JPH ordinary sampler 日程一致。
全部 loss 有限，final checkpoint 配置与冻结协议一致。

| 指标 | JP | 原 JPH | Stopgrad candidate | Candidate − JPH | 冻结门 |
|---|---:|---:|---:|---:|---|
| BoN@16 | `.918` | `.920` | `.912` | `-.008` | 通过：相对 JP `-.006 ≥ -.02` |
| H span token AP | `.192` | `.319` | `.247` | `-.0717` | 失败：未超过 `.3933` |
| H claim mean AP | `.172` | `.338` | `.235` | `-.1033` | 失败：未超过 `.4220` |
| key unit AP | `.432` | `.314` | `.337` | `+.0232` | 失败：恢复不足 `.05`，且距 JP `.0952` |
| complete unit AP | `.946` | `.928` | `.929` | `+.0011` | 通过：未比 JPH 下降 `.05` |

BoN@16 的 paired 500-query bootstrap 中，candidate−JP 为 `-.006`，95% CI
`[-.022,+.008]`；candidate−JPH 为 `-.008`，95% CI `[-.024,+.008]`。这没有建立稳定 ranking
差异，但点估计通过预注册保护门。

最终分类固定为 `condition_route_not_supported_at_frozen_gates_seed42`。工程审计证明路由实现准确，
但效果结果说明局部负梯度 cosine 不能直接推出“切断这条梯度会更好”：阻断 H 更新 condition parameters
只带来小幅 key 回升，同时明显伤害 H localization。该开关不进入保留训练方案，不扩 seeds、不调权重；
原 direct key/complete、双向 mutual `.25` 和 shared-gradient gate `10` 本轮从未被修改，后续仍以原实现为准。

## 7. Artifact

- 冻结协议：`configs/joint_condition_routing_v1/audit_protocol_v1.json`
- 协议 SHA256：`18c017833c4241d896ccca6bf86fc322632a1c8c174672e2c23416a479f2a89a`
- 审计结果：`configs/joint_condition_routing_v1/audit_result_v1.json`
- 结果 SHA256：`661831e72919a93c81e9cb3007976360978227eb56e90283bf07ceb8be987234`
- 审计代码：`scripts/audit_joint_condition_routing_v1.py`
- clean 代码 commit：`a5bf692b12590bfc439127dd527dc8c5da5901c2`
- 冻结训练协议：`configs/joint_condition_routing_v1/training_protocol_v1.json`
- 训练协议 SHA256：`5532bb787d20eb2b049158fdb7c131a1394e094d574aba53a0bce9929d5c528b`
- 训练运行器：`scripts/run_joint_training_pilot_v1.py`
- 结果汇总器：`scripts/summarize_joint_condition_routing_v1.py`
- 训练结果：`configs/joint_condition_routing_v1/training_result_v1.json`
- 训练结果 SHA256：`7ad80a766d872f95f2a48c8ce81daddbabb1d5883ed5fb6f103e7373da157163`
- final checkpoint SHA256：`9b98ccd64d7c1fd92faa94dbb4de9c6461718ec43dcd7685ca448d29c4dfd1ef`

`pilot_test/final_test` 均未访问。
