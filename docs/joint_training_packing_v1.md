# JPH Supervision-Aware Packing v1

最后更新：2026-08-20

状态：`completed_seed42_packing_schedule_not_supported`

## 1. 为什么这样改

`main` 最初的 `SemanticGroupBatchSampler` 已经体现了一个正确原则：如果一个 auxiliary objective 依赖特定
样本共同出现，就由 sampler 显式保证，而不是等随机 shuffle 碰撞。Consistency 因而把同一
`semantic_id` 的两条 view 放进同一 batch。

当前 JPH 的 48 条 mechanism rows 没有 `semantic_id`，所以在 3968-row stream 中几乎总是作为 singleton
散落：五个 epoch 分别有 `48/47/48/45/45` 个 H/prior-active optimizer steps。Standalone S1 则天然是
48 rows、batch 4、每 epoch 12 steps。新 cell 把 main 的“结构感知组批”原则推广到 H/prior，但使用与
语义完全独立的 packing sidecar，绝不伪造 `semantic_id`。

main 默认的 key/complete epoch-level alternate phase 没有进入本轮。采用它会同时改变 direct prior 的
训练 phase 和双向 mutual distillation，不再是 packing 对照；本轮继续固定 `prior_phase=joint`，完整保留
direct key/complete、双向 stop-gradient mutual `.25` 和 shared-gradient gate `10`。

## 2. 唯一新增机制

`mechanism_packing_pool_v1.jsonl` 只含两列：trajectory `id` 和 `packing_pool_id`。它不进入 dataset item、
collate 或模型，只告诉 sampler 哪 48 行属于同一个 exclusive packing pool。每个 epoch：

- 48 行重新 shuffle；
- 每 4 行组成一个纯 mechanism batch，共 12 个；
- 其余 3920 行继续走原 semantic-group packing；
- 全部 batch 再进入同一个 single stream；
- 3968 行各出现一次，总 batch 数仍是 992。

五个 epoch 的静态 materialization audit 均确认：12 个 mechanism batch 的 mechanism row count 全为 4，
没有 1–3 行混合 batch；27 个 positive consistency pairs、26 个 same-style negative pairs保持不变；没有
修改 semantic/style metadata 或任何 supervision target。

## 3. 明确不变与发生变化的量

不变：seed 42 初始化、模型、3968-row manifest、5 epochs、batch size 4、总 optimizer steps 4960、每行
exposure、final checkpoint、全部 loss 公式与配置权重、原始 mutual/gate、C 关闭、tail/progress/
reconstruction 关闭。

发生变化：H/prior-active steps 从五个 epoch 合计 233 降到 60；四条 mechanism row 的 token/prior units
现在在一个 batch 内共同 reduction；它们在 epoch stream 中的位置也变化。因此本实验必须称为
`packing/schedule + effective auxiliary budget` compound diagnostic，不能称为 gradient-budget-matched 的
纯 packing 因果实验。若结果通过，下一轮仍须区分 grouping 与有效预算。

## 4. 冻结采用门

相对冻结 JPH/JP：

- key AP 相对 JPH 至少恢复 `.05`，且相对 JP 回退不超过 `.05`；
- H span/claim AP 严格超过位置基线 `.3932807/.4219877`；
- BoN@16 相对 JP 回退不超过 `.02`。

三类门全部通过才记为 seed-42 support；只通过 key 或 H 的完整门组且 ranking 守住，记为 partial；否则
记为不支持。无论结果如何，本格不自动扩 seed、不自动调 loss、不切 multistream、不做 gradient surgery，
也不读取 `pilot_test/final_test`。

## 5. Artifact

- 冻结协议：`configs/joint_training_packing_v1/training_protocol_v1.json`
- packing sidecar：`configs/joint_training_packing_v1/mechanism_packing_pool_v1.jsonl`
- 静态审计：`configs/joint_training_packing_v1/packing_report_v1.json`
- 执行入口：`scripts/run_joint_training_pilot_v1.py`
- 总结入口：`scripts/summarize_joint_training_packing_v1.py`
- 冻结机器结果：`configs/joint_training_packing_v1/training_result_v1.json`
- seed-42 cell result：
  `run_artifacts/joint_training_packing_v1/seed42_v1/seed_42/jph_supervision_packed/cell_result.json`

执行必须来自 clean committed worktree：

```bash
/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python \
  scripts/run_joint_training_pilot_v1.py \
  --protocol configs/joint_training_packing_v1/training_protocol_v1.json \
  --cell jph_supervision_packed \
  --seed 42 \
  --execute
```

## 6. seed-42 结果

训练从 clean commit `950f5c4` 完成 5 epochs，无 OOM/NaN/runtime failure；未访问
`pilot_test/final_test`。工程上完全达到预期：每个 epoch 都是 992 个 optimizer steps，H 与
prior 都只在 12 个纯 mechanism batch 上 active，而每条数据的 exposure 不变。

| 指标 | JP | JPH | JPH packed | 冻结判定 |
|---|---:|---:|---:|---|
| BoN@16 | `.918` | `.920` | `.924` | 通过 ranking guard；packed−JP `+.006` |
| H span-token AP | `.192` | `.319` | `.206` | 失败；低于 `.393` position baseline |
| H claim-mean AP | `.172` | `.338` | `.448` | 通过；高于 `.422` position baseline |
| key unit AP | `.432` | `.314` | `.096` | 失败；相对 JPH/JP 下降 `.218/.336` |
| complete unit AP | `.946` | `.928` | `.429` | 非主门，但同样显示 prior 拟合大幅损伤 |

BoN@16 的 packed−JPH 配对 bootstrap 差为 `+.004 [-.012,+.020]`，packed−JP 为
`+.006 [-.012,+.024]`；只能说 ranking 没有崩坏，不能宣称改善。冻结分类为
`packing_schedule_not_supported_at_frozen_gates_seed42`，不授权扩 seeds。

## 7. 从 main 得到的真正结论

main 的 `SemanticGroupBatchSampler` 解决的是“relational loss 缺少同批 pair”：consistency 必须同时
看到两个 view 才能定义 positive/negative pair。H sparse BCE 和 dual-prior direct/mutual/gate 的基本
监督单元却都是 per-row/per-token；把四条 mechanism row 同批并不会新增一条跨样本约束。

在当前 loss 以 active mask 均值规约的实现下，原 JPH 约 48 个单条-active batch 产生约 48 次
auxiliary update；packed 变成 12 个四条均值 batch，等价于将每个 epoch 的 auxiliary optimizer
opportunities 压缩到约四分之一。prior 两张 map 同时大幅恶化，与这一机制一致。因此不应把
本结果解释为“结构组批尚需调一个阈值”；它更直接地说明，main 的 pair-aware 原则不能原样
套到非 relational 的 H/prior loss 上。

packing 能力保留为显式 optional 诊断工具，不进入当前保留训练方案。当前方案恢复 JPH
的普通 single-stream sampler，不通过事后乘 4、oversampling 或重复 row 挽救本格；这些都会引入
新的 effective-weight/exposure 变量，需要独立冻结协议。

## 8. 下一个可讨论的最小修复

已有 no-update audit 中，H↔prior-total 在整个 shared encoder 上没有稳定冲突，但在
condition-attention/fusion 子空间的两个模型状态都低于 `-.05`。因此下一个候选不是继续打包、
改原 mutual/gate，也不是全模型 PCGrad，而是一个窄的 JPH condition-branch gradient-routing
control：

- forward 仍使用同一 problem condition；
- H BCE 仍训练 hallucination head、trajectory encoder 和共享 token representation；
- 只阻止 H BCE 更新 `condition_query/key/value` 和 `condition_fusion`；
- final 与项目原 prior 对这些参数的梯度完整保留，direct targets、双向 stop-gradient
  mutual `.25` 和 shared-gradient gate `10` 均不变；
- 恢复原 JPH 普通 sampler 和约 45–48 个 active steps，先做 no-update routing audit，再决定是否
  训练单个 seed-42 cell。

该 no-update routing audit 后续已在全 48 rows 上通过；结果见
`docs/joint_condition_routing_v1.md`。它只授权讨论单个 seed-42 训练格，尚未成为保留方法，也不构成
效果证据。
