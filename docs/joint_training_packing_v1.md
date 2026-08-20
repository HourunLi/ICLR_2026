# JPH Supervision-Aware Packing v1

最后更新：2026-08-20

状态：`frozen_before_training`

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

执行必须来自 clean committed worktree：

```bash
/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python \
  scripts/run_joint_training_pilot_v1.py \
  --protocol configs/joint_training_packing_v1/training_protocol_v1.json \
  --cell jph_supervision_packed \
  --seed 42 \
  --execute
```
