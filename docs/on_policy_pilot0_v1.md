# Route A on-policy equivalence Pilot-0

最后更新：2026-08-17

## 结论

本轮在 `pipeline_pilot` 证据层完成了 Phi 原生等价候选构建、Silver verifier 筛选、exact-token online
manifest 发布和两组同架构训练。流水线可运行，consistency supervision 也确实有 applicable pairs；但
当前 positive cosine loss 在完成全正例 BCE 训练的 no-consistency checkpoint 上约为 0；进一步诊断
确认这不是随机初始化属性，而是全正例训练造成的全局 representation collapse。打开现有 loss 后，
优化主要来自 different-semantic/same-style negative 排斥，并且排斥强到同时拉远 native positive；
不能把本轮解释为 rewrite consistency 有效。

## 数据规模

- `development_32`：32 query、256 条 Phi 原生 trajectory；
- 同题、同 correctness、同归一化答案、文本不同：727 个 correct 高召回候选对；
- 每 query 最多 4 对并按 output-token length gap 排序：125 对；
- Mistral-24B Silver verifier：125/125 可解析，121 accept、4 reject；
- 每 query 最多发布 1 个最大 gap accept：31 semantic groups、62 views、31 query；
- 所有发布 view 都是 correct；`style_id=native_anchor/native_alternate` 只是 view role；
- 未继承 hallucination、onset、advantage、progress 或 dual-prior supervision。

发布 manifest：
`run_artifacts/on_policy_pilot0_v1/published_silver_v1/train_online.jsonl`，SHA256
`e62add6806ea4d35aa90aec5f27ecc13df8209058f14c434b61227c7f32f771e`。

## Tiny training

两组都使用同一 62-row manifest、CLIR layer-transformer、seed 42、batch size 4、2 epochs、query-level
80/20 split、Phi exact-token online hidden states。所有非 consistency 辅助 loss 权重均为 0。

| run | consistency weights | epoch-2 checkpoint train positive | negative | score | total | epoch-2 validation total |
|---|---|---:|---:|---:|---:|---:|
| no-consistency | `0 / 0` | 0.0000 | 0.7669 | 0.00172 | 0.7669 | 0.7969 |
| consistency | `1 / 0.1` | 0.0244 | 0.2028 | 0.02913 | 0.2301 | 0.4671 |

`no-consistency` 的表中 consistency 值只作只读诊断，不进入 total loss。其 positive 在两个 epoch 的
checkpoint evaluation 都是 0。representation diagnostic 表明这是全正例 BCE 训练后的塌缩，而非
训练前已经饱和。consistency run 的 negative 明显下降，但 positive 与 score 项变差，且 validation
total 从 epoch 1 的 `0.4370` 上升到 `0.4671`。

数据全部为 correct，因此 correctness BCE 退化为单类别问题，两个 run 的 health report 都明确记录
`reason=degenerate_class_distribution`。本轮不能评估 correctness ranking 或 BoN 效果。

## 下一门

在扩大 Route A 或运行 Route B 前，先实现 representation diagnostic：在训练前直接报告每个 native
pair 的 trajectory representation cosine、reward-score delta，以及同题不同 candidate、不同题同
view-role 的对照分布。只有确认某个层/聚合位置存在非饱和的等价差异后，才决定调整 representation、
margin/temperature 或 loss。loss 的正式改变仍需单独协议和对照实验。

## Representation diagnostic v1

报告：`run_artifacts/on_policy_pilot0_v1/representation_diagnostic_v1/report.json`。它对 31 个 native
positive pair 和 30 个 different-query/same-view-role 对照，比较 Phi raw mean、encoder mean、
conditioned pooled representation、projected representation，以及同 semantic pair 的 score delta。

projected representation cosine 均值：

| checkpoint | native positive | cross-query control | gap |
|---|---:|---:|---:|
| random init seed 42 | 0.99565 | 0.98040 | 0.01525 |
| no consistency | 0.99954 | 0.99793 | 0.00162 |
| consistency | 0.97492 | 0.32926 | 0.64566 |

Phi raw mean 本身是 `0.99649` positive / `0.98184` control，gap `0.01465`；随机 encoder 没有抹掉这点
差异。no-consistency 的全正例 BCE 把所有方向压得接近一致。现有 consistency objective 成功阻止全局
塌缩并强力分开不同 query，但 native positive 也由 `0.99565` 降至 `0.97492`，其中最差 pair 只有
`0.68218`。同 semantic absolute score delta 的均值则是 random `0.05168`、no-consistency
`0.03656`、consistency `0.12138`；显式 score-consistency 项没有在当前联合目标下实现同分。

因此瓶颈不在 Phi raw representation 或 pooling 单点，而在当前 objective 的梯度平衡：初始 positive
已经较近，`relu(sim-margin)` negative 项有大得多的下降空间，训练优先通过大范围排斥降低 loss，并
牺牲了部分 positive 和 score consistency。下一实验应冻结数据不变，只比较 loss 级最小改动（例如
去掉 negative、降低 negative 权重或使用相对/温度化目标）；不得把继续增加 rewrite 数量当作修复。
