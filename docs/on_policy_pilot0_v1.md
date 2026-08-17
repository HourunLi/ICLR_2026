# Route A on-policy equivalence Pilot-0

最后更新：2026-08-17

## 结论

本轮在 `pipeline_pilot` 证据层完成了 Phi 原生等价候选构建、Silver verifier 筛选、exact-token online
manifest 发布和两组同架构训练。流水线可运行，consistency supervision 也确实有 applicable pairs；但
当前 positive cosine loss 在关闭 consistency 的随机初始化模型上已经约为 0，因此这批数据没有提供
可观的“拉近等价轨迹”梯度。打开现有 loss 后，优化主要来自 different-semantic/same-style negative
排斥，不能把本轮解释为 rewrite consistency 有效。

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
checkpoint evaluation 都是 0；这意味着当前正项在训练前已经饱和。consistency run 的 negative 明显
下降，但 positive 与 score 项变差，且 validation total 从 epoch 1 的 `0.4370` 上升到 `0.4671`。

数据全部为 correct，因此 correctness BCE 退化为单类别问题，两个 run 的 health report 都明确记录
`reason=degenerate_class_distribution`。本轮不能评估 correctness ranking 或 BoN 效果。

## 下一门

在扩大 Route A 或运行 Route B 前，先实现 representation diagnostic：在训练前直接报告每个 native
pair 的 trajectory representation cosine、reward-score delta，以及同题不同 candidate、不同题同
view-role 的对照分布。只有确认某个层/聚合位置存在非饱和的等价差异后，才决定调整 representation、
margin/temperature 或 loss。loss 的正式改变仍需单独协议和对照实验。
