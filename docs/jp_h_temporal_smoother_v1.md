# JP → H Fixed Temporal Smoother v1

最后更新：2026-08-21

## 1. 冻结问题

上一轮冻结 JP 的独立 `Linear(768,1)` H probe 在主要 48 条 OOF 轨迹上稳定读出了 claim-level signal，
但逐 token 输出在 claim 内抖动，导致 span AP 三个 seed 都低于位置基线。用户随后只授权一个单因素工程
pilot：不再训练任何参数，在原 12 个 OOF head 的输出上加入固定 3-token temporal smoother。

本轮协议在读取 logit-smoother 结果前冻结于
`configs/jp_h_temporal_smoother_v1/protocol_v1.json`，SHA256 为
`dc123b0a07ddcb70c974cdbacd6fa3a8ae607dfb087f9b3e6d2205fb8e4c5cfb`；执行 commit 为
`9ad426806ef616d4ddc3d93c460dd4bbc5093d6d`，worktree clean。

## 2. 唯一变化

对每条完整 trajectory，读取上一轮不可变 `dev_scored.jsonl` 中的精确逐 token 概率 `p_t`，执行：

```text
z_t  = logit(p_t)
z'_t = mean(z_{t-1}, z_t, z_{t+1}) over available real tokens
p'_t = sigmoid(z'_t)
```

- 固定 window 为 3、radius 为 1，没有窗口扫描；
- 两端只平均实际存在的 2 个 token，不补零；
- mask 只表示真实生成 token / padding，不使用 sparse supervision mask；因此“哪里被标了”不会泄漏给
  smoother；
- 计算使用 float64。输入是原 float32 sigmoid 输出的稳定 logit 逆变换，避免重跑不同 device 的
  `Linear(h)` 造成数值差异；
- trainable parameters 为 0，12 个 linear head 没有重训；
- JP checkpoint、reward score、token values、direct priors、双向 stop-gradient mutual `.25`、
  shared-gradient gate `10`、tail、progress、reconstruction 和 consistency 全不变；
- centered smoother 会读取后一个 token，只适用于完整 trajectory 的离线 reward scoring。若未来要求生成时
  在线逐 token 决策，必须另发 causal 协议，不能静默复用本结果。

## 3. 预设工程门

主要裁决只使用上一轮同样的 folds 1–3、48 条 OOF 轨迹；fold 0 保持 selection-exposed diagnostic。
每个 seed 必须同时满足四项严格大于 0：

1. smoother span AP − 同批 strongest position AP；
2. smoother claim AP − 同批 strongest position AP；
3. smoother span AP − 配对 raw linear span AP；
4. smoother claim AP − 配对 raw linear claim AP。

至少 2/3 seed 四项全过，且四种跨 seed mean delta 都严格为正，才称为
`engineering_signal_supported`。即使通过，也不采用方法、不接 reward，只进入新增盲标 validation 门。

## 4. 结果

12/12 cells 完成；12 个源 prediction files、12 个源 probe checkpoints 均经 SHA256 复核。192/192 个
scored row views 除四个 H 派生字段和新增 smoother provenance 外逐字段完全一致。参数增加为 0，head
重训为 0。

主要 48 条结果：

| seed | raw span AP | smoother span AP | position span AP | raw claim AP | smoother claim AP | position claim AP | 四门 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | .2357 | .2801 | .2599 | .3582 | .3663 | .3046 | pass |
| 43 | .2329 | .2719 | .2599 | .3409 | .3642 | .3046 | pass |
| 44 | .2343 | .2684 | .2599 | .3249 | .3324 | .3046 | pass |

跨 seed mean：

- smoother span AP `.2735`，相对 position `+.0136`，相对 raw linear `+.0392`；
- smoother claim AP `.3543`，相对 position `+.0497`，相对 raw linear `+.0130`；
- 3/3 seed 通过全部四门，因此状态为 `completed_engineering_signal_supported`。

这个结果支持“当前 token/span 失败主要含有可被极小局部一致性算子修复的高频抖动成分”，不支持
“hallucination localizer 已验证”或“Best-of-N 会提高”。

## 5. 不能省略的反证与边界

把 selection-exposed fold 0 也并入后，64 条 cross-fold pooled diagnostic 的 smoother span/claim AP 仍低于
同批 position baseline：三个 seed 的 span AP 为 `.2695/.2639/.2619`，位置为 `.2868`；claim AP 为
`.3154/.3144/.2935`，位置为 `.3209`。它们都比各自 raw linear 提高，但没有越过 position。

这里不能简单说 smoother “在 64 条失败”：AP 对不同 fold/head 的分数尺度不具线性可合并性，fold 0 又是
selection-exposed；主要 48 条本身也由三个 OOF head 拼接，所以这一现象仍提示真实风险——不同 head 或不同
domain 的校准可能掩盖 localization 排序。新盲标验证应使用预先冻结的单个 final head，或在协议中显式冻结
跨 head calibration；不能看到新标签后再决定。

更根本的限制是 window=3 来自查看同一 64 条上的事后平滑诊断。虽然本轮从概率均值改成了预先冻结的
logit 均值，并且没有再扫窗口，它仍是 post-selection exploratory evidence，不能用 OOF 身份洗掉结构选择
泄漏。plain linear probe 的历史失败也不被追溯改判。

## 6. 冻结裁决与下一门

当前裁决：

- 保留固定 3-token masked centered logit smoother，作为下一批盲标要验证的唯一 H readout 候选；
- `method_adopted=false`；
- `score_coupling_authorized=false`；
- 不重开 full tail、exact onset、progress、reconstruction 或 consistency；
- 不读取 `pilot_test` / `final_test`。

下一道阻塞门是新增一批未参与窗口/结构选择的 localization validation，并由独立第二标注者盲标。优先覆盖
非数学领域；第二标注 prompt 必须要求每完成一条立即 durable 落盘，以避免超时丢失。只有 fixed smoother
在新标签上同时守住 span 与 claim gate，才讨论一个独立冻结的透明 score-coupling 实验。

## 7. Artifact

- 机器结果：`configs/jp_h_temporal_smoother_v1/result_v1.json`，SHA256
  `c3b6add1cbc1fd987e49f67d0e28c6146058992667d7762a83056609a4f66771`；
- ignored matrix：`run_artifacts/jp_h_temporal_smoother_v1/matrix_v1/matrix_run.json`，SHA256
  `2ce147d5feef9fc52a8cc1c5ed9e5ae28ad5150c7841c9cb85290c457a6d884a`；
- runner：`scripts/run_jp_h_temporal_smoother_v1.py`；
- summarizer：`scripts/summarize_jp_h_temporal_smoother_v1.py`；
- 实现：`src/clir_h_temporal_smoother.py`。
