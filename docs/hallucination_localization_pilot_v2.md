# Hallucination Localization Pilot v2

状态：`completed_span_token_gate_passed_onset_gate_failed`

日期：2026-08-18

证据等级：`pipeline pilot`，不是人工 Gold 或 formal mechanism evidence

> 后续状态说明（2026-08-18）：本文件冻结的是 v2 span-target 选择。第 6 节的“不启用 negative-tail”是
> **v2 当时未授权**，不是永久否证。用户要求撤销审计后，v2b 已在当前 S1 target 上完成 matched
> `tail_weight=0/.1/.5` 比较；`.5` 通过 point-estimate pilot guards，现仅保留作扩大 validation 与多
> seed 比较。后续 v2c 扩大复核因 0/3 seed 通过 tail-specific locality，最终选择 T0 并暂缓当前
> absolute-margin T2。见 `docs/hallucination_tail_comparison_v2b.md` 与
> `docs/hallucination_tail_cross_validation_v2c.md`。

## 1. 本轮修复什么

v1 用 `hallucination_onset` 生成 contaminated-tail 标签：首错之前为 0，首错及之后全部为 1。它虽然
能训练，但会把“越靠后越可能为 1”直接写进 target。v1 的 H1/H2 token AP 没有超过绝对位置
baseline，六个 positive dev onset 的 `±5` 命中均为 0。

现有 64 条裁决标签其实保留了每个 material claim 的 exact token span 和
`supported/unsupported/contradicted/non_claim` 状态。v2 不新增 judge 标注，而是按以下固定规则重新物化
token supervision：

```text
supported claim                 -> target 0, mask 1
unsupported / contradicted      -> target 1, mask 1
non_claim / unreviewed token    -> target 0, mask 0
```

因此 v2 监督的是“被审 claim span”，不是把整段后缀都设为 positive。每个位置仍直接索引冻结的
`output_token_ids`；没有重新 tokenize。所有 positive path 的最早 positive token 都必须等于原
`hallucination_onset`，clean path 必须没有 positive token。

## 2. 数据与 shortcut 审计

数据仍使用 v1 冻结的 query-disjoint 48 train / 16 dev，不改变 trajectory、Phi hidden states 或
correctness：

| split | rows | 全部 token | sparse supervised | positive | negative |
|---|---:|---:|---:|---:|---:|
| train | 48 | 14,307 | 6,681 | 922 | 5,759 |
| dev | 16 | 5,008 | 2,451 | 528 | 1,923 |

64 条一共形成 9,132 个 conflict-free sparse token labels。train/dev query overlap 为 0，全部 64 行的
row ID、query ID、`output_token_ids` hash 和 supervision provenance 均通过。

dev shortcut：

| 单位 | shortcut | AUROC | AP |
|---|---|---:|---:|
| token | absolute token index | .586 | .393 |
| token | within-row normalized position | .519 | .241 |
| claim | absolute mean token index | .584 | .422 |
| claim | within-row normalized mean position | .515 | .232 |

稀疏标签显著降低了 v1 contaminated-tail 的位置确定性，但 absolute position 仍然是必须超过的强基线。

## 3. 冻结训练矩阵

所有 cell 使用同一 train/dev manifest、seed 42、5 epochs、batch 4、LR `1e-4`、all-33-layer Phi
features 和 9.55M 参数 CLIR 架构。correctness BCE 始终开启；consistency、tail shaping、pseudo-tail、
progress 和 dual prior 全部关闭。

| cell | token target | positive weight | path MIL |
|---|---|---:|---:|
| S0 | v1 onset-tail | 1.0 | 0 |
| S1 | sparse claim span | 1.0 | 0 |
| S2 | sparse claim span | `5759/922 = 6.2462` | 0 |
| S3 | sparse claim span | `6.2462` | .25 |

主指标是 held-out dev sparse span-token AP。通过门要求候选同时超过 S0、token absolute-position AP，且
claim-mean AP 超过 claim absolute-position AP。exact onset 是独立门：train-only calibrated dev MAE
必须低于 v1 constant-onset `101.5`，同时至少 `1/6` positive row 在 `±5`。

第一次输出目录 `pilot_span_v2` 在打开 dev 指标前发现监督计数器错误：S0 loss 正确使用 14,307 个
onset-tail token，但日志误报 sparse mask 的 6,681。该批训练不作为结果。修复后四个 cell 在新目录
`pilot_span_v2a`、同一 clean commit 下完整重跑；官方结果只认 v2a。

## 4. 结果

| cell | span token AUROC / AP | claim-mean AUROC / AP | path AUROC / AP | fixed onset MAE | onset ±5 |
|---|---:|---:|---:|---:|---:|
| S0 | .706 / .371 | .770 / .375 | .933 / .873 | 134.2 | 0/6 |
| S1 | **.715 / .416** | **.790 / .464** | **.950 / .915** | **82.5** | 0/6 |
| S2 | .656 / .320 | .693 / .370 | .883 / .813 | 99.0 | 0/6 |
| S3 | .682 / .345 | .734 / .404 | .900 / .855 | 98.5 | 0/6 |

S1 的 point estimate 满足预注册 token gate：

- span-token AP `.416` > S0 `.371`，也超过 absolute-position `.393`；
- claim-mean AP `.464` > S0 `.375`，也超过 claim absolute-position `.422`；
- 完整 class balance 会提高固定 `0.5` threshold F1，但明显降低 threshold-free AP/AUROC；
- 增加 path MIL 没有修复这一退化，因此选择最简单的 S1。

上面的选择仍然不稳定。2,000 次 query-row paired bootstrap 中：

- S1−S0 token AP：均值 `+.041`，95% percentile interval `[-.032, +.128]`；
- S1−position token AP：均值 `+.028`，区间 `[-.220, +.265]`；
- S1−S0 claim-mean AP：均值 `+.091`，区间 `[-.056, +.310]`。

区间全部跨 0，所以这是 pilot cell selection signal，不是稳定机制增益。

## 5. 为什么 exact onset 仍未通过

S1 的 fixed-threshold onset MAE 从 S0 的 `134.2` 降到 `82.5`，但六条 positive 都没有进入 `±5`。
train-only threshold 也没有修复：S1 calibrated MAE 为 `172.7`，仍是 `0/6`。

逐行诊断显示，模型能让错误 claim 的平均/最大概率在全局 ranking 中上升，但早期 supported token 仍有
孤立高峰；raw onset decoder 采用“第一个超过阈值的 token”，会被这些高峰过早触发。另一方面，真正
onset token 本身不一定是错误 span 内概率最高的位置。因此：

```text
claim/span discrimination  !=  exact first-boundary decoding
```

使用标注 claim boundary 做 post-hoc claim pooling 可以让少数行 exact，但它依赖评估标注，不能作为当前
部署方案或正式结果。下一轮如果继续 onset，应冻结独立的 causal boundary/segment objective 或 transition-
constrained decoder，并与 raw first-crossing 同时比较；不能把 post-hoc pooling 当成通过。

## 6. 当前裁决

- 保留 S1 unweighted sparse claim-span BCE，作为后续扩大数据与多 seed 验证的 token branch；
- exact onset gate 仍为 false；
- 不启用 pseudo-tail；
- 在 v2 冻结点不启用 negative-tail value shaping；这不是永久否证，后续必须另发 matched protocol；
- 不跑 mixed 3968-row mechanism training；
- 不读取 pilot/final test；
- 不宣称 hallucination localization 改善 Best-of-N。

机器可读结论为
`configs/hallucination_localization_v2/training_result_v2.json`。协议、数据审计和标签分别为：

```text
configs/hallucination_localization_v2/span_target_protocol_v2.json
configs/hallucination_localization_v2/span_target_audit_v2.json
configs/hallucination_localization_v2/span_supervision_annotations_v2.jsonl
configs/hallucination_localization_v2/training_protocol_v2.json
run_artifacts/hallucination_localization_v2/pilot_span_v2a/
```

这些标签继承 v1 双 AI 标注与内部盲审裁决的证据边界，仍然不是人工 Gold。

## 7. 后续 v2b 撤销审计（不回写 v2 结果）

v2b 先确认旧证据不足以永久抛弃 full-tail shaping，再以相同 S1/data/seed/架构/预算直接比较
`tail_weight=0/.1/.5`。`.1` 失败，`.5` 的 span AP `.4535`、explicit-token value-risk AP `.5002`、
correctness AUROC `.9524`，并通过 relative locality；因此保留 `.5` 进入扩大 validation 与多 seed
验证。它仍未获 mixed training 授权，也没有解决 exact onset（`±5 = 0/6`）。

这项后续结果不把 full tail 改写成 token 标签：dev full tail 的 1,460 个 token 中，805 个未审，已审
部分还有 127 个 supported token。其语义只能是 error-contamination reward hypothesis。冻结审计、协议和
结果见：

```text
configs/hallucination_localization_v2/tail_hypothesis_audit_v2b.json
configs/hallucination_localization_v2/tail_comparison_protocol_v2b.json
configs/hallucination_localization_v2/tail_comparison_result_v2b.json
docs/hallucination_tail_comparison_v2b.md
```
