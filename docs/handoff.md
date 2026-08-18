# CLIR 当前交接

最后更新：2026-08-18

本文件只保留当前可执行状态。研究设计见 `docs/proposal.md`，历史路线与弃用原因见
`docs/decision_history.md`；Localization v1 的 contaminated-tail 实验见
`docs/hallucination_localization_pilot_v1.md`。

## 1. 当前停止点

Hallucination Localization Pilot v2 已完成，冻结状态为
`completed_span_token_gate_passed_onset_gate_failed`，证据等级仍是 `pipeline pilot`：

- 64 条裁决 trajectory 形成 41 clean / 23 hallucinated；标签来自双 AI 标注与内部盲审，不是人工 Gold；
- claim reviews 被物化为 9,132 个 conflict-free sparse token labels；train/dev 为 query-disjoint
  48/16，分别监督 6,681/2,451 个 token；
- unweighted sparse claim-span BCE（S1）在 dev 上取得 span-token AP `.416`，超过 onset-tail S0
  `.371` 与 absolute-position `.393`；claim-mean AP `.464`，超过 position `.422`；
- 2,000 次 paired query bootstrap 的差值区间均跨 0，因此 S1 只是当前候选，不是稳定增益；
- exact onset 仍失败：S1 fixed MAE `82.5`，六个 positive dev 的 `±5` 命中仍为 `0/6`；
- pseudo-tail、negative-tail shaping、mixed 3968-row mechanism run 均未获授权，pilot/final test 未读。

当前三种监督不要混称：

| 层级 | canonical 字段 | 含义 | 当前裁决 |
|---|---|---|---|
| 整条序列 | `path_hallucinated` | trajectory 是否包含 unsupported/contradicted material claim | 有 ranking 信号，仅作小样本诊断 |
| token/span | `token_hallucination_target` + `token_hallucination_mask` | 只监督已审 claim span；mask 外不是负例 | S1 point-estimate gate 通过，待扩量和多 seed |
| 首错边界 | `hallucination_onset` | 第一个 unsupported/contradicted claim 的首 token | 独立 gate 未通过 |

## 2. 当前采用的研究路线

1. Phi-3.5-mini 同时作为 task model 与 feature model；正式表示保留 embedding + 32 blocks。
2. Consistency 主路线是 Route A：同一原始 prompt 下从 Phi on-policy candidates 挖掘
   reasoning-equivalent pairs；Route B 的 Phi self-rewrite 只作后续对照。
3. 错误 trajectory 不要求 rewrite。错误机制等价组只有在定向采样和独立 verifier 能稳定判断后才做。
4. Localization 当前保留 S1 sparse span branch；exact onset 作为独立 boundary 问题继续，不与
   consistency 或 dual prior 混训。
5. Dual-prior targets 必须由外部流程生成，并在 consistency/localization 分别验证后再接入。

外部 Qwen/Falcon rewrite、旧 Route A v1 manifest 与 Stage 1B v4 的 outcome-only 数据都不是当前机制
训练入口。

## 3. 当前版本化 artifact

### Route A v1a

- `configs/on_policy_pilot0_v1a/adjudicated_relations_v1.jsonl`：27 accept / 4 reject；
- `configs/on_policy_pilot0_v1a/train_mixed_v1.jsonl`：4096 行，仅 54 行有 consistency metadata；
- `configs/on_policy_pilot0_v1a/training_protocol_v1.json` 与 `training_result_v1.json`；
- 解释边界：`docs/on_policy_pilot0_reaudit_v1.md`。

### Hallucination Localization v2

- 标签规则：`configs/hallucination_localization_v2/span_target_protocol_v2.json`；
- 标签与审计：`span_supervision_annotations_v2.jsonl`、`span_target_audit_v2.json`；
- 训练协议与结论：`training_protocol_v2.json`、`training_result_v2.json`；
- 官方四 cell 输出：`run_artifacts/hallucination_localization_v2/pilot_span_v2a/`；
- `run_artifacts/hallucination_localization_v2/pilot_span_v2/` 是监督计数日志有误的 pre-audit 输出，永久排除；
- 完整解释：`docs/hallucination_localization_pilot_v2.md`。

v2 继承的 v1 labels/split 是来源 artifact，不是当前训练方案；其 hash 与 lineage 已记录在 v2
protocol/audit 中，不在本 handoff 重复展开。

## 4. 下一步

不要原样重跑 seed 42，也不要继续调 class weight 或给 S1 添加 path MIL：S2/S3 已显示这些改动会降低
threshold-free span/claim ranking。

推荐下一轮单独发布 onset Pilot v2b：

1. 保持 S1 sparse token branch、48/16 split、features 与预算不变；
2. 预先冻结一个 causal boundary/segment objective 或 transition-constrained decoder，并保留 raw
   first-crossing 作为对照；
3. 主门仍报告 span-token/claim AP，onset 门单独要求 `±5` 命中与 MAE；
4. 禁止使用 gold claim boundary 做部署时 pooling，禁止从 dev post-hoc 选阈值；
5. v2b 若出现可解释信号，再扩大 positive labels 并跑多 seed；否则先扩标，不进入 mixed training。

## 5. 不可破坏的约束

- `prompt_token_ids/output_token_ids` 是模型输入与 token label 的唯一 source of truth，禁止重分词贴标签。
- sparse target/mask 必须成对、长度严格等于 `T`、值为 binary；mask 外 target 固定 0 且不参与 loss。
- `hallucination_onset` 是生成 token 的零基索引；clean 为 `-1`，uncertain 不得强制写 0。
- `query_id` 与 `semantic_id` 分工固定；condition 来自独立 prompt-only forward 并在 query 内共享。
- 正式 feature 保留全部 33 层；online/precomputed 是协议变量，同一矩阵内必须一致。
- correctness 只能监督 outcome BCE，不能派生 hallucination、progress、prior 或 reconstruction target。
- augmentation view 不继承 source localization label，除非该 view 被独立重标。
- 评分必须绑定 manifest、checkpoint、协议、dtype、candidate order 与 SHA256 provenance。
- 冻结 artifact 不原地修改；schema、阈值、loss 或指标语义变化必须发布新版本。
- `pilot_test/final_test` 在 protocol selection 完成前保持未读。

## 6. 当前证据边界

- correctness-only Stage 1 是 `small-scale real`，没有稳定的 encoded→CLIR 增益。
- Route A v1a、Localization v1/v2 都是 `pipeline pilot`。
- verifier selection 的 Mistral-24B 只获 Silver pilot 授权，不能自动迁移为 hallucination Gold。
- S1 的 span/claim point estimates 有希望，但 16-row dev bootstrap 不稳定；exact onset 明确未通过。
- 不能宣称 consistency、hallucination localization 或 tail shaping 已改善 Best-of-N。

## 7. 验证与入口

使用固定 SWIFT Python：

```bash
cd /prodcpfs/user/panzhixin/ICLR_2027
P=/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python
"$P" -m pytest -q
"$P" -m pip check
```

关键入口：

```text
README.md                                         当前方案与项目入口
docs/handoff.md                                   当前执行交接
docs/hallucination_localization_pilot_v2.md       当前 localization 结果
docs/clir_supervision_protocol.md                  外部监督与 sparse-mask 契约
docs/decision_history.md                          历史路线与转向理由

src/consistency_localized_reward.py               模型与 loss
src/clir_supervision.py                           外部监督绑定与覆盖审计
src/clir_localization_evaluation.py               span/claim/path/onset 指标
scripts/materialize_hallucination_span_targets_v2.py  sparse target 物化
scripts/run_hallucination_localization_pilot_v2.py    S0-S3 launcher
scripts/summarize_hallucination_span_pilot_v2.py      冻结结论与 bootstrap
train_clir.py / score_clir.py / evaluate_clir.py     训练、打分、Best-of-N
```
