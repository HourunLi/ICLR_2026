# CLIR 当前交接

最后更新：2026-08-18

本文件只保留当前可执行状态。研究设计见 `docs/proposal.md`，历史路线与弃用原因见
`docs/decision_history.md`；Localization v1 的 contaminated-tail 实验见
`docs/hallucination_localization_pilot_v1.md`，当前 tail 撤销审计与直接比较见
`docs/hallucination_tail_comparison_v2b.md`，最终 4-fold × 3-seed 复核见
`docs/hallucination_tail_cross_validation_v2c.md`，首个相对锚定修复试验见
`docs/hallucination_relative_tail_pilot_v2d.md`。

## 1. 当前停止点

Hallucination Localization v2/v2b/v2c/v2d 已完成。最新 relative-tail R1 冻结状态为
`completed_fail_keep_t0`，证据等级仍是 `pipeline pilot`：

- 64 条裁决 trajectory 形成 41 clean / 23 hallucinated；标签来自双 AI 标注与内部盲审，不是人工 Gold；
- claim reviews 被物化为 9,132 个 conflict-free sparse token labels；train/dev 为 query-disjoint
  48/16，分别监督 6,681/2,451 个 token；
- unweighted sparse claim-span BCE（S1）在 dev 上取得 span-token AP `.416`，超过 onset-tail S0
  `.371` 与 absolute-position `.393`；claim-mean AP `.464`，超过 position `.422`；
- 2,000 次 paired query bootstrap 的差值区间均跨 0，因此 S1 只是当前候选，不是稳定增益；
- exact onset 仍失败：S1 fixed MAE `82.5`，六个 positive dev 的 `±5` 命中仍为 `0/6`；
- 旧 v1/v2 证据不足以永久否证 full-tail shaping；v2b 在相同 S1/data/seed/预算下比较
  `tail_weight=0/.1/.5`，轻权重 `.1` 失败，`.5` 通过全部预设 point-estimate guards；
- `.5` 相对 control 的 tail−pre gap bootstrap difference 为 `-.642`，95% interval
  `[-1.434,-.087]`；explicit-token value-risk AP difference 为 `+.056`，区间 `[+.008,+.108]`；
- v2c 用 4 folds × 3 seeds 完成 24-cell 复核；fold 0 是 selection-exposed continuity，采用门只看 folds
  1–3 拼接的 48 条 out-of-fold predictions；
- T2 在三个 seed 的 value-risk AP 都提高，跨 seed mean 的 value-risk/span/correctness delta 为
  `+.0357/+.0170/+.0093`，所以它有 regularization/ranking signal；
- 但三个 seed 的 `tail−clean` gap 全部恶化，mean delta `+.2517`。T2 的 clean mean value 分别下移
  `-3.728/-3.552/-4.620`，均比对应 tail 下移更大，0/3 seed 通过 tail-specific locality；
- v2d 的单-cell R1 改用行内 pre-onset-relative margin，relative violation `.586→.091`、`tail−pre`
  delta `-1.712`，所以它确实消除了统一平移捷径；但 clean mean 下移 `-2.300`，大于 tail 的 `-1.374`，
  `tail−clean` delta `+.925`，value-risk/span AP 分别下降 `.109/.100`，未获多 seed 扩跑资格；
- 当前 localization 默认选择 T0（S1 sparse BCE、无 full tail）。absolute T2 与 relative R1 都暂缓而非永久
  否证；pseudo-tail、mixed 3968-row mechanism run 仍未获授权，pilot/final test 未读。

当前三种监督和一种 reward shaping 不要混称：

| 层级 | canonical 字段 | 含义 | 当前裁决 |
|---|---|---|---|
| 整条序列 | `path_hallucinated` | trajectory 是否包含 unsupported/contradicted material claim | 有 ranking 信号，仅作小样本诊断 |
| token/span | `token_hallucination_target` + `token_hallucination_mask` | 只监督已审 claim span；mask 外不是负例 | T0/S1 为当前 standalone 默认 |
| 首错边界 | `hallucination_onset` | 第一个 unsupported/contradicted claim 的首 token | 独立 gate 未通过 |
| onset 后 value shaping | full-tail margin loss | 从 onset 起降低全部 token value，包括 supported/unreviewed token | absolute T2 因全局 shift 暂缓；relative R1 因无 clean anchor 与 sparse AP 损失失败；不是 token 标签 |

## 2. 当前采用的研究路线

1. Phi-3.5-mini 同时作为 task model 与 feature model；正式表示保留 embedding + 32 blocks。
2. Consistency 主路线是 Route A：同一原始 prompt 下从 Phi on-policy candidates 挖掘
   reasoning-equivalent pairs；Route B 的 Phi self-rewrite 只作后续对照。
3. 错误 trajectory 不要求 rewrite。错误机制等价组只有在定向采样和独立 verifier 能稳定判断后才做。
4. Localization 当前选择 T0/S1 sparse span branch；absolute-margin T2 暂缓，简单 pre-onset-relative R1 也未
   通过。exact onset 与显式 clean-matched 的 tail repair 都是独立后续问题。
5. 当前进入 dual-prior。v1 已冻结 64 条 fixed-unit 双标包；key/complete 由外部双标生成，首轮只验证 direct
   targets。mutual distill、gate alignment 和 reconstruction 全关；后者仍只接受独立 768-d target。

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

### Full-tail comparison v2b

- 撤销审计：`configs/hallucination_localization_v2/tail_hypothesis_audit_v2b.json`；
- 冻结协议：`configs/hallucination_localization_v2/tail_comparison_protocol_v2b.json`；
- 机器结论：`configs/hallucination_localization_v2/tail_comparison_result_v2b.json`，SHA256
  `29b5d20e4563f217b22ecd392c381fdd5aaf8f044830705542f21d7031df6bbe`；
- 三个 matched cell：`run_artifacts/hallucination_localization_v2/pilot_tail_v2b/`；
- 完整解释与禁止越界的结论：`docs/hallucination_tail_comparison_v2b.md`。

### Full-tail cross-validation v2c

- 数据划分协议：`configs/hallucination_localization_v2/tail_cv_data_protocol_v2c.json`；
- 训练/采用协议：`configs/hallucination_localization_v2/tail_cv_protocol_v2c.json`；
- 机器结论：`configs/hallucination_localization_v2/tail_cv_result_v2c.json`，SHA256
  `d8264599b61c221958598d8a087ded7617a9cfdf5b8cdee46716ad4ea2d2a5d9`；
- fold 审计：`run_artifacts/hallucination_localization_v2/data_tail_cv_v2c/fold_audit_v2c.json`；
- 22 个新 cell：`run_artifacts/hallucination_localization_v2/pilot_tail_cv_v2c/`；
- 完整解释：`docs/hallucination_tail_cross_validation_v2c.md`。

### Relative full-tail pilot v2d

- 冻结协议：`configs/hallucination_localization_v2/relative_tail_protocol_v2d.json`，SHA256
  `1543f8169606f86e6eee054e5ee59105e345f875a2d0ebca2f02b89bb5de1361`；
- 机器结论：`configs/hallucination_localization_v2/relative_tail_result_v2d.json`，SHA256
  `6ced98914c37ee9cea0ac4686d3b1b079c806fd857589070d8774e18c33e71d1`；
- 单个新 R1 cell：`run_artifacts/hallucination_localization_v2/pilot_relative_tail_v2d/`；T0 严格复用 v2b；
- 结论：执行门通过，但 clean-locality、semantic-value 与 sparse-span guards 失败，不扩跑；
- 完整解释：`docs/hallucination_relative_tail_pilot_v2d.md`。

### Dual-prior evidence v1

- 冻结协议/定义：`configs/dual_prior_evidence_v1/protocol_v1.json`、`annotation_guide_v1.md`；
- 盲包/lineage：`annotation_items_v1.jsonl`、`annotation_lineage_v1.jsonl`、`package_report_v1.json`；
- 规模：64 rows / 64 queries / 1210 fixed units，原 query-disjoint 48/16 membership；
- primary：64/64 schema/token-map valid，`labels_primary_v1.jsonl` SHA256
  `06f24880e226adba33f818aeea9a62df19510f45c599286f3135c9380bd95526`；结构合格但有 late-position 与
  wrong-path guide-alignment 警告，禁止单独采用；
- primary 审计：`primary_report_v1.json`、`primary_semantic_audit_v1.json`；
- 第二标注的冻结语义 prompt：`configs/dual_prior_evidence_v1/secondary_prompt_v1.md`；当前实际发送的是
  `secondary_prompt_resumable_v1a.md`，要求每判完一条立即用
  `scripts/checkpoint_dual_prior_secondary_v1.py` 原子落盘并可按合法前缀续跑；操作附录为
  `secondary_execution_addendum_v1a.json`，标签语义与 64 条输入均未改变；
- 设计与代码审计：`docs/dual_prior_evidence_pilot_v1.md`。

v2 继承的 v1 labels/split 是来源 artifact，不是当前训练方案；其 hash 与 lineage 已记录在 v2
protocol/audit 中，不在本 handoff 重复展开。

## 4. 下一步

不要再重跑 v2c 或继续扫 absolute tail weight。dual-prior 接下来严格按已冻结 v1 做：

1. 用 `secondary_prompt_resumable_v1a.md` 收取独立 secondary；每条判断都立即 checkpoint，先用 helper
   `status` 查看合法前缀与下一 item，64/64 后用 `finalize` 和 `validate_dual_prior_secondary_v1.py` 校验；
2. 计算 eligibility 与 key/complete unit-set agreement，特别检查错误路径的 flaw-vs-terminal 语义，逐项裁决
   set disagreement；
3. 物化 exact Phi token gold，审计正例比例、位置 shortcut、`key ⊆ complete` 与 head 可分性；
4. 跑 D0 correctness-only、D1 key-only、D2 complete-only、D3 joint direct-target；四格都关闭 consistency、
   hallucination、tail、progress、distill、gate alignment 和 reconstruction；
5. direct target 可学后，才另发协议比较 containment collaboration 与旧 mutual MSE。reconstruction 仍要求
   独立 768-d target，禁止 same-candidate pooling；
6. exact onset 与下一代 clean-matched tail repair 进入 backlog，不阻塞 dual-prior，也不进入 mixed run；不要
   扩跑或扫描已经失败的 R1 weight/margin。

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
- Route A v1a、Localization v1/v2/v2b/v2c/v2d 都是 `pipeline pilot`。
- verifier selection 的 Mistral-24B 只获 Silver pilot 授权，不能自动迁移为 hallucination Gold。
- T2 的 AP/ranking signal 在 v2c 仍存在，但 absolute-margin implementation 的 tail locality 0/3 seed 通过；
  R1 虽解决统一 shift，却没有 clean anchor，并明显损伤当前 sparse AP。只能选择 T0 并暂缓这两个实现，
  不能写成 tail hypothesis 永久失败。exact onset 明确未通过。
- 不能宣称 consistency、hallucination localization 或 tail shaping 已改善 Best-of-N，也不能说 tail 已被
  永久证伪或已获 mixed-training 授权。

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
docs/hallucination_tail_comparison_v2b.md         tail 撤销审计与 matched 直接比较
docs/hallucination_tail_cross_validation_v2c.md   4-fold × 3-seed 最终复核
docs/hallucination_relative_tail_pilot_v2d.md     relative R1 单-cell 修复试验
docs/clir_supervision_protocol.md                  外部监督与 sparse-mask 契约
docs/decision_history.md                          历史路线与转向理由

src/consistency_localized_reward.py               模型与 loss
src/clir_supervision.py                           外部监督绑定与覆盖审计
src/clir_localization_evaluation.py               span/claim/path/onset 指标
scripts/materialize_hallucination_span_targets_v2.py  sparse target 物化
scripts/run_hallucination_localization_pilot_v2.py    S0-S3 launcher
scripts/summarize_hallucination_span_pilot_v2.py      冻结结论与 bootstrap
scripts/summarize_hallucination_tail_comparison_v2b.py  T0-T2 护栏与 bootstrap
scripts/summarize_hallucination_tail_cv_v2c.py        out-of-fold 多 seed 采用门
scripts/summarize_hallucination_relative_tail_v2d.py  relative R1 冻结裁决
train_clir.py / score_clir.py / evaluate_clir.py     训练、打分、Best-of-N
```
