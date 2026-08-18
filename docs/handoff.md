# CLIR 当前交接

最后更新：2026-08-19

本文件只保留当前可执行状态。研究设计见 `docs/proposal.md`，历史路线与弃用原因见
`docs/decision_history.md`；Localization v1 的 contaminated-tail 实验见
`docs/hallucination_localization_pilot_v1.md`，当前 tail 撤销审计与直接比较见
`docs/hallucination_tail_comparison_v2b.md`，最终 4-fold × 3-seed 复核见
`docs/hallucination_tail_cross_validation_v2c.md`，首个相对锚定修复试验见
`docs/hallucination_relative_tail_pilot_v2d.md`。

## 1. 当前停止点

Dual-prior direct-target、原始 mutual distillation 与首个 reward-gate integration 均已完成；项目方法已明确
保留 direct targets + 原始 mutual distillation + 原始 shared-gradient reward gate。v1 gate 的冻结状态仍为
`completed_reward_gate_integration_diagnostic_only`，但不再以 head-only 作为当前下一路线：

- secondary 64/64 完成，key/complete macro unit F1 为 `.5469/.8456`；16 条 exact agreement，48 条
  disagreement 全部 role-blind adjudicated；
- 最终 gold 64/64 usable，48/16 query-disjoint，63 条严格 `key ⊂ complete`、1 条相等；
- 4 cells × 3 seeds 的 12/12 训练、评分、exact-token evaluation 全部完成；全部 8 个冻结 guard 为 3/3
  seeds 通过；
- unit AP 均值：key D0/D1/D3/position 为 `.079/.377/.433/.133`；complete D0/D2/D3/position 为
  `.328/.921/.919/.268`；
- D3 correctness AUROC 相对 D0 `+.005`；两图 mean absolute probability difference `.302`、correlation
  `.770`，未塌缩；
- 用户明确裁决保留原始双向 stop-gradient mutual MSE，不以 containment 替换；M0/M1 的 6/6 cells 已完成；
- M1 的 held-out symmetric attention MSE 三 seed 相对下降 `27.0%/24.0%/33.3%`，均值 `28.1%`；
- key/complete unit AP 相对 M0 均值为 `-.0085/-.0008`，correctness AUROC 持平；两图 probability difference
  `.295`、correlation `.793`，全部 guard 3/3 通过；
- G0/G1 复跑同一 M1 control，唯一变化是 gate-to-detached-fused-prior alignment `0→10`；权重在读取任何
  G1 结果前由历史 M1 loss/gradient scale audit 冻结；
- G1 的 held-out gate MSE 三 seed 都下降，均值 `79.6%`；gate effective-token fraction `.929→.370`，与
  fused prior `.379` 接近，证明 gate objective 可优化；
- G1 的 complete unit AP 平均 `+.0102`、correctness AUROC `-.0370`，但 key unit AP 平均 `-.0766`，只有
  1/3 seeds 满足 `delta>=-.05`；因此 v1 不追溯改判为“通过”或正式效果证据；
- 方法身份复核确认 G1 正是 proposal 与核心代码原有的 shared-gradient gate，并非后来提出的 head-only；
  用户裁决优先把项目原方法训起来，head-only 降为未执行的历史 diagnostic ablation；
- original-scale v2 已物化 496 queries / 3968 rows mixed train：48 条 prior Gold 保留，16 个 prior-dev query
  连同其 128 个 candidates 全部排除；独立 ranking validation 为 500 queries × 16 candidates，query overlap=0；
- `training_protocol_v2.json` 冻结 G0/G1 × seeds 42/43/44、5 epochs、全 33 层；尚未启动，因此当前仍没有
  original-scale ranking 结果；
- direct priors 与原 mutual `.25` 不受该裁决影响。reconstruction 仍关闭，pilot/final test 未读。

此前 Hallucination Localization v2/v2b/v2c/v2d 已完成。最新 relative-tail R1 冻结状态为
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

### 为什么当前不采用 tail

当前拒绝的是两个已经跑过的实现，不是整个 full-tail hypothesis：

1. absolute T2 在 selection-unexposed 三 seed 上虽然有 AP/ranking signal，但 `tail−clean` 三个 seed 全部
   恶化；clean value 每次都比 tail 下移更多，所以不能把收益归因为 tail-local shaping；
2. relative R1 精确修掉了整行常数平移捷径，也学会了 `tail<pre`，但没有约束 clean row，最终
   `tail−clean` 仍恶化 `+.925`；
3. R1 的 value-risk AP 与 sparse-span AP 分别下降 `.109/.100`，超过冻结保护门；不能用事后 weight/margin
   sweep 覆盖失败；
4. correctness 没有 catastrophic failure，tail 也不是“完全无效”。不采用的原因是缺少 tail-specific
   locality 并损伤 intended localization，而不是 tail family 已被永久证伪。

正式机器记录为 `configs/hallucination_localization_v2/tail_non_adoption_record_v2d.json`，SHA256
`fed23537b52a2665f8b99883f1750658cf1f386a668e65dfe5d491eb43e90f18`。未来重开必须另发显式
clean-matched positional control 的新协议。

## 2. 当前采用的研究路线

1. Phi-3.5-mini 同时作为 task model 与 feature model；正式表示保留 embedding + 32 blocks。
2. Consistency 主路线是 Route A：同一原始 prompt 下从 Phi on-policy candidates 挖掘
   reasoning-equivalent pairs；Route B 的 Phi self-rewrite 只作后续对照。
3. 错误 trajectory 不要求 rewrite。错误机制等价组只有在定向采样和独立 verifier 能稳定判断后才做。
4. Localization 当前选择 T0/S1 sparse span branch；absolute-margin T2 暂缓，简单 pre-onset-relative R1 也未
   通过。exact onset 与显式 clean-matched 的 tail repair 都是独立后续问题。
5. Dual-prior 保留项目原方法：direct key/complete targets、symmetric stop-gradient mutual MSE，以及
   gate-to-detached-fused-prior 的 shared-gradient alignment。v1 的 key-AP 保护门失败按历史原样保留，但当前
   不换 head-only；先用 v2 mixed outcome data 和 query-grouped ranking 判断原方法的任务效果。containment
   不替代 mutual，reconstruction 继续关闭且仍只接受独立 768-d target。

外部 Qwen/Falcon rewrite 与旧 Route A v1 manifest 不是当前机制训练入口。Stage 1B v4 的旧训练结论仍只作
optimization diagnostic，但其最新 checker-v5 outcome manifests 由 original-scale v2 作为 correctness 数据
carrier 复用；新的 query 排除、prior 合并和实验结论全部受 v2 协议约束。

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
- 非采用裁决：`configs/hallucination_localization_v2/tail_non_adoption_record_v2d.json`，SHA256
  `fed23537b52a2665f8b99883f1750658cf1f386a668e65dfe5d491eb43e90f18`；
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
- secondary：`labels_secondary_v1.jsonl`，64/64，SHA256
  `271ca58e5ffcfd99000c2ff035059f00ff16d5df5de6c9d2cd95dcbb3fa23d1a`；逐条 durable prompt/helper 保留为
  `secondary_prompt_resumable_v1a.md` 与 `scripts/checkpoint_dual_prior_secondary_v1.py`；
- agreement 与裁决：`agreement_report_v1.json`、`adjudications_v1.jsonl`、`adjudication_report_v1.json`；48/48
  disagreement 裁决完成，adjudication SHA256
  `86fbbe0a5e75325b1d0bee23c39fc6e08383fc28bc079587770c6c68539d4e96`；
- exact-token gold：`gold_materialization_protocol_v1.json`、`labels_gold_v1.jsonl`、`gold_report_v1.json`、
  `gold_semantic_audit_v1.json`；gold SHA256
  `da41e1e3061bb7ce321d12211fd5350f116d9510357ed774c7347c214e10cbdc`；
- matched data/protocol：`training_data_protocol_v1.json`、`training_data_report_v1.json`、
  `training_protocol_v1.json`；
- 机器结论：`training_result_v1.json`，SHA256
  `7b11aaf92d5ed6b9d7a3c99cd34c039077e295855b2305efc5a9b4f53595801b`；12 cells 在
  `run_artifacts/dual_prior_evidence_v1/pilot_v1/`；
- 设计与代码审计：`docs/dual_prior_evidence_pilot_v1.md`。

### Dual-prior original mutual distillation v1

- 冻结协议：`configs/dual_prior_mutual_distillation_v1/training_protocol_v1.json`；
- 公式：`MSE(A_key, stopgrad(A_complete)) + MSE(A_complete, stopgrad(A_key))`，完整 trajectory attention，
  joint phase，权重 `.25`；
- 两个 cell：M0 direct BCE control、M1 只增加原始 mutual MSE；均从头训练，3 seeds × 5 epochs；
- 新 evaluator 同时发布 membership localization 与数值匹配训练公式的 symmetric attention MSE；
- 机器结果：`configs/dual_prior_mutual_distillation_v1/training_result_v1.json`，SHA256
  `03c5c3e799f5c4ee7be58be5f6af3a3a89d3248410d8e685efd4da6881e7d11e`；6/6 cells 位于
  `run_artifacts/dual_prior_mutual_distillation_v1/pilot_v1/`；
- 裁决：`completed_pass_original_mutual_distillation`；完整说明见
  `docs/dual_prior_mutual_distillation_pilot_v1.md`。

### Dual-prior reward-gate integration v1

- 量级审计：`configs/dual_prior_reward_gate_v1/loss_scale_audit_v1.json`；默认 `.25` 的 gate-head gradient
  ratio 中位数只有 `.0060`，按预先规则冻结唯一权重 `10.0`，未读取任何 G1 结果；
- 冻结协议：`configs/dual_prior_reward_gate_v1/training_protocol_v1.json`，SHA256
  `46bf816ab55d0f2ba28e63bef83ab820732bd0ad045db4853a25010fb587e751`；
- G0/G1 均保留 direct BCE + 原 mutual `.25`，只改变 gate alignment `0→10`，3 seeds × 5 epochs；
- 机器结果：`configs/dual_prior_reward_gate_v1/training_result_v1.json`，SHA256
  `bc94e06516c31a5f616f6ea6bb91ec67149bff8e052af43d85a69e7965c67c9b`；6/6 cells 位于
  `run_artifacts/dual_prior_reward_gate_v1/pilot_v1/`；
- 裁决：`completed_reward_gate_integration_diagnostic_only`。gate alignment 3/3 有效，但 key-localization
  protection 只有 1/3 通过；完整说明见 `docs/dual_prior_reward_gate_pilot_v1.md`。

### Dual-prior original shared-gradient scale v2

- 数据协议：`configs/dual_prior_original_scale_v2/data_protocol_v2.json`；物化报告
  `data_report_v2.json`；mixed train SHA256
  `3c1ac6068343328ef23d85e380985be641c564ce29d4492e6e95b1526d97f310`；
- 训练数据为 496 queries / 3968 candidates，其中 48 rows 有 adjudicated key+complete Gold；其余 prior
  fields 缺失而非伪造全零；16 个 localization-dev query 的 128 rows 全部从 outcome train 排除；
- ranking validation 复用最新 checker-v5 的 500×16 pool，含 146 mixed queries，与 train/prior-dev
  query-disjoint；
- 训练协议：`configs/dual_prior_original_scale_v2/training_protocol_v2.json`；G0 为 correctness + direct +
  mutual `.25`，G1 只增加原始 shared-gradient gate weight `10`；head-only/containment/reconstruction 均关闭；
- 执行入口：`scripts/run_dual_prior_matrix_v1.py --protocol
  configs/dual_prior_original_scale_v2/training_protocol_v2.json --gpus 0,1,2,3,4,5 --execute`；
- 当前状态：`frozen_before_training`；6 cells 尚未启动，结果路径预留为
  `configs/dual_prior_original_scale_v2/training_result_v2.json`；完整设计见
  `docs/dual_prior_original_scale_v2.md`。

v2 继承的 v1 labels/split 是来源 artifact，不是当前训练方案；其 hash 与 lineage 已记录在 v2
protocol/audit 中，不在本 handoff 重复展开。

## 4. 下一步

不要再重跑 v2c、扫描 tail weight、重跑 M0/M1，或放宽已失败 v1 G1 的 key-AP 门槛。下一步执行已经冻结的
original-scale v2：

1. 先跑 G0/G1 × seeds 42/43/44；两格唯一 loss 差异是原始 shared-gradient gate alignment `0→10`；
2. 每格在 48-row prior-train 与 16-row prior-dev 发布 localization/gate diagnostics；
3. 每格在同一 checker-v5 500×16 validation 发布 k=`1/2/4/8/16` 的 Best-of-N、random、oracle 与
   within-query pairwise accuracy；
4. primary comparison 是 query-paired 的 G1−G0 BoN@16，先在 seed 内配对，再在 query 内平均三 seed 后
   bootstrap；
5. 无论结果正负都不回写 v1；负结果也不自动授权 head-only，只能在新协议下诊断 supervision scale 或优化。

当前 G1 的 key-AP 损失仍是 shared-representation interference 的机制假设，不是已证明因果。用户已选择先
保证项目原方法得到直接训练与 ranking 检验。containment 不替代相互蒸馏，reconstruction 仍等待独立
768-d target。

reconstruction 仍要求独立 768-d target，禁止 same-candidate pooling。exact onset 与下一代 clean-matched tail
repair 留在 backlog，不阻塞当前 collaboration comparison。

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
- Route A v1a、Localization v1/v2/v2b/v2c/v2d、Dual-prior direct-target/mutual-distillation/reward-gate v1 都是
  `pipeline pilot`。
- verifier selection 的 Mistral-24B 只获 Silver pilot 授权，不能自动迁移为 hallucination Gold。
- T2 的 AP/ranking signal 在 v2c 仍存在，但 absolute-margin implementation 的 tail locality 0/3 seed 通过；
  R1 虽解决统一 shift，却没有 clean anchor，并明显损伤当前 sparse AP。只能选择 T0 并暂缓这两个实现，
  不能写成 tail hypothesis 永久失败。exact onset 明确未通过。
- 不能宣称 consistency、hallucination localization 或 tail shaping 已改善 Best-of-N，也不能说 tail 已被
  永久证伪或已获 mixed-training 授权。Dual-prior 当前只能宣称 direct targets 可学、原始 mutual objective
  在 held-out 上有效降低 branch discrepancy，以及首个 gate objective 可被优化；项目已选择原始 gate 接法
  进入 v2，但在 6-cell ranking 结果完成前仍不能宣称最终排序增益。

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
docs/dual_prior_evidence_pilot_v1.md              dual-prior gold、D0–D3 与下一门
docs/dual_prior_mutual_distillation_pilot_v1.md   原始 mutual distillation 三种子结果
docs/dual_prior_reward_gate_pilot_v1.md           shared-gradient gate 三种子结果与 head-only repair
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
scripts/run_dual_prior_matrix_v1.py                   D0–D3 多 GPU launcher
scripts/summarize_dual_prior_pilot_v1.py              direct-target 三种子采用门
train_clir.py / score_clir.py / evaluate_clir.py     训练、打分、Best-of-N
```
