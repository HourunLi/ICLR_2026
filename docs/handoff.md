# CLIR 当前交接

最后更新：2026-08-21

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
- `training_protocol_v2.json` 冻结的 G0/G1 × seeds 42/43/44 已 6/6 完成，训练 commit 为 `ee549af`；每格
  5 epochs、全 33 层，随后完成 48/16 localization 与独立 500×16 ranking；
- G1−G0 BoN@16 三 seed 为 `-.008/-.010/-.008`，mean `-.00867`；aggregate 500-query paired bootstrap
  95% interval `[-.01933,+.00200]`，0/3 seed 正向，因此没有建立稳定 ranking 增益，也未证明稳定负效应；
- G1 的 gate-objective MSE 平均下降 `.00643`，complete/key unit AP 平均提高 `.02150/.03084`，说明原
  shared-gradient objective 被优化，但当前 localization 改善未转化成 ranking 收益；
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
  否证；pseudo-tail 仍未获授权。用户已另行批准把 T0 作为稀疏监督接入 3968-row 联合训练 pilot；这不推翻
  tail 的历史裁决，也不把 sparse span 标签解释成 full-tail ground truth。pilot/final test 未读。

为避免 H 再通过 shared representation 改坏 JP，`JP→H frozen linear probe v1` 已从结果前冻结的 clean
commit `17d0efa` 完成：

- JP epoch-5 checkpoint 只读；抽取前后及 12-cell 结束后的 SHA256 均为
  `a392ad6ad8ccbe020e6a10835463c15829e3684f0a2b9213e6bcefe78666df5c`；
- 当前 forward 对 64/64 reward scores 和全部 token values 与 retained JP scoring bit-identical；每个 probe
  仅为独立 `Linear(768,1)`，769 参数，base optimizer parameters 为 0；
- 4 folds × seeds 42/43/44 的 12/12 cells 完成；主要 folds 1–3 的 48 条 OOF 上，claim AP
  `.358/.341/.325` 全部超过同批位置基线 `.305`，span AP `.236/.233/.234` 全部低于 `.260`；
- 冻结双门 0/3 通过，状态为 `completed_frozen_linear_probe_not_supported`，不能直接接 reward；
- 事后机制诊断显示所有 supervised tokens 都由 claim spans 覆盖，annotated-claim oracle 内取均值可把 span AP 提至
  `.384/.369/.352`，3-token local mean 也提至 `.280/.273/.269`。这定位为 claim 内高频抖动，但属于
  post-hoc，不得追溯改判或在同一 64 条上直接采用窗口；
- 完整协议、结果和下一门见 `docs/jp_h_frozen_probe_v1.md` 与
  `configs/jp_h_frozen_probe_v1/training_result_v1.json`。

固定 `3-token` logit smoother 的后续工程 pilot 也已完成：

- 协议与实现先冻结于 clean commit `9ad4268`；本轮直接复用 12 个不可变 OOF prediction files 和 head
  checkpoints，不重训、不新增参数；
- 192/192 个 row views 除 H 概率/path 派生字段与新增 smoother provenance 外逐字段完全一致，JP reward、
  token values、原 direct/mutual/gate 均未改变；
- 主要 48 条 OOF 上，span AP 从 `.236/.233/.234` 提至 `.280/.272/.268`，claim AP 从
  `.358/.341/.325` 提至 `.366/.364/.332`；三个 seed 都同时超过 position 与各自 raw linear 的
  span/claim 四门；
- 状态为 `completed_engineering_signal_supported`，但 window=3 是看过同一 64 条的 post-hoc diagnostics 后
  选出的，因此 `method_adopted=false`、`score_coupling_authorized=false`；
- all-64 cross-fold diagnostic 虽比 raw linear 改善，仍低于 position，提示跨 head/domain calibration 风险；
- 完整记录见 `docs/jp_h_temporal_smoother_v1.md` 与
  `configs/jp_h_temporal_smoother_v1/result_v1.json`。

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
4. Localization 标签当前选择 T0/S1 sparse span。联合阶段不再让 H 更新 JP shared path：plain frozen
   linear H probe 已确认 claim-level signal、但 token/span 门失败；固定 3-token masked centered logit
   smoother 已获 3/3-seed 工程支持，作为新盲标 validation 的唯一候选，尚未采用且未接 reward。
   absolute-margin T2、pre-onset-relative R1、exact onset 与 clean-matched tail repair 都保持独立，不与
   smoother 混跑。
5. Dual-prior 保留项目原方法：direct key/complete targets、symmetric stop-gradient mutual MSE，以及
   gate-to-detached-fused-prior 的 shared-gradient alignment。v1 的 key-AP 保护门失败按历史原样保留，但当前
   不换 head-only；先用 v2 mixed outcome data 和 query-grouped ranking 判断原方法的任务效果。containment
   不替代 mutual，reconstruction 继续关闭且仍只接受独立 768-d target。
6. Progress 在 primary 中暂时完全移出 reward：显式使用 `progress_weight=0` 与
   `progress_score_weight=0`。原 head 不删除；稳定版 main 的 `progress_score_weight=.5` 只保留为 matched
   control。在没有独立 verifier prefix-delta 等真实 `progress_targets` 前，不得把该分支输出解释为
   局部推理进度。
7. Complete reconstruction 已明确暂时关闭：primary 必须显式使用
   `reconstruction_weight=0`。这是推迟而非删除；只有独立、冻结、宽度为 `model_dim=768` 的完整证据/
   答案 target 经新版数据与训练协议发布后才能重开。same-candidate pooled feature 永久不能作为
   reconstruction target。
8. 当前联合训练采用一个 manifest、一个 sampler、一个 optimizer，从同一初始化联合更新；禁止把三个已训练
   checkpoint 拼接。每行每 epoch 一次，不对 54/48 条辅助样本 oversample；缺失监督用 mask 跳过。

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
- 当前状态：`completed_original_shared_gradient_scale_and_ranking`；6/6 cells 完成，机器结果为
  `configs/dual_prior_original_scale_v2/training_result_v2.json`；完整设计与结果见
  `docs/dual_prior_original_scale_v2.md`。

v2 继承的 v1 labels/split 是来源 artifact，不是当前训练方案；其 hash 与 lineage 已记录在 v2
protocol/audit 中，不在本 handoff 重复展开。

### Joint training pilot v1

- 数据协议：`configs/joint_training_pilot_v1/data_protocol_v1.json`；统一 train 固定为 3968 rows / 496
  queries，其中 3866 correctness-only、54 correctness+consistency、48
  correctness+sparse-hallucination+dual-prior；54 与 48 行零重叠；
- mechanism train/dev 为相同机制标签的 48/16 query-disjoint rows；ranking validation 固定为 500×16，三者
  query overlap 均为 0；
- sampler 固定 batch 4、每 epoch 992 batches、每行恰好一次；Route-A 每 epoch 产生 27 positive pairs 与
  26 same-style negative pairs，不做辅助 oversampling；
- 三格固定为 `j0_correctness`、`jp_original_prior`、`jall_full_retained`，seed 42、5 epochs、BF16、LR
  `1e-4`、final epoch 5；原始 prior 为 direct key/complete + mutual `.25` + shared-gradient gate `10`；
- JALL 只启用 consistency `1`、unweighted sparse hallucination BCE `1` 和 prior outer `1`。MIL、所有 tail、
  token reward、progress、reconstruction 均为 0，`progress_score_weight=0`；
- 训练前必须通过真实 feature 的 no-update gradient routing audit；日志同时保留全 epoch coverage-weighted loss、
  active-supervision batch mean、active batch 数和精确 row/pair/token count；
- 结果等级固定为 `small-scale real integration pilot`。Consistency 没有 held-out relation，只报告训练关系几何；
  seed 42 通过只会授权另发 seeds 43/44 协议；实际结果没有通过。
- 三格均从 clean commit `5a0b5d3` 完成 5 epochs；无 OOM/NaN/运行时失败，真实 feature 梯度路由审计、每
  epoch 3968 rows、27/26 consistency pairs、6681 sparse tokens、14307 prior units 全部匹配冻结协议；
- J0/JP/JALL 的 BoN@16 为 `.920/.918/.912`；JALL 相对两个 control 的回退都在 `.02` 内，consistency
  training-relation gap 为 `.789`，complete AP `.931`；
- 扩展门失败项为：H span AP `.272 < .393` position、H claim AP `.289 < .422` position、key AP
  `.314` 相对 JP `.432` 回退 `.118 > .05`。状态固定为 `completed_seed42_expansion_gates_failed`，seeds
  43/44 未授权；
- standalone S1 在相同 48/16 mechanism split 上曾达到 span/claim AP `.416/.464`，而 JALL active H BCE
  五 epoch 始终高于 standalone；同时 JALL prior 拟合与 key AP 弱于 JP。因此这是联合优化/混合问题的信号，
  不能归因为“代码没接上”或“标签必然无效”；
- 当前 JALL 相对 JP 同时增加 H 与 consistency，不能识别谁造成回退。完整机器结果、loss 轨迹、batching
  confound 与因果边界见 `configs/joint_training_pilot_v1/failure_diagnostic_v1.json` 和
  `docs/joint_training_pilot_v1.md`。
- drop-one JPH/JPC 已从 clean commit `55155d2` 完成。JPH/JPC 的 BoN@16 为 `.920/.918`，相对 JP
  `+.002/.000`，均通过 ranking 保护门；key AP 为 `.314/.275`，相对 JP 分别下降 `.118/.157`，两格都
  复现预注册 key drop。因此在当前 seed-42 条件下，H 和 C 各自都已足够，不是只有 H×C 交互才会失败；
- JPH 的 H span/claim AP 为 `.319/.338`，相对 JALL 恢复约 `.047/.049`，但仍低于 `.393/.422` 位置
  基线。JPC 的 H loss 为 0，却得到 `.435/.442`，必须解释为 16-row dev 对共享表示变化敏感，禁止声称 C
  学会了 H localization；
- drop-one 状态为 `completed_seed42_drop_one_diagnosis`，仍不授权多 seed。完整结果和证据边界见
  `docs/joint_training_drop_one_v1.md` 与
  `configs/joint_training_drop_one_v1/training_result_v1.json`。
- clean commit `74f7583` 的 no-update gradient interaction audit 已完成。H↔prior-total 的 init/JP shared
  cosine 为 `-.024/+.126`，C↔prior-total 为 `+.133/+.268`，所以不授权 blanket shared-encoder
  gradient surgery；H↔prior 的 condition branch 局部为稳定负向，保留为后续 diagnostic；
- C↔mutual 为 `-.400/-.536`、C↔H 为 `-.110/-.200`，但 frozen 5-epoch stream 的 C 与 H/prior batch
  overlap 是 `0/0/0/0/0`。它们是 cross-step pressure，不是 simultaneous conflict，普通 per-batch PCGrad
  无法处理；
- JP 状态 H/C shared norm 为 `1.250/1.713`，prior-total 仅 `.536`。这支持先查 late-stage
  schedule/effective budget，而不是删除或替换原 mutual/gate。完整结果见
  `docs/joint_gradient_interaction_v1.md` 与
  `configs/joint_gradient_interaction_v1/audit_result_v1.json`。
- JPH supervision-aware packing v1 已从 clean commit `950f5c4` 完成。静态与运行审计都确认每 epoch
  992 total batches、12 个纯 4-row mechanism batches、27/26 consistency pairs，且每行一次；
  loss/label/model/direct/mutual/gate 全部未改。结果 BoN@16 `.924`、H span/claim AP `.206/.448`、
  key/complete AP `.096/.429`；因 key 和 H-span 冻结门失败，分类为
  `packing_schedule_not_supported_at_frozen_gates_seed42`，不扩 seeds。
- 这一负结果校正了对 main 的理解：semantic sampler 为必须共批的 pairwise consistency 服务；
  H/prior 是 per-row/per-token objective，直接打包没有新增 relational loss，只把 auxiliary-active
  optimizer opportunities 从约 48 次压缩为 12 次。packing 因此不进入当前保留方案；完整记录见
  `docs/joint_training_packing_v1.md` 和 `configs/joint_training_packing_v1/training_result_v1.json`。
- `H-condition-stopgrad` 已完成全 48-row、2-state no-update 路由审计。实现不 detach 整个
  conditioned feature，而是以 detached weights 重算 H branch，因而只阻断 H BCE→
  `condition_query/key/value + condition_fusion` 参数梯度；input encoder、`feature_norm`、H head 梯度保留。
  24/24 batches 的 forward/loss/H-nonblocked/final/direct/mutual/gate 差异全为 `0`，blocked 梯度从
  非零变为 `0`；参数 checksum 不变且无 `.grad` buffer。详见
  `docs/joint_condition_routing_v1.md` 和 `configs/joint_condition_routing_v1/audit_result_v1.json`。
- 唯一的 seed-42 `JPH + H-condition-stopgrad` 训练格已从 clean commit `df33e7b` 完成。五轮
  H/prior active batches 为 `[48,47,48,45,45]`，原 sampler、direct targets、双向 mutual `.25`、
  shared-gradient gate `10` 和所有 loss weights 均未改。结果 BoN@16 `.912`、complete AP `.929`
  通过保护门，但 key AP `.337` 只比 JPH 恢复 `.023`，H span/claim AP `.247/.235` 比 JPH
  下降 `.072/.103`；分类为 `condition_route_not_supported_at_frozen_gates_seed42`。完整机器结果见
  `configs/joint_condition_routing_v1/training_result_v1.json`。

## 4. 下一步

不要重跑 v2c、扫描 tail weight、重跑 M0/M1、扩跑 packing/condition-stopgrad seeds，或放宽任何历史失败门。
plain frozen linear probe 的历史失败也不能因 smoother 成功而追溯改判。

下一道阻塞门已经从“实现 smoother”推进为“新增真正未见的 localization validation”：

1. 新 batch 的 row 来源、domain 构成、sample size、split 和 exact-token artifact 必须先冻结；优先加入
   非数学领域，不能只从当前 64 条附近重采样；
2. 先冻结一个单个 final H head 及其训练集/seed/checkpoint，再在新标签未打开前冻结 fixed 3-token centered
   logit smoother；不再扫描 window、概率/日志空间、causal/centered 或 MLP；
3. 主标和独立第二标注者都按 claim span 标注。第二标注 prompt 必须要求每完成一条立即 durable 落盘，
   支持超时续跑；分歧在看模型分数前裁决；
4. 采用门仍同时看 span AP 与 claim-mean AP，并同时对比 position 和 raw linear。跨 head calibration 必须在
   协议中预先解决，不能看新标签后再选；
5. 新盲标门通过前，`method_adopted=false` 且不做 `p_hallucination→reward` coupling；通过后才单独冻结一个
   透明 score-coupling 协议，最多学习非负标量 `alpha`，直接测独立 500×16 ranking；
6. consistency、tail/exact onset/progress/reconstruction 不在该格中重开；不读取 `pilot_test/final_test`。

centered smoother 用到未来一个 token，只适用于完整 trajectory 的离线 scoring。若目标改成生成时实时拦截，
causal smoother 是另一个方法，必须单独冻结和验证。

原始 mutual 与 shared-gradient gate 不因联合或 probe 结果被静默替换；containment/head-only 仍只是历史
候选。reconstruction 继续等待独立 768-d target；exact onset 与下一代 clean-matched tail repair 留在 backlog。

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
- fixed 3-token logit smoother 在同一 64 条上的工程门为 3/3 seed 通过，但结构选择已暴露；它只能作为新盲标
  候选，不能称为已采用的 localization 方法或 Best-of-N 证据。
- verifier selection 的 Mistral-24B 只获 Silver pilot 授权，不能自动迁移为 hallucination Gold。
- T2 的 AP/ranking signal 在 v2c 仍存在，但 absolute-margin implementation 的 tail locality 0/3 seed 通过；
  R1 虽解决统一 shift，却没有 clean anchor，并明显损伤当前 sparse AP。只能选择 T0 并暂缓这两个实现，
  不能写成 tail hypothesis 永久失败。exact onset 明确未通过。
- 不能宣称 consistency、hallucination localization 或 tail shaping 已改善 Best-of-N，也不能说 tail 已被
  永久证伪或已获 mixed-training 授权。Dual-prior 当前可以宣称 direct targets 可学、原始 mutual objective
  在 held-out 上有效降低 branch discrepancy、原始 gate objective 可被优化；original-scale v2 已直接检验
  ranking，但 G1−G0 BoN@16 mean `-.00867` 且 paired CI 跨 0，因此没有排序增益证据。

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
docs/jp_h_frozen_probe_v1.md                      冻结 JP 的 H linear probe 与 temporal-smoother 下一门
docs/jp_h_temporal_smoother_v1.md                 固定 3-token smoother 工程结果与新盲标门
docs/dual_prior_evidence_pilot_v1.md              dual-prior gold、D0–D3 与下一门
docs/dual_prior_mutual_distillation_pilot_v1.md   原始 mutual distillation 三种子结果
docs/dual_prior_reward_gate_pilot_v1.md           shared-gradient gate 三种子结果与 head-only repair
docs/dual_prior_original_scale_v2.md              原方法规模训练、500×16 ranking 与冻结结论
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
scripts/run_jp_h_frozen_probe_v1.py                   只读 JP feature cache 与 4-fold × 3-seed probe
scripts/summarize_jp_h_frozen_probe_v1.py             base bit-identity 与 OOF span/claim 采用门
scripts/run_jp_h_temporal_smoother_v1.py              12-cell 固定 logit smoother 只读重评分
scripts/summarize_jp_h_temporal_smoother_v1.py        paired raw/position 工程门与不变性审计
scripts/run_dual_prior_matrix_v1.py                   D0–D3 多 GPU launcher
scripts/summarize_dual_prior_pilot_v1.py              direct-target 三种子采用门
train_clir.py / score_clir.py / evaluate_clir.py     训练、打分、Best-of-N
```
