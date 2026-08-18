# Hallucination Localization Pilot v1

状态：历史基线；双标、裁决、supervision merge 和 H0–H3 已完成，冻结为
`completed_path_signal_onset_gate_failed`。当前方案与后续结果见
`docs/hallucination_localization_pilot_v2.md`。

日期：2026-08-18

证据等级：`pipeline pilot`

本文定义 semantics consistency Pilot-0 之后的下一个机制模块：hallucination localization。
Route A v1a 已完成双标、裁决、4096 行 mixed consistency manifest 和匹配训练门；旧 62 行 manifest
仍禁止训练。本模块已经冻结 selection/annotation protocol，完成 primary、blind secondary、裁决和
matched dense training。
首轮目标不是证明 CLIR 提升 Best-of-N，而是用一批可审计的真实标签跑通
`path_hallucinated`、`hallucination_onset`、token localization loss 和 negative-tail reward
的最小闭环。

## 1. 为什么现在做这个模块

on-policy equivalence mining 已经证明 consistency 数据、真实 Phi hidden states 和训练接口能够闭环。
v1a 修复训练进一步显示，现有 consistency loss 主要分离跨语义表示，而不是提高已饱和的正对 cosine。
继续扩大 consistency 数据之前，
应独立验证 CLIR 的第二个核心假设：轨迹在首个不受支持的推理主张之前仍可能有价值，之后的表示应被
识别并降权。

本轮开始时冻结 train/validation manifest 的 hallucination 监督覆盖为 0；v1 现已给 4096-row base
train 中的 64 行绑定真实 pilot 标签，并在 query-disjoint 48/16 dense split 上完成训练。correctness
仍只描述最终答案，未用于派生这些监督。

## 2. 任务定义与边界

### 2.1 领域通用定义

标注单位是一个模型输出 trajectory。将输出按可判定的 atomic claims / reasoning steps 阅读，定义：

```text
hallucination_onset = 最早一个 unsupported 或 contradicted claim 所对应的第一个 output token
```

其中 unsupported / contradicted 包括：

- 与题目、给定 context、已建立事实或可验证外部证据冲突；
- 引入完成当前推理所需、但输入与此前推导均未提供的事实；
- 计算、逻辑或因果推断的结论不由其前提推出。

以下情况本身不构成 hallucination：

- 为解释清楚而补齐 source 中省略、但可以由输入或此前步骤推出的中间推导；
- 合法的等价改写、不同解题路径或不同表达风格；
- 只改变冗长度而不引入错误主张；
- 不完整、拒答或格式错误，但没有作出 unsupported claim。

因此 correctness 与 hallucination 必须独立标注：正确答案可能包含早先错误后又修正；错误答案也可能
只是未完成、误读输出格式或缺少最终答案，而没有可定位的 hallucinated claim。

### 2.2 路径标签、onset 与尾部状态

- `path_hallucinated=1`：至少存在一个可定位的 unsupported / contradicted claim；
- `path_hallucinated=0`：所有实质性 claim 都可支持；
- 不确定、证据不足或 claim 边界无法可靠映射时保持 unlabeled，不强行写 0；
- 只有 `path_hallucinated=1` 时才写非负 `hallucination_onset`；
- 明确无 hallucination 时写 `hallucination_onset=-1`；
- onset 使用保存的 `output_token_ids` 的零基索引，禁止对 response 重新分词后估算。

当前实现把 onset 后所有 token 监督为 1。严格说这预测的是“首错之后的 contaminated tail”，不是声称
尾部每一个 token 都独立 hallucinated。论文、报告和字段解释必须维持这个区分。

## 3. Pilot-0 数据设计

只从 `train-primary` 取样，不读取 `pilot_test` 或 `final_test`。首版冻结 64 条 Phi 原生 trajectory：

- 32 条 correctness=1，32 条 correctness=0；
- 尽量一题一条并覆盖不同长度分位；
- correctness 只用于分层抽样，不进入 hallucination 判据；
- 不使用 Qwen rewrite，也不要求错误 trajectory 做 rewrite；
- Route A/Route B augmentation row 不继承 source 的 path/onset 标签，除非该 view 被独立重标。

64 条只用于定义校准和工程闭环。若 development-32 的 19 条错误样本不足，剩余错误样本从冻结的
train-primary manifest 中按确定性 seed 采样。发布候选清单时记录 row ID、query ID、token-ID hash、
correctness stratum、长度、源 manifest/hash、抽样 seed 与代码 commit。

### 3.1 已冻结的 v1 selection

机器协议是 `configs/hallucination_localization_v1/protocol_v1.json`，SHA256
`f79ea5b2dd92ca72bfb8b4d0878f952bbad0e95d1891bdc4a8d040515a231252`。seed 1729 从 4096 行
train-primary 中选择 32 correct/32 incorrect；每类按 frozen output-token 长度分 4 个 rank bin，每 bin
8 条，最终 64 个不同 query。correctness 只出现在 private selection/lineage，不进入 blind item。为避免
围绕已反复查看的样本调定义，额外排除 Route A v1a 重新审核的全部 31 个 query。

关键产物：

- `selection_manifest_v1.jsonl`：SHA256
  `bbc4e4503af1abada3a7df3e7cfa9a0891ec796539e19408bf78f3df64fc88df`；
- `annotation_items_v1.jsonl`：SHA256
  `ec7ebe67794810300a8d9ca984ea7f29c5e1017f2e1da0b81118dc2789a38591`，仅含
  schema/item/problem/trajectory；
- `annotation_lineage_v1.jsonl`：SHA256
  `ad786d30c08a2d4112018893fecfcee7a32e13fc22d925bae12334ba3d95faef`，私有绑定
  source/query/correctness/token hash；
- `token_mapping_preflight_v1.json`：SHA256
  `f7e0cb816e7ee62e5dcb18611c637c9a002fc3607e8cd76bc6af9ded0050f1dc`。

Phi fast tokenizer preflight 对 64/64 行验证：visible response 的无 special-token 编码与冻结
`output_token_ids` 精确前缀相等，完整 decode 与 response 相等，offset 覆盖完整；每行恰好尾随一个
decode 后不可见的 `[32007]` control token。因此后续由 exact quote 映射 onset，不接受重新分词估算。

## 4. 标注与裁决

### 4.1 标注器角色

几十 B 模型可以承担 primary annotation，但必须输出结构化证据：

1. atomic claim/step 边界；
2. 每步 `supported / contradicted / unsupported / non-claim / uncertain`；
3. 最早问题 claim 的原文字符区间与理由；
4. path decision；
5. 映射到 exact output token 后的 onset。

已有 `Mistral-Small-24B-Instruct-2501` 只在 reasoning-equivalence selection set 上获得 Silver
授权。它可以作为本任务的首个候选标注器，但该授权不能迁移；必须在新的 hallucination/onset
calibration set 上单独评价。

### 4.2 双标与 Gold/Silver 边界

- 64 条全部由 primary annotator 标注；
- 其中至少 32 条由独立第二标注者盲标，保持候选顺序与 correctness 信息隐藏；
- 分歧由单独 adjudication 文件裁决，不覆盖两份原始标签；
- path agreement 报 Cohen's kappa、正负类 confusion；
- 双方均判 positive 的样本报告 onset exact match、token distance，以及容差
  `±1/±3/±5` token agreement；
- unresolved/uncertain 保持缺失，不进入训练 mask。

未经上述校准的模型标签只能称为 Silver。只有人工或预注册裁决后的共识标签才能称为 Gold。

### 4.3 Candidate-primary 实际结果

冻结的 Mistral-24B 在 clean commit `5fabe9b63258950e6696d067a2e4ade8222102c4` 上以 greedy decoding
处理全部 64 条。原始 artifact 原样保留：60 条通过完整 annotation schema，4 条失败；已通过的类别为
42 clean / 18 hallucinated。4 个失败均不是缺少语义判断：3 条把 trajectory 内换行压成空格，导致
exact quote gate 拒绝；1 条将 8 个 claim 的最后一个索引写成 8，而不是零基索引 7。

commit `0b038af3c607580e9e44ad48b9d94c14cb03f91d` 新增受限合同修复器。它只允许：

- non-whitespace 字符完全相同时，把模型 quote 重新复制为 trajectory 的 exact substring；
- 从原有 claim statuses 派生第一个 `contradicted/unsupported` 的零基索引。

它禁止模糊文本匹配，也不能修改 `path_status`、claim status/reason、confidence 或 summary；修复前后
逐行 semantic decision signature 必须相同。正式运行修复 4 行、10 个 whitespace quote 和 1 个索引，
最终 64/64 annotation/token map 有效，得到 45 clean / 19 hallucinated。onset 的 output-token index
最小 34、最大 337、均值 164.42。private correctness/path 交叉表是：incorrect 16 clean / 16
hallucinated；correct 29 clean / 3 hallucinated。该交叉表只作“不等同于 correctness”的诊断，不作为
标注准确率证据。primary 64 条置信度均为 high，反而说明需要独立标注校准其过度自信。

版本化结果：

- `labels_primary_v1.jsonl`：SHA256
  `1b8129982c15b2b948e4e3ec19fb1ce7da979044014673337b4cfe46a08959f7`；
- `primary_report_v1.json`：SHA256
  `daad3a931496cf79d1b362a3eea0c429a1ee53daefe79d2b9e9545cca4522317`；
- `secondary_items_v1.jsonl`：SHA256
  `ec7ebe67794810300a8d9ca984ea7f29c5e1017f2e1da0b81118dc2789a38591`，与 primary blind items
  byte-identical；
- `secondary_prompt_v1.md`：SHA256
  `17cf885c1bf1852c1c05ce543d1789eb550e41ef6955af42bfe8d2cab5f2e4b4`。

这些 primary 结果仅作为被比较的原始标注保留，未被覆盖或直接当作最终训练标签。

### 4.4 Secondary、agreement 与裁决结果

外部提供的 blind secondary raw labels SHA256 为
`cbc5599a82a928fb3d4433645ffb5101e136659cbc180dd9ecfe4c47e55eb0a6`，64/64 通过 structure-only、
exact-substring validation。映射到同一冻结 Phi token 后：

- primary 45 clean / 19 hallucinated；secondary 41 clean / 23 hallucinated；
- path agreement `52/64 = 0.8125`，Cohen's kappa `0.576626`；
- confusion 为 clean→clean 37、clean→hallucinated 8、hallucinated→clean 4、
  hallucinated→hallucinated 15；
- 15 个共同 positive 的 onset exact/`±1/±3/±5` 都是 `5/15`；median absolute distance `67`、
  mean `55.8`、max `217` tokens。

训练阻塞口径只裁 path 冲突和共同-positive onset-token 冲突；claim span 粗细或
unsupported/contradicted 细分类差异报告但不阻塞。22-row A/B package 经内部盲审得到 A 10、B 8、
revised onset 4，最终 41 clean / 23 hallucinated。裁决者不是独立人工 reviewer，并在协议中披露了
aggregate cross-tab 与 post-decision 两个 role mapping 的可见性限制，因此最终标签只称
`not_gold_internal_blinded_pipeline_pilot`。

关键产物：

- `agreement_report_v1.json`：原始 agreement 与 onset 距离；
- `adjudication_resolutions_v1.jsonl`：22 条 compact decision；
- `labels_adjudicated_v1.jsonl`：SHA256
  `00ec837086a76f042766f1b59821783a5aa9c83810634121a362715c5af9281d`；
- `supervision_annotations_v1.jsonl`：SHA256
  `2447aef866f4a8693d03bd5bdeef7381d5254485966de3ff4f7d4fc0384b72cb`；
- `adjudication_report_v1.json`：64/64 exact token identity，trainability gate passed。

## 5. 数据产物与硬门

计划新增、且全部版本化保留的产物：

```text
configs/hallucination_localization_v1/protocol_v1.json
configs/hallucination_localization_v1/selection_manifest_v1.jsonl
configs/hallucination_localization_v1/annotation_items_v1.jsonl
configs/hallucination_localization_v1/annotation_lineage_v1.jsonl
configs/hallucination_localization_v1/labels_primary_v1.jsonl
configs/hallucination_localization_v1/labels_secondary_raw_v1.jsonl
configs/hallucination_localization_v1/labels_adjudicated_v1.jsonl
configs/hallucination_localization_v1/supervision_annotations_v1.jsonl
configs/hallucination_localization_v1/training_split_manifest_v1.jsonl
configs/hallucination_localization_v1/training_result_v1.json
run_artifacts/hallucination_localization_v1/merge_report_v1.json
run_artifacts/hallucination_localization_v1/coverage_report_v1.json
```

每条可训练 annotation 必须满足 `docs/clir_supervision_protocol.md`：绑定 `id`、`query_id` 和
`SHA256(canonical_json(output_token_ids))`，并记录 protocol SHA256 与 annotator/version。发布前必须通过
`scripts/merge_clir_supervision.py` 和 `scripts/audit_clir_supervision.py`。

进入训练的最低门槛：

- 64 条均成功解析，token 映射无越界；
- path 两类都非空；
- 至少 12 条裁决后的 positive onset；
- 至少 12 条明确 negative；
- 双标 path agreement 与 onset 容差指标完整报告；
- 任一 augmentation row 均未静默继承 source 的 localization 标签；
- 人工抽查确认“补齐合法推导”未被系统性误判为 hallucination。

最后一项人工抽查未发生；按用户“先达到可训练、门禁后续再加强”的决定，本轮用内部盲审替代并把证据
严格降为 `pipeline_pilot/not_gold`，只放行 dense engineering run，不满足 Gold/formal gate。其余可训练性
门通过，但 onset 分歧和后续 held-out 结果均未过定位门，所以 v1 停在 onset repair，不进入
pseudo-tail 或 mixed-data 机制训练。

## 6. 最小训练矩阵

所有 cell 使用同一批 trajectory、Phi features、query-disjoint split、初始化与训练预算。Pilot-0 先跑：

| Cell | 监督 | 目的 |
|---|---|---|
| H0 | correctness only | 同数据基线 |
| H1 | correctness + strong token BCE | 验证 onset/tail 分类信号 |
| H2 | H1 + negative-tail margin | 验证 localization 是否真正影响 token value |
| H3 | correctness + path MIL | 弱标签可用性诊断 |

首轮显式设置 `pseudo_tail_weight=0`。当前 pseudo-tail 是由尚未校准的 hallucination head 自己预测 onset，
若从训练开始即启用，会形成循环自训练 shortcut。只有 H1/H2 在 held-out calibration 上形成稳定 onset
信号后，才能发布新协议增加 `H3 + pseudo-tail` cell。

首轮也不同时加入 consistency 或 dual-prior loss；否则无法归因 localization 是否提供信号。现有
`token_values = token_rewards + progress_score_weight * progress` 的耦合保持为开放问题。本 pilot 没有
progress 标签时不据此修改架构。

### 6.1 v1 实际 split、训练和结果

最终 64 条按冻结 hash priority 划分为 query-disjoint 48 train / 16 dev：train 31 clean / 17
hallucinated，dev 10 clean / 6 hallucinated。唯一 correct+hallucinated row 留在 train；dev 另含 3 个
incorrect-clean 和 6 个 incorrect-hallucinated，可做 correctness shortcut 诊断。完整 base manifest 另生成
排除全部 16 个 dev query 的 3968-row mixed 版本，但 v1 未授权使用它。

H0–H3 均以 seed 42、5 epochs、batch 4、同一 all-33-layer precomputed features 完成；每 epoch 覆盖
48 path labels 和 14,307 onset-supervised tokens，无 non-finite。冻结 `0.5` threshold 下的主要 dev
结果：

| cell | path AUROC / AP | incorrect-only path AUROC | token AUROC / AP | onset MAE | onset ±5 |
|---|---:|---:|---:|---:|---:|
| H0 | .533 / .448 | .333 | .431 / .252 | 87.5 | 0/6 |
| H1 | .933 / .873 | .778 | .710 / .461 | 134.2 | 0/6 |
| H2 | .933 / .873 | .778 | .732 / .497 | 132.7 | 0/6 |
| H3 | .900 / .800 | .722 | .547 / .351 | 343.5 | 0/6 |

shortcut baselines 是 path length AUROC `.700`、incorrectness AUROC `.850`、incorrect-only length AUROC
`.556`，以及 absolute token position AUROC/AP `.699/.514`。因此 H1/H2 的 path ranking 超过这些
shortcut，且 train-only path threshold 在 dev 得到 balanced accuracy `.783`；但 token AP 未超过位置
baseline，任何 cell 都没有一个 onset 落在 `±5`。train-only threshold calibration 同样不能修复 onset，
并且 post-hoc calibration 不替代冻结阈值结果。

H2 将 dev tail margin violation 降为 `0%`，但 mean token value 同时变成 clean `-2.55`、pre-onset
`-3.84`、tail `-4.33`。tail 相对更负，却不是局部下调，故 negative-tail locality 不通过。

机器可读结论在 `training_result_v1.json`：engineering pipeline passed、path ranking promising；
`onset_localization_gate_passed=false`、`negative_tail_locality_gate_passed=false`、
`authorize_pseudo_tail=false`、`authorize_mixed_3968_row_mechanism_run=false`。

## 7. 指标与结论边界

主要工程/定位指标：

- path AUROC、AUPRC、balanced accuracy；
- contaminated-tail token F1 / AUPRC；
- positive rows 的 onset absolute token error 与 `±1/±3/±5` accuracy；
- onset 前后 token value 均值及 margin violation rate；
- 每项 loss 的 applicable row/token counts。

探索性任务指标：同一冻结候选池上的 BoN accuracy、correct-vs-incorrect pairwise ranking 和 score
population std。64 条 pilot 的 BoN 变化不能作为方法效果结论，也不据此选择最终 test 协议。

## 8. 执行顺序与停止点

1. 冻结 64-row selection manifest 与 annotation protocol；
2. 运行 primary annotation，并完成 exact-token span mapping；
3. 导出不含 primary decision 的 blind secondary prompt/package；
4. 收到第二份标签后计算 agreement、裁决并审计；已完成；
5. 数据硬门通过后运行 H0-H3；已完成；
6. 根据 held-out localization 结果决定是否扩大标签、启用 pseudo-tail，或回到定位设计；已裁决为
   扩大/修复 onset，禁止 pseudo-tail。

v1 在此冻结。下一轮必须新建 onset Pilot v2：先不改 loss，扩大 positive onset 标签，并把绝对位置和
归一化位置 baseline 设为硬门；若相同 token BCE 仍不能超过位置 baseline 或形成 onset `±5` 命中，再
单独讨论 claim-boundary objective。也可以先探索独立 dual-prior pipeline，但不能与当前失败分支混训。

## 9. 当前未验证事项

- path signal 在更大、更独立 dev 上能否继续超过 correctness/length shortcuts；
- 增加 positive onset 后，相同 token BCE 能否超过 position baseline；
- claim-boundary onset 是否需要显式结构或 objective，而不是逐 token tail BCE；
- noisy-or 的长度校准如何避免 float32 path probability 饱和；
- tail shaping 如何只下调 onset 后 token，而不整体移动整条错误 trajectory。

本文不授权 mixed 3968-row mechanism run、pseudo-tail、读取测试集或宣称 hallucination localization
已经有效；只允许把 path ranking 称为 promising pipeline diagnostic。
