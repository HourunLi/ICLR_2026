# Hallucination Localization Pilot v1

状态：设计冻结，尚未生成标签或运行训练

日期：2026-08-17

证据等级：`pipeline pilot`

本文冻结 semantics consistency Pilot-0 之后的下一个机制模块：hallucination localization。
首轮目标不是证明 CLIR 提升 Best-of-N，而是用一批可审计的真实标签跑通
`path_hallucinated`、`hallucination_onset`、token localization loss 和 negative-tail reward
的最小闭环。

## 1. 为什么现在做这个模块

on-policy equivalence mining 已经证明 consistency 数据、真实 Phi hidden states 和训练接口能够闭环，
但小样本诊断也表明 consistency 目标的正负项存在明显权重敏感性。继续扩大 consistency 数据之前，
应独立验证 CLIR 的第二个核心假设：轨迹在首个不受支持的推理主张之前仍可能有价值，之后的表示应被
识别并降权。

当前冻结 train/validation manifest 的 hallucination 监督覆盖仍为 0；现有 loss 只是已经实现，尚无
真实标签证据。correctness 只描述最终答案，不能填补该监督。

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

## 5. 数据产物与硬门

计划新增、且全部版本化保留的产物：

```text
configs/hallucination_localization_v1/selection_manifest.jsonl
configs/hallucination_localization_v1/primary_labels.jsonl
configs/hallucination_localization_v1/secondary_labels.jsonl
configs/hallucination_localization_v1/adjudicated_labels.jsonl
configs/hallucination_localization_v1/protocol.json
run_artifacts/hallucination_localization_v1/merge_report.json
run_artifacts/hallucination_localization_v1/coverage_report.json
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

这些是可训练性门，不是 verifier 质量或方法有效性门。若 agreement 很低，停止在 calibration，先修改
定义/prompt 并重新盲标，不扩大数据。

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
4. 收到第二份标签后计算 agreement、裁决并审计；
5. 只有数据硬门通过才运行 H0-H3；
6. 根据 held-out localization 结果决定是否扩大标签、启用 pseudo-tail，或回到定义/标注器选择。

下一次需要用户/第二标注者介入的明确停止点是第 3 步：仓库会给出可直接交给另一个 AI 的完整 prompt
和 blind input，不要求对方理解本仓库实现。

## 9. 当前未验证事项

- Mistral-24B 是否能可靠区分“合法补全推导”与“无依据事实”；
- atomic claim 的字符区间到 Phi token onset 的一致性；
- 64 条中是否有足够多的真实 positive onset；
- noisy-or MIL 的长度偏置是否会在真实标签上出现；
- negative-tail shaping 是否改善排序，还是仅让 localization 指标变好。

因此当前唯一被授权的下一动作是构造 64-row selection manifest 与 annotation protocol；本文不授权
扩量标签、读取测试集或宣称 hallucination localization 已有效。
