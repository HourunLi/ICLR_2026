# Dual-Prior Evidence Pilot v1

状态：64 条独立双标、role-blind adjudication、exact-token gold 与 4 cells × 3 seeds direct-target pilot 均已
完成。冻结结论为 `completed_pass_direct_targets_learnable`；证据等级仍是 `pipeline pilot`。后续用户明确裁决
保留原始双向 stop-gradient mutual distillation，不以 containment 替换；新 M0/M1 协议另见
`docs/dual_prior_mutual_distillation_pilot_v1.md`。

## 1. 先解决了什么代码问题

现有 CLIR head 同时产生两种对象：

- `key_prior_logits` / `complete_prior_logits` 通过 token-wise BCE 学 `[0,1]` membership；
- 同一 logits 再经过全轨迹 softmax，形成 `key_prior` / `complete_prior` attention distribution。

外部二值 token target 可以直接监督前一种对象。但旧默认还会对两个 softmax distribution 做双向 MSE，
并让 reward gate 对齐二者融合结果。若标注语义是 `key ⊆ complete`，双向 MSE 的低成本解却是让两张图相同，
这会主动抹掉 narrow key 与 broad complete 的区别。reconstruction 又是第三种假设，且当前没有独立外部
768-d target。

因此 v1 不直接验证整个旧 proxy，而只回答更基础的问题：外部 key/complete token membership 是否可复现、
非退化、并能被两个独立 head 学到。`train_clir.py` 已暴露以下独立权重：

```text
key_prior_weight
complete_prior_weight
prior_distill_weight
gate_prior_weight
reconstruction_weight
```

首轮把后三项固定为 0。mutual distillation 没有删除，但在新证据支持前不得启用。

## 2. 领域通用的 target 语义

标注对象是 trajectory 内的固定 reasoning units，而不是 GSM8K 专用的数字、等号或算术模板。

- `complete`：从 prompt 出发，复现并审核这条 trajectory 如何得到其结论所需的最小、非冗余推理链；
- `key`：complete 内最小的决定性子集，检查它最直接决定结论是否成立；
- 强约束：`key ⊆ complete`，两者对 usable row 都非空。

错误 trajectory 不被排除。对错误轨迹，decisive error/unsupported leap 是 key，实际使用的 material attempted
chain 是 complete。这样 target 描述“该看哪里”，correctness BCE 再描述“看完后的 outcome 是什么”；不会用
correctness 自动伪造 prior 标签。

纯标题、计划话术、未变换的 prompt 复述、重复计算和重复 final-answer wrapper 默认不选。答案-only 或无法表达
合理证据集合的 row 保持无监督，不写全零 target。

## 3. 为什么采用固定 unit 双标

让两个 annotator 独立自由画 char spans，会把“证据选择分歧”和“边界分词分歧”混在一起。v1 先用一个
domain-agnostic deterministic segmenter 按非空行、句末标点切分，并把常见 `Step N:` / `Answer:` 标题拆开。
每个非空白字符恰好属于一个 unit。annotator 只返回 unit indices；程序随后用冻结 Phi tokenizer 和
`output_token_ids` identity 将 char spans 映射到 token targets。

这不是声称 unitization 已经最优。若必要推理和无关文字无法分开，annotator 必须标
`insufficient_unitization`，该 row 不进入 loss。这样不会为了凑数据静默制造错误 token 标签。

## 4. 冻结 selection set

v1 复用 Hallucination Localization 已经绑定过 exact text/token identity 的 64 条 trajectory 和原有
query-disjoint `48 dense_train / 16 localization_dev` membership。复用只减少工程风险，不继承 correctness、
hallucination 或旧 claim 判断作为 prior 决策。

盲包统计：

- 64 rows / 64 queries；
- 1210 fixed units；
- 每条 9–38 units，中位数 18；
- annotator-visible 字段只有 `item_id/problem/trajectory/units`；
- correctness、path status、membership、query/source identity 全部只在 private lineage。

冻结入口：

```text
configs/dual_prior_evidence_v1/protocol_v1.json
configs/dual_prior_evidence_v1/annotation_guide_v1.md
configs/dual_prior_evidence_v1/annotation_items_v1.jsonl
configs/dual_prior_evidence_v1/annotation_lineage_v1.jsonl
configs/dual_prior_evidence_v1/package_report_v1.json
```

## 5. 双标 gate

两位 annotator 都标全部 64 条。结构合法率必须 100%；每位至少 48 usable，usable overlap 至少 40；eligibility
agreement 至少 `.75`。在双方都 usable 的行上，逐行 unit-set F1 再取宏平均，key 至少 `.45`、complete 至少
`.60`。这些是 module-development gate，不是 paper-quality human agreement 声明。

所有 set disagreement 在物化训练 gold 前都需要裁决；不能对两个二值 target 求平均后冒充 gold。若 agreement
失败，则修订 guide/unitization 并发布 v2，不能通过降低门槛覆盖失败。

### 5.1 primary 结果与不能越过的警告

primary 在冻结 commit `22b74f5` 上由 8 张卡各跑 8 条，64/64 strict schema-valid、64/64 usable，全部
token-map-valid：

- key positive token micro fraction：`.0844`；complete：`.3662`；
- key unit 中位数 1，complete unit 中位数 5；
- 64/64 都是严格 `key ⊂ complete`，没有 key=complete；
- 没有一条只选择最后一个 unit 为 key。

这些说明 target 非空且窄/宽两支没有在标注阶段坍缩。但 primary 不能单独成为 gold，原因也已量化：

- 58/64 只选一个 key unit；所有 key unit 的相对位置中位数 `.8378`，42/64 的全部 key 都在轨迹最后
  四分之一，必须加入 position-only baseline；
- 在旧 localization 私有审计中，23 条 hallucinated row 只有 1 条 key 覆盖 exact onset，12 条 key 与任一
  reviewed problem span 重叠。onset 不是 dual-prior gold，但这个结果说明 primary 常把末端 answer-producing
  calculation 当 key，没有稳定执行 guide 中“错误路径选决定性 flaw”的规则。

所以 primary 当时允许的唯一结论是：结构与 non-degeneracy 通过，语义采用门仍未通过。后续没有直接使用
primary，而是收回独立 secondary，比较其在错误路径上究竟选择 flaw 还是 terminal calculation，再裁决 key。
机器审计见 `configs/dual_prior_evidence_v1/primary_report_v1.json` 与
`primary_semantic_audit_v1.json`。

## 6. Secondary、裁决与 exact-token gold

Secondary 采用 resumable v1a 流程，每判断一条立即结构校验并原子 checkpoint。最终 raw 与 validated 都是
64/64，SHA256 均为
`271ca58e5ffcfd99000c2ff035059f00ff16d5df5de6c9d2cd95dcbb3fa23d1a`。冻结 agreement gate 通过：

- exact full-target agreement：16/64；需要裁决：48/64；
- key macro unit F1：`.546875`，超过冻结门 `.45`；
- complete macro unit F1：`.845640`，超过冻结门 `.60`；
- 结构、usable-yield 与 eligibility gate 均通过。

48 条 disagreement 随后通过只显示匿名 A/B proposal 的 role-blind packet 逐条裁决，并在每条后立即落盘。
裁决结果为 adopt-A 13、adopt-B 24、synthesize 11；46 high-confidence、2 medium-confidence，48/48 usable。
raw/validated adjudication SHA256 均为
`86fbbe0a5e75325b1d0bee23c39fc6e08383fc28bc079587770c6c68539d4e96`。这里的 A/B 角色只存在于私有
lineage，裁决者不能按 primary/secondary 身份择优。

Gold 物化严格区分两类事实：unit-set 语义来自 exact agreement 或盲审裁决；primary 的 `mapped_units` 只作为
已经验证过的 char→Phi-token alignment carrier，不能携带 unilateral semantic decision。最终：

- 64/64 usable；48 dense_train / 16 localization_dev，query-disjoint；
- 63/64 严格 `key ⊂ complete`，1/64 相等；
- key/complete token-positive micro fraction 为 `.0843/.3409`；unit 中位数为 `1/4`；
- exact-token gold SHA256：
  `da41e1e3061bb7ce321d12211fd5350f116d9510357ed774c7347c214e10cbdc`。

语义审计仍保留 shortcut 警告：44/64 的全部 key unit 位于最后四分之一，但 clean/hallucinated 的中位 key
位置分别为 `.882/.615`；错误路径 gold 对 exact onset 的覆盖从 unilateral primary 的 1/23 提高到 13/23，
对 sparse unsupported spans 的覆盖从 12/23 提高到 17/23。onset 与 evidence key 不是同一 target，这些数字
只诊断 guide alignment，因此训练评价必须保留 position-only baseline。

## 7. 首轮训练矩阵与结果

双标与裁决通过后，冻结并完成以下比较：

```text
D0  correctness only
D1  correctness + direct key BCE
D2  correctness + direct complete BCE
D3  correctness + direct key BCE + direct complete BCE
```

四个 cell 使用相同 split、features、seed、epoch、optimizer 与 outcome loss。consistency、hallucination、tail、
progress、distill、gate-prior alignment 和 reconstruction 全部为 0。

训练使用 exact same 48/16 rows、预计算 33×3072 Phi features、5 epochs、seeds 42/43/44。membership 评价读取
`sigmoid(key_prior_logits)` / `sigmoid(complete_prior_logits)`，没有把 trajectory-softmax attention 误当二值
概率。阈值只在 dense_train 校准后一次应用到 localization_dev；AP 本身不调阈值。结果如下：

| 指标（三种子均值） | D0 | 单头 direct BCE | D3 joint | position-only |
|---|---:|---:|---:|---:|
| key unit AP | `.0786` | D1 `.3769` | `.4325` | `.1327` |
| complete unit AP | `.3276` | D2 `.9208` | `.9192` | `.2684` |

D3 key 相对 D1 为 `+.0556`，complete 相对 D2 为 `-.00155`；correctness AUROC 相对 D0 为 `+.00529`。
D3 两张 membership map 的平均绝对概率差为 `.3022`，Pearson correlation 为 `.7704`，没有塌成同一张图。
8 个冻结 guard 都是 3/3 seeds 通过；更严格的“同一个 seed 内全部 guard 同时通过”也是 3/3。完整机器结果：

```text
configs/dual_prior_evidence_v1/training_protocol_v1.json
configs/dual_prior_evidence_v1/training_result_v1.json
run_artifacts/dual_prior_evidence_v1/pilot_v1/
```

`training_result_v1.json` SHA256 为
`7b11aaf92d5ed6b9d7a3c99cd34c039077e295855b2305efc5a9b4f53595801b`；12/12 cells 均来自 clean commit
`f485e54db0181fbcce677b8ae3a0fa895e4e8f82`，没有访问 pilot/final test。

允许结论仅是：adjudicated key/complete direct targets 在这个小型 pipeline pilot 中明显可学，并可在同一模型
内共存。它不证明对最终 Best-of-N 有增益，也不证明 learned map 在新领域泛化。

## 8. reconstruction 与协同 loss 的后续边界

v1 不生成 reconstruction target。same-candidate pooled hidden state 不是外部 target，会允许 uniform prior 走
自重构 shortcut，继续禁止。只有当 supported answer/evidence 经过独立流程生成并冻结成 `model_dim=768` target
后，才发布单独协议测试 reconstruction。

Direct-target gate 已通过。最初曾计划并列比较不协同、mutual MSE 与 directional containment；随后用户裁决
相互蒸馏是必须保留的主方法，不能由 containment 替换。当前实际冻结的是从头复跑的 M0 direct-BCE control 与
M1 原始双向 stop-gradient mutual MSE（权重 `.25`）；direct BCE 始终作为 anchor，gate alignment 与
reconstruction 继续关闭。containment 本轮不实现，若未来需要只能作为另行授权的附加消融。
