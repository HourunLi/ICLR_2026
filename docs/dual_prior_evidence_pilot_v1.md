# Dual-Prior Evidence Pilot v1

状态：标注协议与 64 条盲包已冻结；等待 primary/secondary 双标与 agreement gate，尚未开始训练。

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

## 6. 首轮训练矩阵

双标与裁决通过后，固定比较：

```text
D0  correctness only
D1  correctness + direct key BCE
D2  correctness + direct complete BCE
D3  correctness + direct key BCE + direct complete BCE
```

四个 cell 使用相同 split、features、seed、epoch、optimizer 与 outcome loss。consistency、hallucination、tail、
progress、distill、gate-prior alignment 和 reconstruction 全部为 0。

机制指标至少报告 key/complete token AP、unit-level AP/F1、正例比例、位置分布和 key-vs-complete 区分度；同时
保留 correctness AUROC/accuracy guard。若 D3 只学到两个几乎相同的 map，即使 correctness 不下降，也不能称为
dual-prior 成功。

## 7. reconstruction 与协同 loss 的后续边界

v1 不生成 reconstruction target。same-candidate pooled hidden state 不是外部 target，会允许 uniform prior 走
自重构 shortcut，继续禁止。只有当 supported answer/evidence 经过独立流程生成并冻结成 `model_dim=768` target
后，才发布单独协议测试 reconstruction。

如果 direct targets 可学，再单独比较三种协同方式：不协同、旧 mutual MSE、以及尊重 `key ⊆ complete` 的
directional/containment objective。旧 mutual MSE 不能因为代码已存在就成为默认。
