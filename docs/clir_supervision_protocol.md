# CLIR 外部监督导入与覆盖审计协议

本协议定义“监督标签怎样安全进入真实 trajectory manifest”。它不是标签生成器，也不把
`correctness` 伪装成 token/step 标签。Stage 1B v4 仍是冻结的 outcome-only control；当前真正进入
localization Pilot 的是独立 claim review 物化出的显式 sparse token labels。

## 1. 不变量

- `output_token_ids` 是 token 位置的唯一事实来源；禁止对 `response` 重新分词后贴标签。
- 每条 annotation 同时绑定 trajectory `id`、`query_id` 与
  `SHA256(canonical_json(output_token_ids))`。三者任一漂移都拒绝合并。
- 缺失字段表示“没有该监督”，不会补零。零只能是标注者明确写出的真实标签值。
- 合并器拒绝覆盖 manifest 已有的同类字段，也拒绝未知或重复 trajectory ID。
- correctness 只能监督 outcome BCE；不得由它推导 hallucination、progress、prior 或
  consistency 标签。
- annotation 文件及 annotation protocol 都必须有 SHA256 provenance；先版本化、再合并、
  再审计，不允许直接手改合并后的 manifest。

## 2. Annotation JSONL 契约

每行至少包含一个可选监督字段，并使用 canonical 字段名：

```json
{
  "schema_version": "clir-supervision-annotation-v1",
  "id": "gsm8k-train-00001-cand-0",
  "query_id": "gsm8k-train-00001",
  "output_token_ids_sha256": "<64 lowercase hex>",
  "annotation_provenance": {
    "schema_version": "clir-supervision-provenance-v1",
    "protocol_sha256": "<64 lowercase hex>",
    "annotation_source": "human-or-versioned-verifier-name"
  },
  "path_hallucinated": 1,
  "hallucination_onset": 3,
  "token_hallucination_target": [0, 0, 0, 1, 1],
  "token_hallucination_mask": [1, 1, 0, 1, 1],
  "semantic_id": "semantic-group-17",
  "style_id": "equation-first",
  "domain_id": "gsm8k"
}
```

其余字段可在同一批或后续非覆盖 annotation 中追加；合并后的行会按字段记录各批来源。实际约束
如下：

| 字段 | 约束 | 用途 |
|---|---|---|
| `path_hallucinated` | 数值 `0` 或 `1` | path-level MIL |
| `hallucination_onset` | 无幻觉为 `-1`，有幻觉为 `[0,T-1]`；必须同时给 path 标签且一致 | token localization / negative tail |
| `token_hallucination_target` + `token_hallucination_mask` | 两者必须同时提供、长度严格等于 `T`、值为 binary；mask 至少覆盖一个 token，mask 外 target 必须为 0 | sparse token/span localization |
| `token_advantage` | 长度严格等于 `T=len(output_token_ids)`，全部 finite | token reward target |
| `progress_targets` | 长度严格等于 `T`，全部 finite | progress-head target |
| `key_prior_target` / `complete_prior_target` | 长度严格等于 `T`，值域 `[0,1]` | dual-prior BCE；两者共同覆盖才启用 distill/gate 对齐 |
| `semantic_id` + `style_id` 或 `domain_id` | semantic 必须有一个非空 spurious-attribute ID；`style_id` 优先，`domain_id` 可作兼容 fallback | consistency positive/negative pairs |
| `domain_id` | 提供时非空；同时给 `style_id` 时作为独立诊断元数据 | domain 诊断 |
| `complete_reconstruction_target` | 独立生成的 finite 定长向量；正式 CLIR 配置预期宽度为 `model_dim` | complete-prior reconstruction |

`token_hallucination_mask=1,target=0` 表示“该 token 被明确审核为 non-hallucinated”；mask 为 0 表示
缺失监督，绝不是负例。target vector 虽在 mask 外以 0 占位，collate、loss 与 evaluation 都必须保留 mask，
不得把占位 0 纳入 BCE。具体任务协议还可以增加更强约束；Localization v2 例如要求最早 positive token
等于冻结 onset，且 positive token 的存在性与 path label 一致。

annotation 可以是稀疏的：例如只有 path label 时，不要写 onset 或任何全零 token vector。为了让
consistency loss 真正有适用样本，数据集必须同时形成“同 semantic、不同 style”的正 pairs 与
“不同 semantic、同 style”的负 pairs；仅让字段非空还不够。

顶层字段是严格白名单；额外的 judge rationale、置信度或人工复核信息统一放进可选的
`annotation_metadata` object，避免字段拼写错误被静默忽略。

## 3. 导入与审计

先固定输入 manifest、annotation JSONL 和 annotation protocol 的 SHA256，再运行：

```bash
PYTHON=/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python

"$PYTHON" scripts/merge_clir_supervision.py \
  --input-jsonl /absolute/path/base.v5.jsonl \
  --annotations-jsonl /absolute/path/annotations.v1.jsonl \
  --output-jsonl /absolute/path/supervised.v1.jsonl \
  --output-report /absolute/path/supervised.v1.merge.json \
  --expected-input-sha256 <base_sha256> \
  --expected-annotations-sha256 <annotations_sha256> \
  --expected-reconstruction-dim 768

"$PYTHON" scripts/audit_clir_supervision.py \
  --input-jsonl /absolute/path/supervised.v1.jsonl \
  --expected-input-sha256 <supervised_sha256> \
  --expected-reconstruction-dim 768 \
  --require consistency path_hallucination onset_localization \
            token_advantage progress dual_prior reconstruction \
  --output-json /absolute/path/supervised.v1.coverage.json
```

`--require` 应只列入预注册实验真正声称使用的组件；缺少任一组件时审计会以非零状态退出。
报告同时给出逐字段 row/token 覆盖、sparse mask 内 positive/negative token 数、path 两类计数、
positive onset 数、joint-prior token 数、reconstruction 维度和 consistency pair 数。通过覆盖审计只证明
数据契约成立，不证明 verifier 标签正确；标签质量仍需盲审样本、inter-annotator agreement 或独立
verifier parity。

## 4. 当前状态与下一步

Stage 1B v4 的 train 4096 行与 validation 8000 行仍保持 outcome-only。其冻结 contract 列出当时的
10 个辅助字段且全为 0；当前 auditor 会额外返回新增的两个 sparse-token canonical 字段，v4 launcher
允许这些未写入旧 contract 的字段保持 0，但会拒绝任何非零新增监督。因此历史控制没有被 schema 扩展
悄悄改变。

截至 2026-08-18，Hallucination Localization v2 已按本协议导入 64 行：

- 9,132 个 mask 内 supervised tokens；
- query-disjoint train 48 行为 922 positive / 5,759 negative；
- dev 16 行为 528 positive / 1,923 negative；
- row/query/token hash、annotation provenance、target/mask 长度与取值均通过；
- S1 sparse span BCE 的 point-estimate token gate 通过，但 bootstrap 区间跨 0，exact onset gate 失败。

当前下一步是单独冻结 boundary/segment onset Pilot；在它通过前不启用 pseudo-tail、negative-tail shaping
或 mixed-data mechanism run。完整结果见 `docs/hallucination_localization_pilot_v2.md`。
