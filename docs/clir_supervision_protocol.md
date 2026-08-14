# CLIR 外部监督导入与覆盖审计协议

本协议定义“监督标签怎样安全进入真实 trajectory manifest”。它不是标签生成器，也不把
`correctness` 伪装成 token/step 标签。当前 Stage 1B v3 数据经审计后 7 个 CLIR 组件全部
不可用；真正的机制实验必须先由独立的 rewrite/verifier/judge 流水线产出本协议中的显式标签，
再发布新的冻结实验协议。

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
  "hallucination_onset": 37,
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
| `token_advantage` | 长度严格等于 `T=len(output_token_ids)`，全部 finite | token reward target |
| `progress_targets` | 长度严格等于 `T`，全部 finite | progress-head target |
| `key_prior_target` / `complete_prior_target` | 长度严格等于 `T`，值域 `[0,1]` | dual-prior BCE；两者共同覆盖才启用 distill/gate 对齐 |
| `semantic_id` + `style_id` 或 `domain_id` | semantic 必须有一个非空 spurious-attribute ID；`style_id` 优先，`domain_id` 可作兼容 fallback | consistency positive/negative pairs |
| `domain_id` | 提供时非空；同时给 `style_id` 时作为独立诊断元数据 | domain 诊断 |
| `complete_reconstruction_target` | 独立生成的 finite 定长向量；正式 CLIR 配置预期宽度为 `model_dim` | complete-prior reconstruction |

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
报告同时给出逐字段 row/token 覆盖、path 两类计数、positive onset 数、joint-prior token 数、
reconstruction 维度和 consistency pair 数。通过覆盖审计只证明数据契约成立，不证明 verifier
标签正确；标签质量仍需盲审样本、inter-annotator agreement 或独立 verifier parity。

## 4. 当前状态与下一步

截至 2026-08-14，Stage 1B v3 的 train 4096 行与 validation 8000 行都已用上述审计器检查：
10 个辅助字段的 row count 全为 0，7 个机制组件的 eligibility 全为 false。v3 launcher 会在
preflight 中复算并强制这一 outcome-only 契约，避免将 v3 误报为机制实验。

下一步应先在 train/validation 范围实现并小规模人工审核：

1. semantics-preserving rewrite + answer/evidence relation verifier，生成真实
   `semantic_id/style_id/domain_id`；
2. 独立 step/verifier 标注 path hallucination 和 first unsupported token，并记录原始判据；
3. 从 prefix entailment/evidence coverage 生成 progress、advantage 与 key/complete prior；
4. 冻结 annotation protocol/hash，走本协议的 merge/audit，再为非零 applicable counts 发布
   新的 mechanism protocol。`pilot_test` 在协议锁定前继续保持未读取。
