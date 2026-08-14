# CLIR semantics rewrite 工程 pilot v1

机器配置：`configs/semantic_rewrite_pilot_v1.json`

唯一入口：`scripts/run_semantic_rewrite_pilot.py`

## 1. 目标与结论边界

这个 pilot 只验证 rewrite 数据流水线：独立的 `query_id` / `semantic_id` / `style_id`、精确 token
身份、teacher-forced 全 33 层特征、共享 prompt-only condition，以及 consistency 正负 pair 覆盖。
它使用可逆的表面格式变换，不是研究级 LLM semantic rewrite，配置固定
`formal_mechanism_claim_allowed=false`；任何训练数字都只能算工程 smoke evidence。

`pilot_test` 不在输入范围。v1 只从冻结 train manifest 选样；未来 LLM rewrite 扩展也只能先使用
train/validation，并保持原 query 的全部视图不跨 split。

## 2. 冻结输入与选择

- source：`run_artifacts/stage1b_v3/labels/train_extracted.v5.jsonl`
- SHA256：`6faca803b108675ed7ae08e54416969b483e7b9918133b51a0c9aff92fe02e06`
- 按 source manifest 顺序选择 2 条 correct、2 条 incorrect trajectory；4 条必须来自不同 query
- 每条源 trajectory 产生 `original / markdown_bullets / blockquote` 三个 style，共 12 行

`query_id` 原样保留候选池身份；`semantic_id` 从 source trajectory ID 派生；`style_id` 只表示当前
视图。source `candidate_index` 仅进入 lineage，不复制到顶层，避免 rewrite manifest 被误当作新的
Best-of-N 候选池。

## 3. 可逆性与 answer gate

两个 rewrite transform 只给每个 physical line 加固定 `- ` 或 `> ` 前缀。每行在发布前必须满足：

1. 逆变换逐 UTF-8 字节恢复 source response；
2. 使用冻结 Phi tokenizer 编码后，token IDs 解码精确恢复 rewrite response；
3. v5 checker 的 `correctness` 与 `normalized_candidate_answer` 均与 source 相同；
4. plan、response、source/output token IDs、source manifest 与 protocol 全部有 SHA256 lineage；
5. `semantic_id` 不得等于或替代 `query_id`。

这些条件给表面变换提供确定性 semantic certificate，但不能替代未来 LLM rewrite 的
answer/evidence-relation verifier 或人工盲审。

## 4. 特征契约

- rewrite trajectory 使用保存的精确 output token IDs 做 teacher forcing；
- 输入是 embedding 加 32 个 block outputs，拼接宽度固定 `101376`，BF16 存储；
- original view 复用已校验的 immutable trajectory feature；
- 同一 query 的所有视图复用原 prompt-only condition path/SHA256，不从 response forward 切片；
- audit 会重新加载全部 12 条 trajectory、4 个唯一 condition，检查 SHA256、shape、dtype、token
  对齐和 finite，并要求只有 consistency component eligible。

## 5. 执行

不带 `--execute` 只打印/验证计划：

```bash
P=/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python
"$P" scripts/run_semantic_rewrite_pilot.py --stage prepare
"$P" scripts/run_semantic_rewrite_pilot.py --stage extract
"$P" scripts/run_semantic_rewrite_pilot.py --stage audit
```

实际执行按顺序运行：

```bash
"$P" scripts/run_semantic_rewrite_pilot.py --stage prepare --execute
CUDA_VISIBLE_DEVICES=0 "$P" scripts/run_semantic_rewrite_pilot.py \
  --stage extract --device-map auto --execute
"$P" scripts/run_semantic_rewrite_pilot.py --stage audit --execute
```

已有输出默认拒绝覆盖。开发期确需重建同一 pilot 时显式加 `--overwrite`；研究级数据必须发布新协议
和新目录，不能覆盖 v1。

## 6. 进入真实 semantic rewrite 的门

v1 通过后再发布 LLM rewrite 协议，至少冻结 generator model/revision、prompt、decoding、rewrite
axes、regeneration/teacher-force policy 和 query-atomic resume。每个候选必须通过独立 answer + required
evidence relation verifier，并在分层人工盲审中报告接受率与 agreement。只有 train/validation 的质量
与覆盖门都通过后，才可把 rewrite groups 纳入新的 mechanism experiment。
