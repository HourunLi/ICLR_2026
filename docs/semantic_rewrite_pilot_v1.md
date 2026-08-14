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

## 7. 冻结执行结果

v1 已在 clean commit `bc393cf856b5de45f5e2be281300f913c6724e71` 上完成三个阶段：

- 4 条源 trajectory（2 correct + 2 incorrect）形成 12 行、4 个 semantic group 和 3 个 style；
- 12 个 positive same-semantic/different-style pair 和 18 个 negative
  different-semantic/same-style pair；
- 4 个 original feature 复用，8 个 rewrite feature 按精确 token IDs teacher-force 提取，新增
  `539337712` bytes BF16 trajectory feature；
- audit 重读 12 个 trajectory 和 4 个唯一 condition，共 16 个 payload 的 SHA256 全部匹配；
- supervision audit 只允许 `consistency=true`，其余 6 个机制组件仍为 false，0 条未溯源监督。

关键产物：

- rewrite protocol SHA256：`5609537c02484b06700f8afa31c178c629193a6400630ee3f3088af6813aa512`；
- plan SHA256：`767e699acddf7850111492bc8137b186a5fd73172f6796ffd7a34a7dc66460ba`；
- extracted manifest SHA256：`9e58ede2d04c4d5911ca2d1eb945c42587d22950218ec99d1d33a0765a5bf443`；
- `run_artifacts/semantic_rewrite_pilot_v1/audit_report.json` SHA256：
  `a31e582579d830d4406f6564e3929f59adbc89686a1f856fda64daf69f08d27f`。

最终状态是 `passed_pipeline_only_no_mechanism_claim`，且
`formal_mechanism_claim_allowed=false`。这个结果不允许把格式变换表述为语义增强，只是真实 LLM
rewrite 的工程前置门。
