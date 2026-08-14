# CLIR 项目交接

最后更新：2026-08-15

这份文档只记录当前可执行事实。方法细节看 `docs/proposal.md`，历史审查看
`docs/code_review_panzhixin_*_change.md`，不要从旧审查里的行号或旧协议推断 HEAD 行为。

## 1. 一句话状态

第四轮审查要求的 P0/P1 代码问题已经修复，Stage 1B v4 outcome-only 3×3 已在 clean commit
`b1c4fae` 上执行完成。9 个 cell 中 1 个通过、8 个在 final-train 常数先验门失败，因此结果只允许
作为优化不稳定性诊断，不能回答 CLIR 机制是否有效。train-only rewrite 工程 pilot v1
也已完成 prepare/extract/audit；它只验证流水线，当前要进入带独立 verifier 的真实 LLM
semantics rewrite。`pilot_test` 仍未读取。

## 2. 已建立的证据

### 工程证据

- 自包含 PyTorch 模型、真实 JSONL/feature loader、训练、打分、评估和多 seed 汇总均已实现。
- 真实 feature 使用全部 33 层，宽度 `101376`；大张量在 LayerNorm/投影前分块，不删层。
- 生成/提取按 query 原子分片，可校验、可断点恢复；condition 是 query 级 prompt-only forward。
- Stage 1B feature 完整性门覆盖 12,096 行、13,108 个唯一 payload、
  725,761,877,084 bytes，0 failure。报告：
  `run_artifacts/stage1b_v2/audits/feature_mirror_verification.json`，SHA256
  `e38b14446dca77926d28361aa7cb3877a7970aa7114b8980ca3141fdc7354d25`。
- 同设备 resume 的数值恢复已有测试；当前 checkpoint/run 还记录恢复来源、起始 epoch、训练分段和
  resolved device，并硬拒绝跨设备 resume。

### 研究证据

- Stage 1 small-scale correctness-only：strict/encoded/CLIR 的 BoN@16 分别为
  `88.28±2.07% / 88.54±1.63% / 89.32±2.51%`。encoded→CLIR 平均 `+0.78`
  个百分点，但三个 seed 方向不稳定，不能宣称稳定增益。
- Stage 1B v1 是 pre-audit artifact；v2 没有完成 epoch；v3 没有正式效果结果并已被 v4 取代。
- 当前 train/validation 的 10 个辅助字段覆盖都是 0，7 个 CLIR 机制组件均不可训练。这个降级由
  launcher 复算并强制，不是文档约定。
- Stage 1B v4 的 9 个 cell 均实际执行并绑定 commit `b1c4fae49acebe7e03d87a8ae7664a997ad05358`。
  final summary 为 `incomplete_diagnostic_only`：1 个 included、8 个显式 train health failure、
  0 个 unknown/unrun；`formal_primary_claim_allowed=false`。
- 唯一 included cell 是 `seed=42/encoded_swift`：score population std `1.3618`、query 内 pairwise
  accuracy `0.6680`，BoN@16 `0.912`。单 seed 不能形成稳定 baseline 或模型间主比较。
- semantics rewrite 工程 pilot v1 在 commit
  `bc393cf856b5de45f5e2be281300f913c6724e71` 上完成：12 行/4 个 semantic group/3 个 style，
  12 个正 pair/18 个负 pair，12 个 trajectory 和 4 个 condition payload 全部通过 SHA256、
  shape、dtype、token 对齐和 finite 审计。audit 状态是
  `passed_pipeline_only_no_mechanism_claim`。

## 3. 已冻结完成的 Stage 1B v4

机器协议：`configs/stage1b_validation_v4.json`。说明：`docs/stage1b_v4_protocol.md`。

### 数据

| split | rows / queries / candidates | correct / incorrect | pool composition | SHA256 |
|---|---:|---:|---:|---|
| train | 4096 / 512 / 8 | 3668 / 428 | 111 mixed / 385 all-correct / 16 all-wrong | `6faca803b108675ed7ae08e54416969b483e7b9918133b51a0c9aff92fe02e06` |
| validation | 8000 / 500 / 16 | 7155 / 845 | 146 mixed / 342 all-correct / 12 all-wrong | `1e55ccfea2de2aa8190072e8a8c5784aec3d64059751355f6a5bde23295a6c36` |

checker 固定为 `clir_gsm8k_numeric_v5`，label protocol SHA256 为
`4245dcf4d6f15583d03c1e8088481b597df41a4fe39daf1c2e098e884c3ad8ef`。v4 复用已验证的
v5 manifest/feature，不重复采集；新结果写入 `run_artifacts/stage1b_v4`。

正式 `summary.json` SHA256 为
`0f1e49ad72f71c4b2f51ad904f92f0cd02593e77d41d10caf34fe2e00a3b095c`。执行覆盖是 9/9，
但健康 evaluation 覆盖只有 1/9，所以 `matrix_complete=false`。不要在同一 v4 协议下补跑、改门或
用 snapshot 替代 final；新实验必须使用新协议与新目录。

### 矩阵与预算

- variants：`strict_swift / encoded_swift / clir`
- seeds：`42 / 43 / 44`
- 5 epochs，batch size 2，learning rate `1e-4`，BF16 train，gradient clip 1.0
- FP32 score，batch size 2
- final epoch 是预注册主 checkpoint；每个 epoch snapshot 只用于诊断
- k=`1/2/4/8/16`，query bootstrap 10,000 次，95% CI

### 三道健康门

1. final train correctness BCE 相对常数类别先验至少改善 1%；改善方向错误也失败。
2. validation reward score population std 至少 0.1。
3. max-k pool 内，query 内 correct-vs-incorrect pairwise 排序准确率至少 0.60；micro-average，
   tie 计 0.5。

第三道门来自第四轮审查对历史 v1 的诊断分离区间。它是 v4 预注册门，不得在看过 v4 结果后原地
调整；若要改阈值，发布新协议。

### 失败与诊断语义

- train 或 score 失败时，先写持久化健康证据，再以非零状态退出。
- evaluate 排序失败时，先写 evaluation 报告，再以非零状态退出。
- `--allow-failed-cells` 只接受与 v4 协议 SHA256、阈值和 cell 身份匹配的失败证据。
- 缺失但没有明确失败证据的 cell 仍报错，不能把“没跑”伪装成“健康门失败”。
- 不完整矩阵的 summary 标为 `incomplete_diagnostic_only`，
  `formal_primary_claim_allowed=false`；即使全部 cell 都失败，也会留下矩阵诊断报告。
- final 健康失败不影响读取某个 epoch 自己的 `training_health`；snapshot 结果不能替代正式 final。

## 4. 受保护的不变量

以下约束若要改变，必须同时改测试并发布新版本协议：

- `output_token_ids` 长度、trajectory token 轴和所有 token 标签严格一致；缺失标签保持缺失。
- `query_id` 是候选池身份；`semantic_id` 是 rewrite group 身份，两者不能 fallback 混用。
- 全部 33 层是正式输入；不能为了存储/显存静默截层、pool 或降采样。
- vLLM 候选顺序以 `CompletionOutput.index` 为事实来源，ordered-prefix 必须使用
  `first_k_by_vllm_completion_output_index`。
- scoring provenance 必须绑定 variant、checkpoint/input/protocol SHA256、batch、dtype 和 score
  distribution；summary 会消费并校验这些字段。
- 外部机制标签必须通过 `scripts/merge_clir_supervision.py` 的 row/query/token-ID hash 身份绑定和
  `scripts/audit_clir_supervision.py` 的覆盖审计；不得从 correctness 推导。
- 历史协议和 artifact 只读。Stage 1B v1 scored 文件缺当前 provenance，若需要当前 evaluator 的
  诊断必须重新打分，不能降低 evaluator 契约。

## 5. 最短验证路径

```bash
cd /prodcpfs/user/panzhixin/ICLR_2027
P=/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python

"$P" -m pytest -q
"$P" -m pip check
"$P" scripts/run_stage1b_validation.py --stage preflight
"$P" scripts/run_stage1b_validation.py --stage train --device cuda
```

最后两条分别做只读输入检查和打印九个训练命令；都不会启动训练。正式执行步骤见
`docs/runbook_zh.md`，必须先由用户确认。

## 6. 文件导航

```text
README.md                              项目入口和当前结论边界
docs/handoff.md                        本交接
docs/proposal.md                       方法设计
docs/runbook_zh.md                     最短运行手册
docs/stage1b_v4_protocol.md            当前 Stage 1B 人类协议
configs/stage1b_validation_v4.json     当前 Stage 1B 机器协议
docs/clir_supervision_protocol.md      真实机制监督接入契约
docs/semantic_rewrite_pilot_v1.md      当前 train-only rewrite 工程 pilot

src/consistency_localized_reward.py    模型与 loss
src/clir_data.py                       数据契约与 collate
train_clir.py                          训练、恢复和 checkpoint
score_clir.py                          打分与 score 健康门
evaluate_clir.py                       BoN 与排序健康门
summarize_clir.py                      多 seed / 失败矩阵汇总
scripts/run_stage1b_validation.py      唯一正式 launcher
scripts/run_semantic_rewrite_pilot.py  rewrite prepare/extract/audit 唯一入口
```

已删除的 `scripts/diagnose_stage1b.py` 不再是入口：其候选顺序、排序和矩阵检查已由 evaluator、summary
与 launcher 的强制契约覆盖。不要重新引入另一套诊断口径。

## 7. 当前下一步：真实 LLM semantics rewrite

v4 已证明 outcome-only 训练在当前预算下高度不稳定，可逆格式 pilot 也已把
`semantic_id/style_id`、teacher-forced token 对齐、全 33 层 feature 提取和 consistency pair 审计
跑通。下一个版本不再增加格式 transform，而是冻结 generator model/revision、prompt/decoding、
rewrite axes 和 query-atomic resume，并为每个候选接入独立 answer + required-evidence relation
verifier 及分层人工盲审。建议先做 32 条 train trajectory × 2 个真实 rewrite style 的小型
冻结 pilot，质量门通过后再扩到 train/validation。`query_id` 始终保留原候选池身份，
`semantic_id` 单独表示源 trajectory 的 rewrite group。
