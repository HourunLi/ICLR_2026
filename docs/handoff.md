# CLIR 项目交接

最后更新：2026-08-18

这份文档只记录当前可执行事实。方法细节看 `docs/proposal.md`，历史审查看
`docs/code_review_panzhixin_*_change.md`，不要从旧审查里的行号或旧协议推断 HEAD 行为。

## 1. 一句话状态

第四轮审查最初条目已处理，Stage 1B v4 outcome-only 3×3 也已在 clean commit `b1c4fae` 上执行；
审查报告后来追加的 §13 确认了两个问题：训练健康门使用 epoch-average BCE 而非 final checkpoint 的
完整 train-split BCE，且冻结的 persistent-worker loader 配置下 resume 不 bit-exact。当前代码已经
修复并回归这两项；历史 summary 的 1/9 分类仍不回写，只读重算仍为 4/9（strict 3/3、encoded
1/3、CLIR 0/3），且只能作为优化诊断，不能回答 CLIR 机制是否有效。train-only rewrite 工程 pilot
v1 与真实 LLM 4-source gate 都已执行；Qwen v5、Falcon3 v6 和 Qwen-7B v7 均未通过扩规模门。
`pilot_test` 仍未读取。

真实 rewrite 的模型层已进一步冻结：Phi-3.5-mini 继续作为 task/feature model；Qwen-2.5-3B 首版、
Falcon3-3B fallback 和 Qwen-2.5-7B scale diagnostic 分别冻结在 model-selection v1/v2/v3。
runtime-pinned v4 接受 1/8；结构化 Qwen v5 与保持同 gate 只换模型的 Falcon3 v6 都接受 0/8；
只扩大 Qwen checkpoint 的 v7 接受 1/8。v7 的 source 已暴露，机器协议禁止用它授权扩规模；全部
`scale_allowed=false`。数据接口新增
exact-token online hidden-state 模式，避免为每个 rewrite 永久保存 `[T,101376]` BF16 payload。

当前 semantics augmentation 已换方向：主路线 Route A 从同一原始 prompt 下的 Phi 原生候选池挖掘
reasoning-equivalent trajectories，Route B 让 Phi 自己 rewrite；Qwen rewrite 只保留为后续 off-policy
control。独立 Mistral-24B relation verifier 已完成 64-pair 双标注 gold 盲测，当前仅作为 Silver pilot
gate。`development_32` 的便宜预筛显示 correct 候选充足、incorrect 候选稀少，因此 Pilot-0 先做
correct-only。随后对 operational 31-pair 发布做的高强度复审发现，旧 flat verifier 字段模板化、没有
执行 v8 双向 material-claim/risk 审核，并错误地从 correct outcome 推断内部无错误。旧 121 Silver、
31 组/62 行 manifest 和训练资格已经撤销；原 candidate/source/hash 保留。v1a 双标、裁决和 matched
训练现已完成：30/31 status 一致，最终 27 accept / 4 reject；27 组只作为 metadata 注入原 4096 行
mixed train。单 seed/1 epoch 结果没有出现 correctness 或 ranking 崩坏，表示诊断显示主要效果是分离
不同语义而非进一步提高已饱和的正对 cosine。此处均为 pipeline-pilot 决策，不是 augmentation
泛化或正式有效性证据。当前下一模块是 hallucination localization 的 64-row selection/annotation
protocol；`pilot_test` 仍未读取。

## 2. 已建立的证据

### 工程证据

- 自包含 PyTorch 模型、真实 JSONL/feature loader、训练、打分、评估和多 seed 汇总均已实现。
- 真实 feature 使用全部 33 层，宽度 `101376`；大张量在 LayerNorm/投影前分块，不删层。
- 生成/提取按 query 原子分片，可校验、可断点恢复；condition 是 query 级 prompt-only forward。
- Stage 1B feature 完整性门覆盖 12,096 行、13,108 个唯一 payload、
  725,761,877,084 bytes，0 failure。报告：
  `run_artifacts/stage1b_v2/audits/feature_mirror_verification.json`，SHA256
  `e38b14446dca77926d28361aa7cb3877a7970aa7114b8980ca3141fdc7354d25`。
- checkpoint/run 已记录恢复来源、起始 epoch、训练分段和 resolved device，并硬拒绝跨设备 resume。
  当前 loader 使用显式 `(seed, epoch)` sampler，worker/base-seed generator 与 reward-model RNG 分离；
  `num_workers=4, pin_memory=true, persistent_workers=true` 的真实参数回归已验证模型、优化器和 metrics
  bit-exact。
- `train_clir.py`/`score_clir.py` 同时支持历史 `precomputed` 和新 `online` feature source。online
  manifest 仍必须保存精确 prompt/output token IDs、Phi/tokenizer revision 和 protocol provenance；
  文本只用于审计。一个 105 prompt-token + 221 output-token 的真实 Phi 样本在线重现 trajectory 与
  condition 后，与历史 BF16 payload 均 bitwise 相等；模型冷加载约 29 秒，抽取约 0.39 秒。该数字
  只是一条 pipeline gate，不可外推为全量吞吐。

### 研究证据

- Stage 1 small-scale correctness-only：strict/encoded/CLIR 的 BoN@16 分别为
  `88.28±2.07% / 88.54±1.63% / 89.32±2.51%`。encoded→CLIR 平均 `+0.78`
  个百分点，但三个 seed 方向不稳定，不能宣称稳定增益。
- Stage 1B v1 是 pre-audit artifact；v2 没有完成 epoch；v3 没有正式效果结果并已被 v4 取代。
- 当前 train/validation 的 10 个辅助字段覆盖都是 0，7 个 CLIR 机制组件均不可训练。这个降级由
  launcher 复算并强制，不是文档约定。
- Stage 1B v4 的 9 个 cell 均实际执行并绑定 commit `b1c4fae49acebe7e03d87a8ae7664a997ad05358`。
  冻结的 final summary 为 `incomplete_diagnostic_only`，并按有缺陷的门禁实现记录 1 个 included、
  8 个 train health failure、0 个 unknown/unrun；`formal_primary_claim_allowed=false`。
- 对 final checkpoint 的只读重算显示健康分类应为 strict SWIFT 3/3、encoded SWIFT 1/3、CLIR 0/3，
  总计 4/9。历史 summary 不原地修改；该重算只纠正优化健康诊断，不会把 outcome-only 实验升级为
  CLIR 机制证据。
- 历史 summary 唯一 included cell 是 `seed=42/encoded_swift`：score population std `1.3618`、query
  内 pairwise accuracy `0.6680`，BoN@16 `0.912`。单 seed 不能形成稳定 baseline 或模型间主比较。
- semantics rewrite 工程 pilot v1 在 commit
  `bc393cf856b5de45f5e2be281300f913c6724e71` 上完成：12 行/4 个 semantic group/3 个 style，
  12 个正 pair/18 个负 pair，12 个 trajectory 和 4 个 condition payload 全部通过 SHA256、
  shape、dtype、token 对齐和 finite 审计。audit 状态是
  `passed_pipeline_only_no_mechanism_claim`。
- Route A v1a 的 31 对盲双标 status agreement 为 `0.9677`、Cohen's kappa 为 `0.8394`；唯一分歧裁为
  reject，最终 27 对进入 4096 行 mixed train。matched consistency=0/1 cell 都完整观测 27 个正 pair、
  26 个实际 batch 负 pair。consistency cell 的 train checkpoint/validation BCE 为 `0.2182/0.4600`，
  baseline 为 `0.2389/0.5074`；32-query 验证的 pairwise accuracy 为 `0.5871 vs 0.5702`，BoN@16
  同为 `0.875`。只有 10 个 mixed query，全部结果均为 diagnostic-only。
- v1a projected representation 的 same-semantic cosine 为 baseline `0.9989`、consistency `0.9827`；
  different-semantic/same-style cosine 为 `0.9926`、`0.7305`，正负均值 gap 从 `0.0064` 扩到
  `0.2522`。这说明当前 loss 主要阻止全局 collapse；不能表述成“把正对变得更接近”。机器结果见
  `configs/on_policy_pilot0_v1a/training_result_v1.json`。
- 真实 LLM gate 全程只读 train manifest，并以 source-atomic shard 和 completion marker 支持恢复。
  v1 的 0/8 包含 1 次 incorrect-source repair；v2 的 3/8、v3/v4 的 1/8 都是 0 repair。v4 的 8 个
  generator 输出与 v3 逐条 SHA256 相同，证明运行时切换没有改变生成诊断；self-verifier 只额外拒绝
  1 条，仍对多条删/并行输出给出 accept，不能作为唯一 fidelity 证据。所有 audit report 都明确
  `formal_mechanism_claim_allowed=false`、`scale_allowed=false`。
- Qwen-2.5-7B v7 严格复用 v5 的 source、prompt、greedy decoding、parser 和门槛，仅改变 checkpoint。
  它接受 1/8；3 条 conversational 输出标签不闭合，4 条可解析输出改变数字 occurrence，所有 5 条
  可解析输出的 self-verifier 都给出 accept，且 0 次 incorrect-source repair。protocol SHA256 为
  `576752391ad278eb4b6fa6ee3d9f03e315739a332c2ef983cdbc7e9c63bf3d76`，audit report SHA256 为
  `433d5d2dde4351e7366edca942ef72ca648d41e913bbe697524a990d0ad22788`；报告另强制记录
  `scale_authorization_eligible=false`。
- reasoning-equivalence selection set 位于 `configs/verifier_selection_v1/`：64 条、16 个 base、每个 base
  4 个变体，覆盖 quantitative/code-rules/evidence/logic-procedure；两份独立人工标注的 decision agreement
  为 `81.25%`、Cohen's kappa `0.667`，12 个分歧均经逐项裁决，最终 gold 为 32 accept / 32 reject。
- `mistralai/Mistral-Small-24B-Instruct-2501` 固定 revision
  `9527884be6e5616bdd54de542f9ae13384489724`，在不读取 gold 的 generation stage 上完成 selection：
  parse-valid `64/64`，decision accuracy `56/64 = 87.5%`，auto-accept precision `28/32 = 87.5%`；
  4 false accept、4 false reject。结果保存在
  `run_artifacts/verifier_selection_mistral24_v2/{run.json,score_report.json,scored_records.jsonl}`；runner 已在
  commit `506cb1a` 推送。该结果授权 Silver pilot gate，不授权 Gold/正式自动标注声明。
- `development_32` 只读候选池诊断覆盖 32 query × 8 Phi candidates。按同 `query_id`、同 correctness、
  同 `normalized_candidate_answer` 且 response 不同组合，correct 有 727 对，incorrect 有 8 对。该筛选
  只是 verifier 前的高召回候选生成，不能把“同答案”当成 reasoning equivalence。

## 3. 已冻结完成、但门禁分类有已知缺陷的 Stage 1B v4

机器协议：`configs/stage1b_validation_v4.json`。说明：`docs/stage1b_v4_protocol.md`。

### 数据

| split | rows / queries / candidates | correct / incorrect | pool composition | SHA256 |
|---|---:|---:|---:|---|
| train | 4096 / 512 / 8 | 3668 / 428 | 111 mixed / 385 all-correct / 16 all-wrong | `6faca803b108675ed7ae08e54416969b483e7b9918133b51a0c9aff92fe02e06` |
| validation | 8000 / 500 / 16 | 7155 / 845 | 146 mixed / 342 all-correct / 12 all-wrong | `1e55ccfea2de2aa8190072e8a8c5784aec3d64059751355f6a5bde23295a6c36` |

checker 固定为 `clir_gsm8k_numeric_v5`，label protocol SHA256 为
`4245dcf4d6f15583d03c1e8088481b597df41a4fe39daf1c2e098e884c3ad8ef`。v4 复用已验证的
v5 manifest/feature，不重复采集；新结果写入 `run_artifacts/stage1b_v4`。

冻结的 `summary.json` SHA256 为
`0f1e49ad72f71c4b2f51ad904f92f0cd02593e77d41d10caf34fe2e00a3b095c`。执行覆盖是 9/9，
但其 1/9 健康分类来自已知有缺陷的 BCE 输入，不能继续作为正确分类引用。不要原地修改 v4、补跑、
改门或用 snapshot 替代 final；修正门禁与重新发布结果必须使用新协议与新目录。

### 矩阵与预算

- variants：`strict_swift / encoded_swift / clir`
- seeds：`42 / 43 / 44`
- 5 epochs，batch size 2，learning rate `1e-4`，BF16 train，gradient clip 1.0
- FP32 score，batch size 2
- final epoch 是预注册主 checkpoint；每个 epoch snapshot 只用于诊断
- k=`1/2/4/8/16`，query bootstrap 10,000 次，95% CI

### 三道健康门

1. final checkpoint 在完整 train split 上的 correctness BCE 相对常数类别先验至少改善 1%；改善方向
   错误也失败。当前代码已用 checkpoint no-grad 全量 evaluation 实现；历史 v4 仍保留旧错误口径。
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
- schema 可不持久化 full-width feature，但不能只保存裸 prompt/output 文本：精确
  `prompt_token_ids/output_token_ids` 及冻结 provenance 是 source of truth。online/precomputed 是
  协议变量，同一正式矩阵必须统一。
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
src/clir_hidden_states.py              exact-token online 全层 feature 重现
train_clir.py                          训练、恢复和 checkpoint
score_clir.py                          打分与 score 健康门
evaluate_clir.py                       BoN 与排序健康门
summarize_clir.py                      多 seed / 失败矩阵汇总
scripts/run_stage1b_validation.py      唯一正式 launcher
scripts/run_semantic_rewrite_pilot.py  rewrite prepare/extract/audit 唯一入口
scripts/run_verifier_selection_v1.py   独立 verifier 的 blind run / score 入口
src/clir_reasoning_rewrite.py          reasoning-equivalence 与分层 acceptance 契约
```

已删除的 `scripts/diagnose_stage1b.py` 不再是入口：其候选顺序、排序和矩阵检查已由 evaluator、summary
与 launcher 的强制契约覆盖。不要重新引入另一套诊断口径。

## 7. 当前下一步：hallucination localization Pilot v1

v4 的修正诊断表明 encoded/CLIR outcome-only 训练在当前预算下不稳定，可逆格式 pilot 也已把
`semantic_id/style_id`、teacher-forced token 对齐、全 33 层 feature 提取和 consistency pair 审计
跑通。generator/verifier 的首版 model selection 由
`configs/semantic_rewrite_models_v1.json` 冻结；失败后使用的 Falcon3 fallback 由
`configs/semantic_rewrite_models_v2.json` 冻结；7.61B Qwen scale diagnostic 由
`configs/semantic_rewrite_models_v3.json` 冻结。三者的同 checkpoint verifier 都只是 self-verification，
`formal_mechanism_claim_allowed=false`。`configs/semantic_rewrite_llm_v1` 至 `v7` 与
`scripts/run_llm_semantic_rewrite.py` 已把 prompt、严格 parser、greedy decoding、source-atomic resume、
answer/numeric/equation/line gate、token lineage 和人工盲审模板跑通。

4-source × 2-style 门仍未通过。v5 已实现“模型只返回带 ID 的可编辑行、程序原样回填保护行和空行”，
Qwen 结果为 0/8：6 条可解析输出仍改变数字 occurrence，另 2 条标签输出不完整。v6 保持 v5 的
source、prompt、greedy decoding 和全部 8/8 阈值，仅把 generator/verifier 切到本地
`tiiuae/Falcon3-3B-Instruct@411bb943...`，结果仍为 0/8，且 8 条全部是标签协议解析失败。v5/v6
protocol SHA256 分别为 `e67f01309b2b633304c96425628f90470ce1bc35092f973c16ff426675dc36be`
和 `34f280ddb2d320e152e586e9b93e87f21a2b60bd01a6f33a008332cb0b27dd5f`。所有执行仍是 dirty-tree
train-only diagnostic；失败 artifact 不回写、不删除。

v7 保持 v5 的所有非模型因素，只把 generator/verifier 切到本地
`Qwen/Qwen2.5-7B-Instruct@a09a354...`。结果为 1/8：3 条 generator 标签不完整；其余 4 条拒绝主要来自
模型在叙述行中删掉、增加或重复数字。5 条可解析输出均被同 checkpoint verifier 接受，进一步说明
self-verification 不能替代确定性 gate。模型生成 8 次共约 19.50 秒，5 次 verifier 共约 4.33 秒；
这是单次 L20Z pipeline timing，不是正式吞吐 benchmark。v7 的 audit 明确
`scale_authorization_eligible=false`，不得据此扩规模。

不要继续针对这 4 条已反复查看的 Qwen gate 样本调 prompt，也不要把 v1-v7 的失败输出纳入训练。
Route A v1 的历史 62 行也不得继续训练。v1a 已关闭最小修复门：原 source/hash 不变、双向 material
claim 双标、1 个分歧独立裁决、27 组注入 4096 行 mixed train、匹配 baseline/consistency 训练与评分均
已完成。机器协议/结果分别是 `configs/on_policy_pilot0_v1a/training_protocol_v1.json` 和
`configs/on_policy_pilot0_v1a/training_result_v1.json`。

下一步按 `docs/hallucination_localization_pilot_v1.md` 执行，只从 train-primary 构造 64-row
correct/incorrect 分层 selection manifest，冻结领域通用 hallucination/onset 定义、exact-token span
mapping 和双标协议。selection 现已完成：32/32 outcome 分层、每类 4 个长度 bin 各 8 条、64 个不同
query，并排除已反复审核的 Route A 31 query；blind items 只含 item/problem/trajectory。Phi tokenizer
preflight 对 64/64 行完成 exact encoded-prefix、exact decode 和完整 offset coverage，均只尾随 `[32007]`。
当前执行 24B candidate primary；到 secondary blind package 生成后再请求第二标注者。首轮不把
consistency、localization、dual-prior 同时混训。Phi self-rewrite Route B 仍保留，但不阻塞 localization
Pilot-0。

incorrect trajectory 是否需要 augmentation 仍是开放问题。当前 8 个候选对不足以形成可靠错误机制组，
不阻塞 Pilot-0；本轮让 base manifest 的错误轨迹继续提供 outcome BCE，但不给它们 consistency label。
后续若定向增加 Phi 错误采样，再由 verifier 要求 same error mechanism / semantic error location /
downstream effect，比较 correct-only 与 correct+incorrect augmentation ablation。

模型选择文件只冻结 checkpoint 与声明边界，不等价于完整 rewrite protocol。Qwen-7B v7 的
Apache-2.0 与历史 Qwen/Falcon checkpoint 的 provenance、attribution 和发布条件必须在公开数据或模型
之前再次核对。
