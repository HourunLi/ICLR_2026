# CLIR semantic rewrite v8：reasoning-equivalent 设计协议

最后更新：2026-08-17

状态：第一阶段纯函数契约已实现；尚未实现模型 launcher，且未冻结 generator/verifier checkpoint、source
manifest、prompt、阈值或 blind protocol。

证据层级：`pipeline pilot`。本文不构成 rewrite 质量、consistency loss 合理性或 CLIR 机制效果证据。

## 0. 决策摘要

v8 将 rewrite 关系从“逐行、逐数字保持的同轨迹表面改写”改为
`reasoning_equivalent`：允许改变措辞、行数、局部组织、步骤拆分/合并和非因果顺序，但必须保持核心前提、
推理方法、关键推导、结论、correctness，以及 incorrect source 的核心错误机制。

冻结以下设计方向：

- generator 继续使用约 7B 的本地 instruct 模型，一次生成完整 trajectory；
- verifier 使用独立的几十 B instruct 模型，不能与 generator 共用 checkpoint；
- verifier 做双向 claim alignment，而不是只输出全局 accept/reject；
- 最终状态由程序从结构化 verifier 字段保守派生；
- 数字、单位、实体、否定、比较和量词变化是软风险信号，不是通用硬等价门；
- GSM8K checker 是可选领域证据，不是 `reasoning_equivalent` 的定义；
- 现有 CLIR loss 暂不修改，v8 accepted group 继续映射到 `semantic_id/style_id`；
- rewrite row 不继承 source 的 token-level/onset/prior targets；
- v1--v7 artifact 和原 v8 span-locking 审计方案保留为历史，不覆盖、不改写成已实现事实。

原 span-locking 方案由本文 supersede 为未来可选的 `surface_equivalent` 路径，不再是 v8 主路径。

截至 2026-08-17，`src/clir_reasoning_rewrite.py` 已实现严格 verifier JSON/parser、双向 claim schema、
可选 domain-checker 契约、程序派生的 `accepted/rejected/review_required` 状态，以及第一版通用软风险探针；
`tests/test_clir_reasoning_rewrite.py` 覆盖其核心 fail-closed 行为。该状态只证明纯函数契约通过测试，不代表
任何几十 B verifier 已被选择、运行或校准。

同日，`configs/verifier_selection_v1/` 已生成模型盲的 64-pair 双标注 selection package：16 个 base
sources × 4 个受控 variants，覆盖四个领域、32 correct/32 incorrect source、两个 style 各 32 条，primary
分布为 24 accept/32 reject/8 review。匿名 items 不携带 source correctness 或 gold 字段；secondary 只能
读取 items、annotation guide 和隔离 prompt。当前等待独立 secondary annotation，尚未运行任何候选 verifier。

## 1. 关系定义

### 1.1 接受的变化

`reasoning_equivalent` rewrite 可以：

- 改变语气、句式、术语选择和解释风格；
- 合并或拆分步骤；
- 改变物理行数和篇幅；
- 删除不承载新信息的纯重复复述；
- 增加不承载新事实的连接、过渡和解释性措辞；
- 补充由 source 既有前提和推理方法直接蕴含、但原文省略的推导桥梁；
- 在不改变精度和主张的前提下使用等价表示；
- 重排彼此独立的解释，或按逻辑依赖重新组织叙述；
- 把自然语言关系改写为等价的符号表达，或反过来。

### 1.2 必须保持的不变量

rewrite 必须保持：

1. 同一个问题、任务和目标；
2. source 实际使用的核心前提；
3. source 的核心推理方法，而不是换一条新解法；
4. 关键中间推导及其依赖/因果方向；
5. source 的最终结论；
6. source 的 correctness 状态；
7. source 中的重要限定、不确定性和适用条件；
8. incorrect source 的核心错误机制及其下游后果；
9. 不引入 source 未使用的外部事实、证据或假设。

### 1.3 明确不要求保持的内容

v8 不要求：

- 数字 literal 的字符串、出现次数或全文顺序完全相同；
- 单位或表达形式逐字相同，只要对应主张严格等价；
- 相同行数、相同步骤编号或逐行对应；
- protected math line byte-exact；
- 相同 tokenization、token 数或 token-level 对齐；
- 相同 hallucination onset token index；
- 完整 trajectory representation 在研究意义上应完全相同。

最后一项与现有 loss 存在尚未验证的张力。按项目负责人决定，v8 暂不修改 loss；任何训练结果都必须把
“现有表示一致性目标是否适合 B relation”列为开放研究问题。

### 1.4 关键反例

以下不属于 `reasoning_equivalent`，即使答案相同：

- source 用比例法，rewrite 换成独立方程法；
- source 依赖事实 A，rewrite 删除 A 并引入新事实 B；
- source 在步骤 2 算错，rewrite 修正步骤 2、再在步骤 4 制造另一个错误；
- source 表达“可能”，rewrite 改成“必然”；
- source 的最终答案碰巧与 rewrite 相同，但核心推理方法或错误机制不同。

## 2. 身份和监督契约

### 2.1 身份字段

训练兼容路径继续写入：

```json
{
  "query_id": "source query identity",
  "semantic_id": "semrw-reasoning-v8:<source_trajectory_id>",
  "style_id": "declared style axis"
}
```

另外必须在 lineage/certificate 中写入更明确的关系字段：

```json
{
  "source_trajectory_id": "...",
  "reasoning_group_id": "semrw-reasoning-v8:<source_trajectory_id>",
  "relation_to_source": "reasoning_equivalent",
  "relation_protocol_version": "clir-reasoning-equivalence-v1"
}
```

当前训练代码只消费 `semantic_id/style_id`；`reasoning_group_id` 是防止下游把 B relation 误解释为
token-exact equivalence 的显式元数据。`query_id` 仍只表示 Best-of-N/query split group，不能与
`semantic_id` 或 `reasoning_group_id` 混用。

### 2.2 group 成员

一个可发布的 consistency group 至少包含：

- source original view，`style_id=original`；
- 两个 accepted rewrite views，使用两个不同且预注册的 `style_id`；
- 三者共享同一个 `semantic_id/reasoning_group_id`。

如果当前训练 manifest 不纳入 original view，必须在实现协议里明确 pair 构造方式；不得让两个 rewrite
在没有 source lineage 的情况下形成不可审计 group。

### 2.3 允许继承的监督

accepted rewrite 可以继承或重新计算：

- `query_id` 和 split membership；
- source correctness，前提是 task checker 与 verifier 同时确认未改变；
- `semantic_id/style_id` consistency metadata；
- 从 rewrite 自身 exact output token IDs 在线抽取的 hidden states；
- trajectory-level reward/correctness supervision。

### 2.4 禁止继承的监督

rewrite row 不得从 source 复制：

- `output_token_ids` 或任何 exact-token hash；
- `hallucination_onset` token index；
- onset label mask；
- token advantage/progress target；
- key/complete/reconstruction token target；
- 任何依赖 source token 位置的标签。

若 schema 需要这些字段，必须使用显式 applicability mask 表示“未标注”，不能用 `-1` 冒充缺失；项目中
`hallucination_onset=-1` 的语义是已知无 hallucination，而不是 unknown。

## 3. Generator 协议

### 3.1 模型边界

generator 是 candidate proposer，不是 correctness 或 relation authority。

冻结协议时必须记录：repo ID、不可变 revision、tokenizer revision、license、dtype、runtime、chat template、
decoding 参数、prompt hash、最大上下文和最大输出长度。默认本地 BF16、greedy decoding；任何 sampling、
checkpoint、prompt 或 decoding 变化产生新 protocol hash 和新 output root。

### 3.2 输入

generator 可看到：

- style instruction；
- problem；
- 完整 source trajectory；
- `reasoning_equivalent` 的允许/禁止变化定义。

generator 不得看到：

- reference answer；
- source correctness label；
- task checker 结果；
- verifier 输出；
- blind source 的历史人工诊断。

### 3.3 输出格式

完整 trajectory 使用：

```text
<rewritten_response>
...
</rewritten_response>
```

parser 要求恰好一个 opening/closing tag、tag 外无非空内容、内部非空。内部允许多行。禁止 parser 自动补
tag、截断解释或从多个候选中挑选一个；异常一律 `generator_output_invalid`。

### 3.4 Prompt 语义

system prompt 至少包含：

```text
You are rewriting an existing reasoning trajectory, not solving the problem again.
Create a meaningfully different presentation and organization while preserving the
same core premises, reasoning method, key inferences, intermediate conclusions,
final conclusion, and any original reasoning error. You may merge, split, or reorder
explanatory steps only when logical dependencies remain unchanged. Do not introduce
a new solution method, external fact, correction, unsupported claim, or new error.
```

具体 prompt 文本在 dev 校准后单独冻结；本文只冻结语义要求，不冻结最终 hash。

### 3.5 Style 轴

首版保留两个明显不同但不指定推理方法的 style，例如：

- `terse_formal`：正式、紧凑、减少非必要过渡；
- `explanatory_conversational`：自然解释、允许显式连接词和必要复述。

style 指令不得要求删除事实、增加推导、改变证据或换解法。是否形成足够表面差异由 dev artifact 的文本和
representation diagnostics 报告，不用 fidelity gate 强行要求每行改变。

## 4. 通用确定性检查和软风险探针

### 4.1 通用硬检查

生成后先执行模型无关检查：

- generator 输出严格可解析且 rewrite 非空；
- rewrite 与 source 不完全相同；
- style/prompt 指令未逐字泄漏；
- exact rewrite response SHA256 和 output token IDs 可重建并校验；
- feature tokenizer encode/decode 契约按当前 online hidden-state 路径满足；
- source、query、split、protocol 和 model lineage 完整；
- 不覆盖已存在 source shard 或 completion marker。

极端长度（建议初始诊断边界 `<0.25` 或 `>3.0`）只触发 `review_required`，不直接证明语义错误。边界需在
dev 校准后冻结。

### 4.2 软风险探针

程序为 verifier 生成差异摘要，至少包括：

- numeric/fraction/percent/currency changes；
- unit changes；
- named entity 和主要名词短语变化；
- negation changes；
- quantifier changes；
- comparative/direction changes；
- modality/uncertainty changes；
- causal/temporal connective changes；
- line count、length ratio 和 lexical similarity。

这些信号不能单独 accept 或 reject。它们必须原样进入 artifact，并作为 verifier 重点核查项。领域插件可增加
信号，但不得改变通用 relation 的定义。

### 4.3 可选领域 checker

checker interface 返回：

```json
{
  "available": true,
  "checker_id": "...",
  "checker_revision": "...",
  "source_outcome": "...",
  "rewrite_outcome": "...",
  "same_outcome": true,
  "source_correctness": 0,
  "rewrite_correctness": 0
}
```

GSM8K 使用冻结 numeric checker，并要求 normalized candidate answer 与 correctness 都保持。其他任务可用单元
测试、label verifier、citation checker 或不提供 checker。`available=false` 时不能伪造 correctness 硬证据，
必须由 verifier 加人工校准承担更高风险。

## 5. 独立几十 B Verifier 协议

### 5.1 独立性和模型选择

verifier 必须：

- 与 generator 不同 checkpoint；
- 参数规模为几十 B；
- 优先使用不同模型族以减少相关错误；
- 固定 repo ID、不可变 revision、tokenizer、license 和 runtime；
- 使用 deterministic decoding；
- 支持严格结构化 JSON 输出；
- 不读取 reference answer 或 source correctness label。

若只能使用同模型族的大 checkpoint，必须将 `same_model_family=true` 写入 provenance，并扩大人工校准与
对抗测试；不能描述为完全独立错误来源。

### 5.2 双向审计

每个 view 必须覆盖两个方向：

1. `source_to_rewrite`：所有 essential source claims、推导和错误是否被 rewrite 保留；
2. `rewrite_to_source`：rewrite 的所有 substantive claims 是否得到 source 支持，是否引入新方法/事实/错误。

实现可以是一条请求中的两个独立 section，也可以是两次确定性调用。若一次请求无法稳定覆盖两个方向，
v8-dev 必须切到两次调用；该选择在 blind 前冻结。

### 5.3 Claim relation 枚举

每个 claim alignment 的 `relation` 只能是：

- `equivalent`；
- `preserved_implicitly`；
- `entailed_elaboration`；
- `omitted`；
- `contradicted`；
- `replaced_by_different_reasoning`；
- `uncertain`。

`preserved_implicitly` 必须附 rewrite evidence 和解释，不能成为跳过所有中间推导的默认类别。
`entailed_elaboration` 只用于 rewrite 新增的显式推导桥梁；它必须由 source 既有前提和方法直接蕴含，
不能引入新前提、新证据、例子、类比、假想场景、不同解法或错误修正。

### 5.4 Verifier 输出 schema

以下是语义 schema；实现时转换为严格 JSON Schema 并拒绝未知字段、缺字段、重复 key 和非法枚举：

```json
{
  "schema_version": "clir-reasoning-equivalence-verifier-v1",
  "source_to_rewrite": {
    "claims": [
      {
        "claim_id": "S1",
        "source_claim": "...",
        "essential": true,
        "rewrite_evidence": "...",
        "relation": "equivalent",
        "explanation": "..."
      }
    ],
    "all_essential_claims_preserved": true
  },
  "rewrite_to_source": {
    "claims": [
      {
        "claim_id": "R1",
        "rewrite_claim": "...",
        "substantive": true,
        "source_evidence": "...",
        "relation": "equivalent",
        "explanation": "..."
      }
    ],
    "all_substantive_claims_supported": true
  },
  "global_relation": {
    "same_task_and_goal": true,
    "same_core_premises": true,
    "same_reasoning_method": true,
    "same_key_inferences": true,
    "same_intermediate_conclusions": true,
    "same_final_conclusion": true,
    "introduced_external_fact": false,
    "omitted_essential_claim": false,
    "contradicted_source_claim": false,
    "replaced_with_different_solution": false,
    "introduced_new_error": false
  },
  "error_alignment": {
    "applicable": false,
    "source_error": null,
    "rewrite_error": null,
    "same_error_mechanism": null,
    "same_semantic_error_location": null,
    "same_downstream_effect": null
  },
  "risk_review": [
    {
      "risk_id": "numeric-change-0",
      "resolved": true,
      "explanation": "2/3 and two thirds express the same value and role."
    }
  ],
  "confidence": "high",
  "decision": "accept",
  "issues": []
}
```

`confidence` 只能是 `high/medium/low`，`decision` 只能是 `accept/reject/review`。模型的 decision 仅用于一致性
检查，不是最终权威。

### 5.5 程序派生状态

程序从结构化字段派生：

```text
AUTO_ACCEPT iff
  verifier JSON valid
  AND all essential source claims ∈ {equivalent, preserved_implicitly}
  AND all substantive rewrite claims == equivalent
  AND all risk items resolved
  AND all positive global invariants true
  AND all prohibited global changes false
  AND confidence == high
  AND model decision == accept
  AND applicable domain checker gates pass
  AND (correct source OR strict incorrect-source conditions pass)
```

correct source 的 strict conditions：checker 可用时 correctness 与 outcome 都保持；checker 不可用时必须在
dev 人工校准中单独报告。

incorrect source 的 strict conditions：

- checker 可用时 source/rewrite 都 incorrect，且 outcome 保持；
- `error_alignment.applicable=true`；
- `same_error_mechanism=true`；
- `same_semantic_error_location=true`；
- `same_downstream_effect=true`；
- 没有“修复原错误后另造新错误”。

明确 contradiction/omission/addition/method replacement/new error 派生 `rejected`。其他不完整、不确定、
medium/low confidence、未解决风险或模型/program decision 不一致派生 `review_required`。任何 parser/runtime
异常派生专门错误状态，不能混作语义 reject。

## 6. Calibration、development 和 blind 协议

### 6.1 v8-regression：历史暴露 source

先使用 v1--v7 已暴露的 4 个 train source 和人工构造对抗例，不产生 blind claim。目标是：

- 验证完整 trajectory generation；
- 验证 JSON parser 和程序派生规则；
- 建立 claim alignment/error alignment 失败分类；
- 检查 verifier 是否只看最终答案；
- 测量 correct/incorrect source 的差异；
- 校准 `accept/reject/review`，不授权 scale。

### 6.2 Verifier calibration set

在选择 blind source 前冻结一个仅来自 train/合成反例的校准集，至少覆盖：

- 合法压缩、拆分、合并和独立解释重排；
- 等价数字/单位/符号表达；
- 否定、量词、比较方向、主体、模态和因果方向漂移；
- essential claim omission；
- unsupported addition；
- 相同答案但替换解法；
- incorrect source 被修正；
- 同答案但错误机制/错误位置改变；
- parser 格式攻击和 prompt leakage。

人工 gold 必须先冻结。报告 confusion matrix，并分别报告 correct、incorrect、自动 accept 和 review。scale 前的
首要质量指标是自动 accepted view 的 precision；阈值在观察 blind 文本前冻结。

### 6.3 v8-dev

从 train 选择 query-disjoint、与历史和 future blind 互斥的 development sources。可查看输出和修改 prompt、
schema、模型或派生逻辑；每次语义变化使用新 protocol hash/output root。

v8-dev 是工程开发，不用于估计总体错误率。必须报告：planned/generated/parse-valid/checker-pass/
auto-accept/review/reject、完整 group 数、人工 false acceptance、错误机制保持、调用 tokens、wall time、显存和
每个完整 group 成本。

### 6.4 v9-blind

在 generator prompt、verifier checkpoint/prompt/schema、程序派生规则、soft probes、domain checker、style、
selection algorithm 和 runtime 全部冻结后，从 train manifest 选择未暴露 query。

建议最小 blind：8 sources × 2 styles = 16 views，4 correct + 4 incorrect。主 gate 以训练消费的完整 group
为单位：

```text
complete_reasoning_groups >= 7 / 8
confirmed_semantic_false_accepts == 0
incorrect_source_repairs == 0
accepted_incorrect_error_mechanism_changes == 0
accepted_groups_with_unresolved_risk == 0
```

具体门槛必须结合 calibration 结果预注册；以上数字是待确认设计，不是已冻结协议。

blind 默认只公开聚合计数。per-view 文本和 claim alignment 放入 sealed artifact；显式 unseal 必须写入事件
记录并使这些 source 永久变为 exposed。blind 失败后不能在同一 source 上调协议并再次声称 blind。

## 7. Artifact、resume 和 fail-closed

### 7.1 Source-atomic shard

一个 source trajectory 的 original metadata、两个 generated styles、risk probes、checker、verifier outputs 和
派生状态构成原子 shard。只有所有 planned views 到达终态后才写 completion marker。

每个 marker 至少绑定：

- source payload bytes/SHA256；
- source/query/split identity；
- source manifest SHA256；
- protocol SHA256；
- generator/verifier model revision；
- generator/verifier prompt SHA256；
- runtime provenance；
- 每个 completion text/token IDs/SHA256；
- risk/checker/verifier result SHA256；
- 派生 acceptance policy version；
- shard payload bytes/SHA256。

### 7.2 Resume

- 健康 completed shard 只读复用；
- marker 缺失、payload/hash/schema/config 漂移一律视为 incomplete/corrupt；
- 不覆盖损坏 shard，由显式 quarantine/rebuild 流程写入新 output root 或新 attempt identity；
- 单个 verifier 方向失败不能把另一个方向的结果当成完整证书；
- audit 只消费 marker 验证通过的 shard；
- 重复启动、部分失败或手工改文件必须 fail closed。

### 7.3 Relation certificate

accepted row 的 lineage 必须包含完整 relation certificate hash，证书至少覆盖：

- source/rewrite hashes；
- generator request/response hashes；
- soft risk probe；
- domain checker；
- verifier request/response 与结构化 parse；
- program-derived checks/status；
- human adjudication（若由 review 升级为 accepted）；
- relation、group 和 style identity；
- exact rewrite token-ID hash。

人工不能直接改模型输出或 verifier JSON；只能追加不可变 adjudication record，重新派生最终状态。

## 8. 发布和训练兼容性

只有以下 group 可以进入 consistency manifest：

- source lineage 完整；
- 所需 styles 全部 accepted；
- relation certificate 验证通过；
- rewrite exact token IDs 从 rewrite 自身获得；
- hidden states 对这些 exact IDs 在线 teacher forcing 抽取；
- `semantic_id/style_id` 非空且 group batching 中实际形成 positive pairs；
- 不携带 source token-level/onset labels。

现有 `prism_style_consistency_loss` 把相同 `semantic_id`、不同 `style_id` 当 positive pair，并同时施加表示与
score consistency。v8 暂不改该行为，但所有实验报告必须同时记录：

- positive pair 数和 applicable batch 数；
- pair 的文本/长度/representation distance 分布；
- consistency positive、negative、score 三项 loss；
- 与不使用 rewrite 的相同数据/预算 baseline；
- B relation 与现有 loss 是否匹配仍未验证。

pipeline 通过只说明数据和实现能工作，不说明 rewrite 有益或 loss 合理。

## 9. 实现分层

不得把 v8 直接塞进 v7 的版本分支。建议新增独立模块或明确的新 schema：

1. `relation_protocol`：v8 config 验证、identity 和状态枚举；
2. `generator_io`：完整 trajectory prompt 和 tagged parser；
3. `risk_probe`：通用软差异摘要；
4. `domain_checker`：可选 checker interface，首个 adapter 为 GSM8K；
5. `verifier_io`：严格 JSON Schema、双向 claim alignment parser；
6. `acceptance_policy`：纯函数式程序派生；
7. `artifact_store`：source-atomic shard、marker、resume、seal/unseal event；
8. `publisher`：只发布完整 accepted reasoning groups。

纯函数层必须可在无 GPU/无模型条件下通过 fixture 测试。模型 launcher 只负责 completion 和 runtime
provenance，不能包含隐藏的 acceptance 逻辑。

## 10. 必须测试的攻击面

### 10.1 单元测试

- tagged generator parser 的空输出、重复 tag、tag 外文本和多候选；
- verifier JSON unknown/missing/duplicate key、非法 enum/type；
- 每个 claim relation 对派生状态的影响；
- model decision 与 program decision 不一致；
- checker unavailable 路径；
- correct/incorrect 分支和 error alignment；
- review 经 append-only human adjudication 升级；
- token-level supervision stripping；
- group 只在所有 styles accepted 时发布。

### 10.2 对抗语义样例

- not/never/without；
- more/less、at least/at most；
- each/some/all；
- before/after 和因果方向；
- 主体/宾语交换；
- 可能/一定；
- 数字等价、近似、单位换算和精度丢失；
- 删除 essential intermediate claim；
- 同答案不同解法；
- 修复旧错误并制造新错误；
- 保持答案但改变错误机制；
- 伪造 source quote 或把 prompt 内容写进 rewrite。

### 10.3 集成和 artifact 测试

- tiny 2-source × 2-style fake-backend end to end；
- generator/verifier 中断和 resume；
- payload/marker/request/response 任一字节篡改 fail closed；
- protocol/model/prompt/runtime drift 拒绝 resume；
- healthy shard 不变，只有 incomplete source 被重建；
- deterministic merge order；
- sealed blind aggregate 与显式 unseal event；
- 发布 manifest reload、exact token IDs、semantic-group batching 和真实 positive pair 计数。

## 11. 实施顺序和 gate

1. 冻结本文的 relation/schema 草案，不选择 blind source；
2. 实现纯函数 verifier schema、acceptance policy、risk probe 和测试；
3. 实现完整 trajectory generator path，复用 v7 暴露 source 做 regression；
4. 选定并固定几十 B verifier checkpoint，完成 runtime smoke；
5. 构建人工 gold calibration/对抗集，测自动 accept precision；
6. 在新的 query-disjoint train dev sources 上迭代；
7. 冻结 prompt、模型、阈值、checker、selection 和 artifact 协议；
8. 再选择并封存 v9-blind source manifest；
9. blind gate 通过并完成人工审计后，才讨论 32-source scale；
10. scale 数据验证后，才运行带 rewrite consistency 的对照实验。

当前允许：设计、实现、历史 source regression、calibration 和 v8-dev。

当前不允许：声称 blind、授权 scale、读取 `pilot_test`、发布训练 augmentation、从 pipeline pass 推断 CLIR
机制有效。

## 12. 冻结前未决策项

以下项目必须在实现或 calibration 后由项目负责人裁决：

1. 几十 B verifier 的具体 repo ID、revision、模型族、部署和 license；
2. 双向审计采用一次请求双 section，还是两次独立请求；
3. style prompt 的最终文本；
4. soft-risk 的极端长度边界及是否有其他 review routing 阈值；
5. checker unavailable 任务是否允许自动 accept，还是全部进入人工 review；
6. `review_required` 经何种人工协议升级为 accepted；
7. v9-blind 的 source 数、完整 group 门槛和 unseal 流程；
8. original view 是否与两个 rewrite 一起写入 consistency manifest；
9. 何时对 B relation 与当前 representation loss 的兼容性做 ablation。
