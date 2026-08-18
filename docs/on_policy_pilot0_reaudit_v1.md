# Route A Pilot-0 重新审核与 v1a 修复

最后更新：2026-08-18

证据等级：`pipeline pilot`

状态：**v1 已部分解冻；旧训练资格保持撤销；v1a 双标、裁决和修复训练门已完成。**

## 1. 解冻范围

本次不是推翻 on-policy equivalence mining。以下事实仍有效并继续冻结：

- 125 对候选均来自 Phi-3.5-mini 在原始 task prompt 与正式 decoding policy 下的原生采样；
- source manifest、candidate manifest、原始 row/token IDs、feature 路径和各级 SHA256 血缘有效；
- `query_id` 与 source split 未漂移，候选构建可精确重算；
- Route A 仍是当前主方向，Phi self-rewrite 仍是后续 Route B。

撤销的是旧 verifier 之后的训练资格：

- `verifier_mistral24_v1/all_results.jsonl` 的 121 个 Silver accept 不再作为 relation label；
- `published_silver_v1/train_online.jsonl` 的 31 组/62 行只保留为历史 roster 与诊断，不得训练；
- 基于这 62 行的训练、representation diagnostic 和 loss-weight sweep 只保留为失败机制/几何诊断；
- `negative_consistency_weight=0.5, score_consistency_weight=1.0` 不再视为下一轮冻结配置；
- hallucination localization 设计文件保留，但执行暂停到 rewrite 修复训练门通过以后。

机器可读状态在
`configs/on_policy_pilot0_v1a/repair_protocol_v1.json`。旧 artifact 不删除、不覆盖、不原地改标签。

## 2. 为什么必须修

### 2.1 旧 operational verifier 没有执行已设计的双向审核

`src/clir_reasoning_rewrite.py` 与 v8 设计要求：

1. source→rewrite 检查所有 essential material claims；
2. rewrite→source 检查所有 substantive 新 claim；
3. 单独检查内部错误机制；
4. 对自动发现的数字/否定/量词等风险逐项解释；
5. acceptance 由程序从证书推导。

旧 Route A runner 实际只要求一个 flat Boolean JSON。125 条 annotation 的语义字段几乎完全模板化，
四条 reject 的具体 reason 也没有反映到对应 Boolean。JSON 可解析只能证明 schema 语法有效，不能证明
relation judgment 有证据支撑。

### 2.2 correctness 被误用成“过程无错”

旧指南强制 correct-only pair 的 `error_alignment_applicable=false`。这会漏掉最终答案正确、但中间含有
错误陈述的轨迹。已确认的例子 `OPA-26a8d95781569cde` 在 expanded trajectory 中同时声称
“1350/60 等于 22 且无余数”和“余数是 30”；最终 22 小时 30 分及 1350 分钟都正确，但过程新增错误。

当前 acceptance 契约已修正：

- incorrect outcome 仍必须提供错误对齐；
- correct outcome 可以没有内部错误，也可以显式声明内部错误；
- 一旦声明内部错误，机制、语义位置和 downstream effect 都必须保持；
- correctness 不再决定 `error_alignment.applicable`。

### 2.3 旧 style ID 是候选顺序，不是 style

`native_anchor/native_alternate` 由 candidate index 决定。它不能支持
“different semantic + same style”的负 consistency 解释。v1a 按 exact output-token 数把每对定向为：

- 较短：`native_compact`；
- 较长：`native_expanded`；
- pair target：`native_compact_to_expanded`。

这仍只是相对长度/说明粒度轴，不宣称是领域无关的完整风格 taxonomy，但至少不再用任意候选角色冒充
style。

### 2.4 62 行 all-positive 训练不是有效对照

旧 published manifest 的 62 行全部 correctness=1，导致 outcome BCE 退化为单类别并出现全局
representation collapse。negative consistency 项随后主要学习跨 query 排斥，不能解释为 style
invariance。

修复后不再复制 62 行单独训练。通过的 pair 只给原 4096 行 base train manifest 中对应的原生 row
增加 `semantic_id/style_id`；其余 4096 行、包括 incorrect outcome rows，继续参与 correctness BCE。
首轮不要求错误解法也有 rewrite。

## 3. 最小可训练修复，而不是 Gold 数据工程

按用户确认，本轮采用宽松 Silver 门：

- 只重审历史发布 roster 的 31 对，不重跑 125 对，不为 reject query 回填候选；
- claim inventory 只覆盖前提、方法、关键中间量、结论和内部错误，不逐句穷举；
- 允许步骤拆并、重组、直接蕴含的解释桥梁、以及不改变实际值/结论的近似措辞；
- 不同方法、关键遗漏/矛盾、新错误、错误修复/迁移或不同结论仍为硬拒绝；
- 两份独立标注有分歧才裁决；最终通过多少就训练多少，不设最低 31 组要求。

盲 item 不含 correctness、旧 verifier decision 或旧 reason。每个 item 都附程序生成的软风险提示；它们
不是硬门，但 annotator 必须逐项核算并解释。

## 4. v1a 当前产物

目录：`configs/on_policy_pilot0_v1a/`。

| 产物 | SHA256 | 状态 |
|---|---|---|
| `annotation_items_v1.jsonl` | `8800150af88f021b4ffce23bd1918036eedd58b3da1663bbed5139a475d6a1c8` | 31 个 blind items |
| `annotation_lineage_v1.jsonl` | `3b10b028cf4c2ea2517495411b27d199972448c39641a7d030fdbec8ad5cfe1a` | 私有 source/checker/hash 血缘 |
| `annotation_guide_v1.md` | `90dfe4589d46e1184ef5105df821450a6ead4a8d1fc2ba7eeb742367b597ee18` | material-claim 宽门指南 |
| `labels_primary_v1.jsonl` | `31ddd5f5b5810bb86d340e81ddb3f2629dd20a9014a52c607f88905cf7a4ff9e` | 第一标注，结构校验通过 |
| `secondary_prompt_v1.md` | `5e912d9bdeed567eadd514920513461aa418407a433be9e9584a04a6d5ffa7d0` | 可直接交给第二 AI |
| `labels_secondary_v1.jsonl` | `861578c398a026ac4d9b35dc4e71a9aaf68c230a92491608b71f7a07907c32e7` | 第二标注，结构校验通过 |
| `adjudicated_relations_v1.jsonl` | `b874fd07280e501ff1515ee3ad301fb6b8eeaabf29ae9f7752c62493ccba810e` | 最终 27 accept / 4 reject |
| `training_protocol_v1.json` | `0e2497841e4308c0b0cceff1f19980f2c9d54ecb06970aa87ba75fafdebc25cd` | matched 1-epoch pipeline 协议 |
| `training_result_v1.json` | `9a371b9a7a05df28c84e56ad9471a061088fea79a2cb3b955c5f1ef9fad830ea` | 训练、评分与表示诊断摘要 |

第一标注的程序派生状态为 28 accept / 3 reject。三条 reject 是：

- `OPA-26a8d95781569cde`：新增“1350/60 无余数”的错误；
- `OPA-817dc95df11db9aa`：新增“总票数不受每局票数影响”的错误，而同一轨迹定义总票数为 `5x+5`；
- `OPA-4b4c8fa805f51e49`：把精确的 `33/3=11` 错称为向下取整。

第二份派生状态为 27 accept / 4 reject。两份 status 一致 30/31（`0.9677`），Cohen's kappa
`0.8394`。唯一分歧 `OPA-5d63fcc016593ab6` 的 rewrite 把 `0.6667 * 60` 写成精确等于 40；实际为
40.002，因此按“不得新增错误”裁为 reject。最终不是 Gold verifier 质量结论，只是本轮 Silver
training relation set。

## 5. 已执行的最短路径

1. `scripts/audit_on_policy_reannotations_v1.py` 校验了第二标注并计算 agreement；
2. `scripts/adjudicate_on_policy_reannotations_v1.py` 只裁决唯一分歧，保留两份原始标签；
3. `scripts/materialize_on_policy_mixed_v1.py` 生成 4096 行 mixed precomputed train，仅为 54 个已存在
   source rows 添加 `semantic_id/style_id`，没有复制 trajectory；
4. 原 row/query/token/feature/correctness/checker provenance 保持，validation 取 32 个 query/512 行且与
   train query overlap 为 0；
5. 同一 seed 42、同一 batch 顺序、同一 1-epoch 预算依次跑 consistency=0 与 1；训练参数除输出路径和
   consistency 总权重外一致；
6. 两个 cell 均观测 3668 correct、428 incorrect、27 个正 pair 和 26 个实际 batch 负 pair，且 finite。

该训练仍是 `pipeline pilot`。能运行和 loss 非零只证明模块可训练，不证明 CLIR 改善 Best-of-N。

## 6. 修复训练结果与解释

| 指标 | baseline (`consistency=0`) | consistency (`consistency=1`) |
|---|---:|---:|
| checkpoint full-train BCE | 0.2389 | 0.2182 |
| 32-query validation BCE | 0.5074 | 0.4600 |
| checkpoint consistency total | 0.01064 | 0.00592 |
| validation score population std | 1.6911 | 1.7893 |
| mixed-query pairwise accuracy | 0.5702 | 0.5871 |
| BoN@4 / @8 / @16 | 0.875 / 0.84375 / 0.875 | 0.84375 / 0.90625 / 0.875 |

validation 的 32 个 query 中只有 10 个同时有正确/错误候选；BoN 波动既有正也有负，不能解释成稳定
ranking gain。它足以说明在当前宽 Silver 门下，consistency 模块可以训练且没有把 correctness 学习
打坏。

更重要的是表示机制：baseline 的 same-semantic projected cosine 为 `0.9989`，但
different-semantic/same-style 也达到 `0.9926`，接近全局 collapse。consistency cell 的两者分别为
`0.9827` 和 `0.7305`，正负 gap 从 `0.0064` 扩到 `0.2522`；27 对中只有 1 对的正 cosine 高于
baseline，但 19 对的 absolute score delta 更低。因此本轮主要学到的是**跨语义分离，同时相对保留
正对**，不是“继续提高正对相似度”。该诊断只覆盖训练中的 27 对，仍不能证明 unseen equivalence
generalization。

修复训练门据此通过，允许恢复 hallucination localization Pilot v1。Route A 后续若扩量，必须另建
held-out equivalence pair split；不能用这 27 对的 in-training geometry 当泛化证据。
