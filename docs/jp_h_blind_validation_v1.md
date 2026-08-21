# JP → H Mixed-Domain Blind Validation v1

最后更新：2026-08-21

## 1. 结论

固定 3-token centered logit smoother 没有通过新的 mixed-domain 盲测，状态冻结为
`blind_gate_failed_smoother_rejected`。两套 resolved annotation views 均为 0/3 seed 通过，不能采用
smoother，也不能把 `p_hallucination` 接入 reward。

更重要的诊断是：smoother 虽稳定降低 token-level 高频抖动，但 raw/smoothed H 都明显落后于同一批数据上的
绝对位置基线。当前问题不是再挑一个平滑窗口，而是 H branch 尚未证明学到了可迁移、超越位置捷径的
hallucination signal。

## 2. 这轮回答什么

旧 64 条数据已经参与 linear probe 和 window=3 的结构选择，因此其 3/3-seed 工程结果不能作为采用证据。
本轮在标签打开前冻结以下对象，并在一批全新的 Phi-native trajectories 上检验：

- seeds 42/43/44 的旧 64 条训练所得 final H heads；
- paired raw H probabilities；
- fixed 3-token centered logit mean；
- exact reviewed-claim token AP 与 claim-mean probability AP；
- 同行 absolute/normalized position baselines；
- 每个 seed 的四项严格优势和双 resolved-view 采用门。

本轮不训练新参数、不选阈值、不扫描 window，不使用 correctness、answer key 或 tail target。

## 3. 数据与盲性

盲测共 96 条新轨迹，每个 query 取一条 Phi-3.5-mini 原生 trajectory：

| Domain | Rows |
|---|---:|
| GSM8K | 24 |
| ARC-Challenge | 24 |
| CommonsenseQA | 24 |
| BoolQ | 24 |

总计 96 条，其中 72 条来自非数学领域，输出共 29,148 个 Phi tokens。所有 label span 都映射到保存的
`output_token_ids`；没有通过重新 tokenize 文本生成 target。

冻结预测文件为
`run_artifacts/jp_h_blind_validation_v1/frozen_scoring_v1/predictions_v1.jsonl`，SHA256：
`16057bd28d7c57a8f15ef2f490c6c44402926fd63449688d312dd4923c719345`。预测在标注期间已生成但数值
sealed；比较、裁决和最终 evaluation protocol 均在读取预测数值前冻结。

`pilot_test` 与 `final_test` 未访问。domain 只用于事后描述，不参与门控或方法选择。

## 4. 双标与裁决

secondary 按“每完成一条立即 durable 落盘”的 resumable contract 完成 96/96：

- raw labels SHA256：`118952a3200048675595c618476c75b2e7a4a8349573ccb4868150dd5e864b4b`；
- structural validation：passed；
- primary path：70 clean / 26 hallucinated；
- secondary path：60 clean / 36 hallucinated；
- path agreement：70/96（`.7292`）；
- Cohen's kappa：`.3882`；
- exact positive token set agreement：54/96；
- 42 条需要 blocking adjudication；negative reviewed-token coverage 的非阻塞差异为 94/96。

blocking 只包括 path status 或 exact positive token set 不一致。supported/non-claim coverage 不一致不会被
强行融合，因为不同完整 claim segmentation 会改变 claim-level AP；它们被保留为两套 resolved views 的
annotation sensitivity。

42 条分歧由内部 reviewer 在下列字段不可见时做 A/B role-blind 裁决：annotator lineage、domain、
correctness、reference answer、parsed answer、raw/smoothed predictions 和 position scores。裁决只允许选择一套
完整 A/B view，不允许发明第三套 span。结果为：

- 最终 path：61 clean / 35 hallucinated；
- zero remaining path disagreements；
- zero remaining positive-token disagreements；
- positive claims：133；positive tokens：3,814；
- resolved-primary view：737 claims；
- resolved-secondary view：687 claims。

解封 A/B lineage 后，42 条裁决中选择 secondary 41 条、primary 1 条。这揭示 primary local
Mistral-24B 标注质量明显不足，也意味着最终 reference 不能称为人工 Gold。裁决者不是独立人类，并且裁决前
看过 aggregate agreement statistics；这些限制已写入机器报告。

## 5. 冻结采用门

对每个 resolved view 和每个 seed 分别计算：

1. smoothed token AP − paired raw token AP；
2. smoothed token AP − strongest same-row position token AP；
3. smoothed claim AP − paired raw claim AP；
4. smoothed claim AP − strongest same-row position claim AP。

一个 seed 必须四项都严格大于 0 且四项均值大于 0。一个 view 至少 2/3 seed 通过；两套 views 都通过才可
采用。单 view、单 metric 或 per-domain 的正结果都不能覆盖失败。

token unit 只包含明确审过、非 uncertain claim span 内的 token；未审 token 直接 mask，不当负例。
claim unit 是一个非 uncertain claim，score 为其 exact-span token probabilities 的算术均值。没有使用
full-tail labels。

## 6. 结果

| Resolved view | Unit | Position AP | Seed 42 raw / smooth | Seed 43 raw / smooth | Seed 44 raw / smooth |
|---|---|---:|---:|---:|---:|
| primary | token | `.3817` | `.2252 / .2419` | `.2238 / .2403` | `.2231 / .2394` |
| primary | claim | `.4231` | `.2771 / .2742` | `.2770 / .2738` | `.2746 / .2717` |
| secondary | token | `.3704` | `.2167 / .2334` | `.2154 / .2318` | `.2143 / .2308` |
| secondary | claim | `.4249` | `.2881 / .2848` | `.2880 / .2848` | `.2853 / .2831` |

两套 views 的共同模式完全一致：

- token AP：smoother 相对 raw 每个 seed 提高 `.0163–.0167`；
- claim AP：smoother 相对 raw 每个 seed 下降 `.0022–.0032`；
- token AP：smoother 比 position 低约 `.137–.142`；
- claim AP：smoother 比 position 低约 `.140–.151`；
- resolved-primary：0/3 seed 通过；
- resolved-secondary：0/3 seed 通过；
- overall adoption gate：failed。

evaluation 内部 AP 又用 `sklearn.metrics.average_precision_score` 对 primary/seed42 独立复算，raw、smooth、
position 的 token/claim AP 均在浮点误差内一致。

## 7. 正确解释

可以说：

- fixed smoother 确实消除了一部分 token-level 局部抖动；
- 这种改善没有提升 claim ranking，反而稳定轻微下降；
- 旧 64 条上看似通过的结构没有迁移到新 mixed-domain 数据；
- 当前 H 预测大幅弱于位置捷径，尚无 transferable semantic localization evidence。

不能说：

- sparse hallucination labels 没有价值；
- 所有 H architecture 都已经失败；
- position baseline 是可部署的语义 detector；
- 本轮标签是 human Gold；
- smoother 或 raw H 可以接入 scalar reward。

位置基线强只说明本批错误 claim 在 trajectory 中的位置分布可预测，以及当前 H 没有提供足够增量信号。它是
必须超过的 shortcut control，不是最终方法。

## 8. 下一步

本轮 96 条一旦解封就永久失去 fresh-blind 身份。推荐将旧 64 条与本轮 96 条组成 160 条
`H-v3 development/Silver`，仅用于诊断和候选选择：

1. 冻结 query/domain-balanced cross-validation；
2. 每一格都报告 absolute/normalized position，而不是只和 raw H 比；
3. 直接检验 JP/H features 在 position 之外的增量贡献，例如预注册 position-only、H-only、
   H+position，以及 position-residualized H；
4. domain-balanced 汇总为主，leave-domain-out 作为 transfer 诊断；
5. 在 160 条上只选择一个 H-v3 candidate；
6. 另采完全新的 Phi-native blind-v2，重新双标并在预测 sealed 时冻结采用门。

在 blind-v2 通过前：raw H 只作 diagnostic；smoother、tail、exact onset、progress、reconstruction 和
`p_hallucination→reward` coupling 全部保持关闭。

## 9. 关键 artifact

| Artifact | SHA256 |
|---|---|
| `comparison_protocol_v1.json` | `f445c02db0f97a22e3d35ce249f6b964d7541290eb401210b1c317a49ad889f9` |
| `agreement_report_v1.json` | `364f9ea0755d784076ff7c438cf6f7c0d664c34e7abbf85895ce320856a8d613` |
| `adjudication_selections_v1.jsonl` | `6507f3937f3283deb8d84cb8fb5753c6e182e608141c052d0f1bc3ea4d341b4c` |
| `adjudication_report_v1.json` | `4fcfca3b46386e75d94cfe8952c1a05e92f918c63d1111431cb5499c4ff5e9ae` |
| `labels_resolved_primary_v1.jsonl` | `54d1506c1e9bf66ce3e7358c12341344ec44e2b9f77d7be40ac2a6e0fc1b97fa` |
| `labels_resolved_secondary_v1.jsonl` | `cd20827deda69a4ab22b5568fd180a118ec7550a92af7b30a22accb8da8ec665` |
| `evaluation_protocol_v1.json` | `f28042af2ee927d5465e30e24b75d9a5ee264e87d5b07e97a3b263dfcf8babfd` |
| `evaluation_result_v1.json` | `12dfa039cc62d021d0a1af144679201fc3ec8c7778834fd2791997c926f7d1a7` |

冻结与执行 chronology：annotation comparison `575d760` → comparison result `b4c9393` → blinded
adjudication freeze `f4deeeb` → resolved views `54faac1` → evaluation gate freeze `950dade` → blind result
`0b50419`。
