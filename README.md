# CLIR: Consistency-Localized Intrinsic Rewards

最后更新：2026-08-18

CLIR 研究如何利用真实 LLM hidden states，为 Best-of-N 轨迹排序学习比最终 correctness 更细的监督：
语义一致性、首个 hallucination 的定位，以及后续 dual-prior evidence localization。仓库实现自包含的
SWIFT-style reward baseline，不调用 SWIFT 仓库代码。

接手项目先读 [当前交接](docs/handoff.md)；方法假设见 [研究方案](docs/proposal.md)；已经淘汰的路线、
旧实验和转向理由统一放在 [决策与实验历史](docs/decision_history.md)。

## 当前保留的方案

| 层次 | 当前选择 | 边界 |
|---|---|---|
| task / feature model | Phi-3.5-mini；embedding + 32 blocks，共 33 层 | 精确 token IDs 是输入事实；不静默删层 |
| correctness baseline | `strict_swift / encoded_swift / clir` 共用候选、split 和预算 | correctness 只监督 outcome，不伪造机制标签 |
| semantics consistency | 主路线 Route A：同一原始 prompt 下挖掘 Phi on-policy 等价轨迹 | 由独立 relation verifier 判断 reasoning equivalence |
| rewrite 备选 | Route B：Phi 自己 rewrite 自己的轨迹 | 外部 Qwen/Falcon rewrite 只保留为 off-policy control |
| hallucination localization | T0：S1 sparse token BCE；不加 absolute 或 relative full tail | absolute T2 有全局 shift；首个 pre-onset-relative R1 无 clean anchor 且损伤 sparse AP；exact onset 未通过 |
| dual-prior localization | v1 的 64 条双标、盲审裁决、exact-token gold 与三种子 D0–D3 direct-target pilot 已完成 | `key ⊆ complete`；direct targets 可学，下一步只比较协同形式，reconstruction 仍关闭 |

模块按顺序单独验证：先 consistency，再 hallucination localization，最后 dual prior。首轮不把三族 loss
同时混训，也不在未校准的 hallucination head 上启用 pseudo-tail 自训练。

## 当前状态与证据边界

- Stage A 的真实 Phi rollout、all-layer hidden-state extraction、query-disjoint split、恢复、打分和
  Best-of-N evaluator 已形成可复现工程闭环。
- correctness-only Stage 1 的 BoN@16 三 seed 均值为 strict SWIFT `88.28%`、encoded SWIFT
  `88.54%`、CLIR `89.32%`。encoded→CLIR 只有 `+0.78` 个百分点且跨 seed 方向不稳定，因此没有
  稳定的 CLIR 增益结论。
- Route A v1a 完成 31 对 claim-level 双标和裁决：agreement `30/31`、Cohen's kappa `0.8394`，最终
  27 对通过。它们不复制 trajectory，只给原 4096 行 train 中的 54 行增加 consistency metadata。
- Route A v1a 的 matched 1-epoch pilot 没有造成 correctness/ranking 崩坏，但表示诊断显示主要作用是
  分离不同 semantic groups，而不是继续提高已经饱和的 positive cosine。该结果仍是
  `pipeline pilot`，不能证明泛化或 Best-of-N 增益。
- Hallucination Localization v1 的 64 条双标已完成：path agreement `81.25%`、Cohen's kappa
  `0.5766`；共同判 positive 的 15 条只有 `5/15` onset exact match，说明 path 定义尚可用，但首错
  边界分歧很大。22 条阻塞分歧经内部盲审后得到 41 clean / 23 hallucinated；这是
  `pipeline pilot` Silver，不是人工 Gold。
- Localization v2 从现有 claim reviews 物化了 9,132 个 sparse exact-token labels：supported span 为 0，
  unsupported/contradicted span 为 1，non-claim 与 unreviewed token 保持 mask。train/dev 仍为
  query-disjoint 48/16，所有 identity、token hash、provenance 与 feature gate 通过。
- 四个 matched 5-epoch cell 中，unweighted sparse S1 的 dev span-token AP `.416`，超过 onset-tail S0
  `.371` 和 absolute-position `.393`；claim-mean AP `.464` 也超过 position `.422`。因此 point-estimate
  token gate 通过并保留 S1。2,000 次 query bootstrap 区间仍跨 0，不能视为稳定机制证据。
- v2b 撤销审计确认旧证据不足以永久否证 tail。matched T0/T1/T2 中，T2（S1 + full-tail
  `weight=.5`）的 span AP `.454`、explicit-token value-risk AP `.500`、correctness AUROC `.952`，通过全部
  预设 point-estimate guards；其相对 T0 的 tail−pre gap bootstrap difference 为 `-.642`
  `[-1.434,-.087]`，value-risk AP difference 为 `+.056` `[+.008,+.108]`。因此 T2 仅获扩大 validation 与
  多 seed 比较资格；轻权重 T1 失败，mixed-data run 未获授权。
- v2c 随后完成 4-fold × 3-seed、24-cell 复核，主要采用门只看 48 条 selection-unexposed out-of-fold
  predictions。T2 的 mean value-risk AP、span AP、correctness AUROC 分别比 T0 高 `+.0357/+.0170/+.0093`，
  但三个 seed 的 `tail−clean` gap 全部恶化，mean `+.2517`；clean value 的绝对下移也在三个 seed 都大于
  tail 下移。因此 0/3 seed 通过 tail-specific locality，当前选择 T0，暂缓当前 absolute-margin T2。
  这不是永久否证 tail；不能继续扫描当前 absolute-margin weight。
- v2d 进一步按冻结协议只跑一个 pre-onset-relative R1 cell。它把 relative-margin violation 从 `.586` 降到
  `.091`，`tail−pre` gap 改善 `-1.712`，证明统一平移捷径已被消除；但 `tail−clean` gap 恶化 `+.925`，
  value-risk AP 与 sparse-span AP 分别下降 `.109/.100`。原因是 loss 只锚定 hallucinated row 内部，仍没有
  clean-row anchor。R1 状态为 `completed_fail_keep_t0`，不扩跑、不扫权重；未来 repair 必须显式控制
  matched clean positional baseline，并处理 shared-encoder interference。
- 当前不采用的精确对象是 absolute T2 与 pre-onset-relative R1 两个实现，不是永久否证 full-tail family；
  正式理由与非理由见 `tail_non_adoption_record_v2d.json`。采用标准要求 tail 同时相对 pre 和 clean 局部下降，
  且不以牺牲 sparse localization 为代价，不能只看“tail 自己变低了”。
- exact onset 仍未通过：T0/T2 的 fixed MAE 为 `82.5/71.7`，六个 positive 的 `±5` 均为 `0/6`。
  pseudo-tail 继续禁止；full tail 不得称为 token hallucination ground truth。
- Dual-prior secondary 64/64 完成；独立双标的 key/complete macro unit F1 为 `.5469/.8456`，48 条 set
  disagreement 全部经过 role-blind adjudication。最终 64 条均 usable，63/64 为严格 `key ⊂ complete`。
- D0–D3 direct-target pilot 在固定 48/16 query-disjoint split 上完成 4 cells × 3 seeds。unit AP 三种子均值：
  key 为 D0 `.079`、D1 `.377`、D3 `.433`，位置基线 `.133`；complete 为 D0 `.328`、D2 `.921`、
  D3 `.919`，位置基线 `.268`。全部预设 guard 为 3/3 seeds 通过，D3 correctness AUROC 相对 D0
  `+.005`，两张 prior map 的平均概率差 `.302`、相关系数 `.770`。
- 该 dual-prior 结果只说明 adjudicated direct targets 在当前 pipeline pilot 中可学且可共存，不说明它们已
  改善 Best-of-N，也不授权旧 mutual MSE、gate alignment 或 reconstruction。
- base validation 仍没有 hallucination、progress、dual-prior 或 reconstruction supervision；当前没有
  formal mechanism-efficacy 结论。
- `pilot_test` 和 `final_test` 尚未用于当前模块选择。

## 下一道门

Dual-prior direct-target gate 已以 `completed_pass_direct_targets_learnable` 关闭。下一步另发只改变 collaboration
项的 matched 协议：以 D3（direct key + complete BCE、无协同）为 control，对照旧 symmetric mutual MSE 与
尊重 `key ⊆ complete` 的 directional containment loss。采用门必须同时保护 key/complete held-out AP、两图
可分性、containment violation 与 correctness；不能只因 violation 下降就采用。gate alignment 继续关闭，
reconstruction 继续等待独立 768-d target，不得使用 same-candidate pooling。完整 direct-target 结果见
[Dual-Prior Evidence Pilot v1](docs/dual_prior_evidence_pilot_v1.md) 与
`configs/dual_prior_evidence_v1/training_result_v1.json`。

完整停止条件和标签定义见
[Hallucination Full-Tail v2c](docs/hallucination_tail_cross_validation_v2c.md)、
[Hallucination Relative Full-Tail v2d](docs/hallucination_relative_tail_pilot_v2d.md)、
[Hallucination Full-Tail v2b](docs/hallucination_tail_comparison_v2b.md) 与
[Hallucination Localization Pilot v2](docs/hallucination_localization_pilot_v2.md)。dual-prior 的冻结协议、gold 与
机器结果位于 `configs/dual_prior_evidence_v1/`。

## 受保护的数据契约

- 保存的 `prompt_token_ids` 和 `output_token_ids` 是 hidden-state 与 token label 对齐的唯一事实来源；
  文本用于审计，不能重新 tokenize 后猜 token 轴。
- 正式输入保留 Phi embedding 与全部 32 个 block 输出，宽度 `33 × 3072 = 101376`。
- condition 必须来自独立 prompt-only forward；同一 query 的 candidates 共享同一个 canonical
  condition。
- `query_id` 只表示 Best-of-N candidate pool；`semantic_id` 只表示 consistency group，不能互相 fallback。
- `online` 与 `precomputed` hidden-state source 都受支持，但同一正式比较的所有 variants/seeds 必须一致。
- 外部监督必须按 row ID、query ID、精确 token-ID hash 和协议 provenance 绑定；缺失监督保持缺失。
- 稀疏 token hallucination 监督必须同时写 `token_hallucination_target` 与
  `token_hallucination_mask`；mask 外 target 固定为 0，但不参与 loss。
- 冻结协议和 artifact 只读；任何阈值、schema、loss 或指标语义变化必须发布新版本。

详细契约见 [CLIR supervision protocol](docs/clir_supervision_protocol.md)。

## 主要入口

| 路径 | 职责 |
|---|---|
| `src/consistency_localized_reward.py` | baseline、CLIR heads 与 loss |
| `src/clir_data.py` | JSONL/feature loader、token 对齐与 collate |
| `src/clir_hidden_states.py` | exact-token online all-layer feature 重现 |
| `src/clir_supervision.py` | 外部机制监督绑定与覆盖审计 |
| `src/clir_reasoning_rewrite.py` | reasoning-equivalence verifier 契约 |
| `src/clir_hallucination_annotation.py` | claim schema、exact span、token onset 映射 |
| `src/clir_localization_evaluation.py` | path/token/onset 指标与 shortcut baselines |
| `train_clir.py` | 训练、恢复、健康证据与 checkpoint |
| `score_clir.py` | 逐候选 reward scoring |
| `evaluate_hallucination_localization.py` | localization held-out evaluation |
| `scripts/checkpoint_dual_prior_secondary_v1.py` | secondary 逐条 durable checkpoint、顺序校验与续跑状态 |
| `scripts/run_dual_prior_matrix_v1.py` | D0–D3 × 三种子的独占 GPU launcher |
| `scripts/summarize_dual_prior_pilot_v1.py` | direct-target learnability 与位置基线采用门 |
| `evaluate_clir.py` | ordered-prefix Best-of-N 与 ranking metrics |
| `summarize_clir.py` | 多 seed 汇总与配对比较 |

Best-of-N 选择只在 `evaluate_clir.py` 中实现；`score_clir.py` 发布逐候选分数、localization 诊断数组和
provenance，不执行 candidate 选择。

## 本地验证

使用项目固定的 Python 3.11 环境，不要调用系统 `pytest`：

```bash
cd /prodcpfs/user/panzhixin/ICLR_2027
P=/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python
"$P" -m pytest -q
"$P" -m pip check
```

完整运行说明见 [runbook](docs/runbook_zh.md)。toy 或 pipeline pass 只证明工程闭环，不构成方法效果证据。

## 当前文档

- [docs/handoff.md](docs/handoff.md)：当前可执行状态、artifact 和下一步
- [docs/proposal.md](docs/proposal.md)：研究假设、目标函数与完整 ablation 设计
- [docs/hallucination_localization_pilot_v2.md](docs/hallucination_localization_pilot_v2.md)：当前模块结果
- [docs/hallucination_tail_comparison_v2b.md](docs/hallucination_tail_comparison_v2b.md)：tail 撤销审计、
  matched 比较与保留边界
- [docs/hallucination_tail_cross_validation_v2c.md](docs/hallucination_tail_cross_validation_v2c.md)：
  4-fold × 3-seed tail 复核、全局 shift 诊断与当前裁决
- [docs/hallucination_relative_tail_pilot_v2d.md](docs/hallucination_relative_tail_pilot_v2d.md)：
  pre-onset-relative R1 单-cell 试验、clean-anchor 缺口与冻结负结果
- [docs/dual_prior_evidence_pilot_v1.md](docs/dual_prior_evidence_pilot_v1.md)：双标裁决、exact-token gold、
  D0–D3 direct-target 结果与下一道 collaboration 门
- [docs/hallucination_localization_pilot_v1.md](docs/hallucination_localization_pilot_v1.md)：contaminated-tail 历史基线
- [docs/on_policy_pilot0_reaudit_v1.md](docs/on_policy_pilot0_reaudit_v1.md)：Route A v1a 冻结结果
- [docs/semantic_rewrite_v8_reasoning_equivalent.md](docs/semantic_rewrite_v8_reasoning_equivalent.md)：保留的
  on-policy mining / Phi self-rewrite 契约
- [docs/decision_history.md](docs/decision_history.md)：旧 Stage 1B、外部 rewrite 和路线演化
- [docs/runbook_zh.md](docs/runbook_zh.md)：操作手册

代码和测试是行为事实；README 与 handoff 不再承担 changelog。历史事实新增到 history 文档，当前方案或
状态变化才更新 README/handoff。
