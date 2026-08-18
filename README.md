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
| hallucination localization | T0：S1 sparse token BCE，不加当前 absolute-margin full tail | T2 有 ranking signal 但发生全局 value shift；exact onset 未通过 |
| dual-prior localization | v1 已冻结 64 条 fixed-unit 双标包，先单独验证 direct external key/complete targets | `key ⊆ complete`；首轮 distill/gate/reconstruction 全关 |

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
  这不是永久否证 tail；未来只考虑有 clean/pre-onset anchor 的相对或 centered objective。
- exact onset 仍未通过：T0/T2 的 fixed MAE 为 `82.5/71.7`，六个 positive 的 `±5` 均为 `0/6`。
  pseudo-tail 继续禁止；full tail 不得称为 token hallucination ground truth。
- base validation 仍没有 hallucination、progress、dual-prior 或 reconstruction supervision；当前没有
  formal mechanism-efficacy 结论。
- `pilot_test` 和 `final_test` 尚未用于当前模块选择。

## 下一道门

Hallucination tail v2c 已冻结为 `completed_keep_t0_defer_tail`，当前 localization 模块选择 T0。dual-prior v1
代码审计与 64 条盲包已经冻结：共 1210 个 domain-agnostic fixed units，沿用 query-disjoint 48/16 split。
Mistral-24B primary 已 64/64 完成并通过结构/non-degeneracy，但 key 有明显 late-position skew，且错误路径的
decisive-flaw 规则执行不稳定，因此 primary 不能单独成为 gold。当前等待独立 secondary 全量双标；通过
agreement/裁决后再跑 correctness-only、key-only、complete-only、joint direct-target 四格。mutual
distillation、gate alignment 与 reconstruction 首轮全关。完整定义见
[Dual-Prior Evidence Pilot v1](docs/dual_prior_evidence_pilot_v1.md)。

secondary 当前使用 `secondary_prompt_resumable_v1a.md`：标签语义仍继承冻结 v1，但每判断一条就由
`checkpoint_dual_prior_secondary_v1.py` 校验、原子落盘并报告下一 item，超时后可从合法前缀继续。

完整停止条件和标签定义见
[Hallucination Full-Tail v2c](docs/hallucination_tail_cross_validation_v2c.md)、
[Hallucination Full-Tail v2b](docs/hallucination_tail_comparison_v2b.md) 与
[Hallucination Localization Pilot v2](docs/hallucination_localization_pilot_v2.md)。dual-prior 当前 artifact 与
第二标注说明位于 `configs/dual_prior_evidence_v1/`。

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
- [docs/hallucination_localization_pilot_v1.md](docs/hallucination_localization_pilot_v1.md)：contaminated-tail 历史基线
- [docs/on_policy_pilot0_reaudit_v1.md](docs/on_policy_pilot0_reaudit_v1.md)：Route A v1a 冻结结果
- [docs/semantic_rewrite_v8_reasoning_equivalent.md](docs/semantic_rewrite_v8_reasoning_equivalent.md)：保留的
  on-policy mining / Phi self-rewrite 契约
- [docs/decision_history.md](docs/decision_history.md)：旧 Stage 1B、外部 rewrite 和路线演化
- [docs/runbook_zh.md](docs/runbook_zh.md)：操作手册

代码和测试是行为事实；README 与 handoff 不再承担 changelog。历史事实新增到 history 文档，当前方案或
状态变化才更新 README/handoff。
