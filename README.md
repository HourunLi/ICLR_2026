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
| hallucination localization | 对原始 Phi trajectory 双标 material claims 和首个 unsupported/contradicted claim | onset 映射到冻结 `output_token_ids`；uncertain 必须 mask |
| dual-prior localization | 等 consistency 与 hallucination 模块分别验证后再做 | key/complete/reconstruction 必须来自外部 target |

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
- 裁决标签已通过 identity/token/provenance merge：4096 行 train 中恰好 64 行有 path/onset 监督，
  其余保持缺失。query-disjoint dense pilot 使用 48 train / 16 dev，H0–H3 四个 5-epoch cell 均完成。
- H1/H2 的 dev path AUROC 均为 `0.933`，incorrect-only 为 `0.778`，高于 length shortcut；但
  token AP (`0.461/0.497`) 低于绝对位置 baseline (`0.514`)，六个 positive dev onset 的 `±5`
  命中均为 `0`。H2 虽把 tail margin violation 压到 `0%`，却同时整体下移 clean/pre/tail value。
  因而只保留 path 分支为有希望的诊断，onset 和 localized tail shaping 均未通过。
- base validation 仍没有 hallucination、progress、dual-prior 或 reconstruction supervision；当前没有
  formal mechanism-efficacy 结论。
- `pilot_test` 和 `final_test` 尚未用于当前模块选择。当前没有 formal mechanism-efficacy 结论。

## 下一道门

Hallucination Localization v1 已冻结为 `completed_path_signal_onset_gate_failed`。下一轮先做
position-shortcut-controlled onset repair：在不启用 pseudo-tail、不跑 mixed 3968-row 机制训练的前提下，
扩大 positive onset 标签并预注册绝对/归一化位置 baseline。若同一 token BCE 在扩大数据后仍不能超过
位置 baseline 或形成 onset `±5` 命中，再讨论 claim-boundary objective 或 loss 变更。path-only 分支可
保留，但当前 16-row dev 不足以授权扩量训练。

完整停止条件和标签定义见
[Hallucination Localization Pilot v1](docs/hallucination_localization_pilot_v1.md)。

## 受保护的数据契约

- 保存的 `prompt_token_ids` 和 `output_token_ids` 是 hidden-state 与 token label 对齐的唯一事实来源；
  文本用于审计，不能重新 tokenize 后猜 token 轴。
- 正式输入保留 Phi embedding 与全部 32 个 block 输出，宽度 `33 × 3072 = 101376`。
- condition 必须来自独立 prompt-only forward；同一 query 的 candidates 共享同一个 canonical
  condition。
- `query_id` 只表示 Best-of-N candidate pool；`semantic_id` 只表示 consistency group，不能互相 fallback。
- `online` 与 `precomputed` hidden-state source 都受支持，但同一正式比较的所有 variants/seeds 必须一致。
- 外部监督必须按 row ID、query ID、精确 token-ID hash 和协议 provenance 绑定；缺失监督保持缺失。
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
- [docs/hallucination_localization_pilot_v1.md](docs/hallucination_localization_pilot_v1.md)：当前模块协议
- [docs/on_policy_pilot0_reaudit_v1.md](docs/on_policy_pilot0_reaudit_v1.md)：Route A v1a 冻结结果
- [docs/semantic_rewrite_v8_reasoning_equivalent.md](docs/semantic_rewrite_v8_reasoning_equivalent.md)：保留的
  on-policy mining / Phi self-rewrite 契约
- [docs/decision_history.md](docs/decision_history.md)：旧 Stage 1B、外部 rewrite 和路线演化
- [docs/runbook_zh.md](docs/runbook_zh.md)：操作手册

代码和测试是行为事实；README 与 handoff 不再承担 changelog。历史事实新增到 history 文档，当前方案或
状态变化才更新 README/handoff。
