# CLIR 决策与实验历史

最后更新：2026-08-18

本文件保存已经完成、被替换或不再作为当前入口的决策。它不是 runbook；当前方案看 `README.md`，当前
执行状态看 `docs/handoff.md`。冻结 config、artifact 和原审查文档仍是细节事实来源，不因本摘要而覆盖。

## 1. Baseline 与架构演化

### Stage A 与 Stage 1

项目先完成真实 Phi-3.5-mini rollout、query-atomic hidden-state extraction、精确 token IDs、恢复、训练、
打分和 Best-of-N evaluator。工程阶段证明 pipeline 可运行，不构成方法效果证据。

Stage 1 使用 512 train queries × 8 candidates 和 128 validation queries × 16 candidates，比较
`strict_swift / encoded_swift / clir`。BoN@16 三 seed 均值为 `88.28% / 88.54% / 89.32%`；
strict→encoded 和 encoded→CLIR 的方向都跨 seed 变化。因此保留的结论是“correctness-only baseline
可运行，但没有稳定增益”，而不是 CLIR 已优于 SWIFT。完整结果见 `docs/stage1_results.md`。

早期 raw-width quadratic CLIR 参数和显存成本过高，被正式放弃。当前保留的 layer-axis encoder 仍使用
embedding + 全部 32 blocks，不用删层换存储。取舍记录见 `docs/parameter_blowup_solution_comparison_zh.md`。

### Stage 1B v1–v4

- v1 是 pre-audit diagnostic，缺少后来冻结的 checker、resume、large-LayerNorm、scoring provenance 和
  ordered-prefix 契约，不能形成结论。
- v2 有 8 个 cell 在 epoch 1 前被用户中止，第 9 个未启动；0 个完成 epoch，不能 resume 或计入矩阵。
- v3 没有形成正式效果结果，随后被 v4 替代。
- v4 在 clean commit `b1c4fae` 执行 3 variants × 3 seeds。冻结 summary 当时记录 1/9 healthy，后来
  审查发现训练门误用了最后 epoch 的 batch-average BCE，而非 final checkpoint 的 full-train BCE。
  只读重算为 strict 3/3、encoded 1/3、CLIR 0/3，总计 4/9。历史 summary 不回写；v4 只保留为
  optimization diagnostic，不是机制证据。

v4 的旧协议、健康门和 artifact 细节保存在 `docs/stage1b_v4_protocol.md`、
`configs/stage1b_validation_v4.json` 与 `docs/code_review_panzhixin_fourth_change_resolution.md`。当前代码已
修复 full-train checkpoint evaluation 和 persistent-worker bit-exact resume，但这不追溯升级旧结果。

## 2. Hidden-state 存储决策

最初实验持久化 `[T,101376]` BF16 features。Stage 1B 的 12,096 行镜像约 725.8 GB，继续为每个 rewrite
复制 payload 不可持续。因此新增 exact-token `online` 模式：manifest 保存精确 prompt/output IDs 与
provenance，冻结 Phi 在 batch 内重现 33 层 features，prompt-only condition 按精确 IDs 去重。

单样本已证明 online 与历史 precomputed payload bitwise 等价；尚未得到大规模吞吐结论。最终决策不是
删除 precomputed，而是两种模式并存，并把 source mode 冻结为实验协议变量。

## 3. 外部 LLM rewrite 试错

### 可逆工程 pilot

最早的 train-only reversible pilot 用 4 条 source 构造 12 个视图，跑通 `semantic_id/style_id`、
teacher-forced token alignment、33 层 feature extraction 和 pair audit。它只证明工程接口，不是真实
semantic rewrite。细节见 `docs/semantic_rewrite_pilot_v1.md`。

### Qwen/Falcon v1–v7

外部 generator 路线先后尝试 Qwen-2.5-3B、Falcon3-3B 和 Qwen-2.5-7B：

| 版本 | 结果 | 主要失败 |
|---|---:|---|
| v1 | 0/8 accept | 首版 gate；含一次 incorrect-source repair |
| v2 | 3/8 | 仍是暴露样本上的开发诊断 |
| v3/v4 | 1/8 | runtime 固定后输出不变；self-verifier 不能替代 fidelity gate |
| v5 | 0/8 | 6 条数字 occurrence 漂移，2 条结构无法解析 |
| v6 Falcon3 | 0/8 | 8 条均未遵守标签协议 |
| v7 Qwen-7B | 1/8 | 3 条标签不完整，4 条改变数字 occurrence |

这些运行使用已暴露的 4-source gate，全部 `scale_allowed=false`。失败 artifact 保留，只用于说明：增强
generator 规模改善格式服从不足以解决 fidelity，而且 generator 与 verifier 共用 checkpoint 的
self-verification 不能支持独立质量结论。

更根本的问题是 external-author teacher-forced trajectory 会让 Phi hidden states 携带强作者身份和
off-policy surprisal 信号，使 consistency loss 可能学习“忽略作者”而非“忽略表面风格”。因此外部
Qwen/Falcon rewrite 不再作为主训练路线，只保留为未来 off-policy control。相关设计讨论和冻结配置见
`docs/semantic_rewrite_v8_reasoning_equivalent.md` 与 `configs/semantic_rewrite_llm_v1`–`v7`。

## 4. 转向 on-policy equivalence mining

保留的新主路线 Route A 是：同一原始 problem prompt 下，Phi 按部署 decoding policy 采样 K 条原生
trajectories，再由独立 verifier 对 reasoning equivalence 分组。Route B 让 Phi 自己 rewrite，作为并行
备选。这样 hidden states 保持在 Phi 原生候选分布，避免外部作者与目标 style 完全混淆。

独立 relation verifier 的 64-pair selection set 覆盖 quantitative、code/rules、evidence 和
logic/procedure。两份独立标注 agreement 为 `81.25%`、kappa `0.667`，12 个分歧经裁决得到 32 accept /
32 reject。Mistral-Small-24B 固定 revision 的 blind decision accuracy 和 auto-accept precision 均为
`87.5%`，因此只获 Silver pilot gate 授权，不是 Gold verifier。

`development_32` 的 256 条 Phi 原生 trajectories 经同题、同 correctness、同 normalized answer 的便宜
预筛得到 727 个 correct pairs、8 个 incorrect pairs。这个预筛只提供候选，不等于 reasoning
equivalence。由于错误等价组太少，Pilot-0 决定 correct-only；错误 trajectory 继续提供 outcome BCE，
但不要求 rewrite，也不继承 consistency label。

## 5. Route A v1 撤销与 v1a 修复

旧 Route A v1 的 operational verifier 实际只要求 flat Boolean，121 个 Silver accepts 和 31 组/62 行
manifest 没有执行双向 material-claim/risk 审核，并错误地从 correct outcome 推断内部 reasoning 无错。
因此训练资格全部撤销；原 candidate/source/token/hash lineage 保留，旧 manifest 禁止训练。

v1a 对同一 31 对重新做 claim-level blind double annotation：status agreement `30/31`、kappa
`0.8394`，唯一分歧裁为 reject，最终 27 accept / 4 reject。27 groups 只作为 metadata 注入原 4096 行
mixed train 的 54 行，不复制 trajectory。

matched seed-42/1-epoch pilot 中，consistency cell 与 baseline 的 validation BCE 为 `0.4600 / 0.5074`，
pairwise accuracy `0.5871 / 0.5702`，BoN@16 同为 `0.875`。validation 只有 10 个 mixed queries，不能
声称 ranking gain。表示诊断显示 different-semantic/same-style cosine 从 `0.9926` 降到 `0.7305`，而
same-semantic positive 从 `0.9989` 降到 `0.9827`；当前解释是 loss 主要防止全局 collapse，而不是拉近
已经饱和的 positive pairs。完整记录见 `docs/on_policy_pilot0_reaudit_v1.md`。

## 6. Hallucination Localization 的启动

Route A v1a 关闭最小训练门后，项目转向第二个核心模块。定义从数学题特化门扩展为领域通用的 material
claim：首个 contradicted 或 unsupported claim 是 onset；合法的不同路径、可推出的补充推导、风格变化
和无断言的格式错误本身不算 hallucination。correctness 只用于抽样分层，不是标注证据。

v1 从 train-primary 选择 64 条不同 query，32 correct / 32 incorrect，并排除 Route A 已反复查看的 31
个 query。Mistral-24B primary 原始 run 为 60/64 schema-valid；4 条只含 quote 空白漂移或零基索引错误。
确定性合同修复器完成 10 处 whitespace-equivalent quote alignment 和 1 处 derived onset index，强制
保证 path/status/reason/confidence/summary 不变。最终 64/64 token-map-valid，45 clean / 19
hallucinated。该结果仍是 candidate Silver，等待 secondary、agreement 和裁决。

协议和当前停止点见 `docs/hallucination_localization_pilot_v1.md`。

## 7. 已拒绝或暂缓的选择

- 不再把“换更强的外部 rewrite generator”作为默认扩量方向。
- 不把 correctness 等同于 reasoning equivalence 或 hallucination label。
- 不要求 incorrect trajectories 在 consistency Pilot-0 中必须 rewrite。
- NLL/off-policy score 先作为 diagnostic，不在首轮设硬 gate。
- consistency loss 暂不修改；旧 tiny sweep 的权重不冻结为新默认。
- localization 首轮 `pseudo_tail_weight=0`，避免未校准 head 循环自训练。
- consistency、localization、dual prior 首轮分开训练，避免无法归因。
- dual-prior/reconstruction 不用全零或 same-candidate pooling 伪造缺失 external targets。
- `pilot_test/final_test` 在 validation/calibration 冻结前不用于选择。

## 8. 详细历史入口

- `docs/stage1_results.md`：Stage 1 small-scale real 结果
- `docs/stage1b_v4_protocol.md`：冻结但仅诊断的 Stage 1B v4
- `docs/code_review_panzhixin_*_change.md`：历次独立审查
- `docs/semantic_rewrite_pilot_v1.md`：可逆工程 pilot
- `docs/semantic_rewrite_v8_reasoning_equivalent.md`：路线转向与保留契约
- `docs/on_policy_pilot0_v1.md`：旧 Pilot-0 记录
- `docs/on_policy_pilot0_reaudit_v1.md`：v1 撤销与 v1a 修复
- `configs/verifier_selection_v1/`：relation verifier 双标 selection set
- `configs/on_policy_pilot0_v1a/`：v1a labels、manifest、protocol 和结果
- `configs/hallucination_localization_v1/`：localization selection、labels 和报告
