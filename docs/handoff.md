# CLIR 当前交接

最后更新：2026-08-18

本文件只回答当前如何继续。研究设计见 `docs/proposal.md`，历史决策见
`docs/decision_history.md`，旧审查文件不作为 HEAD 的运行说明。

## 1. 当前停止点

Hallucination Localization Pilot v1 已完成并冻结为
`completed_path_signal_onset_gate_failed`：

- independent secondary 64/64 结构和 exact quote 通过，SHA256
  `cbc5599a82a928fb3d4433645ffb5101e136659cbc180dd9ecfe4c47e55eb0a6`；
- primary/secondary path agreement `52/64 = 81.25%`，Cohen's kappa `0.5766`；15 个共同
  positive 中 onset exact `5/15`，median absolute distance `67` tokens；
- 12 个 path 冲突和 10 个共同-positive onset 冲突组成 22-row A/B package；内部盲审选择 A 10、B 8、
  revised onset 4，最终 41 clean / 23 hallucinated；原始两份标签均未覆盖；
- 64 条最终标签已通过冻结 token identity、provenance、merge 和 coverage 审计；
- query-disjoint dense split 为 48 train / 16 dev；H0–H3 四个 5-epoch cell 全部完成，无 non-finite；
- path ranking 有信号，onset 与 localized tail shaping 未过门。当前停止，不跑 mixed 3968-row，不启用
  pseudo-tail，不读取 pilot/final test。

## 2. 当前采用的研究路线

1. Phi-3.5-mini 同时作为 task model 和 feature model；正式表示保留全部 33 层。
2. Semantics consistency 主路线是 Route A：从同一原始 prompt 的 Phi on-policy candidates 中挖掘
   reasoning-equivalent pairs。Route B 的 Phi self-rewrite 保留为后续对照。
3. 错误 trajectory 不要求 rewrite。当前 Route A correct-only；错误机制等价组只有在定向采样和独立
   verifier 能稳定判断后才做 ablation。
4. Localization v1 已独立跑完；当前只允许 onset Pilot v2 或独立 dual-prior pipeline 探索，不把失败的
   onset/tail 分支与 consistency 或 dual prior 混训。
5. dual-prior targets 必须外部生成；即使先探索，也不因 path diagnostic 有信号而跳过独立校准。

外部 Qwen/Falcon rewrite、旧 Route A v1 manifest 和 Stage 1B v4 的原健康分类均不是当前训练入口；
原因和原始数字见 `docs/decision_history.md`。

## 3. 当前版本化 artifact

### Route A v1a

- `configs/on_policy_pilot0_v1a/adjudicated_relations_v1.jsonl`：27 accept / 4 reject；
- `configs/on_policy_pilot0_v1a/train_mixed_v1.jsonl`：4096 行，仅 54 行带 consistency metadata；
- `configs/on_policy_pilot0_v1a/training_protocol_v1.json`；
- `configs/on_policy_pilot0_v1a/training_result_v1.json`；
- 解释边界：`docs/on_policy_pilot0_reaudit_v1.md`。

### Hallucination Localization v1

- protocol：`configs/hallucination_localization_v1/protocol_v1.json`，SHA256
  `f79ea5b2dd92ca72bfb8b4d0878f952bbad0e95d1891bdc4a8d040515a231252`；
- primary labels：`labels_primary_v1.jsonl`，SHA256
  `1b8129982c15b2b948e4e3ec19fb1ce7da979044014673337b4cfe46a08959f7`；
- secondary raw：`labels_secondary_raw_v1.jsonl`，SHA256
  `cbc5599a82a928fb3d4433645ffb5101e136659cbc180dd9ecfe4c47e55eb0a6`；
- mapped secondary / agreement：`labels_secondary_v1.jsonl`、`agreement_report_v1.json`；
- adjudication：`adjudication_resolutions_v1.jsonl`、`labels_adjudicated_v1.jsonl`、
  `adjudication_report_v1.json`；最终 label SHA256
  `00ec837086a76f042766f1b59821783a5aa9c83810634121a362715c5af9281d`；
- train/dev split：`training_split_protocol_v1.json`、`training_split_manifest_v1.jsonl`、
  `training_split_report_v1.json`；
- H0–H3：`training_protocol_v1.json`、`training_result_v1.json`；详细 run artifact 位于
  `run_artifacts/hallucination_localization_v1/pilot0_dense_v1/`。

`adjudication_report_v1.json` 已明确披露裁决者是内部模型审查而非独立人工 Gold；不得删除该边界或把
最终标签改称 Gold。

## 4. 下一步

不要重复当前 seed/48/16/H0–H3，也不要从 post-hoc calibration 中挑阈值冒充预注册结果。当前结果的
有效读法是：

1. H1/H2 path AUROC `0.933`、incorrect-only `0.778`，高于 length baseline `0.700/0.556`；path
   branch 可以保留，但 dev 只有 16 行；
2. H1/H2 token AP `0.461/0.497` 低于 absolute-position baseline `0.514`；
3. 六个 positive dev 在所有 cell 下 onset `±5=0`；train-only 阈值校准也不能修复；
4. H2 的 tail margin violation 虽为 `0%`，clean/pre/tail value 均整体变负，未证明局部 shaping。

推荐下一轮发布新的 onset Pilot v2，而不是原地改 v1：扩大 positive 标签，保持 query-disjoint split，
把 absolute/normalized position baseline 设为硬门。按用户当前决定先不改 loss；若扩量后的相同 token BCE
仍不超过位置 baseline，再单独商议 claim-boundary objective。若改为先探索 dual prior，也必须作为独立
pipeline pilot，不能与当前失败的 onset/tail 分支混训。

## 5. 不可破坏的约束

- `prompt_token_ids/output_token_ids` 是模型输入与 token label 的唯一 source of truth。
- `hallucination_onset` 是生成 token 的零基索引；clean 为 `-1`，uncertain 不强制写 0。
- `query_id` 和 `semantic_id` 分工固定，不能互相替代。
- condition 来自独立 prompt-only forward，并在 query 内共享。
- 正式 feature 保留 embedding + 32 blocks；online/precomputed 是协议变量，正式矩阵内必须一致。
- correctness 只能监督 outcome BCE，不能派生 hallucination、progress、prior 或 reconstruction target。
- augmentation view 不继承 source localization label，除非该 view 被独立重标。
- scoring/evaluation 必须绑定 manifest、checkpoint、协议、dtype、candidate order 和 SHA256 provenance。
- 冻结 artifact 不原地修改；新 schema、阈值、loss 或指标必须新建版本。
- `pilot_test/final_test` 在当前 calibration 与 protocol selection 完成前保持未读。

## 6. 当前证据边界

- correctness-only Stage 1 是 `small-scale real`，没有稳定的 encoded→CLIR 增益。
- Route A v1a 和 localization v1 都是 `pipeline pilot`。
- verifier selection 的 Mistral-24B 只获 Silver pilot 授权；该授权不能自动迁移成 hallucination Gold。
- 内部盲审 Silver 和 16-row dev 不能支持 verifier 准确率或泛化结论。
- path ranking 是 promising diagnostic；onset localization 和 negative-tail locality 当前明确未通过。
- 不能宣称 consistency、hallucination localization 或 negative-tail shaping 改善 Best-of-N。

## 7. 验证与入口

使用固定 SWIFT Python，不要调用系统 `pytest`：

```bash
cd /prodcpfs/user/panzhixin/ICLR_2027
P=/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python
"$P" -m pytest -q
"$P" -m pip check
```

关键入口：

```text
README.md                                      当前方案与项目入口
docs/handoff.md                                当前执行交接
docs/decision_history.md                       旧路线、旧实验和转向理由
docs/hallucination_localization_pilot_v1.md    当前 localization 协议
docs/on_policy_pilot0_reaudit_v1.md            Route A v1a 结果边界
docs/runbook_zh.md                             通用运行手册

src/consistency_localized_reward.py            模型与 loss
src/clir_hidden_states.py                      exact-token online features
src/clir_supervision.py                        外部监督绑定与覆盖审计
src/clir_hallucination_annotation.py           claim/span/onset contract
scripts/validate_hallucination_annotations_v1.py  secondary blind validator
scripts/compare_hallucination_annotations_v1.py   双标 comparison/A-B package
scripts/adjudicate_hallucination_annotations_v1.py  裁决物化与 supervision
scripts/run_hallucination_localization_pilot_v1.py  H0-H3 单-cell launcher
evaluate_hallucination_localization.py             path/token/onset 指标
train_clir.py / score_clir.py / evaluate_clir.py  训练、打分、Best-of-N
```
