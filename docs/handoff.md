# CLIR 当前交接

最后更新：2026-08-18

本文件只回答当前如何继续。研究设计见 `docs/proposal.md`，历史决策见
`docs/decision_history.md`，旧审查文件不作为 HEAD 的运行说明。

## 1. 当前停止点

当前主线是 Hallucination Localization Pilot v1。已完成：

- 64-row train-only selection，32 correct / 32 incorrect 只用于分层，64 个 query 互不重复；
- blind item、private lineage、annotation guide 和 exact-token preflight；
- Mistral-Small-24B candidate-primary 全量标注；
- 4 条纯格式失败的受限合同修复，且 semantic decision signature 未变；
- 64/64 primary label 到冻结 Phi output tokens 的映射；结果为 45 clean / 19 hallucinated；
- 不含 primary decision/correctness 的 64-row secondary package 和完整 prompt。

尚未完成：secondary 审计、双标 agreement、onset comparison、裁决、supervision merge 和 H0–H3 训练。
primary 全部给出 high confidence，因此它只能算 candidate Silver，不能直接进入训练。

## 2. 当前采用的研究路线

1. Phi-3.5-mini 同时作为 task model 和 feature model；正式表示保留全部 33 层。
2. Semantics consistency 主路线是 Route A：从同一原始 prompt 的 Phi on-policy candidates 中挖掘
   reasoning-equivalent pairs。Route B 的 Phi self-rewrite 保留为后续对照。
3. 错误 trajectory 不要求 rewrite。当前 Route A correct-only；错误机制等价组只有在定向采样和独立
   verifier 能稳定判断后才做 ablation。
4. 当前先单独校准 hallucination path/onset，再训练 localization；不同时加入 consistency 或 dual prior。
5. dual-prior targets 必须外部生成，等前两个模块分别通过后再实现。

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
- primary labels：`configs/hallucination_localization_v1/labels_primary_v1.jsonl`，SHA256
  `1b8129982c15b2b948e4e3ec19fb1ce7da979044014673337b4cfe46a08959f7`；
- primary report：`configs/hallucination_localization_v1/primary_report_v1.json`；
- secondary blind items：`configs/hallucination_localization_v1/secondary_items_v1.jsonl`，SHA256
  `ec7ebe67794810300a8d9ca984ea7f29c5e1017f2e1da0b81118dc2789a38591`；
- secondary prompt：`configs/hallucination_localization_v1/secondary_prompt_v1.md`；
- secondary 约定输出：`configs/hallucination_localization_v1/labels_secondary_raw_v1.jsonl`。

最后一个文件是外部盲标交付物。即使文件已经出现，也必须先完成 structure-only validation，不能直接
stage、读取 private lineage 辅助修正，或覆盖 primary/secondary 原始结果。

## 4. 收到 secondary 后的执行顺序

先运行只读 blind validator：

```bash
cd /prodcpfs/user/panzhixin/ICLR_2027
P=/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python
"$P" scripts/validate_hallucination_annotations_v1.py \
  --items configs/hallucination_localization_v1/secondary_items_v1.jsonl \
  --labels configs/hallucination_localization_v1/labels_secondary_raw_v1.jsonl
```

若通过，下一轮按以下顺序实现并执行：

1. 冻结 secondary 文件 SHA256，保持 primary、secondary 原文件只读；
2. 用与 primary 相同的 Phi tokenizer contract 映射 secondary exact quotes；
3. 报告 path confusion、agreement、Cohen's kappa；共同 positive 报 onset exact match、absolute token
   distance 和 `±1/±3/±5` agreement；
4. 将所有 path、problem-claim 或 onset 分歧写入独立 adjudication package；
5. unresolved/uncertain 保持 mask，裁决结果另存，不覆盖原标注；
6. 通过 `merge_clir_supervision.py` 和 `audit_clir_supervision.py` 的身份、token hash 与 coverage 门；
7. 再冻结 H0 correctness-only、H1 token BCE、H2 negative-tail、H3 path MIL 的 matched protocol。

当前仓库尚没有 hallucination 双标 comparison/adjudication runner；不要用临时手工合并绕过这一步。

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
- primary 类别数量门通过不等于标注准确率通过。
- 在双标裁决和 held-out localization evaluation 前，不能宣称 consistency、hallucination localization 或
  negative-tail shaping 改善 Best-of-N。

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
train_clir.py / score_clir.py / evaluate_clir.py  训练、打分、Best-of-N
```
