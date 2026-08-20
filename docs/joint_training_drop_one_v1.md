# Joint Training Drop-One Diagnosis v1

最后更新：2026-08-20

## 1. 结论

JPH 与 JPC 两个预注册 drop-one cell 已完成。它们都保住了 seed-42 的 BoN@16 ranking 保护线，但也都单独
复现了相对 JP 的 key-prior AP 下降。因此，在当前冻结的 single-stream、seed-42 训练条件下：

- sparse hallucination（H）在保留原始 dual prior 时，单独就足以复现 key AP drop；
- consistency（C）在保留原始 dual prior 时，单独也足以复现 key AP drop，而且本次降幅更大；
- JALL 的 key drop 不需要 H×C 同时启用才出现，不能再归因为纯粹的二者交互；
- 去掉 C 后，JPH 的 H span/claim AP 有所恢复，但仍未超过冻结的位置基线，因此 C 不是 JALL localization
  失败的唯一原因。

状态固定为 `completed_seed42_drop_one_diagnosis`。证据等级仍是
`small-scale real integration pilot`：它用于当前配置下的失败归因，不是一般因果结论，也没有建立任何正式
机制或 ranking 效果。

## 2. 冻结设计与执行边界

两个新 cell 复用父实验的相同初始化、seed 42、3968-row manifest、semantic-group batch 顺序、single
stream、batch 4、5 epochs、BF16、LR `1e-4` 和 final-epoch checkpoint：

| Cell | Correctness | 原始 dual prior | H | C |
|---|---:|---:|---:|---:|
| JP（冻结对照） | 1 | 1 | 0 | 0 |
| JPH（新） | 1 | 1 | 1 | 0 |
| JPC（新） | 1 | 1 | 0 | 1 |
| JALL（冻结对照） | 1 | 1 | 1 | 1 |

原始 dual prior 完整保留为 direct key/complete BCE、双向 stop-gradient mutual MSE（`.25`）和
shared-gradient gate 对 detached `0.5/0.5` fused prior 的 MSE（`10`）。没有替换为 head-only，也没有改
相互蒸馏。MIL、token reward、所有 tail、progress 和 reconstruction 仍为 0。

训练前真实 feature 的 no-update gradient-routing audit 通过；两个新 cell 都完成 5 epochs，无 OOM、NaN 或
运行时失败。每个 epoch 的 3968 rows、27 positive pairs、26 same-style negative pairs、6681 个 sparse H
masked tokens、14307 个 prior units 均与冻结协议一致。JPH/JPC 的训练 commit 均为 clean
`55155d295073b219a70386fc2920b66c53bcf55a`。

## 3. 结果

| Cell | BoN@16 | Pairwise acc. | H span AP | H claim AP | Key AP | Complete AP | C gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| JP | `.918` | `.700` | `.192` | `.172` | `.432` | `.946` | `.023` |
| JPH | `.920` | `.686` | `.319` | `.338` | `.314` | `.928` | `.011` |
| JPC | `.918` | `.686` | `.435` | `.442` | `.275` | `.938` | `.818` |
| JALL | `.912` | `.681` | `.272` | `.289` | `.314` | `.931` | `.789` |

### 3.1 Key-prior 归因

预注册的 key-drop 判据为 `cell key AP - JP key AP < -.05`：

- JALL−JP：`-.11796`；
- JPH−JP：`-.11839`，复现 drop；
- JPC−JP：`-.15736`，复现 drop。

机器分类为 `both_auxiliaries_individually_sufficient_at_seed42`。JPH 几乎复现了 JALL 的全部降幅，JPC 的
降幅还更大；同时，JALL 相对 JPC 又回升约 `.0394`。这说明效应不是可相加的，不能把 H、C 的单独 delta
相加来预测 JALL，也不能据此声称某一个目标在一般意义上“有害”。三个 conditional cell 的 key AP 仍高于
位置基线 `.1327`；当前问题是相对 JP 的保留失败，而不是完全无法学习 key target。

### 3.2 Hallucination 归因

JPH 相对 JALL 的 span/claim AP 分别提高 `.04729/.04876`，说明去掉 C 后有恢复；但 JPH 的
`.319/.338` 仍低于预注册位置基线 `.393/.422`，因此恢复门失败。相对完全不训练 H 的 JP，JPH 又提高
`.1273/.1662`，说明 H supervision 在联合流中确实产生了可见 signal，只是没有保住 standalone S1 的
`.416/.464`。

JPH 与 standalone S1 仍有 correctness/prior mixing 和 packing 差异：S1 每 epoch 只训练 48 rows / 12
batches；JPH 在 3968-row 联合流中，H singleton 分散在每 epoch 45–48 个 active batches。因此本轮还不能
区分共享梯度冲突与 optimizer-step/packing 稀释。

特别需要避免误读 JPC：JPC 的 H outer weight 为 0，H head 没有收到 token-H loss，却得到
`.435/.442`。其共享 encoder 会被 correctness、prior 与 C 更新，所以该结果只能说明当前 16-row H dev AP
对共享表示的偶然变化非常敏感；它不是“consistency 学会了 hallucination localization”的证据。正式 H
机制结论需要更大 dev 和多 seed，不能用 JPC 这一数值替代。

### 3.3 Ranking 与 consistency

- JPH−JP BoN@16 为 `+.002`，paired query bootstrap 95% CI `[-.014,.018]`；
- JPC−JP 为 `.000`，CI `[-.018,.018]`；
- JALL−JPH 为 `-.008`，CI `[-.026,.010]`；
- JALL−JPC 为 `-.006`，CI `[-.022,.010]`。

两个新 cell 都通过最大 `.02` regression 的 ranking 保护门，但区间均跨 0，不能宣称收益。JPC 的 training
relation cosine gap 为 `.818`，JALL 为 `.789`，说明 C 显著改变训练关系几何；因为仍无 held-out relation，
该指标只能作训练诊断。

## 4. 后续梯度审计与当前裁决

后续 no-update audit 已完成：H↔prior-total 为 `-.024/+.126`，C↔prior-total 为 `+.133/+.268`，没有稳定
全局负夹角；C↔mutual 与 C↔H 虽为稳定负向，但五个 epoch 中 C 与 H/prior 从不共批，普通 per-batch PCGrad
无法处理。JP 状态下 H/C norm 又是 residual prior-total 的 `2.33×/3.20×`。

所以当前不授权 blanket gradient surgery。下一格建议比较冻结 JPH 与 JPH supervision-aware
packing/schedule：把 48 mechanism rows 集中成 12 个 4-row active batches，保持每行一次、所有 loss 权重和
原始 prior 不变。由于 active steps 也会下降，它不是严格 gradient-budget-matched 的纯 packing 因果实验，
必须按这个边界解释。完整审计见 `docs/joint_gradient_interaction_v1.md`。

本轮仍不授权 seeds 43/44，不自动改 sampler/stream，不访问 `pilot_test` 或 `final_test`。

## 5. 可复核 artifact

- 冻结协议：`configs/joint_training_drop_one_v1/training_protocol_v1.json`
- 机器结果：`configs/joint_training_drop_one_v1/training_result_v1.json`
- no-update 梯度路由审计：`run_artifacts/joint_training_drop_one_v1/audits/gradient_routing_v1.json`
- 新 cell 训练目录：`run_artifacts/joint_training_drop_one_v1/seed42_v1/seed_42/`
- 父实验结果：`configs/joint_training_pilot_v1/training_result_v1.json`
- 后续梯度审计：`configs/joint_gradient_interaction_v1/audit_result_v1.json`

协议 SHA256 为 `ce6c2d5817be8323803c11a1911ddb9167af1cd228eafb6ea7461b39f6c7e128`；机器结果
SHA256 为 `5258d051129bdb480416a68f267229de249525fd664c498d68ab5511cdb3c289`；梯度路由审计 SHA256 为
`262a92738cfb7f404a9cb42af6e5c3e66aadaf4781688d4d038c014a701486f7`。
