# Dual-Prior Reward-Gate Integration Pilot v1

状态：`completed_reward_gate_integration_diagnostic_only`

证据等级：`pipeline pilot`

日期：2026-08-19

## 1. 结论

当前 shared-gradient reward-gate alignment 不纳入保留方案。它确实让 reward gate 学会了 detached fused
dual prior，但对稀疏 key localization 的损伤超过冻结保护线：3 个 seed 只有 1 个通过，因而整体 gate 失败。

这个失败不影响此前已经通过的 direct key/complete heads 与原始 mutual distillation。以下公式和权重继续保留：

```text
MSE(A_key, stopgrad(A_complete))
+ MSE(A_complete, stopgrad(A_key))
weight = 0.25
```

本轮没有使用 containment replacement，没有启用 reconstruction，也没有读取 pilot/final test。

## 2. 为什么 gate weight 冻结为 10.0

在运行任何 gate-enabled cell 之前，先利用历史 M1 中权重为 0、但仍被记录的 gate diagnostic 做量级审计。
历史 held-out gate MSE 均值约为 `.00794`；更关键的是，在三个旧 M1 checkpoint 的固定首 batch 上：

| gate weight | gate-head gradient / final-BCE gate-head gradient，中位数 | 最大值 | shared-encoder ratio 最大值 |
|---:|---:|---:|---:|
| `.25` | `.0060` | `.0204` | `.00110` |
| `10.0` | `.2410` | `.8160` | `.04396` |

候选权重预先限定为 `.25/1/5/10`；选择规则要求 gate-head ratio 中位数进入 `.15–.50`，每个 seed 都不超过
`1.0`，shared encoder 最大不超过 `.05`。`10.0` 是第一个满足规则的候选。完整机器审计为
`configs/dual_prior_reward_gate_v1/loss_scale_audit_v1.json`。该选择没有读取任何 G1 结果。

## 3. 冻结比较

协议：`configs/dual_prior_reward_gate_v1/training_protocol_v1.json`，SHA256
`46bf816ab55d0f2ba28e63bef83ab820732bd0ad045db4853a25010fb587e751`。

```text
G0 = correctness BCE + direct key/complete BCE + original mutual MSE(.25)
G1 = G0 + 10.0 * MSE(A_gate, stopgrad(.5 A_key + .5 A_complete))
```

两个 cell 都从头训练，使用同一 48/16 query-disjoint split、5 epochs、batch size 4、LR `1e-4`，并运行
seeds `42/43/44`。consistency、hallucination、tail、progress、reconstruction 等 loss 全部关闭。训练来自干净
commit `b0deb7828763a90de7fc6be590dbf814ead399eb`，6/6 cells 完成。

## 4. 结果

三 seed 均值：

| 指标 | G0 | G1 | G1 − G0 / 相对变化 |
|---|---:|---:|---:|
| gate→fused-prior MSE | `.00909` | `.00170` | `-79.58%` |
| gate/fused cosine | `.6022` | `.9524` | `+.3503` |
| gate/fused overlap mass | `.4490` | `.8545` | `+.4055` |
| gate effective-token fraction | `.9288` | `.3702` | `-.5586` |
| fused-prior effective-token fraction | — | `.3789` | gate 与目标接近 |
| key unit AP | `.4240` | `.3474` | `-.0766` |
| complete unit AP | `.9185` | `.9286` | `+.0102` |
| correctness AUROC | `.6931` | `.6561` | `-.0370` |
| key/complete probability MAD | — | `.3043` | 未塌缩 |
| key/complete probability correlation | — | `.8071` | 未塌缩 |

逐 seed 的关键保护项：

| seed | gate MSE 相对下降 | key unit AP delta | complete unit AP delta | correctness AUROC delta |
|---:|---:|---:|---:|---:|
| 42 | `76.32%` | `-.1322` | `+.0092` | `-.0317` |
| 43 | `67.00%` | `-.0109` | `+.0179` | `-.0317` |
| 44 | `95.42%` | `-.0867` | `+.0034` | `-.0476` |

gate alignment、离开均匀分布、complete localization、correctness、map separation、score non-collapse 均为
3/3 seeds 通过；key localization 只有 1/3 通过冻结的 `delta >= -.05`。没有任一 seed 同时通过所有保护项，
所以不能采用 G1，也不能在结果后放宽门槛。

## 5. 机制诊断

`fused_prior.detach()` 只保证 gate loss 不会在当前 backward 中直接更新 key/complete map。当前
`A_gate` 仍由共享 `token_features` 产生，因此 gate loss 同时更新：

```text
gate head
    ↑
shared token representation  ← gate-alignment gradient
    ↓
key head / complete head
```

G1 的 gate 已从近乎均匀变为与 fused prior 高度一致，说明优化目标有效；但 key AP 在两个 seed 明显下降，
而 complete AP 三个 seed 都保持或提高。这符合 shared-representation interference：较强 fused-map 梯度改变了
共享表示，稀疏 key 分支比覆盖更广的 complete 分支更脆弱。这个解释是由梯度路径与结果共同支持的机制假设，
还不是独立因果证明。

## 6. 下一道门

不扫描当前 shared-gradient gate weight，也不改写本轮失败。若继续修复，优先发布新的 head-only gate
alignment 协议：生产打分仍使用同一个 gate head，但 auxiliary gate loss 通过 detached token features 只更新
gate-head 参数，不进入共享 encoder；final correctness BCE 的正常梯度保持不变。

新协议应继续保留 G0、原 mutual `.25`、相同数据/预算/三 seed 与全部保护门，并先用回归测试证明：单独对
gate auxiliary loss backward 时 gate-head 梯度非零，而 shared encoder、key head 与 complete head 梯度为零。
只有该修复通过后，才进入 query-grouped candidate pool 做 ranking/Best-of-N。当前 48/16 数据每题只有一个
candidate，因此本轮没有、也不能提供 Best-of-N 结论。

## 7. Artifact

- 量级审计：`configs/dual_prior_reward_gate_v1/loss_scale_audit_v1.json`
- 冻结协议：`configs/dual_prior_reward_gate_v1/training_protocol_v1.json`
- 机器结果：`configs/dual_prior_reward_gate_v1/training_result_v1.json`
- 机器结果 SHA256：`bc94e06516c31a5f616f6ea6bb91ec67149bff8e052af43d85a69e7965c67c9b`
- 运行目录：`run_artifacts/dual_prior_reward_gate_v1/pilot_v1/`
