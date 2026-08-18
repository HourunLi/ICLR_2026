# Dual-Prior Original Mutual-Distillation Pilot v1

状态：协议在读取结果前冻结，2 cells × 3 seeds 已全部完成。机器裁决为
`completed_pass_original_mutual_distillation`，证据等级仍为 `pipeline pilot`。

## 1. 用户裁决与方法边界

原始相互蒸馏是 dual-prior 方法的保留机制，不得被 containment objective 替换。当前代码中的公式保持不变：

```text
L_mutual
  = MSE(A_key, stopgrad(A_complete))
  + MSE(A_complete, stopgrad(A_key))
```

`A_key/A_complete` 是由相应 logits 在完整 trajectory 上做 masked softmax 得到的 attention distribution。
每个方向先在一条 trajectory 内累加 token squared error，再跨 trajectory 取平均。`joint` phase 同时执行两个
stop-gradient 方向；没有修改为 containment、KL、membership MSE 或其他替代公式。

## 2. 为什么现在可以开始

前一轮 D0–D3 已证明 adjudicated direct key/complete membership targets 可学且可在共享模型中共存。D3 的
key/complete unit AP 三种子均值为 `.4325/.9192`，两张 membership map 的平均绝对概率差为 `.3022`、相关性
为 `.7704`。因此本轮不再改标签、数据或 direct BCE，只检验相互蒸馏带来的显式 branch collaboration。

D3 即使把蒸馏权重设为 0，也记录了未加权的 joint distillation diagnostic；各训练 epoch/seed 约在
`.003–.023`。本轮采用实现原始默认权重 `.25`，预期它是一个温和但非零的辅助项。

## 3. 冻结比较

```text
M0  correctness + direct key BCE + direct complete BCE
M1  M0 + 0.25 * original bidirectional stop-gradient mutual MSE
```

两个 cell 都从头训练，复用同一 48/16 query-disjoint split、同一预计算 33×3072 Phi features、相同
seeds `42/43/44`、5 epochs、batch size 4、LR `1e-4`、joint prior phase 和 final correctness BCE。唯一改变的
训练因子是 `prior_distill_weight: 0 → .25`。

以下项目保持关闭：gate-prior alignment、reconstruction、containment、consistency、hallucination、tail、
progress。这样结果只归因于原始 mutual distillation。

冻结协议：

```text
configs/dual_prior_mutual_distillation_v1/training_protocol_v1.json
```

## 4. 评价与采用门

评价同时读取两类输出：

- `sigmoid(logits)` membership probability：评价 key/complete token 与 unit AP、两图分离度；
- 完整 trajectory softmax attention：直接计算与训练公式数值一致的双向 symmetric attention MSE，检验
  相互蒸馏是否真的缩小 branch discrepancy。

每个 guard 至少 2/3 seeds 通过：

1. M1 held-out symmetric attention MSE 相对 M0 至少下降 2%；
2. key 与 complete unit AP 相对 M0 各自不低于 `-.05`；
3. 两个 unit AP 仍至少超过 position-only `.02`；
4. membership map 平均绝对差至少 `.02`、相关性不超过 `.98`，防止完全坍缩；
5. correctness AUROC 相对 M0 不低于 `-.10`。

通过只表示：原始相互蒸馏在当前小型 held-out pilot 中学到了它自己的协同目标，且没有造成不可接受的定位、
分离度或 correctness 损失。它仍不证明 dual prior 已改善 Best-of-N，因为 reward gate alignment 本轮没有开启。

若未通过，原始 mutual-distillation 机制仍保留，但本冻结权重/调度只能记为 diagnostic；任何 weight 或
joint/alternate schedule 修改都必须另发协议，不能看完结果后修改本轮公式。

## 5. 运行结果

6/6 cells 均从 clean commit `91733a85463991147e6546152a588d58637a09cd` 完成 5 epochs、train/dev scoring
与 exact-token evaluation。M0 的逐 seed key/complete unit AP 与上一轮 D3 完全一致，说明从头复跑 control
精确复现；M1 的唯一训练变化确实是 `prior_distill_weight=.25`。

| 三种子均值 | M0 direct control | M1 original mutual | M1 − M0 |
|---|---:|---:|---:|
| dev symmetric attention MSE | `.015379` | `.010700` | 相对下降 `28.10%` |
| key unit AP | `.432512` | `.424021` | `-.008490` |
| complete unit AP | `.919237` | `.918480` | `-.000757` |
| correctness AUROC | `.693122` | `.693122` | `.000000` |

三个 seed 的 held-out symmetric attention MSE 分别相对下降 `26.999% / 23.997% / 33.313%`，全部超过冻结
2% 门。key AP delta 分别为 `-.0152/-.0225/+.0122`，complete 为 `-.0028/+.0017/-.0012`，均在 `-.05`
non-inferiority 门内。M1 membership map 的三种子平均绝对概率差为 `.29455`、相关性 `.79283`，仍明显分离；
全部六个命名 guard 与 stricter per-seed all-guard 都是 3/3 通过。

机器结果：

```text
configs/dual_prior_mutual_distillation_v1/training_result_v1.json
run_artifacts/dual_prior_mutual_distillation_v1/pilot_v1/
```

`training_result_v1.json` SHA256 为
`03c5c3e799f5c4ee7be58be5f6af3a3a89d3248410d8e685efd4da6881e7d11e`。

允许结论是：保留的原始双向 stop-gradient mutual distillation 在这个小型 held-out pilot 中确实降低了它所
优化的 branch-attention discrepancy，同时没有不可接受的 localization、map-separation 或 correctness 损失，
因此作为 dual-prior collaboration 方法保留。它尚未直接影响 reward gate，也没有证明 Best-of-N 增益。

## 6. 下一道门

下一步不再比较 containment replacement。若继续推进 dual prior，应单独冻结 reward-gate integration：以当前
M1 为 control，只开启 gate 对 fused prior 的对齐，并同时评价 prior localization、gate-prior alignment、最终
correctness/Best-of-N 与 gate 是否发生无意义均匀化。reconstruction 继续等待独立外部 768-d target。
