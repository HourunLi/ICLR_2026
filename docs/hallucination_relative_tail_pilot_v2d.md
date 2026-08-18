# Hallucination Relative Full-Tail R1 Pilot v2d

状态：`completed_fail_keep_t0`

日期：2026-08-18

证据等级：`pipeline pilot`；单 seed、selection-exposed dev，不是人工 Gold、独立 test、Best-of-N 或
formal mechanism evidence

## 1. 本轮回答的问题

v2c 发现 absolute-margin full-tail loss 可以改善部分排序指标，但最省力的解是把所有 token value 一起
下移：三个 confirmatory seed 的 clean value 都比 tail value 下移更多。因此，R1 不再要求 tail 低于一个
绝对常数，而是在每条 hallucinated trajectory 内要求 onset 后 value 低于该行 onset 前的均值：

```text
anchor_i = mean(value[i, :onset_i])
loss_i = mean_t>=onset_i relu(margin + value[i,t] - anchor_i)^2
R1 = mean_i loss_i
```

梯度同时流过 tail 与 pre-onset anchor。给一整行所有 value 加同一个常数不会改变 loss，所以 R1 精确
消除了 absolute T2 的统一负 bias 捷径。`onset=0` 没有行内 anchor：train 的 1 条此类 trajectory 只从
relative loss 排除，correctness BCE 与 sparse-span BCE 仍照常训练。

这里的 full tail 仍表示“首个实质错误后的整段 continuation value 受污染”这一 reward hypothesis；它有意
覆盖 supported 和未审核 token，绝不是 token hallucination ground truth。后者仍只有
`token_hallucination_target + token_hallucination_mask`。

## 2. 冻结设计

- 数据：原 query-disjoint train/dev `48/16`；train 有 `17` 条 hallucinated，其中 `16` 条有合法
  pre-onset anchor、共 `3,110` 个 tail token；dev 为 `6/6` 条、`1,460` 个 tail token。
- Control：严格复用 v2b 的 seed-42 T0（correctness BCE + unweighted S1 sparse-span BCE）。
- Candidate：同样 seed、模型、features、5 epochs 与 batch budget，只增加
  `relative_tail_weight=.5`、`relative_tail_margin=.5`；absolute tail 与 pseudo-tail 都为 0。
- 只授权一个 selection-exposed R1 cell。通过全部 point-estimate guards 才能授权 folds 1–3 × three-seed
  expansion；无论通过与否都不能直接成为模块默认。
- `pilot_test/final_test` 保持未读。

冻结协议为
`configs/hallucination_localization_v2/relative_tail_protocol_v2d.json`，SHA256
`1543f8169606f86e6eee054e5ee59105e345f875a2d0ebca2f02b89bb5de1361`。

## 3. 预注册 guards

相对 T0，R1 必须同时满足：

1. eligible tail 的 relative-margin violation rate 更低；
2. `tail−pre` 与 `tail−clean` gap 都进一步变负；
3. explicit-token value-risk AP 不下降；
4. sparse-span AP 最多下降 `.02`；
5. correctness AUROC 最多下降 `.05`。

exact onset 只报告，不用于选择这个 full-tail reward hypothesis。冻结后没有扫描 margin、weight、epoch 或
阈值。

## 4. 结果

执行门全部通过：R1 完成 epoch 5，loss 全部 finite；每个 train epoch 观察 `6,681` 个 sparse tokens 和
`3,110` 个 relative-tail tokens，dev 分别为 `2,451/1,460`；absolute/pseudo tail 均关闭。

| diagnostic | T0 | R1 | R1 − T0 | guard |
|---|---:|---:|---:|---|
| relative-margin violation rate | .5863 | .0911 | -.4952 | pass |
| row-mean `tail−pre` | +.2317 | -3.2715 | -3.5032 | diagnostic improves |
| pooled `tail−pre` | +.3081 | -1.4034 | -1.7115 | pass |
| pooled `tail−clean` | -2.5660 | -1.6409 | **+.9251** | **fail** |
| explicit-token value-risk AP | .4399 | .3307 | **-.1092** | **fail** |
| sparse-span AP | .4156 | .3160 | **-.0996** | **fail** (`-.02` floor) |
| correctness AUROC | .9206 | .8889 | -.0317 | pass (`-.05` floor) |

R1 确实学会了行内相对下降：relative violation 从 `58.6%` 降到 `9.1%`，且 paired-query bootstrap 中
`tail−pre` pooled-gap difference 的 95% percentile interval 为 `[-4.6747,-.1323]`。所以不能把结果解释成
“relative loss 没被模型学到”。

但它没有产生相对 clean trajectory 的 tail-specific lowering：

| token population | T0 mean | R1 mean | shift |
|---|---:|---:|---:|
| hallucinated-row pre-onset | 1.0859 | 1.4229 | +.3370 |
| hallucinated-row tail | 1.3940 | .0195 | -1.3745 |
| clean trajectory | 3.9600 | 1.6604 | **-2.2996** |

clean 仍比 tail 高，但它比 tail 下移更多，因此 `tail−clean` 的 separation 反而缩小。对应 bootstrap 的
`tail−clean` delta 均值为 `+.8869`，`93.3%` resamples 大于 0。explicit hallucinated-vs-supported token
value gap 也从 `-2.3705` 缩到 `-1.2291`，与 value-risk AP 下降一致。sparse-span AP 的 bootstrap delta
在 `95.1%` resamples 小于 0，说明共享训练受到的损失不是一个孤立 point estimate。

## 5. 机制解释

R1 修掉了旧的 **全局平移自由度**，但没有提供 **clean-row 锚点**：relative loss 只看 hallucinated rows，
只约束它们内部的 tail 与 pre-onset 差值。它对 clean trajectory 的绝对或相对 value 没有任何直接项。
因此模型可以一边提高 hallucinated-row pre-onset anchor、一边降低其 tail，同时让 clean population 经共享
encoder/value head 和其他 losses 发生独立下移。观测到的 `pre +.337 / tail -1.374 / clean -2.300` 正好
符合这个未约束自由度。

sparse hallucination head 虽不是 token-value 标量本身，但它与 value 路径共享 encoder；R1 的辅助梯度因此
可以破坏 sparse-span 表示。当前样本只有 48 条、训练 5 epochs，不能把这解释成 relative-tail family 被
永久否证；但也不能在三个关键 guard 已失败后通过扫 weight 来追一个新的 selection-exposed 结果。

## 6. 冻结裁决与后续边界

- **保持 T0/S1 sparse-span BCE 为 standalone localization 默认。**
- 不扩跑 R1，不把它加入 mixed training，不读取 test，不声称改善 Best-of-N。
- R1 代码保留为可审计的实验开关；默认 `relative_tail_weight=0`，历史 loss 字典和 total 严格不变。
- 不永久否证 full-tail hypothesis。若以后再开，objective 必须显式控制 clean-row positional baseline（例如
  matched clean suffix-vs-prefix 的 difference-in-differences），并单独处理 shared-encoder 对 sparse head
  的干扰；这需要新协议，而不是继续扫 R1 的 weight/margin。
- exact onset 仍是独立未解决问题；R1 的 `±5` 为 `0/6`，不能借 tail 训练宣称 boundary localization。

机器结果为
`configs/hallucination_localization_v2/relative_tail_result_v2d.json`，SHA256
`6ced98914c37ee9cea0ac4686d3b1b079c806fd857589070d8774e18c33e71d1`。R1 checkpoint SHA256 为
`5ef0501c45bdb72dd800ac8e8f75462095c132c84fdfb586016a21f06fa83802`；输出根目录为
`run_artifacts/hallucination_localization_v2/pilot_relative_tail_v2d/`，训练代码 commit 为
`9fc26b54693afa1c25ba2483d5a240827afc6e31`。
