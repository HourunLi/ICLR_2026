# CLIR Stage 1 small-scale correctness-only 结果

日期：2026-08-13

证据等级：`small-scale real`

协议：`configs/stage1_small_scale_v1.json` / `docs/stage1_protocol.md`

代码：`7d1d49e02f317390ac1d94c19bff7739d84d4480`，9 个训练运行均记录
`dirty=false`。

本阶段在任何 rewrite、hallucination onset 或 dual-prior target 进入数据之前，使用完全相同
的真实候选、全 33 层特征、标签和 split 比较 `strict_swift`、`encoded_swift` 与
correctness-only `clir`。结果只来自冻结的 validation；没有生成或检查 `pilot_test`。

## 1. 数据与覆盖门

| split | query × candidates | correct / incorrect | mixed / all-correct / all-wrong | feature bytes |
|---|---:|---:|---:|---:|
| train | 512 × 8 = 4096 | 3658 / 438 | 117 / 379 / 16 | 245,452,377,600 |
| validation | 128 × 16 = 2048 | 1772 / 276 | 39 / 84 / 5 | 123,764,154,752 |

标签覆盖门全部通过：train correct/incorrect 均超过 128，validation mixed query 为 39，超过
预注册下限 32。特征统一为 embedding 加 32 个 block outputs，`33 × 3072 = 101376`，
BF16；总特征 payload 为 369,216,532,352 bytes（343.86 GiB）。

本次训练从已完成 marker、payload size/SHA256、tensor reload/shape/finite 验收过的 shard
复制到本机 scratch 读取。源目录和 scratch 的 6784 个唯一 trajectory/condition payload
逐文件 SHA256 一致，训练 manifest 只改变物理路径，不改变行、标签、token 或 tensor 内容。
checkpoint、日志、评分与评估输出均保存在持久化 `run_artifacts/stage1_small_scale_v1`。

## 2. 训练验收

三种模型均以 seeds 42/43/44、5 epochs、最终 epoch、batch size 2、AdamW、LR 1e-4、
BF16 完成，共 9 个 checkpoint。每个运行均有 5 条 epoch metrics、相同 train/validation
SHA256 和 `status=completed`，未出现 OOM、NaN 或 traceback。

所有 15 个 CLIR train/validation epoch 中，key/complete/distill/gate/reconstruction 的
loss 与 applicable count 都严格为 0。这证明本次比较没有把缺失的 auxiliary target 当作
隐式监督；`clir` 只是 correctness-only backbone。

## 3. Validation Best-of-N

下表为三个训练 seed 的 reward-selected accuracy 均值 ± 样本标准差，单位为百分比。每个
单 seed evaluator 另保存 2000 次 query bootstrap 95% CI。

| 方法 | BoN@1 | BoN@2 | BoN@4 | BoN@8 | BoN@16 |
|---|---:|---:|---:|---:|---:|
| strict SWIFT | 88.28 ± 0.00 | 89.84 ± 0.00 | 90.10 ± 0.45 | 90.89 ± 1.63 | 88.28 ± 2.07 |
| encoded SWIFT | 88.28 ± 0.00 | 88.80 ± 0.45 | 90.36 ± 0.45 | 90.63 ± 0.00 | 88.54 ± 1.63 |
| CLIR correctness-only | 88.28 ± 0.00 | 89.58 ± 0.45 | 90.36 ± 1.80 | 89.32 ± 1.80 | **89.32 ± 2.51** |
| random expected | 88.28 | 89.06 | 89.65 | 88.96 | 86.52 |
| random seeded | 88.28 ± 0.00 | 89.58 ± 0.45 | 89.84 ± 0.00 | 89.84 ± 1.35 | 85.94 ± 1.35 |
| oracle | 88.28 | 90.63 | 93.75 | 95.31 | 96.09 |

BoN@1 对所有方法必然相同，因为第一候选集合中没有排序选择。BoN 随 k 并不单调：候选池
加入更多 distractor 后，三个 reward model 都可能选到错误候选；oracle 的单调上升只说明
正确答案存在，不代表 reward 已能稳定识别它。

## 4. 预注册主比较：BoN@16

正数表示右侧方法优于左侧。均值和样本标准差跨三个训练 seed；CI 是每个 seed 内对相同
128 个 query 做 2000 次配对 bootstrap，单位均为百分点。

| 比较 | seed 42 / 43 / 44 | 跨 seed 均值 ± 标准差 | 各 seed 配对 95% CI |
|---|---:|---:|---|
| random expected → strict | +0.98 / +4.10 / +0.20 | +1.76 ± 2.07 | [-2.78, 4.69] / [0.63, 7.52] / [-4.15, 4.00] |
| random expected → encoded | +3.32 / +0.20 / +2.54 | +2.02 ± 1.63 | [0.10, 6.64] / [-3.91, 4.49] / [-0.68, 5.91] |
| random expected → CLIR | +5.66 / +1.76 / +0.98 | +2.80 ± 2.51 | [2.49, 8.79] / [-1.95, 5.42] / [-2.73, 4.54] |
| strict → encoded（编码器效应） | +2.34 / -3.91 / +2.34 | +0.26 ± 3.61 | [-0.78, 6.25] / [-9.38, 0.78] / [-2.34, 7.03] |
| encoded → CLIR（backbone 效应） | +2.34 / +1.56 / -1.56 | +0.78 ± 2.07 | [0.00, 5.47] / [-3.13, 6.25] / [-6.25, 3.13] |

## 5. 结论与边界

这批真实数据证明 correctness-only reward 能学到**弱的候选排序信号**：CLIR 在三个 seed
的 BoN@16 都高于 random expected，平均高 2.80 个百分点。但只有 seed 42 的配对 CI 明确
高于 0，其余两个 seed 的 CI 均跨 0。

本阶段**不支持“CLIR backbone 稳定优于 encoded SWIFT”**：平均增量只有 0.78 个百分点，
seed 44 方向为负，三个 seed 的配对 CI 都没有给出稳定正证据。也不支持“full-layer encoder
稳定优于 raw strict SWIFT”：平均增量 0.26 点且 seed 43 明显为负。因此不能把当前最高
均值写成已经建立的方法收益，也不应据此查看 `pilot_test`。

主要统计瓶颈是 validation 只有 39/128 个 mixed pools；84 个 all-correct 和 5 个
all-wrong query 在 BoN@16 上对所有选择器都是结构性不敏感的，有效比较样本远小于 128。
训练标签也有 89.31% correct 的明显不平衡，而冻结协议预先规定不做 class reweighting。

## 6. 下一步决策

进入新的、单独版本化的 Stage 1B validation-strengthening/diagnostic，而不是直接宣称通过
Stage 1 或查看 test：

1. 先在现有 39 个 mixed validation pools 上审计逐 query 分歧、长度偏差、score margin 和
   layer attention，明确错误类型；该分析不调参。
2. 冻结一个只扩展 ordered prefix 的 Stage 1B 配置，将 validation 扩到足够多的 mixed
   query，并视容量决定是否同步扩大 train；禁止按答案 cherry-pick。
3. 若修改 class weighting、pointwise objective、epochs 或生成温度，必须新建配置和新的
   validation 口径，不能覆盖本结果。
4. 只有 validation 上的 random 增益及 encoded→CLIR 增量在多 seed 下稳定后，才锁定配置
   并一次性进入 `pilot_test`；否则把本阶段保留为弱信号/负增量证据。

机器可读的九份单 seed 报告与汇总位于
`run_artifacts/stage1_small_scale_v1/evaluations` 和
`run_artifacts/stage1_small_scale_v1/stage1_summary.json`。`summarize_clir.py` 会校验所有
report 的 query/k/baseline 一致性，计算跨 seed 样本标准差和主指标配对 CI。
