# Hallucination Full-Tail 4-Fold × 3-Seed 复核 v2c

状态：`completed_keep_t0_defer_tail`

日期：2026-08-18

证据等级：`pipeline pilot`；不是人工 Gold、独立 test、Best-of-N 或 formal mechanism evidence

## 1. 要关闭的问题

v2b 的单 seed、16-row dev 上，T2（S1 sparse-span BCE + `tail_weight=.5`）同时改善 span AP、
explicit-token value-risk AP 和 correctness AUROC，并通过当轮四道 point-estimate guards。因此 v2b 正确
撤销了“tail 已被永久否证”的过强说法，但不足以把 T2 直接设为默认。

v2c 在打开任何新增 held-out 指标前冻结两个候选：

- T0：S1 sparse-span BCE，`tail_weight=0`；
- T2：相同 S1，增加 full-onset-tail margin，`tail_weight=.5`、margin `.5`。

本轮只回答“当前 T2 实现是否能跨 seed、跨 fold 稳定地产生 **tail-specific** value lowering，同时不破坏
sparse localization 与 correctness”。无论结果如何，都不删除 tail 代码或永久否证其他相对/锚定 tail
objective。

## 2. 扩大的 out-of-fold 设计

现有 64 条 adjudicated trajectory 已经有 exact-token sparse labels。为了不等待新标注，同时避免只重复
原 16-row dev，本轮冻结 4-fold：

- fold 0：原 16-row dev，已经用于选择 `.5`，只作 continuity diagnostic；
- folds 1–3：把原 48-row train 按 `path_hallucinated` 分层、SHA256 key 确定性分为 16/16/16；
- 每个 fold 都在另外 48 条上训练，在该 fold 的 16 条上评估；
- 主要采用门只拼接 folds 1–3，得到每个 seed 48 条不重复的 out-of-fold prediction；
- seeds `42/43/44`；两个 variants、四个 folds，共 24 cells。fold-0/seed-42 的两个 v2b cell 严格复用，
  新跑 22 cells。

四个 dev fold 两两无交集并覆盖全部 64 条，每个 train/dev 都同时包含 path/correctness 两类与 explicit
positive/negative token。数据协议、审计和训练协议分别为：

```text
configs/hallucination_localization_v2/tail_cv_data_protocol_v2c.json
run_artifacts/hallucination_localization_v2/data_tail_cv_v2c/fold_audit_v2c.json
configs/hallucination_localization_v2/tail_cv_protocol_v2c.json
```

训练协议 SHA256 为 `a6d64ae8d6febf8c99d393fd62513972ab16377c8669052165a37c548e49e482`；fold audit
SHA256 为 `4412efeca3ffe21a02e2ab3762ee1964b7d99d4842cc1516d4b76be89f95ebca`。

## 3. 预注册采用门

对每个训练 seed，先拼接 folds 1–3 的 48 条预测，再要求：

1. `tail−pre` 与 `tail−clean` 相对 T0 都进一步变负；
2. `−token_value` 的 explicit-token risk AP 不低于 T0；
3. sparse-span probability AP 最多下降 `.02`；
4. reward-score correctness AUROC 最多下降 `.05`。

T2 被采用还必须满足：至少 `2/3` seed 通过四门、跨 seed mean 通过同样方向、任何 seed 的 span AP 不得
下降超过 `.05`、correctness AUROC 不得下降超过 `.10`。fold 0 明确不得参与采用裁决。exact onset 只报告，
不用于选择 full-tail reward hypothesis。

## 4. 48-row confirmatory 结果

T2−T0 的逐 seed 差值如下：

| seed | tail−pre gap | tail−clean gap | value-risk AP | span AP | correctness AUROC | 四门 |
|---:|---:|---:|---:|---:|---:|---|
| 42 | -.1000 | **+.1517** | +.0545 | -.0179 | +.1252 | fail locality-vs-clean |
| 43 | +.1807 | **+.4297** | +.0307 | +.0375 | -.0591 | fail locality/correctness |
| 44 | -.1911 | **+.1737** | +.0219 | +.0315 | -.0383 | fail locality-vs-clean |
| mean | -.0368 | **+.2517** | **+.0357** | **+.0170** | **+.0093** | fail locality-vs-clean |

这不是“T2 没效果”：

- value-risk AP 在三个 seed 都提高；
- span AP 在两个 seed 提高，一个 seed 下降但仍在 `.02` 保护范围内；
- correctness 没有 catastrophic seed，跨 seed mean 略升；
- tail margin violation 从 T0 的较高水平下降到 T2 的 `.7%–4.4%` 左右。

但它没有稳定地产生 tail-specific lowering。三个 seed 的 `tail−clean` 都变得 **更不负**；换言之，T2
虽然把 tail 压低，却把 clean token 压得更多。没有一个 seed 通过全部四门。

2,000 次 query bootstrap、在每个 resample 内先算三个 seed 的 T2−T0 再取均值：

| diagnostic | mean difference | 95% percentile interval |
|---|---:|---:|
| tail−pre gap | -.0283 | [-.3048, +.2210] |
| tail−clean gap | **+.2608** | [-.1289, +.6745] |
| explicit-token value-risk AP | +.0340 | [-.0175, +.0874] |
| span AP | +.0162 | [-.0158, +.0444] |
| correctness AUROC | +.0098 | [-.0315, +.0530] |

bootstrap 仍是 pipeline diagnostic；采用门失败来自预先冻结的逐 seed locality 条件，不是事后根据区间改门。

## 5. 为什么这是全局 shift，而不是局部 tail 信号

confirmatory 48 rows 的 mean token values 给出直接证据：

| seed | clean T0 → T2 | clean shift | tail T0 → T2 | tail shift |
|---:|---:|---:|---:|---:|
| 42 | +.563 → -3.165 | -3.728 | -.554 → -4.130 | -3.576 |
| 43 | -.713 → -4.266 | -3.552 | -2.145 → -5.267 | -3.122 |
| 44 | +.754 → -3.865 | -4.620 | -.095 → -4.541 | -4.446 |

三个 seed 都是 clean shift 的绝对值大于 tail shift，所以 `tail−clean` gap 必然恶化。这与实现中的最省力
解一致：

- tail loss 是单边 `relu(token_value_tail + margin)^2`，只要求 tail 低于绝对负 margin；
- 同一个 token-value head 作用于所有 token，没有 clean/pre-onset 的相对锚点；
- correctness score 是 gated mean token value 再加独立 `score_residual`，residual 可以补偿全局 value
  偏移；
- 因而共享 head 可以主要学习一个大幅负 bias 来满足 tail margin，而不必学会“只在 onset 后下降”。

这是代码结构与三 seed value population 一致支持的机制解释，而不是仅凭一次异常指标猜测。value-risk 与
span AP 的正向变化说明这个全局 regularization 可能仍有用，但当前实验不能把收益归因于 tail locality。

## 6. 严格裁决

本轮裁决为：

- **当前 standalone hallucination localization 默认选择 T0：S1 sparse-span BCE，不加 full-tail shaping。**
- T2 `.5` 的当前绝对-margin 实现暂缓，不进入 mixed training；原因是 0/3 confirmatory seeds 通过
  tail-specific locality，而不是因为 AP 没有提高。
- 不永久抛弃 tail hypothesis。若未来重开，应测试有 clean/pre-onset anchor 的相对 margin、row-centered
  value 或显式 contrastive tail loss，并发布新协议；不能继续扫当前绝对 margin 权重。
- exact onset 仍是独立未解决问题；本轮 confirmatory T0/T2 的 `±5` 总体仍接近零。
- 不读取 pilot/final test，不声称改善 Best-of-N，不授权 mixed training。
- hallucination 模块的 pipeline-selection gate 至此关闭，按用户授权进入 dual-prior 模块。

机器可读结果为
`configs/hallucination_localization_v2/tail_cv_result_v2c.json`，SHA256
`d8264599b61c221958598d8a087ded7617a9cfdf5b8cdee46716ad4ea2d2a5d9`。22 个新 cell 的 clean training
commit 为 `743c3512995614df4f625888c66bc15912cce31e`；输出根目录为
`run_artifacts/hallucination_localization_v2/pilot_tail_cv_v2c/`。
