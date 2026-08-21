# JP → Hallucination Frozen Probe v1

最后更新：2026-08-21

## 1. 问题与冻结边界

本轮只回答一个问题：项目原始 JP（correctness + direct key/complete + 双向 stop-gradient mutual `.25` +
shared-gradient gate `10`）的 epoch-5 token representation，在完全不更新 JP 的前提下，是否已经线性包含
sparse hallucination localization 信号。

协议在读取结果前提交为 `17d0efa6de14d16dfa18d82a561562949b3eed3f`。冻结协议为
`configs/jp_h_frozen_probe_v1/training_protocol_v1.json`，SHA256
`f42fa15f084111921cdc1901339664bc9b46f730d15ad274f7433e2053cc3983`。

唯一训练对象是：

```text
stopgrad(JP conditioned token_features [T, 768])
    → Linear(768, 1)
    → sparse token BCE
```

- 每个 probe 只有 `768 + 1 = 769` 个参数；
- JP checkpoint 只读，不进入 optimizer；
- 不修改 correctness、direct prior、mutual distillation、reward gate 或最终 score；
- 不使用 full tail、onset tail、path MIL、consistency、progress 或 reconstruction；
- 4 folds × seeds `42/43/44`，每格 100 个 full-batch AdamW steps；
- fold 0 是历史 selection-exposed continuity，主要裁决只拼接 folds 1–3 的 48 条 OOF predictions；
- 没有访问 `pilot_test` 或 `final_test`。

同批位置基线必须随评价行重新计算。历史 `.3933/.4220` 来自 fold-0 的 16 条，不能直接作为另外 48 条的
position baseline。主要 48 条的 strongest position AP 是 span `.259916`、claim `.304641`；历史阈值只作
continuity report。

## 2. 工程与原方法保护门

12/12 cells 完成，12 个 probe checkpoint 均唯一且只含 `weight/bias`。所有格训练 BCE 都下降。

JP checkpoint 的运行前、feature extraction 后和全部 probe 训练后 SHA256 均为：

```text
a392ad6ad8ccbe020e6a10835463c15829e3684f0a2b9213e6bcefe78666df5c
```

当前代码重新 forward 64 条时：

- 64/64 `reward_score` 与原 JP retained scoring bit-identical，最大绝对差 `0`；
- 64/64 每个 `token_value` bit-identical，最大绝对差 `0`；
- 12 个 OOF views 共 192 行的 base reward 字段全部逐值不变；
- base parameters in optimizer 为 `0`，probe checkpoint 不嵌入 base state dict。

因此本轮的负/正结果都不能归因于误改项目原方法。

## 3. 主要结果

主要 48 条 OOF pool：

| Seed | Span AP | Position AP | Delta | Claim-mean AP | Position AP | Delta | 双门 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | `.23567` | `.25992` | `-.02424` | `.35821` | `.30464` | `+.05356` | fail |
| 43 | `.23290` | `.25992` | `-.02702` | `.34085` | `.30464` | `+.03621` | fail |
| 44 | `.23430` | `.25992` | `-.02562` | `.32486` | `.30464` | `+.02022` | fail |
| mean | `.23429` | `.25992` | `-.02563` | `.34131` | `.30464` | `+.03666` | fail |

冻结采用规则要求至少 2/3 seeds 同时通过 span 与 claim，并要求两个 mean delta 都严格大于 0。实际为 0/3，
状态固定为：

```text
completed_frozen_linear_probe_not_supported
```

机器结果为 `configs/jp_h_frozen_probe_v1/training_result_v1.json`，SHA256
`3c12fab6b864ed753485ddee1d03377417adeed49adbd089b4bb96a012015361`。

这不是“JP 完全没有 hallucination signal”。三个 seeds 的 claim-mean AP 都稳定超过同批 position baseline；
失败只集中在更细的 token/span 排序。

## 4. 事后机制诊断（不得用于追溯改判）

### 4.1 不是简单欠优化

每格 90→100 step 的训练 BCE 只继续下降约 `.0004–.0012`，而最终 BCE 已从初始化的约 `.56–.81` 降至
`.376–.442`。因此“原样多跑几轮”不是优先修复方向。

### 4.2 主要问题是 claim 内部高频抖动

所有 9,132 个 supervised tokens 都被一个人工 claim span 覆盖。对主要 48 条 OOF predictions 做只读诊断：

| 读出方式 | Seed 42 span AP | Seed 43 | Seed 44 | 角色 |
|---|---:|---:|---:|---|
| 原始逐 token linear probe | `.2357` | `.2329` | `.2343` | 冻结主结果 |
| centered 3-token mean | `.2802` | `.2728` | `.2691` | post-hoc diagnostic |
| annotated-claim 内取常数均值 | `.3836` | `.3693` | `.3522` | oracle mechanism diagnostic |

人工 positive claim 内概率标准差约 `.089`，negative claim 约 `.0505`。线性头能判断“哪一段 claim 更危险”，
但同一 claim 内 token probability 抖动较大；token micro-AP 会按 span 长度放大这种抖动。简单 3-token
局部平均已经让三个 seeds 都超过 `.2599` 的同批 position baseline，annotated-claim oracle constant
更明显改善。这里的 claim span 仍来自现有 Silver 标注，不是新的 Gold label。

这些平滑结果是在查看主结果后得到的，不能把当前 linear probe 追溯改判为通过，也不能直接选择某个窗口并在
同一 64 条上声称验证成功。它们只用于决定下一种结构应该处理 temporal/claim consistency，而不是继续扫描
共享梯度路由或增加 pointwise MLP 容量。

## 5. 解释与下一步

当前证据支持下面这个更窄的结论：

> 冻结 JP representation 含有可线性读取的 claim-level hallucination risk，但 plain pointwise linear head
> 尚不能稳定形成 token-level localization；瓶颈更像 claim 内局部一致性，而不是 JP/H 必须共享训练。

下一格推荐仍保持 JP 完全冻结，只给 H branch 增加一个极小、域无关的 temporal smoother：

```text
h_t = stopgrad(JP token_features_t)
z_t = Linear(h_t)
z'_t = masked local smoothing(z_1 ... z_T)
p_t = sigmoid(z'_t)
```

优先于大 MLP/Transformer adapter 的理由：

1. 它直接对应本轮观察到的 claim 内抖动；
2. 不需要 GSM8K 数字、等号或领域规则；
3. base reward、prior、mutual 和 gate 仍结构性不变；
4. 可以先用 fixed 或极少参数的 masked smoother，避免在 48 条上训练一个大 adapter；
5. inference 不需要 gold claim span。

但窗口/尺度已经被本轮 post-hoc 诊断触及。严谨采用需要新增、未看过的 localization validation labels；在
现有 64 条上做 nested train-only selection 最多只能作为工程 pilot，不能恢复 confirmatory 身份。建议顺序：

1. 与用户冻结一个最小 temporal smoother，而不是继续 pointwise head 或 shared-gradient surgery；
2. 若只求先验证实现，可在现有 64 条做 nested OOF exploratory pilot，所有窗口选择只能看各 fold train；
3. 同时新增至少一批未参与本轮选择的 H validation labels，并覆盖非数学领域；
4. smoother 在新标签上通过 span/claim gate 后，再讨论把冻结的 `p_t` 显式接入 JP score；
5. score coupling 仍单独冻结，例如只学习一个非负 `alpha`，不得把本轮 localization 结果直接等同于
   Best-of-N 增益。

当前不授权自动运行 temporal smoother、H adapter、score coupling 或 label expansion。下一项仍需用户逐项拍板。
