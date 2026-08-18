# Hallucination Full-Tail 撤销审计与直接比较 v2b

状态：`completed_retain_tail_for_larger_validation`

日期：2026-08-18

证据等级：`pipeline pilot`；不是人工 Gold、formal mechanism evidence 或 Best-of-N 结果

## 1. 为什么必须重新审计

此前文档把两件不同的事都简称为 tail，并据 v1/v2 的小样本结果写成“不启用 negative-tail”：

1. **onset-tail classification**：从首个裁决 hallucination onset 开始，把后续每个 token 的
   hallucination target 都写成 1；
2. **negative-tail reward shaping**：仍用 sparse reviewed span 学 token hallucination probability，但从
   onset 开始把后续每个 token value 压到负 margin 以下。

v2 足以让第一种标签不再成为当前首选，却没有直接比较第二种 shaping 与当前 S1 sparse-span branch。
因此旧证据不足以支持“永久抛弃 tail”。在用户要求下，本轮先冻结撤销审计，再用同一数据、seed、架构和
预算直接比较；禁止根据结果反改门槛。

机器可读撤销审计为
`configs/hallucination_localization_v2/tail_hypothesis_audit_v2b.json`，SHA256
`9d7cd5ca49c2debf1688f5ba83b91271fc5ed0ba13cc9d580114775ef57cc494`。

## 2. 旧证据为什么不充分

### 2.1 onset-tail classification

v2 matched S0/S1 的 sparse reviewed-span dev 指标为：

| cell | token target | span-token AP |
|---|---|---:|
| S0 | onset 后全部为 positive | .3713 |
| S1 | 只监督 reviewed claim spans | .4156 |
| shortcut | absolute token position | .3933 |

S1−S0 的 2,000 次 paired query bootstrap 均值为 `+.0412`，95% percentile interval
`[-.0319, +.1277]`。所以 S1 是当前较合理的 point-estimate 选择，S0 还低于位置 shortcut；但区间跨 0，
这只能支持“当前不选择 S0”，不能支持永久否证 onset-tail classification。

### 2.2 negative-tail reward shaping

v1 的 H2 在 contaminated onset-tail BCE 上增加 `tail_weight=.5`、margin `.5`。它把 dev tail margin
violation 降到 `0%`，但 mean token value 同时成为 clean `-2.55`、pre-onset `-3.84`、tail `-4.33`，
没有通过当时的 locality gate；token AP `.4973` 也低于位置 baseline `.514`。

这个失败说明 v1 实现不能被直接采用，但 H2 从未与当前 S1 sparse-span token target 配对，所以不能由此
推出 full-tail shaping 在当前方案下无效。

## 3. Full tail 到底监督了什么

当前 S1 token label 只把 reviewed supported span 标为 0、reviewed unsupported/contradicted span 标为 1，
其余 token 均 mask。full tail 则覆盖 hallucinated row 上 onset 及之后的全部生成 token：

| split | hallucinated rows | full-tail tokens | 明确 hallucinated | 明确 supported | unreviewed | unreviewed / full tail |
|---|---:|---:|---:|---:|---:|---:|
| train | 17 | 3,220 | 922 | 61 | 2,237 | 69.47% |
| dev | 6 | 1,460 | 528 | 127 | 805 | 55.14% |

dev 已审 tail token 中还有 `127/655 = 19.39%` 是明确 supported。由此得到严格语义边界：

- full tail **不是** token-level hallucination ground truth；
- 它有意降低 supported 和 unreviewed post-onset token 的 value；
- 它检验的是“首错后整段 trajectory 的 reward 应受污染惩罚”这一独立假设；
- 是否应该惩罚首错后的正确恢复步骤，是 reward semantics 的选择，不能从 sparse labels 自动推出。

## 4. 冻结直接比较

三组都使用 query-disjoint train/dev `48/16`、seed 42、5 epochs、batch 4、LR `1e-4`、同一 33-layer
Phi features、同一 9.55M 参数 CLIR 和相同 correctness + S1 sparse-span BCE。每 epoch 精确使用 6,681
sparse train labels；只有 tail loss weight 不同：

| cell | sparse-span BCE | full-tail margin weight |
|---|---:|---:|
| T0 control | 1.0 | 0 |
| T1 light | 1.0 | .1 |
| T2 historical | 1.0 | .5 |

margin 固定为 `.5`。预先冻结四道护栏：

1. 相对 locality：相较 T0，`tail−pre` 与 `tail−clean` 都必须更负，避免把全局平移误称为局部作用；
2. semantic value：以 `−token_value` 作为风险分数时，explicit-token AP 不得低于 T0；
3. sparse span：hallucination-probability AP 最多比 T0 低 `.02`；
4. correctness：reward-score correctness AUROC 最多比 T0 低 `.05`。

协议为 `configs/hallucination_localization_v2/tail_comparison_protocol_v2b.json`，SHA256
`1b283158438d1298a88b01d3f0308b4bd042edd824cb7256659d55f6efbb1ea5`。协议明确规定：单 seed、16-row dev
无论结果如何都不能永久否证 tail，也不能授权 mixed/test training。

## 5. 结果

三个 cell 均完成 epoch 5、loss finite、数据/seed/架构/预算/训练 commit 一致：

| cell | span AP | value-risk AP | correctness AUROC | tail−pre | tail−clean | margin violation | onset MAE / ±5 | 四门 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| T0 | .4156 | .4399 | .9206 | +.3081 | -2.5660 | .6658 | 82.5 / 0/6 | control |
| T1 `.1` | .3697 | .4328 | .9048 | +.0931 | -1.6219 | .2788 | 320.8 / 0/6 | fail |
| T2 `.5` | **.4535** | **.5002** | **.9524** | **-.3097** | **-2.7142** | **.1671** | 71.7 / 0/6 | **pass** |

T1 只改善了相对 pre-onset 的 gap，没有改善相对 clean 的 gap，并且 value-risk 与 span AP 都低于 control，
所以不保留。T2 则通过全部预设 point-estimate guards：tail 相对于 pre-onset 和 clean 都进一步降低，
explicit-token value-risk、sparse span ranking 和 correctness 均未被破坏。

2,000 次 paired query bootstrap 对 T2−T0 给出：

| diagnostic | mean difference | 95% percentile interval |
|---|---:|---:|
| tail−pre gap | -.6421 | [-1.4337, -.0868] |
| explicit-token value-risk AP | +.0560 | [+.0082, +.1077] |
| span hallucination AP | +.0348 | [-.0149, +.0992] |
| correctness AUROC | +.0324 | [.0000, +.1273] |

前两项区间支持当前方向，但这仍只是对 16 条 dev trajectory 的 diagnostic bootstrap，不等于跨训练 seed
稳定或 formal significance。T2 也没有解决 exact onset：`±5` 仍为 `0/6`。

## 6. 严格裁决与后续条件

本轮允许的结论只有：

- **不抛弃 full-tail reward shaping**；保留 T2 `.5` 进入扩大 validation 与多 seed 的 matched T0/T2 比较；
- T1 `.1` 在本 pilot 中失败，不进入下一轮；
- S1 sparse-span classification 继续作为当前 token hallucination branch；
- onset-tail classification S0 仍未被选择，但也没有被永久否证；
- full-tail value 是独立的 error-contamination reward hypothesis，报告时不得称作 token 幻觉标签；
- pseudo-tail、mixed 3968-row mechanism training、pilot/final test 和 Best-of-N 仍未获授权；
- 只有扩大验证中的相对 locality、semantic value、span 与 correctness guards 在多个 seed 上保持，才可讨论
  把 T2 接入 mixed mechanism run；exact onset 必须由独立 boundary/decoder 路线处理。

机器可读结果：
`configs/hallucination_localization_v2/tail_comparison_result_v2b.json`，SHA256
`29b5d20e4563f217b22ecd392c381fdd5aaf8f044830705542f21d7031df6bbe`。训练输出：
`run_artifacts/hallucination_localization_v2/pilot_tail_v2b/`。训练 commit 为 `08b9e82446f5a63bef1c1216d2748d1778df181e`，
结果汇总 commit 为 `dd8e152e9aa59418c772e81b5a2a3862bb370feb`；本文件不改变冻结结果。
