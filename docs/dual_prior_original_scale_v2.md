# Dual-Prior Original Shared-Gradient Scale v2

状态：`frozen_before_training`

证据等级：`small-scale real`

日期：2026-08-19

## 1. 当前裁决

本轮不改项目原方法。保留三部分：

```text
direct key / complete membership BCE
+ MSE(A_key, stopgrad(A_complete))
+ MSE(A_complete, stopgrad(A_key))
+ MSE(A_gate, stopgrad(0.5 A_key + 0.5 A_complete))
```

gate auxiliary 的梯度仍进入 reward gate 与共享 token representation；只对 fused prior target stop-gradient。
没有 head-only detached-feature repair，没有 containment replacement，没有 reconstruction，也不混入 consistency、
hallucination 或 tail loss。

v1 G1 的冻结结论不回写：它确实学会 fused prior，但未通过当时的 key-AP protection。新 v2 回答另一个更直接
的问题：在更大 outcome 分布训练后，原始 gate 是否改善同题多候选排序。

## 2. 数据合同

来源是 checker-v5 的 4096-row / 512-query outcome train。64 条 prior Gold 原本来自其中 64 个不同 query：

- 16 个 localization-dev query 连同各自全部 8 个 candidates 一起排除，共 128 rows；
- 剩余 496 queries / 3968 rows 保持每题完整 8-candidate group；
- 48 条 prior-train Gold 只写回精确对应的 48 rows；
- 其余 3920 rows 不含 prior fields，而不是写全零伪标签；
- ranking validation 是独立 500 queries × 16 candidates，含 146 个 mixed pools，与 train 和 prior-dev 的
  query overlap 都为 0。

物化 manifest SHA256 为
`3c1ac6068343328ef23d85e380985be641c564ce29d4492e6e95b1526d97f310`。协议和审计为：

```text
configs/dual_prior_original_scale_v2/data_protocol_v2.json
configs/dual_prior_original_scale_v2/data_report_v2.json
run_artifacts/dual_prior_original_scale_v2/data_v2/train3968_mixed.jsonl
```

## 3. 冻结比较

```text
G0 = correctness BCE + direct key/complete BCE + original mutual MSE(.25)
G1 = G0 + original shared-gradient gate alignment(10.0)
```

两格从头训练，seeds `42/43/44`，5 epochs，batch size 2，LR `1e-4`，BF16，全 33 层 Phi features。
`10.0` 继承 v1 在读取 gate-enabled 结果前完成的梯度量级审计；本轮不扫描权重。G0/G1 的候选、split、
训练预算和除 gate 之外的 loss 完全相同。

每格发布两类结果：

1. 48-row train / 16-row dev 的 key/complete AP、mutual discrepancy、gate→fused-prior MSE；
2. 500×16 validation 上 k=`1/2/4/8/16` 的 reward Best-of-N、random、oracle 与 within-query pairwise
   accuracy。

Primary comparison 是 G1−G0 BoN@16：seed 内按 query 配对；最终先对每个 query 的三 seed delta 取均值，
再 bootstrap 500 queries。稳定正向要求三 seed 至少 2 个为正、mean 为正、aggregate paired 95% CI 下界
大于 0。

## 4. 解释边界

- 正结果可以称为 original shared-gradient dual-prior 在当前 validation 上产生稳定 ranking signal；仍不是
  pilot/final-test 或跨领域正式结论。
- 负结果只说明当前 48 条 prior Gold + 3968-row mixed training 没有建立稳定 ranking 增益；不得回写 v1、
  事后换门或自动采用 head-only。
- 无论结果正负，项目原始公式都保持实现；后续若改监督采样、权重或架构，必须另发协议。

## 5. 执行入口

```bash
cd /prodcpfs/user/panzhixin/ICLR_2027
P=/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python

"$P" scripts/run_dual_prior_matrix_v1.py \
  --protocol configs/dual_prior_original_scale_v2/training_protocol_v2.json \
  --gpus 0,1,2,3,4,5 \
  --max-parallel 6 \
  --execute
```

矩阵要求 clean committed worktree；6 cells 完成后自动调用
`scripts/summarize_dual_prior_original_scale_v2.py`，机器结果写入
`configs/dual_prior_original_scale_v2/training_result_v2.json`。
