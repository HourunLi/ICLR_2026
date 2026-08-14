# CLIR Stage 1B v4 第四轮审查修订协议

机器配置：`configs/stage1b_validation_v4.json`
唯一正式入口：`scripts/run_stage1b_validation.py`

## 1. 定位与版本关系

v4 在正式矩阵启动前取代 v3，修复第四轮代码审查发现的健康门、失败矩阵、epoch snapshot 和
resume provenance 问题。v3 保持冻结，不原地改写，也没有可报告的正式效果数字。

v4 仍只使用 outcome-level correctness BCE。train/validation 中 consistency、hallucination、
onset、progress/advantage、key/complete prior 和 reconstruction 的适用覆盖都是 0。因此本实验只做
容量/优化控制，不能支持或否证 CLIR 的机制主张。`pilot_test` 保持未读取。

## 2. 冻结输入

v4 复用经过 v3 preflight 验证的 v5 label manifests 和全部 33 层 feature，不重复生成或提取：

| split | rows / queries / candidates | correct / incorrect | mixed / all-correct / all-wrong | SHA256 |
|---|---:|---:|---:|---|
| train | 4096 / 512 / 8 | 3668 / 428 | 111 / 385 / 16 | `6faca803b108675ed7ae08e54416969b483e7b9918133b51a0c9aff92fe02e06` |
| validation | 8000 / 500 / 16 | 7155 / 845 | 146 / 342 / 12 | `1e55ccfea2de2aa8190072e8a8c5784aec3d64059751355f6a5bde23295a6c36` |

checker 为 `clir_gsm8k_numeric_v5`；label protocol SHA256 为
`4245dcf4d6f15583d03c1e8088481b597df41a4fe39daf1c2e098e884c3ad8ef`。

feature integrity gate 已覆盖 12,096 行、13,108 个唯一 payload、725,761,877,084 bytes，
0 failure；报告 SHA256 为
`e38b14446dca77926d28361aa7cb3877a7970aa7114b8980ca3141fdc7354d25`。v4 输出使用新目录
`run_artifacts/stage1b_v4`，不会覆盖历史 artifact。

## 3. 冻结矩阵与计算口径

- variants：`strict_swift`、`encoded_swift`、`clir`
- training seeds：42、43、44
- 5 epochs；final epoch 为预注册主 checkpoint
- batch size 2，learning rate `1e-4`，weight decay 0，gradient clip 1.0
- BF16 training；4 workers、pinned memory、persistent workers
- FP32 scoring，batch size 2
- evaluation k=`1/2/4/8/16`，query bootstrap 10,000 次，95% CI
- primary metric：final-epoch reward BoN accuracy@16
- candidate subset：`first_k_by_vllm_completion_output_index`

每个 epoch 保存 immutable full-state snapshot，但它只用于注明 post-hoc 的训练轨迹诊断。

## 4. 三道预注册健康门

### 4.1 Train：相对常数先验改善

令训练集正例比例为 `p`，常数类别先验 BCE 为二元熵 `H(p)`，最终 train correctness BCE 为
`L`。定义：

```text
relative_improvement = (H(p) - L) / H(p)
```

必须满足 `relative_improvement >= 0.01`。这是一侧门：`L > H(p)` 明确失败，不再因绝对距离大而
通过。阈值从 v3 的 2% 调为 1%，避免健康 run 距硬失败边界过近。

### 4.2 Score：分数散布

validation 全部 reward score 的 population standard deviation 必须至少为 0.1。打分器总是先写
`*.jsonl.health.json`；失败时不发布 scored JSONL。

### 4.3 Evaluate：query 内排序

在每个 query 的 max-k pool 中枚举 correct-vs-incorrect 候选对，correct score 更高记 1，精确
tie 记 0.5，更低记 0；跨所有 pair 做 micro-average。必须满足 pairwise accuracy ≥ 0.60，且至少
存在一个 informative pair。

0.60 位于第四轮审查对历史 v1 的坏 run（0.4763/0.5735）与健康 run（最低 0.6666）之间。
这些历史数字只用于在 v4 运行前冻结诊断门，不能作为 v4 效果证据。

## 5. 失败 cell 的预注册处理

门禁失败是实验结果，不是可以静默覆盖的异常：

- train failure：`run.json` 必须为 `health_gate_failed` 并保存完整 gate evidence；
- score failure：必须保存 scoring health sidecar；
- ranking failure：evaluation JSON 必须保存 ranking health evidence；
- 失败入口随后退出非零，调度者继续运行其他独立 cell。

final summary 默认要求 9/9 完整。只有显式传 `--allow-failed-cells` 时，才允许省略与同一 v4
协议 SHA256、冻结阈值和 cell 身份匹配的健康失败。未知、未运行、损坏或协议不匹配的缺口仍报错。

任何省略后的产物必须满足：

```text
result_status = incomplete_diagnostic_only
matrix_complete = false
formal_primary_claim_allowed = false
```

即使 9/9 全部健康失败，也只发布无指标的矩阵诊断壳。若要改变阈值或重跑失败 cell，必须发布新协议。

## 6. Snapshot 与 resume

指定 `--checkpoint-epoch N` 时，launcher 读取 metrics JSONL 中 epoch N 自己的
`training_health`，不读取 final run 的健康状态。这样 final 失败时仍可诊断较早的健康 snapshot；该
结果不能替代 final primary，也不允许使用 incomplete-summary policy。

full-state checkpoint/run 必须记录：

- `start_epoch`
- `resumed_from.path/sha256/completed_epoch/device`
- 每段 `training_segments` 的 epoch 范围和 resolved device

resume 同时 pin CLI device 选择并校验 resolved device type。CPU↔CUDA、CUDA↔MPS 等跨设备续训
直接拒绝；无法确定历史 resolved device 的 checkpoint 也不进入正式 resume。

## 7. 汇总契约

完整和不完整 summary 都必须验证：

- evaluation schema、candidate subset/index policy、query/k 集合；
- variant、checkpoint SHA256，以及矩阵内 checkpoint 不复用；
- scoring input、batch、AMP/compute dtype、score distribution 和 protocol SHA256；
- score/ranking 阈值与 v4 完全一致，不能混入旧 v2 evaluation 或禁用 ranking gate 的报告；
- candidate-independent baseline 在 cell 间一致；
- 主比较按 query 配对，先在 query 内跨训练 seed 平均，再 bootstrap query。

## 8. 执行路径与状态

只读预检：

```bash
P=/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python
"$P" scripts/run_stage1b_validation.py --stage preflight
```

正式执行要求 clean commit、用户明确 GPU 授权，并先运行：

```bash
"$P" scripts/run_stage1b_validation.py --stage preflight --execute
```

然后由调度器显式执行每个 seed/variant 的 train→score→evaluate；命令见
`docs/runbook_zh.md`。

截至 2026-08-15，v4 已在 commit `b1c4fae49acebe7e03d87a8ae7664a997ad05358` 完成冻结执行。
9 个 cell 均有终态：`seed=42/encoded_swift` 通过 train/score/ranking 三门，其余 8 个在 train 门
失败。`run_artifacts/stage1b_v4/summary.json` 的 SHA256 为
`0f1e49ad72f71c4b2f51ad904f92f0cd02593e77d41d10caf34fe2e00a3b095c`，状态为
`incomplete_diagnostic_only`、`formal_primary_claim_allowed=false`。这是优化稳定性诊断，不是正式
模型主比较；同一协议不得补跑或调门。
