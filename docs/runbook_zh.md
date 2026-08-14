# CLIR 最短运行手册

最后更新：2026-08-15

本手册只给当前入口。研究状态和下一步选择见 `docs/handoff.md`，Stage 1B v4 的结论边界见
`docs/stage1b_v4_protocol.md`。

## 1. 环境

```bash
cd /prodcpfs/user/panzhixin/ICLR_2027
P=/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python
```

必须使用上面的解释器。先验证环境：

```bash
"$P" -m pytest -q
"$P" -m pip check
```

入口参数以 `--help` 为准：

```bash
"$P" train_clir.py --help
"$P" score_clir.py --help
"$P" evaluate_clir.py --help
"$P" summarize_clir.py --help
"$P" scripts/run_stage1b_validation.py --help
```

## 2. Toy 工程闭环

Toy 数据是随机 feature，只能验证加载、训练和打分，不可用于论文结论。

```bash
TOY_DIR=run_artifacts/toy_manual

"$P" examples/create_toy_clir_data.py \
  --output_jsonl "$TOY_DIR/toy.jsonl" \
  --feature_dir "$TOY_DIR/features" \
  --hidden_dim 8

"$P" train_clir.py \
  --train_jsonl "$TOY_DIR/toy.jsonl" \
  --output_model "$TOY_DIR/clir.pt" \
  --metrics_jsonl "$TOY_DIR/clir.metrics.jsonl" \
  --run_json "$TOY_DIR/clir.run.json" \
  --hidden_dim 8 \
  --projection_dim 4 \
  --batch_size 4 \
  --epochs 2 \
  --lr 1e-3 \
  --device cpu

"$P" score_clir.py \
  --input_jsonl "$TOY_DIR/toy.jsonl" \
  --model "$TOY_DIR/clir.pt" \
  --output_jsonl "$TOY_DIR/clir.scores.jsonl" \
  --batch_size 4 \
  --device cpu
```

输出存在时各入口默认拒绝覆盖。重复运行请使用新的目录；不要把 toy artifact 混入正式目录。

`score_clir.py` 不做 Best-of-N 选择。正式候选池的选择、tie 统计、random/oracle baseline 和
bootstrap 统一由 `evaluate_clir.py` 完成。

## 3. Stage 1B v4：只读检查

默认 launcher 已指向 `configs/stage1b_validation_v4.json`。下面命令会核对协议组件、split、v5
manifest/hash、所有 payload 是否存在、feature 完整性报告和零机制监督覆盖，但不写 formal
preflight、不启动 GPU：

```bash
"$P" scripts/run_stage1b_validation.py --stage preflight
```

打印九个 cell 的精确训练命令：

```bash
"$P" scripts/run_stage1b_validation.py --stage train --device cuda
```

同样可打印 score/evaluate/summarize 命令：

```bash
"$P" scripts/run_stage1b_validation.py --stage score --device cuda
"$P" scripts/run_stage1b_validation.py --stage evaluate
"$P" scripts/run_stage1b_validation.py --stage summarize
```

这些命令不带 `--execute` 时只审计命令，不执行阶段。

## 4. Stage 1B v4：正式执行

正式执行前必须同时满足：

- 用户明确批准 3 variants × 3 seeds 的 GPU 预算；
- 当前代码和 v4 协议已完成审计并形成 clean commit；
- `run_artifacts/stage1b_v4` 的 9-cell 训练矩阵没有冲突 artifact；
- 未读取 `pilot_test`。

获批后先写原子 formal preflight：

```bash
"$P" scripts/run_stage1b_validation.py --stage preflight --execute
```

launcher 不允许一次隐式执行整批 cell。调度器必须明确给出 seed 和 variant。一个 cell 的标准流程：

```bash
"$P" scripts/run_stage1b_validation.py \
  --stage train --seed 42 --variant clir --device cuda --execute

"$P" scripts/run_stage1b_validation.py \
  --stage score --seed 42 --variant clir --device cuda --execute

"$P" scripts/run_stage1b_validation.py \
  --stage evaluate --seed 42 --variant clir --execute
```

对 9 个冻结 cell 分别执行。阶段产物均为 append/atomic publication 语义；已有 checkpoint、score、
健康证据或 evaluation 时默认拒绝覆盖。

所有 cell 健康并完成后：

```bash
"$P" scripts/run_stage1b_validation.py --stage summarize --execute
```

## 5. 健康门失败

v4 的健康门依次为：train BCE 相对常数先验改善至少 1%、score population std 至少 0.1、
query 内 pairwise 排序准确率至少 0.60。

任何门失败都应保留证据并退出非零；这是预期结果，不应关门重跑。继续完成其他独立 cell，最后用：

```bash
"$P" scripts/run_stage1b_validation.py \
  --stage summarize --allow-failed-cells --execute
```

汇总器只省略具有明确失败证据的 cell，并验证证据的协议 SHA256、阈值和 cell 身份。未知、未运行、
损坏或协议不匹配的 cell 仍会报错。任何不完整 summary 都是 diagnostic-only，不能支持正式主结论；
若要改变门禁或重跑失败 cell，必须发布新协议。

## 6. Epoch snapshot 诊断

final checkpoint 是预注册主结果。若 final 健康失败，可读取某个 epoch 自己在 metrics 中的
`training_health`，而不是 final run status：

```bash
"$P" scripts/run_stage1b_validation.py \
  --stage score --seed 42 --variant clir --checkpoint-epoch 2 --device cuda --execute

"$P" scripts/run_stage1b_validation.py \
  --stage evaluate --seed 42 --variant clir --checkpoint-epoch 2 --execute
```

只有该 epoch 的训练健康证据通过时才允许打分。完整 snapshot 矩阵可用
`--stage summarize --checkpoint-epoch 2 --execute` 汇总；snapshot 不能覆盖 final summary，且
snapshot 不支持 `--allow-failed-cells`。

## 7. Resume

`--resume_from` 只接受完整 full-state checkpoint，并严格校验模型、数据 hash、协议、训练参数和
resolved device。续训必须复用原 metrics/run 路径，`--epochs` 表示新的总目标 epoch：

```bash
"$P" train_clir.py \
  <与原运行完全相同的冻结参数> \
  --epochs 5 \
  --resume_from /path/to/cell.pt \
  --output_model /path/to/cell.pt \
  --metrics_jsonl /path/to/cell.metrics.jsonl \
  --run_json /path/to/cell.run.json
```

run/checkpoint 会记录 `resumed_from`、`start_epoch` 和 `training_segments`。CPU↔CUDA 或其他
resolved device 变化会被拒绝；不要通过手工编辑 checkpoint 绕过。

`--force` 不是通用覆盖开关，只能重启 `status=failed, completed_epoch=0` 且没有 checkpoint/
metrics 的旧记录。真实部分训练或完成产物必须 resume 或使用新目录。

## 8. 真实机制监督

Stage 1B v4 不含机制监督。外部 annotation 必须先绑定样本身份再审计覆盖：

```bash
"$P" scripts/merge_clir_supervision.py --help
"$P" scripts/audit_clir_supervision.py --help
```

必须遵循 `docs/clir_supervision_protocol.md`：绑定 `id/query_id/output_token_ids SHA256`，检查 token
长度、onset/path 一致性、prior 值域和 reconstruction 维度；缺失项保持缺失。通过小样本人工盲审、
覆盖门和质量门之后，发布新的 mechanism protocol，不能原地修改 v4。

## 9. 历史 artifact

- `configs/stage1b_validation_v1/v2/v3.json` 和对应文档是冻结历史。
- Stage 1B v1 scored 文件缺少当前 `clir-reward-scoring-v2` provenance，当前 evaluator 会正确拒绝；
  如确需诊断，需要重新打分，不能放宽验证。
- `run_artifacts/` 是实验证据目录，不因“仓库瘦身”删除。
- `docs/code_review_panzhixin_*_change.md` 是审查证据，不是运行说明；当前行为以代码、测试和 v4
  协议为准。
