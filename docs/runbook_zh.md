# CLIR 本地运行手册

本手册讲如何在当前 DSW 实例中跑通 toy 闭环，并完成第一条 Phi-3.5-mini 真实数据
对齐 gate：

1. 激活 `SWIFT` Conda 环境；
2. 运行 smoke tests；
3. 生成 toy 数据；
4. 训练 toy 模型；
5. 打分并完成 Best-of-N 选择。
6. 保存 GSM8K rollout 的原始 token IDs，并提取全层 hidden states。

这里的 toy hidden states 是随机数，只能验证代码管线是否正常，不能用于判断模型效果。

## 1. 每次打开新终端后先执行

下面整段可以直接复制：

```bash
export CONDA_ROOT=/prodcpfs/user/panzhixin/miniconda3
export PROJECT_ROOT=/prodcpfs/user/panzhixin/ICLR_2027
export RUN_DIR="$PROJECT_ROOT/run_artifacts/toy"
export CLIR_DEVICE=cpu

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate SWIFT
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_DIR"
```

本手册仍设置 `PYTHONPATH`，这样交互式 `python -c` 和任何工作目录都能稳定导入。当前
`examples/*.py` 与 `scripts/*.py` 已加入项目根自定位，即使从别的目录直接运行也不会再因
入口路径报错；测试仍推荐在项目根使用 `python -m pytest`。旧版本曾出现：

```text
ModuleNotFoundError: No module named 'src'
```

确认当前环境和模块路径：

```bash
which python
python --version
python -c "import src, torch, numpy, pytest; print('src:', src); print('torch:', torch.__version__); print('numpy:', numpy.__version__); print('pytest:', pytest.__version__)"
```

预期 `which python` 输出：

```text
/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python
```

## 2. 跑 smoke tests

```bash
cd "$PROJECT_ROOT"
python -m pytest -q tests/test_clir_smoke.py tests/test_clir_real_data.py
```

预期结果：

```text
............... [100%]
15 passed
```

请使用 `python -m pytest`，不要直接使用 `pytest`。前者能明确使用当前 Conda 环境的 Python；配合第 1 节设置的 `PYTHONPATH`，可以稳定找到 `src`。

## 3. 生成 toy 数据

```bash
cd "$PROJECT_ROOT"
python examples/create_toy_clir_data.py \
  --output_jsonl "$RUN_DIR/toy_clir.jsonl" \
  --feature_dir "$RUN_DIR/features" \
  --hidden_dim 8
```

预期最后输出：

```text
wrote /prodcpfs/user/panzhixin/ICLR_2027/run_artifacts/toy/toy_clir.jsonl
```

检查产物：

```bash
wc -l "$RUN_DIR/toy_clir.jsonl"
find "$RUN_DIR/features" -maxdepth 1 -type f | sort
```

JSONL 应有 6 行，`features` 下应有 12 个 `.pt` 文件。

这里故意给 `--output_jsonl` 和 `--feature_dir` 传绝对路径。不要照旧手册混用 `examples/toy_clir.jsonl` 和 `examples/features` 这样的相对路径；JSONL 内的特征路径还会被数据加载器解析一次，混用相对路径容易得到错误的 `examples/examples/features/...`。

## 4. 训练 toy 模型

```bash
cd "$PROJECT_ROOT"
python train_clir.py \
  --train_jsonl "$RUN_DIR/toy_clir.jsonl" \
  --output_model "$RUN_DIR/clir_toy.pt" \
  --hidden_dim 8 \
  --projection_dim 4 \
  --batch_size 4 \
  --epochs 3 \
  --lr 1e-3 \
  --group_by_semantic_id \
  --prior_phase_mode alternate \
  --condition_attention_temperature 1.0 \
  --progress_score_weight 0.5 \
  --device "$CLIR_DEVICE"
```

预期会依次看到 3 个 epoch，最后出现：

```text
saved /prodcpfs/user/panzhixin/ICLR_2027/run_artifacts/toy/clir_toy.pt
```

确认 checkpoint 存在：

```bash
ls -lh "$RUN_DIR/clir_toy.pt"
```

loss 不需要和某次运行完全相同，但应该是有限数值，不能出现 `nan` 或 `inf`。

## 5. 打分和 Best-of-N 选择

```bash
cd "$PROJECT_ROOT"
python score_clir.py \
  --input_jsonl "$RUN_DIR/toy_clir.jsonl" \
  --model "$RUN_DIR/clir_toy.pt" \
  --output_jsonl "$RUN_DIR/clir_toy_scores.jsonl" \
  --batch_size 4 \
  --device "$CLIR_DEVICE"
```

预期最后输出：

```text
wrote /prodcpfs/user/panzhixin/ICLR_2027/run_artifacts/toy/clir_toy_scores.jsonl
```

检查输出行数及 Best-of-N 结果：

```bash
python -c "import json, pathlib; p=pathlib.Path('$RUN_DIR/clir_toy_scores.jsonl'); rows=[json.loads(line) for line in p.read_text().splitlines()]; print('rows:', len(rows)); print('scored:', sum('clir_score' in row for row in rows)); print('selected:', sum(row['clir_selected_best_of_n'] for row in rows))"
```

预期：

```text
rows: 6
scored: 6
selected: 3
```

toy 数据有 3 个 `query_id`，每个 query 有 2 条候选，因此最终应该选出 3 条。

## 6. 当前机器上的 CPU/GPU 选择

toy 训练仍默认使用 CPU，避免环境差异影响最小闭环：

```bash
export CLIR_DEVICE=cpu
```

真实 rollout/hidden-state extraction 必须先检查：

```bash
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda)"
```

截至 2026-08-13，当前实例可以看到 8 张 NVIDIA L20Z（每张约 80GB），且
`torch.cuda.is_available()` 为 `True`。只有上述检查仍然通过时再执行：

```bash
export CLIR_DEVICE=cuda
```

然后运行真实数据命令。当前环境中的 PyTorch 是 `2.3.1+cu121`，对应 CUDA 12.1
运行时；如果以后换回无驱动实例，toy 流程仍可用 CPU，但不能运行 vLLM/Phi 提取。

## 7. 常见报错

### `No module named 'src'`

重新执行：

```bash
export PROJECT_ROOT=/prodcpfs/user/panzhixin/ICLR_2027
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -c "import src; print(src)"
```

如果是跑测试，请确认使用的是：

```bash
python -m pytest -q tests/test_clir_smoke.py tests/test_clir_real_data.py
```

### 找不到 `.pt`，路径里出现 `examples/examples/features`

这是相对路径被重复拼接。回到第 3 节，使用绝对的 `RUN_DIR` 重新生成 toy JSONL 和特征文件。

### `CUDA initialization` 或 NVIDIA driver 报错

当前机器应使用：

```bash
export CLIR_DEVICE=cpu
```

并确保训练、打分命令都有：

```text
--device "$CLIR_DEVICE"
```

### `which python` 不是 `SWIFT` 环境

执行：

```bash
source /prodcpfs/user/panzhixin/miniconda3/etc/profile.d/conda.sh
conda activate SWIFT
which python
```

### 检查依赖是否损坏

```bash
python -m pip check
```

正常结果是：

```text
No broken requirements found.
```

## 8. 一次性完整命令

如果只想从头到尾跑一遍，可复制下面整段：

```bash
set -e

export CONDA_ROOT=/prodcpfs/user/panzhixin/miniconda3
export PROJECT_ROOT=/prodcpfs/user/panzhixin/ICLR_2027
export RUN_DIR="$PROJECT_ROOT/run_artifacts/toy"
export CLIR_DEVICE=cpu

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate SWIFT
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_DIR"

python -m pytest -q tests/test_clir_smoke.py

python examples/create_toy_clir_data.py \
  --output_jsonl "$RUN_DIR/toy_clir.jsonl" \
  --feature_dir "$RUN_DIR/features" \
  --hidden_dim 8

python train_clir.py \
  --train_jsonl "$RUN_DIR/toy_clir.jsonl" \
  --output_model "$RUN_DIR/clir_toy.pt" \
  --hidden_dim 8 \
  --projection_dim 4 \
  --batch_size 4 \
  --epochs 3 \
  --lr 1e-3 \
  --group_by_semantic_id \
  --prior_phase_mode alternate \
  --condition_attention_temperature 1.0 \
  --progress_score_weight 0.5 \
  --device "$CLIR_DEVICE"

python score_clir.py \
  --input_jsonl "$RUN_DIR/toy_clir.jsonl" \
  --model "$RUN_DIR/clir_toy.pt" \
  --output_jsonl "$RUN_DIR/clir_toy_scores.jsonl" \
  --batch_size 4 \
  --device "$CLIR_DEVICE"

python -c "import json, pathlib; p=pathlib.Path('$RUN_DIR/clir_toy_scores.jsonl'); rows=[json.loads(line) for line in p.read_text().splitlines()]; print('rows:', len(rows)); print('scored:', sum('clir_score' in row for row in rows)); print('selected:', sum(row['clir_selected_best_of_n'] for row in rows))"
```

本手册中的 toy 闭环已在 `SWIFT` 环境中实际验证；当前全仓库 28 个测试全部通过，成功生成 6 条
toy 数据，训练 3 个 epoch，完成 6 条打分，并为 3 个 query 各选出 1 条候选。

## 9. 第一条 Phi-3.5-mini + GSM8K 真实对齐 gate

研究口径以 `docs/pilot_protocol.md` 和
`configs/phi35_gsm8k_pilot_v1.json` 为准。下面只跑 test split 第一题，生成默认 16 个
候选并提取前两条候选的全层特征。该 gate 已于 2026-08-13 通过；下面是复跑命令，
使用新的 `_rerun` 目录以免覆盖已验收 artifact。

```bash
export CONDA_ROOT=/prodcpfs/user/panzhixin/miniconda3
export PROJECT_ROOT=/prodcpfs/user/panzhixin/ICLR_2027
export REAL_RUN_DIR="$PROJECT_ROOT/run_artifacts/phi35_gsm8k_gate_rerun"
export HF_CACHE=/prodcpfs/user/panzhixin/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate SWIFT
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$REAL_RUN_DIR" "$HF_CACHE"

python -m pip check
python -m pytest -q tests/test_clir_smoke.py tests/test_clir_real_data.py

python scripts/generate_gsm8k_rollouts.py \
  --protocol-config "$PROJECT_ROOT/configs/phi35_gsm8k_pilot_v1.json" \
  --split test \
  --max-queries 1 \
  --n-rollouts 16 \
  --tensor-parallel-size 1 \
  --output-jsonl "$REAL_RUN_DIR/rollouts.jsonl" \
  --cache-dir "$HF_CACHE"

python scripts/extract_hidden_states.py \
  --protocol-config "$PROJECT_ROOT/configs/phi35_gsm8k_pilot_v1.json" \
  --input-jsonl "$REAL_RUN_DIR/rollouts.jsonl" \
  --output-jsonl "$REAL_RUN_DIR/extracted.jsonl" \
  --feature-dir "$REAL_RUN_DIR/features" \
  --max-rows 2 \
  --cache-dir "$HF_CACHE"
```

这里用单卡是因为 3.8B 模型和单题 gate 轻松放入一张 80GB L20Z；主路径仍然提取全部
33 组状态，不是最后 4 层。当前 DSW 的 `HF_HUB_ENABLE_HF_TRANSFER=1` 曾让 Xet signed
URL 返回 403，所以命令显式关闭 transfer/Xet 加速并使用普通 HTTP。权重缓存完成后可继续
复用 `HF_CACHE`。

提取命令会生成：

- `extracted.jsonl`：可直接被 `CLIRTrajectoryDataset` 读取的真实 manifest；
- `features/trajectories/*.pt`：严格对应 `output_token_ids` 的 `[T,D]`；
- `features/conditions/*.pt`：严格对应 `prompt_token_ids` 的 `[C,D]`；
- `extracted.jsonl.extraction.json`：模型/协议 hash、层数、维度、dtype、总字节数和
  bytes-per-output-token 报告。

验收时必须确认：

```bash
python -c "from src.clir_data import CLIRTrajectoryDataset; d=CLIRTrajectoryDataset('$REAL_RUN_DIR/extracted.jsonl'); x=d[0]; print(x['hidden_states'].shape, x['condition_states'].shape, x['correctness'])"
```

该命令能成功只说明真实 token/feature 契约通过，不说明 SWIFT 或 CLIR 已经有效。

已验收目录 `run_artifacts/phi35_gsm8k_gate` 的结果是：

- 16 条 rollout 全部保存 exact prompt/output IDs；prompt 长度 113；
- 前两条 trajectory 为 `[162,101376]`、`[250,101376]`；
- 两条都引用同一个 `[113,101376]` canonical condition；
- `101376 = 33 * 3072`，落盘 bfloat16，全部 finite；
- `extracted.jsonl.extraction.json` 报告 2 条 trajectory + 1 份 condition 共
  106,450,149 bytes。

不要被生成摘要里的 `decode_mismatches` 误导：当前 vLLM 的 `candidate.text` 统一比按原始
IDs decode 的 `response` 多一个前导空格，272/272 条的正文均完全一致。协议规定原始 IDs
及其 decode 结果是事实来源。

### 在真实特征上验收 reward 架构

下面的 gate 读取同一个 manifest，依次运行 strict SWIFT、encoded SWIFT 和 CLIR。它只用
correctness BCE，不会在缺少外部 target 时偷偷启用 reconstruction：

```bash
python scripts/gate_reward_architecture.py \
  --manifest "$PROJECT_ROOT/run_artifacts/phi35_gsm8k_gate/extracted.jsonl" \
  --output_json "$PROJECT_ROOT/run_artifacts/phi35_gsm8k_gate/reward_architecture_gate.json" \
  --rows 2 \
  --device cuda:0 \
  --input_dtype bfloat16
```

2026-08-13 的已验收结果：

| variant | encoder | 参数量 | 最大参数矩阵 | peak allocated |
|---|---:|---:|---:|---:|
| `strict_swift` | identity | 202,754 | `[2,101376]` | 0.22 GB |
| `encoded_swift` | layer Transformer | 3,435,266 | `[256,3072]` | 0.98 GB |
| `clir` | layer Transformer | 9,547,273 | `[768,3073]` | 1.28 GB |

三者的 score、loss 和梯度全部 finite；layer pooling attention 对 33 层的概率和误差不超过
`1.2e-7`。这证明维度与反向链路成立，不证明训练后能改善 Best-of-N。

正式训练时三个命令必须明确写出不同的 `--model_variant`。`strict_swift` 使用默认 identity；
另外两个至少写出以下共同编码器参数：

```bash
--encoder_type layer_transformer \
--hidden_dim 101376 \
--model_dim 768 \
--num_feature_layers 33 \
--per_layer_dim 3072 \
--layer_encoder_dim 256 \
--layer_encoder_blocks 2 \
--layer_encoder_heads 8 \
--layer_pool_queries 4
```

第一阶段 manifest 没有 `complete_reconstruction_target` 是预期状态：它必须来自独立生成的
固定证据/答案摘要，形状为 `[768]`，不能拿当前候选自己的 mean/pool 特征代替。缺失时
`prior_reconstruction` 必须为 0，correctness-only baseline 仍可正常训练。

## 10. Checker 审计与 SWIFT parity

当前已生成的审计集位于
`run_artifacts/phi35_gsm8k_checker_audit/rollouts.jsonl`，包含 test index 1--16 的
16×16 条候选；加上 gate 第一题共 272 条。冻结 checker 的审计结果是 196 positive、
75 numeric negative、1 non-numeric negative。

若本地已有 checkout 到固定 commit 的官方 SWIFT 仓库，可复跑 parity：

```bash
export SWIFT_REPO=/path/to/aster2024/SWIFT
git -C "$SWIFT_REPO" checkout 41f7c9f7e13734267450870f977e5dd7d62ac23e

python scripts/audit_swift_checker_parity.py \
  --input-jsonl "$PROJECT_ROOT/run_artifacts/phi35_gsm8k_gate/rollouts.jsonl" \
  --input-jsonl "$PROJECT_ROOT/run_artifacts/phi35_gsm8k_checker_audit/rollouts.jsonl" \
  --swift-repo "$SWIFT_REPO" \
  --expected-swift-commit 41f7c9f7e13734267450870f977e5dd7d62ac23e
```

实测一致 260/272（95.59%）。12 个分歧全部是 CLIR v2 判对、SWIFT 判错，内容是
`3 bolts`、`366 downloads` 等正确数值带单位/尾随文本；没有 CLIR 判错而 SWIFT 判对的
样本。正式报告既使用所有方法共享的 v2 标签，也单独披露官方 checker parity。
