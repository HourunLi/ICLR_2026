# CLIR Stage 1B v3 第三轮审查修订协议

机器可读配置：`configs/stage1b_validation_v3.json`。正式入口：
`scripts/run_stage1b_validation.py`。

## 1. 实验定位

第三轮审查证明，历史 Stage 1B v1 不能被解释为 CLIR 机制实验：CLIR 的 seed 43 基本退化到
类别先验，seed 44 部分退化；与此同时，4096 条训练数据没有 consistency、hallucination、
progress、key prior、complete prior 或 reconstruction target，所有 CLIR 专属 loss 的适用计数
均为 0。

因此 v3 只回答一个较窄的问题：在共享 outcome-level correctness BCE 监督下，三个 backbone
的容量与优化行为如何。它可以发现训练塌缩、比较最终排序、验证工程链路，但不能支持或否证
consistency localization、hallucination localization 或 dual-prior localization。任何报告都必须
保留这条 claim boundary。

Stage 1B v1 是 pre-audit diagnostic；v2 在 0 个完成 epoch 时中止并被本协议取代。二者都不参与
v3 正式结论。`pilot_test` 继续保持未读取状态。

## 2. 标签与特征

correctness checker 固定为 `clir_gsm8k_numeric_v5`，label protocol SHA256 为
`4245dcf4d6f15583d03c1e8088481b597df41a4fe39daf1c2e098e884c3ad8ef`。v5 在 v4 的 placeholder
与 percent 规则上增加 governed/first-number 解析，同时把 leading mixed number 与复合小时/分钟
视为完整数值。这样既修复尾随限定词造成的 15 个 validation 漏判，也不会把 `21 1/2` 当成
`21`。

冻结 manifest：

| split | rows / queries / candidates | correct / incorrect | pool composition | SHA256 |
|---|---:|---:|---:|---|
| train | 4096 / 512 / 8 | 3668 / 428 | 111 mixed / 385 all-correct / 16 all-wrong | `6faca803b108675ed7ae08e54416969b483e7b9918133b51a0c9aff92fe02e06` |
| validation | 8000 / 500 / 16 | 7155 / 845 | 146 mixed / 342 all-correct / 12 all-wrong | `1e55ccfea2de2aa8190072e8a8c5784aec3d64059751355f6a5bde23295a6c36` |

v5 只改标签与 label provenance，不改 prompt/output token IDs、trajectory/condition 路径或 payload
SHA256。launcher 会逐行比对 v4 source 与 v5 output 的这些受保护字段。已有全量 feature gate
继续有效：12,096 行、13,108 个唯一 payload、725,761,877,084 bytes、0 failure；报告 SHA256
为 `e38b14446dca77926d28361aa7cb3877a7970aa7114b8980ca3141fdc7354d25`。

历史 v1/v2 训练实际引用的四份 local-path manifest 已从 `/tmp` 原样归档到各自
`run_artifacts/stage1b_v*/manifests/`，哈希保持不变。v3 直接使用持久化、绝对 payload 路径的
versioned manifest，不再依赖 `/tmp`。

v3 配置还冻结了 `supervision_audit`：train/validation 的 10 个辅助字段 row count 必须全为 0，
7 个 CLIR 组件 eligibility 必须全为 false。launcher 在每次 preflight 直接从 manifest 复算；
任何非零覆盖都会拒绝执行 v3，因为带真实辅助监督的数据必须发布新的 mechanism protocol，
不能悄悄改变本容量对照的实验语义。外部标签接入契约见
`docs/clir_supervision_protocol.md`。

## 3. 训练、健康门与 checkpoint

矩阵仍为 `strict_swift`、`encoded_swift`、`clir` × seeds 42/43/44。共享训练项固定为：

- 5 epochs，batch size 2，learning rate 1e-4，weight decay 0；
- BF16 training，global gradient clipping 1.0；
- 4 DataLoader workers、pinned memory、persistent workers；
- feature finite scan 在已通过 725.8 GB gate 后跳过，但 loss 与每个 parameter gradient 的 finite
  gate 仍始终开启；
- `group_by_semantic_id=false`、`prior_phase_mode=joint`、只在第 5 epoch 跑 validation；
- final epoch 是预注册主 checkpoint；每个 epoch 另存 immutable full-state snapshot，供训练轨迹
  诊断，不能事后替换主结果而不标注 post-hoc。

训练完成后必须通过类别先验门：最终 train correctness BCE 若落在常数类别先验 entropy 的相对
2% 区间内，run 记为 `health_gate_failed`，虽保留 checkpoint 供诊断，但不得打分或进入汇总。
`--force` 只允许重启 `status=failed, completed_epoch=0` 且没有 checkpoint/metrics 的记录；任何
真实产物都不会被覆盖。

## 4. 打分、评估与汇总

打分固定 batch size 2、FP32 compute（`--amp_dtype none`），不再使用会随 GEMM batch 形状产生
真实 BoN 翻转的 BF16 score。validation 全体 reward score 的 population std 必须至少为 0.1；
低于阈值时在写 scored manifest 之前失败。

每行 scoring provenance 必须包含并一致地验证：model variant、checkpoint SHA256、validation
manifest SHA256、batch size、AMP/compute dtype、experiment protocol SHA256 与 score distribution。
evaluator 会拒绝 provenance 缺失或混合 checkpoint 的文件；summary 会拒绝旧 evaluation schema、
错误 candidate subset/policy、跨报告 scoring contract 漂移，以及同一 seed 多 variant 复用同一
checkpoint。

评估固定 k=1/2/4/8/16、query bootstrap 10,000 次、95% CI。top score 精确并列时按最小
`candidate_index` 选择，并显式报告 tie query 数、比例和最大 tie size。主结果是 final-epoch
BoN@16。epoch snapshots 只能作为注明 post-hoc 的训练轨迹诊断。

## 5. 唯一允许的执行路径

先做不写文件的快速检查；它会读取 v5 manifests、核对所有 payload 是否存在、对照 relabel
source 的受保护字段，复算辅助监督零覆盖，并验证 component hashes 与已有 feature gate：

```bash
PYTHON=/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python
"$PYTHON" scripts/run_stage1b_validation.py --stage preflight
```

正式运行要求代码已形成 clean commit，并且已经明确批准本次 3×3 GPU 开销。随后写入整矩阵
原子 preflight；只要任一 cell 有冲突，9 个 cell 全部不允许启动：

```bash
"$PYTHON" scripts/run_stage1b_validation.py --stage preflight --execute
```

打印九条完全展开的训练命令用于调度审计：

```bash
"$PYTHON" scripts/run_stage1b_validation.py --stage train --device cuda
```

实际 scheduler 中，每个 cell 都必须显式给 seed 与 variant；以下以一个 cell 为例：

```bash
"$PYTHON" scripts/run_stage1b_validation.py \
  --stage train --seed 42 --variant clir --device cuda --execute

"$PYTHON" scripts/run_stage1b_validation.py \
  --stage score --seed 42 --variant clir --device cuda --execute

"$PYTHON" scripts/run_stage1b_validation.py \
  --stage evaluate --seed 42 --variant clir --execute
```

九份 final evaluation 都完成后才能汇总：

```bash
"$PYTHON" scripts/run_stage1b_validation.py --stage summarize --execute
```

若要审计某个 epoch snapshot，在 score/evaluate/summarize 三个 stage 加
`--checkpoint-epoch N`。不得给 train stage 传该参数，也不得用 snapshot 结果覆盖 final primary。

## 6. 当前状态与下一门

截至 2026-08-14，v3 配置、v5 labels、launcher 与代码健康门已准备好，快速 preflight 已通过；
formal preflight 按设计因 worktree 尚未形成 clean commit 而不会执行。尚未消耗正式 v3 GPU
训练算力，也没有任何 v3 效果数字。

外部 supervision 的身份绑定、严格合并与 coverage audit 入口现已完成，且已确认 v5 train/
validation 都是零覆盖。真正推进 CLIR 的下一项研究工作是生成并人工审核专属 supervision：
rewrite/semantic-style groups、path/onset verifier labels、token advantage/progress targets，以及
key/complete/reconstruction priors。只有这些字段真实存在、applicable counts 非零且标签质量门
通过后，才能发布“CLIR mechanism”实验协议。
