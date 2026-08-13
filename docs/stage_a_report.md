# CLIR Stage A 验收报告

日期：2026-08-13

证据等级：`pipeline pilot`

协议：`phi35-gsm8k-pilot-v1`

Stage A 的六个工程退出条件均已完成。本报告只证明真实数据、恢复、训练和评估接口可用，
不证明 CLIR 改善 Best-of-N。

## 1. 冻结 query split

- manifest：`configs/splits/gsm8k_phi35_v1.json`
- manifest SHA256：`cb7f23e3da36c253d3fe7d3a33675db2e43eafb05685bf55d65511e35f553186`
- train-primary / validation / reserve：6000 / 500 / 973
- development-32：train-primary 的前 32 个冻结 query，仅用于工程验收
- pilot-test / final-test：500 / 1319
- split 单位：原始 GSM8K query；每项保存 source split/index 和 question SHA256

## 2. Query 原子生成与抽取

每个 query 目录分别包含 rollout/extraction rows、trajectory tensors、唯一共享 condition
tensor，以及最后原子发布的成功 marker。marker 保存所有 payload 的相对路径、bytes 和
SHA256。严格合并器验证：

- query/protocol/split hash；
- 候选数和连续 `candidate_index`；
- 每个 payload 的 size/SHA256；
- 冻结 membership 顺序；
- 全层 metadata 的一致性。

全完成恢复测试中，8 个 shard 均报告 `all_complete`、每 shard 跳过 4 query，所有 marker
mtime 不变。模拟故障时移走一个 marker；生成和抽取均精确报告 `skipped_queries=3`、
`queries=1`、`rows=8`，mtime diff 只有该 query。抽取恢复后的合并 SHA 与恢复前相同。

该测试实际发现并修复了一个恢复循环缩进到 `except ImportError` 后成为不可达代码的 bug。

## 3. Development-32 真实数据报告

| 项目 | 数值 |
|---|---:|
| queries | 32 |
| candidates/query | 8 |
| rows | 256 |
| correct / incorrect | 237 / 19 |
| mixed / all-correct / all-wrong query | 7 / 25 / 0 |
| prompt token min / mean / max | 70 / 102.59375 / 141 |
| output token min / mean / max | 135 / 271.1484375 / 492 |
| layers / per-layer dim / total dim | 33 / 3072 / 101376 |
| storage dtype | bfloat16 |
| feature bytes | 14,739,981,408 |

- rollout manifest SHA256：`1eed7a0680d369d5183ed3ddec233e21ca19742578ea9239e22b2e62ab30332d`
- extraction manifest SHA256：`a6de1dad416b3bd4813b2df3f0bf6a06cd70b343a3acee262763eef827f08ad2`

vLLM 0.5.3.post1 的 `candidate.text` 比精确 output IDs decode 多一个前导空格，因此摘要中
`decode_mismatches=256`。正文与尾部相同；协议规定 IDs 及其独立 decode 的 `response` 是
事实来源，抽取也再次验证该 decode。

## 4. 训练状态与数据路径

`train_clir.py` 现在接受显式 `--val_jsonl` 并拒绝 train/validation query 交集。full-state
checkpoint 包含：

- model 与 optimizer state；
- completed epoch；
- Python、NumPy、Torch 和 CUDA RNG；
- model config、training args、Git 状态；
- train/validation 文件绝对路径、SHA256、row/query counts；
- 每 epoch 各 loss、适用样本/token/pair counts。

单测验证 1 epoch checkpoint 恢复至 2 epochs 与不间断 2 epochs 的全部模型参数逐 tensor
相等。

真实运行中还发现并修复两处性能问题：加载器不再把 BF16 全层 tensor 强制扩展成 FP32；
只有已通过 marker/checksum/reload/finite 验收的 manifest 才显式跳过每 epoch 重复 finite
全量扫描。默认安全扫描仍开启，训练仍拒绝 non-finite total loss。

## 5. 三模型工程闭环

从 development-32 冻结顺序派生 24-query train / 8-query engineering validation；两者零
query 重叠，不是正式 validation。三种变体均使用 101376 维真实输入，完成 1 epoch：

| variant | 参数量 | train loss | validation loss |
|---|---:|---:|---:|
| strict_swift | 202,754 | 0.44747 | 0.12368 |
| encoded_swift | 3,435,266 | 0.38268 | 0.09252 |
| clir | 9,547,273 | 0.41894 | 0.14159 |

CLIR 的 key/complete/distill/gate/reconstruction loss 和适用计数均为 0，符合当前没有外部
prior/reconstruction target 的数据契约。

## 6. 独立 evaluator

`evaluate_clir.py` 对每个冻结 query 的前 k 个候选分别报告：reward-selected accuracy、
random expected/seeded accuracy、oracle、pool coverage、逐 query 选择记录和 query-level
bootstrap CI。

8-query engineering validation 中 7 题的 8 个候选全对，1 题为 7/8 对。三模型的
reward BoN@1/2/4/8 都为 1.0；random expected@8 为 0.984375，oracle@8 为 1.0。这个候选
池过易且样本极小，因此这些数值只验收选择逻辑，不能用于方法比较。

## 7. 下一阶段

下一门是 Stage 1 correctness-only baseline：在冻结 train-primary/validation 上构造更大且
足够困难的共享候选池，以多个训练 seed 比较 strict SWIFT、encoded SWIFT、CLIR；锁定配置
后再在 pilot-test 报告 BoN@1/2/4/8/16、random、oracle、checker parity 和 query bootstrap
CI。在这之前不应把 rewrite、hallucination onset 或 dual-prior 标签混入主比较。
