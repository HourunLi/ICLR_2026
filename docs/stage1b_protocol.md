# CLIR Stage 1B validation-strengthening 协议

版本：`clir-stage1b-validation-v1`

机器可读配置：`configs/stage1b_validation_v1.json`

本阶段只回答一个问题：在修复 candidate 顺序和 correctness 标签之后，扩大 held-out
validation 能否支持 correctness-only CLIR backbone 相对 encoded SWIFT 的稳定增量。它不引入
rewrite、onset、dual-prior target，也不查看 `pilot_test`。

## 1. Stage 1 历史结果的适用边界

旧 vLLM 0.5.3 会按 cumulative log-probability 对 `request_output.outputs` 排序。Stage 1
脚本曾用排序后的位置覆盖原始 `CompletionOutput.index`，所以历史 `candidate_index` 不是生成
顺序。由此：

- 旧数据的 BoN@1/2/4/8 不能解释为随机生成顺序的 ordered prefix；
- BoN@16 包含同一题的全部 16 个候选，候选集合没有丢失，因此 full-pool 选择仍可诊断；
- 旧数据无法事后恢复原始 index，Stage 1B validation 必须重新生成。

历史 9 个 checkpoint 已在不调参条件下，用 v3 标签对完整 16-candidate pools 做诊断。v3
修正后共有 38 个 mixed、85 个 all-correct、5 个 all-wrong query。mixed-pool 选择准确率跨
三 seed 的均值为 strict SWIFT 73.68%、encoded SWIFT 74.56%、CLIR 78.07%，但 CLIR 的
sample standard deviation 为 7.60 个百分点，且其三个 seed 在 38 道题上没有一次选择完全
一致。该结果是扩大 validation 的依据，不是稳定优越性证据。

## 2. 冻结数据

- query membership 与 v1 完全一致；v2 split 只增加组件哈希并更新协议版本。
- train 继续使用 `train_primary` 前 512 题、每题全部 8 个候选和已有全层特征，只以 v3
  checker 重标。候选内部顺序不参与 pointwise BCE，因此不会改变 train membership。
- validation 使用冻结 `validation` 的全部 500 题，每题 16 个候选，在 v2 协议下重新生成、
  重新抽取；禁止复用旧 validation 特征。
- 每个新 validation row 必须记录 `candidate_index_policy=vllm_completion_output_index`，每题
  index 必须严格为 0--15。
- rollout 完成、特征抽取开始前，validation 至少要有 120 个 mixed query、256 个正确候选和
  256 个错误候选。若 gate 不满足，只能先停下并冻结新版本，不能按标签挑题。

新 validation 全 33 层 BF16 特征预计约 483.5GB，硬停止上限为 600GiB。

## 3. 标签与 hidden-state 口径

主标签统一为 `clir_gsm8k_numeric_v3`。v3 相对历史 v2 在已有 6,144 条回答中只把 9 条
`0→1`，没有 `1→0`：百分比写法 6 条，带答案文字且尾部另有数字 3 条。与固定官方 SWIFT
commit 的全量 parity 中，没有出现“CLIR 判错、SWIFT 判对”；官方 checker 的假阴性另作
诊断披露，不能把 v3 称为官方原始标签。

Phi-3.5 仍保留 embedding output 加 32 个 block/model outputs，共 33×3072。第 33 个
`model_outputs.hidden_states` 已经经过模型自身 final RMSNorm；提取器不再额外归一化。新
metadata 分开记录“模型返回语义”和“提取器是否额外 norm”，不再使用含糊的
`apply_final_norm=false`。

## 4. 模型、训练和评估

模型、优化器和训练轮数保持 Stage 1 不变：strict SWIFT、encoded SWIFT、correctness-only
CLIR；seeds 42/43/44；5 epochs；batch size 2；AdamW；LR 1e-4；无 class reweighting、
无 early stopping，使用最终 epoch。三个模型必须共享完全相同的 train/validation rows 和
v3 标签。

validation 报告 BoN@1/2/4/8/16、random expected/seeded、oracle、全 query 与 mixed-query
结果、score margin、长度相关性、跨 seed 选择稳定性和 layer attention。CI 以 query 为单位
做 10,000 次 paired bootstrap。

只有同时满足以下三项，才能称 correctness-only CLIR 相对 encoded SWIFT 有稳定增量：

1. 三个 seed 中至少两个方向为正；
2. 跨 seed 平均增量为正；
3. query-paired aggregate bootstrap 95% CI 下界大于 0。

否则把结果记录为“尚无稳定 backbone 增量”，不查看 `pilot_test`，再决定是否进入带真实
CLIR auxiliary target 的下一阶段。
