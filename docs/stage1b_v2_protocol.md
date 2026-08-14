# CLIR Stage 1B v2 审计修订协议

机器可读配置：`configs/stage1b_validation_v2.json`。

本修订发生在正式 Stage 1B 指标被查看之前。v1 的 9 组训练与打分已完成，但第二轮代码审查随后指出 checker、CUDA resume、LayerNorm 大输入、打分 provenance 和 ordered-prefix 声明等新问题。因此 v1 输出只保留为 pre-audit artifact，不参与正式结论；不读取 `pilot_test`。

## 1. 标签修订

checker v4 只忽略明确来自 prompt 的空 `\boxed{}`、`\boxed{Answer}` 或 `\boxed{Your Answer}`。`\boxed{x}`、`\boxed{Impossible}` 等实质性文本仍是模型最终答案，不能回退到中间算式。小数/分数与百分数的等价也必须有明确数值百分号证据，或处在 probability/odds/chance 的百分比上下文中。

对 v3 冻结数据全量重算后：

- train 4096 条中仅 2 条 `0→1`，没有 `1→0`；
- validation 8000 条中仅 16 条 `0→1`，没有 `1→0`；
- validation pool 变为 151 mixed、337 all-correct、12 all-wrong。

审查报告估计 validation 有 18 条假阴性。逐题核对表明其中只有 16 条正确；另外两条虽然带占位符，但模型的实际数值计算错误，所以继续标 0：`gsm8k-train-05191-cand-002` 把 80 个百分点乘以 2 美元算成 1.60 美元，`gsm8k-train-03921-cand-010` 给出 3.34 而参考题意所需的平均值是 4。不能为了追平外部 checker 而把错误答案改成正确。

## 2. 不重新采集 hidden states

v4 只改变 correctness 标签，不改变 prompt、原始生成 token IDs 或特征。Stage 1B v1 的 validation acquisition hash `8d4afea9…` 保持有效；train 继续复用原有 512×8 特征。所有本地 trajectory/condition 文件必须先通过记录在行内的 SHA256 全量校验，任何缺失或不一致都停止训练。

正式训练前的全量门禁已经通过：12,096 行共引用 13,108 个独立 payload，13,108 个全部匹配，失败数为 0；实际读取 725,761,877,084 bytes（约 725.8 GB / 676.0 GiB）。持久化报告为 `run_artifacts/stage1b_v2/audits/feature_mirror_verification.json`，SHA256 为 `e38b14446dca77926d28361aa7cb3877a7970aa7114b8980ca3141fdc7354d25`。

## 3. 训练和打分冻结项

仍训练 `strict_swift`、`encoded_swift`、`clir`，seeds 42/43/44，共 9 组，5 epochs，LR 1e-4，batch size 2，BF16，final epoch checkpoint。由于不做 early stopping，只在第 5 epoch 跑 validation；`strict_swift` 和 `encoded_swift` 不读取其 forward 明确丢弃的 condition payload。这两项只减少 I/O，不改变训练样本、梯度或模型定义。

打分固定 batch size 2 + BF16，并在每行写 checkpoint SHA、input manifest SHA、batch size、dtype、device 和代码 commit。ordered-prefix 评估必须验证每行的 `candidate_index_policy=vllm_completion_output_index`，不再写死一个未经验证的声明。

## 4. 统计判据不变

主指标仍为 500 个 validation query 的 BoN@16，报告 k=1/2/4/8/16，query-level 10,000 次 bootstrap。跨 seed aggregate CI 固定为：先在每个 query 内对三个训练 seed 的 paired delta 求平均，再 bootstrap queries。

只有同时满足以下三项，才称 correctness-only CLIR 相对 encoded SWIFT 有稳定增益：

1. 3 个 seed 中至少 2 个增量为正；
2. 三 seed 平均增量为正；
3. aggregate query-paired 95% CI 下界大于 0。

否则结论是“没有稳定的 correctness-only backbone 增益”，不得转去查看或调参 `pilot_test`。
