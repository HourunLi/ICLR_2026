# CLIR 代码与实验独立审查报告（第三轮，独立完整版）

- **审查对象**：`/prodcpfs/user/panzhixin/ICLR_2027`，分支 `panzhixin`
- **代码状态**：`77af559`（git HEAD 为 `c9bf99c`，但该提交只改 `README.md` 与 `docs/handoff.md`，代码与 `77af559` 完全一致）
- **本轮定位**：**独立状态审查**，不是"上一轮问题清单的对照表"。本报告描述"今天这份代码库处于什么状态"，包括新提交是否真的修好、以及在当前 HEAD 上任何值得关注的风险（不论何时引入）。
- **审查目标**（沿用你的要求）：**能跑起来实验和训练，代码逻辑正确**。
  - 在范围内：崩溃、卡死、OOM、数字悄悄算错、标签错、指标错、断点续训坏掉、数据损坏、规模化阻塞、"看起来修好其实没修好"、实验有效性。
  - 不在范围内：命名、风格、注释、格式、优雅性。
- **方法**：9 个维度并行审查 → 每个维度的主要发现交由独立的"反驳型"验证 agent 逐条复现（共 36 次对抗验证，22 条被驳回或降级）→ 我本人独立复算关键结论。**所有结论都有可复现的命令与数字**；无法复现的一律不写。
- 环境：`/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python`（torch 2.3.1+cu121, py3.11.15），8×L20Z 80GB。测试：**77 passed**。

---

## 0. 总结论

**代码质量这一轮有实质性提升，三个 P0 修复中有两个完全正确。但当前状态下你有一个立刻的阻塞，以及一个比任何代码 bug 都更重要的问题：Stage 1B 实验的结论不成立。**

分三句话说：

1. **能跑**。训练/打分/评估/汇总全链路在 HEAD 上都能跑通，三个 variant 在真实 101376 维 bf16 特征上都正常收敛，CUDA 断点续训已做到位精确。
2. **有一个立刻的阻塞**：Stage 1B v2 的 9 个训练任务里有 8 个**现在就无法重新启动**，因为上次 Ctrl-C 留下了 8 个 `*.run.json`，而 `train_clir.py` 没有 `--force`。
3. **最重要的问题不是代码 bug**：已完成的 Stage 1B 实验里，**CLIR 的 3 个 seed 有 2 个训练塌缩到类别先验**（排序能力 = 随机），而且**CLIR 专属监督信号在数据里一条都没有**。所以那个"CLIR 比基线差 2.8pp"的结论，测的既不是 CLIR 机制也不是架构，而是优化失败。**照现在的 v2 协议重跑 9 次，会得到同样的塌缩。**

### 优先级总表

| 级别 | # | 问题 | 位置 | 状态 |
|---|---|---|---|---|
| **P0** | 1 | Stage 1B v2 矩阵被 8 个残留 run.json 卡死，无 `--force` | `train_clir.py:611` | 现在就挡着你 |
| **P0** | 2 | CLIR 2/3 seed 训练塌缩到类别先验，实验结论无效 | 协议/优化，非单点代码 | 需在重跑前处理 |
| **P0** | 3 | 数据里没有任何 CLIR 专属监督，实验没在测 CLIR | 数据契约 | 需重新设计实验 |
| **P1** | 4 | bf16 打分让 BoN 选择依赖 batch 形状，已实测出真实翻转 | `src/consistency_localized_reward.py:276` | 指标有 0.2–1.0pp 噪声底 |
| **P1** | 5 | checker 取"答案区间最后一个数"，冻结池里还有 15 行标签错 | `src/clir_real_data.py:289` | 新冻结的 v2 池已含错 |
| **P1** | 6 | 打分 provenance 只写不读，跨阶段 checkpoint 混用查不出来 | `evaluate_clir.py:80` | 静默错数字 |
| **P1** | 7 | `RESUME_PINNED_ARGS` 让全部 18 个已完成 checkpoint 永久无法续训 | `train_clir.py:478` | 新修复引入的回归 |
| **P1** | 8 | v2 协议没有任何可执行命令序列，7 个 CLI 默认值偏离冻结协议 | `train_clir.py:58` | 会浪费整个矩阵 |
| **P1** | 9 | README/handoff 的"最小验证闭环"直接崩 | `README.md:234` | 新人第一步就失败 |
| **P2** | 10 | MIL loss 随 T 线性增长，T=1024 时梯度占比 3686× | `src/consistency_localized_reward.py:960` | 上轮未修 |
| **P2** | 11 | `prior_distill`/`prior_gate` 量级 1e-7，实际惰性 | `src/consistency_localized_reward.py:1073` | 上轮未修 |
| **P2** | 12 | `FlatLinearFeatureEncoder` 的 `LayerNorm(101376)` 仍未加保护 | `src/consistency_localized_reward.py:145` | 修复只加在一个 encoder |
| **P2** | 13 | Stage 1B 训练用的 manifest 只存在于 `/tmp` | `run_artifacts/stage1b_v1` | 数据持久性风险 |
| **P2** | 14 | vLLM 生成对 batch 成员不变性仍未成立 | `scripts/generate_gsm8k_rollouts.py:293` | `--resume`/分片会改数据 |
| **P2** | 15 | 9 次运行 19.4 TiB 逻辑读；协议钉死 `num_workers=0` → 顺序跑 4.7 天 | `configs/stage1b_validation_v2.json:114` | 可无损去掉 |
| **P2** | 16 | Stage 1 的 9 份归档报告仍写着错误的 `candidate_subset`，且 `summarize_clir` 照收 | `summarize_clir.py:105` | 聚合路径缺 gate |

---

## 1. 当前状态速览

### 1.1 提交时间线

```
c9bf99c 08-14 15:47  Update CLIR state after Stage 1B audit   ← 仅 README.md + docs/handoff.md
77af559 08-14 15:22  Freeze audit-amended Stage 1B v2 protocol
41bc463 08-14 15:15  Harden CLIR Stage 1B reproducibility
4f415dc 08-14 13:59  Add aggregate query-paired bootstrap summary
a82f37c 08-14 11:04  Fix CLIR audit issues and freeze Stage 1B   ← 上一轮审查基线
```

三个代码提交共 1329 插入 / 152 删除，跨 29 个文件。

### 1.2 三个实验的真实状态

| 实验 | 训练 | 打分 | 评估 | 汇总 | 说明 |
|---|---|---|---|---|---|
| Stage 1 (`stage1_small_scale_v1`) | 9/9 完成 | 9/9 | 9/9 | 有 | 数据带 `candidate_index` 旧 bug；HEAD 已（正确地）拒绝重新评估 |
| **Stage 1B v1** (`stage1b_v1`) | **9/9 完成** | 9/9 | **8/9**（缺 `seed_44/clir`） | 无 | 在 `a82f37c` 训练；打分/评估在修复提交之前完成 |
| **Stage 1B v2** (`stage1b_v2`) | **0/9，8 个 failed** | — | — | — | 在 `77af559` 启动后被 Ctrl-C，**现在卡死** |

关键时间对照（这是判断归档结果新鲜度的依据）：

- Stage 1B v1 打分完成于 **14:53**，评估完成于 **14:53**
- `score_clir` 的 provenance 修复在 **15:15**，`evaluate_clir` 的 policy gate 修复在 **15:15/15:22**

所以**归档的 Stage 1B 结果是"修复前的代码"产出的**。我实测确认：

```
run_artifacts/stage1b_v1/evaluation/seed_42/clir.json
  candidate_subset = "first_k_in_generation_order"     ← 旧字符串
  candidate_index_policy 字段不存在
run_artifacts/stage1b_v1/scored/seed_42/clir.jsonl
  "reward_scoring_provenance" not in row               ← 无打分 provenance
```

好消息是底层数据是修好的那批（`generation.candidate_index_policy = vllm_completion_output_index`），所以**只要重跑 evaluate 就行，不用重新生成数据**。我实测重跑一份 18.5 秒，9 份约 3 分钟。

---

## 2. 【P0-2 / P0-3】最重要的发现：Stage 1B 的实验结论不成立

这一节是整份报告里最重要的内容。它不是代码 bug，但它直接决定"你的实验有没有意义"。

### 2.1 我先把完整结果算了出来（作者还没有）

归档评估缺 `seed_44/clir`。我用 HEAD 的代码补齐了这一格，并按冻结协议（k=1,2,4,8,16，10000 次 bootstrap）重算了全部 9 格，然后跑了 `summarize_clir.py` 得到决策规则 #3：

**Stage 1B 完整结果，k=16，aggregate query-paired bootstrap（10000 次重采样，500 query，3 seed）**

| 对比 | 均值 | 95% CI | CI 排除 0？ |
|---|---|---|---|
| random → strict_swift | **+0.0362** | [+0.0222, +0.0506] | 是 |
| random → encoded_swift | **+0.0322** | [+0.0192, +0.0453] | 是 |
| random → **clir** | **+0.0042** | **[−0.0065, +0.0147]** | **否** |
| strict_swift → encoded_swift | −0.0040 | [−0.0160, +0.0080] | 否 |
| **encoded_swift → clir** | **−0.0280** | **[−0.0420, −0.0140]** | **是（显著更差）** |

逐 seed 的 k=16 增益（相对随机）：

| variant | seed42 | seed43 | seed44 | 均值 | 样本标准差 |
|---|---|---|---|---|---|
| strict_swift | +0.0395 | +0.0315 | +0.0375 | +0.0362 | 0.0042 |
| encoded_swift | +0.0335 | +0.0375 | +0.0255 | +0.0322 | 0.0061 |
| **clir** | +0.0295 | **−0.0085** | **−0.0085** | +0.0042 | **0.0219** |

表面结论是"CLIR 显著差于两个基线，且与随机选择无法区分，seed 方差是基线的 5 倍"。**但这个结论测的不是 CLIR。**有三个独立原因。

> 顺带的交叉校验：我用完全独立重写的 BoN 计算复现了 8 格归档数字，**逐格精确一致**（例如 seed42/strict_swift 0.9300、seed42/clir 0.9200、seed43/clir 0.8820）。所以仓库的评估器本身是对的，上面的数字可信。

### 2.2 原因一：CLIR 有 2 个 seed 训练塌缩到类别先验

`seed_43/clir` 根本没学到排序函数：

- 8000 个验证分数跨度 1.5837–2.3221，**标准差仅 0.0598**，**98.3% 的分数落在最大值的 1e-3 之内**
- 诊断出的 pairwise（正确样本分数 > 错误样本分数）准确率 = **0.5023**，即**精确的随机水平**
- 正确 vs 错误的平均 margin = **+0.0001**
- train loss 从第 2 个 epoch 起单调上升：`0.3416, 0.2903, 0.2957, 0.3395, 0.3366`
- 最终 0.3366 = 训练集（89.45% 正确）**类别先验 BCE 熵 0.3369 的 0.9985 倍** → 模型收敛到了预测基准率

`seed_44/clir` 部分塌缩（std 0.574，pairwise 0.5843，884 行共享同一个分数值）。只有 `seed_42/clir` 真正训起来了（std 1.067，pairwise 0.6734）。对照组 `encoded_swift` 三个 seed 全在 0.6864–0.7074。

而冻结的 v2 协议里 `max_grad_norm=0.0`（不裁剪）、`early_stopping=false`、`checkpoint_selection=final_epoch` —— 也就是说**塌缩那次运行最差的那个 epoch 正好是被拿去打分的那个**，3-seed 的 CLIR 均值由两个失败的 seed 主导。

我自己独立看训练曲线也得到同样结论，并且发现一个相关现象：**验证 loss 与 BoN 准确率是反相关的**。`strict_swift` 的验证 loss 最差（seed42 第 5 epoch 达到 1.062，且震荡上升），但 BoN 增益最好（+0.0395）：

```
seed42/strict_swift   0.451/0.764  0.335/0.553  0.341/0.656  0.321/0.651  0.278/1.062   (train/val)
seed42/encoded_swift  0.337/0.295  0.232/0.272  0.193/0.269  0.181/0.280  0.170/0.311
seed42/clir           0.351/0.277  0.256/0.275  0.228/0.279  0.226/0.288  0.203/0.304
```

BoN 只需要 query 内部的**排序**，不需要校准；而训练/选型用的是 BCE。**协议里没有任何一步用 BoN 或排序指标做模型选择或早停**，取的是最后一个 epoch。这意味着模型选择准则和你真正关心的指标是脱钩的。

### 2.3 原因二：数据里没有任何 CLIR 专属监督

我扫了 Stage 1B 全部 4096 行真实训练数据：

```
TOKEN_LABEL_FIELDS: ['advantages','complete_prior','complete_prior_target','key_prior',
                     'key_prior_target','progress','progress_target','progress_targets',
                     'token_advantage','token_advantages']
rows scanned: 4096
CLIR-supervision-looking fields present in real data: NONE
```

对应地，9 次运行的 metrics 里，**全部 5 个 epoch、全部 CLIR 专属 loss 都精确等于 0，且 `applicable_counts` 全部为 0**：

```
NONZERO:        train.losses.final 0.203208    validation.losses.final 0.304294
EXACTLY ZERO:   prior_complete, prior_distill, prior_gate, prior_key,
                prior_reconstruction, prior_total   （train 与 validation 都是）
```

也就是说，`clir` 这一臂拿到的监督**和两个基线完全一样**（只有 outcome-level 的 correctness），却多背了 610 万个参数（9,547,273 vs 3,435,266）。一致性、定位、双先验、幻觉 MIL —— 四个机制**一个都没有被监督到**。

补充一点架构上的好消息：我追了 score 的计算路径，`key_prior_head` / `complete_prior_head` / `hallucination_head` / `complete_reconstructor` **不参与 `scores` 的计算**，只进入那些恒为 0 的 aux loss。所以它们是**惰性的，而不是用随机初始化污染分数**。分数只依赖 `token_features` + `token_reward_head` + `progress_head` + `final_score_head`，这些都能从 outcome loss 拿到梯度。

**结论：Stage 1B 实验无法支持也无法否证 CLIR 的核心假设。**它实际测的是"在只有 outcome 监督的情况下，多加一个 condition cross-attention 和 610 万惰性参数会不会更差"。答案是会更差，但这与论文想论证的东西无关。

### 2.4 这意味着什么

如果照现在冻结的 v2 协议重跑 9 次（约 10+ GPU-hour × 9），会得到同样的塌缩和同样的负数结论，然后这个结论会被归因到 CLIR 的设计上。**在重跑之前必须先处理 2.2 和 2.3。**

建议（这三条改变了对比语义，应当作为 `configs/stage1b_validation_v3.json` 声明，而不是编辑已冻结的文件）：

1. **训练健康门禁（preregistered）**：任何 run 若最终 train loss 落在类别先验 BCE 熵的 2% 以内，或验证分数 std 低于阈值，则标记/拒绝。这条现在就能拦住 seed_43。
2. **打开梯度裁剪**（`max_grad_norm > 0`），至少对 CLIR variant；并同时报告 best-epoch 与 final-epoch。
3. **要么给 CLIR 提供它需要的监督**（token/step 级标签，例如 PRM 风格的过程标注），**要么明确把这次实验降级**为"架构容量对照"，不要写成 CLIR 机制的验证。GSM8K rollout 本身不含过程标签，这是数据契约层面的缺口，不是调参能解决的。

---

## 3. P0：现在就挡住你的阻塞

### 3.1 【P0-1】Stage 1B v2 矩阵被 8 个残留 run.json 卡死

**这是你现在最该先解决的一条。**

现场状态：`run_artifacts/stage1b_v2/models/` 下有 8 个 `*.pt.run.json`，全是 `status=failed, completed_epoch=0, error_type=KeyboardInterrupt`，在 `77af559` 启动，**没有任何 `.pt` 和 `.metrics.jsonl`**：

```
seed_42/{strict_swift,encoded_swift,clir}.pt.run.json
seed_43/{strict_swift,encoded_swift,clir}.pt.run.json
seed_44/{strict_swift,encoded_swift}.pt.run.json      ← seed_44/clir 从未启动
```

`train_clir.py:610-611` 的守卫是：

```python
elif metrics_path.exists() or run_path.exists() or output.exists():
    raise FileExistsError("Training outputs already exist; use --resume_from with the full checkpoint")
```

我用最小复现证明了两条路都死：

```
ATTEMPT 1（重新启动）：
  FileExistsError: Training outputs already exist; use --resume_from with the full checkpoint
ATTEMPT 2（按报错提示用 --resume_from）：
  FileNotFoundError: [Errno 2] No such file or directory: '/tmp/mine_wedge/clir.pt'
```

`grep -c 'force\|overwrite' train_clir.py` = **0**，没有任何 override。**报错信息给出的补救办法是不可能执行的。**

更糟的是**失败是不对称的**：如果你照文档重新提交这 9 个命令，那 8 个会瞬间死掉，而 `seed_44/clir`（没有残留记录）会正常开跑 —— 你得到的是一个"1/9 的矩阵"，而不是一个干净的失败或干净的开始。

**立刻的解法**（数据已就绪，我核对过 hash 完全匹配）：

```bash
# 这 8 个文件不含任何实验信息（completed_epoch=0，无权重），删掉即可
rm run_artifacts/stage1b_v2/models/seed_4{2,3}/{strict_swift,encoded_swift,clir}.pt.run.json
rm run_artifacts/stage1b_v2/models/seed_44/{strict_swift,encoded_swift}.pt.run.json
# 或者更稳妥：写到全新目录 run_artifacts/stage1b_v2/models_run1/
```

v4 数据已经准备好且校验一致：

```
/tmp/clir_stage1b_v2/labels/train_extracted.v4.local.jsonl       sha 2ce546c1ff9b... ✓ 与 run.json 记录一致
/tmp/clir_stage1b_v2/labels/validation_extracted.v4.local.jsonl  sha 42d83ab29bcb... ✓ 与 run.json 记录一致
```

**代码层面的修复建议**：让守卫区分"有 run 记录但 `completed_epoch==0` 且无 checkpoint"与"真的有产出"，前者直接覆盖；并且在整个矩阵里只要有一个成员被挡住就拒绝启动全部，而不是启动没被挡住的子集。

---

## 4. P1：会悄悄给出错数字

### 4.1 【P1-4】bf16 打分让 BoN 选择依赖 batch 形状（已实测出真实翻转）

冻结协议要求 `scoring.amp_dtype = "bfloat16"`。在 bf16 下 cuBLAS 会依 M 维（= `batch*time`）选不同的 GEMM 算法，于是**同一个候选的分数会因为它落在哪个 batch 里而改变**。

实测（真实 stage1b_v1 seed_42 checkpoint + 真实特征，`max|score(bs) − score(bs=2)|`）：

| variant | bf16 | fp32 |
|---|---|---|
| strict_swift | bs1 1.87e-2, bs4 9.53e-3, bs8 2.36e-2, bs16 8.46e-3 | 8.2e-5 – 1.4e-4 |
| encoded_swift | bs4 1.66e-2 | 7.66e-6 |
| clir | 2.33e-3 – 2.73e-3 | ≤ 2.83e-6 |

而 top-1/top-2 的分数间隔很小：`encoded_swift` 的 p1=0.0012、p5=0.0063，**500 个 query 里有 60 个（12.0%）间隔小于 1.66e-2**，其中 5 个的 top-2 正确性不同。

**已实测出真实的 BoN 翻转**（`encoded_swift` seed_42，query `gsm8k-train-06364`，16 个候选在不同 batch size 下重新打分）：

```
picks by batch_size        {1: 2, 2: 2, 3: 2, 4: 14, 8: 2, 16: 2}
correctness of pick        {1: 0, 2: 0, 3: 0, 4: 1, 8: 0, 16: 0}   *** BoN 翻转 ***
```

这不是不确定性问题：固定 batch size 重复 5 次的散度是 **0.000e+00**，已归档的分数在 bs=2 下**逐位可复现**。

**影响**：冻结的主指标 `reward_bon_accuracy@16` 有约 **0.2–1.0pp 的噪声底**，而决策规则要检验的 `encoded_swift→clir` 差值就在同一量级（−2.8pp，CI 宽度 2.8pp）。已记录的数字是安全的（bs=2 下逐位可复现），但**距离改变只差一个配置旋钮**。

**修复**：打分改用 fp32（`--amp_dtype none`），batch 敏感度降到 ≤1.4e-4，比 p1 间隔低 2–4 个数量级。特征本来就是 bf16 存盘，输入信息不变，代价只是 GEMM。若必须保留 bf16，至少把 `batch_size` 写进指标 provenance，并加一个"固定 query 在 bs∈{1,2,4,16} 下 argmax 不变"的回归测试。

> 我自己也从另一个角度独立确认了同一机制：把 encoder 的 LayerNorm 分块阈值人为调小，强制走分块分支，输出与不分块相差 **1.7e-3**，梯度相差 1.9e-4 —— 同样是 cuBLAS 按 batch 形状换算法导致的。这也说明上一轮报告的"打分随 batch size 变化"**根因没有被修掉，只是把默认值钉住了**。

### 4.2 【P1-5】checker 取"答案区间最后一个数"，冻结池里还有 15 行标签错

`41bc463` 修好了 `\boxed{Your Answer}` 占位符那一类假阴性（20 处翻转，全部 0→1，逐条人工核对正确，**零回归**）。但同一缺陷家族在"答案句"形式的 boxed 上没修。

`_last_numeric_expression`（`src/clir_real_data.py:289`）返回**结束位置最晚**的数值表达式。对 Phi 很常见的 `\boxed{<答案> for/in/over <限定语>}` 句式，限定语里的数字会胜出：

| id | 参考答案 | boxed 内容 | 被抽取成 |
|---|---|---|---|
| `gsm8k-train-00009-cand-009` | 990 | `Tina makes $990.00 for working 10 hours every day for 5 days.` | `5` |
| `gsm8k-train-03358-cand-013` | 15 | `Bob pays $15 for the 10 nose sprays...` | `10` |
| `gsm8k-train-01359-cand-005` | 30 | `Carter can read 30 pages in 1 hour.` | `1` |
| `gsm8k-train-02816-cand-009` | 40 | `40 frogs hatch out of the 800 eggs` | `800` |
| `gsm8k-train-06908-cand-001..011` | 10 | `10:1` | `1`（7 行） |

全量扫 14,144 行，命中 30 行，其中 **15 行在新冻结的 `run_artifacts/stage1b_v2/labels/validation_extracted.v4.jsonl` 里**，全部标为 0，抽 10 条人工核对**全是假阴性**。修正这 15 行后：池组成 **151/337/12 → 146/342/12**，BoN@16 变动约 +0.2–0.4pp。

镜像方向的假阳性也可构造：`C('Final answer: \boxed{Bob pays $15 for the 10 nose sprays.}', '#### 10') == 1`。

**影响**：这次审计修正移动了 12 个 query（163/325/12 → 151/337/12），而**同一类缺陷还剩约 5 个 query，相当于刚冻结的这次修正本身的 40%**。v2 协议把 `mixed_queries=151` 冻进了配置，决策规则又是在 mixed 池上评的，所以即将开跑的 v4 实验起点池组成仍然是错的。

**修复**：非数值 boxed 路径里，若 span 含 `=` 保持"取 = 之后最后一个"（`\boxed{... = \$29}` 需要这个），否则取 span 里**第一个**数值字面量。然后 bump checker 版本、重跑 relabel、重新冻结计数，并把上面 10 条加成回归测试。

### 4.3 【P1-6】打分 provenance 只写不读

`score_clir.py:123-135` 现在会把 `checkpoint_sha256`/`batch_size`/`amp_dtype` 写进每一行。但全仓库对 `reward_scoring_provenance` 和 `reward_model_variant` 的引用**只有 `score_clir.py:180-181` 那两处写入**。`evaluate_clir.py` 会检查 `candidate_index_policy` 的一致性（`:80-88`），但**从不读 provenance**，不检查一个 pool 内所有行是否来自同一个 checkpoint，也不把 checkpoint 身份写进报告。

实测（两个阶段的文件名完全一样，都叫 `models/seed_42/clir.pt`）：

```bash
# 用 Stage 1 的 checkpoint 给 Stage 1B 的验证集打分
score_clir.py --model run_artifacts/stage1_small_scale_v1/models/seed_42/clir.pt ... → exit 0
evaluate_clir.py --input-jsonl wrongstage_clir.jsonl ...                            → exit 0, bon@16 = 1.0
# 报告里的 key：无任何 checkpoint 字段
```

把两个不同 checkpoint 打的分**交错混进同一个文件**，`evaluate_clir.py` 依然 exit 0。

另外 `--expected_checkpoint_sha256` 是可选 flag，而 `configs/stage1b_validation_v2.json:127` 写了 `checkpoint_sha256_required: true` —— 但**没有任何 Python 文件读取这个配置**，操作者也没有 9 个期望 sha256 的清单可粘。已完成的 9 个 Stage 1B 打分文件里**完全没有 `reward_scoring_provenance` 这个 key**（它们在 `a82f37c` 打的分），所以 `:129` 的 `per_row_provenance_required` 用现有产物根本无法满足。

**修复**：在 `evaluate_candidate_rows` 里照 `candidate_index_policy` 的同款做法，收集 `{checkpoint_sha256}` 与 `{reward_model_variant}`，非单值就报错，并把它们写进报告；再让 `summarize_clir._validate_reports` 断言三个 variant 的报告带三个不同的 checkpoint sha256。

### 4.4 【P1-7】`RESUME_PINNED_ARGS` 让全部 18 个已完成 checkpoint 永久无法续训

`41bc463` 新增的 `RESUME_PINNED_ARGS`（`train_clir.py:465-480`）要求 `val_every_n_epochs` 精确相等。而这个 CLI flag**是同一个提交引入的**，所以 `a82f37c` 写的 checkpoint 里 `training_args.get("val_every_n_epochs")` 是 `None`；parser 把它定为 int 且 `main()` 拒绝 ≤0（`:510-511`），**不存在任何 CLI 取值能等于 None**。

实测（用 checkpoint 里记录的原始 CLI 重跑）：

```
ValueError: Resume training arguments differ: {"val_every_n_epochs": {"checkpoint": null, "current": 1}}
--val_every_n_epochs 0  →  ValueError: val_every_n_epochs must be positive
```

逐个 checkpoint 检查缺失的 pinned key：

```
stage1b_v1/seed_42/clir.pt             → ['val_every_n_epochs']
stage1_small_scale_v1/seed_42/clir.pt  → ['num_workers','pin_memory','persistent_workers','max_grad_norm','val_every_n_epochs']
```

**影响**：9 个 Stage 1B + 9 个 Stage 1 checkpoint 全部无法 `--resume_from`。要把 5 epoch 的矩阵延长，或节点故障后重启，都只能手改 checkpoint。考虑到这 18 个 run 都已 5/5 跑完，实际损失有限（主要影响"想加 epoch 继续训"），所以我把它定为 P1 而非 P0，但它确实是**新修复引入的回归**。

**修复**：把"key 在旧 checkpoint 里不存在"与"key 值不同"区分开（`if key in checkpoint_args and checkpoint_args[key] != ...`），或加 `--allow_resume_arg_drift`。把 `val_every_n_epochs` 用历史默认值 1 回填进比较字典，即可精确解锁全部 9 个 Stage 1B checkpoint。

### 4.5 【P1-8】v2 协议没有可执行命令序列，7 个 CLI 默认值偏离冻结协议

`configs/stage1b_validation_v2.json` 是唯一的正式 Stage 1B 规格，但**没有任何文档给出对应的 `train_clir.py`/`score_clir.py` 调用**。`README.md:243-254`、`docs/runbook_zh.md:97-110`、`:269-281`、`docs/handoff.md:393-398` 里的 train 命令全是 toy 命令（`hidden_dim 8`, `batch 4`, `lr 1e-3`, `epochs 3`）。仓库里也没有任何启动脚本（`find . -name '*.sh'` 为空），9 个 run 全靠手敲。

裸默认值与冻结协议的偏离：

| flag | CLI 默认 | 协议冻结值 | 位置 |
|---|---|---|---|
| `--batch_size` | 16 | **2** | `train_clir.py:58` |
| `--lr` | 1e-3 | **1e-4** | `:73` |
| `--amp_dtype` | none | **bfloat16** | `:85` |
| `--group_by_semantic_id` | True | **false** | `:91` |
| `--prior_phase_mode` | alternate | **joint** | `:92` |
| `--val_every_n_epochs` | 1 | **5** | `:69` |
| `--bootstrap-replicates` | 2000 | **10000** | `evaluate_clir.py:28`, `summarize_clir.py:29` |

实测同一份真实数据、仅改 batch size：`--batch_size 2` → `train_total=0.8640`；`--batch_size 16`（默认）→ `train_total=1.2293`。**两者都 exit 0，都写了 checkpoint，没有任何警告。**

**影响**：照文档补全参数的人会得到一个学习率差 10 倍、batch 差 8 倍、AMP 路径不同的 run，而且从任何输出都看不出偏离了协议。9 个这样的 run 每个 10+ GPU-hour，发现时整个矩阵得作废。

**修复**：把 9 条精确命令（或一个读 `configs/stage1b_validation_v2.json` 的 launcher）写进 runbook；并给三个入口加 `--protocol_config`，断言每个冻结字段都匹配，不匹配就快速失败。

### 4.6 【P1-9】README/handoff 的"最小验证闭环"直接崩

`README.md:234-238` 用相对路径写 toy 数据（`--output_jsonl examples/toy_clir.jsonl --feature_dir examples/features`），`create_toy_clir_data.py` 把路径原样存成 `examples/features/q0_c0.pt`，而 `CLIRTrajectoryDataset` 会把它相对 JSONL 所在目录（`examples/`）解析，得到 `examples/examples/features/...`。

实测在干净目录里逐字执行 README 第 2、3 步：

```
FileNotFoundError: [Errno 2] No such file or directory: 'examples/examples/features/q2_c1.pt'
```

`docs/handoff.md:387-406` 把同一个坏序列作为"最小验证闭环"重复了一遍，`README.md:176-177` 还声称这条流程"端到端跑通"，而 `handoff.md:408` 明确告诉读者"这五步任一失败就说明环境或代码坏了"。`docs/runbook_zh.md:91` 其实已经记录了这个坑并要求用绝对路径 —— 修法存在，只是没有传播到新人最先读的两份文档。

**修复**：把 README 和 handoff 的相应命令改成绝对路径，或加 `--feature_root .`。

---

## 5. P2：工程与效率

| # | 问题 | 证据 |
|---|---|---|
| 10 | **MIL loss 随 T 线性增长**（上轮未修）。`negative_loss = -Σ logsigmoid(-logit)` 是对 token 求和，而所有兄弟 loss 都是均值。T=1024 时 MIL = 709.78，而真实总 loss 只有 0.203，在共享 encoder 上的梯度占比 **3686×**。新加的测试反而把这个未归一化行为锁定了下来。当前 Stage 1B 里 MIL 恒为 0（无监督），所以暂无实际影响；一旦启用会立刻主导训练。 | `src/consistency_localized_reward.py:960` |
| 11 | **`prior_distill`/`prior_gate` 惰性**（上轮未修）。按 1/T² 衰减，真实 T 下量级 8.3e-7 / 1.2e-7，比其他项低 5–6 个数量级。DPCL 四个目标里有两个实际上无法影响训练。 | `:1073` |
| 12 | **LayerNorm 保护只加在一个 encoder**。`LayerAxisFeatureEncoder` 已分块，但兄弟类 `FlatLinearFeatureEncoder` 有同样的 `LayerNorm(101376)` 且**未加保护**。当前 pilot 用的是 `layer_transformer`，所以没触发。 | `:145` |
| 13 | **Stage 1B 训练 manifest 只存在于 `/tmp`**。9 个 run 记录的 `train_sha256=7257737f...` 对应 `/tmp/clir_stage1b_v1/train_extracted.v3.local.jsonl`；`run_artifacts/stage1b_v1/` 下**没有** train manifest，且 `validation_extracted.jsonl` 的 sha（`7a2a4cb5`）与 run 记录的（`4362f943`）不同（前者是 canonical 路径版）。好消息：审查确认这两个文件都能从 `run_artifacts` 里的 canonical 版**逐位重建**（只差路径前缀）。但 `/tmp` 一清，记录的 hash 就无法直接验证。建议把 local 版归档进 `run_artifacts`。 | — |
| 14 | **vLLM 对 batch 成员仍不不变**。`SamplingParams(seed=42)` 确实修好了 batch **位置**不变性（同一 query 在 3-prompt batch 里换槽位，16/16 候选完全一致；跨进程、跨 GPU 逐位一致）。但**成员**变化仍改数据：从 3-query n=16 的 batch 里去掉一个 query，留下来的 query 有 13–15 个候选发生变化 —— 而这正是 `--resume` 和 `--num-shards` 会做的事。acquisition hash 却宣称 `seed_scope=LLM_and_SamplingParams`。（另有"所有 query 共享同一噪声流导致候选槽位按距贪心程度系统排序"的疑虑，验证 agent 复现后判定为低危、非指标有效性阻塞。） | `scripts/generate_gsm8k_rollouts.py:293` |
| 15 | **I/O 是唯一的规模瓶颈**。9-run 矩阵 **19.4 TiB 逻辑读**。协议钉死 `num_workers=0` 且默认每 epoch 做 finite check，实测单进程冷读 50 MiB/s → **顺序跑约 113 小时（4.7 天）**；9 路并发 + workers + `--skip_feature_finite_check` 约 **8–15 小时，能进一天**。瓶颈是单进程对 57MB `.pt` 的读延迟，不是 GPU（4 个并发作业每个都比单作业更快）、不是 CPFS 带宽（32 线程 9.3 GiB/s）、不是磁盘空间、不是校验（726 GB 约 13 分钟）。另外 `clir` variant 每 epoch 重复读 **162.0 GiB** 的 condition（unique 仅 10.1 GiB，16 倍放大）。 | `configs/stage1b_validation_v2.json:114`, `src/clir_data.py:127` |
| 16 | **Stage 1 归档报告仍带错误的 `candidate_subset`**。9 份报告仍写 `first_k_in_generation_order`（实际是 logprob 排序），而 `summarize_clir.py` 在 HEAD 上**照样能把它们聚合成 summary（exit 0，per-seed CI 与归档逐位一致）** —— policy gate 只加在 `evaluate_clir` 上，聚合路径漏了。 | `summarize_clir.py:105` |
| 17 | 其他已确认的小问题：`seed_44/clir` 有 **18.2% 的 query 存在精确的 top-score 并列**，`evaluate_clir` 静默按最小 `candidate_index` 解决且不报告；HEAD 的 `clir_gsm8k_numeric_v3` 在**相同版本字符串、相同 `label_protocol_sha256`** 下产出与 `a82f37c` 的 v3 不同的标签；两份互相矛盾的 v3 标签文件仍在盘上（`af554ad9` vs `f31d8535`，8/6144 行不一致）；metrics JSONL 先写、checkpoint 后写，写 checkpoint 失败会让 resume 永久卡住；可见 GPU 数变少时 resume 抛 `IndexError`。 | 多处 |

---

## 6. 已验证确实正确 —— 不要再动，也不要再怀疑

这一节的目的是让你知道哪些地方可以放心。全部有实测支撑。

**三个 P0 修复中的两个完全正确：**

- **CUDA 断点续训真的修好了，而且是位精确的**。跑 3-epoch 参考 run，对同配置 run 在 epoch 2 中途 SIGKILL，再在 CUDA 上 resume：**0/69 个参数不一致，0/165 个 optimizer state 条目不一致，metrics 逐位相同**。在 `--val_every_n_epochs 5 --epochs 5` 下 kill-at-2 再 resume 同样位精确；开 `encoder_dropout=0.1`（走 CUDA RNG）也位精确。AdamW 的 `exp_avg`/`exp_avg_sq` 在 `map_location="cpu"` 加载后正确落在 `cuda:0`。
- **LayerNorm 分块保护正确且必要**。我独立复现了原 bug：**恰好在 42367 行（4,294,996,992 元素 > 2³²）开始静默损坏，最大偏差 4.06**，而 42366 行干净；分块本身**逐位无误差**（LayerNorm 按行独立）。`rows_per_chunk = (2³¹−1) // 101376 = 21183`，即 2.147e9 元素，安全落在 2³¹ 以下、距真实的 2³² 悬崖有 2 倍余量。batch_size 2 下无吞吐损失。我还验证了分块路径**不丢梯度**：21/21 行、693/693 个 (row, layer) 对全部拿到梯度。
- **`_log1mexp` 的 double-where 修复在原先出问题的区间是有效的**。上一轮 `T=162, logit=-30, label=0` 是 loss 0.0000 + grad_norm=nan；现在 grad_norm = **1.191e-12（有限）**。我在 fp32/bf16、CPU/CUDA 上扫 `log_x ∈ [-50, -1e-14]`，NaN 全部消失。
  - 残留（低危）：`log_x == 0.0` 恰好时（需要全部 token logit ≤ −103，fp32 下 `logsigmoid` 下溢到 0）梯度为 inf；`label=0` 行会报 loss 0.0 + NaN 梯度。现在被新增的 per-parameter 梯度守卫抓住（崩溃而非静默污染），且我无法证明 logit 能到 −103。

**模型与数据层面（多轮交叉验证，勿重复排查）：**

- 全部 **18 个 checkpoint**（2 实验 × 3 seed × 3 variant）在 HEAD 上加载**零 config 漂移**，并**逐位复现**记录的 `reward_score`（`max|repro − recorded| = 0.000e+00`）。
- 14 个 CLIR loss 项在真实宽度、T=1024、bf16 autocast 下全部有限；目标缺失时精确为 0 且无梯度，存在时有非零梯度；无除零；无 in-place 破坏 backward；混合 condition 存在性与全 padding 行都不产生 NaN；GPU forward 对固定 batch **逐位确定**。
- 训练可复现性：同 seed 两次运行 metrics **与权重都逐位一致**，seed 43 不同。三个 variant 在真实特征上的性能：`strict_swift` 0.114 s/step、`encoded_swift` 0.138、`clir` 0.198（batch 2, bf16, workers 2），峰值 torch alloc 0.50/2.38/2.68 GiB。
- 上一轮 P1-7 **确实修好了**：12 个 pinned 参数（lr、weight_decay、seed、batch_size、prior_phase_mode、group_by_semantic_id、num_workers、max_grad_norm、val_every_n_epochs、amp_dtype、skip_feature_finite_check、pin/persistent_workers）不匹配时全部拒绝，且被拒的 resume 不会碰 run.json/metrics。
- 上一轮 P1-9 对 consistency 与 localization 两族**已修**：通过 `clir_collate` 重建 batch，报告的 count 与真实 reduction 分母精确相等（含 ragged padding）。
- Stage 1B v1 **内部一致、provenance 严密**：9 个 run 全 `status=completed, completed_epoch=5/5, code=a82f37c, dirty=false`，`train_sha256` 与 `val_sha256` **9 份完全相同**；9 份 metrics 各 5 条 epoch 记录、零非有限值；train/validation **query 完全不相交**（512 vs 500，交集 0），与 `configs/splits/gsm8k_phi35_v2.json` 的 membership 精确一致；8500 个验证特征 payload 全部存在、零孤儿。三个 seed 的权重确实不同（`max|w43−w44| = 0.20`），"seed43 与 seed44 的 BoN 数字相同"只是 1/500 的指标粒度巧合。
- **checker v4 的 `\boxed{Your Answer}` 修复是正确的**：24 行触及新代码路径，20 处标签翻转，**全部 0→1，逐条人工核对为真修复，全库零 1→0 回归**（v2→v4 共 123 处翻转，全部 0→1）。percent×100 的"仅凭词"假阳性在 v4 **已修**。
- **与 SWIFT 官方 checker 的 parity**（14,144 行，pinned commit `41f7c9f`）：(1,1) 11374、(0,0) 1558、(1,0) 1209、(0,1) **仅 3**；3 个反向个案全部人工核对 —— 2 个是 SWIFT 自身 `math_equal` 的 100× bug，1 个是 "cents" 换算差异，**该类别下 v4 零假阳性**。
- 决策规则 #3 的实现**正确**：独立重写的 aggregate paired bootstrap 在两个实验的全部 5 个对比上与仓库结果**精确一致**（均值差 <1e-15，CI 边界相同）。实现的是 `configs/stage1b_validation_v2.json:141` 与 `docs/stage1b_v2_protocol.md §4` 冻结的那个定义（每个 query 内先跨 training seed 取均值，再对 query 做 bootstrap），两臂天然配对，seed 推导稳定，replicates 已是冻结的 10000。
- acquisition 侧的三项修复**真的生效**：四个脚本现在默认 `configs/phi35_gsm8k_pilot_v3.json`，漏掉 `--protocol-config` 不再静默产出污染数据而是报 `Acquisition protocol hash mismatch`；`candidate_count_for_membership()` 让 `--membership validation` 正确取 16（旧值 8 会大声失败）；`SamplingParams(seed=42)` 修好了 batch 位置不变性。
- `scripts/verify_feature_mirror.py` **功能正确**（我在副本里改一个字节，它 exit 1 抓住；截断与缺失 payload 也能抓），归档的 gate 报告**逐位复现**（13,108 payload / 725,761,877,084 字节）。局限：纯手工，未接入 `train_clir`/`score_clir`，全量 725.8 GB 约 75 分钟。
- 上一轮 "791 GiB shard 不可达且无 override" 这条**被验证 agent 驳回**：override 存在且可用（`--protocol-config configs/phi35_gsm8k_pilot_v2.json`，已 pin 在 `configs/stage1b_validation_v2.json:7-11` 与 `docs/stage1b_v2_protocol.md §2`），冻结的 merged manifest 带逐行绝对路径 + SHA256，725.8 GB 的 gate 已通过。v3 默认切换只留下**文档陈旧**的残留，没有把任何东西弄坏。
- 11 个入口的 `--help` 全部干净；误配置守卫很好（`clir`+`identity`、`encoded_swift`+`identity`、`layer_transformer` 缺 `num_feature_layers` 都会快速失败并给出精确信息）；resume / no-overwrite / sha256 / ordered-prefix 各 gate 都正确触发；`--skip_feature_finite_check` 与 `--num_workers>0` 与协议路径逐位一致。
- Stage 1B v1 **在 HEAD 上仍可完整评估**（数据带 `candidate_index_policy`）；Stage 1 **不可**在 HEAD 重新评估 —— 这是**正确**的，因为它的候选顺序确实是 logprob 排序。存储无瓶颈。

---

## 7. 建议的处理顺序

**第一步（分钟级，解除阻塞）**

1. 删掉 8 个 `completed_epoch=0` 的残留 run.json，或改用全新输出目录（§3.1）。同时给 `train_clir.py` 加 `--force`，并让守卫区分"零 epoch 记录"与"真实产出"。
2. 补齐 `seed_44/clir` 的评估，并**用 HEAD 的代码重跑全部 9 份 evaluate + summarize**（约 3 分钟），让归档结果与当前代码一致（§1.2）。我已经跑过一遍，结果见 §2.1，输出在 `/tmp/mine_eval/`。

**第二步（在重跑 v2 矩阵之前，必须做）**

3. 处理 CLIR 训练塌缩：加 preregistered 训练健康门禁 + 打开梯度裁剪 + 同时报告 best-epoch（§2.2）。
4. 决定 CLIR 实验的定位：要么补上 token/step 级监督，要么把这次实验明确降级为容量对照，不要写成 CLIR 机制验证（§2.3）。
5. 修 checker 的"最后一个数"缺陷，重跑 relabel，重新冻结池计数（§4.2）。**这一步会改变冻结配置，应当发 v3 而不是编辑 v2。**
6. 打分改 fp32（`--amp_dtype none`），消除 BoN 的 batch 形状依赖（§4.1）。这一步也会改变冻结配置。
7. 把 9 条精确命令写进 runbook，或加 `--protocol_config` 断言（§4.5）。否则手敲 9 个 run 极易偏离协议。

**第三步（工程健壮性）**

8. `evaluate_clir` 增加 checkpoint 同源性 gate 并把 checkpoint 身份写进报告；`summarize_clir` 补上 policy gate（§4.3、§5-16）。
9. resume pinned-args 对缺失 key 宽容处理，解锁 18 个 checkpoint（§4.4）。
10. 修 README/handoff 的 toy 路径（§4.6）。
11. 跑矩阵时用 9 路并发 + `num_workers>0` + `--skip_feature_finite_check`（前置跑一次 `verify_feature_mirror`），把 4.7 天压到 8–15 小时（§5-15）。
12. 把 `/tmp` 里的 local manifest 归档进 `run_artifacts`（§5-13）。

---

## 8. 一句话回答三个问题

- **今天在 HEAD 上跑训练，会崩什么？** Stage 1B v2 的 8/9 个 run 会立刻 `FileExistsError`（第 9 个正常开跑，给你一个静默的 1/9 矩阵）；已完成的 18 个 checkpoint 全部无法 `--resume_from`；照 README 第一步走会 `FileNotFoundError`。其余全链路能跑通。
- **什么会悄悄给出错数字？** bf16 打分让 BoN 选择随 batch 形状翻转（已实测真实翻转，指标噪声底 0.2–1.0pp，与要检验的 2.8pp 同量级）；checker 在冻结池里还留着 15 行错标签；跨阶段 checkpoint 混用完全查不出来；Stage 1 的归档报告带着错误的 `candidate_subset` 仍能被聚合。
- **什么会浪费几天算力？** 照 CLI 默认值手敲 9 个 run（lr 差 10 倍、batch 差 8 倍），发现时整个矩阵作废；按协议的 `num_workers=0` 顺序跑要 4.7 天而并发只需 8–15 小时；**以及最贵的一项 —— 在不修训练塌缩、不补 CLIR 监督的前提下重跑 9 次，只会把同一个无效结论再生产一遍。**
