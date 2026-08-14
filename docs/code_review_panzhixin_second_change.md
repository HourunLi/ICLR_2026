# 代码审查报告（第二轮）：`panzhixin` 分支 bf66282 → a82f37c

- 审查日期：2026-08-14
- 审查范围：上一轮审查（`docs/code_review_panzhixin_first_change.md`，基准 `bf66282`）之后的 5 个提交
  `3fb97c6` → `a1031da` → `7d1d49e` → `7a72ad4` → `a82f37c`，代码 diff 6475 行，工作树干净
- 审查基准快照：`/tmp/clir_review_v2/`
- 审查目标不变：**能否真正跑起来实验和训练，代码逻辑是否正确**。风格/命名不在范围内
- 审查方式：11 个维度并行审查 + 44 次对抗性复核（默认试图推翻每条结论）+ 1 次完备性批判，共 56 个 agent，全部结论要求可执行复现

> 审查期间发现 **Stage 1B 训练矩阵正在运行**：9 个 `train_clir.py` 任务（seeds 42/43/44 × strict/encoded/clir），
> 截至写稿时 7 个 `completed`，`seed_42/clir` 在 epoch 4/5，`seed_44/clir` 刚从 epoch 0 重启。
> 本报告的第 1 条（CUDA resume 失效）对这批正在跑的任务是**当下就有影响**的。

---

## 0. 总体结论

上一轮 16 条findings，**13 条真修好了，2 条部分修，1 条按设计保留**。修复不是表面功夫——每一条我都用真实数据复现验证过。而且这一版已经真正跑出了完整的 Stage 1 结果（343.86 GiB 特征、9 个训练运行），`docs/stage1_results.md` 里的**每一个数字我都从原始 artifact 独立重算过，全部逐字节吻合**，包括对 candidate 排序 bug 的自我审计更正。这个工程质量是可以信任的。

问题出在新增的 ~6.5k 行代码上：

| 优先级 | 数量 | 性质 |
| --- | --- | --- |
| P0 | 3 | 会让正在跑/下一步的实验直接崩或白跑 |
| P1 | 9 | 数字算得出来但是错的，或者浪费几天 |
| P2 | 8 | 效率/卫生，但在下一个 scale 上决定可行性 |

一句话：**修复是真的，但新代码引入了 3 个新的硬问题，其中 2 个属于"崩了没法恢复"和"静默算错"。**

---

## 1. 上一轮 16 条的最终判定

| 上轮编号 | 判定 | 证据 |
| --- | --- | --- |
| P0-1 candidate_index 是 logprob 排名 | **已修** | 生成器改用 `CompletionOutput.index`（`scripts/generate_gsm8k_rollouts.py:318`），加了 0..n-1 连续性校验，并持久化 `cumulative_logprob`。新数据实测干净：500×16 上 concordance(index, 更短输出)=**0.501**、concordance(index, 更高 cumlogprob)=**0.506**（旧数据 0.666/0.672）；per-index 正确率 sd=0.0074，低于 iid 期望 0.0140；oracle@k 严格单调。旧数据被 quarantine，`docs/stage1_results.md` 顶部有明确审计更正 |
| P0-2 缺 correctness 静默当负样本 | **已修** | `require_correctness=True` 硬编码在 `train_clir.py:432/439`，无法关闭。且 loss 真的尊重 mask：逐行输入梯度 = 1.215（有标签）vs **精确 0**（无标签），strict_swift 和 clir 都验过 |
| P0-3 别名绕过长度校验 | **已修** | `TOKEN_LABEL_FIELDS` 改为从 `TOKEN_LABEL_ALIASES` 推导（`src/clir_real_data.py:38`）。程序化枚举全部 10 个别名 × 长度 10 数组 vs T=162，**10/10 全部拒绝**，包括上轮报告的 `key_prior`，严格路径和 legacy 路径都拒 |
| P0-4 checker 误判 | **已修** | 26 案例表：`60%`/`60\%`/`36\text{ cm}^2`/`m^{3}` 全部修正，且**没有回归**上轮警告的 `\boxed{Profit Difference: \$125 - \$96 = \$29}`。在 **14,144 条真实 Phi-3.5 completion** 上 v2→v3 只有 16 处 label 翻转，全为 0→1，逐条人工审计全部是正确修复，**零回归** |
| P1-1 noisy-or 梯度为 0 | **已修，但引入新 bug** | label=0 的 grad_norm 从 0.0 变成 6.9。但新的 `_log1mexp` 有 NaN 梯度陷阱 → 见 P1-1（新） |
| P1-2 protocol hash 范围过宽 | **部分修** | 组件化 hash 对新数据完全正确。但两个整文件 hash 仍在把关已存储的 shard，且 `generation.*_candidates` 仍在 acquisition hash 内 → 见 P0-3（新） |
| P1-3 encoder 配置无交叉校验 | **已修** | `RewardConfig` 现在会对 `layer_transformer + 101376 + 无 layer 契约` 报错；`train_clir.py:181-188` 会用真实 manifest 报 `train layer contract is 33 x 3072, but model configuration is 32 x 3168`。残留：`train_clir.py:162 if not present: return`，manifest 缺 `feature_metadata` 时整个校验被跳过 |
| P1-4 val split 按行切 | **已修** | 按 query 分组 + `train_clir.py:449-451` 显式泄漏断言。真实数据实测 query_overlap=0（4096 行/512 query），同 seed 确定性；`configs/splits/gsm8k_phi35_v2.json` 的 train/validation/pilot_test 三者互斥 |
| P1-5 CLIR 目标全部不激活 | **按设计保留，但已诚实披露** | `applicable_counts` 是真的（造了带 target 的 batch，每个 count 从 0 变成 4/48），且 15 个真实 clir epoch 的 aux loss 与 count 严格为 0 —— 与 doc 声明一致。残留：两族 count 会**高报** → 见 P1-9 |
| P2-1 condition_cache 不清理 | **已修**（对关键路径） | shard 模式下 `_extract_rows` 每 query 调一次，所有 marker 记录 `stats.queries == 1`，缓存最多驻留一个 24MB condition。仅 legacy `--input-jsonl` 模式仍无界 |
| P2-2 无分片/续跑/原子写 | **已修** | 7 案例损坏矩阵（截断/单字节翻转/尾字节篡改/删 marker/删 payload/丢最后一行 JSONL）**全部被拒**；用真实 Phi-3.5 重新提取一条候选得到 `torch.equal(recomputed, stored) == True` |
| P2-3 DataLoader 无并行 | **已修** | 正在跑的任务实际使用 `--num_workers 2 --pin_memory --persistent_workers` |
| P2-4 condition 重复读取 | **提取端已修，训练端未修** | 见 P2-1（新）：每 epoch 重复读 243.5 GiB，唯一字节只有 20.3 GiB |
| P2-5 每次 getitem 全量 isfinite | **已修** | 可用 `--skip_feature_finite_check` 跳过（正在跑的任务已启用） |
| P2-6 只在最后 epoch 存盘 / 无 resume / 无裁剪 | **存盘与裁剪已修，resume 在 CUDA 上完全不可用** | 见 P0-1（新） |
| P2-7 无 .gitignore / 无 BoN 评测脚本 | **已修** | `.gitignore` 干净（`git ls-files run_artifacts` 返回 0 个文件）；`evaluate_clir.py` + `summarize_clir.py` 存在且能逐位复现已归档结果 |
| 附：layer 32 是 final-norm 后的状态 | **已修（文档口径）** | 新 protocol/metadata 已改正，且无代码读取这些字段，**343 GiB 特征不需要重切** |

---

## 2. P0：会让正在跑 / 下一步的实验直接崩或白跑

### P0-1（新）`--resume_from` 在 CUDA 上 100% 失效

- 位置：`train_clir.py:487` + `train_clir.py:400`（经 `:497` 到达）
- 机制：`torch.load(args.resume_from, map_location=device, weights_only=False)` 把存档里的 **CPU uint8 RNG ByteTensor 搬到了 GPU 上**，随后 `torch.set_rng_state(state["torch"])` 抛
  `TypeError: RNG state must be a torch.ByteTensor`。
- 触发条件比看起来更宽：`--device` 默认是 `auto`，`resolve_device` 在有 CUDA 时返回 `cuda`，所以**默认命令行就是坏的**。
- CPU 回退也不是出路：用 `--device cpu` 恢复一个 cuda/bf16 checkpoint 会绕过 RNG 报错，然后在 `train_clir.py:343` 撞上 `ValueError: --amp_dtype bfloat16 currently requires CUDA`。**真实 bf16 GPU 场景下不存在任何可用的 resume 路径。**
- 已用**真实的冻结 checkpoint** `run_artifacts/stage1_small_scale_v1/models/seed_42/clir.pt` 复现，6 个独立 agent 各自复现一致。CPU 路径的 resume 是位精确正确的——所以这不是设计问题，只是一个设备搬运 bug。
- 为什么没被测出来：**整个测试套件里没有任何一处出现 `cuda`**，`tests/test_clir_training_state.py` 只跑 CPU。
- 连带后果：任务被杀后不仅不能 resume，连重启都要先手动删文件——`train_clir.py:503` 的 `FileExistsError` 会拒绝，而且没有 `--force/--overwrite`。若在第一个 epoch 存盘之前被 SIGKILL，目录处于"既不能重启也不能恢复"的死结。
- 修复（两行）：

```python
def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())                     # <-- .cpu()
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])   # <-- .cpu()
```

  更稳的做法是 `torch.load(..., map_location="cpu")` 再把 model/optimizer state 搬到 device。
- **对当前正在跑的任务**：这个补丁只改加载路径，不改 checkpoint 格式，现在打上不影响在跑的进程，且能让已经存下的 checkpoint 变得可恢复。建议立刻打。

### P0-2（新）LayerNorm 在 `batch × T × 33 × 3072 > 2^32` 时静默算错

- 位置：`src/consistency_localized_reward.py:214`（`self.input_projection(self.input_norm(layer_states))`，输入被 reshape 成 `[batch*time, 33, 3072]`）
- 实测（fp32，L20Z，真实形状）：

| `batch*time` | LayerNorm 输入元素数 | 前段行最大误差 | 尾部行是否全 0 |
| ---: | ---: | ---: | :--- |
| 16,384 | 1.66e9（0.39×2^32） | 0.000000 | 否 |
| 32,768 | 3.32e9（0.77×2^32） | 0.000000 | 否 |
| 42,367 | 4.29e9（1.000×2^32） | **0.145693** | 否 |
| 65,536 | 6.64e9（1.55×2^32） | **0.234001** | **是** |

  注意越界后**不只是尾部变 0，边界之前的行也算错**。全程不抛任何异常。
- 现状安全但余量很小：正在跑的任务 `batch_size 2` → `batch*time ≤ 2048`，安全约 20 倍。但 `score_clir.py` 的**默认 `--batch_size 32`** 在 T=1024 时是 32,768，只有 1.3 倍余量；而把 batch 提到 48 以上（这正是加速 I/O bound 训练最自然的做法）会**静默产出垃圾 reward score**。
- 修复：在 `LayerAxisFeatureEncoder.forward` 里对 `batch*time` 分块跑 norm+projection，或至少加一条断言 `batch*time*num_feature_layers*per_layer_dim < 2**32`。

### P0-3（新）冻结的 Stage 1B 判据 #3 没有实现，而且 protocol hash 会锁死下一步

**(a) 判据 #3 无实现** — `configs/stage1b_validation_v1.json:130` 把
`query_paired_aggregate_bootstrap_95pct_ci_lower_bound_above_zero` 列为宣称 CLIR 增益的三个**强制**条件之一，`docs/stage1b_protocol.md` §4 重复了它。但仓库里没有任何脚本算这个量：`summarize_clir.py:181` 只输出 `paired_query_bootstrap_ci_by_seed`（每个 seed 一个区间）。

用 9 份真实 evaluation report 重算 encoded_swift→clir @k=16，三种同样合理的聚合定义给出的区间宽度差约 50%：

| 定义 | 均值 | 95% CI |
| --- | ---: | --- |
| A：先对每个 query 跨 seed 取均值再 bootstrap | +0.00781 | [-0.01042, +0.02604] |
| B：把 seed×query 混池 bootstrap | +0.00781 | [-0.01562, +0.03125] |
| C：paired-query 后取 seed 均值 | +0.00781 | [-0.01042, +0.02604] |

也就是说：**算力花完之后才临时决定 go/no-go 统计量的定义，而这个选择可能直接翻转论文的中心结论。** 另外冻结配置要求 `bootstrap_replicates=10000`，而 `evaluate_clir.py:25` 和 `summarize_clir.py:29` 默认 2000，已归档的数字不符合冻结规格。

**(b) hash 会锁死已有的 805 GiB** — 上一轮 P1-2 的组件化 hash 只解决了一半：

- `generation.formal_eval_candidates / pilot_eval_candidates / train_candidates` 仍在 acquisition hash 子树内（`src/clir_real_data.py:104`），所以**计划中的下一步 BoN@64 会拒绝全部已生成的 shard，且没有任何 override 开关**。
- 更糟：`validate_completion_marker`（`src/clir_stage_a.py:287`）按**整文件** `split_manifest_sha256` 把关。`configs/splits/gsm8k_phi35_v{1,2}.json` 的 `queries` 与 `memberships` **逐字节相同**，只有 `protocol_version` 和记录的 4 个 hash 不同，但自哈希从 `cb7f23e3` 变成 `8cb622a3`。因此在冻结的 (pilot_v2 + split v2) 组合下，**Stage 1 的 512 个 train shard（345 GiB）现在已经不可达**：

```
merge_query_shards.py --stage extraction --membership train_primary --max-queries 2
  (protocol v1, split v1) -> 成功
  (v1, v2)                -> ValueError: Acquisition protocol hash mismatch
  (v2, v1)                -> ValueError: Legacy full protocol hash mismatch
  (v2, v2)  <- 冻结组合    -> ValueError: Completion marker split_manifest_sha256 mismatch
```

  `--resume` 和 `--overwrite` 都无法绕过。**今天能跑，只是因为复用的 train pool 被当成一个 out-of-band 重标注 manifest（`train_extracted.v3.jsonl`）喂进去，完全绕开了 shard 校验机制。**
- 修复：marker 改按 `membership_sha256`（manifest 里已有，且 `validation_membership_sha256` 在 v1/v2 间不变）把关，而不是整文件 `manifest_sha256`；acquisition hash 改用显式的数值白名单，把纯文档字段（如 `apply_final_norm` 的描述性替代键）排除；作为止损先加 `--allow-provenance-drift` 并记录两个 hash。

---

## 3. P1：数字算得出来但是错的，或者浪费几天

### P1-1（新）`_log1mexp` 的 `torch.where` 双分支 NaN 梯度陷阱

- 位置：`src/consistency_localized_reward.py:954-958`
- 机制：`torch.where` 会计算**两个**分支。未被选中的 `torch.log1p(-torch.exp(log_x))` 在 `exp(log_x)` 舍入到恰好 1.0 时返回 `-inf`，其梯度是 `-inf`，再被 where 乘以 0 → **NaN**。
- 实测（fp32）：

| `log_x` | 返回值 | 梯度 | 有限？ |
| ---: | ---: | ---: | :--- |
| -1.0e-06 | -13.8155 | -1.0000e+06 | 是 |
| -6.0e-08 | -16.6289 | -1.6667e+07 | 是 |
| **-1.0e-08** | -18.4207 | **nan** | **否** |
| **0.0** | -inf | **nan** | **否** |

  即 `log_x ∈ (-6e-8, 0]` 时梯度为 NaN。折算到 MIL loss：T=162 且所有 hallucination logit 低于约 **-22.5** 时触发。
- 比原始描述更糟的一点：**label=0 的行自己也会被毒化**。实测 `T=162, logit=-30, label=0.0` → `loss=0.0000, grad_norm=nan`。而 negative 分支的最优解恰好就是把 logit 推向 -inf（`negative_loss = -log_no_hallucination`），**梯度下降会主动把模型送进这个 NaN 区间**。
- 守卫失效：`run_epoch` 只检查 `torch.isfinite(losses["total"])`（`train_clir.py:359`），而这里 **loss 是有限的（24.91）**，只有梯度是 NaN。所以守卫被绕过，一次 `optimizer.step()` 就把 19-28 个参数张量变成 NaN；`clip_grad_norm_` 也救不了（`total_norm` 本身是 nan）。
- 可达性：目前真实数据里没有 `path_hallucinated` 标签（即上轮 P1-5），所以这条**今天不会触发，Stage 2 加 onset/hallucination 标签时必然触发**。新加的回归测试只覆盖 `logits=0, label=0`，恰好避开了 NaN 区间。
- 修复（double-where，实测梯度恢复为有限值 -1e10/-1e12）：

```python
def _log1mexp(log_x):
    log_half = -0.6931471805599453
    lo = torch.where(log_x < log_half, log_x, torch.full_like(log_x, -1.0))
    hi = torch.where(log_x < log_half, torch.full_like(log_x, -1.0), log_x)
    return torch.where(log_x < log_half, torch.log1p(-torch.exp(lo)), torch.log(-torch.expm1(hi)))
```

  建议同时在 `run_epoch` 里加一次梯度有限性检查，把故障定位在根因而不是下一个 batch。

### P1-2（新）同一次重写让 MIL loss 变成随 T 线性增长

`negative_loss = -log_no_hallucination = -Σ logsigmoid(-logits)` 不再是有界的 BCE，实测 **T=224 时为 117.2**，在配置的 `mil_weight` 下占总目标的约 **85%**。同样在 Stage 2 才可达，但一旦可达就会静默主导整个目标函数。修复：按有效 token 数归一化，或对 path 概率用有界的 BCE 形式。

### P1-3（新）`score_clir.py` 的分数依赖 batch_size，且不写任何 provenance

- reward_score 随 batch 组成变化最多 **2.8e-3**（另一个 agent 在真实 101376 维特征上测到 3.1e-2），原因是 padding 到 per-batch 最大 T 改变了 kernel 形状（**不是 mask 泄漏** —— bf16 下的 padding 不变性我单独验证过是正确的）。
- 后果是实打实的：seed 42 CLIR 的 39 个 mixed pool 里有 **3 个的胜负margin 只有 1e-4 ~ 2.2e-3**；把 batch 换成 16 会**翻转 12/128 个 BoN@16 选择**，即最多 2.34pp —— **比论文报告的 +0.78pp backbone 效应更大**。
- 冻结的 scored 文件只在 `--batch_size 2 --amp_dtype bfloat16` 下可复现，而**默认是 32**。
- `score_clir.py` 不写 checkpoint 路径/sha、不写 amp_dtype、不写 batch_size、不写 device，日志只有一行 `wrote <path>`。而且两个 stage 用**完全相同的文件名**（`models/seed_42/clir.pt`），所以拿 Stage 1 的 checkpoint 去给 Stage 1B validation 打分会得到一个**貌似合理且无法察觉**的数字。
- 修复：在 scored 输出里写一个 provenance 块（checkpoint sha256、batch_size、amp_dtype、device、代码 commit），并把 `--batch_size` 默认改成与冻结配置一致的值。

### P1-4（新）`evaluate_clir.py:160` 硬编码 `candidate_subset` 声明，无任何校验

- `"candidate_subset": "first_k_in_generation_order"` 是无条件写死的字典字面量，**全仓库对该字段只有这一处引用**，没有任何 validator/assert/test 把关——而 `scripts/diagnose_stage1b.py:140` 的 `_candidate_order_status()` 已经实现了完全相同的 3 行 gate。
- 后果：9 份已归档的 Stage 1 report 全都断言了一个对那批数据**为假**的声明（其 provenance commit 是 `7d1d49e`，该版本用 `enumerate`）。实测偏差：`random_expected@1..8` 被抬高最多 **+3.1pp**，而且 `docs/stage1_results.md` §3 里"BoN 随 k 非单调"这个观察**在无偏候选顺序下消失**。
- 另外这个字符串与冻结配置自己声明的 `first_k_by_vllm_completion_output_index`（`configs/stage1b_validation_v1.json:110`）不一致。
- 修复：把 `_candidate_order_status()` 那三行搬进 `evaluate_clir.py`，从行的 `generation.candidate_index_policy` 推导该字段，policy 缺失或不符时直接报错。

### P1-5（新）磁盘上有两份互相矛盾、却都标着 `clir_gsm8k_numeric_v3` 的标签

两份 v3 标签在 6144 行里有 **8 行不一致**，而 `checker_version` 和 `label_protocol_sha256` **都无法区分它们**；`docs/stage1b_protocol.md` 引用的恰好是**不在**冻结 stage1 运行目录里的那一份。另外实测 `_validate_rollout_provenance` 会接受一个 v3 标注的行，即使 protocol 声明的是 v2。修复：把标签文件的 sha256 作为唯一内容标识强制校验（配置里已经 pin 了 `f31d8535...`，只是没有强制），并删除被取代的副本。

### P1-6（新）checker v3 的两处残留误判

**(a) 百分号 ×100 宽松规则会造成假阳性（0→1）** — `src/clir_real_data.py:369-378`。触发条件是：解析出的答案里没有 `%`、绝对值 ≤ 1、且**整个 response 里任何地方**出现 `%` 或 `percent`。正则 `(?i)(?:%|percent)` 没有词边界，所以 `percentage`/`percentile` 也会命中。用真实文本构造出的假阳性：参考答案 `12 years`、`\boxed{0.12}` → **判为正确**。真实数据里有 **576/8000** 行落在暴露区间。

**(b) 无数字的 `\boxed{Your Answer}` 占位符造成假阴性（真值被判错）** — `src/clir_real_data.py:197-199` 的 `extract_gsm8k_candidate_answer` **无条件返回最后一个** `\boxed{...}`，早于 `final answer|answer (is|=|:)` 线索；而 pilot prompt 本身就写着 `Present your final answer as \boxed{Your Answer}`，Phi-3.5 经常把占位符原样回显。实测 14,144 行中 **21 行**被误判为错，其中 **18 行在冻结的 8000 行 Stage 1B validation 里**（例：`gsm8k-train-02657-cand-014`，参考 16，response 尾部是 "16 doughnuts left in the box"）。

  这一条对**当下的决策有直接影响**：修正这 18 行会把 pool 构成从 `163 mixed / 325 all-correct / 12 all-wrong` 变成 `152/337/11`，即 **163 个 mixed query 里有 11 个是伪 mixed**，而 mixed-query BoN 是冻结协议里的 required_diagnostic。SWIFT 参考实现在其中 2 例上是对的。

  修复：`\boxed{}` 内容不含数字时回退到线索匹配 / 全文最后一个数值表达式。

### P1-7（新）resume 的守卫不完整（等 P0-1 修好之后才会显现）

`train_clir.py:486-504` 只校验 model config 和 manifest hash/行数。`batch_size`、`seed`、`prior_phase_mode`、`group_by_semantic_id`、`max_grad_norm`、`amp_dtype`、worker 相关开关**全都可以静默改变**。其中最坑的：`--lr` / `--weight_decay` 会被**静默忽略**（`optimizer.load_state_dict` 恢复了旧的 param_groups），但 `run.json` 里仍然记录为已生效。`--persistent_workers`（正在跑的任务都开着）会破坏位精确 resume。legacy `--val_fraction` 路径的 split 身份没有被 pin，换个 `--seed` 会静默换掉哪些 query 是 validation。

### P1-8（新）没有任何消费者校验记录在案的 `feature_sha256`

每行 extracted row 都带 `feature_sha256`/`condition_sha256`，shard/marker 机制会校验，但**训练和打分侧从不校验**：`CLIRTrajectoryDataset.__getitem__` 只检查 ndim、`T == len(output_token_ids)`、`P == len(prompt_token_ids)`，以及可选的 isfinite。而正在跑的 8 个任务读的是 `/tmp` 下 **793 GiB 的镜像**（`/tmp/clir_stage1b_v1` 448G + `/tmp/clir_stage1_small_scale_v1` 345G），该镜像是在 marker 机制之外拷出来的，且这些任务都传了 `--skip_feature_finite_check`。

实测：在一条真实 trajectory 中间翻转 128 字节，sha 从 `ab1c37fd` 变成 `1fc356bc`，而 dataset **照常接受**（shape 正确、0 个非有限值、max 3.39e38）。前向下去，clir/encoded_swift 因为有 LayerNorm 吸收所以 6 位有效数字不变，但 **strict_swift 的 total 变成约 1e35 的有限值**，一次 AdamW step 就污染动量估计。目前**没有实际损坏**（抽样 16/16 payload 全部吻合），所以这是一个有界的完整性缺口，不是当下的错数。修复：加 `--verify_feature_sha256` 或一个 `scripts/verify_feature_mirror.py`，每次做本地镜像后跑一次。

### P1-9（新）`applicable_counts` 有两族会高报

`train_clir.py:292-298` 的 consistency 计数和 localization 家族会高报，而 `docs/stage1_protocol.md:36` 与 `docs/stage1_results.md:45` 的"这个目标到底有没有生效"结论正是建立在这个字段上的。目前真实数据下这两族都是 0，所以已发表的数字没受影响，但这个字段的可信度需要修。

---

## 4. P2：效率与卫生（在下一个 scale 上决定可行性）

1. **I/O 是唯一瓶颈，且大部分是白费的**。按 202,752 bytes/token 精算：train 218.4 GiB 轨迹 + 81.5 GiB condition 重复读 = 299.9 GiB/epoch；validation 437.2 + 162.0 = 599.2 GiB/epoch；合计 **899 GiB/epoch/run**，validation 占 **66.6%**。9 runs × 5 epochs = **39.5 TiB**。而总 GPU 计算量不到 **2 GPU-hour**。
   - `run_epoch` 每个 epoch 都在全部 8000 行 validation 上跑一遍（`train_clir.py:543-546`），没有 `--val_every_n_epochs` / `--skip_val`；而冻结配置写的是 `checkpoint_selection="final_epoch"`、`early_stopping=false`，**5 次里有 4 次除了画曲线什么都不影响**。
   - `condition_states` 每行都从盘上读（`src/clir_data.py:125`），但 `StrictSwiftReward.forward` 和 `EncodedSwiftReward.forward` 开头就 `del` 掉它（`src/consistency_localized_reward.py:345/373`）：每 epoch 读 243.5 GiB，唯一字节只有 20.3 GiB。
   - 在下一个 scale（train_primary 6000 query，约 2.6 TiB/epoch）上，这两个旋钮就是"一天能跑完"和"跑不完"的差别。
2. **vLLM 生成对 batch 位置不不变，即使设了 `LLM(seed=42)`**。实测：把一个固定 query 从 batch slot 0 移到 slot 1，它的 4 个候选**全部改变**。加 `SamplingParams(seed=...)` 可以修好。所以 provenance 里的 `generation.seed = 42` 高估了可复现性，换分片布局不是位可复现的。
3. **所有采集脚本的默认 protocol 仍是未修复的 v1**（checker v2、无 candidate_index_policy）：`scripts/generate_gsm8k_rollouts.py:43`、`extract_hidden_states.py:44`、`merge_query_shards.py:35`、`freeze_gsm8k_splits.py:20`。谁忘了传 `--protocol-config` 就会重新生成被污染的数据。
4. **默认候选数会取错**：`extract_hidden_states.py:315` 和 `generate_gsm8k_rollouts.py:127` 按 GSM8K **源 split** 推导默认候选数，而 `validation` membership 是从 GSM8K train 里切出来的，所以两者都默认 `train_candidates=8`，而 `configs/stage1b_validation_v1.json` 要求 16。
5. `.gitignore` 忽略 `/run_artifacts/`，恰好**隐藏了那个被冻结配置 pin 了 sha256 的 manifest**。
6. `max_grad_norm` / `num_workers` 没有写进冻结的 stage1b 配置（正在跑的任务传的是 `--max_grad_norm 0`，即**裁剪关闭**）。
7. Stage 1B 没有任何文档化命令；runbook 里有一条命令在仓库自己接受的 artifact 上会失败；`docs/runbook_zh.md` 没有真实（非 toy）`train_clir.py` 的示例。补充一条实测结论：在冻结的 `batch_size 2` 下 **不要**开 `--num_workers/--pin_memory`（实测慢 20%）。
8. `diagnose_stage1b.py` 的跨 run 一致性 signature（`:342-349`）不包含 `correctness`/`checker_version`；`ordered_prefix_valid` 只是打印出来的建议值，不是断言。

---

## 5. 我验证为**确实正确**的部分（不要再返工）

这些我都实测过，可以放心：

- **Masking 是对的**。逐行 padding 不变性在全部 3 个 variant 的 **23 个前向输出**和 **18 个 loss 项**上都成立（T=3 与 T=200 混批，最大偏差 5e-7，即 fp32 舍入）；bf16 下也成立。`correctness_mask=False` 的行产生的梯度与单独跑有标签行**逐位相同**。
- **`LayerAxisFeatureEncoder` 契约正确**。33 层全部收到梯度；pooling attention 归一化到 1±1.2e-7；layer-major reshape 与 `clir_real_data.py` 的 `torch.cat` 顺序在真实特征上被证明一致（per-layer RMS 0.031/1.289/10.93/1.373，若轴搞反会是 2.75/3.04/5.28）。
- **指标代码的公式全对**。独立重写一份实现后比对：**180 个聚合值 + 23,040 个 per-query 值 + 15 个 paired CI 全部逐位吻合**。BoN@k 按 candidate_index 取前 k 且对文件行序不变（打乱文件验证过），oracle@k 是前缀 max，random_expected 是精确解析期望，random_seeded 均匀（χ²=14.24, df=15），paired bootstrap 是**按 query 重采样且两臂共用同一组索引**，± 是样本标准差（n−1），与文档一致（用总体标准差会得到 1.69/1.33/2.05 而不是 2.07/1.63/2.51）。
- **`docs/stage1_results.md` 的每个数字都从原始 artifact 重算过**：行/标签/pool 计数（4096/3658/438/117/379/16 与 2048/1772/276/39/84/5）、特征字节 369,216,532,352 = 343.86 GiB **精确到字节**、202,752 B/token、6784 个唯一 payload 源与 scratch 的 SHA256 逐字节一致（抽样 24 个全中，0 孤儿 0 缺失）、9 个运行 dirty=false/status=completed/5 条 epoch metrics/无 OOM-NaN-traceback、30 条 CLIR train+val epoch 记录的 aux loss 与 count 精确为 0、完整 BoN 表、全部 paired bootstrap CI。
- **损坏抵抗能力是真的**：7 案例损坏矩阵全部被拒；用真实 Phi-3.5 重新提取得到 `torch.equal == True`。
- **不存在 cherry-pick 路径**：query 选择是对一个 hash 校验过的冻结列表做 `query_ids[:max_queries]`（`src/clir_stage_a.py:230-236`），且 merge 要求每个预期 query 都有成功 marker。
- **checker v3 相对 v2 零回归**（14,144 条真实 completion，16 处翻转全部是正确修复）。相对 SWIFT 参考实现有 1196 处分歧，抽样 40 条审计后结论是：绝大多数是 **SWIFT 自己没能剥离单位/句子**（本仓库对），5 条反向分歧里 3 条是 SWIFT 的 bug（`math_equal("2000","20") == True`），2 条本仓库确实错。
- **旧 checkpoint 在 HEAD 上仍能加载并复现记录的分数**（打分代码与 `7d1d49e` 逐位一致），所以不存在静默的打分回归。
- **同 seed 的 GPU 运行逐字节一致**，不同 seed 有差异 → 3-seed 的 ± 是真实的 seed 效应，不是运行间噪声。
- **正在跑的 Stage 1B 数据的标签/provenance 链是干净的**：配置 pin 的 `f31d8535` 与仓库 v3 manifest sha 一致；`/tmp` 镜像 manifest 与配置 pin 的 artifact **只在 feature 路径上不同**（4096 train + 8000 val 行 **0 处 label 分歧**，id 顺序一致）；冻结 split 从 HEAD 可逐字节重现。
- **存储不是瓶颈**（约 199-208 TB 可用）。
- `scripts/materialize_development_split.py` 逻辑正确（此前无人审过）。
- 上轮的 layer-32 口径问题已修，且无代码读取那些字段，**343 GiB 不需要重切**。

---

## 6. 立刻要做的事（因为矩阵正在跑）

1. **马上打 P0-1 的两行 `.cpu()` 补丁。** 它只改加载路径，对在跑的进程无影响，但能让已经存下的 checkpoint 变成可恢复的。现在 `seed_42/clir` 在 epoch 4/5、`seed_44/clir` 在 epoch 0/5，任何一个挂掉现在都只能从 0 重跑。
2. **打分时显式写 `--batch_size 2 --amp_dtype bfloat16`。** 默认 32 既不可复现（P1-3），又距 LayerNorm 悬崖只有 1.3 倍余量（P0-2），而且在 8 个训练进程仍占着显存时实测峰值接近 76 GiB / 80 GiB。
3. **在看结果之前**先定下判据 #3 的聚合 CI 定义和 replicate 数（P0-3a）。这是唯一能避免"看到数字再选统计量"的时机。
4. **决定那 18 行冻结 validation 假阴性怎么办**（P1-6b）：要么在决策前修掉并记录 relabel drift，要么明确预注册"保持不变"。它影响 163 个 mixed query 中的 11 个，而 mixed-query BoN 是 required_diagnostic。

## 7. 之后的修复顺序

1. P0-2（LayerNorm 断言/分块）、P0-3b（marker 改按 membership hash + `--allow-provenance-drift`）——后者决定你能不能复用那 345 GiB。
2. P1-4（candidate_subset 校验）、P1-3（打分 provenance + 默认 batch）、P1-5（删掉重复标签、强制 sha 校验）。
3. P1-1 + P1-2（`_log1mexp` double-where + MIL 归一化 + 梯度有限性检查）——Stage 2 之前必须修完。
4. P1-6a（百分号规则加词边界并收紧）、P1-7（resume 守卫补齐）、P1-8（镜像校验脚本）、P1-9（applicable_counts）。
5. P2-1（`--val_every_n_epochs` + condition 按 query 缓存/按 variant 跳过）——这是下一个 scale 能否在一天内跑完的关键。
6. P2 其余项（vLLM `SamplingParams(seed=)`、默认 protocol 改 v2、默认候选数、文档与 Stage 1B 命令）。
