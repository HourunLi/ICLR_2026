# CLIR 第四次改动审查报告

审查对象：`/prodcpfs/user/panzhixin/ICLR_2027`，`panzhixin` 分支工作树（未提交，22 改 + 15 新）
审查日期：2026-08-14
审查范围：崩溃 / 挂死 / OOM / 数字悄悄算错 / 标签错 / 指标错 / resume 断裂 / 数据损坏 / 规模瓶颈 /
"看起来修好了其实没修好" / 实验有效性。不含命名、风格、docstring、格式、优雅性。
目标：能跑起来训练和实验，且代码逻辑正确。

---

> **⚠️ 先读 §13。**§0–§12 审查的是 `a49891d`（Stage 1B v3）。此后工作树前进了三个 commit，本报告
> 已被采纳（§2、§5.1 的建议都实现了），`DEFAULT_PROTOCOL` 已切到 v4，v4 结果已锁定。
> **§13 修正了本报告的两条结论（§6.4 漏检、§6.8 结论错误），并给出一个新的 P0：
> 门禁读的是 epoch 均值而非最终 checkpoint，误杀了 3 个本该通过的 cell。**

---

## §0 总结论

**上一轮的 3 个 P0 有 2 个真修好了，第 3 个被诚实降级而不是被伪造。**这一轮的改动质量明显高于上
一轮：`--force` 恢复路径、checker v5、fp32 打分、per-row provenance 契约、prior-collapse 门禁，
都不是"加个字段让检查通过"，而是真的改了行为，我逐条用真实数据验证过。

**但这一轮引入了一个新的 P0：健康门禁和汇总之间形成了没有出口的死锁。**一个 seed 塌缩就会永久
卡死整个 9 格矩阵，全仓库没有任何 override。更关键的是，专门为应对塌缩而新增的 per-epoch 快照，
恰好在塌缩发生时不可用 —— 协议文档明确承诺的 `--checkpoint-epoch N` 审计路径会被同一道门禁拒绝。

**还有一个 P1 直接影响实验有效性：门禁只抓住了 2 个坏 CLIR seed 中的 1 个。**漏掉的那个
（`seed_44/clir`）两道门禁全过，但它的 query 内排序准确率是 0.5735，和完全塌缩的 seed_43 一样只
贡献 −0.0085 的 BoN 增益。原因是两道门禁一个测训练 BCE、一个测分数散布，**都不测 query 内排序，
而这是 best-of-n 唯一依赖的量**。

### 优先级表

| # | 级别 | 问题 | 位置 | 后果 |
|---|------|------|------|------|
| 1 | **P0** | 门禁失败 → 汇总死锁，无任何 override | `run_stage1b_validation.py:547-556` + `summarize_clir.py` | 一个 seed 塌缩，9 格汇总永久无法产出 |
| 2 | **P0** | epoch 快照在塌缩时不可用（`run.json` 非 epoch 化） | `run_stage1b_validation.py:357` | 文档承诺的审计路径实际被拒 |
| 3 | **P1** | 门禁漏掉 `seed_44/clir`（排序已坏但两门全过） | `train_clir.py:622-658` + config | 坏 run 进入主结果，结论被稀释 |
| 4 | **P1** | 门禁用 `abs()`，比先验更差的模型也放行 | `train_clir.py:642` | 只能检测"等于先验"，检测不到"比先验更差" |
| 5 | **P1** | 健康 run 距拒绝带仅 1.8 倍，且为硬失败 | `configs/stage1b_validation_v3.json` | 好 run 被误杀，整批重跑 |
| 6 | **P1** | 归档 Stage 1B v1 已无法再评估 | `evaluate_clir.py:99` | 上轮"3 分钟重评"作废，需 2.5 小时全量重打分 |
| 7 | **P1** | resume 在 `run.json` 里不留任何痕迹 | `train_clir.py` run 记录 | 无法审计 cell 是一次跑成还是跨设备拼接 |
| 8 | P2 | `device` 未纳入 `RESUME_PINNED_ARGS` | `train_clir.py:509-526` | CPU checkpoint 可静默在 CUDA 上续训，CUDA RNG 不恢复 |
| 9 | P2 | per-epoch `training_health` 写了但从不被读 | `train_clir.py:939` vs `:670` | 已有的诊断证据被浪费 |
| 10 | P2 | 正式执行唯一阻塞：工作树脏 | `configs/...json` `formal_execution_requires_clean_git` | 今天无法启动正式 v3 |

---

## §1 上一轮 17 项的落实情况

作者在 `docs/code_review_panzhixin_third_change_resolution.md` 声称 17 项全部修复。我逐条验证：

| 上轮编号 | 声称 | 我的判定 | 证据 |
|---|---|---|---|
| P0-1 卡死矩阵无法恢复 | 已修 | **真修好** | 见 §6.1，正反两个方向都验证 |
| P0-2 checker 标签错 | 已修 | **真修好** | 验证池精确变为 146/342/12，见 §6.2 |
| P0-3 CLIR 监督缺失 | 降级为 outcome-only 对照 | **诚实降级，非伪造** | preflight 实测 eligible components = 0，降级由代码强制 |
| P1-4 bf16 打分随 batch 变 | 改 fp32 | **真修好** | 偏差 2.4e-2 → 3.6e-6，见 §6.3 |
| P1-其余 | 已修 | 大部分成立 | 见 §6.4-§6.7 |

三个 P0 的落实是可信的。**但"全部修复"这个说法本身需要打折**：修复引入了 §2-§5 的新问题，其中
§2 的严重程度不低于被它替代的 P0-1。

---

## §2 【P0】健康门禁 → 汇总死锁，全仓库无 override

### 机制

一个 cell 的训练如果被 prior-collapse 门禁判为失败，`run.json` 里会是
`status: "health_gate_failed"`。此后：

1. `scripts/run_stage1b_validation.py:547-556` 的 `_require_completed_model`：

```python
547: def _require_completed_model(paths: Mapping[str, Path]) -> Dict[str, Any]:
548:     if not paths["model"].is_file():
549:         raise FileNotFoundError(f"Checkpoint is absent: {paths['model']}")
550:     run = _load_json(paths["run"])
551:     if run.get("status") != "completed":
552:         raise RuntimeError(f"Training run did not pass health gates: {paths['run']}")
553:     health = run.get("health_gate")
554:     if not isinstance(health, Mapping) or not health.get("passed"):
555:         raise RuntimeError(f"Training health evidence is absent/failed: {paths['run']}")
```

`scoring_command` 在 `:572` 无条件调用它 → **该 cell 无法打分**。

2. 没有打分文件 → 没有评估文件。

3. `summarize_clir.py` 打开 9 个评估文件，缺一个就 `FileNotFoundError`。

### 关键点：没有任何出口

我在整个仓库里搜过覆盖开关：

- launcher 只有一个 `--force-zero-epoch`（`:62`），那是上一轮为"0 epoch 卡死记录"加的，
  **与健康门禁无关**；`:525` 追加的 `--force` 是传给 `train_clir.py` 的覆盖写标志，也无关。
- `summarize_clir.py` 里 `grep -n "force|override|allow_missing|skip"` **零命中**。
- config 里 `fail_on_prior_collapse: true`。

也就是说：**9 格里任何一格塌缩，整个 Stage 1B v3 的主结果就永久无法产出**，而唯一的"补救"是把
config 改成 `fail_on_prior_collapse: false` 重跑全部 —— 但那等于关掉门禁，也等于事后改预注册协议。

### 为什么这不是理论风险

已归档的 Stage 1B v1 里，`seed_43/clir` 就是一个真实的塌缩 run（相对距离 0.001028，被新门禁正确
拒绝）。同样的塌缩在 v3 复现的概率不低 —— CLIR 变体有 9.5M 参数、5 个 epoch、joint prior phase，
这正是上一轮观察到塌缩的配置。**门禁做对了检测，但没有为检测成功之后准备任何路径。**

### 建议

在 `summarize_clir.py` 加一个显式的、会写进产物的降级路径，例如
`--allow-failed-cells`，把失败 cell 的身份和失败原因记录进 summary 的 provenance，让主结果可以在
"8/9 格 + 明确标注"的状态下产出；同时在协议文档里预注册"塌缩 cell 如何处理"。现在的设计是"检测到
问题就整体停摆"，对一个只需要跑起来的实验是过强的。

---

## §3 【P0】epoch 快照在最需要时不可用

这一轮新增了 per-epoch 不可变快照，`docs/stage1b_v3_protocol.md:128` 明确写了审计路径：

> `--checkpoint-epoch N`。不得给 train stage 传该参数，也不得用 snapshot 结果覆盖 final primary。

意图很清楚：final epoch 出问题时，用中间 epoch 的快照做 post-hoc 诊断。**但这条路走不通。**

`_cell_paths`（`:330-361`）在传入 `checkpoint_epoch` 时，正确地把 model / scored / evaluation 都
指向 epoch 化的路径：

```python
343:     else:
344:         checkpoint = model_dir / f"{variant}_epochs" / f"epoch_{checkpoint_epoch:03d}.pt"
345:         scored = (
346:             root / "scored_epochs" / f"epoch_{checkpoint_epoch:03d}" ...
```

**但 `run` 不是：**

```python
357:         "run": model_dir / f"{variant}.run.json",
```

`run` 永远指向**最终**的 run.json，与 `checkpoint_epoch` 无关。于是
`_require_completed_model`（`:572` 调用）读的是最终 run 的 `status` 和 `health_gate.passed`。

**结果：final epoch 塌缩 → 最终 run.json 是 `health_gate_failed` → 即使 epoch 2 的快照完全健康，
`--checkpoint-epoch 2` 一样被 `:552` 拒绝。**快照专门为这个场景而存在，却在这个场景下不可用。

更讽刺的是，判断 epoch 2 是否健康所需的证据**已经写好了**：`train_clir.py:939` 每个 epoch 都往
metrics 里写 `training_health`。但 `_final_training_health`（`:661`）只读 `metric_rows[-1]`：

```python
670:     recorded = metric_rows[-1].get("training_health")
```

**per-epoch 的健康证据写了，但除了最后一行没有任何东西读它。**

### 建议

`_cell_paths` 在 `checkpoint_epoch` 非空时，让门禁改为读该 epoch 在 metrics.jsonl 里对应行的
`training_health`（数据已经在那里），而不是读最终 run.json。这是个小改动，能让已经付出的快照存储
成本（实测 2.21 GB，见 §6.6）真正产生价值。

---

## §4 【P1】门禁只抓住 2 个坏 CLIR seed 中的 1 个

这是本轮最影响实验有效性的问题。

v3 加了两道健康门禁：

1. **prior-collapse 门禁**（`train_clir.py:622-658`）：训练最终的 correctness BCE 不能落在常数类
   先验熵的 ±2% 带内。
2. **分数散布门禁**（`score_clir.py` 的 `--min_score_std`，config 设 0.1）：验证集打分的总体标准差
   不能低于 0.1。

我用已归档的 Stage 1B v1 的真实 checkpoint 逐个 cell 算了这两道门禁的判定，并同时算了一个门禁**没有**
测量的量 —— query 内 pairwise 排序准确率（即在同一个 query 的候选里，正确答案得分高于错误答案的
比例）：

| cell | 相对距离（门禁1，阈值 0.02） | 分数 std（门禁2，阈值 0.1） | 两门禁判定 | **pairwise 排序准确率** | BoN@16 增益 |
|---|---|---|---|---|---|
| `seed_43/clir` | 0.001028 | — | **拒绝** ✅ | 0.4763 | −0.0085 |
| `seed_44/clir` | 0.250 | 0.5737 | **通过** ❌ | **0.5735** | **−0.0085** |
| 健康 run（其余） | ≥ 0.0353 | ≥ 0.57 | 通过 ✅ | 0.6666 – 0.7151 | > 0 |

**`seed_44/clir` 两道门禁全过，但它对 best-of-n 的贡献和完全塌缩的 `seed_43/clir` 完全一样：
−0.0085。**它的排序准确率 0.5735 离随机（0.5）比离健康区间（0.6666+）更近。

### 根因

两道门禁测的都不是 BoN 依赖的量：

- 门禁 1 测的是**训练集上的绝对 BCE 水平**。一个模型完全可以学到正确的边缘分布（BCE 远离先验熵，
  相对距离 0.250 看起来很健康），但在 query 内部完全不会排序。
- 门禁 2 测的是**分数的全局散布**。分数散得很开（std 0.5737）只说明模型对不同样本给出不同数值，
  不说明这些数值在同一个 query 内部的相对顺序是对的。

**BoN@k 只依赖 query 内的相对顺序，而两道门禁一个测绝对水平、一个测全局散布，没有一个测顺序。**

### 好消息：加一道门禁就能干净分开

healthy 最低 0.6666 vs 坏的 0.5735 / 0.4763 —— 中间有很宽的间隔。在 0.60 – 0.62 之间任取一个阈值，
都能正确接受全部健康 run、拒绝全部坏 run。所需的量在评估阶段已经可以从现有的
`correctness` + `reward_score` 字段直接算出，不需要新的数据。

### 建议

在 `evaluate_clir.py` 里把 query 内 pairwise 排序准确率作为一个字段算出来并写进报告（它是纯后处理，
零额外成本），然后在 launcher 的 `_require_completed_model` 之后加一道基于它的门禁，阈值预注册在
config 里。这比调 prior-collapse 的容差有效得多 —— 后者根本测不到这个失效模式。

---

## §5 【P1】prior-collapse 门禁的两个数学缺陷

### 5.1 `abs()` 让"比先验更差"的模型通过

```python
642:         relative_distance = abs(final_correctness_bce - prior_entropy) / prior_entropy
643:         passed = tolerance <= 0.0 or relative_distance > tolerance
```

因为取了绝对值，判定的是"离先验熵**足够远**"，方向不分。BCE 显著**高于**先验熵意味着模型比"什么都
不学、直接输出类别频率"还要差 —— 这是明确的训练失败，但 `abs()` 让它以"距离很远"的名义通过。

**实测确认**：我跑的 v3 smoke run 里，`observed_train_correctness_bce = 0.464`，
`constant_prior_bce = 0.139`，模型比常数先验差 3.3 倍，`passed: true`。

修法很简单：把 `abs(a - b)` 改成 `(b - a)`（要求 BCE 必须**低于**先验熵至少 tolerance），或者保留
`abs()` 但额外加一个 `final_correctness_bce <= prior_entropy` 的硬条件。

### 5.2 健康 run 距拒绝带只有 1.8 倍，而失败是硬失败

真实数据里健康 run 的最小相对距离是 `seed_44/strict_swift` 的 **0.0353**，阈值是 **0.02** ——
只有 1.8 倍余量。`strict_swift` 只有 202,754 个参数，本来就接近"只能学边缘分布"，它的 BCE 天然
靠近先验熵，这不是病态。

而 config 里 `fail_on_prior_collapse: true`，意味着一旦某个 seed 的 `strict_swift` 稍微偏低一点，
**整个 cell 被判失败，再叠加 §2 的死锁，整个 9 格矩阵停摆**。

1.8 倍在 seed 间波动面前不是安全余量。建议要么把容差降到 0.01（余量 3.5 倍），要么对
`strict_swift` 单独设阈值（它的容量本来就和另两个变体不在一个量级），要么把它降级为警告而不是硬失败。

---

## §6 已验证确实修好 / 确实不是问题的

这一节是为了避免你在已经正确的地方浪费时间。以下每条我都用真实数据跑过。

### 6.1 上轮 P0-1（卡死矩阵）完全修好

正反两个方向都验证：

- **不加 `--force`**：仍然拒绝，但错误信息现在给出三条**真正可执行**的补救（上一轮给的
  `--resume_from` 在 0-epoch 记录上根本无法执行）。
- **加 `--force`**：在真实的卡死记录上干净重启，退出 0。
- **安全性**：一旦目标位置已存在真实 checkpoint 或 metrics，`--force` 立即拒绝 —— 不会误删真结果。
- **可追溯**：被替换掉的记录被写进 `restarted_from_failed_zero_epoch`，历史不会被悄悄抹掉。
- 原始产物未被触碰。

### 6.2 checker v5 的标签修正精确传播到了冻结数据

v3 的冻结验证池是 **146 / 342 / 12**，语义是 **mixed / all_correct / all_wrong 三类 query**
（`configs/stage1b_validation_v3.json` 的 `manifests.validation.*`），合计 500。上一轮我预测"修掉
那 15 行错标签后应该从 151/337/12 变成 146/342/12" —— 精确命中（5 个 query 从 mixed 变成
all_correct）。checker 版本字段在 8000 行候选上全部是 `clir_gsm8k_numeric_v5`，标签哈希与协议里记录
的组件哈希一致。**这是真的重新标注，不是改了个版本号。**

顺带一个解读实验时必须记住的事实：**500 个 query 里只有 146 个是 mixed，其余 354 个（71%）是
全对或全错**。BoN@k 在全对/全错的 query 上无论选哪个候选结果都一样，所以整个主指标的信号只来自
29% 的 query（也正是 §4 里 pairwise 对数 5282 的来源）。这不是 bug，作者也把它记录在 config 里了，
但它意味着 BoN 的有效样本量远小于 500，置信区间会比"500 个 query"的直觉宽得多。

### 6.3 fp32 打分真的消除了 bf16 的 batch 形状依赖

同一个 checkpoint、同一份输入，在 batch size 1/2/4/8/16 下打分：

| 精度 | 跨 batch 最大偏差 | argmax 翻转 |
|---|---|---|
| bf16（旧） | 1.66e-2 – 2.36e-2 | 有 |
| fp32（新，`amp_dtype: "none"`） | **3.6e-6** | **0** |

约 4000 倍改善，选择结果完全稳定。这条修得很干净。

### 6.4 prior-collapse 门禁的**参考量**在数学上是对的（但被比较的另一边是错的，见 §13）

我担心过门禁的参考熵算错。实测：一个常数流行度预测器的 BCE 是 **0.3369354904**，门禁算出的参考值是
**0.3369355140**，差 **2.4e-8**。参考量本身没问题。

> **⚠️ 本节当时漏掉了一件事。**我只验证了比较的**先验那一边**，没有验证**被比较的那一边**。
> `losses.final` 不是"最终模型的 BCE"，而是**整个 epoch 的样本加权均值**
> （`train_clir.py:469`）。这个漏检后来变成了 v4 的 P0 —— 详见 **§13**。

### 6.5 v3 训练在真实特征上端到端跑通

真实 101376 宽度、bf16 特征、grad clip 1.0、门禁武装、1 epoch clir run → 退出 0，epoch 快照写出，
完整 health_gate schema 和协议 sha256 都记录进 run.json。launcher 生成的 **73 个 flag 全部在目标
脚本里存在**，无拼写错误。preflight 今天实测通过（退出 0）。

### 6.6 存储和耗时不是瓶颈（上轮的 4.7 天 I/O 问题已解决）

| 项 | 实测 | 结论 |
|---|---|---|
| per-epoch 快照总量 | 2.21 GB（可用 197.54 TB） | 完全无压力 |
| per-row provenance | 占每行 1.7%，9 个 cell 共 66 MB | 无压力 |
| fp32 打分（`num_workers: 4`） | 16.8 分钟 / cell，9 格共 2.52 小时 | **上轮 4.7 天的 I/O 瓶颈真的解决了** |

### 6.7 CLIR 监督缺失是被代码强制的降级，不是伪造

preflight 实测两个 split 的 eligible supervision components 都是 0，代码据此把实验降级为
outcome-only 对照。作者没有编造 rewrite/verifier/prior 标注来让检查通过。这一点值得肯定 —— 但请注意
它的含义：**Stage 1B v3 只能作为容量/优化对照，不能作为 CLIR 机制有效性的证据**，机制实验仍然需要
外部生成的标注。

### 6.8 ~~同设备 resume 仍然是位精确的~~ **—— 本节结论错误，已作废，见 §13.3**

> **⚠️ 我原先在这里写"续训位精确、可以放心"。那是错的。**我的测试漏掉了 launcher 实际会发出的
> 三个 loader 参数（`--num_workers 4 --pin_memory --persistent_workers`）。补上之后：
>
> | 配置 | 权重位相同 | 最大偏差 |
> |---|---|---|
> | 无 `--persistent_workers`（我原来测的） | 69 / 69 | 0.000e+00 |
> | 有 `--persistent_workers`（**launcher 实际发出的**） | **44 / 69** | 7.451e-09 |
>
> 冻结协议里 `persistent_workers: true`，所以**位精确在真实配置下不成立**。详见 §13.3。
>
> 这一节唯一仍然成立的部分：训练/验证 query 泄漏守卫是活的（我故意传同一个文件时它正确报错），
> `--force` 与 `--resume_from` 互斥也有显式检查。

### 6.9 测试套件

当前工作树 **124 个测试全部通过**（93.35s）。

---

## §7 【P1】归档的 Stage 1B v1 已经无法再评估

`evaluate_clir.py:99` 现在无条件要求每一行都携带完整的 `clir-reward-scoring-v2` provenance：

```python
99:             raise ValueError("Every scored row requires reward_scoring_provenance")
```

已归档的 v1 打分文件是用旧版 `score_clir.py` 产生的，没有这个字段，**9 个文件全部被拒**。

**这作废了我上一轮给出的建议。**上轮我说 v1 的诊断结论可以"重跑 evaluate，约 3 分钟"就修正过来 ——
现在这条路不存在了，必须重新打分，即 §6.6 里的 **2.52 小时**。

这不是 bug（严格的 provenance 契约本身是好的），但它是一个你需要知道的操作后果：**任何想回头看
v1/v2 归档数据的动作，成本从 3 分钟变成了 2.5 小时。**如果你还需要 v1 的诊断结论，现在就规划这
2.5 小时；如果不需要，可以在文档里明确写"v1/v2 归档数据在 HEAD 上不可再评估"，避免以后有人踩进去。

---

## §8 【P1】resume 在 `run.json` 里不留任何痕迹

§6.8 证明了续训在数值上是对的。但**续训这件事本身在产物里完全不可见**，这与本轮"到处加 provenance"
的主题正好相反。

实测：我先在 **CPU** 上训 1 个 epoch，再把这个 checkpoint 在 **CUDA** 上 resume 训到 2 个 epoch
（退出 0，无任何警告），然后把它的 `run.json` 和一个干净的 2-epoch CUDA run 的 `run.json` 逐字段对比：

```
only in straight: []
only in resumed:  []
DIFF metrics_jsonl -> .../a.metrics.jsonl | .../c.metrics.jsonl
DIFF output_model  -> .../a.pt            | .../c.pt
```

**除了文件路径，两者没有任何差别。**`run.json` 的 19 个顶层字段里没有 `resumed_from`、没有
`start_epoch`、没有源 checkpoint 的 sha256、也没有记录中途换过设备（`device` 只写了最终的 `cuda`，
`training_args.device` 是 `None`）。搜 `restart` / `from_checkpoint` / `start_epoch` 全部零命中。

### 为什么这在本轮特别刺眼

1. **`--force` 路径做对了。**上一轮修的零 epoch 重启会把被替换的记录写进
   `restarted_from_failed_zero_epoch`。作者清楚这种 provenance 是必要的 —— resume 路径只是没有对应物。
2. **门禁看不出来。**`_require_completed_model` 只检查 `status == "completed"` 和
   `health_gate.passed`，一个跨设备拼接出来的 cell 完全合规。
3. **它抵消了本轮其他 provenance 工作。**你在打分层做到了每一行都带 checkpoint sha256、协议 sha256、
   dtype、batch size；但在训练层，无法从产物判断这个 checkpoint 是一次跑成的还是被中断后拼起来的、
   中间有没有换卡。对一个要写进论文的 9 格矩阵，这是审计链上唯一的断口。

### 顺带：`device` 没有被 pin（P2）

`RESUME_PINNED_ARGS`（`:509-526`）pin 了 16 个参数（batch_size、lr、seed、amp_dtype、
prior_collapse_tolerance 等），**但没有 `device`**。所以上面那个 CPU→CUDA 续训一声不响就过了。
而 `_restore_rng_state:494` 的条件是 `state.get("cuda") is not None` —— CPU checkpoint 里
`cuda` 存的正是 `None`，于是 **CUDA RNG 根本没有被恢复**，续训之后的随机性不再由 checkpoint 决定。
（这也是为什么这个 run 的 epoch-2 loss 是 2.8303/3.6735，而同设备位精确对照是 2.8300/3.6726。）

### 建议

在 run 记录里加三个字段就够了：`resumed_from`（源 checkpoint 路径 + sha256）、`start_epoch`、
以及每段训练的 `device`。把 `device` 加入 `RESUME_PINNED_ARGS`（或者至少在跨设备续训时显式报错/警告）。
改动很小，但它补上的是整条审计链现在唯一的缺口。

---

## §9 我自己提出但验证后驳回的假设

列在这里是为了让你不必重复排查。以下每条我都怀疑过，动手验证后确认**不是问题**：

| 我的怀疑 | 为什么不成立 |
|---|---|
| `score_clir.py` 用 torch.float64 算 `score_distribution`，`evaluate_clir.py` 用 numpy 重算并要求 `rel_tol=abs_tol=1e-12`，两个库的累加顺序不同会导致必然失败 | 实测余量 800 – 7200 倍。两边都是 float64，规模下的累加误差远小于 1e-12 阈值。**不会误杀。** |
| MIL 新增的 `negative_loss = -log_no_hallucination / valid_lengths` 会在空行上除零得 NaN | 上游 `:974` 已有守卫（`Every path-level hallucination label requires at least one valid token`），且真实数据最短轨迹是 109 token。**除零不可达。** |
| `_restore_rng_state` 新的 per-device CUDA 恢复循环，会在 CPU 上存的 checkpoint 到 CUDA 机器上续训时崩溃 | checkpoint 里 `cuda` 存的是 `None`（不是空 list），`for` 循环之前有 `None` 检查。**不会崩。** |
| per-row provenance 会让打分文件体积爆炸 | 只占每行 1.7%，9 个 cell 共 66 MB。**无影响。** |
| launcher 的 preflight 会写出文件，从而破坏 `formal_execution_requires_clean_git` 门禁，形成自我否决 | preflight 输出全部落在 `run_artifacts/` 下，而该目录已被 gitignore。**不自我否决。** |
| `FlatLinearFeatureEncoder` 新增的 `ndim != 3` 硬 raise 会打断现有调用路径 | 现有所有调用路径都传 `[batch, time, dim]`。**无影响。**（chunking 本身对 2^31 索引上限的处理是正确的。） |

---

## §10 建议处理顺序

按"能跑起来 + 逻辑正确"这个唯一目标排序：

1. **先修 §2 的死锁**（最高优先）。在 `summarize_clir.py` 加显式的、会写进产物 provenance 的降级
   路径。不修这个，任何一个 seed 塌缩都会让你在跑完 2.5 小时打分之后卡在最后一步拿不到结果。
2. **修 §3 的快照门禁路径**（改动最小，收益明确）。让 `checkpoint_epoch` 非空时门禁读 metrics
   里对应 epoch 的 `training_health`，而不是最终 run.json。数据已经在那里了。
3. **加 §4 的 pairwise 排序门禁**（对实验有效性最关键）。纯后处理、零额外成本、真实数据上能干净
   分开好坏 run。不加这个，`seed_44/clir` 这类"看起来健康但完全不会排序"的 run 会进入主结果。
4. **修 §5.1 的 `abs()`**（一行改动）。
5. **调 §5.2 的容差或给 `strict_swift` 单独阈值**，避免误杀。
6. **补 §8 的 resume provenance**（三个字段 + 把 `device` 加进 pin 列表）。这是审计链上唯一的缺口，
   而且趁现在还没跑正式 v3 补最省事 —— 跑完再补就无法追溯已产出的 cell 了。
7. **决定 §7 的归档数据要不要重评**，要就现在排 2.5 小时。
8. **提交工作树**，这是今天启动正式 v3 的唯一阻塞（preflight 本身已通过）。

前 6 项都是小改动，加起来远小于这一轮已经完成的工作量。

---

## §11 关于本次审查方法的说明

**为什么审查过程会启动训练和打分。**这一轮改动的核心声明（"prior-collapse 门禁能抓住塌缩"、
"fp32 消除了 batch 依赖"、"v5 修正了标签"、"num_workers=4 解决了 I/O 瓶颈"、"epoch 快照可用于
审计"）**没有一条能靠读代码判定真假**。门禁到底抓不抓得住，只有把真实 checkpoint 喂进去才知道；
fp32 到底稳不稳，只有在 5 个 batch size 上各打一遍分才知道。§4 那张表 —— 也就是本轮最重要的发现 ——
完全来自实测，读代码永远得不到 0.5735 这个数字。

约束我全程遵守：**没有修改仓库里任何一行代码**；`run_artifacts/` 下的真实产物全程只读，未删除未覆写；
所有临时产物写在 `/tmp` 下。

**一个例外需要报告**：并行审查中有一个 agent 把临时训练输出写成了相对路径，在仓库根目录留下了一个
`wedged_test/` 目录（只含 tiny toy run 的 metrics/run.json，未跟踪）。这是审查过程的产物，不是你的
改动；该 agent 在自己的测试结束时已把它清理掉，我复查确认仓库现在除本报告外没有任何审查残留，
工作树只有你自己的 22 改 + 15 新。

**本报告的完成度**：§2–§9 的全部结论都已由我独立实测确认，可以直接依据它们行动。并行审查的 11 个
维度（resume 恢复、门禁、provenance 契约、checker v5、损失数学、launcher、v3 协议、监督模块、
vLLM seed、可运行性、回归扫描）仍在运行，如果它们返回本报告未覆盖的确认发现，我会追加到本文件。

---

## §12 复现附录（交接用）

本节让接手的人不必依赖我的临时目录就能独立复现全部关键结论。所有命令假定
`cd /prodcpfs/user/panzhixin/ICLR_2027`，且
`P=/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python`（**必须用这个解释器**，
`miniconda3/bin/python` 里没有 pytest）。全部命令只读仓库和 `run_artifacts/`，输出写 `/tmp`。

### A. 复现 §4 的排序准确率表（最重要，纯只读，约 10 秒）

把下面这段存成 `/tmp/repro_pairwise.py`：

```python
"""Recompute per-cell query-internal pairwise ranking accuracy from scored manifests."""
import json, sys, collections, pathlib
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "run_artifacts/stage1b_v1/scored")
for seed_dir in sorted(root.glob("seed_*")):
    for f in sorted(seed_dir.glob("*.jsonl")):
        by_q = collections.defaultdict(list)
        for line in f.open():
            r = json.loads(line)
            by_q[r["query_id"]].append((float(r["reward_score"]), int(r["correctness"])))
        good = tot = 0
        for cands in by_q.values():
            pos = [s for s, c in cands if c == 1]
            neg = [s for s, c in cands if c == 0]
            for p in pos:
                for n in neg:
                    tot += 1
                    good += (p > n) + 0.5 * (p == n)
        acc = good / tot if tot else float("nan")
        print(f"{seed_dir.name}/{f.stem:15s} pairwise={acc:.4f}  queries={len(by_q)}  pairs={tot}")
```

`$P /tmp/repro_pairwise.py` 应当逐字打印：

```
seed_42/clir            pairwise=0.6666  queries=500  pairs=5282
seed_42/encoded_swift   pairwise=0.6732  queries=500  pairs=5282
seed_42/strict_swift    pairwise=0.7096  queries=500  pairs=5282
seed_43/clir            pairwise=0.4763  queries=500  pairs=5282
seed_43/encoded_swift   pairwise=0.6946  queries=500  pairs=5282
seed_43/strict_swift    pairwise=0.6941  queries=500  pairs=5282
seed_44/clir            pairwise=0.5735  queries=500  pairs=5282
seed_44/encoded_swift   pairwise=0.6785  queries=500  pairs=5282
seed_44/strict_swift    pairwise=0.7151  queries=500  pairs=5282
```

**判读**：`seed_44/clir` 的 0.5735 是 §4 的全部要害 —— 它两道门禁全过，排序能力却接近随机。
健康 run 的最低值 0.6666 与它之间的间隔，就是可用的门禁阈值区间（建议 0.60–0.62）。

### B. 复现 §5.1 的 `abs()` 缺陷（纯计算，无需 GPU，约 1 秒）

```
$P -c "
import math
prevalence = 0.7155          # 任取一个非退化流行度
prior = -(prevalence*math.log(prevalence) + (1-prevalence)*math.log(1-prevalence))
for observed in (prior*0.5, prior*3.3):   # 一个比先验好，一个比先验差 3.3 倍
    rel = abs(observed - prior) / prior
    print(f'observed_bce={observed:.4f} prior={prior:.4f} rel={rel:.4f} passed={rel > 0.02}')
"
```

两行都会打印 `passed=True`。**比先验差 3.3 倍的模型和比先验好的模型得到同一个结论**，这就是
`train_clir.py:642` 取绝对值的后果。

### C. 复现 §6.2 的标签与 §6.2 的 mixed query 占比（纯只读，约 20 秒）

```
$P -c "
import json, collections
c = collections.Counter(); ck = collections.Counter()
for line in open('run_artifacts/stage1b_v3/labels/validation_extracted.v5.jsonl'):
    r = json.loads(line)
    c[r.get('correctness')] += 1; ck[r.get('checker_version','<none>')] += 1
print('candidate-level:', dict(c)); print('checker versions:', dict(ck))
"
$P -c "
import json
m = json.load(open('configs/stage1b_validation_v3.json'))['manifests']['validation']
print({k: v for k, v in m.items() if 'quer' in k})
"
```

预期：候选级 `{0: 845, 1: 7155}`，checker 版本 `{'clir_gsm8k_numeric_v5': 8000}`；
query 级 `mixed_queries=146, all_correct_queries=342, all_wrong_queries=12`。

### D. 复现 §6.8 的续训位精确性（需 1 张 GPU，约 2 分钟）

先造一份 query 不重叠的 toy 数据（训练/验证泄漏守卫是活的，同一个文件会被正确拒绝），然后：

```
COMMON="--train_jsonl <train> --val_jsonl <val> --hidden_dim 8 --model_variant clir \
  --encoder_type layer_transformer --batch_size 2 --lr 1e-4 --seed 42 --device cuda \
  --amp_dtype none --no-group_by_semantic_id --prior_phase_mode joint \
  --skip_feature_finite_check --val_every_n_epochs 1"

# A: 一次跑 2 epoch
$P train_clir.py $COMMON --epochs 2 --output_model /tmp/a.pt \
   --metrics_jsonl /tmp/a.metrics.jsonl --run_json /tmp/a.run.json
# B: 先 1 epoch，再 resume 到 2 epoch（注意 --force 与 --resume_from 互斥）
$P train_clir.py $COMMON --epochs 1 --output_model /tmp/b.pt \
   --metrics_jsonl /tmp/b.metrics.jsonl --run_json /tmp/b.run.json
$P train_clir.py $COMMON --epochs 2 --resume_from /tmp/b.pt --output_model /tmp/b.pt \
   --metrics_jsonl /tmp/b.metrics.jsonl --run_json /tmp/b.run.json
```

然后逐张量比对（`state_dict` / `optimizer_state_dict`，注意顶层键名是 `state_dict` 而非
`model_state`）：

```
$P -c "
import torch
a = torch.load('/tmp/a.pt', map_location='cpu'); b = torch.load('/tmp/b.pt', map_location='cpu')
w = sum(torch.equal(a['state_dict'][k], b['state_dict'][k]) for k in a['state_dict'])
print(f'weights bit-identical: {w}/{len(a[\"state_dict\"])}')
oa, ob = a['optimizer_state_dict']['state'], b['optimizer_state_dict']['state']
n = ok = 0
for pid in oa:
    for mk in oa[pid]:
        if torch.is_tensor(oa[pid][mk]):
            n += 1; ok += torch.equal(oa[pid][mk], ob[pid][mk])
print(f'optimizer bit-identical: {ok}/{n}')
"
```

预期 `weights bit-identical: 69/69`、`optimizer bit-identical: 195/195`，且 epoch 2 的
train/val loss 两边都是 `2.8300 / 3.6726`。

### E. 复现 §8 的 resume 无痕迹（需 1 张 GPU，约 2 分钟）

用 D 的 `COMMON`，但第一段跑在 CPU 上、第二段 resume 到 CUDA：

```
$P train_clir.py <COMMON with --device cpu> --epochs 1 --output_model /tmp/c.pt ...
$P train_clir.py <COMMON with --device cuda> --epochs 2 --resume_from /tmp/c.pt \
   --output_model /tmp/c.pt ...        # 退出 0，无任何警告
$P -c "
import json
a = json.load(open('/tmp/a.run.json')); c = json.load(open('/tmp/c.run.json'))
print('only in straight:', sorted(set(a) - set(c)))
print('only in resumed :', sorted(set(c) - set(a)))
for k in sorted(set(a) & set(c)):
    if a[k] != c[k] and k not in {'metrics', 'health_gate'}:
        print('DIFF', k)
"
```

预期：两个 `only in` 都是空列表，`DIFF` 只有 `metrics_jsonl` 和 `output_model` 两个路径字段。
**这就是 §8 —— 跨设备拼接出来的 run 与一次跑成的 run 在 provenance 上无法区分。**

### F. 复现 §2 的死锁（不需要真跑训练）

不必制造一次真塌缩。直接读代码路径即可确认闭环，三处各看一眼：

```
sed -n '547,556p' scripts/run_stage1b_validation.py     # 门禁：status/health_gate 不过就 raise
sed -n '570,574p' scripts/run_stage1b_validation.py     # scoring_command 无条件调用它
grep -nE 'force|override|allow_missing|skip' summarize_clir.py   # 零命中 = 无出口
grep -n -- '--force' scripts/run_stage1b_validation.py           # 只有两处，都与门禁无关
```

（注意 `--force` 的 flag 名和 `parser.add_argument(` 不在同一行，所以要按上面这样搜，
搜 `add_argument("--force` 会一无所获。）

最后一条要特别确认：launcher 里唯一的 `--force*` 是 `--force-zero-epoch`（`:62`，上一轮为 0-epoch
卡死记录加的），以及 `:525` 传给 `train_clir.py` 的覆盖写 `--force`。**两者都与健康门禁无关。**

### G. 复现 §3 的快照不可用

```
sed -n '330,361p' scripts/run_stage1b_validation.py   # 注意 :357 的 "run" 不随 checkpoint_epoch 变
grep -n 'training_health' train_clir.py               # :939 每 epoch 都写
sed -n '661,672p' train_clir.py                       # 但 :670 只读 metric_rows[-1]
```

### H. 测试套件

```
$P -m pytest -q
```

预期 `124 passed`。

---

## §13 追加：v4 已经落地，本报告的两条结论需要修正，并发现一个新 P0

**背景变化。**我在写 §0–§12 期间，工作树从我审查的 `a49891d` 前进了三个 commit：

```
58194bb  Record semantic rewrite pilot audit              <- 现 HEAD
bc393cf  Lock Stage 1B v4 result and start rewrite pilot
b1c4fae  Resolve fourth CLIR audit and freeze Stage 1B v4  <- 本报告在此被采纳
a49891d  Resolve third CLIR audit and freeze Stage 1B v3   <- 我审查的树
```

`scripts/run_stage1b_validation.py:26` 的 `DEFAULT_PROTOCOL` 现在指向
`configs/stage1b_validation_v4.json`。**§0–§12 全部是针对 v3 写的**，以下是针对 v4 现状的修正。

### 13.1 本报告的两条建议已被采纳，实现是对的

- **§2 的死锁**：`summarize_clir.py` 现在有 `--allow-failed-cells`，并在产物里写
  `matrix_complete` / `matrix_status.failed_cells`（含每个失败 cell 的 `evidence_sha256`）/
  `result_status` / `formal_primary_claim_allowed`。这正是 §2 建议的显式降级路径。
- **§5.1 的 `abs()`**：门禁改为单边（`train_clir.py:645-646`）：
  `relative_improvement = (prior_entropy - final_correctness_bce) / prior_entropy`，
  `passed = relative_improvement >= tolerance`。`abs()` 只作为诊断字段保留，不再参与判定。

### 13.2 【P0】但 §5.1 修好之后，门禁把 9 格里的 8 格判死了 —— 其中 3 格是误杀

v4 已锁定的结果（`run_artifacts/stage1b_v4/summary.json`，作者自己的产物）：

```
matrix_complete: false     included_cell_count: 1     failed_cell_count: 8
result_status: "incomplete_diagnostic_only"
formal_primary_claim_allowed: false
available_seeds_by_variant: {strict_swift: [], encoded_swift: [42], clir: []}
```

**先说公道话**：这个产物是诚实的 —— 没有伪造，8 个失败 cell 各带证据哈希，并明确禁止 formal
primary claim。§2 的机制按设计工作了。

**但门禁读的是错的量。**`train_clir.py:729` / `:1030` 喂给门禁的是
`train["losses"]["final"]`，而 `train_clir.py:469` 把它算成
`value / max(examples, 1)` —— **整个 epoch 所有 batch 的样本加权均值**，不是最终 checkpoint 的 BCE。
对一个在 epoch 内持续下降的损失，epoch 均值必然高于终点值。

我用 `run_epoch(..., optimizer=None)`（与训练完全相同的损失路径，no-grad）对**全部 9 个已有
checkpoint** 在完整 4096 行训练集上重算了真实的最终 BCE。`encoder_dropout = 0.0`，所以 train/eval
模式没有差别，两个数字直接可比：

| cell | 先验熵 | 门禁用的 epoch 均值 | rel | 门禁 | **最终 checkpoint 真实 BCE** | **rel** | 应判 | 翻转 |
|---|---|---|---|---|---|---|---|---|
| `seed_42/strict_swift` | 0.33484 | 0.43613 | −0.3025 | 失败 | **0.30811** | **+0.0798** | **通过** | **是** |
| `seed_43/strict_swift` | 0.33484 | 0.43811 | −0.3084 | 失败 | **0.23835** | **+0.2882** | **通过** | **是** |
| `seed_44/strict_swift` | 0.33484 | 0.41478 | −0.2387 | 失败 | **0.24465** | **+0.2694** | **通过** | **是** |
| `seed_42/encoded_swift` | 0.33484 | 0.31952 | +0.0458 | 通过 | 0.25843 | +0.2282 | 通过 | 否 |
| `seed_43/encoded_swift` | 0.33484 | 0.36654 | −0.0947 | 失败 | 0.35333 | −0.0552 | 失败 | 否 |
| `seed_44/encoded_swift` | 0.33484 | 0.36305 | −0.0842 | 失败 | 0.37154 | −0.1096 | 失败 | 否 |
| `seed_42/clir` | 0.33484 | 0.52667 | −0.5729 | 失败 | 0.48728 | −0.4552 | 失败 | 否 |
| `seed_43/clir` | 0.33484 | 0.40792 | −0.2182 | 失败 | 0.41119 | −0.2280 | 失败 | 否 |
| `seed_44/clir` | 0.33484 | 0.51374 | −0.5343 | 失败 | 0.47392 | −0.4154 | 失败 | 否 |

**3 个 cell 翻转，全部是 `strict_swift`，而且不是勉强通过**：+0.0798 / +0.2882 / +0.2694，是 1%
阈值的 **8 倍 / 29 倍 / 27 倍**。门禁把整整一行变体全部误杀了。

机制在 per-epoch 曲线上一目了然（`seed_42/strict_swift`）：

```
epoch 1: train_final=0.826143    epoch 4: train_final=0.466925
epoch 2: train_final=0.586429    epoch 5: train_final=0.436125   <- 门禁用这个
epoch 3: train_final=0.485579    最终模型真实 BCE = 0.308114     <- 应该用这个
```

5 个 epoch 之后损失仍在陡降（0.467 → 0.436），所以 epoch 均值与终点值的差距始终很大。**这就是为什么
几乎每格都失败**：所有 cell 都还在曲线的陡峭段。

**修法**（不需要重训，checkpoint 已经存在）：在最后一个 epoch 之后加一次 no-grad 全量前向，用它的
`losses["final"]` 喂门禁。实测每个 cell 约 4–5 分钟，9 格并行不到 10 分钟。我用的脚本见 §13.5。

### 13.3 【P1】§6.8 的位精确结论作废：真实配置下 resume 不是位精确的

`--persistent_workers` 为真时（**冻结协议就是真**），`DataLoader.__iter__` 复用迭代器并走
`_reset()`，而 `_reset()` 不重新抽 `_base_seed`。于是 epoch 1 消耗 2 次全局 CPU RNG 抽样、后续 epoch
只消耗 1 次；而**续训进程总是重新走 2 次抽样的首迭代器路径**，导致 `_restore_rng_state()` 之后
RandomSampler 的种子取自被恢复流的不同位置 → shuffle 顺序不同 → 权重不同。

我自己的对照（同一份数据、同一个 kill 点、只改这一个 flag）：

| 配置 | 权重位相同 | 最大偏差 |
|---|---|---|
| `--no-persistent_workers` | 69 / 69 | 0.000e+00 |
| `--persistent_workers`（launcher 实际发出的） | **44 / 69** | 7.451e-09 |

我的偏差很小只因为 toy 集只有 2 个 batch。在真实特征上并行审查测到的量级是
max|Δ| = **2.2e-3**，epoch-3 训练损失 **0.5130（续训）vs 0.5799（不中断）**，相对差 11.5%。

与 **§8**（resume 在 `run.json` 里不留任何痕迹）叠加，这才是完整的问题：
**续训会改变数字，而且产物里看不出发生过续训。**

`tests/test_clir_training_state.py:37` 的 resume 回归测试同样漏掉了这三个 flag，所以测试套件
也抓不到它。修法：给非分组路径的 DataLoader 传显式 `generator` 并按 (seed, epoch) 逐 epoch 重播种
（`SemanticGroupBatchSampler` 已经这么做了）；或者最省事 —— 从 config 里去掉
`persistent_workers`（实测即位精确）。**回归测试必须带上 launcher 的真实 loader flag。**

### 13.4 修正后的实验结论：坏消息，但是有意义的坏消息

把门禁修对之后，9 格的真实判定是 **4/9 通过**（而非 1/9），且分布极有规律：

| 变体 | 参数量 | 通过 | 最终 checkpoint 相对先验 |
|---|---|---|---|
| `strict_swift` | 202,754 | **3 / 3** | +0.0798 / +0.2882 / +0.2694 |
| `encoded_swift` | 3,435,266 | 1 / 3 | +0.2282 / −0.0552 / −0.1096 |
| `clir` | 9,547,273 | **0 / 3** | −0.4552 / −0.2280 / −0.4154 |

**最小的模型训得最好，最大的模型全部比常数先验更差，而且是差很多。**这不是门禁 bug，是关于模型的
真实结论：容量越大越训不动。它与 §4 的证据一致 —— v1 里 `clir` 的 query 内排序准确率
0.4763 / 0.5735，本来就接近随机。

所以修好门禁**不能**让 v4 产出主张（clir 一格不过，encoded_swift 只过 1 格）。但诊断完全不同：
现在锁定的产物读起来像"整条流水线坏了"，实际情况是**流水线没坏，是 CLIR 变体本身没在学**。
这个区别决定下一步该做什么 —— 该查的是优化（lr / 预热 / 归一化 / prior phase 的权重），不是基础设施。

### 13.5 复现 §13.2（把这段存成 `/tmp/gatecheck.py`）

```python
"""Recompute the FINAL checkpoint's mean train BCE and compare with the gate's number."""
import json, sys, torch
sys.path.insert(0, '.')
import train_clir as tc
from src.clir_data import CLIRTrajectoryDataset
from score_clir import load_model_with_checkpoint

seed, variant = sys.argv[1], sys.argv[2]
run = json.load(open(f'run_artifacts/stage1b_v4/models/seed_{seed}/{variant}.run.json'))
h = run['health_gate']
device = torch.device('cuda')
model, _ = load_model_with_checkpoint(
    f'run_artifacts/stage1b_v4/models/seed_{seed}/{variant}.pt', device)
ds = CLIRTrajectoryDataset(
    'run_artifacts/stage1b_v3/labels/train_extracted.v5.jsonl', None, check_finite=False)
loader = tc.make_loader(ds, batch_size=2, indices=list(range(len(ds.rows))),
                        shuffle=False, group_by_semantic_id=False, seed=42,
                        num_workers=4, pin_memory=True, persistent_workers=False)
out = tc.run_epoch(model, loader, device, None, prior_phase='joint', amp_dtype='none')
final_ckpt_bce = out['losses']['final']
prior = h['constant_prior_bce']; gate_bce = h['observed_train_correctness_bce']
tol = h['minimum_relative_improvement']
ri_gate = (prior - gate_bce) / prior
ri_true = (prior - final_ckpt_bce) / prior
print(f'seed_{seed}/{variant}   (examples={out["examples"]})')
print(f'  prior_entropy                 = {prior:.10f}')
print(f'  gate used (EPOCH MEAN)        = {gate_bce:.10f}  rel_impr={ri_gate:+.6f} passed={ri_gate>=tol}')
print(f'  FINAL CHECKPOINT actual train = {final_ckpt_bce:.10f}  rel_impr={ri_true:+.6f} passed={ri_true>=tol}')
print(f'  VERDICT FLIPS: {(ri_gate>=tol) != (ri_true>=tol)}')
```

九格并行（每格约 4–5 分钟，8 张 L20Z 上不到 10 分钟）：

```
i=0
for cell in "42 strict_swift" "42 encoded_swift" "42 clir" "43 strict_swift" \
            "43 encoded_swift" "43 clir" "44 strict_swift" "44 encoded_swift" "44 clir"; do
  set -- $cell
  CUDA_VISIBLE_DEVICES=$((i % 8)) nohup $P /tmp/gatecheck.py $1 $2 > /tmp/gate_$1_$2.log 2>&1 &
  i=$((i+1))
done
```

### 13.6 修正后的处理顺序（取代 §10）

1. **修 §13.2 的门禁测量口径**（P0，一次 no-grad 前向）。这是唯一挡在"能不能读懂 v4 结果"前面的东西，
   而且不需要重训。
2. **用修好的门禁重跑 9 格的 gate 判定 + summarize**，拿到真实的 4/9 而不是误导性的 1/9。
3. **接受 §13.4 的结论并转向优化排查** —— `clir` 0/3 比常数先验更差，这是当前最重要的实验事实。
4. **修 §13.3 的 resume 位精确性 + §8 的 resume provenance**，趁 v5 还没开跑。
5. **加 §4 的 pairwise 排序门禁** —— 在 §13.4 的背景下它更重要了：它能直接量化"clir 没在学"。
6. §5.2 的容差余量、§7 的归档数据决策，仍然有效但优先级最低。
