# CLIR 项目交接文档（写给下一个接手的人 / AI）

最后更新：2026-08-12（对应 commit `be0dc7f`，分支 `claude/code-review-logic-bugs-nmc2ea`，基于已合并进 `main` 的 `d948d40`）

这份文档是专门写给**完全没有上下文**的下一个开发者或 AI 看的。目标是让你不用重新翻一遍 commit history、不用重新做一遍代码审查，就能知道：现在做到哪一步了、为什么是这样设计的、踩过哪些坑、接下来该做什么。

如果你只有时间看一个文件，看这个文件。如果还有时间，再看：

- `README.md`：项目状态的滚动记录（每次改代码都会同步更新"当前进展/遇到的问题/未来解决方向"），信息和这份文档有重叠，但更新粒度更细、更像 changelog。
- `docs/proposal.md`：研究方法设计文档，公式、符号定义、论文对照关系写得比这里详细。
- `tests/test_clir_smoke.py`：目前所有行为契约都在这里，比文字描述更权威。

## 1. 这个项目在做什么（30 秒版本）

一句话：给 LLM reasoning trajectory 做一个轻量级、不依赖大模型的 reward model，用于 Best-of-N sampling 选答案，目标是比 baseline SWIFT 更抗"风格偏见"和"幻觉"。

三篇论文，三个来源：

| 论文 | 在 CLIR 里对应什么 |
|---|---|
| **SWIFT**（*Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling*，KDD'26，直接 baseline） | token-level gate/reward 线性头 + 加权平均聚合成标量 reward 的架构，`R(H) = Σ g_t·r_t / max(Σ g_t, ε)` |
| **PRISM**（*Reducing Spurious Implicit Biases in Vision-Language Models with LLM-Guided Embedding Projection*，ICCV'25） | 用 LLM 改写题干/风格制造"同语义不同风格"的样本对，用对比式 loss 让 reward representation 对风格不变、对语义敏感 |
| **DPCL / MLLM4WTAL**（*Weakly Supervised Temporal Action Localization via Dual-Prior Collaborative Learning*，CVPR'25） | key prior（最关键证据 token）+ complete prior（完整支持证据 token span）两个 attention head，互蒸馏来缓解"定位不全"和"定位过多"这两种互相矛盾的偏差 |

CLIR = SWIFT 的 token reward/gate 架构 + PRISM 的一致性 loss + DPCL 的 dual-prior 定位 + 一个原创的"幻觉起始点之后 reward 强制为负"的 tail-shaping 机制。

**代码不依赖、不调用 SWIFT/PRISM/DPCL 任何一方的官方仓库**，全部在本仓库自包含实现，只是架构上参考。

## 2. 现在的状态：能跑，但只能跑在合成（随机）数据上

这是最重要的一句话，必须先说清楚：

> **目前仓库里没有任何真实数据。** `examples/create_toy_clir_data.py` 生成的是 `torch.randn` 随机向量，不是从任何 LLM 抽取的真实 hidden states。所有的正确性验证目前都停留在"代码逻辑对不对、张量形状对不对、loss 会不会变成 NaN"这个层面，**完全没有验证过"这套方法真的能学出更好的 reward model"这个研究问题本身**。

也就是说：软件工程层面（模型代码、数据管线、训练脚本）已经比较扎实了，但 research 层面（这个方法到底有没有效）一步都还没开始。下一个人接手，最大的任务不是继续抠代码细节，而是**把真实数据接进来，跑第一次真正的实验**。

### 2.1 已经实现、且验证过能跑通的部分

以下内容我在装有 `torch`（2.13, CPU）/ `pytest` 的环境里实际跑过（不是只看代码）：

- `pytest tests/test_clir_smoke.py`：11/11 通过（含一个专门防止条件化模块参数量再次退化成平方级的回归测试，见 7.2 节）。
- 端到端流程：`examples/create_toy_clir_data.py` → `train_clir.py`（带全部 CLI 参数）→ `score_clir.py`，全程无崩溃、无 NaN。
- 针对 `SemanticGroupBatchSampler` 单独做过 300 组随机场景（组数 1-12、组大小 1-6、`batch_size` 2-10、`shuffle`/`drop_last` 全组合）共 1200 次采样的压力测试，确认 `__len__()` 与 `__iter__()` 实际产出的 batch 数一致，且每个样本在非 `drop_last` 模式下恰好被覆盖一次。

### 2.2 文件地图

```
README.md                            项目状态滚动记录（changelog 风格）
docs/proposal.md                     研究方法设计文档（公式、符号、评估计划）
docs/handoff.md                      本文件：交接文档
requirements.txt                     依赖版本（核心库对齐 SWIFT，见下方说明）
src/consistency_localized_reward.py  模型定义 + 所有 loss（核心文件，765 行）
src/clir_data.py                     JSONL 数据集、collate、SemanticGroupBatchSampler（509 行）
train_clir.py                        训练入口（233 行）
score_clir.py                        打分 + Best-of-N 选择入口（130 行）
examples/create_toy_clir_data.py     合成（随机）toy 数据生成脚本，仅用于 smoke test
tests/test_clir_smoke.py             全部测试（292 行，11 个 test）
```

没有别的隐藏文件、没有 CI 配置。有一个 `requirements.txt`，`torch`/`numpy` 版本对齐了 SWIFT 官方仓库（[aster2024/SWIFT](https://github.com/aster2024/SWIFT)）的 pin，因为 CLIR 的架构是照着 SWIFT 参考实现的；`transformers`/`accelerate` 等要等第 6 节的 P0 任务（真实 hidden-state 抽取脚本）实现了再启用，细节和踩过的坑（SWIFT 自己的 `numpy`/`accelerate` 版本互相冲突）写在 `requirements.txt` 的注释和 `README.md` 的"运行代码"一节里，不重复贴一遍。

## 3. 模型架构详解（对着代码看）

入口：`ConsistencyLocalizedReward.forward()`，`src/consistency_localized_reward.py:103-188`。

### 3.1 输入

- `hidden_states: [batch, time, hidden_dim]`：**假设已经是抽取好的 frozen LLM hidden states**（哪几层、要不要 concat、要不要拼 logits，都是调用方决定，模型本身不关心）。
- `mask: [batch, time]`：有效 token 的 0/1 mask。
- 可选的 query/context 条件输入，二选一或都给：
  - `condition_states/condition_mask: [batch, C, hidden_dim]`：变长的 condition token 序列（比如 query+context 拼起来的所有 token）。
  - `condition_embedding/condition_embedding_mask: [batch, hidden_dim]`：单个已经池化好的 condition 向量。
  - 两者都不给：模型退化成纯 SWIFT（`token_features = hidden_states` 原样传下去），这是刻意保留的 fallback，不是 bug。

### 3.2 Query/Context 条件化（`_condition_token_features`，313-383 行）

**这是全文件里最容易被下一个人重新写错的地方，务必先看这段再动手改。这里已经踩过两次坑了，一次是"数学上零效果"，一次是"参数量爆炸"，两次都很隐蔽，纯读代码/小 hidden_dim 的 toy 测试都看不出来。**

**坑 1：条件化对 dual-prior 数学上零效果（已修复）。** 最早一版实现是把 query/context 池化成一个向量，**原样加到每个 token 上**（uniform bias）。这是错的——因为所有下游 head（reward/gate/hallucination/key_prior/complete_prior）都是对 `token_features` 做 `nn.Linear`，而 `Linear(h_t + c) = Linear(h_t) + Linear(c)`，`c` 对同一行所有 t 都一样，所以这种"加常数"的条件化**只能给整条轨迹加一个统一偏置，不可能让模型区分"这个 token 是否和 query/context 相关"**。更严重的是，`key_prior`/`complete_prior` 这两个 head 后面还要过 softmax 归一化，而 softmax 对"所有位置加同一个常数"是严格平移不变的——也就是说旧版本的条件化对 dual-prior **数学上是零效果**。修复方式：每个 generated token 对 condition tokens 做**逐 token 的 cross-attention**（`condition_query`/`condition_key`/`condition_value`，75-77 行），attention 权重逐 token 不同，所以 softmax 不再是平移不变的。**千万不要把条件化重新简化成"池化+相加"这种写法**，会静默地回退到这个 bug。

**坑 2：条件化模块参数量对 `hidden_dim` 是平方级（已修复）。** 加完 cross-attention 之后，`condition_query`/`condition_key`/`condition_value` 一开始是 `Linear(hidden_dim, hidden_dim)`，`condition_fusion` 第一层是 `Linear(4*hidden_dim+1, hidden_dim)`，`complete_reconstructor` 也是两层 `Linear(hidden_dim, hidden_dim)`——这些全是 `hidden_dim` 的平方级参数量。`hidden_dim=8` 的 toy 数据完全看不出问题，但 `hidden_dim` 本来就该是 SWIFT 那种"拼接所有 transformer 层"的量级（Llama-3.1-8B 是 33 层 × 4096 = 135,168）。实测过：不修的话，在这个真实量级下光条件化模块就要 **约 1827 亿参数**（比 GPT-3 还大），而 SWIFT 自己在同样输入维度下整个 reward model 只有 2.7×10⁵ 参数——完全违背"轻量级 reward model"这个立项初衷，也不可能训得动（几千条样本喂 1800 亿参数纯属灾难性过拟合，不是加几张 GPU 能解决的）。修复方式：新增一个独立的瓶颈维度 `condition_attention_dim`（`RewardConfig` 新字段，默认 256），`condition_query`/`condition_key`/`condition_value`/`condition_hidden_proj` 先把 `hidden_dim` 投影到这个小维度，交互特征和 `condition_fusion` 都在这个小维度里算，最后用 `condition_delta_out: Linear(condition_attention_dim, hidden_dim)` 投影回 `hidden_dim`；`complete_reconstructor` 同理改成 `hidden_dim -> condition_attention_dim -> hidden_dim` 的沙漏结构。实测同样的 Llama-3.1-8B 量级（`hidden_dim=135,168`），修复后整个模型只有 **约 2.79 亿参数**，比修之前少了约 655 倍，单卡随便训。`tests/test_clir_smoke.py::test_condition_module_params_scale_linearly_with_hidden_dim` 是这个问题的回归测试，以后再改这块代码要保证这个测试还过。**千万不要在这几层里直接用 `hidden_dim` 做输出维度，一定要过 `condition_attention_dim` 这个瓶颈。**

关键产出（`forward()` 里对应的键名，这两次修复都没变过）：

- `relevance: [batch, time]`：每个 token 和它 attend 到的 condition 内容的匹配分数（query 和 attention-加权 key 的点积），对应你最初想法里"看有没有相关性"这一诉求，输出在 `forward()` 返回值里叫 `condition_relevance`，也会写进 `score_clir.py` 的打分结果。
- `context`：现在是 `[batch, time, condition_attention_dim]`（不是 `hidden_dim` 了），每个 token attend 到的 condition 内容的加权和。
- `hidden_states` 先投影成 `hidden_proj: [batch, time, condition_attention_dim]`，和 `context` 拼起来（`[hidden_proj, context, hidden_proj*context, hidden_proj-context, relevance]`）过 `condition_fusion` 这个小 MLP，再过 `condition_delta_out` 投影回 `hidden_dim` 得到 `delta`，加回 `hidden_states` 后过 `feature_norm`（LayerNorm）得到 `token_features`（`[batch, time, hidden_dim]`，这个外部契约没变）。

有条件的行（`has_condition=True`）会走 LayerNorm 归一化，完全没提供 condition 的行会原样保留 `hidden_states`（不过 LayerNorm）。这在同一个 batch 里混用时会有轻微的分布不一致（有条件的行是 LayerNorm 过的，没条件的行不是），目前判断这是可接受的 graceful-degradation 设计，不是 bug——但如果真实数据里"有没有 condition"这件事本身就有意义（比如某些任务就是没有 context），要留意这个尺度差异会不会影响下游 head。

### 3.3 Reward / Gate（SWIFT 部分，135-149 行）

```python
token_head = token_reward_head(token_features)   # 一个 Linear(hidden_dim, 2) 同时出 gate 和 reward
gate_logits, token_rewards = token_head[...,0], token_head[...,1]
gates = sigmoid(gate_logits) * mask

hallucination_logits = hallucination_head(token_features)
progress = progress_head(token_features)
token_values = token_rewards + progress_score_weight * progress   # 见 3.5 的已知问题

token_scores = Σ(gates * token_values) / max(Σ gates, eps)        # SWIFT 公式，只是把 r_t 换成了 token_values
score_residual = final_score_head(mean_pool(token_features))       # 额外加的残差头，见下
scores = token_scores + score_residual
```

`final_score_head` 是 CLIR 独有的，SWIFT 原文没有。加它的原因：`scores` 同时要满足两个目标——(a) 作为 BCE 的 logit 去拟合 correctness（希望能发散到较大的正/负值），(b) 它的组成部分 `token_rewards`/`progress` 又被幻觉 tail loss 用 MSE 拉向绝对数值很小的 `token_advantage`/`-negative_tail_margin`（默认量级在 -0.5～1 之间）。这两个目标在同一个张量上打架。`final_score_head` 相当于给模型一个"减压阀"：让 correctness 这个粗粒度信号更多地走 `score_residual`，`token_rewards`/`progress` 可以更专心地去拟合幻觉定位这种细粒度信号。这不是完全解决了张力，只是缓解——见第 5 节的开放问题。

### 3.4 PRISM 一致性（`prism_style_consistency_loss`，437-485 行）

对照 PRISM 论文 Eq.7 实现，公式忠实：

- `L_pos`：同 `semantic_id`、不同 `style_id` 的样本对，cosine similarity 拉近（`1 - cos`）。
- `L_neg`：不同 `semantic_id`、同 `style_id` 的样本对，margin-based 拉远（`relu(cos - margin)`）。
- `L_score`：CLIR 自己加的，同语义不同风格的样本对，最终标量 `scores` 也要接近（`(R_i - R_j)²`）。

**这个 loss 只在同一个 mini-batch 里找样本对**，所以能不能触发完全取决于 batch 里有没有同 `semantic_id`、不同 `style_id` 的行。这就是为什么需要 `SemanticGroupBatchSampler`（见 3.6）——如果用普通随机 shuffle 的 DataLoader，数据量一大，同语义的行很难凑到同一个 batch 里，这个 loss 长期会是 0（`positive_pairs.any()` 为 False 时函数会静默返回 0，不报错，容易被忽略）。

### 3.5 幻觉定位 + tail reward shaping（`hallucination_localization_losses`/`path_level_hallucination_mil`/`pseudo_onset_tail_loss`，487-621 行）

两档监督：

1. **强监督**：有明确的 `hallucination_onset`（第一个不被支持的 claim 出现的 token 位置，-1 表示整条轨迹没有幻觉）。`onset` 越界（≥ 该行实际有效长度）会直接 `raise ValueError`（515 行），不会静默吞掉。token-level BCE 监督 `hallucination_logits`，`t >= onset` 之后的所有 token 的目标标签都是 1。同时 `token_values`（不是单独的 `token_rewards`，见下面的已知问题）在 tail 区间被 MSE 拉向 `-negative_tail_margin`，非 tail 区间如果有 `token_advantage` 就拟合 `token_advantage`。
2. **弱监督**：只有整条轨迹级别的 `path_hallucinated`（是/否），用 noisy-or MIL（`path_prob = 1 - Π(1-p_t)`）做 BCE。再从 `hallucination_logits` 里推一个 pseudo onset（第一个 sigmoid 概率超过阈值的 token），对 pseudo tail 区间用更小的权重（`pseudo_tail_weight`，默认 0.1）做同样的 tail-margin 惩罚。

**已知问题**（README 里标成"仍需观察，不算 bug"）：`hallucination_localization_losses`/`pseudo_onset_tail_loss` 现在接收的是 `token_values = token_rewards + progress_score_weight * progress`（3.3 节提到的组合量），而 `progress` 同时又被 `progress_targets` 单独做回归监督（233-280 行 `loss()` 里两条独立的 if 分支）。toy 数据里 `progress_targets` 和 `token_advantage` 是同一份数据，所以两个 head 之间没有显式的分工约束，训练时只是被动地推出一个隐式拆分（大致是 `progress ≈ 目标值`，`token_rewards ≈ 目标值 - progress_score_weight*目标值`）。真实数据接进来、`progress_targets` 和 `token_advantage` 语义上真的不同之后，要重新想清楚这两个 head 各自该学什么、要不要拆开。

### 3.6 Dual-prior 定位（DPCL 部分，`dual_prior_losses`，623-708 行）

`key_prior_head`/`complete_prior_head` 各出一个 attention 分布（`masked_softmax` 归一化过，各自在有效 token 上加和为 1）。四块子 loss：

- `key`/`complete`：有外部 `key_prior_target`/`complete_prior_target`（0/1 或 soft 标签）时才算，纯 BCE。**没有标签时这两项严格是 0**（不是"接近 0"，是精确的 0，`test_prior_loss_is_zero_without_external_targets` 验证过）——早期版本这里是无标签也常开、用自己重构自己的方式退化成"complete_prior 学成均匀分布就能拿到 0 loss"的平凡解，已经改掉了。
- `distill`：key/complete 互相用 stop-gradient MSE 拉近（对应论文里 KSM/CSR 的交替蒸馏）。**只在 key 与 complete 都有标签覆盖的 token 上算**（`shared_prior_mask`），且比较的是"完整轨迹归一化后的原始概率"，不会对标签子集重新做一次 softmax/归一化（这也是踩过的坑，第二版实现曾经对子集重新归一化，导致标签只覆盖一小部分 token 时会人为放大那部分的权重，已经改掉）。
- `gate`：`token_reward_head` 的 gate 被正则化去贴近 `fused_prior = α·key_prior + (1-α)·complete_prior`（`prior_fusion_alpha`，默认 0.5），这一项不需要标签也会算（只要 key 和 complete 都存在，即 head 存在，不要求有 target），但 `fused_prior` 是 `.detach()` 过的，梯度只流向 gate，不会反过来污染两个 prior head。
- `reconstruction`：`complete_prior` 加权池化 token 特征后过一个小 MLP，回归一个**外部提供**的 `complete_reconstruction_target`（一个 `[hidden_dim]` 的向量，代表"这条轨迹该有的完整支持证据的某种摘要表示"，目前生成方式还没实现，见第 4 节）。早期版本这里没有外部 target，是拿 `complete_prior` 去重构自己池化出来的均值特征——这是一个平凡解陷阱：只要 `complete_prior` 学成均匀分布，`complete_prior` 加权池化就精确等于均值池化，reconstruction loss 直接归零，完全不需要学到任何有意义的定位。已经改成必须有外部 target 才会算这项 loss。

`phase` 参数（`joint`/`key`/`complete`）控制 DPCL 论文里"交替训练"的思路：`key`/`complete` 各自的 BCE 只在对应 phase 里算，`distill` 的两个方向也各自绑定到对应 phase，`gate`/`reconstruction` 目前不跟随 phase 切换（`gate` 靠 `.detach()` 天然不参与交替，`reconstruction` 绑定在 `complete` phase）。`train_clir.py` 默认用 `--prior_phase_mode alternate`：奇数 epoch 是 `key` phase，偶数 epoch 是 `complete` phase。

### 3.7 `SemanticGroupBatchSampler`（`src/clir_data.py:393-492`）

自定义 `BatchSampler`，按 `semantic_id` 分组，尽量把同语义不同风格的行打包进同一个 batch（每组内部再切成最多 `max(2, batch_size//2)` 大小的 chunk，避免一个特别大的组独占整个 batch），从而让 3.4 节的 PRISM 一致性 loss 在真实规模数据上也能稳定触发，而不是赌随机 shuffle 的运气。

`__len__()` 和 `__iter__()` 共用同一套 `_build_batches()` 逻辑（437-492 行），关键技巧是 `chunks.sort(key=len, reverse=True)`（461 行）——这一步让贪心装箱的"批次数量和每批大小"完全由 chunk 长度决定，和要不要 shuffle 无关，所以 `__len__(shuffle=False 的构造)` 和 `__iter__(shuffle=True 的构造)` 算出的批次数量必然一致。这是修复过的一个真实 bug（旧版本 `__len__` 直接用 `ceil(N/batch_size)` 估算，在很多同大小语义组和 `batch_size` 不能整除时会低估实际产出的 batch 数），已经用一个手工构造的回归测试 + 我做的 1200 次随机压力测试验证过。

`train_clir.py` 里 `--group_by_semantic_id`（默认开）控制训练集要不要用这个 sampler；验证集固定不用（`shuffle=False` 时直接退化成普通 `Subset` + `DataLoader`）。

## 4. 数据 schema（`src/clir_data.py` 的 JSONL 格式）

一行 JSONL = 一条生成的 trajectory。核心字段（`extract_metadata`，`clir_data.py:167-197`，支持多个别名，取第一个出现的）：

| 字段 | 别名 | 含义 | 谁在用 |
|---|---|---|---|
| `id` | - | 行的唯一标识 | 输出结果 |
| `query_id` | `candidate_group_id`、`prompt_id` | **Best-of-N 候选分组**：同一个原始问题的多条候选轨迹共享同一个 `query_id` | `score_clir.py` 选 Best-of-N |
| `hidden_states` / `hidden_states_path` | - | `[time, hidden_dim]`，二选一，支持 `.pt`/`.npy`/`.json` | 模型主输入 |
| `condition_states`/`condition_states_path`、`condition_embedding`/`condition_embedding_path` | - | query/context 条件输入，见 3.2 | 条件化模块 |
| `correctness` | `label`、`final_correct` | 0/1，整条轨迹最终答案对不对 | SWIFT-style BCE |
| `semantic_id` | `semantic_ids`、`augmentation_group`、`augmentation_group_id`、`group_id` | **一致性分组**：同一个原始语义、不同风格改写的行共享同一个值 | PRISM consistency + `SemanticGroupBatchSampler` |
| `style_id` | `style_ids`、`augmentation_style`、`rewrite_style`、`domain_id`、`domain`、`style` | 风格/领域标签，用于区分同语义的不同改写 | PRISM consistency |
| `hallucination_onset` | `hallucination_start`、`onset` | 第一个幻觉 token 的下标，-1 = 无幻觉 | 强监督幻觉定位 |
| `path_hallucinated` | `hallucinated`、`hallucination` | 0/1，整条轨迹级别的幻觉标签 | 弱监督 MIL |
| `token_advantage` | `token_advantages`、`advantages` | 逐 token 的 progress/advantage 目标值 | tail reward MSE |
| `progress_targets` | `progress`、`progress_target` | 逐 token 的 progress head 目标值 | progress head 回归（注意和 `token_advantage` 目前语义重叠，见 3.5） |
| `key_prior_target` | `key_prior` | 逐 token 0/1 或 soft 标签，最关键证据 | dual-prior key BCE |
| `complete_prior_target` | `complete_prior` | 逐 token 0/1 或 soft 标签，完整支持证据 | dual-prior complete BCE |
| `complete_reconstruction_target` | `csr_target` | `[hidden_dim]` 向量，完整支持证据的摘要表示 | dual-prior reconstruction（3.6 节提到的"必须外部提供"） |

**重要**：`query_id`（Best-of-N 分组）和 `semantic_id`（一致性增强分组）是两个不同粒度的概念，之前踩过混用的坑（`query_id` 曾经会 fallback 到 `semantic_id`），现在已经拆开，`query_id` 只 fallback 到 `candidate_group_id`/`prompt_id`。真实数据构造时不要把这两个字段填成一样的值，除非它们本来就该一样（toy 数据里刚好一样，容易造成误解）。

## 5. 已知的开放问题（不是 bug，是要做研究判断的地方）

1. **`token_values` 把 `token_rewards` 和 `progress` 合并后做幻觉监督，但 `progress` 又被单独回归**（3.5/3.3 节已详述）。等真实数据接入、`progress_targets` 和 `token_advantage` 真正不同之后需要重新设计。
2. **`_condition_token_features` 里"有 condition 的行过 LayerNorm、没 condition 的行不过"造成的轻微分布不一致**（3.2 节）。如果真实数据里"有没有 condition"本身有意义，需要评估这个不一致有没有影响。
3. **`gate`/`reconstruction` 两项 dual-prior loss 不跟随 `prior_phase` 交替**（3.6 节），是否需要让它们也分 phase，目前没有定论。
4. Best-of-N 选择目前是纯 pointwise：训练目标只有单条轨迹的 BCE + 各种辅助 loss，没有 pairwise/listwise 目标。SWIFT 原论文里 pairwise/DPO/InfoNCA/NCA 都试过，效果和 BCE 接近但没有明显更好，所以优先级不高，但如果要和 SWIFT 做严格对齐实验，这块需要补。

## 6. 目前完全没做、下一步必须做的事（按优先级）

**P0（不做这个，后面所有实验都是空中楼阁）**

1. **写一个从真实 LLM 抽取 hidden states 的脚本。** 输入 query/context/trajectory，跑一遍生成模型（teacher forcing 或真实生成），把 SWIFT 用的那种"每层 hidden state 拼起来"的 per-token 向量存下来，同时可选地存 query/context 的 condition hidden states。可以参照 SWIFT 论文 4.2 节的做法（`h_t = [h_t^1; ...; h_t^L]`，拼接所有层）。目前 `create_toy_clir_data.py` 完全是随机数占位，这一步做完才有真实数据可用。
2. **接一个 correctness/verifier 标注流程**，给每条 trajectory 打 `correctness` 标签（答案对不对，可以用规则化的 answer checker，比如 MATH/GSM8K 那种数值比对）。这是 SWIFT baseline 复现的最低要求，不需要幻觉/一致性这些 CLIR 独有的标签也能先把 SWIFT-only baseline 跑起来。

**P1（跑通 SWIFT baseline 之后，逐步加 CLIR 的三个模块）**

3. **LLM rewrite augmentation 流水线**：给每条原始 trajectory 生成 K 个"改写题干风格/长度/context 顺序/domain 表面形式"的版本，用 verifier 过滤掉改变了答案或证据关系的改写，产出 `semantic_id`/`style_id`/`domain_id` 元数据。做完这一步才能真正测试 PRISM 一致性 loss 有没有用。
4. **幻觉标注流水线**：至少要有 path-level 的 `path_hallucinated`（弱监督），有条件的话再加 token-level 的 `hallucination_onset`（强监督）。可以用 verifier/LLM judge 对着 query+context 检查每一步 claim 有没有支持依据。
5. **dual-prior target 生成**：`key_prior_target`/`complete_prior_target`/`complete_reconstruction_target` 目前都要外部提供，需要设计怎么从 query/context entailment 或 verifier score 生成弱 prior（可以参考 `docs/proposal.md` 里 Dual-Prior Localization View 一节的思路）。

**P2（跑完第一轮实验之后再看）**

6. 把训练目标从纯 pointwise BCE 扩展成 pairwise/listwise（DPO/InfoNCA/NCA），和 SWIFT 论文附录 E.1 的消融对齐。
7. 在真 GPU + 完整依赖环境跑一遍 end-to-end small-scale training，看这套方法在 Best-of-N accuracy / 幻觉率 / worst-augmentation accuracy 上到底有没有比 SWIFT-only backbone 好（`docs/proposal.md` 的 Evaluation Plan 一节已经定义好了要看哪些指标、哪些消融）。
8. 解决第 5 节列的开放问题（token_values 拆分、condition LayerNorm 不一致、gate/reconstruction 要不要分 phase）。

## 7. Bug 修复历史（给想知道"为什么现在长这样"的人）

按时间顺序，每一条都是真实发生过、被发现并修复的问题，不是假设性的：

| # | 问题 | 根因 | 修复方式 | 验证状态 |
|---|---|---|---|---|
| 1 | Query/context 条件化对 dual-prior 数学上零效果 | 池化成常数向量加到所有 token 上，softmax 对常数偏移平移不变 | 改成逐 token cross-attention（3.2 节） | 有专门回归测试 `test_condition_changes_prior_distribution`，我复核过 |
| 2 | dual-prior 的 distill/reconstruction 无标签时也常开，且 reconstruction 有平凡解（均匀 attention 可以零成本重构均值特征） | reconstruction target 用的是自己池化出来的均值特征，不是外部真实标签 | 无标签时严格为 0；reconstruction 改成必须有外部 `complete_reconstruction_target` | 有 `test_prior_loss_is_zero_without_external_targets`，我复核过 |
| 3 | PRISM 一致性 loss 依赖随机 shuffle 让同语义样本凑到同一 batch，数据量大时基本不触发 | 普通 `DataLoader(shuffle=True)` 不保证分组 co-occurrence | 加 `SemanticGroupBatchSampler`（3.7 节） | 有专门测试，我复核过 |
| 4 | `hallucination_onset` 越界时静默当成"无幻觉"处理 | 没有边界检查 | 越界直接 `raise ValueError` | `test_invalid_hallucination_onset_raises`，我复核过 |
| 5 | `query_id` 会 fallback 到 `semantic_id`，混淆 Best-of-N 分组和一致性分组 | fallback 链设计不当 | 拆开两条独立的 fallback 链（第 4 节表格） | 静态检查确认 |
| 6 | `SemanticGroupBatchSampler.__len__()` 低估实际 batch 数 | `__len__` 用 `ceil(N/batch_size)` 估算，`__iter__` 的贪心装箱在特定组大小/batch_size 组合下会产出更多、更碎的 batch | `__len__`/`__iter__` 复用同一套构造逻辑 + 排序让计数与 shuffle 无关（3.7 节） | 手工回归测试 + 我做的 1200 次随机压力测试，全部一致 |
| 7 | 两个新 `RewardConfig` 字段（`condition_attention_temperature`/`progress_score_weight`）没接到训练 CLI | 加字段时漏了改 `train_clir.py` | 补上 `--condition_attention_temperature`/`--progress_score_weight` 参数 | `test_train_cli_exposes_new_reward_config_fields` + 我端到端跑过确认写进了 checkpoint |
| 8 | dual-prior 的 distill/gate-prior 在标签部分覆盖时，对标签子集重新做 softmax 归一化，会人为放大子集内的相对权重，失真 | 用 `normalize_attention(x, shared_prior_mask)` 而不是直接比较完整轨迹归一化后的值 | 直接用 `outputs["key_prior"]`/`outputs["complete_prior"]`/`outputs["fused_prior"]`（已经是完整轨迹归一化过的），只在 `attention_mse` 内部做子集选择，不重新归一化 | `test_dual_prior_partial_mask_preserves_full_attention_mass`，我手工验算过数值 + 复核过 |
| 9 | 条件化模块（`condition_query`/`key`/`value`/`condition_fusion`/`complete_reconstructor`）参数量是 `hidden_dim` 的平方级，真实 LLM 拼接维度下会爆炸到约 1827 亿参数 | 这几层都直接用 `Linear(hidden_dim, hidden_dim)`，toy 数据 `hidden_dim=8` 完全看不出问题 | 加一个独立的瓶颈维度 `condition_attention_dim`（新 `RewardConfig` 字段，默认 256），相关层先降维到这个小维度再算，最后投影回 `hidden_dim`（3.2 节坑 2） | `test_condition_module_params_scale_linearly_with_hidden_dim` + 我实测了 hidden_dim=135,168（Llama-3.1-8B 拼接全部层的量级）下从约 1827 亿降到约 2.79 亿参数 |

**目前没有已知的、未修复的逻辑 bug。** 第 5 节列的是设计层面的开放问题，不是逻辑错误。

### 7.1 依赖版本对齐 SWIFT（不是 bug，是基础设施补全）

之前仓库没有 `requirements.txt`，只在 README 里写了一行 `pip install torch numpy pytest`，没有锁版本。因为 CLIR 的架构是照着 SWIFT 参考实现的，现在把 `torch`/`numpy` 的版本对齐到 SWIFT 官方仓库（[aster2024/SWIFT](https://github.com/aster2024/SWIFT)）的 `requirements.txt`，减少以后接真实 SWIFT-style hidden states 时的行为差异风险。核对过程中发现 SWIFT 自己的 `requirements.txt` 里 `numpy==2.2.6` 和 `accelerate==0.32.1` 两个 pin 互相冲突（`accelerate==0.32.1` 要求 `numpy<2.0.0`），照抄会直接装不上；本仓库的 `requirements.txt` 把 `accelerate` 换成了 `>=1.0.0` 来绕开这个冲突，其余核心版本不变。完整版本列表、哪些包现在用得上/哪些要等第 6 节 P0 任务做完才用得上，见仓库根目录 `requirements.txt` 的注释，这里不重复贴。

### 7.2 条件化模块参数量爆炸（bug #9 的详细版本）

这个问题是这样被发现的：有人问"要不要把训练代码改成多卡"，因为担心"大模型可能跑不起来"。查下来发现问题根本不在要不要多卡，而是 `_condition_token_features` 里 `condition_query`/`condition_key`/`condition_value`（当时还是 `Linear(hidden_dim, hidden_dim)`）、`condition_fusion` 第一层（`Linear(4*hidden_dim+1, hidden_dim)`）、`complete_reconstructor`（两层 `Linear(hidden_dim, hidden_dim)`）全都是 `hidden_dim` 的平方级参数量。

`hidden_dim` 在这个项目里指的是 SWIFT 论文里 `Ld`（`L` 层 transformer 拼接、每层维度 `d`）这种量级，不是随便一个小数字——SWIFT 论文 Table 3 报告 Llama-3.1-8B（33 层 × 4096 维）下它自己的 reward model 是 `2.7×10⁵` 参数（因为 SWIFT 的 `W_SWIFT ∈ R^{2×Ld}` 是纯线性的）。我用 `sum(p.numel() for p in model.parameters())` 实测了 CLIR 在修复前的架构下、同样 `Ld=135,168` 这个量级，参数量是 **约 1827 亿**，比 SWIFT 自己在同样输入维度下的模型大了约 **67.6 万倍**，比 GPT-3（1750 亿）还大——完全不是"轻量级 reward model"，而且这个规模的模型不管有多少张 GPU 都不该用几千条样本去训（灾难性过拟合），所以"改成多卡"从一开始就是问错了方向的解法。

修复：加一个独立的 `condition_attention_dim`（`RewardConfig` 新字段，默认 256），把 3.2 节提到的几层全部通过这个小维度做瓶颈（细节见 3.2 节坑 2 的描述）。修完之后同样 `Ld=135,168` 实测是 **约 2.79 亿参数**，是修复前的约 1/655，能在单卡甚至 CPU 上正常训练。加了 `test_condition_module_params_scale_linearly_with_hidden_dim` 防止这个问题再次出现——它检查 `hidden_dim` 变 8 倍时参数量不能涨超过 16 倍（线性预期约 7.5 倍，平方级会是 64 倍），阈值留了充分余量。

## 8. 怎么跑起来（最小验证闭环）

```bash
# 1. 装依赖（torch/numpy 版本对齐 SWIFT，见 requirements.txt 注释）
pip install -r requirements.txt

# 2. 跑测试，确认环境没问题
pytest tests/test_clir_smoke.py    # 应该 11/11 通过

# 3. 生成 toy 数据（纯随机数，只用来验证管线通不通，不能用来判断方法有没有效）
python examples/create_toy_clir_data.py \
  --output_jsonl examples/toy_clir.jsonl \
  --feature_dir examples/features \
  --hidden_dim 8

# 4. 训练
python train_clir.py \
  --train_jsonl examples/toy_clir.jsonl \
  --output_model outputs/clir_toy.pt \
  --hidden_dim 8 --projection_dim 4 --batch_size 4 --epochs 3 --lr 1e-3 \
  --group_by_semantic_id --prior_phase_mode alternate \
  --condition_attention_temperature 1.0 --progress_score_weight 0.5

# 5. 打分 + Best-of-N
python score_clir.py \
  --input_jsonl examples/toy_clir.jsonl \
  --model outputs/clir_toy.pt \
  --output_jsonl outputs/clir_toy_scores.jsonl \
  --batch_size 4
```

如果这五步任何一步跑不通，说明环境或者代码本身出了问题（和"方法有没有效"无关），应该优先排查。所有命令行参数的完整列表和说明见 `README.md` 的"运行代码"一节。

## 9. 给下一个接手的人/AI 的建议

- 不要重新审查已经在第 7 节列表里的问题，除非你怀疑修复本身不彻底——可以直接去看对应的测试是不是还在、还过不过。
- 改 `src/consistency_localized_reward.py` 或 `src/clir_data.py` 之前，先跑一遍 `pytest tests/test_clir_smoke.py`，改完再跑一遍，别指望"看起来对"就是真的对（第 6 项 bug 就是一个纯静态阅读很难发现、必须实际跑数据才能验证的例子）。
- 按"维护规则"（`README.md` 最后一节），改完代码要同步更新 `README.md` 的对应章节；如果改动大到影响这份交接文档里描述的架构/状态，也请同步更新这份 `docs/handoff.md`，不要让它过时——过时的交接文档比没有更糟。
- 第 6 节的 P0 任务（真实 hidden state 抽取 + correctness 标注）是当前最大的阻塞点，建议下一步直接从这里开始，而不是继续打磨模型代码细节。
