# CLIR 第四次审查处理记录

处理日期：2026-08-15
输入：`docs/code_review_panzhixin_fourth_change.md`

本轮只修复审查中影响可运行性、实验有效性和 provenance 的问题；没有启动正式 Stage 1B GPU
矩阵，也没有读取 `pilot_test`。v3 保持冻结，由新发布的 v4 协议取代。

## 逐项结论

| 审查项 | 处理 | 结果 |
|---|---|---|
| P0：健康失败导致汇总死锁 | 增加显式 `--allow-failed-cells`；train/score/evaluate 均持久化失败证据；summary 只接受协议/阈值/cell 身份匹配的证据 | 已修。部分或全部健康失败都能产出 diagnostic-only 矩阵；未知/未运行 cell 仍拒绝 |
| P0：final 失败使 epoch snapshot 不可用 | snapshot gate 改读 metrics 中目标 epoch 的 `training_health` | 已修。snapshot 与 final run status 解耦，但仍只允许诊断 |
| P1：两道门漏掉 query 内排序坏 run | evaluator 增加 correct-vs-incorrect micro pairwise accuracy，tie=0.5；v4 阈值 0.60 | 已修。报告先落盘，门失败后退出非零 |
| P1：prior gate 用绝对距离 | 改为 `(prior_bce - observed_bce) / prior_bce` 的一侧改善门 | 已修。比常数先验更差的模型必失败 |
| P1：2% 硬阈值余量过小 | v4 冻结为至少 1% 相对改善 | 已修。v3 不原地改动 |
| P1：历史 v1 不能直接重评 | 保留严格 provenance，不为历史 artifact 降低 evaluator 契约 | 记录为操作后果。若需要 v1 当前口径诊断，必须重新打分；本轮未消耗该 GPU 成本 |
| P1：resume 在 run provenance 中不可见 | run/checkpoint 增加 `start_epoch`、`resumed_from` 和 `training_segments` | 已修。源路径/SHA256/epoch/device 均可审计 |
| P2：device 未 pin | CLI device 纳入 resume pinned args，同时记录并校验 resolved device | 已修。跨设备或 resolved device 不可判定的 resume 拒绝 |
| P2：per-epoch health 无消费者 | launcher 的 snapshot 路径消费目标 epoch health | 已修 |
| P2：工作树脏阻止 formal execution | 保留 clean-git/formal-preflight 门，不绕过 | 正确的程序性阻塞；本轮代码需先完成用户审计、提交，再决定是否授权正式运行 |

## 额外的证据链加固

- complete summary 也接收冻结 protocol config，并核对每份 evaluation 的 protocol SHA256、score
  threshold、observed score distribution 和 ranking threshold；旧 evaluation schema 或未启用 ranking
  gate 的报告不能混入 v4。
- failure summary 不把“文件缺失”当作失败。train failure 必须有匹配 run record；score failure 必须
  有 sidecar 且 checkpoint hash 对应本 cell；ranking failure 必须有匹配 seed/variant/protocol 的
  evaluation report。
- train health schema 使用 `minimum_relative_improvement`，保留旧 v3 threshold 名仅用于读取冻结
  历史配置，不再把“离先验足够远”当作健康。
- scoring health sidecar 在失败前原子写出；通过后才发布 scored JSONL。
- v4 loader 冻结三项阈值和 failed-cell policy。要改门禁必须发布新协议版本。

## 仓库收敛

- 删除 `scripts/diagnose_stage1b.py` 及其重复测试。候选顺序、排序健康、矩阵一致性和失败 provenance
  已由 launcher/evaluator/summary 的强制路径覆盖，不再维护第二套诊断实现。
- 删除 `select_best_of_n` 和 score 阶段的 `*_selected_best_of_n` 字段。Best-of-N 只由
  `evaluate_clir.py` 按冻结 prefix/tie 口径选择。
- README、handoff 和 runbook 从重复 changelog 收敛为当前状态、最短操作和保护边界；历史协议与审查
  报告保留为可追溯证据。

## 协议版本

- 新机器协议：`configs/stage1b_validation_v4.json`
- 新人类协议：`docs/stage1b_v4_protocol.md`
- 输出根：`run_artifacts/stage1b_v4`
- v3：冻结、未正式执行、由 v4 取代
- v4：prepared，等待代码审计和明确 formal-run 授权

## 验证

当前已完成：

- 相关 Python 入口 `py_compile`；
- `git diff --check`；
- 训练恢复、launcher、评分失败证据、排序门和完整/不完整 summary 的 52 个定向测试通过；
- 完整测试套件 `133 passed`；
- `pip check` 无依赖冲突；
- v4 JSON 与全部 CLI 通过，真实输入 dry preflight 复核 train 4096 / validation 8000、
  component/manifest/hash/payload 和零监督覆盖；
- launcher 正确生成 9 条 train、9 条 score、9 条 evaluate 和 1 条 summary 命令，未执行 GPU 阶段。

这里的验证只证明代码和协议契约可执行，不构成 Stage 1B v4 效果证据。

## §13 追加审查的后续修复

审查报告后来追加的 §13 指出了两个在上述记录之后发现的问题。当前代码已处理，但历史 Stage 1B v4
artifact 不原地修改：

- 训练健康门不再读取 epoch 内边训练边累计的平均 BCE。每个 epoch 保存前，final checkpoint 会在
  完整 train split 上以 eval/no-grad 模式重新评估；`clir-training-health-v3` 明确记录
  `measurement=checkpoint_full_train_split_no_grad_eval`，并校验 evaluation example count。
- 普通 shuffle loader 改用显式 `(seed, epoch)` sampler；DataLoader worker/base-seed generator 与
  reward-model 全局 RNG 分离。覆盖 `num_workers=4, pin_memory=true, persistent_workers=true` 的
  中断/续训回归对模型 state、optimizer state 和 metrics 做递归 bit-exact 比较并通过。

历史 v4 summary 的 1/9 仍是冻结证据；只读重算的 4/9 只用于纠正诊断口径。任何重新发布都必须使用
新协议、新提交和新 output root。
