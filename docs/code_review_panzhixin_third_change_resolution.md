# 第三轮代码审查闭环记录

对应原报告：`docs/code_review_panzhixin_third_change.md`。闭环日期：2026-08-14。

## 结论

第三轮报告中的可执行代码问题均已修复并有回归测试。报告指出的“真实数据没有 CLIR 专属
监督”不是能靠伪造字段修掉的代码 bug：它已被机器审计并隔离为实验语义门禁。Stage 1B v3
因此只允许作为 outcome-only capacity/optimization control；真正的 mechanism experiment 仍需
先生成、人工复核并冻结外部 rewrite/verifier/prior annotation。

历史 Stage 1B v1 继续只算 pre-audit diagnostic，v2 在 0 个完成 epoch 时废止。当前没有任何
Stage 1B v3 正式效果数字，也没有读取 `pilot_test`。

## 逐项闭环

| 原报告 # | 问题 | 处理结果 | 主要验证 |
|---:|---|---|---|
| 1 | 8 个零 epoch `run.json` 卡死矩阵 | `train_clir.py --force` 只接受 `failed + completed_epoch=0 + 无 checkpoint/metrics`；v3 使用新目录；launcher 对 3×3 矩阵原子预检 | force 正/反例与 launcher tests |
| 2 | CLIR 2/3 seed 塌缩到类别先验 | v3 打开 grad clip 1.0；冻结 final primary + 每 epoch immutable snapshot；最终 train BCE 距常数先验 2% 门与 validation score population std 0.1 门拒绝坏 run | review collapse 数值回归、epoch checkpoint、score gate tests |
| 3 | 数据没有 CLIR 专属监督 | v3 明确降级为 outcome-only；preflight 复算 10 字段/7 组件零覆盖；新增外部 annotation token-ID 绑定、逐字段 provenance、非覆盖 merge 与 coverage audit | v5 train/validation 实测均 0 个 eligible component；监督契约 tests |
| 4 | BF16 score 随 batch 形状翻转 | v3 scoring 固定 FP32；provenance 记录 AMP/compute dtype、score distribution 与门阈值，evaluator 重新计算并核对 | score/evaluation provenance tests |
| 5 | checker 取答案 span 最后数字 | 发布 `clir_gsm8k_numeric_v5`：equality RHS、governed/first number、mixed-number 与 compound-duration guard | validation 真修复 15 条 `0→1`、0 条 `1→0`；假阳性回归测试 |
| 6 | scoring provenance 只写不读 | evaluation 要求每行完整且完全一致的 v2 provenance，并验证 checkpoint/variant/input/batch/dtype/protocol/distribution；summary 再做矩阵级 gate | 混 checkpoint、篡改 distribution、跨 cell checkpoint 复用拒绝测试 |
| 7 | 旧 checkpoint 因新增 pinned args 无法 resume | 缺失字段按当时真实默认值回填；仍严格拒绝实际训练参数漂移 | legacy resume 与 changed-arg tests |
| 8 | 冻结协议无可执行入口、CLI 默认漂移 | `scripts/run_stage1b_validation.py` 从 v3 配置生成精确 train/score/evaluate/summary 命令，执行必须 clean git + formal preflight + 显式 cell | launcher contract + 真实 manifest dry preflight |
| 9 | README/handoff toy 路径双前缀崩溃 | toy generator 将 manifest/feature 目录解析成绝对路径；文档统一使用绝对 `RUN_DIR` | 相对目录回归测试 + CPU create/train/score 闭环 |
| 10 | negative MIL 随长度线性放大 | negative path loss 按有效 token 数归一化；exact-zero survival 增加 finite fallback | 长序列 loss/gradient tests |
| 11 | prior distill/gate 在真实 T 下趋近惰性 | `attention_mse` 改为每 trajectory 求 token error 总和，再跨 trajectory 平均；仍不对子集重归一化 | 量级与 partial-mask 手算回归 |
| 12 | Flat encoder 的大 LayerNorm 无分块 | flat 与 layer-axis encoder 都在 2^31−1 元素以下分块 raw normalize/projection | 真实宽度与强制分块 forward/backward tests |
| 13 | 历史训练 manifest 只在 `/tmp` | v1/v2 local-path manifests 已逐字节归档到对应 `run_artifacts/.../manifests/`，原 SHA256 不变 | archive report/hash 核对 |
| 14 | vLLM resume/分片受 batch 成员影响 | base seed 与 `query_id` 经 SHA256 派生 request-local seed，vLLM 使用逐 request `SamplingParams` | seed 纯函数与生成参数 tests |
| 15 | `num_workers=0` 与重复 finite scan 放大 I/O | v3 固定 4 workers、pin/persistent workers；725.8 GB feature gate 通过后跳过重复 payload finite scan，loss/gradient finite gate 保留 | launcher command contract + feature report gate |
| 16 | summary 接受错误 historical subset | summary 只接受 evaluation v2、`first_k_by_vllm_completion_output_index` 与冻结 index policy | historical subset/policy rejection tests |
| 17 | ties、版本漂移、metrics 发布顺序、少 GPU resume | 显式 tie 数与最低 index 规则；checker v5 新版本/新 label hash；checkpoint 先写且可恢复缺失 secondary artifacts；CUDA RNG 只恢复可见设备交集 | tie、checkpoint recovery、CUDA-state tests |

## 本轮继续推进的研究基础设施

`src/clir_supervision.py`、`scripts/merge_clir_supervision.py` 和
`scripts/audit_clir_supervision.py` 建立了真实辅助监督的入口：

- annotation 必须绑定非空 `id/query_id` 和 canonical `output_token_ids` SHA256；
- token targets 必须与 trajectory token 数严格相等，全部 finite；prior 值域固定为 `[0,1]`；
- onset 必须与 path label 同时存在并一致；reconstruction target 必须定宽；
- 缺失标签保持缺失，不补零，不从 correctness 推导；
- 多批 annotation 可以追加非重叠字段，每个字段保留唯一 source/protocol/artifact provenance；
- coverage 报告区分字段 rows/tokens、consistency pairs、path classes、positive onset、joint prior、
  reconstruction dimensions 和最终 component eligibility。

完整使用契约见 `docs/clir_supervision_protocol.md`。这一步完成了 ingestion/audit，不代表真实
rewrite/verifier 标签已经生成。

## 最终验证

- `/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python -m pytest -q`：
  **124 passed in 93.35s**。
- `python -m py_compile`：所有变更入口通过。
- 所有相关 CLI `--help`：通过。
- `python -m pip check`：`No broken requirements found.`
- CPU toy `create -> train(1 epoch) -> score`：6/6 行闭环通过；重复 score 默认拒绝覆盖。
- Stage 1B v3 真实输入 dry preflight：train 4096、validation 8000、全部 payload 存在、组件 hash/
  manifest hash/feature gate/v5 relabel protected fields/辅助监督零覆盖均通过。当前 protocol file
  SHA256 为 `b43ff52992d393002bc54199e81829102a62123c6b6cdeb0ed4c69c8a8378c23`。
- formal preflight 未执行：工作树尚未形成 clean commit，且 3×3 GPU 正式开销尚未获得明确授权。

## 仍然开放的研究任务

下一优先级不是重复生成 Stage 1B 特征，也不是直接烧 v3 GPU，而是在 train/validation 上构建
并审核真实 annotation：semantics-preserving rewrite、path/onset verifier、prefix progress/
advantage、key/complete prior 和独立 reconstruction target。质量与 coverage 都通过后发布新的
mechanism protocol；在此之前保持 `pilot_test` 未读取。
