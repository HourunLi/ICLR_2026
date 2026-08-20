# Joint Shared-Gradient Interaction Audit v1

最后更新：2026-08-20

## 1. 结论

冻结的 no-update shared-gradient audit 已完成。结果不支持给整个 shared encoder 直接套一层统一的
PCGrad-style 修复：H 与整体 prior 没有稳定同批冲突，C 与整体 prior 的平均方向在初始化和 JP epoch 5
反而都同向。

真正出现的信号有三类：

1. C 与原始 mutual-distillation 分支存在稳定的跨 stream 反向压力，shared cosine 为
   `-.400/-.536`；
2. C 与 H 也存在稳定跨 stream 反向压力，shared cosine 为 `-.110/-.200`；
3. H 与整体 prior 虽不冲突，但在 condition attention/fusion 子空间中，两个模型状态的 aggregate cosine
   都低于 `-.05`；JP 状态下 H/C 的 shared gradient norm 又分别是 residual prior-total 的 `2.33×/3.20×`。

这里的 C↔prior 与 C↔H 不能称为“同批梯度冲突”：冻结 single-stream 的五个 epoch 中，它们的 active
batches 重叠数始终是 0。普通 per-batch PCGrad 根本看不到这两条梯度，直接采用既没有被本审计授权，也不
会处理已观察到的 cross-step pressure。

工程状态为 `completed_no_update_diagnostic`，两个模型状态的参数 checksum 前后完全一致，且没有留下
`.grad` buffer。证据等级为 `small-scale real` diagnostic，不是机制有效性或一般因果结论。

## 2. 冻结测量口径

审计在两个模型状态上运行：

- seed-42 deterministic initialization，checksum 与既有真实 feature routing audit 完全一致；
- 冻结 JP epoch-5 checkpoint，SHA256 为
  `a392ad6ad8ccbe020e6a10835463c15829e3684f0a2b9213e6bcefe78666df5c`。

覆盖数据不是抽样：

- 全部 48 条 mechanism-train rows，按每 4 条共享一次 forward、每条独立算 loss 后等权平均，共 12 个
  controlled batches；
- 全部 54 条 consistency rows，覆盖 27 positive pairs、26 negative pairs 和 epoch-1 的 14 个 active
  batch groupings；
- 冻结 stream 的五个 epoch 均重新检查，mechanism active batches 为 `48/47/48/45/45`，C 均为 `14`，
  H/prior 与 C overlap 均为 `0`。

所有 gradient 都是在 BF16 autocast forward 后、global-norm clipping 与 AdamW preconditioning 之前测得的
raw effective gradient。原始 prior 权重保持 `key=1`、`complete=1`、双向 stop-gradient mutual=`.25`、
shared-gradient gate=`10`，outer prior=`1`；没有更改项目方法。

预注册的 aggregate cosine 阈值是 `-.05`。同批 stable conflict 还要求每个模型状态至少 70% controlled
batches 为负、中位数为负，并且两个状态都满足。跨 stream 只按两个状态的 aggregate cosine 分类，同时
强制标记为 `simultaneous_batch_conflict=false`。

## 3. 主要结果

### 3.1 同批比较

| Pair | Init cosine | JP cosine | 冻结分类 |
|---|---:|---:|---|
| H ↔ prior total | `-.024` | `+.126` | no stable conflict |
| H ↔ key direct | `-.037` | `+.209` | no stable conflict |
| H ↔ complete direct | `+.010` | `+.127` | no stable conflict |
| H ↔ mutual `.25` | `+.037` | `-.069` | state-specific conflict |
| H ↔ gate `10` | `-.113` | `-.030` | state-specific conflict |
| final ↔ H（mechanism rows） | `+.117` | `+.003` | no stable conflict |
| final ↔ prior total（mechanism rows） | `-.114` | `-.223` | state-specific conflict |
| final ↔ C（C rows） | `+.058` | `+.694` | no stable conflict |

H↔prior-total 在 JP 上不但没有反向，还是正向；因此 JPH 的 key AP 下降不能用“两个总目标直接打架”解释。
final↔prior 本身在没有 H/C 的 JP 路径中就有明显反向分量，尤其集中在 input projection；它是原 baseline
已有的 tension，不能事后归罪于新 auxiliary。

### 3.2 跨 stream 比较

| Pair | Init cosine | JP cosine | 冻结分类 |
|---|---:|---:|---|
| C ↔ prior total | `+.133` | `+.268` | no stable opposition |
| C ↔ key direct | `+.042` | `+.125` | no stable opposition |
| C ↔ complete direct | `+.165` | `+.233` | no stable opposition |
| C ↔ mutual `.25` | `-.400` | `-.536` | stable cross-stream opposition |
| C ↔ gate `10` | `-.079` | `+.661` | state-specific opposition |
| C ↔ H | `-.110` | `-.200` | stable cross-stream opposition |

C 与 total prior 同向，是因为 direct key/complete 与 gate 的合成方向盖过了 mutual 分支；不能因为 total
cosine 为正就忽略 mutual 的局部反向。已有 held-out diagnostic 与这条方向一致：JPC 的 symmetric prior
attention MSE 为 `.004040`，比 JP 的 `.002530` 更高。不过 JALL 又降到 `.000190`，说明 H×C×prior 的最终
效应高度非线性，不能据此删除、降权或替换原 mutual-distillation。

### 3.3 梯度范数发生了阶段反转

| Objective（shared-all norm） | Init | JP epoch 5 |
|---|---:|---:|
| final on mechanism rows | `8.000` | `8.158` |
| H | `16.903` | `1.250` |
| C | `.516` | `1.713` |
| prior key | `21.834` | `.127` |
| prior complete | `16.905` | `.442` |
| prior mutual `.25` | `.0054` | `.150` |
| prior gate `10` | `.083` | `.159` |
| prior total | `27.753` | `.536` |

初始化时 prior direct gradient 很大，C 几乎可以忽略；JP 拟合 prior 后，residual prior-total norm 降到
`.536`，而未训练的 H/C 到达时分别是其 `2.33×/3.20×`。这更像“后期 auxiliary 更新相对 residual prior
过强且含大量正交分量”，而不是一个稳定的全局负 cosine。它可以解释为什么 loss 都接通、总方向甚至同向，
held-out key AP 仍可能被表示漂移伤害；但 raw norm 尚未包含 AdamW moments 和 clipping，不能直接换算成
真实 parameter displacement。

### 3.4 分层信号

- H↔prior-total 的 shared-all 为 `-.024/+.126`，但 condition-all 为 `-.117/-.107`，condition-attention
  为 `-.139/-.210`，condition-fusion 为 `-.115/-.106`。这是稳定的局部 aggregate opposition，但没有
  预注册的逐 batch subgroup fraction，暂不足以授权 targeted gradient surgery。
- C↔mutual 的主要反向集中在 input encoder，尤其 input projection（`-.513/-.545`）。
- C↔H 在 JP input projection、transformer block 0/1 分别为 `-.367/-.459/-.536`，而 condition branch
  已转为正向；所以也不能把它简化成全模型统一冲突。

## 4. 对下一步的裁决

本审计否定了“先给 shared encoder 全面做 gradient projection”这一直接跳步。更合适的下一格是只针对 JPH
做 supervision-aware packing/schedule 对照，复用冻结 JPH 作为 control：

- 把 48 条 mechanism rows 从当前每 epoch 45–48 个 singleton-scattered active batches，集中成 12 个
  4-row mechanism batches；
- 每条 row 每 epoch 仍恰好一次，数据、初始化、5 epochs、所有 loss 权重和原始 prior 实现都不变；
- C 继续关闭，从而只回答 H+prior 的 batching/schedule 是否导致 key/H failure；
- 仍不访问 `pilot_test/final_test`，不扩 seed，不自动调权重。

这个比较必须诚实命名为 packing/schedule test，而不是“纯 packing”：虽然 row exposure 与写在 config 里的
loss weight 不变，auxiliary-active optimizer steps 会从约 48 降为 12，因此有效优化预算也随之改变。如果它
恢复 H/key，下一轮还需要专门区分 grouping 与 effective budget；如果它仍失败，才有理由进一步考虑 condition
branch 的 targeted conflict repair。

C 的 cross-stream mutual opposition 暂不修。普通 per-batch PCGrad 无法处理从不共批的目标；若以后处理，
必须另行比较 co-scheduling、gradient memory 或明确的 multistream 方案，并完整保留 direct、双向
stop-gradient mutual 和 shared-gradient gate。

## 5. 可复核 artifact

- 冻结协议：`configs/joint_gradient_interaction_v1/audit_protocol_v1.json`
- 机器结果：`configs/joint_gradient_interaction_v1/audit_result_v1.json`
- 执行代码：`scripts/audit_joint_gradient_interactions_v1.py`
- 冻结代码 commit：`74f758366ae4a070f042f47332aaf2e8ed8701e4`

协议 SHA256 为 `3595dd8435dd2e889158da664775513ac1e251145547168653e08d1f0efbe45f`；结果
SHA256 为 `de368ef395de9032afd5453479ee71122328502e4d3ab72ec62a959bcdb8144b`。审计没有执行 optimizer step，
没有训练新 checkpoint，也没有读取 `pilot_test` 或 `final_test`。
