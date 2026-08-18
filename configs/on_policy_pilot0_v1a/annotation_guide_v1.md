# Route A Pilot-0 v1a 独立 claim-level 标注指南

## 任务

每个 item 都包含同一道题下由 Phi 正式 decoding policy 原生采样的两条轨迹。为了让表面轴可解释，item
已经按精确 output token 数定向：较短轨迹是 `native_compact` source，较长轨迹是
`native_expanded` rewrite。你要判断它们是否可以作为“推理基本相同、表达路径不同”的 consistency
正对。

这是一轮快速 Silver 训练门，不要求逐句或逐 token 对齐。只列和检查会影响前提、方法、关键中间量、
结论或内部错误的 **material claims**；紧密相连的一段算式可以合并成一个 claim。

## 宽松允许

- 换措辞、调整段落、拆分或合并步骤；
- 删除纯重复说明；
- 把 source 已有前提和方法直接蕴含的推导桥梁说清楚；
- 不改变实际数值、方法或结论的近似措辞，例如把精确值在叙述中称为“约”，但最终计算仍保持精确；
- expanded 轨迹更啰嗦、更有过渡语，只要确实存在可见的 compact/expanded 表面差异。

## 必须拒绝

- 换成另一种独立解法，而不是同一推理的重组；
- 漏掉、否定或替换会影响依赖链的关键前提/中间结论；
- 新增外部事实、未经 source 支持的假设或不同结论；
- 新增算术/逻辑错误，修复 source 的旧错误，或把旧错误移动到另一个语义位置；
- 只有长度变化、但实质是近乎复制且没有可见组织/表述差异。

最终答案正确 **不代表过程无错**。要局部核算所有新增或改写的数字陈述。例如一个轨迹最终余数正确，
但中间声称“整除无余数”，仍是新错误，必须拒绝。不要因为两条轨迹都被 checker 判为 correct 就把
`error_alignment.applicable` 自动设为 false。

## 双向 material-claim 检查

`source_to_rewrite.claims` 至少列出 source 中驱动方法、关键中间量和最终结论的 material claims，并在
rewrite 中给出证据。`essential=true` 表示缺失会改变推理身份或结论。

`rewrite_to_source.claims` 至少列出 rewrite 的同类 material claims，尤其是 source 中没有逐字出现的新增
说明、数值和推断，并在 source 中给出支持。纯过渡语不必单独列 claim。

relation 只能是：

- `equivalent`：双向含义相同；
- `entailed_elaboration`：rewrite 新明说的桥梁由 source 既有前提和同一方法直接推出；
- `preserved_implicitly`：source claim 在 rewrite 中没有逐字复述，但明确由其步骤保留；
- `omitted`、`contradicted`、`replaced_by_different_reasoning`：明确失败；
- `uncertain`：只有证据确实不足时使用。

accept 时：每个 essential source claim 必须是 `equivalent` 或 `preserved_implicitly`；每个 substantive
rewrite claim 必须是 `equivalent` 或 `entailed_elaboration`。

## 内部错误

- 两条轨迹都没有观察到内部错误：`error_alignment.applicable=false`，其余五个字段为 null。
- source 有内部错误：设为 true，描述 source 与 rewrite 中的对应错误，并分别判断机制、语义位置、
  downstream effect 是否相同。只要一项不同就不能 accept。
- 只有 rewrite 新增错误：`error_alignment.applicable=false`，同时
  `global_relation.introduced_new_error=true`，在双向 claim 和 issues 中写清楚。

## 风险提示

item 的 `risk_probe.risks` 是程序生成的注意事项，不是自动拒绝条件。输出的 `risk_review` 必须逐个覆盖
`expected_risk_ids`，不多不少：

- 若只是无害的表面变化，`resolved=true` 并简要解释；
- 若暴露了新数字错误、否定/量词/因果含义变化等，`resolved=false`，并据此 reject；
- 没有风险时输出空数组。

## style 判断

固定 `target_style=native_compact_to_expanded`。较长并不自动通过；expanded 还应有实际的措辞、组织、
步骤拆分或说明粒度差异。这个标签只是当前 pilot 的相对 compact/expanded 轴，不是通用文风分类。

## 输出格式

每个输入 item 输出一行严格 JSON，顺序和 `item_id` 不变：

```json
{
  "schema_version": "clir-on-policy-claim-annotation-v1",
  "item_id": "OPA-...",
  "report": {
    "schema_version": "clir-reasoning-equivalence-verifier-v1",
    "source_to_rewrite": {
      "claims": [
        {
          "claim_id": "S1",
          "source_claim": "material source claim",
          "essential": true,
          "rewrite_evidence": "matching text or calculation",
          "relation": "equivalent",
          "explanation": "why the relation holds or fails"
        }
      ],
      "all_essential_claims_preserved": true
    },
    "rewrite_to_source": {
      "claims": [
        {
          "claim_id": "R1",
          "rewrite_claim": "material rewrite claim",
          "substantive": true,
          "source_evidence": "matching source text or premises",
          "relation": "equivalent",
          "explanation": "why the relation holds or fails"
        }
      ],
      "all_substantive_claims_supported": true
    },
    "global_relation": {
      "same_task_and_goal": true,
      "same_core_premises": true,
      "same_reasoning_method": true,
      "same_key_inferences": true,
      "same_intermediate_conclusions": true,
      "same_final_conclusion": true,
      "introduced_external_fact": false,
      "omitted_essential_claim": false,
      "contradicted_source_claim": false,
      "replaced_with_different_solution": false,
      "introduced_new_error": false
    },
    "error_alignment": {
      "applicable": false,
      "source_error": null,
      "rewrite_error": null,
      "same_error_mechanism": null,
      "same_semantic_error_location": null,
      "same_downstream_effect": null
    },
    "style_assessment": {
      "target_style": "native_compact_to_expanded",
      "satisfied": true,
      "evidence": "specific visible surface/organization contrast"
    },
    "risk_review": [],
    "confidence": "high",
    "decision": "accept",
    "issues": []
  }
}
```

严格约束：

- report 只能使用上述字段；claim、global、error、style 和 risk item 也不能增删字段；
- `decision` 为 `accept|reject|review`，`confidence` 为 `high|medium|low`；
- accept 只用于你有充分证据的 pair；真正边界不清楚时用 review；
- accept 的 `issues=[]`；reject/review 应具体写出问题；
- 输出 JSONL，不要 Markdown fence、总结或计数。

## 独立性

只能读取 item 文件和本指南。不要读取第一份标注、旧 Mistral verifier 结果、旧 62 行训练 manifest、
package report、lineage、仓库历史或任何预期通过率。不要调用另一个模型或拆给其他 agent。
