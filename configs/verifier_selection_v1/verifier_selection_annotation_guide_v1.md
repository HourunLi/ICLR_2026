# Independent annotation guide: CLIR reasoning equivalence v1

## Task

For every item, decide whether the rewrite preserves the source's reasoning while satisfying the requested style. Judge the relationship between the supplied source and rewrite; do not solve the problem from scratch and do not prefer a correct rewrite over a faithfully preserved incorrect source.

## Labels

- `accept`: the rewrite preserves the same core premises, reasoning method, key inferences, intermediate and final conclusions, qualifications, and (when present) the same error mechanism; it also satisfies the target style.
- `reject`: there is a clear material change, including an essential omission, contradiction, new premise/evidence, different solution method, correction, new or relocated error, changed conclusion, or target-style failure.
- `review`: available evidence does not support a confident accept or reject. Use it for a genuinely ambiguous boundary, not as a substitute for analysis.

## Allowed changes

The rewrite may change wording, line count, organization, and the splitting or merging of steps. It may reorder independent explanations while preserving logical dependencies. It may state an omitted bridge explicitly when that bridge follows directly from the source's existing premises and method (`entailed_elaboration`). It may remove purely redundant wording.

An entailed elaboration cannot introduce a new premise, external fact, example, analogy, hypothetical scenario, independent solution method, correction, or new error.

## Incorrect sources

Do not repair them. An acceptable rewrite must preserve:

1. the same core error mechanism;
2. the same semantic location in the reasoning dependency chain (not necessarily the same sentence or token);
3. the same downstream effect and final conclusion.

Keeping the same wrong answer is insufficient if the rewrite fixes the old error and creates a new one, invents a premise, or reaches that answer through a different mistake.

## Style

- `terse_formal`: concise and formal; related steps may be merged and redundant wording removed, but the reasoning cannot become an answer-only summary.
- `explanatory_conversational`: clear, natural, and explanatory; dense steps may be split and directly entailed bridges made explicit, but no example, analogy, external evidence, correction, or different method may be added.

A reasoning-equivalent rewrite with an unmet target style is `reject`. Do not impose a numeric edit-distance or length threshold.

## Output schema

Write exactly one JSON object per input item, in the same order, with exactly these fields:

```json
{
  "item_id": "VS001",
  "decision": "accept|reject|review",
  "same_task_and_goal": true,
  "same_core_premises": true,
  "same_reasoning_method": true,
  "same_key_inferences": true,
  "same_intermediate_conclusions": true,
  "same_final_conclusion": true,
  "entailed_elaboration_present": false,
  "entailed_elaboration_valid": null,
  "introduced_new_premise_or_evidence": false,
  "omitted_essential_claim": false,
  "contradicted_source_claim": false,
  "replaced_with_different_solution": false,
  "introduced_new_error": false,
  "error_alignment_applicable": false,
  "same_error_mechanism": null,
  "same_semantic_error_location": null,
  "same_downstream_effect": null,
  "target_style": "terse_formal|explanatory_conversational",
  "style_satisfied": true,
  "confidence": "high|medium|low",
  "reason": "Specific evidence-based explanation."
}
```

Constraints:

- Copy `item_id` and `target_style` exactly from the item.
- If `entailed_elaboration_present=false`, set `entailed_elaboration_valid=null`.
- If it is true, `entailed_elaboration_valid` must be Boolean.
- For a correct source, set `error_alignment_applicable=false` and the three `same_error_*` fields to null.
- For an incorrect source, set `error_alignment_applicable=true` and all three `same_error_*` fields to Boolean values.
- `confidence` does not override `decision`.
- The reason must identify the concrete preserved or changed premise, inference, method, conclusion, error, or style. Do not write only “same meaning” or “different.”
- Output JSONL only: no Markdown fence, summary, counts, or commentary.

## Independence requirements

Do not inspect primary labels, expected label counts, manifest statistics, candidate-verifier outputs, repository history, or another annotator's reasoning. Use only the supplied items and this guide. Do not modify the items file.
