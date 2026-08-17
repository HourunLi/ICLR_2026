# CLIR native-pair reasoning-equivalence verifier guide v1

Judge the relationship between two trajectories produced for the same problem. Do not solve the problem again and do not use whether either trajectory is correct as a shortcut.

Return `accept` only when both conditions hold:

1. The alternate preserves the anchor's core premises, reasoning method, key inferences, intermediate conclusions, final conclusion, and qualifications. It introduces no unsupported premise, contradiction, different solution method, omission, correction, or new error. A directly entailed explanatory bridge is allowed.
2. The pair has meaningful surface/presentation difference (`native_surface_difference`), such as substantially different wording, organization, step splitting/merging, or level of explanation. A near-copy or only trivial lexical substitution fails this condition.

Use `reject` for a clear relation or surface-difference failure. Use `review` only for genuine ambiguity. The anchor/alternate direction is an audit convention, not an authorship or quality ranking.

Return exactly one JSON object and no Markdown. It must contain exactly these fields:

```json
{
  "item_id": "OPA-...",
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
  "target_style": "native_surface_difference",
  "style_satisfied": true,
  "confidence": "high|medium|low",
  "reason": "Specific evidence from both trajectories."
}
```

Copy `item_id` and `target_style` exactly. When `entailed_elaboration_present=false`, its validity must be null. For this correct-only pilot, set `error_alignment_applicable=false` and all three error-alignment values to null. The reason must cite a concrete preserved or changed inference, intermediate result, conclusion, or surface difference.
