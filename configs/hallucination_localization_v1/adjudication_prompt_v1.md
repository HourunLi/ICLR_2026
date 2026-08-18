# Blind A/B adjudication prompt — hallucination localization v1

You are adjudicating only trajectory items where two independent annotations disagree about path status, the first
material problem claim, or its exact Phi-token onset.

Read only:

- `/prodcpfs/user/panzhixin/ICLR_2027/configs/hallucination_localization_v1/annotation_guide_v1.md`
- `/prodcpfs/user/panzhixin/ICLR_2027/configs/hallucination_localization_v1/adjudication_items_v1.jsonl`

Do not read adjudication lineage, primary/secondary source files, correctness, reference answers, source/query IDs,
README/handoff, or model identities. `annotation_a` and `annotation_b` are deterministically shuffled per item.

For each row, independently re-evaluate the supplied problem and trajectory. The two annotations are evidence, not
votes. A different valid derivation and a derivable explanatory step are allowed. Select the earliest material
`contradicted` or `unsupported` claim; use `uncertain` if it cannot be resolved from the item.

Write one JSONL row per item, in input order, to:

`/prodcpfs/user/panzhixin/ICLR_2027/configs/hallucination_localization_v1/adjudications_raw_v1.jsonl`

Each row must have exactly this form:

```json
{
  "schema_version": "clir-hallucination-adjudication-decision-v1",
  "item_id": "copy exactly",
  "relation_to_views": "annotation_a",
  "final_annotation": {
    "item_id": "copy exactly",
    "claim_reviews": [
      {
        "claim_text": "exact trajectory substring",
        "occurrence": 0,
        "status": "supported",
        "reason": "item-specific evidence"
      }
    ],
    "path_status": "clean",
    "earliest_problem_claim_index": null,
    "confidence": "high",
    "summary": "item-specific final judgment"
  },
  "rationale": "why this resolves the disagreement"
}
```

`relation_to_views` must be one of `annotation_a`, `annotation_b`, `revised`, or `unresolved`. Even when selecting an
existing view, copy a complete valid `final_annotation`. Every claim quote must be an exact trajectory substring and
the earliest index must follow the annotation guide. Use `unresolved` with a valid `path_status=uncertain` annotation
instead of forcing a class.
