# CLIR JP-H blind A/B adjudication prompt v1

You are the third, score-blind adjudicator for a mixed-domain hallucination-localization validation set.

Read only:

- `/prodcpfs/user/panzhixin/ICLR_2027/configs/jp_h_blind_validation_v1/annotation_guide_v1.md`
- `/prodcpfs/user/panzhixin/ICLR_2027/configs/jp_h_blind_validation_v1/adjudication_items_v1.jsonl`

Do not read any lineage, primary/secondary source-label files, answer keys, correctness/domain fields, README or
handoff documents, model identities, or prediction/score files. `annotation_a` and `annotation_b` are shuffled.

Each input row has a disagreement that changes path status or at least one positive contradicted/unsupported token.
Independently re-evaluate the complete problem and trajectory. Treat A and B as evidence, not votes. Return a
complete final annotation covering all material claims under the guide; do not merely choose a span mechanically.

After judging each individual row, append and fsync that one decision immediately to `/prodcpfs/user/panzhixin/ICLR_2027/configs/jp_h_blind_validation_v1/adjudication_resolutions_raw_v1.jsonl` before moving to
the next row. Never hold multiple finished decisions only in memory. Resume by `item_id`; never rewrite a completed
valid row. Finish in input order and produce exactly one output row per input row.

Each JSONL row must contain exactly:

```json
{
  "schema_version": "clir-jp-h-blind-adjudication-decision-v1",
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
  "rationale": "why this resolves the target-changing disagreement"
}
```

`relation_to_views` must be `annotation_a`, `annotation_b`, `revised`, or `unresolved`. Even when choosing A or B,
copy a complete valid `final_annotation`. Every claim quote must be an exact non-empty trajectory substring and use
the correct zero-based occurrence. If the evidence cannot settle the first possible problem, use `unresolved` with
a valid `path_status="uncertain"` annotation instead of forcing a decision.
