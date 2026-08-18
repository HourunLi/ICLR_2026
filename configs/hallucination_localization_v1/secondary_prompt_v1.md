# Independent secondary annotation prompt — hallucination localization v1

You are the independent second annotator for 64 blind trajectory items.

Read only these task files:

- `/prodcpfs/user/panzhixin/ICLR_2027/configs/hallucination_localization_v1/annotation_guide_v1.md`
- `/prodcpfs/user/panzhixin/ICLR_2027/configs/hallucination_localization_v1/secondary_items_v1.jsonl`

Do not read the selection manifest, annotation lineage, primary labels, primary model run, package reports,
README/handoff discussion, source correctness, reference answers, or any previous annotation. Do not browse for a
hidden answer key. The problem and trajectory in each blind item are the complete annotation evidence.

Follow the guide exactly. For each input row, output one strict JSON object with exactly these top-level keys:

`item_id`, `claim_reviews`, `path_status`, `earliest_problem_claim_index`, `confidence`, `summary`.

Important requirements:

1. Preserve the 64-item input order and copy every `item_id` exactly.
2. Write JSONL: exactly 64 one-line JSON objects, with no Markdown fences and no extra prose.
3. Every `claim_text` must be an exact non-empty substring of that item's `trajectory`; use zero-based
   `occurrence` when a quote repeats.
4. Order `claim_reviews` by response position. For `hallucinated`, the earliest index must point to the first
   `contradicted` or `unsupported` reviewed claim.
5. Do not infer path status from whether the final answer looks correct. A valid different derivation is allowed;
   an unsupported or false intermediate claim is not.
6. Use `uncertain` instead of forcing `clean` whenever the earliest possible bad claim cannot be resolved.

Write the completed file to:

`/prodcpfs/user/panzhixin/ICLR_2027/configs/hallucination_localization_v1/labels_secondary_raw_v1.jsonl`

Then run this structure-only blind validator:

```bash
/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python   /prodcpfs/user/panzhixin/ICLR_2027/scripts/validate_hallucination_annotations_v1.py   --items /prodcpfs/user/panzhixin/ICLR_2027/configs/hallucination_localization_v1/secondary_items_v1.jsonl   --labels /prodcpfs/user/panzhixin/ICLR_2027/configs/hallucination_localization_v1/labels_secondary_raw_v1.jsonl
```

Fix only schema, ordering, or exact-quote errors reported by the validator; do not consult prohibited files. Stop only
after it prints `"status": "passed"`, and report the output path and SHA256.
