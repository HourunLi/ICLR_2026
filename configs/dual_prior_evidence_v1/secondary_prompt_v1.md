# Independent secondary annotation prompt — CLIR dual-prior evidence v1

You are the independent second annotator. Label all 64 JSON objects in:

`/prodcpfs/user/panzhixin/ICLR_2027/configs/dual_prior_evidence_v1/annotation_items_v1.jsonl`

Follow this guide exactly:

`/prodcpfs/user/panzhixin/ICLR_2027/configs/dual_prior_evidence_v1/annotation_guide_v1.md`

Important independence rules:

- Do not inspect primary labels, reports, hallucination labels, correctness labels, reference answers, source lineage, or repository training results.
- Judge only the problem, trajectory, fixed units, and the annotation guide.
- Wrong reasoning remains eligible if it has an auditable path. Localize the decisive flaw as key and the attempted material chain as complete; do not output a correctness verdict.
- Select whole unit indices only. Never rewrite, split, or merge units.
- For every usable row, both arrays must be non-empty, sorted, duplicate-free, and `key_unit_indices` must be a subset of `complete_unit_indices`.
- For either ineligible status, both arrays must be empty.
- Preserve input order and copy every `item_id` exactly.

Write exactly 64 JSONL rows to:

`/prodcpfs/user/panzhixin/ICLR_2027/configs/dual_prior_evidence_v1/labels_secondary_raw_v1.jsonl`

Each line must be one strict JSON object with exactly these seven fields and no surrounding Markdown:

```json
{"schema_version":"clir-dual-prior-evidence-annotation-v1","item_id":"COPY_EXACTLY","eligibility":"usable","key_unit_indices":[3],"complete_unit_indices":[1,2,3],"confidence":"high","rationale":"Concise reason that explains both the sufficient complete chain and decisive key subset."}
```

Before reporting completion, validate:

1. exactly 64 non-empty lines;
2. every line parses independently as one JSON object;
3. item IDs and row order exactly match the input;
4. field names exactly match the example;
5. all indices are in range for that item's units;
6. all usable rows satisfy non-empty `key ⊆ complete`;
7. all ineligible rows have two empty arrays.

When done, report the output path, SHA256, and whether these structural checks passed. Do not summarize or reveal your labeling decisions in the chat.
