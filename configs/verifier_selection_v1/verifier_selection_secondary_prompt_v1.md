You are the independent secondary annotator for a CLIR reasoning-equivalence verifier selection set.

Read only these two files:

1. `/prodcpfs/user/panzhixin/ICLR_2027/configs/verifier_selection_v1/verifier_selection_items_v1.jsonl`
2. `/prodcpfs/user/panzhixin/ICLR_2027/configs/verifier_selection_v1/verifier_selection_annotation_guide_v1.md`

Follow the annotation guide completely. Annotate every item independently and in input order. The source may be correct or incorrect. Your job is to compare the source reasoning with its rewrite, not to replace an incorrect source with a correct solution. A rewrite may reorganize, split, merge, or directly elaborate reasoning already entailed by the source, but it must not introduce a new premise, evidence, example, analogy, method, correction, or error. It must preserve an incorrect source's error mechanism, semantic error location, downstream effect, and final conclusion. It must also satisfy the declared target style.

Write your result to:

`/prodcpfs/user/panzhixin/ICLR_2027/configs/verifier_selection_v1/verifier_selection_labels_secondary_v1.jsonl`

The output must contain exactly one strict JSON object per line, one for every input item, using exactly the schema in the guide. Preserve item order and IDs. Do not include Markdown, prose, a summary, or aggregate counts in the output file.

Independence is mandatory. Do not read or search for any of the following:

- `verifier_selection_labels_primary_v1.jsonl`
- `verifier_selection_manifest_v1.json`
- `SHA256SUMS`
- candidate verifier outputs
- git history or repository documents discussing expected labels
- any previous annotator's decisions or reasons

Do not run a candidate verifier model, do not ask another agent to label subsets, and do not infer an expected class balance. If a case is genuinely ambiguous under the guide, use `review` and explain the exact ambiguity. Do not alter the items or guide.

Before finishing, validate locally that:

- there is one output row for every input row;
- IDs and order match exactly;
- every row has exactly the required fields;
- enum, Boolean, null, and conditional fields obey the guide;
- every reason is non-empty and evidence-specific.

After writing and validating the JSONL, respond only with the output path and whether all structural checks passed. Do not report label counts, because those could influence later adjudication.
