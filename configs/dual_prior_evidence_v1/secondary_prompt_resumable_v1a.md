# Independent secondary annotation prompt — CLIR dual-prior evidence v1a

You are the independent second annotator. Label all 64 JSON objects in:

`/prodcpfs/user/panzhixin/ICLR_2027/configs/dual_prior_evidence_v1/annotation_items_v1.jsonl`

Follow this frozen semantic guide exactly:

`/prodcpfs/user/panzhixin/ICLR_2027/configs/dual_prior_evidence_v1/annotation_guide_v1.md`

This v1a prompt changes only how progress is saved. It does not change any label definition, item, order, or independence rule from v1.

## Critical reliability rule: judge one row, then save that row immediately

Do **not** keep multiple completed judgments only in memory or wait until all 64 rows are finished before writing. For every item, use this exact cycle:

1. Ask the checkpoint helper which single row is next.
2. Read and judge exactly that input item.
3. Write exactly one annotation JSON object to the scratch file.
4. Immediately append and durably checkpoint that one row with the helper.
5. Confirm that `completed_rows` increased by one before judging another item.

At the beginning of every run, including every resumed run, execute:

```bash
mkdir -p /prodcpfs/user/panzhixin/ICLR_2027/run_artifacts/dual_prior_evidence_v1/secondary_checkpoint

/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python \
  /prodcpfs/user/panzhixin/ICLR_2027/scripts/checkpoint_dual_prior_secondary_v1.py status
```

The response identifies `next_row_number_1_based` and `next_item_id`. Locate that exact row in the frozen input. Do not skip ahead or redo an already checkpointed row.

After judging that one item, write its one strict JSON object, with no Markdown, to:

`/prodcpfs/user/panzhixin/ICLR_2027/run_artifacts/dual_prior_evidence_v1/secondary_checkpoint/current_row.json`

Then immediately execute:

```bash
/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python \
  /prodcpfs/user/panzhixin/ICLR_2027/scripts/checkpoint_dual_prior_secondary_v1.py append \
  --annotation-file /prodcpfs/user/panzhixin/ICLR_2027/run_artifacts/dual_prior_evidence_v1/secondary_checkpoint/current_row.json
```

Only a successful `append` makes that judgment complete. The helper validates the next frozen item, atomically publishes the valid JSONL prefix, flushes it to storage, rereads it, and reports the new progress. If the process times out or context is running low, stop only after a successful append. A later run must begin with `status` and continue from the reported next item.

If a scratch file remains after an interruption, first run `status`. Append the scratch object only if its `item_id` exactly equals the newly reported `next_item_id`; otherwise replace the scratch object with the judgment for the reported next item. Never manually concatenate, reorder, truncate, or repair the checkpoint output.

The durable checkpoint/output path is:

`/prodcpfs/user/panzhixin/ICLR_2027/configs/dual_prior_evidence_v1/labels_secondary_raw_v1.jsonl`

## Frozen independence and labeling rules

- Do not inspect primary labels, reports, hallucination labels, correctness labels, reference answers, source lineage, or repository training results.
- Judge only the problem, trajectory, fixed units, and the annotation guide.
- Wrong reasoning remains eligible if it has an auditable path. Localize the decisive flaw as key and the attempted material chain as complete; do not output a correctness verdict.
- Select whole unit indices only. Never rewrite, split, or merge units.
- For every usable row, both arrays must be non-empty, sorted, duplicate-free, and `key_unit_indices` must be a subset of `complete_unit_indices`.
- For either ineligible status, both arrays must be empty.
- Preserve input order and copy every `item_id` exactly. The helper enforces this order.

Each scratch file must contain exactly one JSON object with exactly these seven fields:

```json
{"schema_version":"clir-dual-prior-evidence-annotation-v1","item_id":"COPY_EXACTLY","eligibility":"usable","key_unit_indices":[3],"complete_unit_indices":[1,2,3],"confidence":"high","rationale":"Concise reason that explains both the sufficient complete chain and decisive key subset."}
```

After the 64th successful append, execute:

```bash
/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python \
  /prodcpfs/user/panzhixin/ICLR_2027/scripts/checkpoint_dual_prior_secondary_v1.py finalize

/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python \
  /prodcpfs/user/panzhixin/ICLR_2027/scripts/validate_dual_prior_secondary_v1.py \
  --input /prodcpfs/user/panzhixin/ICLR_2027/configs/dual_prior_evidence_v1/labels_secondary_raw_v1.jsonl
```

Report completion only after both commands pass. Report the output path, SHA256, and whether structural validation passed. Do not summarize or reveal labeling decisions in chat.
