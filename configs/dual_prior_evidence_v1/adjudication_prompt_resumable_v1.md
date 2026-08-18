# Role-blind adjudication prompt — CLIR dual-prior evidence v1

You are the semantic adjudicator for 48 primary/secondary target-set disagreements. The preregistered agreement gate has
already passed. You must resolve every disagreement before any training gold is materialized.

Read and follow this semantic guide exactly:

`/prodcpfs/user/panzhixin/ICLR_2027/configs/dual_prior_evidence_v1/annotation_guide_v1.md`

Judge only the role-blinded disagreement packet:

`/prodcpfs/user/panzhixin/ICLR_2027/configs/dual_prior_evidence_v1/adjudication_items_v1.jsonl`

Do not inspect `adjudication_lineage_v1.jsonl`, the primary or secondary label files, correctness/reference-answer fields,
hallucination labels, split membership, source lineage, or training results. `annotation_a` and `annotation_b` are
deterministically role-shuffled proposals. Neither letter has priority.

## Semantic decision order

For each item:

1. Independently solve/audit the problem and trace the trajectory's actual reasoning before choosing between proposals.
2. Determine the minimal non-redundant `complete` chain needed to reproduce and audit that actual path.
3. Determine the smallest decisive `key` subset. For a flawed path, prefer the earliest or most causally decisive flaw,
   not a downstream terminal calculation merely because it states the answer.
4. Compare your decision with A and B:
   - use `resolution="adopt_a"` only when your eligibility/key/complete target exactly equals A;
   - use `resolution="adopt_b"` only when it exactly equals B;
   - use `resolution="synthesize"` only for a genuinely different target, which must differ from both A and B.
5. Keep whole unit indices, non-empty nested sets for usable rows, and `key ⊆ complete`.

Rationale and confidence differences alone are not the disagreement. Adjudicate the target semantics.

## Critical durability rule: decide one row, then checkpoint it immediately

At the start or after any interruption, run:

```bash
mkdir -p /prodcpfs/user/panzhixin/ICLR_2027/run_artifacts/dual_prior_evidence_v1/adjudication_checkpoint

/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python \
  /prodcpfs/user/panzhixin/ICLR_2027/scripts/checkpoint_dual_prior_adjudication_v1.py status
```

Read exactly the reported `next_item_id` from the frozen packet. After deciding it, write exactly one JSON object to:

`/prodcpfs/user/panzhixin/ICLR_2027/run_artifacts/dual_prior_evidence_v1/adjudication_checkpoint/current_row.json`

Then immediately run:

```bash
/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python \
  /prodcpfs/user/panzhixin/ICLR_2027/scripts/checkpoint_dual_prior_adjudication_v1.py append \
  --adjudication-file /prodcpfs/user/panzhixin/ICLR_2027/run_artifacts/dual_prior_evidence_v1/adjudication_checkpoint/current_row.json
```

Only a successful append completes the decision. Confirm `completed_rows` increases by one before reading the next item.
Never keep multiple completed decisions only in context or memory, and never manually concatenate/reorder the output.

The durable output is:

`/prodcpfs/user/panzhixin/ICLR_2027/configs/dual_prior_evidence_v1/adjudications_raw_v1.jsonl`

Each scratch object must contain exactly these eight fields and no Markdown:

```json
{"schema_version":"clir-dual-prior-evidence-adjudication-v1","item_id":"COPY_EXACTLY","resolution":"adopt_a","eligibility":"usable","key_unit_indices":[3],"complete_unit_indices":[1,2,3],"confidence":"high","rationale":"Concise semantic reason for the final complete chain and decisive key subset."}
```

Allowed `resolution`: `adopt_a`, `adopt_b`, `synthesize`. Allowed eligibility and confidence values remain those in the
frozen annotation guide.

After all 48 successful appends, run:

```bash
/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python \
  /prodcpfs/user/panzhixin/ICLR_2027/scripts/checkpoint_dual_prior_adjudication_v1.py finalize

/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python \
  /prodcpfs/user/panzhixin/ICLR_2027/scripts/validate_dual_prior_adjudication_v1.py \
  --input /prodcpfs/user/panzhixin/ICLR_2027/configs/dual_prior_evidence_v1/adjudications_raw_v1.jsonl
```

Do not begin gold materialization or training yourself. Stop only after finalization and structural validation pass.
