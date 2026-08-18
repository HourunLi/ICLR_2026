# CLIR dual-prior evidence annotation guide v1

## 1. Your task

For one problem and one candidate trajectory, assign fixed trajectory units to two nested evidence sets:

- `complete_unit_indices`: the non-redundant candidate-side reasoning needed to reconstruct and audit how the trajectory reaches its conclusion from the problem;
- `key_unit_indices`: the smallest subset of that complete set whose verification most directly determines whether the conclusion is warranted.

You are localizing reasoning evidence. You are **not** asked to output whether the final answer is correct, and you must not rewrite the trajectory.

Every selected key unit must also be complete: `key ⊆ complete`.

## 2. Work in this order

1. Identify the conclusion actually asserted by the trajectory.
2. Trace the reasoning path actually used, even when it is wrong.
3. Mark the minimal non-redundant chain needed to reproduce and audit that path as `complete`.
4. Within that chain, mark the smallest decisive unit or units as `key`.
5. Check that both arrays are sorted, have no duplicates, are non-empty, and key is a subset of complete.

## 3. What counts as complete evidence

Include a unit when removing it would leave a real gap in reconstructing or auditing the trajectory's argument. This can include:

- a material transformation of a problem fact;
- an intermediate result used later;
- a rule, assumption, factual assertion, calculation, code step, or logical inference that the conclusion depends on;
- a decisive erroneous or unsupported step in a flawed trajectory;
- the only unit that explicitly states the conclusion, if no earlier selected unit already states it.

Do not include:

- headings such as `Step 2:` or `Final Answer:` by themselves;
- planning chatter, transition prose, confidence statements, or explanations of what will happen next;
- a unit that only copies information already present in the problem and performs no transformation;
- duplicated equations, paraphrases, or final-answer wrappers when an earlier selected unit already carries the same result;
- an alternative derivation that the trajectory does not actually use.

`complete` means a minimal sufficient audit chain, not every relevant-looking sentence.

## 4. What counts as key evidence

Key is not simply the last unit and not simply every calculation. It is the smallest part of the complete chain that is most decisive for the verdict on the conclusion.

- For a sound trajectory, choose the decisive answer-producing inference or calculation. If that inference depends on two independently decisive branches, include both.
- For a flawed trajectory, choose the earliest or most causally decisive erroneous/unsupported inference. Add a downstream unit only when it is independently needed to see how the flaw determines the conclusion.
- If several units merely repeat the same result, choose the first clear decisive one.
- A pure conclusion restatement is not key when the reasoning unit that produced that conclusion is available.

Key need not reproduce the whole argument alone; complete supplies the dependencies.

## 5. Fixed-unit and eligibility rules

You must select whole supplied units. Do not edit text, invent spans, merge indices, or split a unit yourself.

Use `eligibility = "usable"` when the units permit both non-empty nested sets.

Use `eligibility = "insufficient_unitization"` only when a necessary reasoning fragment is absent or inseparably fused with unrelated content so that the fixed units cannot express defensible targets. Then both arrays must be empty.

Use `eligibility = "no_auditable_reasoning"` when the trajectory gives only an answer, refusal, or otherwise contains no reasoning path to localize. Then both arrays must be empty.

Do not mark a row ineligible merely because its reasoning is wrong. Wrong reasoning is still auditable reasoning.

## 6. Domain-general examples

### Correct arithmetic

Units: `(0) Step 1:`; `(1) 4 bags at 3 apples each give 4 * 3 = 12 apples.`; `(2) Therefore the answer is 12.`

- complete: `[1]`
- key: `[1]`
- unit 2 is a duplicate conclusion wrapper, so it is neither.

### Multi-step reasoning

Units: `(0) The rate is 60/3 = 20 per hour.`; `(1) At that rate, 5 hours gives 20 * 5 = 100.`; `(2) The answer is 100.`

- complete: `[0, 1]`
- key: `[1]`
- unit 2 repeats the result.

### Flawed reasoning

Units: `(0) The ticket costs $40 per person.`; `(1) Because the trip lasts three days, ticket cost is 40 * 3 = 120 per person.`; `(2) Adding the hotel gives $200.`

- complete: `[1, 2]` if unit 2 uses the erroneous 120;
- key: `[1]` because multiplying a one-time ticket by days is the decisive flaw;
- unit 0 merely copies the problem fact and is unnecessary when unit 1 contains the value and transformation.

### Factual or coding task

The same rule applies outside mathematics. For factual QA, complete contains the material factual/inferential chain and key contains the decisive claim. For code reasoning, complete contains the transformations/conditions needed to derive behavior and key contains the branch, invariant, or operation that most directly determines the answer.

## 7. Output contract

Return exactly one JSON object and no Markdown or commentary:

```json
{"schema_version":"clir-dual-prior-evidence-annotation-v1","item_id":"COPY_EXACTLY","eligibility":"usable","key_unit_indices":[3],"complete_unit_indices":[1,2,3],"confidence":"high","rationale":"One concise explanation of why the complete chain is sufficient and the key subset is decisive."}
```

Allowed eligibility values: `usable`, `insufficient_unitization`, `no_auditable_reasoning`.

Allowed confidence values: `high`, `medium`, `low`.
