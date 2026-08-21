# CLIR mixed-domain hallucination-localization annotation guide v1

This task asks whether a model trajectory contains a material claim that is unsupported or contradicted, and where
the first such claim occurs. It does **not** ask whether the final answer matches a hidden answer key. Judge only from
the supplied problem, any supplied passage, the trajectory's valid earlier claims, ordinary stable knowledge, and
checkable logic or arithmetic.

The item may be a math problem, a science or commonsense multiple-choice question, or a passage-grounded yes/no
question. Apply one domain-independent standard: every material step must be licensed by the problem/passage,
stable knowledge, or a valid earlier derivation.

## Core definition

A path is `hallucinated` when at least one material claim is:

- `contradicted`: false given the problem, supplied passage, checkable stable facts, arithmetic, logic, units, or a
  valid earlier derivation; or
- `unsupported`: introduces a needed premise, fact, causal link, intermediate value, or conclusion that does not
  follow from the available evidence.

The onset is the first such material claim in response order. Later claims must still be reviewed when material, but
the onset must point to the actual first bad claim rather than a later consequence.

Use exactly these claim statuses:

- `supported`: follows from the problem/passage, stable domain knowledge, or the valid earlier reasoning prefix;
- `contradicted`: makes a checkably false statement or inference;
- `unsupported`: may not be directly false, but lacks the premise/evidence needed for the reasoning;
- `non_claim`: heading, transition, intention, or formatting text with no material assertion;
- `uncertain`: the available evidence is insufficient to decide reliably.

## Cross-domain rules

- Treat a passage supplied in the problem as the authoritative evidence for a passage-grounded question. Do not
  replace it with remembered facts. If the passage itself does not settle a necessary inference, mark that inference
  `unsupported` or `uncertain` as appropriate.
- For multiple choice, option elimination is a material inference. Review the factual or logical reason used to
  eliminate or select an option, and review the final selected option as a conclusion.
- Stable, elementary facts may support a step. Specialized, disputed, time-sensitive, or obscure facts that are not
  established by the problem should not be silently supplied; use `unsupported` or `uncertain`.
- A concise inference is not unsupported merely because optional intermediate prose is omitted. It is unsupported
  only when a genuinely necessary premise or link is absent.
- Never infer a path label from whether the final answer merely looks right or wrong. A correct answer can follow a
  bad path; an incomplete or malformed answer can make no hallucinated claim.

## What is not hallucination by itself

- a different but valid solution path;
- an omitted optional explanation or compressed valid inference;
- adding an intermediate step that follows from available information;
- approximate language that is explicitly approximate and does not change the operative conclusion;
- verbosity, style, ordering, or formatting differences;
- an incomplete response, refusal, missing final answer, or answer-format mistake that makes no unsupported claim.

## Review procedure

1. Read the complete problem and trajectory independently.
2. Identify **all material reasoning and conclusion claims** needed to audit the path, including suspicious premises,
   factual assertions, eliminations, equations, unit conversions, causal steps, and the final conclusion. Purely
   stylistic text need not be listed.
3. For each reviewed claim, copy a complete auditable clause or sentence as an exact non-empty substring from
   `trajectory`. Preserve case, punctuation, symbols, and internal whitespace. Do not paraphrase or repair it, and do
   not quote only a single ambiguous word when a complete clause is available.
4. If that exact substring occurs more than once, set `occurrence` to its zero-based occurrence in the full
   trajectory; otherwise use `0`.
5. Keep `claim_reviews` ordered by the first character of each quote in the trajectory. Do not duplicate overlapping
   quotes for the same claim.
6. If a definite contradicted/unsupported claim exists, set `path_status="hallucinated"` and set
   `earliest_problem_claim_index` to the zero-based list index of the first definite problem claim.
7. If every reviewed material claim is supported/non-claim, set `path_status="clean"` and the earliest index to
   `null`.
8. If the earliest possible problem cannot be decided reliably, use `path_status="uncertain"`. Do not force a clean
   or hallucinated label. The earliest index may be `null` or may point to an `uncertain` claim.

For a hallucinated path, an `uncertain` review cannot appear before the selected definite onset: that would make the
true earliest onset uncertain. Reasons must be item-specific and state the supporting evidence, missing premise, or
contradiction.

## Required output schema

Return exactly one JSON object with exactly these six top-level fields, with no Markdown or commentary:

```json
{
  "item_id": "copy exactly",
  "claim_reviews": [
    {
      "claim_text": "exact trajectory substring",
      "occurrence": 0,
      "status": "supported",
      "reason": "specific evidence-based reason"
    }
  ],
  "path_status": "hallucinated",
  "earliest_problem_claim_index": 1,
  "confidence": "high",
  "summary": "concise item-specific path judgment"
}
```

Allowed `confidence` values are `high`, `medium`, and `low`. Do not add keys. Do not reveal or guess a hidden
correctness label. Do not repair the trajectory.
