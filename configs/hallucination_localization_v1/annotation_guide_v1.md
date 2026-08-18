# Hallucination localization annotation guide v1

This task asks whether a trajectory contains a claim that is unsupported or contradicted, and where the first such
claim begins. It does not ask whether the final answer matches a hidden answer key. Judge only from the supplied
problem, the trajectory's valid earlier claims, ordinary stable domain knowledge, and checkable logic or arithmetic.

## Core definition

A path is `hallucinated` when at least one material claim is:

- `contradicted`: false given the problem, checkable facts, arithmetic, logic, units, or a valid earlier derivation; or
- `unsupported`: introduces a needed premise, fact, causal link, intermediate value, or conclusion that does not follow
  from the problem or the valid reasoning prefix.

The onset is the first such material claim in response order. Everything after that onset may be treated as a
contaminated tail by training, but you must still identify the actual first bad claim rather than a later consequence.

Use these claim statuses:

- `supported`: follows from the problem, stable domain knowledge, or the valid earlier reasoning prefix;
- `contradicted`: makes a checkably false statement or inference;
- `unsupported`: may not be directly false, but lacks the premise/evidence needed for the reasoning;
- `non_claim`: heading, transition, intention, or formatting text with no material assertion;
- `uncertain`: the available evidence is insufficient to decide reliably.

## What is not hallucination by itself

- a different but valid solution path;
- an omitted optional explanation or compressed algebra;
- adding an intermediate step that follows from available information;
- approximate language that is explicitly approximate and does not change the operative value;
- verbosity, style, ordering, or formatting changes;
- an incomplete response, refusal, missing final answer, or answer-format mistake that makes no unsupported claim.

Correct final answers can still contain an earlier bad claim. Incorrect final answers can be `clean` when they are only
incomplete or malformed and never assert an unsupported conclusion. Never infer the path label from apparent final
correctness.

## Review procedure

1. Read the problem and trajectory independently.
2. List the material claims needed to establish the reasoning path and conclusion. Include any suspicious premise,
   equation, unit conversion, causal step, or factual assertion. You do not need to list purely stylistic sentences.
3. For each listed claim, copy an exact non-empty substring from `trajectory`, preserving case, punctuation, symbols,
   and whitespace inside the quote.
4. If that exact substring occurs more than once, set `occurrence` to its zero-based occurrence in the full trajectory.
   Otherwise use `0`.
5. Keep `claim_reviews` ordered by the first character of each quote in the trajectory.
6. If there is a definite contradicted/unsupported claim, set `path_status="hallucinated"` and set
   `earliest_problem_claim_index` to the zero-based list index of the first definite problem claim.
7. If all material claims are supported/non-claims, set `path_status="clean"` and the earliest index to `null`.
8. If the earliest possible problem cannot be decided reliably, use `path_status="uncertain"`. Do not force a clean
   label. The earliest index may be `null` or may point to an `uncertain` claim.

For a hallucinated path, an `uncertain` review cannot appear before the selected definite onset: that would make the
true earliest onset uncertain. Explanations must be item-specific and should state the missing premise or contradiction.

## Required output schema

Return exactly one JSON object, with no Markdown or commentary:

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
