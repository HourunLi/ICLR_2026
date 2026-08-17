#!/usr/bin/env python3
"""Build the frozen, model-blind v1 verifier selection annotation package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "verifier_selection_v1"


def variant(text: str, decision: str, reason: str, **flags: Any) -> dict[str, Any]:
    return {"rewrite": text, "decision": decision, "reason": reason, **flags}


# Within every domain the four bases use decision patterns ARRV, AARR, ARAV,
# and ARRR.  This gives 6 accept / 8 reject / 2 review per domain while each
# base contributes two terse and two conversational targets.
BASES = [
    # Quantitative reasoning: two correct, two incorrect.
    {
        "domain": "quantitative", "correct": True,
        "problem": "A theater has 6 rows with 8 seats in each row. How many seats are there?",
        "source": "There are 6 equal rows, and each row has 8 seats. Multiplying 6 by 8 gives 48, so the theater has 48 seats.",
        "variants": [
            variant("Six rows of eight seats give 6 × 8 = 48 seats in total.", "accept", "Concise reorganization preserves the multiplication method and conclusion."),
            variant("Think of the seats as six equal groups, with eight seats in every group. Multiplying the six groups by eight seats per group gives 48 seats altogether.", "reject", "The reasoning is equivalent, but this conversational expansion does not satisfy the terse target style.", style_failure=True),
            variant("There are 6 rows and 8 seats, so adding 6 and 8 gives 14 seats.", "reject", "It replaces multiplication with addition and changes the conclusion.", method_changed=True, conclusion_changed=True),
            variant("With six rows holding eight seats apiece, we combine the equal groups by multiplying 6 by 8. That works out to 48 seats in the theater.", "review", "The relation is valid, but whether the wording is sufficiently conversational rather than merely neutral is borderline.", style_uncertain=True),
        ],
    },
    {
        "domain": "quantitative", "correct": True,
        "problem": "A 24-liter tank is one quarter full. How many liters are currently in it?",
        "source": "One quarter of the 24-liter capacity is filled. Dividing 24 by 4 gives 6, so the tank currently contains 6 liters.",
        "variants": [
            variant("The filled amount is one quarter of 24 liters: 24 ÷ 4 = 6 liters.", "accept", "It formally compresses the same fraction-of-capacity calculation."),
            variant("The tank's full capacity is 24 liters, and only one of four equal quarters is occupied. Splitting 24 into four equal parts makes each part 6 liters, so 6 liters are in the tank now.", "accept", "It validly elaborates the implicit equal-parts interpretation without adding a premise.", entailed_elaboration=True),
            variant("A quarter of the tank is empty, so three quarters of 24 liters, or 18 liters, are currently inside.", "reject", "It reverses filled and empty and changes the conclusion.", contradiction=True, conclusion_changed=True),
            variant("The tank is one quarter full. Using the standard density of water, its current contents weigh about 6 kilograms.", "reject", "It introduces an external density fact and answers a different quantity.", new_evidence=True, conclusion_changed=True),
        ],
    },
    {
        "domain": "quantitative", "correct": False,
        "problem": "Five boxes contain 7 pens each. How many pens are there?",
        "source": "There are 5 boxes with 7 pens per box. Multiplying 5 by 7 gives 30, so there are 30 pens.",
        "variants": [
            variant("Five equal boxes at seven pens each are evaluated as 5 × 7 = 30 pens.", "accept", "It preserves the multiplication method, erroneous arithmetic result, and conclusion."),
            variant("There are five boxes, and each one holds seven pens. Multiplying those values actually gives 35 pens, so the total is 35.", "reject", "It repairs the source arithmetic error.", corrected_error=True, conclusion_changed=True),
            variant("We count seven pens in each of five boxes. Following the stated calculation, the five groups are taken to total 30 pens.", "accept", "It makes the grouping bridge explicit while preserving the same erroneous result."),
            variant("Five boxes of seven would normally make 35, but assume five pens are missing; that leaves 30 pens.", "review", "It preserves the conclusion but invents a missing-pens premise; the deliberate contrast makes the error relation atypical and unsuitable for automatic acceptance.", new_premise=True),
        ],
    },
    {
        "domain": "quantitative", "correct": False,
        "problem": "A runner covers 3 kilometers each day for 4 days. How far does the runner travel?",
        "source": "The runner goes 3 kilometers on each of 4 days. Adding 3 and 4 gives 7, so the runner travels 7 kilometers.",
        "variants": [
            variant("The stated method adds the daily distance and day count: 3 + 4 = 7 kilometers.", "accept", "It preserves the source's addition error and erroneous conclusion."),
            variant("The runner covers 3 kilometers on four separate days. Since the daily distance repeats four times, 3 × 4 gives 12 kilometers altogether.", "reject", "It replaces and corrects the source method." , corrected_error=True, method_changed=True, conclusion_changed=True),
            variant("The runner travels 7 kilometers because 3 kilometers on the first day plus 4 kilometers over the remaining days totals 7.", "reject", "It retains 7 but changes the error mechanism and invents a distribution across days.", new_premise=True, error_changed=True),
            variant("Across the four days, the runner's recorded total is seven kilometers.", "reject", "It omits the essential erroneous 3+4 inference rather than preserving the reasoning.", omission=True),
        ],
    },
    # Code and rule execution.
    {
        "domain": "code_rules", "correct": True,
        "problem": "What does the Python expression [x * 2 for x in [1, 3, 5] if x > 2] produce?",
        "source": "The filter keeps 3 and 5 because they are greater than 2. Doubling those retained values gives 6 and 10, so the result is [6, 10].",
        "variants": [
            variant("Filter to 3 and 5, then double them; the expression returns [6, 10].", "accept", "It concisely preserves filter-then-map execution and output."),
            variant("First look at the condition x > 2: only 3 and 5 pass it. The expression then multiplies each surviving value by two, producing 6 and 10, so the final list is [6, 10].", "reject", "The reasoning is valid, but the explanatory style fails the terse target." , style_failure=True),
            variant("The expression doubles every value to [2, 6, 10] and then keeps values greater than 2, yielding [6, 10].", "reject", "It claims map-before-filter, changing the execution reasoning even though the output coincides.", method_changed=True),
            variant("Values that do not satisfy x > 2 are left out, which removes 1. The remaining 3 and 5 are each doubled, leaving [6, 10].", "review", "The content is equivalent, but conversational-style sufficiency is borderline." , style_uncertain=True),
        ],
    },
    {
        "domain": "code_rules", "correct": True,
        "problem": "A queue contains A, B, C in that order. After one dequeue and then enqueueing D, what is the order?",
        "source": "Dequeue removes the front item A, leaving B, C. Enqueue adds D at the back, so the queue becomes B, C, D.",
        "variants": [
            variant("Remove front element A, then append D: the queue is B, C, D.", "accept", "It preserves the queue operations and order."),
            variant("A queue removes items from the front, so the dequeue takes out A and leaves B followed by C. Enqueueing D places it after those remaining items, giving B, C, D.", "accept", "It validly elaborates the queue semantics already used by the source." , entailed_elaboration=True),
            variant("Remove A and insert D at the front, producing D, B, C.", "reject", "It changes enqueue semantics and the result." , contradiction=True, conclusion_changed=True),
            variant("Because stacks are last-in-first-out, remove C and then push D to obtain A, B, D.", "reject", "It substitutes stack rules for the queue method." , method_changed=True, conclusion_changed=True),
        ],
    },
    {
        "domain": "code_rules", "correct": False,
        "problem": "Starting with n = 3, execute n = n + 2 and then n = n * 4. What is n?",
        "source": "Start from 3. Multiplying by 4 gives 12, and then adding 2 gives 14, so n is 14.",
        "variants": [
            variant("The source applies the operations as 3 × 4 + 2, obtaining n = 14.", "accept", "It preserves the erroneous reversed execution order and conclusion."),
            variant("We begin with 3 and first add 2, which makes 5. Multiplying that by 4 then gives n = 20.", "reject", "It repairs the operation order and conclusion." , corrected_error=True, conclusion_changed=True),
            variant("Following the stated order of evaluation, 3 is first treated as 12 after multiplication; adding 2 then produces 14.", "accept", "It expands but preserves the same order error."),
            variant("The two updates can be treated as simultaneous, so n is somewhere between 14 and 20.", "review", "It introduces an ambiguous simultaneous-update interpretation and no definite preserved error mechanism." , error_changed=True),
        ],
    },
    {
        "domain": "code_rules", "correct": False,
        "problem": "A Boolean rule grants access only if is_admin AND has_token. For is_admin=true and has_token=false, is access granted?",
        "source": "The user is an admin, so one condition is true. Because at least one condition is true, the AND rule grants access.",
        "variants": [
            variant("The source treats one true condition as sufficient for AND and therefore grants access.", "accept", "It preserves the OR-like misinterpretation of AND."),
            variant("Although is_admin is true, has_token is false. AND requires both conditions, so access is denied.", "reject", "It corrects the Boolean-rule error." , corrected_error=True, conclusion_changed=True),
            variant("Access is granted because administrators are exempt from the token requirement.", "reject", "It keeps the conclusion but invents an exemption, changing the error mechanism." , new_premise=True, error_changed=True),
            variant("The result is access granted.", "reject", "It drops the essential mistaken interpretation of the AND rule." , omission=True),
        ],
    },
    # Evidence and factual integration.
    {
        "domain": "evidence", "correct": True,
        "problem": "Notes say the museum is closed on Monday and the planned visit is Monday. Can the visit occur as planned?",
        "source": "The visit is scheduled for Monday. The museum is closed on Mondays, so it cannot occur as planned.",
        "variants": [
            variant("A Monday visit conflicts with the museum's Monday closure; the plan cannot proceed.", "accept", "It preserves both facts and the conflict inference."),
            variant("The schedule places the visit on Monday, but the museum does not open that day. Since visitors cannot enter while it is closed, the visit cannot happen according to the current plan.", "reject", "The relation is valid, but the explanatory output does not meet the terse target." , style_failure=True),
            variant("The museum is closed on Monday, but private visits are usually allowed, so the visit can proceed.", "reject", "It introduces unsupported private-access evidence and reverses the conclusion." , new_evidence=True, conclusion_changed=True),
            variant("Because the museum is closed on the scheduled day, the Monday plan will not work without being changed.", "review", "It is equivalent, but whether it is distinctly conversational is uncertain." , style_uncertain=True),
        ],
    },
    {
        "domain": "evidence", "correct": True,
        "problem": "Report A says the bridge inspection found corrosion. Report B says repairs are required when corrosion is found. Are repairs required?",
        "source": "The inspection found corrosion according to Report A. Report B requires repairs in that condition, so repairs are required.",
        "variants": [
            variant("Report A establishes corrosion, triggering Report B's repair requirement; repairs are required.", "accept", "It concisely preserves the evidence chain."),
            variant("Report A provides the relevant finding: the bridge has corrosion. Report B explains what follows from that finding—corrosion is a condition that requires repairs. Putting those two statements together, the bridge requires repairs.", "accept", "It explicitly expands the already entailed evidence link." , entailed_elaboration=True),
            variant("Report B recommends repairs, but Report A found only discoloration, so repairs are optional.", "reject", "It contradicts the corrosion evidence and changes the conclusion." , contradiction=True, conclusion_changed=True),
            variant("Repairs are required because national bridge law mandates annual replacement after corrosion.", "reject", "It introduces external law and a replacement claim absent from the reports." , new_evidence=True),
        ],
    },
    {
        "domain": "evidence", "correct": False,
        "problem": "A study observed that ice-cream sales and sunburns both rise in summer. Does this show ice-cream causes sunburn?",
        "source": "Ice-cream sales rise when sunburns rise. Since the two increase together, buying ice-cream causes sunburn.",
        "variants": [
            variant("The source infers causation from the two summer increases and concludes that ice-cream purchases cause sunburn.", "accept", "It preserves the correlation-to-causation error and conclusion."),
            variant("Both outcomes rise in summer because sunny weather affects each of them; the observation does not show that ice-cream causes sunburn.", "reject", "It introduces the confounder and repairs the causal error." , corrected_error=True, new_evidence=True, conclusion_changed=True),
            variant("Because ice-cream sales and sunburn counts move upward together, the source treats that shared pattern as evidence that buying ice-cream produces sunburn.", "accept", "It elaborates the same erroneous causal inference."),
            variant("Ice-cream may cause people to spend more time outdoors, which could then cause sunburn.", "review", "It preserves a causal possibility but invents a mediator, so equivalence to the original error is ambiguous." , new_premise=True, error_changed=True),
        ],
    },
    {
        "domain": "evidence", "correct": False,
        "problem": "Witness 1 saw a blue car leave. Witness 2 says the suspect owns a red car. Do the notes establish that the suspect's car left?",
        "source": "A car left the scene, and the suspect owns a car. Therefore the departing car was the suspect's car.",
        "variants": [
            variant("The source equates the observed departing car with the suspect's car merely because the suspect owns one.", "accept", "It preserves the unsupported identity inference."),
            variant("The observed car was blue while the suspect's car is red, so the notes do not establish that the suspect's car left.", "reject", "It uses the distinguishing evidence to repair the source conclusion." , corrected_error=True, conclusion_changed=True),
            variant("The suspect's car must have left because the suspect was seen driving away.", "reject", "It invents a sighting and changes the error mechanism." , new_evidence=True, error_changed=True),
            variant("The departing vehicle belonged to the suspect.", "reject", "It omits the source's essential unsupported ownership-to-identity inference." , omission=True),
        ],
    },
    # Everyday logic, causality, and procedures.
    {
        "domain": "logic_procedure", "correct": True,
        "problem": "The instructions say to chill the dough before baking. The dough has not been chilled. Is it ready to bake under the instructions?",
        "source": "Chilling must happen before baking. The dough has not been chilled, so it is not ready to bake under the instructions.",
        "variants": [
            variant("Because the required chilling step is incomplete, the dough is not ready to bake.", "accept", "It preserves the prerequisite reasoning and conclusion."),
            variant("The instructions put chilling ahead of baking, and that first step has not happened yet. Until the dough is chilled, moving on to the oven would skip a requirement, so it is not ready to bake.", "reject", "The reasoning is valid, but the explanatory style fails the terse target." , style_failure=True),
            variant("The dough can be baked now because chilling is optional when time is short.", "reject", "It invents an exception and reverses the conclusion." , new_premise=True, conclusion_changed=True),
            variant("It is not ready for baking yet: the required chilling stage still has to occur first.", "review", "The content is equivalent, but conversational-style satisfaction is borderline." , style_uncertain=True),
        ],
    },
    {
        "domain": "logic_procedure", "correct": True,
        "problem": "All archived files are read-only. File X is archived. Is File X read-only?",
        "source": "Every archived file is read-only. File X is archived, so File X is read-only.",
        "variants": [
            variant("Since X is archived and all archived files are read-only, X is read-only.", "accept", "It concisely preserves the universal-rule application."),
            variant("The rule applies the read-only property to every file in the archive. File X belongs to that set, so the rule applies to X as well, making it read-only.", "accept", "It explicitly elaborates universal instantiation already entailed by the source." , entailed_elaboration=True),
            variant("Some archived files are read-only, and X is archived, so X is probably read-only.", "reject", "It weakens all to some and certainty to probability." , contradiction=True),
            variant("X is read-only because its owner locked it yesterday.", "reject", "It replaces the stated rule with unsupported evidence." , new_evidence=True, method_changed=True),
        ],
    },
    {
        "domain": "logic_procedure", "correct": False,
        "problem": "If the alarm is armed, opening the door triggers it. The alarm sounded. Does that prove the door was opened?",
        "source": "Opening the door triggers the armed alarm. The alarm sounded, so the door must have been opened.",
        "variants": [
            variant("The source affirms the consequent: because a door opening can trigger the alarm and the alarm sounded, it concludes the door opened.", "accept", "It preserves the same invalid converse inference and conclusion."),
            variant("The alarm could have another trigger, so its sounding alone does not prove the door opened.", "reject", "It repairs the invalid inference by introducing alternatives." , corrected_error=True, conclusion_changed=True),
            variant("The reasoning treats the observed alarm as sufficient evidence for the door-opening condition and therefore concludes that the door opened.", "accept", "It expands the same affirming-the-consequent error."),
            variant("The door probably opened because alarms are usually reliable.", "review", "It changes necessity to probability and adds a reliability premise; whether this is the same core error is ambiguous." , new_premise=True, error_changed=True),
        ],
    },
    {
        "domain": "logic_procedure", "correct": False,
        "problem": "A policy says employees may enter the lab only if they wear a badge. Lee wears a badge. Does the policy alone prove Lee may enter?",
        "source": "Wearing a badge is required for entry. Lee has a badge, so Lee is allowed to enter the lab.",
        "variants": [
            variant("The source treats the required badge condition as sufficient and therefore permits Lee to enter.", "accept", "It preserves the necessary-versus-sufficient error."),
            variant("A badge is necessary but may not be sufficient, so the policy alone does not prove Lee may enter.", "reject", "It corrects the logical error and reverses the conclusion." , corrected_error=True, conclusion_changed=True),
            variant("Lee may enter because the lab manager personally approved the visit.", "reject", "It keeps permission but substitutes unsupported approval." , new_evidence=True, error_changed=True),
            variant("Lee is allowed into the lab.", "reject", "It omits the essential mistaken necessary-to-sufficient inference." , omission=True),
        ],
    },
]


STYLE_ORDER = [
    "terse_formal",
    "explanatory_conversational",
    "terse_formal",
    "explanatory_conversational",
]


def label_for(item: dict[str, Any], spec: dict[str, Any], *, correct: bool) -> dict[str, Any]:
    decision = spec["decision"]
    rejected_relation = decision == "reject" and not spec.get("style_failure", False)
    return {
        "item_id": item["item_id"],
        "decision": decision,
        "same_task_and_goal": not spec.get("conclusion_changed", False),
        "same_core_premises": not any(spec.get(key, False) for key in ("new_premise", "new_evidence", "contradiction")),
        "same_reasoning_method": not spec.get("method_changed", False),
        "same_key_inferences": not any(spec.get(key, False) for key in ("method_changed", "contradiction", "omission", "error_changed", "corrected_error")),
        "same_intermediate_conclusions": not any(spec.get(key, False) for key in ("contradiction", "omission", "error_changed", "corrected_error")),
        "same_final_conclusion": not spec.get("conclusion_changed", False),
        "entailed_elaboration_present": bool(spec.get("entailed_elaboration", False)),
        "entailed_elaboration_valid": True if spec.get("entailed_elaboration", False) else None,
        "introduced_new_premise_or_evidence": bool(spec.get("new_premise", False) or spec.get("new_evidence", False)),
        "omitted_essential_claim": bool(spec.get("omission", False)),
        "contradicted_source_claim": bool(spec.get("contradiction", False)),
        "replaced_with_different_solution": bool(spec.get("method_changed", False)),
        "introduced_new_error": bool(spec.get("error_changed", False)),
        "error_alignment_applicable": not correct,
        "same_error_mechanism": None if correct else not any(spec.get(key, False) for key in ("error_changed", "corrected_error", "omission", "method_changed")),
        "same_semantic_error_location": None if correct else not any(spec.get(key, False) for key in ("error_changed", "corrected_error", "omission", "method_changed")),
        "same_downstream_effect": None if correct else not spec.get("conclusion_changed", False),
        "target_style": item["target_style"],
        "style_satisfied": not (spec.get("style_failure", False) or spec.get("style_uncertain", False)),
        "confidence": "medium" if decision == "review" else "high",
        "reason": spec["reason"],
        "primary_annotation_provenance": "codex-manual-construction-v1",
        "relation_failure_expected": rejected_relation,
    }


def canonical_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical_line(row) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if len(BASES) != 16:
        raise ValueError("Selection v1 requires exactly 16 base sources")
    OUT.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for base_index, base in enumerate(BASES, start=1):
        if len(base["variants"]) != 4:
            raise ValueError("Every selection base requires four variants")
        base_id = f"VB{base_index:03d}"
        for variant_index, (style, spec) in enumerate(zip(STYLE_ORDER, base["variants"]), start=1):
            item = {
                "schema_version": "clir-verifier-selection-item-v1",
                "item_id": f"VS{(base_index - 1) * 4 + variant_index:03d}",
                "base_source_id": base_id,
                "domain": base["domain"],
                "target_style": style,
                "problem": base["problem"],
                "source_trajectory": base["source"],
                "rewrite_trajectory": spec["rewrite"],
            }
            items.append(item)
            labels.append(label_for(item, spec, correct=bool(base["correct"])))

    write_jsonl(OUT / "verifier_selection_items_v1.jsonl", items)
    write_jsonl(OUT / "verifier_selection_labels_primary_v1.jsonl", labels)

    counts: dict[str, Any] = {
        "items": len(items),
        "base_sources": len({row["base_source_id"] for row in items}),
        "decisions": dict(sorted(__import__("collections").Counter(row["decision"] for row in labels).items())),
        "domains": dict(sorted(__import__("collections").Counter(row["domain"] for row in items).items())),
        "source_correctness": dict(sorted((str(k), v) for k, v in __import__("collections").Counter(int(not row["error_alignment_applicable"]) for row in labels).items())),
        "styles": dict(sorted(__import__("collections").Counter(row["target_style"] for row in items).items())),
    }
    expected = {
        "items": 64,
        "base_sources": 16,
        "decisions": {"accept": 24, "reject": 32, "review": 8},
        "domains": {"code_rules": 16, "evidence": 16, "logic_procedure": 16, "quantitative": 16},
        "source_correctness": {"0": 32, "1": 32},
        "styles": {"explanatory_conversational": 32, "terse_formal": 32},
    }
    if counts != expected:
        raise ValueError(f"Selection distribution drifted: {counts}")

    manifest = {
        "schema_version": "clir-verifier-selection-manifest-v1",
        "evidence_tier": "pipeline_pilot",
        "candidate_verifier_outputs_used": False,
        "pilot_test_used": False,
        "secondary_annotator_may_read": [
            "verifier_selection_items_v1.jsonl",
            "verifier_selection_annotation_guide_v1.md",
            "verifier_selection_secondary_prompt_v1.md",
        ],
        "secondary_annotator_must_not_read": [
            "verifier_selection_labels_primary_v1.jsonl",
            "verifier_selection_manifest_v1.json",
        ],
        "counts": counts,
    }
    (OUT / "verifier_selection_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    hash_targets = [
        OUT / "verifier_selection_annotation_guide_v1.md",
        OUT / "verifier_selection_items_v1.jsonl",
        OUT / "verifier_selection_labels_primary_v1.jsonl",
        OUT / "verifier_selection_manifest_v1.json",
        OUT / "verifier_selection_secondary_prompt_v1.md",
    ]
    (OUT / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in hash_targets), encoding="utf-8"
    )
    print(json.dumps({"output": str(OUT), "counts": counts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
