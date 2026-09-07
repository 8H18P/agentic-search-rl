#!/usr/bin/env python3
"""Build frozen, manually labelled V3 input-isolation calibration sets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RETRIEVAL_EXPECTED = {
    # Named/scope regressions, manually reviewed against query-local evidence.
    "d3bbec65-30b6-584f-b091-ac6a837b107c::action_12::query_2": (1, "known_scope_regression", "Result explicitly gives Cincinnati Bengals, third round, 83rd overall."),
    "b93bb253-5259-5448-8f6f-476e56d5228a::action_2::query_1": (1, "known_scope_regression", "Result identifies the 2012 World Grand Prix as the first major title."),
    "82c71c02-84aa-531b-9251-55e138b55206::action_8::query_1": (1, "known_scope_regression", "Result establishes the 1535 Sforza succession context and Charles V."),
    "304e018e-9547-582c-8192-e12f14b439a0::action_15::query_3": (0, "known_scope_regression", "Result contains no Horkheimer naturalization date."),
    "e0cc9b79-ee60-512d-89c2-a35b6bc682a9::action_6::query_0": (0, "known_scope_regression", "Result contains no Weekly Shonen Magazine founding date."),
    "48dfa947-d045-5f52-b2e8-b6d06116542c::action_5::query_1": (0, "known_scope_regression", "Result says roughly 30,000 years, not the requested 36,000 BP fact."),
    "87a5476a-d070-511f-a441-d61158d790cb::action_8::query_0": (0, "known_scope_regression", "Result gives multiple censored papers but does not identify the exact heavily-censored newspaper requested."),
    # Ten clear query-local successes.
    "d7c037fa-12db-5fc2-b53e-4c6b9d944c2e::action_3::query_0": (1, "clear_success", "Result directly states If launched in March 1952 by Quinn Publications."),
    "27f8e5f2-b27b-5493-8af2-8c9b0a575d21::action_3::query_0": (1, "clear_success", "Result identifies Adam Hyde and Reuben Styles/Peking Duk collaboration."),
    "c725f655-e497-5ee1-bead-dbe7d843b829::action_3::query_0": (1, "clear_success", "Result directly supplies the queried Blyton work and allegory relation."),
    "f3dac62b-4950-5559-82ff-df8bc5556022::action_4::query_0": (1, "clear_success", "Result supplies Masud's seizure of the Ghaznavid throne."),
    "f18ea4b1-dc9a-5390-83c9-f55cfbd18ca2::action_4::query_0": (1, "clear_success", "Result explicitly describes Tamagotchi as egg-shaped."),
    "fa1fdb89-80da-55d2-b34d-344e6d7c6cbc::action_3::query_0": (1, "clear_success", "Result confirms Yuen Woo-ping choreographed Ip Man 3."),
    "b9e33635-40eb-5c01-9b82-218676e93d98::action_3::query_0": (1, "clear_success", "Result confirms Robert Altman directed The Company."),
    "cedd6ad7-5c5c-5095-9620-cc38c724ec42::action_1::query_0": (1, "clear_success", "Result identifies Antony King's band association."),
    "40557ecd-008f-5101-9fe4-6a6241de668a::action_3::query_0": (1, "clear_success", "Result gives the location of 1221 Avenue of the Americas."),
    "aeba3422-f47f-5917-9c06-f9da2b3f4f0d::action_3::query_1": (1, "clear_success", "Result places Casa de Sierra Nevada in Guanajuato."),
    # Ten clear retrieval misses.
    "fcffc6de-8bff-5154-b2bb-9e98338afa05::action_9::query_1": (0, "clear_miss", "Results do not identify the film director."),
    "7ef656bb-5362-564d-8092-172009a38c9c::action_5::query_1": (0, "clear_miss", "Results do not establish the Penge mine discovery fact."),
    "de02cf87-ef7b-5d93-aa12-3abc51249d98::action_12::query_4": (0, "clear_miss", "Results do not give Heliothis Schrank 1802 authority."),
    "94ea873a-f4a9-5ffb-a664-d499469c2bce::action_4::query_1": (0, "clear_miss", "Results do not identify Dragasakis's father."),
    "51381e0e-1600-576b-a51f-a9aa21730fcb::action_13::query_2": (0, "clear_miss", "Results do not identify the actor for Peppino Corradini."),
    "0c3cc99f-f7e5-5f37-b7bc-476fa2e5de2b::action_10::query_1": (0, "clear_miss", "Results are unrelated to the requested 2011 actor."),
    "2ce48921-d254-5572-91e0-57c5aefb6030::action_6::query_2": (0, "clear_miss", "Results omit the requested New Zealand chart fact."),
    "6fc00b53-260d-5654-a059-e7561bd4a174::action_10::query_0": (0, "clear_miss", "Results do not provide Mario Party Superstars minigame count."),
    "0176ebf1-d106-516d-b14a-690968d9d196::action_11::query_2": (0, "clear_miss", "Results do not supply the requested McGill postdoc dates."),
    "4b811096-0abf-5167-9851-3f68ecfce8ac::action_3::query_0": (0, "clear_miss", "Results do not locate Molla Amirkhan."),
    # Ten partial/entity-mismatch negatives.
    "fb2d9601-5b1c-52f3-99ac-b69e24d332a6::action_1::query_1": (0, "partial_or_entity_mismatch", "Result does not provide the RoboCop 3 starring cast requested."),
    "6186e5ad-6d5b-54ab-be28-8034b840df25::action_1::query_1": (0, "partial_or_entity_mismatch", "Result is ambiguous across similarly titled songs."),
    "d931560b-9912-503e-899f-cb9854c155ac::action_9::query_3": (0, "partial_or_entity_mismatch", "Result does not confirm zero league titles."),
    "9307c276-1ad8-5dfe-98f7-662c60609057::action_1::query_0": (0, "partial_or_entity_mismatch", "Result does not identify the shared film."),
    "a41f924a-38f3-569b-a297-c15a63d5fe1f::action_3::query_1": (0, "partial_or_entity_mismatch", "Result does not establish the goalkeeper/cyclist relation."),
    "22eb38fa-4677-5b8e-90f5-e94fcbc1acc7::action_4::query_0": (0, "partial_or_entity_mismatch", "Result gives 1874 but not the exact requested signing date."),
    "7c301053-3a0b-58c6-8f30-e206632b972c::action_4::query_3": (0, "partial_or_entity_mismatch", "Result lacks the requested highest Vienna percentage."),
    "fbe0e2d6-a10f-57ba-9f08-4c5df693318d::action_10::query_0": (0, "partial_or_entity_mismatch", "Result gives year/opponent but not the requested exact debut fact."),
    "3edc1ead-debb-53e0-8054-881b338ef8aa::action_3::query_0": (0, "partial_or_entity_mismatch", "Result does not establish the queried Fragile X intervention claim."),
    "1dd65519-eaf2-5d3d-8b91-7fb6dd024e07::action_3::query_1": (0, "partial_or_entity_mismatch", "Result gives 1969 but not the requested exact first-fight date."),
}


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def history_text(history: list[dict]) -> str:
    if not history:
        return "(No previous Search action.)"
    return "\n\n".join(
        f"Previous action {s['action_idx']} reasoning:\n{s['reasoning']}\nPrevious executed action:\n{json.dumps(s['executed_action'], ensure_ascii=False)}\nPrevious authoritative observation:\n{s['observation']}"
        for s in history
    )


def retrieval_prompt(query: str, result: str) -> str:
    return f"""You are a query-local retrieval evaluator. Judge only whether CURRENT QUERY-SPECIFIC RESULT provides the information requested by CURRENT EXECUTED QUERY.

Use no unstated original question, history, sibling query, future constraint, or final answer. The result need not solve any larger task. Related entities alone are insufficient; the intended and retrieved entity must match. If the query asks several explicit fields, all are needed. Natural search intent may be read normally, but do not add requirements absent from the query.

Regression examples:
- Query: Mohamed Sanu 2012 NFL Draft round pick. Result: selected by Cincinnati Bengals in the third round of the 2012 NFL Draft. => retrieval=1. Do not require proof of bowl MVP status.
- Query: Michael van Gerwen 2012 breakthrough first major title. Result: won the 2012 World Grand Prix, his first major title. => retrieval=1. Do not require the final opponent unless queried.
- Query: Max Horkheimer US naturalization before Adorno. Result: only says he moved to the US in 1934, with no naturalization date. => retrieval=0.
- Query: Weekly Shonen Magazine founding date. Result: contains no founding date. => retrieval=0.

Return exactly:
<retrieval>0 or 1</retrieval>
<retrieval_failure_type>none, retrieval_miss, retrieval_partial, result_entity_mismatch, or other</retrieval_failure_type>
<explanation>brief query-local justification</explanation>

CURRENT EXECUTED QUERY:
{query}

CURRENT QUERY-SPECIFIC RESULT:
{result}
"""


def intent_prompt(unit: dict) -> str:
    return f"""You are an intent evaluator. Judge whether CURRENT EXECUTED QUERY was a reasonable, task-relevant, actionable information search before it was executed.

Use only the original question, golden answer, authoritative history before this Search action, current action reasoning, and current query. You are not shown and must not infer whether retrieval succeeded. Retrieval failure cannot make intent invalid. Verification or repetition is not an intent failure and belongs to a separate efficiency diagnostic.

Intent=0 only for a clearly irrelevant query, a clearly wrong entity already disproved by the shown context, a malformed/nonsensical query, or a query with no reasonable connection to the task.

Return exactly:
<intent>0 or 1</intent>
<intent_failure_type>none, irrelevant, wrong_entity, malformed, or other</intent_failure_type>
<explanation>brief justification about intent only</explanation>

ORIGINAL QUESTION:
{unit['question']}

GOLDEN ANSWER:
{json.dumps(unit['golden_answer'], ensure_ascii=False)}

AUTHORITATIVE HISTORY BEFORE CURRENT SEARCH ACTION:
{history_text(unit['authoritative_history_before_action'])}

CURRENT ACTION REASONING:
{unit['current_action_reasoning'] or '(none recorded)'}

CURRENT EXECUTED QUERY:
{unit['current_executed_query']}
"""


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows: fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--v2-units", type=Path, required=True); p.add_argument("--v2-results", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True); args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise SystemExit("refusing non-empty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_units = [json.loads(x) for x in args.v2_units.read_text(encoding="utf-8").splitlines() if x.strip()]
    by_id = {u["unit_id"]: u for u in all_units}
    v2_results = {r["unit_id"]: r for r in (json.loads(x) for x in args.v2_results.read_text(encoding="utf-8").splitlines() if x.strip())}
    if set(RETRIEVAL_EXPECTED) - set(by_id): raise SystemExit("missing manual retrieval units")
    retrieval = []
    for uid, (expected, stratum, rationale) in RETRIEVAL_EXPECTED.items():
        u = by_id[uid]; pr = retrieval_prompt(u["current_executed_query"], u["current_query_result"])
        retrieval.append({"calibration_id": "retrieval::" + uid, "unit_id": uid, "trajectory_id": u["trajectory_id"], "action_idx": u["action_idx"], "query_idx": u["query_idx"], "current_executed_query": u["current_executed_query"], "current_query_result": u["current_query_result"], "expected_retrieval": expected, "expected_label_source": "manual_pre_api", "expected_rationale": rationale, "stratum": stratum, "efficiency_note": v2_results[uid]["parsed"]["efficiency_note"], "efficiency_affects_score": False, "judge_prompt": pr, "prompt_sha256": hash_text(pr)})
    # Twenty unmistakably task-relevant natural positives: ten clear-success and ten clear-miss queries.
    positive_ids = [uid for uid, (_, s, _) in RETRIEVAL_EXPECTED.items() if s == "clear_success"] + [uid for uid, (_, s, _) in RETRIEVAL_EXPECTED.items() if s == "clear_miss"]
    intent = []
    for uid in positive_ids:
        u = by_id[uid]; pr = intent_prompt(u)
        intent.append({"calibration_id": "intent_positive::" + uid, "case_type": "natural_positive", "source_unit_id": uid, "trajectory_id": u["trajectory_id"], "question": u["question"], "golden_answer": u["golden_answer"], "authoritative_history_before_action": u["authoritative_history_before_action"], "current_action_reasoning": u["current_action_reasoning"], "current_executed_query": u["current_executed_query"], "expected_intent": 1, "expected_label_source": "manual_pre_api", "judge_prompt": pr, "prompt_sha256": hash_text(pr)})
    # Controlled negatives: preserve a real context/reasoning but rotate in an unrelated real query.
    for i, context_uid in enumerate(positive_ids):
        query_uid = positive_ids[(i + 10) % len(positive_ids)]
        context, query_source = by_id[context_uid], by_id[query_uid]
        synthetic = dict(context); synthetic["current_executed_query"] = query_source["current_executed_query"]
        pr = intent_prompt(synthetic)
        intent.append({"calibration_id": f"intent_synthetic_negative::{context_uid}::{query_uid}", "case_type": "synthetic_counterfactual_negative", "context_source_unit_id": context_uid, "query_source_unit_id": query_uid, "trajectory_id": context["trajectory_id"], "question": context["question"], "golden_answer": context["golden_answer"], "authoritative_history_before_action": context["authoritative_history_before_action"], "current_action_reasoning": context["current_action_reasoning"], "current_executed_query": query_source["current_executed_query"], "expected_intent": 0, "expected_label_source": "controlled_counterfactual_pre_api", "judge_prompt": pr, "prompt_sha256": hash_text(pr)})
    # Ten paired natural positives with reasoning removed, for a narrow necessity ablation.
    for uid in positive_ids[:10]:
        u = dict(by_id[uid]); u["current_action_reasoning"] = ""
        pr = intent_prompt(u)
        intent.append({"calibration_id": "intent_reasoning_ablated::" + uid, "case_type": "natural_positive_reasoning_ablated", "source_unit_id": uid, "trajectory_id": u["trajectory_id"], "question": u["question"], "golden_answer": u["golden_answer"], "authoritative_history_before_action": u["authoritative_history_before_action"], "current_action_reasoning": "", "current_executed_query": u["current_executed_query"], "expected_intent": 1, "expected_label_source": "manual_pre_api", "judge_prompt": pr, "prompt_sha256": hash_text(pr)})
    if len(retrieval) != 37 or len(intent) != 50: raise SystemExit("unexpected calibration counts")
    # Physical input-boundary audits.
    audits = {
        "retrieval_units": len(retrieval), "intent_units": len(intent),
        "retrieval_forbidden_fields_present": sum(any(k in r for k in ["question", "golden_answer", "authoritative_history_before_action", "current_action_reasoning", "final_answer"]) for r in retrieval),
        "intent_result_fields_present": sum(any(k in r for k in ["current_query_result", "query_result", "final_answer"]) for r in intent),
        "retrieval_prompt_contains_original_question_marker": sum("ORIGINAL QUESTION:" in r["judge_prompt"] for r in retrieval),
        "intent_prompt_contains_result_marker": sum("CURRENT QUERY-SPECIFIC RESULT:" in r["judge_prompt"] for r in intent),
    }
    audits["all_pass"] = not any(v for k, v in audits.items() if k.endswith("present") or k.endswith("marker"))
    if not audits["all_pass"]: raise SystemExit(f"input isolation failed: {audits}")
    write_jsonl(args.output_dir / "retrieval_calibration_units.jsonl", retrieval); write_jsonl(args.output_dir / "intent_calibration_units.jsonl", intent)
    (args.output_dir / "input_isolation_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"status": "v3_targeted_calibration_unsubmitted", "api_called": False, "model": "qwen3.5-flash", "temperature": 0, "enable_thinking": False, "retrieval_units": 37, "intent_units": 50, "intent_breakdown": {"natural_positive": 20, "synthetic_counterfactual_negative": 20, "natural_positive_reasoning_ablated": 10, "natural_negative": 0}, "natural_negative_note": "No confidently labelled natural invalid-intent cases were found in the fixed structural-pass pilot; synthetic negatives are sanity-check only.", "expected_labels_frozen_before_api": True, "input_isolation": audits, "process_score": "deterministic intent AND retrieval; not model-generated", "efficiency_affects_score": False}
    (args.output_dir / "calibration_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
