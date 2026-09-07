#!/usr/bin/env python3
"""Offline consistency audit for the frozen V3 Process Judge."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("v3_builder_for_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--artifact-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    builder_path = args.repo / "scripts/prm/build_query_level_judge_v3_calibration.py"
    runner_path = args.repo / "scripts/prm/run_query_level_judge_v3_channel.py"
    aggregate_path = args.repo / "scripts/prm/aggregate_query_level_judge_v3_calibration.py"
    reconstruction_path = args.repo / "scripts/prm/build_query_level_judge_pilot.py"
    builder = load_module(builder_path)
    sentinels = {
        "question": "SENTINEL_ORIGINAL_QUESTION",
        "golden_answer": ["SENTINEL_GOLD"],
        "authoritative_history_before_action": [{"action_idx": 1, "reasoning": "SENTINEL_HISTORY_REASONING", "executed_action": {"name": "search", "arguments": {"query": ["old"]}}, "observation": "SENTINEL_PRIOR_RESULT"}],
        "current_action_reasoning": "SENTINEL_CURRENT_REASONING",
        "current_executed_query": "SENTINEL_CURRENT_QUERY",
        "current_query_result": "SENTINEL_CURRENT_RESULT",
    }
    intent = builder.intent_prompt(sentinels)
    retrieval = builder.retrieval_prompt(sentinels["current_executed_query"], sentinels["current_query_result"])
    runner_source = runner_path.read_text(encoding="utf-8")
    aggregate_source = aggregate_path.read_text(encoding="utf-8")
    reconstruction_source = reconstruction_path.read_text(encoding="utf-8")
    aggregate_tree = ast.parse(aggregate_source)
    has_and = any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitAnd) for n in ast.walk(aggregate_tree))
    checks = {
        "intent_contains_question": sentinels["question"] in intent,
        "intent_contains_gold": "SENTINEL_GOLD" in intent,
        "intent_contains_prior_history": "SENTINEL_PRIOR_RESULT" in intent,
        "intent_contains_reasoning": sentinels["current_action_reasoning"] in intent,
        "intent_contains_query": sentinels["current_executed_query"] in intent,
        "intent_excludes_current_result": sentinels["current_query_result"] not in intent,
        "retrieval_contains_query": sentinels["current_executed_query"] in retrieval,
        "retrieval_contains_current_result": sentinels["current_query_result"] in retrieval,
        "retrieval_excludes_original_question": sentinels["question"] not in retrieval and "ORIGINAL QUESTION:" not in retrieval,
        "retrieval_excludes_gold": "SENTINEL_GOLD" not in retrieval and "GOLDEN ANSWER:" not in retrieval,
        "retrieval_excludes_history": "SENTINEL_PRIOR_RESULT" not in retrieval and "AUTHORITATIVE HISTORY" not in retrieval,
        "retrieval_excludes_reasoning": sentinels["current_action_reasoning"] not in retrieval and "CURRENT ACTION REASONING:" not in retrieval,
        "retrieval_builder_signature_is_query_result_only": list(__import__('inspect').signature(builder.retrieval_prompt).parameters) == ["query", "result"],
        "intent_schema_has_no_answer_tag": "<answer>" not in intent,
        "retrieval_schema_has_no_answer_tag": "<answer>" not in retrieval,
        "separate_intent_parser_present": "INTENT_RE" in runner_source,
        "separate_retrieval_parser_present": "RETRIEVAL_RE" in runner_source,
        "process_score_deterministic_and": has_and and "model_generated_answer_field':False".replace("'", '"') not in aggregate_source,
        "query_specific_positional_parser_present": "def parse_sections" in reconstruction_source and "zip(queries, matches)" in reconstruction_source,
        "history_before_action_only": "sanitized_history(steps[: action_idx - 1])" in reconstruction_source,
        "raw_textual_action_not_serialized": '"raw_textual_action" in json.dumps(unit' in reconstruction_source,
        "artifact_input_isolation_pass": json.loads((args.artifact_dir / "input_isolation_audit.json").read_text())["all_pass"],
    }
    # The aggregate implementation is the executed evidence for deterministic AND.
    checks["process_score_deterministic_and"] = has_and and "'process_score':x['parsed']['intent'] & rr[rid]['parsed']['retrieval']" in aggregate_source
    passed = all(checks.values())
    lines = ["# V3 Freeze Consistency Audit", "", f"Overall: {'PASS' if passed else 'FAIL'}", ""]
    for name, value in checks.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")
    lines += ["", "No API, Retriever, calibration, held-out validation, or full Judge call was made by this audit."]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pass": passed, "checks": checks}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
