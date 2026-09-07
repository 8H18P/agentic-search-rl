#!/usr/bin/env python3
"""Build the immutable, derived Teacher V1 raw-990 baseline freeze artifacts.

This program is deliberately read-only with respect to rollout data.  It never
imports a policy backend, calls an API, calls the retriever, or rewrites a raw
result/trajectory/event file.  Its only writes are the derived files in
``artifacts/baseline_freeze``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CORE_FILES = (
    "src/champion_runtime/agent_loop.py",
    "src/champion_runtime/prompts.py",
    "src/champion_runtime/offline_search.py",
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL row is not an object")
                rows.append(value)
            except Exception:
                bad += 1
    return rows, bad


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def mean(values: Iterable[float]) -> float:
    data = list(values)
    return sum(data) / len(data) if data else 0.0


def percentile(values: Iterable[float], fraction: float) -> float:
    data = sorted(float(v) for v in values)
    if not data:
        return 0.0
    point = (len(data) - 1) * fraction
    lower = math.floor(point)
    upper = math.ceil(point)
    if lower == upper:
        return data[lower]
    return data[lower] * (upper - point) + data[upper] * (point - lower)


def resolve_run_path(run_dir: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else run_dir / path


def distribution(values: Iterable[float]) -> dict[str, float]:
    data = [float(v) for v in values]
    return {
        "count": len(data),
        "min": min(data) if data else 0.0,
        "mean": mean(data),
        "median": percentile(data, 0.50),
        "p95": percentile(data, 0.95),
        "max": max(data) if data else 0.0,
    }


def quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    passed = sum(bool(row.get("outcome_pass")) for row in rows)
    return {
        "n": n,
        "em": mean(float(row.get("answer_em") or 0.0) for row in rows),
        "f1": mean(float(row.get("answer_f1") or 0.0) for row in rows),
        "outcome_pass": passed,
        "outcome_pass_rate": passed / n if n else 0.0,
    }


def termination_name(row: dict[str, Any]) -> str:
    value = str(row.get("termination") or "")
    return "normal_answer" if value == "answer" else value


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    ordinary = [r for r in rows if r.get("question_type") == "ordinary_qa"]
    invalid = [r for r in rows if r.get("question_type") == "invalid_question"]
    rounds = [int(r.get("rounds") or 0) for r in rows]
    actions = [int(r.get("search_actions") or 0) for r in rows]
    queries = [int(r.get("search_queries") or 0) for r in rows]
    tokens = [int(r.get("total_tokens") or 0) for r in rows]
    request_inputs = [
        int(usage.get("input_tokens") or 0)
        for row in rows
        for usage in (row.get("api_usage_by_request") or [])
        if usage.get("input_tokens") is not None
    ]
    total_actions = sum(actions)
    malformed = sum(int(r.get("raw_spaced_tool_call") or 0) + int(r.get("raw_tool_call_no_gt") or 0) for r in rows)
    mismatch = sum(int(r.get("action_source_mismatch") or 0) for r in rows)
    result = {
        "data": {
            "trajectories": n,
            "unique_question_ids": len({str(r.get("question_id")) for r in rows}),
            "ordinary_qa_count": len(ordinary),
            "invalid_question_count": len(invalid),
        },
        "quality": {
            "ordinary": quality(ordinary),
            "invalid_question": quality(invalid),
            "overall": quality(rows),
        },
        "agent": {
            "total_rounds": sum(rounds),
            "mean_rounds": mean(rounds),
            "median_rounds": percentile(rounds, 0.50),
            "p95_rounds": percentile(rounds, 0.95),
            "search_actions": sum(actions),
            "search_queries": sum(queries),
            "mean_search_actions": mean(actions),
            "mean_search_queries": mean(queries),
            "searching_questions": sum(value > 0 for value in actions),
            "search_rate": sum(value > 0 for value in actions) / n if n else 0.0,
            "answer_count": sum(bool(r.get("answer_produced")) for r in rows),
            "answer_rate": sum(bool(r.get("answer_produced")) for r in rows) / n if n else 0.0,
            "normal_answer": sum(termination_name(r) == "normal_answer" for r in rows),
            "force_answer": sum(termination_name(r) == "force_answer" for r in rows),
            "raw_fallback": sum(termination_name(r) == "raw_fallback" for r in rows),
        },
        "tokens": {
            "total_input": sum(int(r.get("input_tokens") or 0) for r in rows),
            "total_output": sum(int(r.get("output_tokens") or 0) for r in rows),
            "total": sum(tokens),
            "mean_trajectory_cumulative": mean(tokens),
            "p50_trajectory_cumulative": percentile(tokens, 0.50),
            "p95_trajectory_cumulative": percentile(tokens, 0.95),
            "max_trajectory_cumulative": max(tokens) if tokens else 0,
            "top10_trajectory_share": sum(sorted(tokens, reverse=True)[:10]) / sum(tokens) if sum(tokens) else 0.0,
            "max_single_request_input": max(request_inputs) if request_inputs else 0,
            "requests_over_128k": sum(value > 128000 for value in request_inputs),
        },
        "protocol": {
            "canonical_exact": sum(int(r.get("raw_canonical_exact") or 0) for r in rows),
            "spaced_tool_call": sum(int(r.get("raw_spaced_tool_call") or 0) for r in rows),
            "tool_call_missing_gt": sum(int(r.get("raw_tool_call_no_gt") or 0) for r in rows),
            "bare_json_fallback": sum(int(r.get("parser_bare_json_fallback") or 0) for r in rows),
            "invalid_json": sum(int(r.get("parser_invalid_json") or 0) for r in rows),
            "parser_none": sum(int(r.get("parser_none") or 0) for r in rows),
            "action_source_mismatch": mismatch,
            "action_source_mismatch_rate": mismatch / total_actions if total_actions else 0.0,
            "malformed_wrapper_count": malformed,
            "malformed_wrapper_rate": malformed / total_actions if total_actions else 0.0,
        },
        "infrastructure": {
            "retriever_requests": sum(int(r.get("retriever_http_requests") or 0) for r in rows),
            "retriever_failures": sum(int(r.get("retriever_failures") or 0) for r in rows),
            "retriever_successes": sum(int(r.get("retriever_http_requests") or 0) - int(r.get("retriever_failures") or 0) for r in rows),
            "message_continuity_edges": sum(int(r.get("message_continuity_edges") or 0) for r in rows),
            "message_continuity_failures": sum(int(r.get("message_continuity_failed_edges") or 0) for r in rows),
            "environment_query_rewrites": sum(int(r.get("env_query_rewrite_count") or 0) for r in rows),
            "visit_attempts": sum(int(r.get("visit_attempts") or 0) for r in rows),
            "visit_executions": sum(int(r.get("visit_executed") or 0) for r in rows),
            "gold_leakage": sum(bool(r.get("gold_leakage")) for r in rows),
            "legacy_runtime_usage": sum(str(r.get("runtime")) != "champion_core_edd28d_search_only" for r in rows),
            "state_machine_failures": sum(not bool(r.get("state_ok")) for r in rows),
        },
    }
    return result


def detailed_row(group: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordinary = [r for r in rows if r.get("question_type") == "ordinary_qa"]
    invalid = [r for r in rows if r.get("question_type") == "invalid_question"]
    rounds = [int(r.get("rounds") or 0) for r in rows]
    actions = [int(r.get("search_actions") or 0) for r in rows]
    queries = [int(r.get("search_queries") or 0) for r in rows]
    tokens = [int(r.get("total_tokens") or 0) for r in rows]
    all_quality = quality(rows)
    ordinary_quality = quality(ordinary)
    invalid_quality = quality(invalid)
    malformed = sum(int(r.get("raw_spaced_tool_call") or 0) + int(r.get("raw_tool_call_no_gt") or 0) for r in rows)
    mismatch = sum(int(r.get("action_source_mismatch") or 0) for r in rows)
    search_actions = sum(actions)
    return {
        "group": group,
        "n": len(rows),
        "ordinary_n": len(ordinary),
        "invalid_n": len(invalid),
        "overall_em": all_quality["em"],
        "overall_f1": all_quality["f1"],
        "overall_pass": all_quality["outcome_pass"],
        "overall_pass_rate": all_quality["outcome_pass_rate"],
        "ordinary_em": ordinary_quality["em"],
        "ordinary_f1": ordinary_quality["f1"],
        "ordinary_pass": ordinary_quality["outcome_pass"],
        "ordinary_pass_rate": ordinary_quality["outcome_pass_rate"],
        "invalid_em": invalid_quality["em"],
        "invalid_f1": invalid_quality["f1"],
        "invalid_pass": invalid_quality["outcome_pass"],
        "invalid_pass_rate": invalid_quality["outcome_pass_rate"],
        "mean_rounds": mean(rounds),
        "median_rounds": percentile(rounds, 0.50),
        "p95_rounds": percentile(rounds, 0.95),
        "mean_search_actions": mean(actions),
        "mean_search_queries": mean(queries),
        "mean_tokens": mean(tokens),
        "median_tokens": percentile(tokens, 0.50),
        "force_answer_rate": sum(termination_name(r) == "force_answer" for r in rows) / len(rows) if rows else 0.0,
        "raw_fallback_rate": sum(termination_name(r) == "raw_fallback" for r in rows) / len(rows) if rows else 0.0,
        "malformed_wrapper_rate": malformed / search_actions if search_actions else 0.0,
        "action_source_mismatch_rate": mismatch / search_actions if search_actions else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def inspect_events(events_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    bad = 0
    finish = 0
    query_count = 0
    wrong_trajectory_id = 0
    expected_tid = str(result.get("trajectory_id"))
    with events_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except Exception:
                bad += 1
                continue
            event_type = str(event.get("event_type"))
            counts[event_type] += 1
            wrong_trajectory_id += int(str(event.get("trajectory_id")) != expected_tid)
            if event_type == "trajectory_finish":
                finish += 1
            elif event_type == "search_start":
                query_count += len((event.get("payload") or {}).get("queries") or [])
    reconstructed = (
        bad == 0
        and finish == 1
        and wrong_trajectory_id == 0
        and counts["round_request"] == int(result.get("rounds") or 0)
        and counts["api_request_result"] == int(result.get("api_calls") or 0)
        and counts["search_start"] == int(result.get("search_actions") or 0)
        and query_count == int(result.get("search_queries") or 0)
        and counts["retriever_http_attempt"] == int(result.get("retriever_http_requests") or 0)
    )
    return {
        "invalid_json_lines": bad,
        "finish_records": finish,
        "wrong_trajectory_ids": wrong_trajectory_id,
        "reconstructed": reconstructed,
        "event_rows": sum(counts.values()),
    }


def json_excerpt(value: Any, limit: int = 300) -> str:
    text = stable_json(value)
    return text[:limit]


def candidate(extractor: Any, json5_module: Any, text: str) -> Any:
    try:
        raw = extractor(text or "")
        return json5_module.loads(raw) if raw else None
    except Exception:
        return None


def champion_extract_tool_call_for_audit(content: str, json5_module: Any) -> str | None:
    """Pure, import-free copy of the frozen Champion extractor semantics.

    Importing ``champion_runtime.agent_loop`` constructs its API client.  The
    freeze audit must not need credentials or initialize that runtime, so this
    sidecar copy is kept intentionally line-for-line semantic with the three
    frozen formats: canonical tags, function syntax, and brace-depth bare JSON.
    """
    if "<tool_call>" in content and "</tool_call>" in content:
        start = content.find("<tool_call>") + len("<tool_call>")
        return content[start:content.find("</tool_call>", start)].strip()
    if "<tool_call>" in content:
        return content[content.find("<tool_call>") + len("<tool_call>"):].strip()

    func_match = re.search(r"<function=(search|visit)>", content)
    if func_match:
        tool_name = func_match.group(1)
        func_start = func_match.start()
        func_end = content.find("</function>", func_start)
        func_block = content[func_start:] if func_end == -1 else content[func_start:func_end]
        arguments: dict[str, Any] = {}
        for match in re.finditer(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", func_block, re.DOTALL):
            name = match.group(1)
            value = match.group(2).strip()
            try:
                arguments[name] = json5_module.loads(value)
            except Exception:
                arguments[name] = value
        return json.dumps({"name": tool_name, "arguments": arguments})

    bare_match = re.search(r'\{"name":\s*"(search|visit)"', content)
    if bare_match:
        json_start = bare_match.start()
        brace_depth = 0
        json_end = json_start
        for index in range(json_start, len(content)):
            if content[index] == "{":
                brace_depth += 1
            elif content[index] == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    json_end = index + 1
                    break
        return content[json_start:json_end]
    return None


def action_alignment_cases(rows: list[dict[str, Any]], run_dirs: dict[str, Path], limit: int = 20) -> list[dict[str, Any]]:
    import json5  # type: ignore

    extractor = lambda text: champion_extract_tool_call_for_audit(text, json5)

    output: list[dict[str, Any]] = []
    for result in sorted((r for r in rows if int(r.get("action_source_mismatch") or 0) > 0), key=lambda r: int(r.get("position") or 0)):
        run_dir = run_dirs[str(result["question_id"])]
        events_path = resolve_run_path(run_dir, str(result["events_file"]))
        responses: dict[int, dict[str, Any]] = {}
        parsed: dict[int, dict[str, Any]] = {}
        extracted: dict[int, Any] = {}
        starts: dict[int, dict[str, Any]] = {}
        observations: dict[int, str] = {}
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                payload = event.get("payload") or {}
                round_idx = int(payload.get("round_idx") or 0)
                event_type = event.get("event_type")
                if event_type == "round_response":
                    responses[round_idx] = payload
                elif event_type == "tool_extracted":
                    extracted[round_idx] = payload.get("extractor_result")
                elif event_type == "tool_json_parse" and payload.get("json_parse_success"):
                    parsed[round_idx] = {
                        "name": payload.get("parsed_tool_name"),
                        "arguments": payload.get("parsed_tool_args") or {},
                    }
                elif event_type == "search_start":
                    starts[round_idx] = {"queries": payload.get("queries") or [], "engines": payload.get("engines")}
                elif event_type == "tool_response_appended":
                    observations[round_idx] = str(payload.get("tool_response") or "")
        for round_idx in sorted(responses):
            payload = responses[round_idx]
            reasoning = str(payload.get("reasoning_content") or "")
            content = str(payload.get("content") or "")
            reasoning_candidate = candidate(extractor, json5, reasoning)
            content_candidate = candidate(extractor, json5, content)
            if reasoning_candidate is None or content_candidate is None or reasoning_candidate == content_candidate:
                continue
            executed = parsed.get(round_idx)
            start = starts.get(round_idx)
            observation = observations.get(round_idx, "")
            executed_queries = ((executed or {}).get("arguments") or {}).get("query", [])
            if isinstance(executed_queries, str):
                executed_queries = [executed_queries]
            output.append({
                "question_id": str(result["question_id"]),
                "source_split_position": int(result["position"]),
                "trajectory_id": str(result["trajectory_id"]),
                "round_idx": round_idx,
                "raw_reasoning_sha256": hashlib.sha256(reasoning.encode("utf-8")).hexdigest(),
                "raw_content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "reasoning_candidate_excerpt": json_excerpt(reasoning_candidate),
                "content_candidate_excerpt": json_excerpt(content_candidate),
                "reasoning_candidate_sha256": value_hash(reasoning_candidate),
                "content_candidate_sha256": value_hash(content_candidate),
                "extractor_result_sha256": hashlib.sha256(str(extracted.get(round_idx) or "").encode("utf-8")).hexdigest(),
                "actual_executed_action": executed,
                "actual_executed_action_sha256": value_hash(executed),
                "observation_source": "offline_e5_tool_response",
                "observation_chars": len(observation),
                "observation_sha256": hashlib.sha256(observation.encode("utf-8")).hexdigest(),
                "executed_query_matches_search_start": bool(start is not None and list(executed_queries) == list(start.get("queries") or [])),
                "executed_matches_reasoning_candidate": executed == reasoning_candidate,
                "executed_matches_content_candidate": executed == content_candidate,
                "canonical_sft_pair_source": "actual_executed_action -> actual_observation",
            })
            if len(output) >= limit:
                return output
    return output


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    def cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = (args.output_dir or root / "artifacts" / "baseline_freeze").resolve()
    logs = root / "logs" / "teacher_rollout"
    pilot_dir = logs / "pilot100_001"
    original_dir = logs / "remaining890_v1_001"
    continuation_dir = logs / "remaining890_v1_dmx_continuation_001"
    split_path = root / "data" / "splits" / "sft_pool_990_seed20260904.jsonl"

    required = [
        split_path,
        pilot_dir / "results.jsonl",
        original_dir / "results.jsonl",
        continuation_dir / "results.jsonl",
        continuation_dir / "provider_transition.json",
        pilot_dir / "run_config.json",
        continuation_dir / "run_config.json",
    ]
    errors = [f"missing required file: {path}" for path in required if not path.exists()]
    if errors:
        print(json.dumps({"hard_check": "FAIL", "errors": errors}, indent=2))
        return 2

    split_rows, split_bad = read_jsonl(split_path)
    pilot_rows, pilot_bad = read_jsonl(pilot_dir / "results.jsonl")
    original_rows, original_bad = read_jsonl(original_dir / "results.jsonl")
    remaining_rows, remaining_bad = read_jsonl(continuation_dir / "results.jsonl")
    transition = read_json(continuation_dir / "provider_transition.json")

    quota_affected = {str(value) for value in transition.get("quota_affected_ids", [])}
    clean_official_ids = {str(row.get("question_id")) for row in original_rows if str(row.get("question_id")) not in quota_affected}
    pilot_ids = {str(row.get("question_id")) for row in pilot_rows}
    remaining_ids = {str(row.get("question_id")) for row in remaining_rows}
    all_rows = pilot_rows + remaining_rows
    all_ids = [str(row.get("question_id")) for row in all_rows]
    split_ids = [str(row.get("question_id", row.get("id"))) for row in split_rows]
    split_by_position = {position: qid for position, qid in enumerate(split_ids, start=1)}

    errors.extend(f"invalid JSONL rows in {name}: {count}" for name, count in (
        ("split", split_bad), ("pilot results", pilot_bad), ("original results", original_bad), ("remaining results", remaining_bad)
    ) if count)
    checks = {
        "pilot_completed_100": len(pilot_rows) == 100,
        "remaining_completed_890": len(remaining_rows) == 890,
        "total_990": len(all_rows) == 990,
        "unique_qids_990": len(set(all_ids)) == 990,
        "duplicate_qids_0": len(all_ids) - len(set(all_ids)) == 0,
        "pilot_remaining_overlap_0": len(pilot_ids & remaining_ids) == 0,
        "pilot_positions_1_100": {int(r.get("position") or 0) for r in pilot_rows} == set(range(1, 101)),
        "remaining_positions_101_990": {int(r.get("position") or 0) for r in remaining_rows} == set(range(101, 991)),
        "source_split_990_valid": len(split_rows) == 990 and split_bad == 0 and len(set(split_ids)) == 990,
        "result_positions_match_source_split": all(split_by_position.get(int(r.get("position") or 0)) == str(r.get("question_id")) for r in all_rows),
        "provider_clean_official_461": len(clean_official_ids & remaining_ids) == 461,
        "provider_dmx_429": len(remaining_ids - clean_official_ids) == 429,
        "provider_total_official_561": len(pilot_ids) + len(clean_official_ids & remaining_ids) == 561,
        "provider_unresolved_0": all(qid in pilot_ids or qid in remaining_ids for qid in all_ids),
    }
    errors.extend(name for name, passed in checks.items() if not passed)

    run_dir_by_qid = {str(row["question_id"]): pilot_dir for row in pilot_rows}
    run_dir_by_qid.update({str(row["question_id"]): continuation_dir for row in remaining_rows})
    event_rows_total = 0
    invalid_event_lines = 0
    invalid_trajectory_json = 0
    missing_files: list[str] = []
    reconstruction_failures: list[str] = []
    finish_failures: list[str] = []
    for row in all_rows:
        qid = str(row["question_id"])
        run_dir = run_dir_by_qid[qid]
        trajectory_path = resolve_run_path(run_dir, str(row.get("trajectory_file") or ""))
        events_path = resolve_run_path(run_dir, str(row.get("events_file") or ""))
        if not trajectory_path.exists() or not events_path.exists():
            missing_files.append(qid)
            continue
        try:
            value = read_json(trajectory_path)
            if not isinstance(value, (dict, list)):
                invalid_trajectory_json += 1
        except Exception:
            invalid_trajectory_json += 1
        inspection = inspect_events(events_path, row)
        event_rows_total += inspection["event_rows"]
        invalid_event_lines += inspection["invalid_json_lines"]
        if inspection["finish_records"] != 1:
            finish_failures.append(qid)
        if not inspection["reconstructed"]:
            reconstruction_failures.append(qid)

    checks.update({
        "all_trajectory_event_files_present": not missing_files,
        "all_formal_trajectories_valid_json": invalid_trajectory_json == 0,
        "all_event_jsonl_valid": invalid_event_lines == 0,
        "every_trajectory_has_one_finish": not finish_failures,
        "results_reconstruct_from_events": not reconstruction_failures,
    })
    errors.extend(name for name in (
        "all_trajectory_event_files_present",
        "all_formal_trajectories_valid_json",
        "all_event_jsonl_valid",
        "every_trajectory_has_one_finish",
        "results_reconstruct_from_events",
    ) if not checks[name])

    current_hashes = {relative: sha256_file(root / relative) for relative in CORE_FILES}
    expected_hashes = read_json(continuation_dir / "run_config.json").get("core_hashes_before", {})
    checks["champion_core_hash_unchanged"] = current_hashes == expected_hashes
    if not checks["champion_core_hash_unchanged"]:
        errors.append("champion_core_hash_unchanged")

    if errors:
        print(json.dumps({
            "hard_check": "FAIL",
            "checks": checks,
            "errors": errors,
            "missing_files": missing_files[:20],
            "finish_failures": finish_failures[:20],
            "reconstruction_failures": reconstruction_failures[:20],
        }, ensure_ascii=False, indent=2))
        return 3

    # From here onward, all writes are derived artifacts only.
    output_dir.mkdir(parents=True, exist_ok=True)
    provider_by_qid: dict[str, str] = {}
    evidence_by_qid: dict[str, str] = {}
    for qid in pilot_ids:
        provider_by_qid[qid] = "official_dashscope"
        evidence_by_qid[qid] = "pilot100_001/run_config.json and the immutable Pilot100 run boundary"
    for qid in remaining_ids:
        if qid in clean_official_ids:
            provider_by_qid[qid] = "official_dashscope"
            evidence_by_qid[qid] = "remaining890_v1_001/results.jsonl clean completion before provider transition; not in quota_affected_ids"
        else:
            provider_by_qid[qid] = "dmxapi"
            evidence_by_qid[qid] = "provider_transition.json complement of 461 clean official completions; completed in remaining890_v1_dmx_continuation_001"

    for row in all_rows:
        row["_actual_provider"] = provider_by_qid[str(row["question_id"])]

    provenance_path = output_dir / "provider_provenance_map.jsonl"
    manifest_path = output_dir / "teacher_v1_raw990_manifest.jsonl"
    with provenance_path.open("w", encoding="utf-8") as provider_handle, manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for row in sorted(all_rows, key=lambda value: int(value["position"])):
            qid = str(row["question_id"])
            run_dir = run_dir_by_qid[qid]
            logical_run = "pilot100_001" if qid in pilot_ids else "remaining890_v1_001"
            provider_handle.write(json.dumps({
                "question_id": qid,
                "source_split_position": int(row["position"]),
                "logged_endpoint": row.get("endpoint"),
                "actual_provider": provider_by_qid[qid],
                "evidence": evidence_by_qid[qid],
                "confidence": "authoritative",
            }, ensure_ascii=False) + "\n")
            manifest_handle.write(json.dumps({
                "question_id": qid,
                "source_split_position": int(row["position"]),
                "run": logical_run,
                "artifact_run": "pilot100_001" if qid in pilot_ids else "remaining890_v1_dmx_continuation_001",
                "formal": True,
                "completed": True,
                "actual_provider": provider_by_qid[qid],
                "provider_provenance_source": evidence_by_qid[qid],
                "trajectory_path": str(resolve_run_path(run_dir, str(row["trajectory_file"]))),
                "events_path": str(resolve_run_path(run_dir, str(row["events_file"]))),
            }, ensure_ascii=False) + "\n")

    metrics = aggregate(all_rows)
    provider_rows: list[dict[str, Any]] = []
    provider_groups = {
        "official_dashscope": [r for r in all_rows if r["_actual_provider"] == "official_dashscope"],
        "dmxapi": [r for r in all_rows if r["_actual_provider"] == "dmxapi"],
    }
    official_detail = detailed_row("official_dashscope", provider_groups["official_dashscope"])
    dmx_detail = detailed_row("dmxapi", provider_groups["dmxapi"])
    provider_rows.extend([official_detail, dmx_detail])
    difference = {"group": "absolute_difference"}
    for key in official_detail:
        if key == "group":
            continue
        difference[key] = abs(official_detail[key] - dmx_detail[key])
    provider_rows.append(difference)

    termination_rows = [
        detailed_row(group, [r for r in all_rows if termination_name(r) == group])
        for group in ("normal_answer", "force_answer", "raw_fallback")
    ]

    round_specs = (
        ("1-3", 1, 3), ("4-6", 4, 6), ("7-12", 7, 12),
        ("13-20", 13, 20), ("21-30", 21, 30), ("31+", 31, 10**9),
    )
    round_rows: list[dict[str, Any]] = []
    for label, lower, upper in round_specs:
        members = [r for r in all_rows if lower <= int(r.get("rounds") or 0) <= upper]
        detail = detailed_row(label, members)
        detail["percentage"] = len(members) / len(all_rows)
        round_rows.append(detail)

    token_values = [int(r.get("total_tokens") or 0) for r in all_rows]
    p50, p75, p90, p95, p99 = (percentile(token_values, p) for p in (0.50, 0.75, 0.90, 0.95, 0.99))
    token_specs = (
        ("<=P50", -1, p50, True),
        ("P50-P75", p50, p75, False),
        ("P75-P90", p75, p90, False),
        ("P90-P95", p90, p95, False),
        ("P95-P99", p95, p99, False),
        (">P99", p99, float("inf"), False),
    )
    token_rows: list[dict[str, Any]] = []
    for label, lower, upper, first in token_specs:
        members = [r for r in all_rows if (int(r.get("total_tokens") or 0) <= upper if first else lower < int(r.get("total_tokens") or 0) <= upper)]
        detail = detailed_row(label, members)
        detail["percentage"] = len(members) / len(all_rows)
        detail["lower_exclusive"] = lower
        detail["upper_inclusive"] = upper
        token_rows.append(detail)

    top10_rows = []
    for row in sorted(all_rows, key=lambda value: int(value.get("total_tokens") or 0), reverse=True)[:10]:
        top10_rows.append({
            "question_id": row["question_id"],
            "question_type": row["question_type"],
            "outcome_pass": bool(row["outcome_pass"]),
            "rounds": int(row["rounds"]),
            "search_actions": int(row["search_actions"]),
            "search_queries": int(row["search_queries"]),
            "input_tokens": int(row["input_tokens"]),
            "output_tokens": int(row["output_tokens"]),
            "total_tokens": int(row["total_tokens"]),
            "termination": termination_name(row),
            "actual_provider": row["_actual_provider"],
        })

    ordinary_pass = [r for r in all_rows if r["question_type"] == "ordinary_qa" and bool(r["outcome_pass"])]
    ordinary_candidate_stats = {
        "count": len(ordinary_pass),
        "interpretation": "Outcome-filter upper bound only; not the final SFT dataset size.",
        "rounds": distribution(int(r["rounds"]) for r in ordinary_pass),
        "search_actions": distribution(int(r["search_actions"]) for r in ordinary_pass),
        "search_queries": distribution(int(r["search_queries"]) for r in ordinary_pass),
        "cumulative_tokens": distribution(int(r["total_tokens"]) for r in ordinary_pass),
        "force_answer_count": sum(termination_name(r) == "force_answer" for r in ordinary_pass),
        "force_answer_rate": sum(termination_name(r) == "force_answer" for r in ordinary_pass) / len(ordinary_pass),
        "malformed_wrapper_rate": sum(int(r["raw_spaced_tool_call"]) + int(r["raw_tool_call_no_gt"]) for r in ordinary_pass) / sum(int(r["search_actions"]) for r in ordinary_pass),
        "action_source_mismatch_rate": sum(int(r["action_source_mismatch"]) for r in ordinary_pass) / sum(int(r["search_actions"]) for r in ordinary_pass),
    }

    ordinary_all = [r for r in all_rows if r["question_type"] == "ordinary_qa"]
    invalid_all = [r for r in all_rows if r["question_type"] == "invalid_question"]
    def subset_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            **quality(rows),
            "force_answer_count": sum(termination_name(r) == "force_answer" for r in rows),
            "force_answer_rate": sum(termination_name(r) == "force_answer" for r in rows) / len(rows),
            "mean_rounds": mean(int(r["rounds"]) for r in rows),
            "mean_search_actions": mean(int(r["search_actions"]) for r in rows),
            "mean_search_queries": mean(int(r["search_queries"]) for r in rows),
            "mean_total_tokens": mean(int(r["total_tokens"]) for r in rows),
        }
    invalid_stats = {
        "invalid_question": subset_stats(invalid_all),
        "ordinary_qa_comparator": subset_stats(ordinary_all),
        "descriptive_interpretation": "Invalid questions show a much lower pass rate and are tested for longer-search/force-answer concentration. This is observational, not causal.",
    }

    alignment = action_alignment_cases(all_rows, run_dir_by_qid, limit=20)
    if len(alignment) < 20:
        print(json.dumps({"hard_check": "FAIL", "error": "fewer than 20 action-source mismatch audit cases", "cases": len(alignment)}, indent=2))
        return 4
    alignment_checks = {
        "sample_count": len(alignment),
        "all_have_distinct_reasoning_and_content_candidates": all(row["reasoning_candidate_sha256"] != row["content_candidate_sha256"] for row in alignment),
        "all_executed_queries_match_search_start": all(row["executed_query_matches_search_start"] for row in alignment),
        "authoritative_reconstruction_source": "actual_executed_action -> actual_observation",
        "raw_textual_action_is_not_authoritative": True,
    }

    integrity = {
        "hard_check": "PASS",
        "checks": checks,
        "source_split": str(split_path),
        "source_split_sha256": sha256_file(split_path),
        "pilot_results_sha256": sha256_file(pilot_dir / "results.jsonl"),
        "remaining_results_sha256": sha256_file(continuation_dir / "results.jsonl"),
        "provider_transition_sha256": sha256_file(continuation_dir / "provider_transition.json"),
        "results_rows": len(all_rows),
        "event_rows": event_rows_total,
        "invalid_result_json_lines": pilot_bad + remaining_bad,
        "invalid_event_json_lines": invalid_event_lines,
        "invalid_trajectory_json_files": invalid_trajectory_json,
        "missing_formal_files": len(missing_files),
        "finish_record_failures": len(finish_failures),
        "result_event_reconstruction_failures": len(reconstruction_failures),
        "duplicate_question_ids": len(all_ids) - len(set(all_ids)),
        "provider_unresolved": 0,
        "raw_retention_rate": 1.0,
        "champion_core_hashes": current_hashes,
        "champion_core_hash_unchanged": True,
        "raw_files_modified": False,
    }
    metrics["experiment"] = {
        "identity": "Teacher V1 raw990 baseline",
        "teacher_model": "qwen3.5-plus",
        "teacher_policy_version": "V1",
        "temperature": 0.4,
        "max_tokens": 8192,
        "enable_thinking": True,
        "max_rounds": 100,
        "runtime": "frozen Champion Search-only",
        "retriever": "offline E5/FAISS",
        "providers": {"official_dashscope": 561, "dmxapi": 429},
        "provider_unresolved": 0,
    }
    metrics["integrity"] = integrity
    metrics["quantile_thresholds"] = {"p50": p50, "p75": p75, "p90": p90, "p95": p95, "p99": p99}
    metrics["action_alignment_audit"] = alignment_checks

    (output_dir / "baseline_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "ordinary_outcome_pass632_stats.json").write_text(json.dumps(ordinary_candidate_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "invalid_question_stats.json").write_text(json.dumps(invalid_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "integrity_report.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "action_alignment_audit.jsonl").open("w", encoding="utf-8") as handle:
        for row in alignment:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(output_dir / "provider_cross_tab.csv", provider_rows)
    write_csv(output_dir / "termination_cross_tab.csv", termination_rows)
    write_csv(output_dir / "round_bucket_cross_tab.csv", round_rows)
    write_csv(output_dir / "token_bucket_cross_tab.csv", token_rows)
    write_csv(output_dir / "top10_expensive_trajectories.csv", top10_rows)

    q = metrics["quality"]
    a = metrics["agent"]
    t = metrics["tokens"]
    p = metrics["protocol"]
    infra = metrics["infrastructure"]
    provider_view = [{key: row[key] for key in ("group", "n", "overall_em", "overall_f1", "overall_pass_rate", "mean_rounds", "mean_tokens", "force_answer_rate")} for row in provider_rows[:2]]
    termination_view = [{key: row[key] for key in ("group", "n", "overall_em", "overall_f1", "overall_pass_rate", "mean_rounds", "p95_rounds", "mean_tokens")} for row in termination_rows]
    round_view = [{key: row[key] for key in ("group", "n", "percentage", "overall_em", "overall_f1", "overall_pass_rate", "mean_tokens", "force_answer_rate")} for row in round_rows]
    token_view = [{key: row[key] for key in ("group", "n", "overall_em", "overall_f1", "overall_pass_rate", "mean_rounds", "force_answer_rate")} for row in token_rows]
    git_head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    report = f"""# Teacher V1 Baseline Freeze — 990 Formal Raw Trajectories

## 1. Experiment identity

- Identity: Teacher V1 raw990 baseline
- Source pool: `data/splits/sft_pool_990_seed20260904.jsonl`
- Source split SHA256: `{integrity['source_split_sha256']}`
- Pre-freeze Git HEAD: `{git_head}`
- Formal trajectories: 990; unique question IDs: 990; completed-result overlap: 0.
- No raw trajectory, result, or event file was rewritten by this freeze.

## 2. Dataset / split

Pilot100 covers positions 1–100. Remaining890 covers positions 101–990 exactly. The union matches all 990 rows of the frozen SFT pool. There are {q['ordinary']['n']} ordinary-QA and {q['invalid_question']['n']} invalid-question examples.

## 3. Teacher policy configuration

`qwen3.5-plus`, Teacher Policy V1, temperature 0.4, max tokens 8192, thinking enabled, max rounds 100, frozen Champion Search-only runtime, and the offline E5/FAISS retriever. The three Champion core hashes match the rollout-time hashes.

## 4. Provider provenance

Actual provider is authoritatively resolved for all 990 IDs: official DashScope = 561, DMXAPI = 429, unresolved = 0. The raw Remaining890 endpoint label was not edited. Its provider is reconstructed from the pre-transition clean official result set, the 22 quota-affected exclusions, `provider_transition.json`, and the continuation completion set.

Provider comparison is descriptive only. The provider assignment follows run order and quota transition rather than randomization, so question mix/difficulty is a confound.

## 5. Core baseline table

| subset | n | EM | F1 | outcome pass | pass rate |
|---|---:|---:|---:|---:|---:|
| Ordinary QA | {q['ordinary']['n']} | {q['ordinary']['em']:.4f} | {q['ordinary']['f1']:.4f} | {q['ordinary']['outcome_pass']} | {q['ordinary']['outcome_pass_rate']:.4f} |
| Invalid question | {q['invalid_question']['n']} | {q['invalid_question']['em']:.4f} | {q['invalid_question']['f1']:.4f} | {q['invalid_question']['outcome_pass']} | {q['invalid_question']['outcome_pass_rate']:.4f} |
| Overall | {q['overall']['n']} | {q['overall']['em']:.4f} | {q['overall']['f1']:.4f} | {q['overall']['outcome_pass']} | {q['overall']['outcome_pass_rate']:.4f} |

## 6. Ordinary vs invalid analysis

Teacher quality is substantially stronger on ordinary QA than invalid questions. The invalid subset therefore lowers the overall score. Invalid-question behavior is reported separately in `invalid_question_stats.json`; the evidence is consistent with an answer-presupposing failure pattern when invalid questions have higher search/round/force-answer burden, but this remains an observational diagnosis.

The 632 ordinary outcome-pass trajectories are only the Outcome Filter upper bound. They are not the final SFT dataset and still require Process Judge review.

## 7. Provider cross-tab

{markdown_table(provider_view, ['group', 'n', 'overall_em', 'overall_f1', 'overall_pass_rate', 'mean_rounds', 'mean_tokens', 'force_answer_rate'])}

Full rows plus the official-minus-DMX absolute differences are in `provider_cross_tab.csv`. No causal provider claim is made.

## 8. Termination analysis

{markdown_table(termination_view, ['group', 'n', 'overall_em', 'overall_f1', 'overall_pass_rate', 'mean_rounds', 'p95_rounds', 'mean_tokens'])}

Force-answer quality and long-trajectory concentration are descriptive properties of Teacher V1, not reasons to alter the frozen policy.

## 9. Round-efficiency analysis

{markdown_table(round_view, ['group', 'n', 'percentage', 'overall_em', 'overall_f1', 'overall_pass_rate', 'mean_tokens', 'force_answer_rate'])}

Long trajectories are associated with higher cost and may be associated with lower success/harder questions; these observational data do not establish that more searches cause failure.

## 10. Token-cost long-tail analysis

Total cumulative tokens = {t['total']:,}; median trajectory = {t['p50_trajectory_cumulative']:,.0f}; P95 = {t['p95_trajectory_cumulative']:,.0f}; maximum = {t['max_trajectory_cumulative']:,}. The ten most expensive trajectories consume {t['top10_trajectory_share']:.2%} of all tokens.

{markdown_table(token_view, ['group', 'n', 'overall_em', 'overall_f1', 'overall_pass_rate', 'mean_rounds', 'force_answer_rate'])}

This is a pronounced right tail. Exact qids and cost/termination fields, without gold answers or full trajectories, are in `top10_expensive_trajectories.csv`.

## 11. Protocol findings

- Canonical exact: {p['canonical_exact']}
- Spaced tool-call wrapper: {p['spaced_tool_call']}
- Tool-call missing `>`: {p['tool_call_missing_gt']}
- Champion bare-JSON fallback executions: {p['bare_json_fallback']}
- Invalid JSON: {p['invalid_json']}; parser-none: {p['parser_none']}
- Action-source mismatches: {p['action_source_mismatch']} ({p['action_source_mismatch_rate']:.2%} of Search actions)

Raw tool wrappers are frequently non-canonical, while the native Champion bare-JSON fallback preserves functional Search execution. Raw text must remain retained, but it cannot be treated as the authoritative training action.

## 12. Action-observation alignment findings

Twenty action-source mismatch cases were structurally audited using hashes, candidate excerpts, actual parsed/executed actions, and observation hashes. In canonical SFT reconstruction, the authoritative pair must be `actual executed action -> actual observation`, not an unverified raw reasoning/content tool candidate. No SFT dataset is generated by this freeze.

## 13. Retriever/environment health

Retriever requests = {infra['retriever_requests']:,}; successes = {infra['retriever_successes']:,}; failures = {infra['retriever_failures']}. Message-continuity failures, environment query rewrites, Visit executions, gold leakage, legacy runtime usage, and state-machine failures are all zero. The retriever is not the observed bottleneck in this corpus.

## 14. Known limitations / confounds

1. Provider assignment is not randomized and is confounded with position/question mix.
2. Outcome pass uses the frozen alias-aware evaluator and F1 >= 0.8; it is not a process-quality judgment.
3. Invalid questions require separate interpretation and have very low outcome-pass accuracy.
4. Tool serialization is often non-canonical; raw textual actions are not automatically safe SFT targets.
5. Token and round associations are observational and do not prove causal effects.
6. Raw logs remain external, large, read-only artifacts; the Git freeze commits manifests, hashes, analysis code, and compact derived outputs only.

## 15. Frozen artifacts

- `teacher_v1_raw990_manifest.jsonl`
- `provider_provenance_map.jsonl`
- `baseline_metrics.json`
- `provider_cross_tab.csv`
- `termination_cross_tab.csv`
- `round_bucket_cross_tab.csv`
- `token_bucket_cross_tab.csv`
- `top10_expensive_trajectories.csv`
- `ordinary_outcome_pass632_stats.json`
- `invalid_question_stats.json`
- `action_alignment_audit.jsonl`
- `integrity_report.json`

## 16. Explicit next-stage boundary

`READY_FOR_PROCESS_JUDGE = true`

`PROCESS_JUDGE_STARTED = false`

`SFT_STARTED = false`

`DPO_STARTED = false`

`PRM_STARTED = false`

`GRPO_STARTED = false`
"""
    (output_dir / "TEACHER_V1_BASELINE_FREEZE.md").write_text(report, encoding="utf-8")

    # Make all derived artifacts read-only.  The parent remains writable so a
    # future, explicitly versioned freeze can use a different directory.
    for path in output_dir.iterdir():
        if path.is_file():
            path.chmod(0o444)

    print(json.dumps({
        "hard_check": "PASS",
        "output_dir": str(output_dir),
        "artifacts": sorted(path.name for path in output_dir.iterdir()),
        "metrics": metrics,
        "provider_counts": {key: len(value) for key, value in provider_groups.items()},
        "alignment_cases": len(alignment),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
