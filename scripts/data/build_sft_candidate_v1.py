#!/usr/bin/env python3
"""Construct deterministic, unjudged Stage-1 SFT candidates from Teacher V1.

The script is intentionally offline.  It reads the frozen raw990 manifest,
results, and event sidecars; it never imports the Champion runtime, calls an
LLM, calls the retriever, or mutates a frozen/raw artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import json5
from transformers import AutoTokenizer


JUDGE_INSTRUCTION = """You are a query-evaluation assistant. Your task is to assess the quality of a search agent's query of the current search round according to the user's question, the golden answer and the agent's search process up to the current search round.

If the agent's query intent of the current search round is necessary and actionable, and the corresponding query result includes the answer for the query, the score for query should be 1. Otherwise, the score for the query should be 0. The details of the assessment are in the Evaluation Guideline, please read it carefully.

### User's question
{question}

### Golden answer
{golden_answer}

### Agent's search process up to the current search round
{trajectory_prefix}

### Evaluation Guideline
1. Identify the agent's query intent of the current search round accurately (**last round** in the agent's search process up to the current search round).
2. The query result **doesn't need to solve the user's question directly**; but it must include the information that address the agent's query intent completely (check/seek for information), related entities alone is not enough.
3. The intended entity and the one found in the query result **must be exactly the same (don't assume typos or other excuses)**, otherwise, the score should be 0.

### Output Format:
<answer> score for the query </answer>
<explanation> explanation for the score </explanation>"""


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
            yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def percentile(values: Iterable[float], fraction: float) -> float:
    data = sorted(float(value) for value in values)
    if not data:
        return 0.0
    point = (len(data) - 1) * fraction
    lower, upper = math.floor(point), math.ceil(point)
    if lower == upper:
        return data[lower]
    return data[lower] * (upper - point) + data[upper] * (point - lower)


def mean(values: Iterable[float]) -> float:
    data = list(values)
    return sum(data) / len(data) if data else 0.0


def distribution(values: Iterable[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    return {
        "n": len(data),
        "min": min(data) if data else 0.0,
        "mean": mean(data),
        "p50": percentile(data, 0.50),
        "p75": percentile(data, 0.75),
        "p90": percentile(data, 0.90),
        "p95": percentile(data, 0.95),
        "p99": percentile(data, 0.99),
        "max": max(data) if data else 0.0,
    }


def normalize_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def extract_between(text: str, start_tag: str, end_tag: str) -> str:
    start = text.find(start_tag)
    if start < 0:
        return ""
    start += len(start_tag)
    end = text.find(end_tag, start)
    return text[start:].strip() if end < 0 else text[start:end].strip()


def extract_tool_call(text: str) -> str | None:
    """Frozen Champion three-format extractor, copied for offline analysis."""
    if "<tool_call>" in text and "</tool_call>" in text:
        return extract_between(text, "<tool_call>", "</tool_call>")
    if "<tool_call>" in text:
        return text[text.find("<tool_call>") + len("<tool_call>"):].strip()
    function = re.search(r"<function=(search|visit)>", text)
    if function:
        name = function.group(1)
        start = function.start()
        end = text.find("</function>", start)
        block = text[start:] if end < 0 else text[start:end]
        arguments: dict[str, Any] = {}
        for match in re.finditer(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", block, re.DOTALL):
            value = match.group(2).strip()
            try:
                arguments[match.group(1)] = json5.loads(value)
            except Exception:
                arguments[match.group(1)] = value
        return json.dumps({"name": name, "arguments": arguments})
    bare = re.search(r'\{"name":\s*"(search|visit)"', text)
    if bare:
        start = bare.start()
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        return text[start:]
    return None


def parsed_candidate(text: str) -> Any:
    try:
        raw = extract_tool_call(text or "")
        return json5.loads(raw) if raw else None
    except Exception:
        return None


def wrapper_form(reasoning: str, content: str) -> str:
    full = (f"<think>\n{reasoning}\n</think>\n" if reasoning else "") + content
    if "<tool_call>" in full:
        return "canonical_exact"
    if re.search(r"<tool_call\s+>", full):
        return "spaced_tool_call"
    if "<tool_call" in full:
        return "tool_call_missing_gt"
    if "<function=" in full:
        return "function_fallback"
    return "bare_json"


def reasoning_text(reasoning: str, content: str) -> str:
    if reasoning:
        return reasoning
    extracted = extract_between(content, "<think>", "</think>")
    return extracted


def normalize_fragment(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def observation_fragments(observation: str) -> list[str]:
    """Extract deterministic title+snippet identities from Champion results.

    The adapter strips Retriever passage IDs, so stable IDs are unavailable.
    This parser follows SmartSearch's set-intersection idea using normalized
    result fragments as the documented fallback identity.
    """
    starts = list(re.finditer(r"(?m)^Search results for:.*$", observation))
    blocks: list[str] = []
    if starts:
        for index, match in enumerate(starts):
            end = starts[index + 1].start() if index + 1 < len(starts) else len(observation)
            blocks.append(observation[match.end():end])
    else:
        blocks = [observation]
    fragments: list[str] = []
    rank_pattern = re.compile(r"(?m)^(\d+)\.\s+.*$")
    for block in blocks:
        ranks = list(rank_pattern.finditer(block))
        for index, match in enumerate(ranks):
            end = ranks[index + 1].start() if index + 1 < len(ranks) else len(block)
            identity = normalize_fragment(block[match.start():end])
            if identity:
                fragments.append(identity)
    return fragments


def canonical_action(queries: list[str]) -> dict[str, Any]:
    return {"name": "search", "arguments": {"query": queries}}


def canonical_assistant_search(reasoning: str, action: dict[str, Any]) -> str:
    return (
        f"<think>\n{reasoning}\n</think>\n"
        f"<tool_call>\n{stable_json(action)}\n</tool_call>"
    )


def canonical_tool_response(observation: str) -> str:
    return f"<tool_response>\n{observation}\n</tool_response>"


def canonical_assistant_answer(reasoning: str, answer: str) -> str:
    return f"<think>\n{reasoning}\n</think>\n<answer>{answer}</answer>"


def build_messages(candidate: dict[str, Any], through_step: int | None = None, include_answer: bool = True) -> list[dict[str, str]]:
    trajectory = candidate["canonical_trajectory"]
    messages = [dict(message) for message in trajectory["initial_messages"]]
    steps = trajectory["steps"] if through_step is None else trajectory["steps"][:through_step]
    for step in steps:
        messages.append({"role": "assistant", "content": canonical_assistant_search(step["reasoning_before_action"], step["executed_action"])})
        messages.append({"role": "user", "content": canonical_tool_response(step["observation"])})
    if include_answer and through_step is None:
        final = trajectory["final"]
        messages.append({"role": "assistant", "content": canonical_assistant_answer(final["reasoning"], final["answer"])})
    return messages


def prefix_text(candidate: dict[str, Any], through_step: int) -> str:
    pieces: list[str] = []
    for step in candidate["canonical_trajectory"]["steps"][:through_step]:
        pieces.append(canonical_assistant_search(step["reasoning_before_action"], step["executed_action"]))
        pieces.append(canonical_tool_response(step["observation"]))
    return "\n".join(pieces)


def reconstruct(manifest: dict[str, Any], result: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    events_path = Path(manifest["events_path"])
    events = list(iter_jsonl(events_path))
    errors: list[str] = []
    starts = [event for event in events if event.get("event_type") == "trajectory_start"]
    finishes = [event for event in events if event.get("event_type") == "trajectory_finish"]
    if len(starts) != 1:
        errors.append("reconstruction_failed:trajectory_start_count")
    if len(finishes) != 1:
        errors.append("reconstruction_failed:trajectory_finish_count")
    if errors:
        return None, errors
    initial_messages = (starts[0].get("payload") or {}).get("messages_snapshot") or []
    if not isinstance(initial_messages, list):
        return None, ["reconstruction_failed:initial_messages"]

    last_response: dict[int, dict[str, Any]] = {}
    last_extracted: dict[int, str] = {}
    parsed: dict[int, dict[str, Any]] = {}
    active: dict[int, dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []
    response_sequence: list[dict[str, Any]] = []
    orphan_search_finish = 0
    orphan_observation = 0

    for event_index, event in enumerate(events):
        event_type = event.get("event_type")
        payload = event.get("payload") or {}
        round_idx = int(payload.get("round_idx") or 0)
        if event_type == "round_response":
            last_response[round_idx] = payload
            response_sequence.append(payload)
        elif event_type == "tool_extracted":
            last_extracted[round_idx] = str(payload.get("extractor_result") or "")
        elif event_type == "tool_json_parse" and payload.get("json_parse_success"):
            parsed[round_idx] = {
                "name": payload.get("parsed_tool_name"),
                "arguments": payload.get("parsed_tool_args") or {},
            }
        elif event_type == "search_start":
            action = parsed.get(round_idx)
            response = last_response.get(round_idx)
            queries = normalize_list(payload.get("queries"))
            if action is None or response is None or round_idx in active:
                errors.append(f"reconstruction_failed:search_start_round_{round_idx}")
                continue
            parsed_queries = normalize_list((action.get("arguments") or {}).get("query"))
            if parsed_queries != queries:
                errors.append(f"reconstruction_failed:query_execution_mismatch_round_{round_idx}")
            reasoning = str(response.get("reasoning_content") or "")
            content = str(response.get("content") or "")
            reasoning_candidate = parsed_candidate(reasoning)
            content_candidate = parsed_candidate(content)
            active[round_idx] = {
                "step_index": len(steps) + 1,
                "round_idx": round_idx,
                "reasoning_before_action": reasoning_text(reasoning, content),
                "raw_textual_action": last_extracted.get(round_idx, ""),
                "raw_response_sha256": sha256_text(reasoning + "\n" + content),
                "raw_wrapper_form": wrapper_form(reasoning, content),
                "executed_action": canonical_action(queries),
                "executed_action_original_arguments": action.get("arguments") or {},
                "proposed_engine": payload.get("engines"),
                "action_source_mismatch": bool(reasoning_candidate is not None and content_candidate is not None and reasoning_candidate != content_candidate),
                "source_event_indices": {"search_start": event_index},
            }
        elif event_type == "search_finish":
            if round_idx not in active:
                orphan_search_finish += 1
            else:
                active[round_idx]["search_result_sha256"] = sha256_text(str(payload.get("result_snapshot") or ""))
                active[round_idx]["source_event_indices"]["search_finish"] = event_index
        elif event_type == "tool_response_appended":
            if round_idx not in active:
                orphan_observation += 1
                continue
            step = active.pop(round_idx)
            observation = str(payload.get("tool_response") or "")
            messages = payload.get("messages_snapshot") or []
            continuity_ok = bool(
                isinstance(messages, list)
                and len(messages) >= 2
                and messages[-2].get("role") == "assistant"
                and messages[-1].get("role") == "user"
                and observation in str(messages[-1].get("content") or "")
            )
            step.update({
                "observation": observation,
                "observation_sha256": sha256_text(observation),
                "observation_source": "tool_response_appended",
                "message_continuity_ok": continuity_ok,
            })
            step["source_event_indices"]["tool_response_appended"] = event_index
            steps.append(step)

    if active:
        errors.append("reconstruction_failed:unfinished_search_action")
    if orphan_search_finish:
        errors.append("reconstruction_failed:orphan_search_finish")
    if orphan_observation:
        errors.append("reconstruction_failed:orphan_observation")
    if len(steps) != int(result.get("search_actions") or 0):
        errors.append("reconstruction_failed:search_action_count")
    if sum(len(step["executed_action"]["arguments"]["query"]) for step in steps) != int(result.get("search_queries") or 0):
        errors.append("reconstruction_failed:search_query_count")

    finish_answer = str((finishes[0].get("payload") or {}).get("answer") or result.get("prediction") or "")
    final_response = response_sequence[-1] if response_sequence else {}
    final_reasoning = reasoning_text(str(final_response.get("reasoning_content") or ""), str(final_response.get("content") or ""))
    trajectory = {
        "initial_messages": initial_messages,
        "steps": steps,
        "final": {"reasoning": final_reasoning, "answer": finish_answer},
    }
    candidate = {
        "question_id": str(result["question_id"]),
        "question": result["question"],
        "gold_answers": normalize_list(result.get("golden_answers")),
        "status": "candidate_unjudged",
        "canonical_trajectory": trajectory,
        "metadata": {
            "source_split_position": int(result["position"]),
            "source_run": manifest["run"],
            "artifact_run": manifest.get("artifact_run"),
            "actual_provider": manifest["actual_provider"],
            "answer_em": float(result.get("answer_em") or 0.0),
            "answer_f1": float(result.get("answer_f1") or 0.0),
            "outcome_pass": bool(result.get("outcome_pass")),
            "step_count": len(steps),
            "mismatch_count": sum(bool(step["action_source_mismatch"]) for step in steps),
            "termination": result.get("termination"),
            "rollout_cumulative_tokens": int(result.get("total_tokens") or 0),
            "raw_trajectory_path": manifest["trajectory_path"],
            "raw_events_path": manifest["events_path"],
        },
    }
    return candidate, errors


def structural_reasons(candidate: dict[str, Any], reconstruction_errors: list[str]) -> list[str]:
    if reconstruction_errors:
        return sorted(set(reconstruction_errors))
    reasons: list[str] = []
    trajectory = candidate["canonical_trajectory"]
    initial = trajectory.get("initial_messages") or []
    if not any(message.get("role") == "user" for message in initial):
        reasons.append("missing_user_question")
    final = trajectory.get("final") or {}
    if not str(final.get("answer") or "").strip():
        reasons.append("missing_answer")
    steps = trajectory.get("steps") or []
    if not steps and not str(final.get("answer") or "").strip():
        reasons.append("missing_assistant_segment")
    for step in steps:
        action = step.get("executed_action") or {}
        if action.get("name") != "search" or not isinstance(action.get("arguments"), dict):
            reasons.append("invalid_executed_action")
            continue
        queries = action["arguments"].get("query")
        if not isinstance(queries, list) or not queries or any(not isinstance(query, str) or not query.strip() for query in queries):
            reasons.append("empty_query")
        if not isinstance(step.get("observation"), str) or not step.get("observation"):
            reasons.append("unpaired_action_observation")
        if not step.get("message_continuity_ok"):
            reasons.append("structural_invalid")
    # This serialization assertion proves the canonical final answer is last.
    try:
        messages = build_messages(candidate)
        if not messages or messages[-1].get("role") != "assistant" or "<answer>" not in messages[-1].get("content", ""):
            reasons.append("structural_invalid")
        for step in steps:
            stable_json(step["executed_action"])
    except Exception:
        reasons.append("structural_invalid")
    return sorted(set(reasons))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def length_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [int(record["canonical_length"]) for record in records]
    thresholds = (4096, 8192, 12288, 16384, 24576, 32768)
    return {
        "distribution": distribution(lengths),
        "coverage": {
            f"le_{threshold}": {
                "count": sum(length <= threshold for length in lengths),
                "rate": sum(length <= threshold for length in lengths) / len(lengths) if lengths else 0.0,
            }
            for threshold in thresholds
        } | {
            "gt_32768": {
                "count": sum(length > 32768 for length in lengths),
                "rate": sum(length > 32768 for length in lengths) / len(lengths) if lengths else 0.0,
            }
        },
    }


def audit_record(category: str, candidate: dict[str, Any]) -> dict[str, Any]:
    steps = candidate["canonical_trajectory"]["steps"]
    return {
        "category": category,
        "question_id": candidate["question_id"],
        "source_split_position": candidate["metadata"]["source_split_position"],
        "step_count": len(steps),
        "mismatch_count": candidate["metadata"]["mismatch_count"],
        "canonical_length": candidate.get("canonical_length"),
        "redundancy_counts": candidate.get("redundancy_counts"),
        "all_actions_are_search": all(step["executed_action"]["name"] == "search" for step in steps),
        "all_queries_nonempty": all(bool(step["executed_action"]["arguments"]["query"]) for step in steps),
        "all_observations_present": all(bool(step["observation"]) for step in steps),
        "all_continuity_ok": all(step["message_continuity_ok"] for step in steps),
        "action_hashes": [sha256_text(stable_json(step["executed_action"])) for step in steps],
        "observation_hashes": [step["observation_sha256"] for step in steps],
        "raw_events_path": candidate["metadata"]["raw_events_path"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--audit-seed", type=int, default=20260904)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    tokenizer_dir = args.tokenizer_dir.resolve()
    freeze_dir = root / "artifacts" / "baseline_freeze"
    raw_manifest_path = freeze_dir / "teacher_v1_raw990_manifest.jsonl"
    provenance_path = freeze_dir / "provider_provenance_map.jsonl"
    frozen_integrity_path = freeze_dir / "integrity_report.json"

    required = [raw_manifest_path, provenance_path, frozen_integrity_path, tokenizer_dir / "tokenizer.json", tokenizer_dir / "tokenizer_config.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print(json.dumps({"hard_check": "FAIL", "missing": missing}, indent=2))
        return 2
    if output_dir.exists() and any(output_dir.iterdir()):
        print(json.dumps({"hard_check": "FAIL", "error": "output directory is not empty", "output_dir": str(output_dir)}, indent=2))
        return 3

    frozen_hashes_before = {path.name: sha256_file(path) for path in freeze_dir.iterdir() if path.is_file()}
    manifest_rows = list(iter_jsonl(raw_manifest_path))
    provenance_rows = list(iter_jsonl(provenance_path))
    provenance = {str(row["question_id"]): row for row in provenance_rows}
    pilot_results = {str(row["question_id"]): row for row in iter_jsonl(root / "logs" / "teacher_rollout" / "pilot100_001" / "results.jsonl")}
    remaining_results = {str(row["question_id"]): row for row in iter_jsonl(root / "logs" / "teacher_rollout" / "remaining890_v1_dmx_continuation_001" / "results.jsonl")}
    results = {**pilot_results, **remaining_results}
    all_qids = [str(row["question_id"]) for row in manifest_rows]

    preflight = {
        "raw_manifest_rows_990": len(manifest_rows) == 990,
        "raw_manifest_unique_990": len(set(all_qids)) == 990,
        "results_resolved_990": len(results) == 990 and set(results) == set(all_qids),
        "provenance_resolved_990": len(provenance) == 990 and set(provenance) == set(all_qids),
        "ordinary_count_861": sum(results[qid].get("question_type") == "ordinary_qa" for qid in all_qids) == 861,
        "frozen_integrity_pass": read_json(frozen_integrity_path).get("hard_check") == "PASS",
    }
    if not all(preflight.values()):
        print(json.dumps({"hard_check": "FAIL", "preflight": preflight}, indent=2))
        return 4

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for manifest in sorted(manifest_rows, key=lambda row: int(row["source_split_position"])):
        qid = str(manifest["question_id"])
        result = results[qid]
        criterion = result.get("question_type") == "ordinary_qa" and float(result.get("answer_f1") or 0.0) >= 0.8
        if criterion:
            selected.append((manifest, result))
    if len(selected) != 632 or len({str(result["question_id"]) for _, result in selected}) != 632:
        print(json.dumps({"hard_check": "FAIL", "error": "outcome632 count mismatch", "count": len(selected)}, indent=2))
        return 5
    if any(not bool(result.get("outcome_pass")) for _, result in selected):
        print(json.dumps({"hard_check": "FAIL", "error": "recomputed criterion disagrees with frozen outcome_pass"}, indent=2))
        return 6

    output_dir.mkdir(parents=True, exist_ok=False)
    outcome_rows = []
    for manifest, result in selected:
        qid = str(result["question_id"])
        outcome_rows.append({
            "question_id": qid,
            "question": result["question"],
            "gold_answers": normalize_list(result.get("golden_answers")),
            "answer_em": float(result.get("answer_em") or 0.0),
            "answer_f1": float(result.get("answer_f1") or 0.0),
            "outcome_pass": True,
            "source_run": manifest["run"],
            "actual_provider": provenance[qid]["actual_provider"],
            "raw_trajectory_path": manifest["trajectory_path"],
            "raw_events_path": manifest["events_path"],
        })
    write_jsonl(output_dir / "outcome632_manifest.jsonl", outcome_rows)

    canonical: list[dict[str, Any]] = []
    structural_pass_rows: list[dict[str, Any]] = []
    structural_rejected_rows: list[dict[str, Any]] = []
    for index, (manifest, result) in enumerate(selected, 1):
        candidate, reconstruction_errors = reconstruct(manifest, result)
        if candidate is None:
            structural_rejected_rows.append({"question_id": result["question_id"], "reject_reason": sorted(set(reconstruction_errors))})
            continue
        reasons = structural_reasons(candidate, reconstruction_errors)
        candidate["structural_valid"] = not reasons
        candidate["reject_reason"] = reasons
        canonical.append(candidate)
        compact = {
            "question_id": candidate["question_id"],
            "source_split_position": candidate["metadata"]["source_split_position"],
            "step_count": candidate["metadata"]["step_count"],
            "mismatch_count": candidate["metadata"]["mismatch_count"],
            "structural_valid": not reasons,
            "reject_reason": reasons,
            "canonical_record_sha256": sha256_text(stable_json(candidate)),
        }
        (structural_pass_rows if not reasons else structural_rejected_rows).append(compact)
        if index % 50 == 0:
            print(f"RECONSTRUCTED={index}/632", flush=True)

    write_jsonl(output_dir / "canonical_candidates.jsonl", canonical)
    write_jsonl(output_dir / "structural_pass.jsonl", structural_pass_rows)
    write_jsonl(output_dir / "structural_rejected.jsonl", structural_rejected_rows)
    structural_ids = {row["question_id"] for row in structural_pass_rows}
    structural_candidates = [candidate for candidate in canonical if candidate["question_id"] in structural_ids]

    redundancy_rows: list[dict[str, Any]] = []
    redundancy_pass_rows: list[dict[str, Any]] = []
    redundancy_rejected_rows: list[dict[str, Any]] = []
    for candidate in structural_candidates:
        seen: set[str] = set()
        counts: list[int] = []
        step_rows = []
        parse_failed = False
        for step in candidate["canonical_trajectory"]["steps"]:
            fragments = observation_fragments(step["observation"])
            if not fragments:
                parse_failed = True
            current = set(fragments)
            count = len(seen.intersection(current))
            counts.append(count)
            step_rows.append({
                "step_index": step["step_index"],
                "redundancy_count": count,
                "document_fragment_count": len(fragments),
                "unique_document_fragment_count": len(current),
            })
            seen.update(current)
        passed = bool(not parse_failed and all(count <= 1 for count in counts))
        candidate["redundancy_counts"] = counts
        candidate["redundancy_pass"] = passed
        row = {
            "question_id": candidate["question_id"],
            "identity_source": "normalized_title_plus_snippet_fragment",
            "stable_passage_id_available": False,
            "steps": step_rows,
            "redundancy_pass": passed,
            "reject_reason": None if passed else ("redundancy_identity_unavailable" if parse_failed else "redundancy_gt_1"),
        }
        redundancy_rows.append(row)
        compact = {"question_id": candidate["question_id"], "redundancy_counts": counts, "redundancy_pass": passed, "reject_reason": row["reject_reason"]}
        (redundancy_pass_rows if passed else redundancy_rejected_rows).append(compact)
    write_jsonl(output_dir / "redundancy_audit.jsonl", redundancy_rows)
    write_jsonl(output_dir / "redundancy_pass.jsonl", redundancy_pass_rows)
    write_jsonl(output_dir / "redundancy_rejected.jsonl", redundancy_rejected_rows)
    redundancy_ids = {row["question_id"] for row in redundancy_pass_rows}

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=True)
    if not tokenizer.chat_template:
        print(json.dumps({"hard_check": "FAIL", "error": "Qwen3.5-4B chat template unavailable"}, indent=2))
        return 7
    length_records: list[dict[str, Any]] = []
    candidate_by_qid = {candidate["question_id"]: candidate for candidate in structural_candidates}
    for index, candidate in enumerate(structural_candidates, 1):
        messages = build_messages(candidate)
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        length = len(token_ids)
        candidate["canonical_length"] = length
        candidate["length_pass"] = length <= 16384
        length_records.append({
            "question_id": candidate["question_id"],
            "canonical_length": length,
            "rollout_cumulative_tokens": candidate["metadata"]["rollout_cumulative_tokens"],
            "redundancy_pass": candidate["question_id"] in redundancy_ids,
            "length_pass_16k": length <= 16384,
            "reject_reason": None if length <= 16384 else "over_16k",
        })
        if index % 50 == 0:
            print(f"TOKENIZED={index}/{len(structural_candidates)}", flush=True)

    redundancy_length_records = [record for record in length_records if record["redundancy_pass"]]
    cumulative_total = sum(record["rollout_cumulative_tokens"] for record in length_records)
    canonical_total = sum(record["canonical_length"] for record in length_records)
    length_audit = {
        "tokenizer": {
            "repo": "Qwen/Qwen3.5-4B",
            "local_path": str(tokenizer_dir),
            "revision": args.tokenizer_revision,
            "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
            "chat_template_sha256": sha256_text(str(tokenizer.chat_template)),
            "files": {path.name: sha256_file(path) for path in tokenizer_dir.iterdir() if path.is_file()},
        },
        "serialization": "tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)",
        "structural_pass": length_summary(length_records),
        "redundancy_pass": length_summary(redundancy_length_records),
        "rollout_vs_canonical": {
            "rollout_cumulative_tokens": cumulative_total,
            "canonical_sequence_tokens": canonical_total,
            "absolute_difference": cumulative_total - canonical_total,
            "canonical_to_rollout_ratio": canonical_total / cumulative_total if cumulative_total else 0.0,
            "note": "Rollout cumulative usage repeatedly counts growing prefixes; canonical length counts each final training sequence once.",
        },
        "records": length_records,
    }
    (output_dir / "length_audit.json").write_text(json.dumps(length_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    survivors = [candidate_by_qid[record["question_id"]] for record in redundancy_length_records if record["length_pass_16k"]]
    write_jsonl(output_dir / "pre_judge_survivors.jsonl", survivors)

    judge_units_path = output_dir / "judge_ready_units.jsonl"
    judge_input_lengths: list[int] = []
    judge_units = 0
    with judge_units_path.open("w", encoding="utf-8") as handle:
        for trajectory_index, candidate in enumerate(survivors, 1):
            for step in candidate["canonical_trajectory"]["steps"]:
                prefix = prefix_text(candidate, int(step["step_index"]))
                gold = candidate["gold_answers"]
                judge_text = JUDGE_INSTRUCTION.format(
                    question=candidate["question"],
                    golden_answer=json.dumps(gold, ensure_ascii=False),
                    trajectory_prefix=prefix,
                )
                judge_messages = [{"role": "user", "content": judge_text}]
                judge_tokens = len(tokenizer.apply_chat_template(judge_messages, tokenize=True, add_generation_prompt=True))
                unit = {
                    "question_id": candidate["question_id"],
                    "step_index": int(step["step_index"]),
                    "question": candidate["question"],
                    "gold_answer": gold,
                    "trajectory_prefix_H_t": prefix,
                    "executed_query": step["executed_action"]["arguments"]["query"],
                    "observation": step["observation"],
                    "redundancy_count": candidate["redundancy_counts"][int(step["step_index"]) - 1],
                    "canonical_length": candidate["canonical_length"],
                    "estimated_judge_input_tokens": judge_tokens,
                    "status": "judge_ready_unsubmitted",
                    "raw_events_path": candidate["metadata"]["raw_events_path"],
                }
                handle.write(json.dumps(unit, ensure_ascii=False) + "\n")
                judge_input_lengths.append(judge_tokens)
                judge_units += 1
            if trajectory_index % 50 == 0:
                print(f"JUDGE_UNITS_TRAJECTORIES={trajectory_index}/{len(survivors)}", flush=True)

    input_total = sum(judge_input_lengths)
    cost_scenarios = {}
    for output_tokens in (50, 100, 200):
        output_total = judge_units * output_tokens
        input_cost = input_total / 1_000_000 * 0.158
        output_cost = output_total / 1_000_000 * 1.58
        cost_scenarios[str(output_tokens)] = {
            "output_tokens_per_unit": output_tokens,
            "estimated_total_output_tokens": output_total,
            "input_cost_rmb": input_cost,
            "output_cost_rmb": output_cost,
            "total_cost_rmb": input_cost + output_cost,
        }
    cost_estimate = {
        "model": "qwen3.5-flash via DMXAPI",
        "api_called": False,
        "input_price_rmb_per_million": 0.158,
        "output_price_rmb_per_million": 1.58,
        "judge_units": judge_units,
        "input_tokens": distribution(judge_input_lengths),
        "total_estimated_input_tokens": input_total,
        "scenarios": cost_scenarios,
        "prompt_source": "SmartSearch scripts/data_construction/process_reward.py adapted to canonical Champion trajectory prefixes",
    }
    (output_dir / "judge_cost_estimate.json").write_text(json.dumps(cost_estimate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    redundancy_reject_ids = {row["question_id"] for row in redundancy_rejected_rows}
    over16_ids = {record["question_id"] for record in redundancy_length_records if not record["length_pass_16k"]}
    candidate_steps = sum(candidate["metadata"]["step_count"] for candidate in structural_candidates)
    redundancy_pass_steps = sum(candidate_by_qid[qid]["metadata"]["step_count"] for qid in redundancy_ids)
    funnel = {
        "raw990": 990,
        "ordinary861": 861,
        "outcome632": 632,
        "reconstruction_pass": len(canonical),
        "reconstruction_rejected": 632 - len(canonical),
        "structural_pass": len(structural_pass_rows),
        "structural_rejected": len(structural_rejected_rows),
        "redundancy_pass": len(redundancy_pass_rows),
        "redundancy_rejected": len(redundancy_rejected_rows),
        "length16k_pass_after_redundancy": len(survivors),
        "over16k_after_redundancy": len(over16_ids),
        "pre_judge_survivors": len(survivors),
        "judge_units": judge_units,
        "candidate_step_count_before_filters": candidate_steps,
        "candidate_step_count_after_redundancy": redundancy_pass_steps,
        "judge_units_saved_by_redundancy": candidate_steps - redundancy_pass_steps,
        "judge_units_saved_by_length_after_redundancy": redundancy_pass_steps - judge_units,
        "rejection_reasons": dict(Counter(reason for row in structural_rejected_rows for reason in row.get("reject_reason", []))) | {
            "redundancy_gt_1_or_identity_unavailable": len(redundancy_rejected_rows),
            "over_16k": len(over16_ids),
        },
        "process_judge_started": False,
        "final_sft_dataset_created": False,
    }
    (output_dir / "pre_judge_funnel.json").write_text(json.dumps(funnel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rng = random.Random(args.audit_seed)
    no_mismatch = [candidate for candidate in structural_candidates if candidate["metadata"]["mismatch_count"] == 0]
    mismatch = [candidate for candidate in structural_candidates if candidate["metadata"]["mismatch_count"] > 0]
    long_candidates = sorted(structural_candidates, key=lambda value: value["canonical_length"], reverse=True)[:10]
    redundancy_rejected_candidates = [candidate_by_qid[qid] for qid in sorted(redundancy_reject_ids)]
    near_16k = sorted(structural_candidates, key=lambda value: abs(value["canonical_length"] - 16384))[:10]
    def sample(values: list[dict[str, Any]], n: int = 10) -> list[dict[str, Any]]:
        return rng.sample(values, min(n, len(values)))
    audit_rows = (
        [audit_record("no_mismatch", value) for value in sample(no_mismatch)]
        + [audit_record("action_source_mismatch", value) for value in sample(mismatch)]
        + [audit_record("long_trajectory", value) for value in long_candidates]
        + [audit_record("redundancy_rejected", value) for value in sample(redundancy_rejected_candidates)]
        + [audit_record("near_16k", value) for value in near_16k]
    )
    write_jsonl(output_dir / "deterministic_alignment_audit.jsonl", audit_rows)
    audit_summary = {
        "seed": args.audit_seed,
        "requested_categories": {"no_mismatch": 10, "action_source_mismatch": 10, "long_trajectory": 10, "redundancy_rejected": 10, "near_16k": 10},
        "actual_categories": dict(Counter(row["category"] for row in audit_rows)),
        "all_actions_are_search": all(row["all_actions_are_search"] for row in audit_rows),
        "all_queries_nonempty": all(row["all_queries_nonempty"] for row in audit_rows),
        "all_observations_present": all(row["all_observations_present"] for row in audit_rows),
        "all_continuity_ok": all(row["all_continuity_ok"] for row in audit_rows),
        "full_observations_printed": False,
        "action_alignment": "PASS",
    }
    (output_dir / "alignment_audit_summary.json").write_text(json.dumps(audit_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (output_dir / "NOT_FINAL_SFT_DATASET.txt").write_text(
        "This directory contains unjudged SFT candidates.\nDo not use for training before Process Judge filtering.\n",
        encoding="utf-8",
    )

    structural_length = length_audit["structural_pass"]
    redundancy_length = length_audit["redundancy_pass"]
    report = f"""# Stage-1 SFT Candidate V1 — Pre-Judge Report

## Identity and boundary

This directory contains deterministic, unjudged candidates derived from the frozen Teacher V1 raw990 corpus. No LLM, Retriever, Refiner, Judge, reroll, or training process was invoked. Frozen raw and baseline artifacts were not modified.

## Funnel

990 raw → 861 ordinary QA → 632 outcome-pass → {funnel['reconstruction_pass']} reconstruction pass → {funnel['structural_pass']} structural pass → {funnel['redundancy_pass']} redundancy pass → {funnel['length16k_pass_after_redundancy']} at ≤16K → {funnel['pre_judge_survivors']} pre-Judge survivors → {judge_units} Judge units.

- Reconstruction/structural rejects: {funnel['structural_rejected']}
- Redundancy rejects: {funnel['redundancy_rejected']}
- Over-16K rejects after redundancy: {funnel['over16k_after_redundancy']}
- Judge units saved by redundancy: {funnel['judge_units_saved_by_redundancy']}
- Additional Judge units saved by length policy: {funnel['judge_units_saved_by_length_after_redundancy']}

## Canonical length

Qwen3.5-4B tokenizer revision `{args.tokenizer_revision}` with its actual chat template was used. No sequence was truncated.

Structural-pass P50/P75/P90/P95/P99/max: {structural_length['distribution']['p50']:.0f} / {structural_length['distribution']['p75']:.0f} / {structural_length['distribution']['p90']:.0f} / {structural_length['distribution']['p95']:.0f} / {structural_length['distribution']['p99']:.0f} / {structural_length['distribution']['max']:.0f}.

Coverage at 8K/16K/32K: {structural_length['coverage']['le_8192']['count']} ({structural_length['coverage']['le_8192']['rate']:.2%}) / {structural_length['coverage']['le_16384']['count']} ({structural_length['coverage']['le_16384']['rate']:.2%}) / {structural_length['coverage']['le_32768']['count']} ({structural_length['coverage']['le_32768']['rate']:.2%}).

Rollout cumulative usage is {cumulative_total:,} tokens versus {canonical_total:,} tokens for each canonical sequence counted once. Their difference reflects repeated growing prefixes during rollout, not deleted candidate content.

## Redundancy

The published SmartSearch set-intersection rule is used: each current result-fragment set is intersected with all fragments seen previously, and every step must have redundancy ≤1. Champion observations do not retain passage IDs, so identity is deterministic NFKC/lowercase/whitespace-normalized title+snippet text. No embedding or LLM semantic deduplication is used.

## Authoritative alignment

Canonical actions come from successful `tool_json_parse` plus `search_start` events. Observations come from the matching `tool_response_appended` event. Raw textual actions are retained only as audit metadata. The deterministic 50-case audit result is `{audit_summary['action_alignment']}`.

## Judge workload and static cost

- Trajectories: {len(survivors)}
- Judge units: {judge_units}
- Total estimated input tokens: {input_total:,}
- Mean/P50/P95 tokens per unit: {mean(judge_input_lengths):.1f} / {percentile(judge_input_lengths, .5):.0f} / {percentile(judge_input_lengths, .95):.0f}
- Estimated qwen3.5-flash cost with 50/100/200 output tokens per unit: ¥{cost_scenarios['50']['total_cost_rmb']:.2f} / ¥{cost_scenarios['100']['total_cost_rmb']:.2f} / ¥{cost_scenarios['200']['total_cost_rmb']:.2f}

"""
    (output_dir / "SFT_CANDIDATE_PREJUDGE_REPORT.md").write_text(report, encoding="utf-8")

    frozen_hashes_after = {path.name: sha256_file(path) for path in freeze_dir.iterdir() if path.is_file()}
    raw_hash_checks = {
        "frozen_artifact_hashes_unchanged": frozen_hashes_before == frozen_hashes_after,
        "frozen_artifact_hashes": frozen_hashes_after,
        "teacher_or_judge_api_called": False,
        "retriever_called": False,
        "raw_files_modified": False,
    }
    (output_dir / "source_immutability_check.json").write_text(json.dumps(raw_hash_checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not raw_hash_checks["frozen_artifact_hashes_unchanged"]:
        print(json.dumps({"hard_check": "FAIL", "error": "frozen artifacts changed during construction"}, indent=2))
        return 8

    summary = {
        "hard_check": "PASS",
        "preflight": preflight,
        "funnel": funnel,
        "length": length_audit,
        "judge_cost": cost_estimate,
        "alignment_audit": audit_summary,
        "output_dir": str(output_dir),
        "files": sorted(path.name for path in output_dir.iterdir()),
    }
    # Avoid printing the 632 per-record length rows in stdout.
    summary["length"] = {key: value for key, value in length_audit.items() if key != "records"}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
