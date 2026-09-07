#!/usr/bin/env python3
"""Build a deterministic, leakage-audited query-level Process Judge pilot."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path


HEADER_RE = re.compile(r"(?m)^Search results for: (.*)$")
WORD_RE = re.compile(r"[\w]+", re.UNICODE)
VERIFY_RE = re.compile(r"\b(verif(?:y|ication)|confirm|double[- ]check|cross[- ]check|check whether)\b", re.I)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def words(text: str) -> set[str]:
    return {w.casefold() for w in WORD_RE.findall(text) if len(w) > 1}


def jaccard(a: str, b: str) -> float:
    aa, bb = words(a), words(b)
    return len(aa & bb) / len(aa | bb) if aa and bb else 0.0


def parse_sections(observation: str, queries: list[str]) -> list[dict]:
    matches = list(HEADER_RE.finditer(observation))
    if len(matches) != len(queries):
        raise ValueError(f"query/result count mismatch: {len(queries)} != {len(matches)}")
    sections = []
    for idx, (query, match) in enumerate(zip(queries, matches)):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(observation)
        section = observation[match.start():end].rstrip()
        header = match.group(1).strip()
        if header != query.strip():
            raise ValueError(f"header mismatch at query {idx}: {header!r} != {query!r}")
        sections.append({
            "query_idx": idx,
            "query": query,
            "section_header": header,
            "result_section": section,
            "result_sha256": sha256_text(section),
        })
    return sections


def sanitized_history(steps: list[dict]) -> list[dict]:
    result = []
    for step in steps:
        result.append({
            "action_idx": int(step["step_index"]),
            "reasoning": step.get("reasoning_before_action") or "",
            "executed_action": {
                "name": "search",
                "arguments": {"query": list(step["executed_action"]["arguments"]["query"])},
            },
            "observation": step["observation"],
            "observation_sha256": step["observation_sha256"],
        })
    return result


def format_history(history: list[dict]) -> str:
    if not history:
        return "(No previous Search action.)"
    chunks = []
    for step in history:
        action = json.dumps(step["executed_action"], ensure_ascii=False, separators=(",", ":"))
        chunks.append(
            f"Previous action {step['action_idx']} reasoning:\n{step['reasoning']}\n"
            f"Previous executed action:\n{action}\n"
            f"Previous authoritative observation:\n{step['observation']}"
        )
    return "\n\n".join(chunks)


def judge_prompt(unit: dict) -> str:
    gold = json.dumps(unit["golden_answer"], ensure_ascii=False)
    return f"""You are a search-query usefulness evaluator. Evaluate exactly ONE executed search query using only the information shown below.

Judge two binary components:

1. INTENT: Is the current query a necessary or reasonable next information need for answering the original question, given the history before this action? It must be actionable and target the correct entity or relation.
2. RETRIEVAL: Does the current query-specific result actually provide enough relevant information to fulfill that query's intent, with the intended entity correctly matched?

The result does not need to answer the final user question directly; it only needs to fulfill the current query intent. Do not infer from sibling queries, sibling results, future trajectory steps, or the agent's final answer. Do not judge redundancy as a general policy. However, if the history already contains direct and sufficiently trustworthy evidence that fully resolves the same fact and there is no concrete need to verify a conflict, entity, or new constraint, mark INTENT=0 as already_resolved. Verification may pass only when prior evidence is incomplete, conflicting, entity-ambiguous, insufficient, or a genuinely new constraint is being checked.

Allowed failure_type values:
none, irrelevant_intent, already_resolved, wrong_entity, underspecified_query, retrieval_miss, retrieval_partial, result_entity_mismatch, other

The final answer must equal INTENT AND RETRIEVAL. Return exactly these tags with a brief explanation and no chain-of-thought:
<answer>0 or 1</answer>
<intent>0 or 1</intent>
<retrieval>0 or 1</retrieval>
<failure_type>one allowed value</failure_type>
<explanation>brief explanation about this query only</explanation>

ORIGINAL QUESTION:
{unit['question']}

GOLDEN ANSWER (Judge metadata only):
{gold}

AUTHORITATIVE HISTORY BEFORE CURRENT SEARCH ACTION:
{format_history(unit['authoritative_history_before_action'])}

CURRENT ACTION REASONING:
{unit['current_action_reasoning'] or '(none recorded)'}

CURRENT EXECUTED QUERY:
{unit['current_executed_query']}

CURRENT QUERY-SPECIFIC RESULT:
{unit['current_query_result']}
"""


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def bucket_action(action_idx: int, total: int) -> str:
    ratio = action_idx / total
    if action_idx == 1 or ratio <= 1 / 3:
        return "early"
    if ratio >= 2 / 3:
        return "late"
    return "middle"


def bucket_query(query_idx: int, count: int) -> str:
    if count == 1:
        return "single"
    if query_idx == 0:
        return "first"
    if query_idx == count - 1:
        return "last"
    return "middle"


def bucket_count(count: int) -> str:
    return "single" if count == 1 else (str(count) if count in {2, 3} else ">3")


def bucket_result(query: str, section: str) -> tuple[str, float]:
    body = section.split("\n", 1)[1] if "\n" in section else ""
    qwords = words(query)
    overlap = len(qwords & words(body)) / len(qwords) if qwords else 0.0
    if overlap >= 0.75:
        return "high_query_term_coverage", overlap
    if overlap >= 0.40:
        return "partial_query_term_coverage", overlap
    return "weak_query_term_coverage", overlap


def build_candidates(canonical: Path, disagreement: Path) -> tuple[list[dict], dict]:
    disagreement_actions = set()
    if disagreement.exists():
        for line in disagreement.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                disagreement_actions.add((str(row["question_id"]), int(row["action_index"])))

    candidates = []
    totals = collections.Counter()
    for line in canonical.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        qid = str(row["question_id"])
        trajectory = row["canonical_trajectory"]
        steps = trajectory["steps"]
        trajectory_id = Path(row["metadata"]["raw_trajectory_path"]).stem
        previous_queries: list[str] = []
        for action_idx, step in enumerate(steps, start=1):
            queries = list(step["executed_action"]["arguments"]["query"])
            sections = parse_sections(step["observation"], queries)
            reasoning = step.get("reasoning_before_action") or ""
            for query_idx, (query, section) in enumerate(zip(queries, sections)):
                history_sim = max((jaccard(query, old) for old in previous_queries), default=0.0)
                result_bucket, result_overlap = bucket_result(query, section["result_section"])
                sibling_queries = [q for i, q in enumerate(queries) if i != query_idx]
                sibling_in_reasoning = [q for q in sibling_queries if q and q.casefold() in reasoning.casefold()]
                unit_id = f"{trajectory_id}::action_{action_idx}::query_{query_idx}"
                candidate = {
                    "unit_id": unit_id,
                    "trajectory_id": trajectory_id,
                    "question_id": qid,
                    "action_idx": action_idx,
                    "query_idx": query_idx,
                    "question": row["question"],
                    "golden_answer": row["gold_answers"],
                    "authoritative_history_before_action": sanitized_history(steps[: action_idx - 1]),
                    "current_action_reasoning": reasoning,
                    "current_executed_query": query,
                    "current_query_result": section["result_section"],
                    "current_query_result_sha256": section["result_sha256"],
                    "source_action_query_count": len(queries),
                    "query_position_in_action": query_idx,
                    "selection_strata": {
                        "action_position": bucket_action(action_idx, len(steps)),
                        "query_position": bucket_query(query_idx, len(queries)),
                        "action_query_count": bucket_count(len(queries)),
                        "result_characteristic": result_bucket,
                        "query_term_coverage": round(result_overlap, 6),
                        "verification_candidate": bool(VERIFY_RE.search(query + "\n" + reasoning)),
                        "already_resolved_candidate": bool(previous_queries) and history_sim >= 0.70,
                        "high_history_similarity": history_sim >= 0.55,
                        "max_prior_query_jaccard": round(history_sim, 6),
                        "action_level_disagreement_related": (qid, action_idx) in disagreement_actions,
                        "sibling_query_literal_in_reasoning": bool(sibling_in_reasoning),
                    },
                    "source_refs": {
                        "canonical_question_id": qid,
                        "raw_events_path": row["metadata"]["raw_events_path"],
                        "action_observation_sha256": step["observation_sha256"],
                    },
                    "status": "query_level_pilot_unsubmitted",
                    "_siblings": [s["result_section"] for i, s in enumerate(sections) if i != query_idx],
                    "_sibling_queries": sibling_queries,
                    "_future_steps": steps[action_idx:],
                    "_teacher_final_answer": trajectory["final"].get("answer") or "",
                }
                candidates.append(candidate)
                totals["queries"] += 1
            previous_queries.extend(queries)
            totals["actions"] += 1
        totals["trajectories"] += 1
    return candidates, dict(totals)


def select_pilot(candidates: list[dict], count: int, seed: int) -> list[dict]:
    targets = {
        "action_position": {"early": 40, "middle": 40, "late": 40},
        "query_position": {"single": 4, "first": 38, "middle": 40, "last": 38},
        "action_query_count": {"single": 4, "2": 32, "3": 42, ">3": 42},
        "result_characteristic": {
            "high_query_term_coverage": 40,
            "partial_query_term_coverage": 40,
            "weak_query_term_coverage": 40,
        },
    }
    special_targets = {
        "verification_candidate": 20,
        "already_resolved_candidate": 20,
        "high_history_similarity": 20,
        "action_level_disagreement_related": 15,
    }
    counts = {name: collections.Counter() for name in targets}
    specials = collections.Counter()
    selected = []
    used_trajectories = set()

    def add(candidate: dict):
        selected.append(candidate)
        used_trajectories.add(candidate["trajectory_id"])
        strata = candidate["selection_strata"]
        for name in targets:
            counts[name][strata[name]] += 1
        for name in special_targets:
            specials[name] += int(bool(strata[name]))

    # Preserve rare single-query actions first, one unit per trajectory.
    rare = [c for c in candidates if c["source_action_query_count"] == 1]
    for c in sorted(rare, key=lambda x: stable_key(seed, x["unit_id"])):
        if c["trajectory_id"] not in used_trajectories and len(selected) < count:
            add(c)

    # Include distinct trajectories associated with prior action-level disagreements.
    disagreement = [c for c in candidates if c["selection_strata"]["action_level_disagreement_related"]]
    for c in sorted(disagreement, key=lambda x: stable_key(seed + 1, x["unit_id"])):
        if specials["action_level_disagreement_related"] >= special_targets["action_level_disagreement_related"]:
            break
        if c["trajectory_id"] not in used_trajectories and not c["selection_strata"]["sibling_query_literal_in_reasoning"]:
            add(c)

    pool = [c for c in candidates if not c["selection_strata"]["sibling_query_literal_in_reasoning"]]
    while len(selected) < count:
        best = None
        best_key = None
        for c in pool:
            if c["trajectory_id"] in used_trajectories:
                continue
            s = c["selection_strata"]
            score = 0.0
            for name, goal in targets.items():
                target = goal[s[name]]
                deficit = max(0, target - counts[name][s[name]])
                score += deficit / max(1, target)
            for name, target in special_targets.items():
                if s[name]:
                    score += 1.5 * max(0, target - specials[name]) / max(1, target)
            score += int(c["action_idx"] > 1) * 0.02
            tie = stable_key(seed + len(selected), c["unit_id"])
            key = (score, tie)
            if best is None or best_key is None or key > best_key:
                best, best_key = c, key
        if best is None:
            raise RuntimeError("could not select requested count with unique trajectories and leakage-safe reasoning")
        add(best)
    return selected


def audit_and_clean(selected: list[dict]) -> tuple[list[dict], dict, list[dict]]:
    cleaned, audits = [], []
    for source in selected:
        unit = {k: v for k, v in source.items() if not k.startswith("_")}
        prompt = judge_prompt(unit)
        unit["judge_prompt"] = prompt
        violations = []
        # Audit the current-result payload boundary rather than the whole prompt:
        # a prior, legitimately visible observation can equal a later sibling result.
        current_result_tail = prompt.rsplit("CURRENT QUERY-SPECIFIC RESULT:\n", 1)[-1].rstrip("\n")
        if current_result_tail != unit["current_query_result"]:
            violations.append("current_result_boundary_mismatch")
        if source["_siblings"] and any(section in current_result_tail for section in source["_siblings"]):
            violations.append("sibling_result_leakage")
        if source["selection_strata"]["sibling_query_literal_in_reasoning"]:
            violations.append("sibling_query_literal_in_reasoning")
        # Future steps are never serialized. Literal matching against the full
        # prompt would falsely flag a valid repeated result from prior history.
        if any(k in unit for k in ("future_steps", "future_observations")):
            violations.append("future_observation_leakage")
        if "raw_textual_action" in json.dumps(unit, ensure_ascii=False):
            violations.append("raw_textual_action_present")
        # The final teacher answer is excluded structurally. Literal matching
        # would be invalid because retrieval evidence may naturally contain it.
        if any(k in unit for k in ("teacher_final_answer", "final_answer", "prediction")):
            violations.append("teacher_final_answer_field_present")
        if unit["current_executed_query"] not in prompt or unit["current_query_result"] not in prompt:
            violations.append("current_query_or_result_missing")
        audit = {
            "unit_id": unit["unit_id"],
            "question_id": unit["question_id"],
            "action_idx": unit["action_idx"],
            "query_idx": unit["query_idx"],
            "prompt_sha256": sha256_text(prompt),
            "current_result_sha256": unit["current_query_result_sha256"],
            "history_step_count": len(unit["authoritative_history_before_action"]),
            "sibling_result_count_excluded": len(source["_siblings"]),
            "future_step_count_excluded": len(source["_future_steps"]),
            "violations": violations,
            "pass": not violations,
        }
        audits.append(audit)
        cleaned.append(unit)
    summary = {
        "units": len(cleaned),
        "unique_unit_ids": len({u["unit_id"] for u in cleaned}),
        "unique_trajectories": len({u["trajectory_id"] for u in cleaned}),
        "unique_actions": len({(u["trajectory_id"], u["action_idx"]) for u in cleaned}),
        "sibling_result_leakage": sum("sibling_result_leakage" in a["violations"] for a in audits),
        "sibling_query_literal_in_reasoning": sum("sibling_query_literal_in_reasoning" in a["violations"] for a in audits),
        "future_observation_leakage": sum("future_observation_leakage" in a["violations"] for a in audits),
        "raw_textual_action_present": sum("raw_textual_action_present" in a["violations"] for a in audits),
        "teacher_final_answer_field_present": sum("teacher_final_answer_field_present" in a["violations"] for a in audits),
        "all_pass": all(a["pass"] for a in audits),
    }
    return cleaned, summary, audits


def distributions(units: list[dict]) -> dict:
    output = {}
    for name in ["action_position", "query_position", "action_query_count", "result_characteristic"]:
        output[name] = dict(sorted(collections.Counter(u["selection_strata"][name] for u in units).items()))
    for name in ["verification_candidate", "already_resolved_candidate", "high_history_similarity", "action_level_disagreement_related"]:
        output[name] = sum(bool(u["selection_strata"][name]) for u in units)
    return output


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--disagreements", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates, source_totals = build_candidates(args.canonical, args.disagreements)
    if source_totals != {"queries": 14279, "actions": 4723, "trajectories": 632}:
        raise RuntimeError(f"authoritative source totals changed: {source_totals}")
    selected = select_pilot(candidates, args.count, args.seed)
    units, leakage, audits = audit_and_clean(selected)
    if not leakage["all_pass"]:
        raise RuntimeError(f"leakage audit failed: {leakage}")
    manifest = {
        "status": "query_level_pilot_unsubmitted",
        "api_called": False,
        "seed": args.seed,
        "source": str(args.canonical),
        "source_sha256": sha256_file(args.canonical),
        "disagreement_source": str(args.disagreements),
        "disagreement_source_sha256": sha256_file(args.disagreements),
        "source_totals": source_totals,
        "pilot": {
            "query_units": len(units),
            "trajectories": len({u["trajectory_id"] for u in units}),
            "actions": len({(u["trajectory_id"], u["action_idx"]) for u in units}),
        },
        "strata": distributions(units),
        "leakage_audit": leakage,
        "authority": "executed_action + matching authoritative observation section",
        "excluded": ["raw textual current action", "sibling queries", "sibling results", "future steps", "teacher final answer"],
        "judge_config_planned": {
            "models": ["qwen3.5-flash", "qwen3.5-plus"],
            "temperature": 0,
            "enable_thinking": False,
            "max_tokens": 384,
        },
    }
    write_jsonl(args.output_dir / "pilot_units.jsonl", units)
    write_jsonl(args.output_dir / "schema_leakage_audit.jsonl", audits)
    (args.output_dir / "pilot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
