"""Query-level process reward 的输入隔离与 prompt 构造。"""
from __future__ import annotations

import hashlib
import json
import re


HEADER_RE = re.compile(r"(?m)^Search results for: (.*)$")


def parse_sections(observation: str, queries: list[str]) -> list[dict]:
    matches = list(HEADER_RE.finditer(observation))
    if len(matches) != len(queries):
        raise ValueError(f"query/result count mismatch: {len(queries)} != {len(matches)}")
    sections = []
    for index, (query, match) in enumerate(zip(queries, matches)):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(observation)
        section = observation[match.start():end].rstrip()
        header = match.group(1).strip()
        if header != query.strip():
            raise ValueError(f"header mismatch at query {index}: {header!r} != {query!r}")
        sections.append({
            "query_idx": index,
            "query": query,
            "section_header": header,
            "result_section": section,
            "result_sha256": hashlib.sha256(section.encode()).hexdigest(),
        })
    return sections


def history_text(history: list[dict]) -> str:
    if not history:
        return "(No previous Search action.)"
    return "\n\n".join(
        f"Previous action {step['action_idx']} reasoning:\n{step['reasoning']}\n"
        f"Previous executed action:\n{json.dumps(step['executed_action'], ensure_ascii=False)}\n"
        f"Previous authoritative observation:\n{step['observation']}"
        for step in history
    )


def retrieval_prompt(query: str, result: str) -> str:
    return f"""You are a query-local retrieval evaluator. Judge only whether CURRENT QUERY-SPECIFIC RESULT provides the information requested by CURRENT EXECUTED QUERY.

Use no unstated original question, history, sibling query, future constraint, or final answer. Related entities alone are insufficient; the intended and retrieved entity must match. If the query asks several explicit fields, all are needed.

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

Use only the original question, golden answer, authoritative history before this Search action, current action reasoning, and current query. You are not shown and must not infer whether retrieval succeeded. Verification or repetition is handled by a separate efficiency diagnostic.

Intent=0 only for a clearly irrelevant query, a clearly wrong entity already disproved by the shown context, a malformed query, or a query with no reasonable connection to the task.

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
