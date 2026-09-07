"""严格聚合 Base/SFT/DPO/GRPO 的统一逐题评测产物。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


STAGES = ("Base", "SFT", "DPO", "GRPO")
REQUIRED = {
    "question_id",
    "answer_em",
    "answer_f1",
    "task_success",
    "search_action_count",
    "search_query_count",
    "valid_query_count",
    "repeated_query_count",
    "retrieval_success_count",
    "process_scores",
    "trajectory_reward",
    "termination_reason",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path):
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise ValueError(f"评测文件为空: {path}")
    for index, row in enumerate(values, 1):
        missing = sorted(REQUIRED - row.keys())
        if missing:
            raise ValueError(f"{path}:{index} 缺少字段: {missing}")
        if row["search_query_count"] < row["valid_query_count"]:
            raise ValueError(f"{path}:{index} valid_query_count 超过 search_query_count")
        if row["search_query_count"] < row["repeated_query_count"]:
            raise ValueError(f"{path}:{index} repeated_query_count 超过 search_query_count")
        if any(value not in (0, 1) for value in row["process_scores"]):
            raise ValueError(f"{path}:{index} process_scores 必须是 0/1")
    return values


def aggregate(path: str | Path):
    path = Path(path)
    rows = _rows(path)
    count = len(rows)
    query_count = sum(row["search_query_count"] for row in rows)
    process = [value for row in rows for value in row["process_scores"]]
    return {
        "examples": count,
        "answer_em": sum(row["answer_em"] for row in rows) / count,
        "answer_f1": sum(row["answer_f1"] for row in rows) / count,
        "task_success_rate": sum(bool(row["task_success"]) for row in rows) / count,
        "average_search_actions": sum(row["search_action_count"] for row in rows) / count,
        "valid_query_ratio": sum(row["valid_query_count"] for row in rows) / query_count if query_count else 0.0,
        "repeated_query_ratio": sum(row["repeated_query_count"] for row in rows) / query_count if query_count else 0.0,
        "retrieval_success_rate": sum(row["retrieval_success_count"] for row in rows) / query_count if query_count else 0.0,
        "mean_process_reward": sum(process) / len(process) if process else 0.0,
        "mean_trajectory_reward": sum(row["trajectory_reward"] for row in rows) / count,
        "termination_reasons": {
            reason: sum(row["termination_reason"] == reason for row in rows)
            for reason in sorted({row["termination_reason"] for row in rows})
        },
        "source_path": str(path.resolve()),
        "source_sha256": _sha256(path),
        "question_ids_sha256": hashlib.sha256(
            "\n".join(sorted(str(row["question_id"]) for row in rows)).encode()
        ).hexdigest(),
    }


def compare(stage_paths: dict[str, str | Path]):
    if set(stage_paths) != set(STAGES):
        raise ValueError(f"必须且只能提供四阶段: {STAGES}")
    result = {stage: aggregate(stage_paths[stage]) for stage in STAGES}
    question_sets = {
        stage: result[stage]["question_ids_sha256"] for stage in STAGES
    }
    if len(set(question_sets.values())) != 1:
        raise ValueError(f"四阶段问题 ID 集合不一致: {question_sets}")
    return {"status": "complete", "same_question_set": True, "stages": result}


def _markdown(report):
    lines = [
        "# 四阶段统一评测",
        "",
        "| Model | EM | F1 | Success | Avg Search | Valid Query | Process Reward |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        row = report["stages"][stage]
        lines.append(
            f'| {stage} | {row["answer_em"]:.4f} | {row["answer_f1"]:.4f} | '
            f'{row["task_success_rate"]:.4f} | {row["average_search_actions"]:.4f} | '
            f'{row["valid_query_ratio"]:.4f} | {row["mean_process_reward"]:.4f} |'
        )
    lines.extend(["", "所有阶段使用相同 question ID 集合；详细 provenance 与扩展指标见同目录 JSON。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", action="append", required=True, metavar="NAME=JSONL",
        help="必须分别提供 Base、SFT、DPO、GRPO，可重复四次",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    mappings = {}
    for value in args.stage:
        name, separator, path = value.partition("=")
        if not separator or name in mappings:
            raise ValueError(f"非法或重复的 --stage: {value}")
        mappings[name] = path
    report = compare(mappings)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
