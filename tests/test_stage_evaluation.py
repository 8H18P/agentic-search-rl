import json

import pytest

from agentic_search_rl.evaluation.stages import aggregate, compare


def write_rows(path, question_ids=("q1", "q2")):
    rows = []
    for index, question_id in enumerate(question_ids):
        rows.append({
            "question_id": question_id,
            "answer_em": float(index == 0),
            "answer_f1": 0.5,
            "task_success": index == 0,
            "search_action_count": 1,
            "search_query_count": 2,
            "valid_query_count": 1,
            "repeated_query_count": 0,
            "retrieval_success_count": 1,
            "process_scores": [1, 0],
            "trajectory_reward": 0.25,
            "termination_reason": "final_answer",
        })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_aggregate_exact_metrics(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_rows(path)
    result = aggregate(path)
    assert result["answer_em"] == 0.5
    assert result["valid_query_ratio"] == 0.5
    assert result["mean_process_reward"] == 0.5


def test_compare_rejects_different_question_sets(tmp_path):
    paths = {}
    for stage in ("Base", "SFT", "DPO", "GRPO"):
        path = tmp_path / f"{stage}.jsonl"
        write_rows(path, ("different",) if stage == "GRPO" else ("q1", "q2"))
        paths[stage] = path
    with pytest.raises(ValueError, match="问题 ID 集合不一致"):
        compare(paths)
