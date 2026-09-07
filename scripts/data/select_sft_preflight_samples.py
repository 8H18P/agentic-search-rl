#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cutoff", type=int, default=16384)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("canonical_builder", root / "scripts/data/build_sft_candidate_v1.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=True
    )
    candidates = [json.loads(line) for line in Path(args.candidates).open() if line.strip()]
    enriched = []
    for index, candidate in enumerate(candidates):
        messages = builder.build_messages(candidate)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        enriched.append((tokens, index, candidate, messages))
    ordered = sorted(enriched)
    selected = [
        ordered[0],
        ordered[len(ordered) // 2],
        min((item for item in ordered if item[0] > args.cutoff), key=lambda item: item[0]),
    ]

    rows, records = [], []
    for kind, (tokens, index, candidate, messages) in zip(("short", "medium", "over_cutoff"), selected):
        trajectory = candidate["canonical_trajectory"]
        raw_path = candidate.get("metadata", {}).get("raw_trajectory_path")
        trajectory_id = (
            candidate.get("trajectory_id")
            or trajectory.get("trajectory_id")
            or (Path(raw_path).stem if raw_path else None)
        )
        row = {
            "sample_id": f"sft-preflight-{kind}",
            "trajectory_id": trajectory_id,
            "question_id": candidate.get("question_id"),
            "question": candidate.get("question"),
            "messages": messages,
            "metadata": {
                "source_dataset": "canonical_candidates.jsonl",
                "source_row_index": index,
                "source_of_truth": {
                    "action": "canonical_trajectory.steps[].executed_action",
                    "observation": "canonical_trajectory.steps[].observation",
                    "final": "canonical_trajectory.final",
                },
                "original_token_count": tokens,
                "selection_bucket": kind,
            },
        }
        rows.append(row)
        records.append(
            {
                "sample_id": row["sample_id"],
                "question_id": row["question_id"],
                "trajectory_id": trajectory_id,
                "source_row_index": index,
                "original_token_count": tokens,
                "will_truncate": tokens > args.cutoff,
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    source_digest = hashlib.sha256(Path(args.candidates).read_bytes()).hexdigest()
    Path(args.manifest).write_text(
        json.dumps(
            {
                "source": str(Path(args.candidates).resolve()),
                "source_sha256": source_digest,
                "source_count": len(candidates),
                "cutoff": args.cutoff,
                "selection_policy": "shortest, median, shortest strictly over cutoff",
                "records": records,
                "validation400_used": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
