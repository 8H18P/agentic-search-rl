#!/usr/bin/env python3
"""Serialize all authoritative canonical candidates without raw response parsing."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from transformers import AutoTokenizer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=True)
    candidates = [json.loads(line) for line in Path(args.candidates).open() if line.strip()]
    rows, records = [], []
    original_total = post_cutoff_total = truncated = 0
    for index, candidate in enumerate(candidates):
        messages = builder.build_messages(candidate)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        raw_path = candidate.get("metadata", {}).get("raw_trajectory_path")
        trajectory_id = Path(raw_path).stem if raw_path else None
        rows.append(
            {
                "sample_id": f"formal-sft-{index + 1:04d}",
                "question_id": candidate.get("question_id"),
                "trajectory_id": trajectory_id,
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
                    "original_token_count": length,
                },
            }
        )
        records.append({"sample_id": rows[-1]["sample_id"], "question_id": candidate.get("question_id"), "trajectory_id": trajectory_id, "original_tokens": length, "post_cutoff_tokens": min(length, args.cutoff), "truncated": length > args.cutoff})
        original_total += length
        post_cutoff_total += min(length, args.cutoff)
        truncated += int(length > args.cutoff)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    Path(args.manifest).write_text(
        json.dumps(
            {
                "dataset": "formal_stage1_sft_632",
                "source": str(Path(args.candidates).resolve()),
                "source_sha256": digest(args.candidates),
                "serialized_sha256": digest(output),
                "count": len(rows),
                "cutoff": args.cutoff,
                "truncated_sample_count": truncated,
                "truncation_rate": truncated / len(rows),
                "original_total_tokens": original_total,
                "post_cutoff_total_tokens": post_cutoff_total,
                "provenance": "build_messages(executed_action, observation, canonical final); no raw response parsing",
                "records": records,
                "validation400_used": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"count": len(rows), "truncated": truncated, "original_tokens": original_total, "post_cutoff_tokens": post_cutoff_total}))


if __name__ == "__main__":
    main()
