#!/usr/bin/env python3
"""Expand a selected real canonical subset for one auditable accumulation cycle."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--micro-batches", type=int, default=64)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).open() if line.strip()]
    if len(rows) < 3 or args.micro_batches < len(rows):
        raise ValueError("trial needs at least three selected rows and enough micro-batches")
    ordered = rows + [rows[0]] * (args.micro_batches - len(rows))
    materialized = []
    for index, source in enumerate(ordered, 1):
        row = dict(source)
        row["sample_id"] = f"production-trial-micro-{index:04d}"
        row["metadata"] = dict(source.get("metadata", {}))
        row["metadata"].update(
            {
                "engineering_trial_only": True,
                "trial_micro_step": index,
                "repeated_from_sample_id": source.get("sample_id"),
            }
        )
        materialized.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in materialized),
        encoding="utf-8",
    )
    counts = {}
    for row in materialized:
        key = row["metadata"]["repeated_from_sample_id"]
        counts[key] = counts.get(key, 0) + 1
    Path(args.manifest).write_text(
        json.dumps(
            {
                "engineering_trial_only": True,
                "source_subset": str(Path(args.input).resolve()),
                "micro_batches": len(materialized),
                "policy": "include short, medium, and over-cutoff once; fill remaining cycle with short",
                "repeat_counts": counts,
                "validation400_used": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
