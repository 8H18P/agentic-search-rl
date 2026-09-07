#!/usr/bin/env python3
"""Download the public policy models used by this workspace to local folders."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--model", choices=("qwen", "smartsearch", "all"), default="all"
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.model in {"qwen", "all"}:
        from modelscope import snapshot_download as modelscope_snapshot_download

        target = args.output_root / "Qwen2.5-3B-Instruct"
        print(f"downloading Qwen/Qwen2.5-3B-Instruct -> {target}", flush=True)
        modelscope_snapshot_download(
            "Qwen/Qwen2.5-3B-Instruct", local_dir=str(target)
        )

    if args.model in {"smartsearch", "all"}:
        from huggingface_hub import snapshot_download as hf_snapshot_download

        target = args.output_root / "SmartSearch-3B"
        print(f"downloading vvv111222/SmartSearch-3B -> {target}", flush=True)
        hf_snapshot_download(
            repo_id="vvv111222/SmartSearch-3B", local_dir=str(target)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
