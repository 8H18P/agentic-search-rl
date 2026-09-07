#!/usr/bin/env python3
"""运行 GRPO 数据、奖励和 objective 的只读预检查。"""
from __future__ import annotations

import argparse
from pathlib import Path

from runner import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.resume, False))
