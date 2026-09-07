#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "用法: $0 <grpo-config.json> [--resume]" >&2
  exit 2
fi
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
python_bin=${GRPO_PYTHON:-python}
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" "$repo_root/scripts/grpo_runtime/train.py" "$@"
