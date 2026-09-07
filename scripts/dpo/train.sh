#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "用法: $0 <dpo-config.json>" >&2
  exit 2
fi
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
python_bin=${DPO_PYTHON:-python}
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m canonical_sft.dpo_runner --config "$1"
