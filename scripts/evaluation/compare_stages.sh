#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
python_bin=${EVAL_PYTHON:-python}
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m agentic_search_rl.evaluation.stages "$@"
