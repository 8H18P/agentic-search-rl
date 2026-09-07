#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/research-agent"

CORPUS="$ROOT/retriever_data/corpus/wiki18_100w.jsonl"
INDEX="$ROOT/retriever_data/index/e5_flat/e5_flat_inner.index"
E5_MODEL="$ROOT/retriever_data/models/e5-base-v2"

SMARTSEARCH="$ROOT/src/smartsearch"
SERVER="$SMARTSEARCH/scripts/serving/retriever_serving.py"
CONFIG="$SMARTSEARCH/scripts/serving/retriever_config_offline.yaml"

LOG_DIR="$ROOT/logs/retriever"
LOG_FILE="$LOG_DIR/offline_8766.log"

PORT=8766

EXPECTED_CORPUS_SIZE=14393573105
EXPECTED_INDEX_SIZE=64559075373

mkdir -p "$LOG_DIR"

echo "=================================================="
echo " Offline Retriever Deployment"
echo "=================================================="

cd "$ROOT"

# --------------------------------------------------
# 1. Hardware / resources
# --------------------------------------------------

echo
echo "[1/8] Hardware"

echo "CPU cores:"
nproc

echo
echo "Memory:"
free -h || true

echo
echo "cgroup memory.max:"
MEM_MAX=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo "unknown")
echo "$MEM_MAX"

if [[ "$MEM_MAX" =~ ^[0-9]+$ ]]; then
    python3 - <<PY
m = int("$MEM_MAX")
print(f"cgroup memory limit = {m / 1024**3:.2f} GiB")
PY

    # 80 GiB is not a theoretical requirement,
    # but below this normal read_index of a ~60 GiB Flat index is risky.
    MIN_MEM=$((80 * 1024 * 1024 * 1024))
    if (( MEM_MAX < MIN_MEM )); then
        echo
        echo "ERROR: cgroup memory limit < 80 GiB."
        echo "The FAISS index alone is ~60 GiB."
        echo "Do NOT attempt normal faiss.read_index() on this instance."
        exit 20
    fi
fi

echo
echo "Disk:"
df -h "$ROOT" || true

echo
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true


# --------------------------------------------------
# 2. Verify files
# --------------------------------------------------

echo
echo "[2/8] Verify offline retrieval resources"

for p in "$CORPUS" "$INDEX" "$E5_MODEL" "$SERVER"; do
    if [[ ! -e "$p" ]]; then
        echo "ERROR: missing:"
        echo "  $p"
        exit 21
    fi
done

CORPUS_SIZE=$(stat -c '%s' "$CORPUS")
INDEX_SIZE=$(stat -c '%s' "$INDEX")

echo "Corpus:"
echo "  $CORPUS"
echo "  size = $CORPUS_SIZE"

echo "FAISS index:"
echo "  $INDEX"
echo "  size = $INDEX_SIZE"

echo "E5:"
echo "  $E5_MODEL"

if [[ "$CORPUS_SIZE" != "$EXPECTED_CORPUS_SIZE" ]]; then
    echo
    echo "ERROR: corpus size mismatch."
    echo "Expected: $EXPECTED_CORPUS_SIZE"
    echo "Actual:   $CORPUS_SIZE"
    exit 22
fi

if [[ "$INDEX_SIZE" != "$EXPECTED_INDEX_SIZE" ]]; then
    echo
    echo "ERROR: FAISS index size mismatch."
    echo "Expected: $EXPECTED_INDEX_SIZE"
    echo "Actual:   $INDEX_SIZE"
    exit 23
fi

echo "Resource sizes: OK"


# --------------------------------------------------
# 3. Find a usable Python environment
# --------------------------------------------------

echo
echo "[3/8] Find Python environment"

CANDIDATES=(
    "$ROOT/.venv"
    "$ROOT/.venv-faiss"
    "$ROOT/.venv-retriever"
)

PYTHON_BIN=""

for ENV_DIR in "${CANDIDATES[@]}"; do
    PY="$ENV_DIR/bin/python"

    if [[ ! -x "$PY" ]]; then
        continue
    fi

    echo
    echo "Testing: $ENV_DIR"

    if "$PY" - <<'PY' >/tmp/retriever_env_check.log 2>&1
import torch
import transformers
import numpy
import faiss
import fastapi
import uvicorn
import pydantic

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("faiss", faiss.__version__)
print("fastapi", fastapi.__version__)
print("uvicorn", uvicorn.__version__)
print("pydantic", pydantic.__version__)
PY
    then
        cat /tmp/retriever_env_check.log
        PYTHON_BIN="$PY"
        echo
        echo "Selected environment: $ENV_DIR"
        break
    else
        echo "Not usable:"
        cat /tmp/retriever_env_check.log
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    echo
    echo "=================================================="
    echo "NO EXISTING ENV HAS ALL RETRIEVER DEPENDENCIES"
    echo "=================================================="
    echo
    echo "This script intentionally did NOT modify your environments."
    echo
    echo "We need one Python environment containing at least:"
    echo
    echo "  torch"
    echo "  transformers"
    echo "  numpy"
    echo "  faiss"
    echo "  fastapi"
    echo "  uvicorn"
    echo "  pydantic"
    echo
    echo "Paste the error above back to ChatGPT and we will"
    echo "create/fix .venv-retriever without touching the training env."
    exit 30
fi


# --------------------------------------------------
# 4. Verify local FlashRAG import
# --------------------------------------------------

echo
echo "[4/8] Verify SmartSearch / FlashRAG"

export PYTHONPATH="$SMARTSEARCH/src:${PYTHONPATH:-}"

"$PYTHON_BIN" - <<PY
import sys

print("python =", sys.executable)

import flashrag
print("flashrag import OK")

from flashrag.config import Config
from flashrag.utils import get_retriever

print("FlashRAG Config/get_retriever import OK")
PY


# --------------------------------------------------
# 5. Inspect retriever config-related source
# --------------------------------------------------

echo
echo "[5/8] Inspect FlashRAG retriever implementation"

echo "Relevant local source references:"
grep -R \
    -n \
    -E "retrieval_model_path|retrieval_method|pooling_method|faiss_gpu|index_path|query.*prefix|query:" \
    "$SMARTSEARCH/src/flashrag" \
    2>/dev/null \
    | head -80 \
    || true


# --------------------------------------------------
# 6. Generate OFFLINE config
# --------------------------------------------------

echo
echo "[6/8] Generate offline retriever config"

cat > "$CONFIG" <<EOF
# Auto-generated offline retrieval configuration.
# Stage-1 environment:
#   engine routing is non-causal
#   all searches use this same offline E5 backend.

gpu_id: "0"

retrieval_method: "e5"

retrieval_model_path: "$E5_MODEL"

index_path: "$INDEX"

faiss_gpu: false

corpus_path: "$CORPUS"

retrieval_topk: 5

retrieval_use_fp16: false

retrieval_query_max_length: 128

retrieval_pooling_method: "mean"
EOF

echo "Created:"
echo "  $CONFIG"

cat "$CONFIG"


# --------------------------------------------------
# 7. Minimal FAISS header/load validation
# --------------------------------------------------

echo
echo "[7/8] Load FAISS index"

echo
echo "WARNING:"
echo "This step may consume ~60+ GiB RAM and can take some time."
echo

"$PYTHON_BIN" - <<PY
import os
import time
import faiss

path = r"$INDEX"

print("index_path =", path)
print("index_size_GiB =", os.path.getsize(path) / 1024**3)

t0 = time.time()

index = faiss.read_index(path)

dt = time.time() - t0

print()
print("FAISS LOAD OK")
print("load_seconds =", round(dt, 2))
print("type =", type(index).__name__)
print("d =", index.d)
print("ntotal =", index.ntotal)

try:
    print("metric_type =", index.metric_type)
except Exception:
    pass

if index.d != 768:
    raise RuntimeError(
        f"Unexpected index dimension: {index.d}; expected 768 for e5-base-v2"
    )
PY


# --------------------------------------------------
# 8. Start server
# --------------------------------------------------

echo
echo "[8/8] Start Offline Retriever :$PORT"

# Clean only our own old screen if it exists.
screen -S offline_retriever_8766 -X quit >/dev/null 2>&1 || true

# Refuse to start if some unrelated process already owns the port.
if command -v lsof >/dev/null 2>&1; then
    if lsof -iTCP:$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "ERROR: TCP port $PORT is already in use."
        lsof -iTCP:$PORT -sTCP:LISTEN || true
        exit 40
    fi
fi

screen -dmS offline_retriever_8766 bash -lc "
cd '$ROOT'
export PYTHONPATH='$SMARTSEARCH/src':\${PYTHONPATH:-}
exec '$PYTHON_BIN' '$SERVER' \
  --config '$CONFIG' \
  --num_retriever 1 \
  --port '$PORT' \
  >> '$LOG_FILE' 2>&1
"

echo
echo "Waiting for server initialization..."

for i in $(seq 1 120); do

    if curl -fsS \
        --max-time 2 \
        "http://127.0.0.1:$PORT/health" \
        >/tmp/offline_retriever_health.txt 2>/dev/null
    then
        echo
        echo "=============================================="
        echo "OFFLINE RETRIEVER IS READY"
        echo "=============================================="
        cat /tmp/offline_retriever_health.txt
        echo
        break
    fi

    if ! screen -ls 2>/dev/null | grep -q "offline_retriever_8766"; then
        echo
        echo "ERROR: retriever server exited during startup."
        echo
        echo "Last log lines:"
        tail -100 "$LOG_FILE" || true
        exit 41
    fi

    printf "\rwaiting... %3d/120" "$i"
    sleep 2

    if [[ "$i" == "120" ]]; then
        echo
        echo "ERROR: server did not become healthy."
        tail -100 "$LOG_FILE" || true
        exit 42
    fi
done


# --------------------------------------------------
# Smoke search
# --------------------------------------------------

echo
echo
echo "Running smoke query..."

curl -sS \
    -X POST \
    "http://127.0.0.1:$PORT/search" \
    -H 'Content-Type: application/json' \
    -d '{
          "query": "Marie Curie husband",
          "top_n": 5,
          "return_score": true
        }'

echo
echo
echo "=============================================="
echo "DEPLOYMENT FINISHED"
echo "=============================================="
echo
echo "Retriever:"
echo "  http://127.0.0.1:$PORT"
echo
echo "screen:"
echo "  offline_retriever_8766"
echo
echo "logs:"
echo "  $LOG_FILE"
echo
echo "Useful commands:"
echo
echo "  screen -ls"
echo
echo "  tail -f $LOG_FILE"
echo
echo "  curl http://127.0.0.1:$PORT/health"
echo
echo "  screen -S offline_retriever_8766 -X quit"
echo

