#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${RAG_PYTHON:-}" ]; then
  PYTHON_BIN="$RAG_PYTHON"
elif [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
elif [ -x "$PROJECT_DIR/../../work/rag-runtime/venv/bin/python" ]; then
  PYTHON_BIN="$PROJECT_DIR/../../work/rag-runtime/venv/bin/python"
  export RAG_CACHE="${RAG_CACHE:-$PROJECT_DIR/../../work/rag-runtime/cache}"
else
  echo '请按 README 创建 .venv，或设置 RAG_PYTHON。' >&2
  exit 1
fi
exec "$PYTHON_BIN" "$PROJECT_DIR/rag.py" "$@"
