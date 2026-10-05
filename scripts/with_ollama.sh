#!/usr/bin/env bash
# Run a command with a local Ollama server alive for its duration.
# Usage: scripts/with_ollama.sh <command...>
set -euo pipefail
OLLAMA_BIN="${OLLAMA_BIN:-/projects/ollama/bin/ollama}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-/projects/ollama/models}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"

if ! curl -sf localhost:11434/api/version >/dev/null 2>&1; then
  "$OLLAMA_BIN" serve >/tmp/ollama.log 2>&1 &
  SERVER_PID=$!
  trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
  for _ in $(seq 1 30); do
    curl -sf localhost:11434/api/version >/dev/null 2>&1 && break
    sleep 1
  done
fi
"$@"
