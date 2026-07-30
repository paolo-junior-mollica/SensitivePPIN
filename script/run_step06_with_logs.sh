#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
RUN_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_ID="step06_${RUN_TIMESTAMP}"
RUNS_DIR="${RUNS_DIR:-drug_disease_validation/run_logs/step06_runs}"
RUN_DIR="${RUNS_DIR}/${RUN_ID}"
LOG_FILE="${RUN_DIR}/step06.log"
META_FILE="${RUN_DIR}/run_metadata.txt"
COMMAND_FILE="${RUN_DIR}/command.txt"

mkdir -p "${RUN_DIR}"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="false"
if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
  GIT_DIRTY="true"
fi

COMMAND=(
  "poetry"
  "run"
  "${PYTHON_BIN}"
  -m
  drug_disease_validation.src.06_find_negative_pathways
  "$@"
)

{
  echo "run_id=${RUN_ID}"
  echo "run_timestamp=${RUN_TIMESTAMP}"
  echo "repo_root=."
  echo "python_bin=${PYTHON_BIN}"
  echo "git_branch=${GIT_BRANCH}"
  echo "git_commit=${GIT_COMMIT}"
  echo "git_dirty=${GIT_DIRTY}"
  echo "host=$(hostname)"
  echo "user=${USER:-unknown}"
  echo "started_at=$(date +"%Y-%m-%d %H:%M:%S %Z")"
} > "${META_FILE}"

printf '%q ' "${COMMAND[@]}" > "${COMMAND_FILE}"
printf '\n' >> "${COMMAND_FILE}"

echo "[${RUN_ID}] starting Step 6 run"
echo "[${RUN_ID}] logs: ${LOG_FILE}"
echo "[${RUN_ID}] metadata: ${META_FILE}"

"${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
COMMAND_EXIT_CODE=${PIPESTATUS[0]}

{
  echo "finished_at=$(date +"%Y-%m-%d %H:%M:%S %Z")"
  echo "exit_code=${COMMAND_EXIT_CODE}"
} >> "${META_FILE}"

echo "[${RUN_ID}] completed with exit code ${COMMAND_EXIT_CODE}"
exit "${COMMAND_EXIT_CODE}"
