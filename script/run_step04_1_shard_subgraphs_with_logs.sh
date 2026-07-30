#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_ID="step04_1_${RUN_TIMESTAMP}"
RUNS_DIR="${RUNS_DIR:-drug_disease_validation/run_logs/step04_1_runs}"
RUN_DIR="${RUNS_DIR}/${RUN_ID}"
LOG_FILE="${RUN_DIR}/step04_1.log"
META_FILE="${RUN_DIR}/run_metadata.txt"
COMMAND_FILE="${RUN_DIR}/command.txt"

mkdir -p "${RUN_DIR}"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

STEP04_1_MANIFEST="${STEP04_1_MANIFEST:-drug_disease_validation/data/processed_step04_external/subgraph_manifest_io+emb1024.tsv}"
STEP04_1_OUTPUT_DIR="${STEP04_1_OUTPUT_DIR:-drug_disease_validation/data/processed_step04_external}"
STEP04_1_SHARD_SIZE="${STEP04_1_SHARD_SIZE:-1000}"
STEP04_1_DELETE_ORIGINALS="${STEP04_1_DELETE_ORIGINALS:-false}"
STEP04_1_DELETE_APPLEDOUBLE="${STEP04_1_DELETE_APPLEDOUBLE:-true}"
STEP04_1_OVERWRITE="${STEP04_1_OVERWRITE:-false}"
STEP04_1_RESUME="${STEP04_1_RESUME:-false}"
STEP04_1_SKIP_UNREADABLE="${STEP04_1_SKIP_UNREADABLE:-false}"
STEP04_1_MAX_SKIPPED_WARNINGS="${STEP04_1_MAX_SKIPPED_WARNINGS:-20}"
STEP04_1_STOP_AFTER_CONSECUTIVE_SKIPS="${STEP04_1_STOP_AFTER_CONSECUTIVE_SKIPS:-0}"

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
  drug_disease_validation.src.04_1_shard_subgraphs
  --manifest
  "${STEP04_1_MANIFEST}"
  --output-dir
  "${STEP04_1_OUTPUT_DIR}"
  --shard-size
  "${STEP04_1_SHARD_SIZE}"
)

if [[ "${STEP04_1_OVERWRITE}" == "true" || "${STEP04_1_OVERWRITE}" == "1" ]]; then
  COMMAND+=(--overwrite)
fi

if [[ "${STEP04_1_RESUME}" == "true" || "${STEP04_1_RESUME}" == "1" ]]; then
  COMMAND+=(--resume)
fi

if [[ "${STEP04_1_SKIP_UNREADABLE}" == "true" || "${STEP04_1_SKIP_UNREADABLE}" == "1" ]]; then
  COMMAND+=(
    --skip-unreadable
    --max-skipped-warnings "${STEP04_1_MAX_SKIPPED_WARNINGS}"
    --stop-after-consecutive-skips "${STEP04_1_STOP_AFTER_CONSECUTIVE_SKIPS}"
  )
fi

if [[ "${STEP04_1_DELETE_ORIGINALS}" == "true" || "${STEP04_1_DELETE_ORIGINALS}" == "1" ]]; then
  COMMAND+=(--delete-originals)
  if [[ "${STEP04_1_DELETE_APPLEDOUBLE}" == "true" || "${STEP04_1_DELETE_APPLEDOUBLE}" == "1" ]]; then
    COMMAND+=(--delete-appledouble)
  fi
fi

COMMAND+=("$@")

{
  echo "run_id=${RUN_ID}"
  echo "run_timestamp=${RUN_TIMESTAMP}"
  echo "repo_root=."
  echo "python_bin=${PYTHON_BIN}"
  echo "step04_1_manifest=${STEP04_1_MANIFEST}"
  echo "step04_1_output_dir=${STEP04_1_OUTPUT_DIR}"
  echo "step04_1_shard_size=${STEP04_1_SHARD_SIZE}"
  echo "step04_1_delete_originals=${STEP04_1_DELETE_ORIGINALS}"
  echo "step04_1_delete_appledouble=${STEP04_1_DELETE_APPLEDOUBLE}"
  echo "step04_1_overwrite=${STEP04_1_OVERWRITE}"
  echo "step04_1_resume=${STEP04_1_RESUME}"
  echo "step04_1_skip_unreadable=${STEP04_1_SKIP_UNREADABLE}"
  echo "step04_1_max_skipped_warnings=${STEP04_1_MAX_SKIPPED_WARNINGS}"
  echo "step04_1_stop_after_consecutive_skips=${STEP04_1_STOP_AFTER_CONSECUTIVE_SKIPS}"
  echo "git_branch=${GIT_BRANCH}"
  echo "git_commit=${GIT_COMMIT}"
  echo "git_dirty=${GIT_DIRTY}"
  echo "host=$(hostname)"
  echo "user=${USER:-unknown}"
  echo "started_at=$(date +"%Y-%m-%d %H:%M:%S %Z")"
} > "${META_FILE}"

printf '%q ' "${COMMAND[@]}" > "${COMMAND_FILE}"
printf '\n' >> "${COMMAND_FILE}"

echo "[${RUN_ID}] starting Step 4.1 sharding run"
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
