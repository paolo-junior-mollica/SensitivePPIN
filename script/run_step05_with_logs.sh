#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_ID="step05_${RUN_TIMESTAMP}"
RUNS_DIR="${RUNS_DIR:-drug_disease_validation/run_logs/step05_runs}"
RUN_DIR="${RUNS_DIR}/${RUN_ID}"
LOG_FILE="${RUN_DIR}/step05.log"
META_FILE="${RUN_DIR}/run_metadata.txt"
COMMAND_FILE="${RUN_DIR}/command.txt"

mkdir -p "${RUN_DIR}"
mkdir -p "${RUN_DIR}/cache/matplotlib" "${RUN_DIR}/cache/xdg"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

DEFAULT_STEP05_MANIFEST="drug_disease_validation/data/processed_step04_external/subgraph_manifest_io+emb1024.tsv"
if [[ -f "drug_disease_validation/data/processed_step04_external/subgraph_manifest_io+emb1024_sharded.tsv" ]]; then
  DEFAULT_STEP05_MANIFEST="drug_disease_validation/data/processed_step04_external/subgraph_manifest_io+emb1024_sharded.tsv"
fi
STEP05_MANIFEST="${STEP05_MANIFEST:-${DEFAULT_STEP05_MANIFEST}}"
STEP05_NEGATIVE_MANIFEST="${STEP05_NEGATIVE_MANIFEST:-}"
STEP05_OUTPUT_DIR="${STEP05_OUTPUT_DIR:-drug_disease_validation/data/processed}"
STEP05_LIMIT="${STEP05_LIMIT:-100}"
STEP05_NEGATIVE_LIMIT="${STEP05_NEGATIVE_LIMIT:-}"
STEP05_ACCELERATOR="${STEP05_ACCELERATOR:-cpu}"
STEP05_PREDICTION_CHUNK_SIZE="${STEP05_PREDICTION_CHUNK_SIZE:-1000}"
STEP05_LIMIT_MODE="limited"
STEP05_LIMIT_DISPLAY="${STEP05_LIMIT}"
if [[ "${STEP05_LIMIT}" == "all" || "${STEP05_LIMIT}" == "full" || "${STEP05_LIMIT}" == "none" ]]; then
  STEP05_LIMIT_MODE="all"
  STEP05_LIMIT_DISPLAY="all"
elif [[ -z "${STEP05_LIMIT}" ]]; then
  echo "STEP05_LIMIT cannot be empty. Use STEP05_LIMIT=all for a full run." >&2
  exit 2
fi

EFFECTIVE_OUTPUT_DIR="${STEP05_OUTPUT_DIR}"
EXTRA_ARGS=("$@")
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  case "${EXTRA_ARGS[$i]}" in
    --output-dir)
      if (( i + 1 < ${#EXTRA_ARGS[@]} )); then
        EFFECTIVE_OUTPUT_DIR="${EXTRA_ARGS[$((i + 1))]}"
      fi
      ;;
    --output-dir=*)
      EFFECTIVE_OUTPUT_DIR="${EXTRA_ARGS[$i]#--output-dir=}"
      ;;
  esac
done

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
  drug_disease_validation.src.05_predict
  --manifest
  "${STEP05_MANIFEST}"
  --output-dir
  "${STEP05_OUTPUT_DIR}"
  --accelerator
  "${STEP05_ACCELERATOR}"
  --prediction-chunk-size
  "${STEP05_PREDICTION_CHUNK_SIZE}"
)

if [[ -n "${STEP05_NEGATIVE_MANIFEST}" ]]; then
  COMMAND+=(--negative-manifest "${STEP05_NEGATIVE_MANIFEST}")
fi

if [[ "${STEP05_LIMIT_MODE}" == "limited" ]]; then
  COMMAND+=(--limit "${STEP05_LIMIT}")
fi

if [[ -n "${STEP05_NEGATIVE_LIMIT}" ]]; then
  COMMAND+=(--negative-limit "${STEP05_NEGATIVE_LIMIT}")
fi

COMMAND+=("$@")

{
  echo "run_id=${RUN_ID}"
  echo "run_timestamp=${RUN_TIMESTAMP}"
  echo "repo_root=."
  echo "python_bin=${PYTHON_BIN}"
  echo "step05_manifest=${STEP05_MANIFEST}"
  echo "step05_negative_manifest=${STEP05_NEGATIVE_MANIFEST:-none}"
  echo "step05_output_dir=${EFFECTIVE_OUTPUT_DIR}"
  echo "step05_limit=${STEP05_LIMIT_DISPLAY}"
  echo "step05_negative_limit=${STEP05_NEGATIVE_LIMIT:-default}"
  echo "step05_limit_mode=${STEP05_LIMIT_MODE}"
  echo "step05_accelerator=${STEP05_ACCELERATOR}"
  echo "step05_prediction_chunk_size=${STEP05_PREDICTION_CHUNK_SIZE}"
  echo "git_branch=${GIT_BRANCH}"
  echo "git_commit=${GIT_COMMIT}"
  echo "git_dirty=${GIT_DIRTY}"
  echo "host=$(hostname)"
  echo "user=${USER:-unknown}"
  echo "started_at=$(date +"%Y-%m-%d %H:%M:%S %Z")"
} > "${META_FILE}"

printf '%q ' "${COMMAND[@]}" > "${COMMAND_FILE}"
printf '\n' >> "${COMMAND_FILE}"

touch "${LOG_FILE}"

log_line() {
  echo "$*" | tee -a "${LOG_FILE}"
}

print_step05_summary() {
  local output_dir="$1"
  poetry run "${PYTHON_BIN}" - "${output_dir}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

output_dir = Path(sys.argv[1])
raw_path = output_dir / "predictions.tsv"
aggregated_path = output_dir / "predictions_aggregated.tsv"
aggregated_all_path = output_dir / "predictions_aggregated_all.tsv"
report_path = output_dir / "step05_report.json"

print("")
print("========== Step 5 Report ==========")
print(f"output_dir: {output_dir}")
print(f"raw_predictions: {raw_path}")
print(f"aggregated_by_experiment: {aggregated_path}")
print(f"aggregated_all: {aggregated_all_path}")
print(f"report_json: {report_path}")

if not report_path.exists():
    print("report_status: missing step05_report.json")
    sys.exit(0)

report = json.loads(report_path.read_text(encoding="utf-8"))
distribution = report.get("score_distribution", {})

print("")
print("Counts")
print(f"  raw_subgraph_rows: {report.get('n_raw_rows', 0)}")
print(f"  aggregated_clean_rows: {report.get('n_aggregated_clean', 0)}")
print(f"  aggregated_extended_rows: {report.get('n_aggregated_extended', 0)}")
print(f"  aggregated_mixed_rows: {report.get('n_aggregated_mixed', 0)}")
print(f"  aggregated_all_rows: {report.get('n_aggregated_all_rows', 0)}")

print("")
print("Score distribution")
print(f"  min: {distribution.get('min', 0.0):.6f}")
print(f"  mean: {distribution.get('mean', 0.0):.6f}")
print(f"  median: {distribution.get('median', 0.0):.6f}")
print(f"  max: {distribution.get('max', 0.0):.6f}")
print(f"  std: {distribution.get('std', 0.0):.6f}")
print(f"  all_zero: {distribution.get('all_zero', False)}")
print(f"  all_one: {distribution.get('all_one', False)}")

if raw_path.exists():
    raw = pd.read_csv(raw_path, sep="\t")
    if not raw.empty and "Score_Sigmoid" in raw.columns:
        label_counts = raw["Label"].value_counts(dropna=False).to_dict() if "Label" in raw.columns else {}
        experiment_counts = raw["ExperimentType"].value_counts(dropna=False).to_dict() if "ExperimentType" in raw.columns else {}
        print("")
        print("Raw row breakdown")
        print(f"  by_label: {label_counts}")
        print(f"  by_experiment_type: {experiment_counts}")
        print(f"  unique_pairs: {raw[['DrugTarget_UniProt', 'DiseaseProtein_UniProt']].drop_duplicates().shape[0] if {'DrugTarget_UniProt', 'DiseaseProtein_UniProt'}.issubset(raw.columns) else 'n/a'}")
        print(f"  unique_pathways: {raw['PathwayID'].nunique() if 'PathwayID' in raw.columns else 'n/a'}")

        if {"Score_FoldStd", "Score_FoldRange"}.issubset(raw.columns):
            print("")
            print("Fold uncertainty")
            print(f"  fold_std_mean: {raw['Score_FoldStd'].mean():.6f}")
            print(f"  fold_std_median: {raw['Score_FoldStd'].median():.6f}")
            print(f"  fold_std_p95: {raw['Score_FoldStd'].quantile(0.95):.6f}")
            print(f"  fold_range_mean: {raw['Score_FoldRange'].mean():.6f}")
            print(f"  fold_range_p95: {raw['Score_FoldRange'].quantile(0.95):.6f}")
            print(f"  fold_range_ge_0.9_rows: {(raw['Score_FoldRange'] >= 0.9).sum()}")

        display_columns = [
            "DrugName",
            "DiseaseName",
            "DrugTarget_UniProt",
            "DiseaseProtein_UniProt",
            "PathwayID",
            "ExperimentType",
            "Score_Sigmoid",
            "Score_FoldStd",
            "Score_FoldRange",
            "Score_PathwayNormalized",
        ]
        display_columns = [column for column in display_columns if column in raw.columns]

        print("")
        print("Top 5 raw subgraph scores")
        print(raw.sort_values("Score_Sigmoid", ascending=False)[display_columns].head(5).to_string(index=False))

        print("")
        print("Bottom 5 raw subgraph scores")
        print(raw.sort_values("Score_Sigmoid", ascending=True)[display_columns].head(5).to_string(index=False))

if aggregated_path.exists():
    aggregated = pd.read_csv(aggregated_path, sep="\t")
    if not aggregated.empty and "MeanScore" in aggregated.columns:
        print("")
        print("Top 5 aggregated pair scores by experiment")
        display_columns = [
            "DrugName",
            "DiseaseName",
            "DrugTarget_UniProt",
            "DiseaseProtein_UniProt",
            "ExperimentType",
            "NumPathways",
            "MeanScore",
            "MaxScore",
            "MeanPathwayNormalizedScore",
            "MeanScoreFoldStd",
            "MaxScoreFoldRange",
        ]
        display_columns = [column for column in display_columns if column in aggregated.columns]
        print(aggregated.sort_values("MeanScore", ascending=False)[display_columns].head(5).to_string(index=False))

        if "ExperimentType" in aggregated.columns:
            clean = aggregated[aggregated["ExperimentType"] == "clean"]
            extended = aggregated[aggregated["ExperimentType"] == "extended"]
            pair_cols = ["DrugTarget_UniProt", "DiseaseProtein_UniProt"]
            if not clean.empty and not extended.empty and set(pair_cols).issubset(aggregated.columns):
                shared = clean[pair_cols + ["MeanScore"]].merge(
                    extended[pair_cols + ["MeanScore"]],
                    on=pair_cols,
                    suffixes=("_clean", "_extended"),
                )
                if not shared.empty:
                    shared["DeltaExtendedMinusClean"] = shared["MeanScore_extended"] - shared["MeanScore_clean"]
                    print("")
                    print("Clean vs extended on shared pairs")
                    print(f"  shared_pairs: {len(shared)}")
                    print(f"  mean_delta_extended_minus_clean: {shared['DeltaExtendedMinusClean'].mean():.6f}")
                    print(shared.sort_values("DeltaExtendedMinusClean", ascending=False).head(5).to_string(index=False))

if aggregated_all_path.exists():
    aggregated_all = pd.read_csv(aggregated_all_path, sep="\t")
    if not aggregated_all.empty and "MeanScore" in aggregated_all.columns:
        print("")
        print("Top 5 aggregated pair scores across all pathways")
        display_columns = [
            "DrugName",
            "DiseaseName",
            "DrugTarget_UniProt",
            "DiseaseProtein_UniProt",
            "ExperimentType",
            "NumPathways",
            "MeanScore",
            "MaxScore",
            "MeanPathwayNormalizedScore",
            "MeanScoreFoldStd",
            "MaxScoreFoldRange",
        ]
        display_columns = [column for column in display_columns if column in aggregated_all.columns]
        print(aggregated_all.sort_values("MeanScore", ascending=False)[display_columns].head(5).to_string(index=False))

print("===================================")
PY
}

log_line "[${RUN_ID}] starting Step 5 run"
log_line "[${RUN_ID}] logs: ${LOG_FILE}"
log_line "[${RUN_ID}] metadata: ${META_FILE}"

set +e
MPLCONFIGDIR="${MPLCONFIGDIR:-${RUN_DIR}/cache/matplotlib}" \
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUN_DIR}/cache/xdg}" \
PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}" \
  "${COMMAND[@]}" 2>&1 | tee -a "${LOG_FILE}"
COMMAND_EXIT_CODE=${PIPESTATUS[0]}
set -e

if [[ "${COMMAND_EXIT_CODE}" -eq 0 ]]; then
  print_step05_summary "${EFFECTIVE_OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"
else
  log_line ""
  log_line "========== Step 5 Report =========="
  log_line "Skipped report because Step 5 exited with code ${COMMAND_EXIT_CODE}."
  log_line "==================================="
fi

{
  echo "finished_at=$(date +"%Y-%m-%d %H:%M:%S %Z")"
  echo "exit_code=${COMMAND_EXIT_CODE}"
} >> "${META_FILE}"

log_line "[${RUN_ID}] completed with exit code ${COMMAND_EXIT_CODE}"
exit "${COMMAND_EXIT_CODE}"
