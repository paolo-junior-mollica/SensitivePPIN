# Digital Health Lab Sensitive PPIN

This repository contains a reproducible research pipeline for building drug-disease validation data and running sensitivity prediction on protein-protein interaction networks (PPINs) with Deep Graph Networks (DGN).

The project combines biomedical data integration scripts, PPIN subgraph construction, and an adapted DGN inference module for sensitivity analysis on protein-protein interaction networks.

## Project Overview

The codebase is organized around two connected workflows:

- `drug_disease_validation/`: builds drug-disease protein pairs, resolves protein identifiers, extracts pathway-aware PPIN subgraphs, shards large subgraph collections, runs DGN predictions, and searches for negative pathways.
- `api_clients/`: provides clients and data-processing utilities for external biomedical sources such as UniProt, ChEMBL, OpenTargets, Reactome, CT.gov, Protein Atlas, STRING, repoDB, and BioGRID-related downloads.
- `script/`: contains runnable step scripts and logging wrappers for the data preparation and prediction pipeline.

The main validation workflow prepares drug-target and disease-protein pairs, maps identifiers to UniProt, builds BioGRID/NetworkX subgraphs, converts them to PyTorch Geometric data, and scores each subgraph with the DGN checkpoints.

## Repository Layout

```text
.
├── api_clients/                         # External biomedical API and download helpers
├── drug_disease_validation/
│   ├── config/                          # YAML configuration files for selected steps
│   ├── model_artifacts/                 # Small preserved DGN model metadata/artifacts
│   └── src/                             # Pipeline source code and DGN inference module
├── script/                              # Executable step wrappers and logging scripts
├── pyproject.toml                       # Poetry environment definition
└── poetry.lock                          # Locked dependency resolution
```

## Requirements

Use Poetry from the repository root. The supported Python range is:

```text
Python >=3.10,<3.12
```

Python 3.12 and newer are not currently supported by the pinned scientific and graph-learning stack in this repository.

Install the environment with:

```bash
poetry env use 3.10
poetry install
```

Python 3.11 is also allowed by the project metadata:

```bash
poetry env use 3.11
poetry install
```

## Verified Dependency Versions

The versions below were checked against `pyproject.toml` and `poetry.lock`.

| Dependency | Version |
| --- | --- |
| Python | `>=3.10,<3.12` |
| lightning | `2.5.0.post0` |
| matplotlib | `3.7.5` |
| networkx | `3.1` |
| numpy | `1.26.4` |
| pandas | `2.0.2` |
| h5py | `3.16.0` |
| ray | `2.42.0` |
| scikit-learn | `1.2.2` |
| scipy | `>=1.10,<2.0` |
| seaborn | `0.13.2` |
| torchmetrics | `1.6.1` |
| torch-geometric | `2.7.0` |
| tqdm | `4.67.1` |
| wandb | `0.19.5` |

PyTorch is platform-specific:

| Platform | PyTorch version |
| --- | --- |
| macOS / Darwin | `2.9.1` |
| Linux | `2.2.0+cu121` from the `pytorch-cu121` source |
| Other non-Linux, non-macOS platforms | `2.2.0` |

The canonical environment for this repository is the root `pyproject.toml`.

## Data Availability

Large raw datasets and generated subgraph collections are intentionally not versioned. They are excluded because of file size, generated volume, and source-specific licensing or access terms.

Expected raw inputs should be downloaded from their official sources and placed under `drug_disease_validation/data/raw/` or passed explicitly through the relevant script arguments. The pipeline expects data from sources such as:

- BioGRID human protein-protein interactions
- CTD chemical-disease associations
- DisGeNET curated gene-disease associations
- repoDB positive drug-disease pairs
- DrugBank drug-target relationships
- UniProt protein embeddings used during subgraph construction and prediction

Generated subgraph directories and intermediate files under `drug_disease_validation/data/processed*/` are also not versioned. Local DGN checkpoints and embedding files should be kept under `drug_disease_validation/model_data/`, which is excluded from version control.

This public repository intentionally excludes local-only material such as temporary runs, private notes, tests, and the original imported Di Palma project directory. The DGN code required by Step 5 is kept inside `drug_disease_validation/src/dgn_model/`; reference prediction scripts are preserved in `drug_disease_validation/src/dgn_prediction_reference/`. Small Di Palma model metadata and the preserved `predictions.pt` artifact are kept in `drug_disease_validation/model_artifacts/dipalma/`. External checkpoints and embeddings should be supplied locally through `drug_disease_validation/model_data/` or explicit command-line paths.

## Running the Pipeline

### Step 1: Fetch and Prepare Source Data

```bash
poetry run python script/step01_fetch_data.py
```

This step normalizes drug-disease pairs, prepares drug-target associations, maps disease proteins to UniProt, filters human physical BioGRID interactions, and writes the initial network and report outputs.

Logging wrapper:

```bash
script/run_step01_with_logs.sh
```

### Step 2: Build Drug-Disease Protein Pairs

```bash
script/run_step02_with_logs.sh
```

### Step 3: Find Pathways

```bash
script/run_step03_with_logs.sh
```

### Step 4: Build PPIN Subgraphs

```bash
script/run_step04_with_logs.sh
```

Subgraph sharding helper:

```bash
script/run_step04_1_shard_subgraphs_with_logs.sh
```

### Step 5: Run DGN Predictions

```bash
poetry run python drug_disease_validation/src/05_predict.py
```

Or use the logging wrapper:

```bash
script/run_step05_with_logs.sh
```

By default, the wrapper uses:

- manifest: `drug_disease_validation/data/processed_step04_external/subgraph_manifest_io+emb1024.tsv`
- limit: `100`
- accelerator: `cpu`

Override these defaults with environment variables:

```bash
STEP05_LIMIT=10 STEP05_ACCELERATOR=cpu script/run_step05_with_logs.sh
```

Run all manifest rows:

```bash
STEP05_LIMIT=all script/run_step05_with_logs.sh
```

Step 5 automatically looks for DGN checkpoint files under:

```text
drug_disease_validation/model_data/Sensitivity Prediction on Protein Protein Interaction Networks/prediction_data/ckpts
```

It expects fold directories such as `0/`, `1/`, `2/`, and `3/`, each containing `params.pkl` and `last.ckpt`.

Main Step 5 outputs are written under `drug_disease_validation/data/processed/`:

- `predictions.tsv`: per-subgraph scores and fold-level uncertainty metrics
- `predictions_aggregated.tsv`: aggregated scores by drug target, disease protein, and experiment type
- `predictions_aggregated_all.tsv`: pair-level aggregation across pathways
- `step05_report.json`: score distribution, counts, and sanity checks

### Step 6: Find Negative Pathways

```bash
script/run_step06_with_logs.sh
```

## Runtime Notes

- On macOS, the runtime detects Metal Performance Shaders (`mps`) when available.
- On CUDA-enabled Linux systems, the project uses the Linux PyTorch build pinned as `2.2.0+cu121`.
- Ray temporary and result paths are configured to use local writable directories instead of absolute machine-specific paths.
- The module `api_clients/reactome.py` imports `ppi_dataset.io_utils`, which is not included in this repository. That external dependency is therefore not part of the Poetry environment.

## License

This repository is distributed under the terms of the license included in `LICENSE`.
