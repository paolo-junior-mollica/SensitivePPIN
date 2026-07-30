from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from drug_disease_validation.src.subgraph_sharding import SHARD_FORMAT
from drug_disease_validation.src.schemas import CANONICAL_FILENAMES
from drug_disease_validation.src.utils import configure_logging, ensure_parent_dir, read_pickle


logger = logging.getLogger(__name__)

DEFAULT_STEP05_REPORT = "step05_report.json"
DEFAULT_STEP05_SUMMARY = "step05_report.md"
DEFAULT_DGN_MODEL_DATA_DIRNAME = "Sensitivity Prediction on Protein Protein Interaction Networks"
REQUIRED_MANIFEST_COLUMNS = [
    "SubgraphID",
    "DrugTarget_UniProt",
    "DiseaseProtein_UniProt",
    "PathwayID",
    "ExperimentType",
    "SubgraphPath",
]


def _resolve_existing_manifest(processed_dir: Path) -> Path:
    candidates = [
        processed_dir / "subgraph_manifest_io_sharded.tsv",
        processed_dir / "subgraph_manifest_io.tsv",
        processed_dir / "subgraph_manifest_sharded.tsv",
        processed_dir / CANONICAL_FILENAMES["subgraph_manifest"],
        processed_dir / "subgraph_manifest_io+emb1024_sharded.tsv",
        processed_dir / "subgraph_manifest_io+emb1024.tsv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_default_checkpoint_dir(model_data_root: Path) -> Path:
    candidates = [
        model_data_root / DEFAULT_DGN_MODEL_DATA_DIRNAME / "prediction_data" / "ckpts",
        model_data_root / "prediction_data" / "ckpts",
        model_data_root / "prediction_data" / "ckpts" / "io",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_manifest_subgraph_path(raw_path: Any, manifest_path: Path) -> Path:
    text = str(raw_path).strip()
    if not text or text.lower() == "nan":
        raise ValueError("Manifest row has an empty SubgraphPath.")

    path = Path(text)
    manifest_dir = manifest_path.parent
    if path.is_absolute():
        candidates = []
        if path.parent.name and path.name:
            candidates.append(manifest_dir / path.parent.name / path.name)
            candidates.append(manifest_dir / path.name)
        candidates.append(path)
    else:
        candidates = [manifest_dir / path, path]
        if path.parent.name and path.name:
            candidates.append(manifest_dir / path.parent.name / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _has_value(value: Any) -> bool:
    text = str(value).strip()
    return bool(text and text.lower() != "nan")


def _coerce_label(value: Any) -> int:
    if value is None:
        return 1
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return 1
    return int(float(text))


def load_manifest(manifest_path: Path, *, limit: int | None = None, dataset_split: str | None = None) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, sep="\t", low_memory=False)
    missing_columns = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in manifest.columns]
    if missing_columns:
        raise ValueError(f"Manifest {manifest_path} is missing required columns: {missing_columns}")

    manifest = manifest.copy()
    manifest["SubgraphPathResolved"] = manifest["SubgraphPath"].apply(
        lambda value: str(_resolve_manifest_subgraph_path(value, manifest_path))
    )
    if "SubgraphShardPath" in manifest.columns:
        manifest["SubgraphShardPathResolved"] = manifest["SubgraphShardPath"].apply(
            lambda value: str(_resolve_manifest_subgraph_path(value, manifest_path)) if _has_value(value) else ""
        )
        if "SubgraphShardKey" not in manifest.columns:
            manifest["SubgraphShardKey"] = manifest["SubgraphID"].astype(str)
    if "SensitivityLabel" in manifest.columns and "Label" not in manifest.columns:
        manifest["Label"] = manifest["SensitivityLabel"].apply(_coerce_label)
    elif "Label" not in manifest.columns:
        manifest["Label"] = 1
    if dataset_split is not None:
        manifest["DatasetSplit"] = dataset_split

    if limit is not None:
        manifest = manifest.head(int(limit)).copy()
    return manifest.reset_index(drop=True)


def load_prediction_manifests(
    *,
    manifest_path: Path,
    negative_manifest_path: Path | None = None,
    limit: int | None = None,
    negative_limit: int | None = None,
) -> pd.DataFrame:
    positive = load_manifest(manifest_path, limit=limit, dataset_split="positive")
    if negative_manifest_path is None:
        return positive

    resolved_negative_limit = limit if negative_limit is None else negative_limit
    negative = load_manifest(negative_manifest_path, limit=resolved_negative_limit, dataset_split="negative")
    combined = pd.concat([positive, negative], ignore_index=True)
    if combined["SubgraphID"].duplicated().any():
        combined["SubgraphID"] = [
            f"{row.DatasetSplit}:{row.SubgraphID}"
            for row in combined[["DatasetSplit", "SubgraphID"]].itertuples(index=False)
        ]
    return combined.reset_index(drop=True)


def _load_graph_from_shard(shard_path: Path, shard_key: str, shard_cache: dict[Path, dict[str, Any]]) -> nx.Graph:
    if shard_path not in shard_cache:
        payload = read_pickle(shard_path)
        if isinstance(payload, dict) and "graphs" in payload:
            payload_format = payload.get("format")
            if payload_format not in {SHARD_FORMAT, None}:
                raise ValueError(f"Unsupported subgraph shard format in {shard_path}: {payload_format}")
            graphs = payload["graphs"]
        elif isinstance(payload, dict):
            graphs = payload
        else:
            raise ValueError(f"Expected shard pickle {shard_path} to contain a dict payload.")
        if not isinstance(graphs, dict):
            raise ValueError(f"Expected shard pickle {shard_path} to contain a graph mapping.")
        shard_cache[shard_path] = graphs
    try:
        graph = shard_cache[shard_path][shard_key]
    except KeyError as exc:
        raise KeyError(f"Shard key {shard_key} not found in {shard_path}") from exc
    return graph


def load_subgraph_from_manifest_row(row: Any, *, shard_cache: dict[Path, dict[str, Any]] | None = None) -> nx.Graph:
    row_dict = row._asdict() if hasattr(row, "_asdict") else dict(row)
    if _has_value(row_dict.get("SubgraphShardPathResolved", "")):
        shard_path = Path(str(row_dict["SubgraphShardPathResolved"]))
        raw_shard_key = row_dict.get("SubgraphShardKey")
        shard_key = str(raw_shard_key if _has_value(raw_shard_key) else row_dict.get("SubgraphID"))
        return _load_graph_from_shard(shard_path, shard_key, shard_cache if shard_cache is not None else {})
    return read_pickle(row_dict["SubgraphPathResolved"])


def _resolve_feature_vector(
    *,
    node: str,
    drug_target: str,
    disease_protein: str,
    payload: dict[str, Any],
    embeddings: dict[str, np.ndarray] | None,
    embedding_dim: int | None,
) -> list[float]:
    io_features = [
        1.0 if node == drug_target else 0.0,
        1.0 if node == disease_protein else 0.0,
    ]
    if embedding_dim is None:
        raw_features = payload.get("features")
        if raw_features is None:
            return io_features
        return [float(value) for value in raw_features]
    if embedding_dim == 0:
        return io_features

    vector = embeddings.get(node) if embeddings is not None else None
    if vector is None:
        embedding = np.zeros(embedding_dim, dtype=float)
    else:
        embedding = np.asarray(vector, dtype=float).reshape(-1)
        if int(embedding.shape[0]) != int(embedding_dim):
            raise ValueError(f"Embedding for {node} has dimension {embedding.shape[0]}, expected {embedding_dim}.")
    return io_features + [float(value) for value in embedding]


def graph_to_pyg_data(
    graph: nx.Graph,
    *,
    drug_target: str,
    disease_protein: str,
    subgraph_id: str,
    embeddings: dict[str, np.ndarray] | None = None,
    embedding_dim: int | None = None,
) -> Any:
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise RuntimeError("Step 5 requires torch and torch-geometric to convert subgraphs for DGN inference.") from exc

    node_order = graph.graph.get("node_order")
    if not node_order:
        node_order = sorted(str(node) for node in graph.nodes())
    node_order = [str(node) for node in node_order if str(node) in graph]
    if drug_target not in node_order:
        raise ValueError(f"Drug target {drug_target} missing from subgraph {subgraph_id}.")
    if disease_protein not in node_order:
        raise ValueError(f"Disease protein {disease_protein} missing from subgraph {subgraph_id}.")

    node_indices = {node: index for index, node in enumerate(node_order)}
    features: list[list[float]] = []
    for node in node_order:
        payload = graph.nodes[node]
        features.append(
            _resolve_feature_vector(
                node=node,
                drug_target=drug_target,
                disease_protein=disease_protein,
                payload=payload,
                embeddings=embeddings,
                embedding_dim=embedding_dim,
            )
        )

    edge_pairs = [
        (node_indices[str(source)], node_indices[str(target)])
        for source, target in graph.edges()
        if str(source) in node_indices and str(target) in node_indices
    ]
    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=edge_index,
        input_species=drug_target,
        output_species=disease_protein,
        subgraph_id=subgraph_id,
    )


def manifest_to_datalist(
    manifest: pd.DataFrame,
    *,
    embeddings: dict[str, np.ndarray] | None = None,
    embedding_dim: int | None = None,
) -> list[Any]:
    datalist = []
    shard_cache: dict[Path, dict[str, Any]] = {}
    for row in manifest.itertuples(index=False):
        graph = load_subgraph_from_manifest_row(row, shard_cache=shard_cache)
        datalist.append(
            graph_to_pyg_data(
                graph,
                drug_target=str(getattr(row, "DrugTarget_UniProt")),
                disease_protein=str(getattr(row, "DiseaseProtein_UniProt")),
                subgraph_id=str(getattr(row, "SubgraphID")),
                embeddings=embeddings,
                embedding_dim=embedding_dim,
            )
        )
    return datalist


def _load_fold_configs(checkpoint_dir: Path, *, batch_size: int) -> dict[int, dict[str, Any]]:
    folds: dict[int, dict[str, Any]] = {}
    for fold in range(4):
        fold_dir = checkpoint_dir / str(fold)
        params_path = fold_dir / "params.pkl"
        checkpoint_path = fold_dir / "last.ckpt"
        if not params_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing DGN checkpoint files for fold {fold}: expected {params_path} and {checkpoint_path}"
            )
        with params_path.open("rb") as handle:
            config = pickle.load(handle)
        config["batch_size"] = int(batch_size)
        config["workers"] = 0
        folds[fold] = {"config": config, "ckpt": checkpoint_path}
    return folds


def resolve_checkpoint_embedding_dim(checkpoint_dir: Path) -> int:
    fold_configs = _load_fold_configs(checkpoint_dir, batch_size=1)
    embedding_dims = {int(payload["config"].get("embeddings_len", 0) or 0) for payload in fold_configs.values()}
    if len(embedding_dims) != 1:
        raise ValueError(f"Checkpoint folds disagree on embeddings_len: {sorted(embedding_dims)}")
    return embedding_dims.pop()


def load_embedding_dict(path: Path, *, expected_dim: int) -> dict[str, np.ndarray]:
    raw_embeddings = read_pickle(path)
    if not isinstance(raw_embeddings, dict):
        raise ValueError(f"Expected embedding file {path} to contain a dict.")
    embeddings: dict[str, np.ndarray] = {}
    for key, value in raw_embeddings.items():
        vector = np.asarray(value, dtype=float).reshape(-1)
        if int(vector.shape[0]) != int(expected_dim):
            raise ValueError(f"Embedding file {path} contains {key} with dimension {vector.shape[0]}, expected {expected_dim}.")
        embeddings[str(key)] = vector
    return embeddings


def resolve_embedding_file(
    args: argparse.Namespace,
    *,
    model_data_root: Path,
    checkpoint_dir: Path,
    embedding_dim: int,
) -> Path | None:
    if embedding_dim == 0:
        return None
    configured = getattr(args, "embedding_file", None)
    if configured:
        return Path(configured)
    if embedding_dim == 128:
        candidates = [
            checkpoint_dir.parent / "uniprot_embeddings_pca_128.pkl",
            model_data_root / DEFAULT_DGN_MODEL_DATA_DIRNAME / "prediction_data" / "uniprot_embeddings_pca_128.pkl",
            model_data_root / "prediction_data" / "uniprot_embeddings_pca_128.pkl",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
    raise ValueError(
        f"Checkpoint expects embeddings_len={embedding_dim}, but no --embedding-file was provided."
    )


def predict_with_dgn_checkpoints(
    datalist: Sequence[Any],
    *,
    checkpoint_dir: Path,
    batch_size: int,
    accelerator: str = "cpu",
) -> pd.DataFrame:
    if not datalist:
        return pd.DataFrame(columns=["Score_Sigmoid"])

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Step 5 requires torch for DGN inference.") from exc

    from lightning import Trainer
    from drug_disease_validation.src.dgn_model.datamodule import GraphDataModule
    from drug_disease_validation.src.dgn_model.dgn.dgn import DGN

    if accelerator == "auto":
        try:
            from lightning.pytorch.accelerators import MPSAccelerator

            resolved_accelerator = "mps" if MPSAccelerator.is_available() else "cpu"
        except Exception:
            resolved_accelerator = "cpu"
    elif accelerator == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS was requested but torch reports it unavailable; falling back to CPU.")
        resolved_accelerator = "cpu"
    else:
        resolved_accelerator = accelerator

    fold_configs = _load_fold_configs(checkpoint_dir, batch_size=batch_size)
    fold_predictions = []
    for fold, payload in fold_configs.items():
        logger.info("Running DGN prediction fold %s on %s subgraphs with accelerator=%s", fold, len(datalist), resolved_accelerator)
        config = payload["config"]
        if "aggr" not in config:
            config["aggr"] = config["SAGE_aggr"]
        config.setdefault("uniform_bound", None)
        config.setdefault("weight_initializer", "kaiming_uniform")
        config["weighted_sampler"] = None

        input_dim = datalist[0].x.shape[1]
        model_device = torch.device("mps" if resolved_accelerator == "mps" else "cpu")
        data = GraphDataModule(datalist, datalist, datalist, config)
        data.setup()
        model = DGN.load_from_checkpoint(
            payload["ckpt"],
            input_dim=input_dim,
            output_dim=1,
            config=config,
            map_location=model_device,
            strict=False,
        )
        trainer = Trainer(
            accelerator=resolved_accelerator,
            devices=1,
            enable_progress_bar=False,
            logger=False,
        )
        fold_output = torch.cat(trainer.predict(model, data.test_dataloader()))
        fold_predictions.append(fold_output.detach().cpu())

    predictions = torch.sigmoid(torch.stack(fold_predictions))
    predictions_mean = torch.mean(predictions, dim=0).detach().cpu().numpy()
    predictions_std = torch.std(predictions, dim=0, unbiased=False).detach().cpu().numpy()
    predictions_range = (torch.max(predictions, dim=0).values - torch.min(predictions, dim=0).values).detach().cpu().numpy()
    output = {"Score_Sigmoid": [float(value) for value in predictions_mean]}
    for fold in range(predictions.shape[0]):
        output[f"Score_Fold{fold}"] = [float(value) for value in predictions[fold].detach().cpu().numpy()]
    output["Score_FoldStd"] = [float(value) for value in predictions_std]
    output["Score_FoldRange"] = [float(value) for value in predictions_range]
    return pd.DataFrame(output)


def _iter_manifest_chunks(manifest: pd.DataFrame, chunk_size: int) -> Sequence[pd.DataFrame]:
    if chunk_size <= 0 or len(manifest) <= chunk_size:
        return [manifest]
    return [manifest.iloc[start : start + chunk_size].copy() for start in range(0, len(manifest), chunk_size)]


def _resolve_prediction_accelerator(accelerator: str) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Step 5 requires torch for DGN inference.") from exc

    if accelerator == "auto":
        try:
            from lightning.pytorch.accelerators import MPSAccelerator

            return "mps" if MPSAccelerator.is_available() else "cpu"
        except Exception:
            return "cpu"
    if accelerator == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS was requested but torch reports it unavailable; falling back to CPU.")
        return "cpu"
    return accelerator


def _prediction_frame_from_fold_outputs(fold_predictions: Sequence[Any]) -> pd.DataFrame:
    import torch

    predictions = torch.sigmoid(torch.stack(list(fold_predictions)))
    predictions_mean = torch.mean(predictions, dim=0).detach().cpu().numpy()
    predictions_std = torch.std(predictions, dim=0, unbiased=False).detach().cpu().numpy()
    predictions_range = (torch.max(predictions, dim=0).values - torch.min(predictions, dim=0).values).detach().cpu().numpy()
    output = {"Score_Sigmoid": [float(value) for value in predictions_mean]}
    for fold in range(predictions.shape[0]):
        output[f"Score_Fold{fold}"] = [float(value) for value in predictions[fold].detach().cpu().numpy()]
    output["Score_FoldStd"] = [float(value) for value in predictions_std]
    output["Score_FoldRange"] = [float(value) for value in predictions_range]
    return pd.DataFrame(output)


def predict_manifest_with_dgn_checkpoints(
    manifest: pd.DataFrame,
    *,
    checkpoint_dir: Path,
    batch_size: int,
    accelerator: str = "cpu",
    embeddings: dict[str, np.ndarray] | None = None,
    embedding_dim: int | None = None,
    prediction_chunk_size: int = 0,
) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame(columns=["Score_Sigmoid"])

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Step 5 requires torch for DGN inference.") from exc

    from lightning import Trainer
    from drug_disease_validation.src.dgn_model.datamodule import GraphDataModule
    from drug_disease_validation.src.dgn_model.dgn.dgn import DGN

    resolved_accelerator = _resolve_prediction_accelerator(accelerator)
    fold_configs = _load_fold_configs(checkpoint_dir, batch_size=batch_size)
    chunks = _iter_manifest_chunks(manifest, int(prediction_chunk_size or 0))
    first_datalist = manifest_to_datalist(chunks[0].head(1), embeddings=embeddings, embedding_dim=embedding_dim)
    input_dim = first_datalist[0].x.shape[1]
    del first_datalist
    gc.collect()

    fold_predictions = []
    for fold, payload in fold_configs.items():
        logger.info(
            "Running DGN prediction fold %s on %s subgraphs with accelerator=%s chunk_size=%s",
            fold,
            len(manifest),
            resolved_accelerator,
            prediction_chunk_size if prediction_chunk_size and prediction_chunk_size > 0 else "all",
        )
        config = payload["config"]
        if "aggr" not in config:
            config["aggr"] = config["SAGE_aggr"]
        config.setdefault("uniform_bound", None)
        config.setdefault("weight_initializer", "kaiming_uniform")
        config["weighted_sampler"] = None

        model_device = torch.device("mps" if resolved_accelerator == "mps" else "cpu")
        model = DGN.load_from_checkpoint(
            payload["ckpt"],
            input_dim=input_dim,
            output_dim=1,
            config=config,
            map_location=model_device,
            strict=False,
        )
        trainer = Trainer(
            accelerator=resolved_accelerator,
            devices=1,
            enable_progress_bar=False,
            logger=False,
        )

        fold_chunk_outputs = []
        for chunk_index, chunk in enumerate(chunks, start=1):
            logger.info("Predicting fold %s chunk %s/%s rows=%s", fold, chunk_index, len(chunks), len(chunk))
            datalist = manifest_to_datalist(chunk, embeddings=embeddings, embedding_dim=embedding_dim)
            data = GraphDataModule(datalist, datalist, datalist, config)
            data.setup()
            fold_chunk_outputs.append(torch.cat(trainer.predict(model, data.test_dataloader())).detach().cpu())
            del datalist, data
            gc.collect()

        fold_predictions.append(torch.cat(fold_chunk_outputs))
        del model, trainer, fold_chunk_outputs
        gc.collect()

    return _prediction_frame_from_fold_outputs(fold_predictions)


def predict_manifest_with_callable(
    manifest: pd.DataFrame,
    *,
    predictor: Callable[[Sequence[Any]], pd.DataFrame],
    embeddings: dict[str, np.ndarray] | None = None,
    embedding_dim: int | None = None,
    prediction_chunk_size: int = 0,
) -> pd.DataFrame:
    outputs = []
    chunks = _iter_manifest_chunks(manifest, int(prediction_chunk_size or 0))
    for chunk_index, chunk in enumerate(chunks, start=1):
        logger.info("Predicting callable chunk %s/%s rows=%s", chunk_index, len(chunks), len(chunk))
        datalist = manifest_to_datalist(chunk, embeddings=embeddings, embedding_dim=embedding_dim)
        outputs.append(predictor(datalist))
        del datalist
        gc.collect()
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(columns=["Score_Sigmoid"])


def add_pathway_normalized_scores(raw_predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = raw_predictions.copy()
    pathway_max = predictions.groupby("PathwayID")["Score_Sigmoid"].transform("max")
    predictions["Score_PathwayNormalized"] = predictions["Score_Sigmoid"] / pathway_max.replace(0, pd.NA)
    predictions["Score_PathwayNormalized"] = predictions["Score_PathwayNormalized"].fillna(0.0).astype(float)
    return predictions


def _first_non_empty(values: pd.Series) -> str:
    for value in values:
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def _count_dict(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).to_dict().items()}


def _numeric_summary(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0, "std": 0.0}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "max": float(values.max()),
        "std": float(values.std(ddof=0)),
    }


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = ranks[labels == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    true_positive_cumsum = np.cumsum(sorted_labels)
    precision = true_positive_cumsum / (np.arange(len(sorted_labels)) + 1)
    return float((precision * sorted_labels).sum() / positives)


def _threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = (scores >= threshold).astype(int)
    tp = int(((predicted == 1) & (labels == 1)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def build_binary_classification_metrics(predictions: pd.DataFrame, *, score_column: str) -> dict[str, Any]:
    if predictions.empty or "Label" not in predictions.columns or score_column not in predictions.columns:
        return {"available": False, "reason": "missing Label or score column"}
    frame = predictions[["Label", score_column]].copy()
    frame["Label"] = pd.to_numeric(frame["Label"], errors="coerce")
    frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
    frame = frame.dropna()
    frame = frame[frame["Label"].isin([0, 1])]
    labels = frame["Label"].astype(int).to_numpy()
    scores = frame[score_column].astype(float).to_numpy()
    label_set = set(int(value) for value in labels)
    if label_set != {0, 1}:
        return {
            "available": False,
            "reason": "requires both positive and negative labels",
            "labels_present": sorted(label_set),
            "n_rows": int(len(frame)),
        }

    threshold_metrics = _threshold_metrics(labels, scores, 0.5)
    candidate_thresholds = np.unique(scores)
    best = threshold_metrics
    for threshold in candidate_thresholds:
        metrics = _threshold_metrics(labels, scores, float(threshold))
        if (metrics["f1"], metrics["accuracy"]) > (best["f1"], best["accuracy"]):
            best = metrics

    return {
        "available": True,
        "score_column": score_column,
        "n_rows": int(len(frame)),
        "label_counts": {str(key): int(value) for key, value in pd.Series(labels).value_counts().to_dict().items()},
        "roc_auc": _roc_auc(labels, scores),
        "average_precision": _average_precision(labels, scores),
        "threshold_0_5": threshold_metrics,
        "best_f1_threshold": best,
    }


def _aggregate_group(group: pd.DataFrame, *, experiment_type: str | None = None) -> pd.Series:
    if experiment_type is None:
        experiment_types = sorted(str(value).strip().lower() for value in group["ExperimentType"].dropna().unique())
        if experiment_types == ["clean"]:
            resolved_experiment_type = "clean"
        elif experiment_types == ["extended"]:
            resolved_experiment_type = "extended"
        else:
            resolved_experiment_type = "mixed"
    else:
        resolved_experiment_type = experiment_type

    row = {
        "DrugName": _first_non_empty(group.get("DrugName", pd.Series(dtype=str))),
        "DiseaseName": _first_non_empty(group.get("DiseaseName", pd.Series(dtype=str))),
        "DrugTarget_UniProt": _first_non_empty(group["DrugTarget_UniProt"]),
        "DiseaseProtein_UniProt": _first_non_empty(group["DiseaseProtein_UniProt"]),
        "NumPathways": int(group["PathwayID"].nunique()),
        "MeanScore": float(group["Score_Sigmoid"].mean()),
        "MaxScore": float(group["Score_Sigmoid"].max()),
        "MeanPathwayNormalizedScore": float(group["Score_PathwayNormalized"].mean()),
        "MaxPathwayNormalizedScore": float(group["Score_PathwayNormalized"].max()),
        "Label": int(group["Label"].max()),
        "ExperimentType": resolved_experiment_type,
    }
    if "Score_FoldStd" in group.columns:
        row["MeanScoreFoldStd"] = float(group["Score_FoldStd"].mean())
        row["MaxScoreFoldStd"] = float(group["Score_FoldStd"].max())
    if "Score_FoldRange" in group.columns:
        row["MeanScoreFoldRange"] = float(group["Score_FoldRange"].mean())
        row["MaxScoreFoldRange"] = float(group["Score_FoldRange"].max())
    return pd.Series(row)


def aggregate_predictions_by_experiment(raw_predictions: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["DrugTarget_UniProt", "DiseaseProtein_UniProt", "ExperimentType"]
    rows = []
    for (*_, experiment_type), group in raw_predictions.groupby(group_columns, dropna=False, sort=True):
        rows.append(_aggregate_group(group, experiment_type=str(experiment_type).strip().lower()))
    return pd.DataFrame(rows)


def aggregate_predictions_all(raw_predictions: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["DrugTarget_UniProt", "DiseaseProtein_UniProt"]
    rows = []
    for _, group in raw_predictions.groupby(group_columns, dropna=False, sort=True):
        rows.append(_aggregate_group(group, experiment_type=None))
    return pd.DataFrame(rows)


def build_step05_report(
    *,
    raw_predictions: pd.DataFrame,
    aggregated: pd.DataFrame,
    aggregated_all: pd.DataFrame,
    manifest: pd.DataFrame | None = None,
    run_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scores = raw_predictions["Score_Sigmoid"] if "Score_Sigmoid" in raw_predictions else pd.Series(dtype=float)
    fold_std = raw_predictions["Score_FoldStd"] if "Score_FoldStd" in raw_predictions else pd.Series(dtype=float)
    fold_range = raw_predictions["Score_FoldRange"] if "Score_FoldRange" in raw_predictions else pd.Series(dtype=float)
    by_type = aggregated["ExperimentType"].value_counts().to_dict() if not aggregated.empty else {}
    report = {
        "run": run_info or {},
        "n_raw_rows": int(len(raw_predictions)),
        "n_aggregated_clean": int(by_type.get("clean", 0)),
        "n_aggregated_extended": int(by_type.get("extended", 0)),
        "n_aggregated_mixed": int((aggregated_all["ExperimentType"] == "mixed").sum()) if not aggregated_all.empty else 0,
        "n_aggregated_all_rows": int(len(aggregated_all)),
        "inputs": {},
        "outputs": {},
        "breakdowns": {
            "raw_by_label": _count_dict(raw_predictions["Label"]) if "Label" in raw_predictions.columns else {},
            "raw_by_experiment_type": _count_dict(raw_predictions["ExperimentType"]) if "ExperimentType" in raw_predictions.columns else {},
            "raw_by_dataset_split": _count_dict(raw_predictions["DatasetSplit"]) if "DatasetSplit" in raw_predictions.columns else {},
            "raw_by_source": _count_dict(raw_predictions["Source"]) if "Source" in raw_predictions.columns else {},
        },
        "score_distribution": {
            "min": float(scores.min()) if len(scores) else 0.0,
            "mean": float(scores.mean()) if len(scores) else 0.0,
            "median": float(scores.median()) if len(scores) else 0.0,
            "max": float(scores.max()) if len(scores) else 0.0,
            "std": float(scores.std(ddof=0)) if len(scores) else 0.0,
            "all_zero": bool((scores == 0).all()) if len(scores) else False,
            "all_one": bool((scores == 1).all()) if len(scores) else False,
        },
        "score_distribution_by_label": {
            str(label): _numeric_summary(group["Score_Sigmoid"])
            for label, group in raw_predictions.groupby("Label", dropna=False)
        } if "Label" in raw_predictions.columns and "Score_Sigmoid" in raw_predictions.columns else {},
        "metrics": {
            "raw": build_binary_classification_metrics(raw_predictions, score_column="Score_Sigmoid"),
            "aggregated_by_experiment_mean": build_binary_classification_metrics(aggregated, score_column="MeanScore"),
            "aggregated_all_mean": build_binary_classification_metrics(aggregated_all, score_column="MeanScore"),
            "aggregated_all_max": build_binary_classification_metrics(aggregated_all, score_column="MaxScore"),
        },
    }
    if manifest is not None:
        report["inputs"]["manifest_rows"] = int(len(manifest))
        report["inputs"]["manifest_by_label"] = _count_dict(manifest["Label"]) if "Label" in manifest.columns else {}
        report["inputs"]["manifest_by_dataset_split"] = _count_dict(manifest["DatasetSplit"]) if "DatasetSplit" in manifest.columns else {}
        report["inputs"]["manifest_unique_pairs"] = int(
            manifest[["DrugTarget_UniProt", "DiseaseProtein_UniProt"]].drop_duplicates().shape[0]
        ) if {"DrugTarget_UniProt", "DiseaseProtein_UniProt"}.issubset(manifest.columns) else 0
        report["inputs"]["manifest_unique_pathways"] = int(manifest["PathwayID"].nunique()) if "PathwayID" in manifest.columns else 0
    if len(fold_std) and len(fold_range):
        report["fold_uncertainty"] = {
            "fold_std_mean": float(fold_std.mean()),
            "fold_std_median": float(fold_std.median()),
            "fold_std_p95": float(fold_std.quantile(0.95)),
            "fold_std_max": float(fold_std.max()),
            "fold_range_mean": float(fold_range.mean()),
            "fold_range_median": float(fold_range.median()),
            "fold_range_p95": float(fold_range.quantile(0.95)),
            "fold_range_max": float(fold_range.max()),
            "fold_range_ge_0_9_rows": int((fold_range >= 0.9).sum()),
        }
    return report


def build_step05_markdown_report(report: dict[str, Any]) -> str:
    metrics = report.get("metrics", {})
    agg_metrics = metrics.get("aggregated_all_mean", {})
    threshold = agg_metrics.get("threshold_0_5", {}) if agg_metrics.get("available") else {}
    lines = [
        "# Step 5 Prediction Report",
        "",
        "## Inputs",
        f"- Manifest rows: {report.get('inputs', {}).get('manifest_rows', report.get('n_raw_rows', 0))}",
        f"- Manifest by label: {report.get('inputs', {}).get('manifest_by_label', {})}",
        f"- Manifest by split: {report.get('inputs', {}).get('manifest_by_dataset_split', {})}",
        f"- Unique manifest pairs: {report.get('inputs', {}).get('manifest_unique_pairs', 0)}",
        f"- Unique manifest pathways: {report.get('inputs', {}).get('manifest_unique_pathways', 0)}",
        "",
        "## Outputs",
        f"- Raw prediction rows: {report.get('n_raw_rows', 0)}",
        f"- Aggregated rows by experiment: {report.get('n_aggregated_clean', 0) + report.get('n_aggregated_extended', 0)}",
        f"- Aggregated rows across pathways: {report.get('n_aggregated_all_rows', 0)}",
        f"- Output files: {report.get('outputs', {})}",
        "",
        "## Score Distribution",
        f"- Overall: {report.get('score_distribution', {})}",
        f"- By label: {report.get('score_distribution_by_label', {})}",
        "",
        "## Binary Metrics",
    ]
    if agg_metrics.get("available"):
        lines.extend(
            [
                f"- ROC AUC: {agg_metrics.get('roc_auc')}",
                f"- Average precision: {agg_metrics.get('average_precision')}",
                f"- Threshold 0.5: accuracy={threshold.get('accuracy')}, precision={threshold.get('precision')}, recall={threshold.get('recall')}, f1={threshold.get('f1')}",
                f"- Confusion matrix @0.5: {threshold.get('confusion_matrix')}",
                f"- Best F1 threshold: {agg_metrics.get('best_f1_threshold')}",
            ]
        )
    else:
        lines.append(f"- Not available: {agg_metrics.get('reason', 'unknown')}")
    lines.extend(["", "## Breakdowns", f"- {report.get('breakdowns', {})}", ""])
    return "\n".join(lines)


def build_step05_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    processed_dir = default_repo_root / "drug_disease_validation" / "data" / "processed"
    model_data_root = default_repo_root / "drug_disease_validation" / "model_data"
    parser = argparse.ArgumentParser(
        description="Run DGN sensitivity predictions over Step 4 subgraphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", default=str(_resolve_existing_manifest(processed_dir)))
    parser.add_argument(
        "--negative-manifest",
        default=None,
        help="Optional Step 4 manifest for negative pairs. When provided, Step 5 predicts positives and negatives together and computes binary metrics.",
    )
    parser.add_argument("--output-dir", default=str(processed_dir))
    parser.add_argument("--model-data-root", default=str(model_data_root))
    parser.add_argument("--checkpoint-dir", default=str(resolve_default_checkpoint_dir(model_data_root)))
    parser.add_argument(
        "--embedding-file",
        default=None,
        help="Optional UniProt embedding pickle used to rebuild node features to match checkpoint embeddings_len.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--prediction-chunk-size",
        type=int,
        default=0,
        help="Number of manifest rows to convert and predict at a time. Use 0 to process all rows at once.",
    )
    parser.add_argument(
        "--accelerator",
        choices=["cpu", "mps", "auto"],
        default="cpu",
        help="Lightning accelerator for prediction. CPU is the safest default on macOS.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of manifest rows processed.")
    parser.add_argument("--negative-limit", type=int, default=None, help="Optional separate limit for --negative-manifest rows.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def run_step05(
    args: argparse.Namespace,
    *,
    predictor: Callable[[Sequence[Any]], pd.DataFrame] | None = None,
) -> None:
    configure_logging(getattr(args, "log_level", "INFO"))
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Step05 started")
    logger.info("Manifest: %s", manifest_path)
    negative_manifest_path = Path(args.negative_manifest) if getattr(args, "negative_manifest", None) else None
    if negative_manifest_path is not None:
        logger.info("Negative manifest: %s", negative_manifest_path)
    manifest = load_prediction_manifests(
        manifest_path=manifest_path,
        negative_manifest_path=negative_manifest_path,
        limit=args.limit,
        negative_limit=getattr(args, "negative_limit", None),
    )
    logger.info("Loaded %s manifest rows", len(manifest))

    checkpoint_dir = Path(args.checkpoint_dir)
    model_data_root = Path(args.model_data_root)
    embeddings = None
    embedding_dim = None
    if predictor is None:
        embedding_dim = resolve_checkpoint_embedding_dim(checkpoint_dir)
        embedding_file = resolve_embedding_file(
            args,
            model_data_root=model_data_root,
            checkpoint_dir=checkpoint_dir,
            embedding_dim=embedding_dim,
        )
        if embedding_file is not None:
            logger.info("Rebuilding node features with embeddings_len=%s from %s", embedding_dim, embedding_file)
            embeddings = load_embedding_dict(embedding_file, expected_dim=embedding_dim)
        else:
            logger.info("Using I/O-only features from subgraph pickles.")

    if predictor is None:
        prediction_scores = predict_manifest_with_dgn_checkpoints(
            manifest,
            checkpoint_dir=checkpoint_dir,
            batch_size=int(args.batch_size),
            accelerator=str(getattr(args, "accelerator", "cpu")),
            embeddings=embeddings,
            embedding_dim=embedding_dim,
            prediction_chunk_size=int(getattr(args, "prediction_chunk_size", 0) or 0),
        )
    else:
        prediction_scores = predict_manifest_with_callable(
            manifest,
            predictor=predictor,
            embeddings=embeddings,
            embedding_dim=embedding_dim,
            prediction_chunk_size=int(getattr(args, "prediction_chunk_size", 0) or 0),
        )

    if len(prediction_scores) != len(manifest):
        raise ValueError(
            f"Prediction count mismatch: got {len(prediction_scores)} scores for {len(manifest)} manifest rows."
        )

    metadata_columns = [
        "SubgraphID",
        "DrugName",
        "DiseaseName",
        "DrugTarget_UniProt",
        "DiseaseProtein_UniProt",
        "PathwayID",
        "PathwayName",
        "Label",
        "ExperimentType",
        "DatasetSplit",
        "Source",
        "EvidenceTier",
    ]
    raw_predictions = pd.concat(
        [manifest[[column for column in metadata_columns if column in manifest.columns]].reset_index(drop=True), prediction_scores],
        axis=1,
    )
    raw_predictions = add_pathway_normalized_scores(raw_predictions)
    aggregated = aggregate_predictions_by_experiment(raw_predictions)
    aggregated_all = aggregate_predictions_all(raw_predictions)
    report = build_step05_report(
        raw_predictions=raw_predictions,
        aggregated=aggregated,
        aggregated_all=aggregated_all,
        manifest=manifest,
        run_info={
            "manifest": str(manifest_path),
            "negative_manifest": str(negative_manifest_path) if negative_manifest_path is not None else None,
            "output_dir": str(output_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "model_data_root": str(model_data_root),
            "batch_size": int(args.batch_size),
            "accelerator": str(getattr(args, "accelerator", "cpu")),
            "prediction_chunk_size": int(getattr(args, "prediction_chunk_size", 0) or 0),
            "limit": getattr(args, "limit", None),
            "negative_limit": getattr(args, "negative_limit", None),
        },
    )

    raw_path = ensure_parent_dir(output_dir / CANONICAL_FILENAMES["predictions_raw"])
    aggregated_path = ensure_parent_dir(output_dir / CANONICAL_FILENAMES["predictions_aggregated"])
    aggregated_all_path = ensure_parent_dir(output_dir / CANONICAL_FILENAMES["predictions_aggregated_all"])
    report_path = ensure_parent_dir(output_dir / DEFAULT_STEP05_REPORT)
    markdown_report_path = ensure_parent_dir(output_dir / DEFAULT_STEP05_SUMMARY)

    raw_predictions.to_csv(raw_path, sep="\t", index=False)
    aggregated.to_csv(aggregated_path, sep="\t", index=False)
    aggregated_all.to_csv(aggregated_all_path, sep="\t", index=False)
    report["outputs"] = {
        "predictions_raw": str(raw_path),
        "predictions_aggregated": str(aggregated_path),
        "predictions_aggregated_all": str(aggregated_all_path),
        "report_json": str(report_path),
        "report_markdown": str(markdown_report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_report_path.write_text(build_step05_markdown_report(report), encoding="utf-8")

    logger.info("Raw predictions: %s", raw_path)
    logger.info("Aggregated predictions by experiment: %s", aggregated_path)
    logger.info("Aggregated predictions across all pathways: %s", aggregated_all_path)
    logger.info("Report: %s", report_path)
    logger.info("Markdown report: %s", markdown_report_path)
