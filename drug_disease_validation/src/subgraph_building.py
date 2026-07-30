from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import networkx as nx
import numpy as np
import pandas as pd

from api_clients.data_processing import load_tabular_file
from drug_disease_validation.src.subgraph_sharding import SHARD_FORMAT
from drug_disease_validation.src.utils import configure_logging, load_yaml_file, read_pickle, write_pickle


logger = logging.getLogger(__name__)

DEFAULT_STEP04_CONFIG = {
    "include_bridge_nodes_for_extended": True,
    "save_subgraphs": True,
    "subgraphs_dirname": "subgraphs",
    "enable_cache": True,
    "min_subgraph_nodes": 5,
    "max_subgraph_nodes": 50,
    "bridge_max_intermediate_nodes": 1,
    "use_embeddings": False,
    "embedding_source": "local_file",
    "embedding_file": "",
    "missing_embedding_policy": "zeros",
    "log_missing_embeddings": True,
}

STEP04_PROGRESS_EVERY = 5_000


def attach_file_logger(log_path: Path) -> None:
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    root_logger = logging.getLogger()
    resolved_path = str(log_path.resolve())
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == resolved_path:
            return

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def summarize_numeric(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "mean": 0.0, "median": 0.0, "max": 0}
    series = pd.Series(values, dtype="int64")
    return {
        "count": int(series.count()),
        "min": int(series.min()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "max": int(series.max()),
    }


def resolve_dataset_variant(config: dict[str, Any], embedding_dim: int) -> str:
    if not config["use_embeddings"]:
        return "io"
    if embedding_dim > 0:
        return f"io+emb{embedding_dim}"
    return "io+emb"


def sanitise_variant_for_filename(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "default"
    return "".join(char if char.isalnum() or char in {"_", "-", "+"} else "_" for char in text)


def _log_step04_header(*, config: dict[str, Any], output_dir: Path, dataset_variant: str, subgraphs_dir: Path) -> None:
    logger.info("Step04 started")
    logger.info(
        "Config: embeddings=%s | bridge_nodes=%s | save_subgraphs=%s | cache=%s | size_range=[%s,%s] | bridge_max_intermediate_nodes=%s | variant=%s",
        config["use_embeddings"],
        config["include_bridge_nodes_for_extended"],
        config["save_subgraphs"],
        config["enable_cache"],
        config["min_subgraph_nodes"],
        config["max_subgraph_nodes"],
        config["bridge_max_intermediate_nodes"],
        dataset_variant,
    )
    logger.info("Output: %s", output_dir)
    if config["save_subgraphs"]:
        logger.info("Subgraphs dir: %s", subgraphs_dir)


def _log_step04_inputs(
    *,
    pair_pathway_rows: int,
    positive_pairs: int,
    self_loops: int,
    reactome_rows: int,
    biogrid_nodes: int,
    biogrid_edges: int,
    pathways_with_biogrid_overlap: int,
) -> None:
    logger.info(
        "Inputs: pair_pathways=%s | positive_pairs=%s | self_loops=%s | reactome_rows=%s | biogrid_nodes=%s | biogrid_edges=%s",
        pair_pathway_rows,
        positive_pairs,
        self_loops,
        reactome_rows,
        biogrid_nodes,
        biogrid_edges,
    )
    logger.info("Pathways with BioGRID overlap: %s", pathways_with_biogrid_overlap)


def _log_step04_progress(*, processed: int, total: int, built: int, skipped: int, cached: int) -> None:
    pct = (100.0 * processed / total) if total else 100.0
    logger.info(
        "Progress: %s/%s rows (%.1f%%) | built=%s | cached=%s | skipped=%s",
        processed,
        total,
        pct,
        built,
        cached,
        skipped,
    )


def _log_step04_footer(
    *,
    built: int,
    skipped: int,
    cached: int,
    manifest_path: Path,
    report_path: Path,
    skipped_rows: dict[str, int],
) -> None:
    logger.info("Step04 finished")
    logger.info("Built: %s | Cached: %s | Skipped: %s", built, cached, skipped)
    logger.info("Manifest: %s", manifest_path)
    logger.info("Report: %s", report_path)
    if skipped_rows:
        logger.info("Skip reasons: %s", skipped_rows)


def _normalize_manifest_cache_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized.setdefault("BridgeExtended", False)
    normalized.setdefault("BridgePathNodes", "")
    normalized.setdefault("ConnectivityExtended", False)
    normalized.setdefault("ConnectivityExtensionNodeCount", 0)
    normalized.setdefault("ConnectivityExtensionNodes", "")
    for column in [
        "BridgeNeeded",
        "BridgeExtended",
        "HasDrugTargetEmbedding",
        "HasDiseaseProteinEmbedding",
        "InputOutputConnected",
    ]:
        if column in normalized:
            normalized[column] = str(normalized[column]).strip().lower() in {"true", "1", "yes"}
    for column in [
        "NumNodes",
        "NumEdges",
        "BridgeNodeCount",
        "ConnectivityExtensionNodeCount",
        "PathwayNodesInBioGRID",
        "MissingEmbeddingCount",
        "SensitivityLabel",
    ]:
        if column in normalized and normalized[column] != "":
            normalized[column] = int(normalized[column])
    if "ConnectivityExtended" in normalized:
        normalized["ConnectivityExtended"] = str(normalized["ConnectivityExtended"]).strip().lower() in {"true", "1", "yes"}
    return normalized


def _resolve_manifest_subgraph_path(raw_path: object, manifest_dir: Path) -> Path | None:
    text = str(raw_path).strip()
    if not text or text.lower() == "nan":
        return None

    path = Path(text)
    candidates = [path] if path.is_absolute() else [manifest_dir / path, path]

    # Older manifests stored absolute paths from the machine that generated them.
    # If the same subgraph directory exists under the current output directory,
    # prefer that local copy and rewrite the cached row to the portable form.
    if path.parent.name and path.name:
        candidates.append(manifest_dir / path.parent.name / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _format_subgraph_path_for_manifest(subgraph_path: Path | None, output_dir: Path) -> str:
    if subgraph_path is None:
        return ""
    try:
        return str(subgraph_path.resolve().relative_to(output_dir.resolve()))
    except ValueError:
        return str(subgraph_path)


def _write_subgraph_shard(
    *,
    graphs: dict[str, nx.Graph],
    shard_path: Path,
    source_manifest: Path,
    shard_index: int,
) -> int:
    payload = {
        "format": SHARD_FORMAT,
        "metadata": {
            "source_manifest": str(source_manifest),
            "shard_index": int(shard_index),
            "shard_size": len(graphs),
            "subgraph_ids": list(graphs.keys()),
        },
        "graphs": graphs,
    }
    write_pickle(payload, shard_path)
    return int(shard_path.stat().st_size)


def load_manifest_cache(
    manifest_path: Path,
    *,
    save_subgraphs: bool,
    min_subgraph_nodes: int | None = None,
    max_subgraph_nodes: int | None = None,
) -> dict[str, dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    manifest_df = load_tabular_file(manifest_path)
    if manifest_df.empty or "SubgraphID" not in manifest_df.columns:
        return {}

    manifest_dir = manifest_path.parent
    cache: dict[str, dict[str, Any]] = {}
    for row in manifest_df.to_dict(orient="records"):
        subgraph_id = str(row.get("SubgraphID", "")).strip()
        if not subgraph_id:
            continue
        normalized = _normalize_manifest_cache_row(row)
        subgraph_path = _resolve_manifest_subgraph_path(normalized.get("SubgraphPath", ""), manifest_dir)
        if save_subgraphs and (subgraph_path is None or not subgraph_path.exists()):
            continue
        if subgraph_path is not None and subgraph_path.exists():
            normalized["SubgraphPath"] = _format_subgraph_path_for_manifest(subgraph_path, manifest_dir)
        num_nodes = int(normalized.get("NumNodes", 0) or 0)
        if min_subgraph_nodes is not None and num_nodes < min_subgraph_nodes:
            continue
        if max_subgraph_nodes is not None and num_nodes > max_subgraph_nodes:
            continue
        cache[subgraph_id] = normalized
    return cache


def load_skip_cache(skip_cache_path: Path) -> dict[str, str]:
    if not skip_cache_path.exists():
        return {}
    with skip_cache_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in data.items()
        if str(key).strip() and str(value).strip()
    }


class H5EmbeddingStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = h5py.File(path, "r")
        self.keys_view = self.handle.keys()
        first_key = next(iter(self.keys_view), None)
        self.embedding_dim = 0
        if first_key is not None:
            dataset = self.handle[first_key]
            self.embedding_dim = int(np.asarray(dataset[()]).reshape(-1).shape[0])

    def __len__(self) -> int:
        return len(self.handle.keys())

    def get(self, protein_id: str) -> np.ndarray | None:
        if protein_id not in self.handle:
            return None
        return np.asarray(self.handle[protein_id][()], dtype=float).reshape(-1)

    def close(self) -> None:
        self.handle.close()


def _resolve_step04_config(args: argparse.Namespace, default_repo_root: Path) -> dict[str, Any]:
    config_path = Path(args.config) if args.config else (
        default_repo_root / "drug_disease_validation" / "config" / "step04.yml"
    )
    config = DEFAULT_STEP04_CONFIG.copy()
    loaded = load_yaml_file(config_path)
    step04_cfg = loaded.get("step04", {})
    if step04_cfg and not isinstance(step04_cfg, dict):
        raise ValueError(f"Expected mapping under 'step04' in config: {config_path}")
    config.update(step04_cfg)
    if getattr(args, "disable_cache", False):
        config["enable_cache"] = False

    config["config_path"] = config_path
    config["min_subgraph_nodes"] = int(config["min_subgraph_nodes"])
    config["max_subgraph_nodes"] = int(config["max_subgraph_nodes"])
    config["bridge_max_intermediate_nodes"] = int(config["bridge_max_intermediate_nodes"])
    if config["min_subgraph_nodes"] < 1:
        raise ValueError("min_subgraph_nodes must be >= 1")
    if config["max_subgraph_nodes"] < config["min_subgraph_nodes"]:
        raise ValueError("max_subgraph_nodes must be >= min_subgraph_nodes")
    if config["bridge_max_intermediate_nodes"] < 0:
        raise ValueError("bridge_max_intermediate_nodes must be >= 0")
    if config["missing_embedding_policy"] != "zeros":
        raise ValueError("Only missing_embedding_policy='zeros' is supported at the moment.")
    if config["embedding_source"] not in {"local_file"}:
        raise ValueError("Only embedding_source='local_file' is supported at the moment.")
    return config


def build_step04_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build pathway-based BioGRID subgraphs and optional node features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed_dir = default_repo_root / "drug_disease_validation" / "data" / "processed"
    parser.add_argument("--pair-pathway-mapping", default=str(processed_dir / "pair_pathway_mapping.tsv"))
    parser.add_argument("--positive-pairs", default=str(processed_dir / "positive_pairs.tsv"))
    parser.add_argument("--positive-pairs-self-loops", default=str(processed_dir / "positive_pairs_self_loops.tsv"))
    parser.add_argument("--reactome-pathways", default=str(processed_dir / "reactome_uniprot2AllPathways.tsv"))
    parser.add_argument("--biogrid-graph", default=str(processed_dir / "biogrid_graph.pkl"))
    parser.add_argument("--output-dir", default=str(processed_dir))
    parser.add_argument("--config", default=str(default_repo_root / "drug_disease_validation" / "config" / "step04.yml"))
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of pathway rows processed.")
    parser.add_argument("--disable-cache", action="store_true", help="Ignore existing Step04 manifest and skip caches.")
    parser.add_argument("--run-log-dir", default=None, help="Directory for Step04's internal log file.")
    parser.add_argument(
        "--subgraph-storage",
        choices=["files", "shards"],
        default="files",
        help="Store each subgraph as one pickle file, or write larger shard pickles directly.",
    )
    parser.add_argument("--shard-size", type=int, default=1000, help="Number of subgraphs per shard when --subgraph-storage=shards.")
    parser.add_argument("--shards-dir", default=None, help="Shard output directory when --subgraph-storage=shards.")
    parser.add_argument("--logging-level", default="INFO")
    return parser


def load_embeddings(embedding_file: Path) -> tuple[Any, int]:
    if not embedding_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_file}")

    if embedding_file.suffix.lower() in {".h5", ".hdf5"}:
        store = H5EmbeddingStore(embedding_file)
        return store, store.embedding_dim

    if embedding_file.suffix.lower() == ".pkl":
        raw = read_pickle(embedding_file)
        if not isinstance(raw, dict):
            raise ValueError("Expected pickle embedding file to contain a dict[UniProtID] -> vector.")
        embeddings: dict[str, np.ndarray] = {}
        embedding_dim = 0
        for key, value in raw.items():
            vector = np.asarray(value, dtype=float).reshape(-1)
            embeddings[str(key)] = vector
            if embedding_dim == 0:
                embedding_dim = int(vector.shape[0])
        return embeddings, embedding_dim

    table = load_tabular_file(embedding_file)
    if table.empty:
        return {}, 0
    if "UniProtID" in table.columns:
        id_column = "UniProtID"
    else:
        id_column = str(table.columns[0])
    feature_columns = [column for column in table.columns if column != id_column]
    if not feature_columns:
        raise ValueError(f"Embedding table {embedding_file} does not contain feature columns.")

    embeddings = {}
    for row in table.itertuples(index=False):
        row_dict = row._asdict()
        protein_id = str(row_dict[id_column]).strip()
        if not protein_id:
            continue
        embeddings[protein_id] = np.asarray([row_dict[column] for column in feature_columns], dtype=float)
    return embeddings, len(feature_columns)


def build_pathway_to_proteins(reactome_df: pd.DataFrame, biogrid_nodes: set[str]) -> dict[str, set[str]]:
    if reactome_df.empty:
        return {}
    required = {"UniProtID", "PathwayID"}
    missing = required.difference(reactome_df.columns)
    if missing:
        raise ValueError(f"Reactome file missing columns: {sorted(missing)}")

    filtered = reactome_df[reactome_df["UniProtID"].astype(str).isin(biogrid_nodes)].copy()
    mapping: dict[str, set[str]] = {}
    for row in filtered.itertuples(index=False):
        pathway_id = str(row.PathwayID).strip()
        protein_id = str(row.UniProtID).strip()
        if not pathway_id or not protein_id:
            continue
        mapping.setdefault(pathway_id, set()).add(protein_id)
    return mapping


def build_pair_metadata(positive_pairs_df: pd.DataFrame) -> pd.DataFrame:
    if positive_pairs_df.empty:
        return pd.DataFrame(
            columns=[
                "DrugTarget_UniProt",
                "DiseaseProtein_UniProt",
                "DrugName",
                "DrugBankID",
                "DiseaseName",
                "DiseaseID",
            ]
        )

    sort_columns = [column for column in ["EvidenceTier", "Source", "DrugName", "DiseaseName"] if column in positive_pairs_df.columns]
    frame = positive_pairs_df.sort_values(sort_columns, kind="stable").copy() if sort_columns else positive_pairs_df.copy()
    return frame.drop_duplicates(
        subset=["DrugTarget_UniProt", "DiseaseProtein_UniProt"],
        keep="first",
    )[
        [column for column in ["DrugTarget_UniProt", "DiseaseProtein_UniProt", "DrugName", "DrugBankID", "DiseaseName", "DiseaseID"] if column in frame.columns]
    ]


def resolve_bridge_path_to_pathway(
    *,
    graph: nx.Graph,
    pathway_nodes: set[str],
    endpoint: str,
    max_intermediate_nodes: int,
) -> list[str]:
    if endpoint in pathway_nodes or endpoint not in graph:
        return []

    best_path: list[str] | None = None
    for pathway_node in pathway_nodes:
        if pathway_node not in graph:
            continue
        try:
            path = nx.shortest_path(graph, source=endpoint, target=pathway_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        intermediate_nodes = path[1:-1]
        if len(intermediate_nodes) > max_intermediate_nodes:
            continue
        if best_path is None or len(path) < len(best_path):
            best_path = path

    return best_path or []


def resolve_connectivity_extension_nodes(
    *,
    graph: nx.Graph,
    source: str,
    target: str,
    existing_nodes: set[str],
) -> list[str]:
    if source not in graph or target not in graph:
        return []
    try:
        shortest_path = nx.shortest_path(graph, source=source, target=target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    return [node for node in shortest_path[1:-1] if node not in existing_nodes]


def build_sample_subgraph(
    *,
    row: pd.Series,
    biogrid_graph: nx.Graph,
    pathway_nodes: set[str],
    include_bridge_nodes_for_extended: bool,
    min_subgraph_nodes: int,
    max_subgraph_nodes: int,
    bridge_max_intermediate_nodes: int,
) -> tuple[nx.Graph | None, dict[str, Any]]:
    pathway_node_set = set(pathway_nodes)
    drug_target = str(row["DrugTarget_UniProt"])
    disease_protein = str(row["DiseaseProtein_UniProt"])
    experiment_type = str(row.get("ExperimentType", "")).strip().lower()
    bridge_needed = bool(row.get("BridgeNeeded", False))

    all_nodes = set(pathway_node_set)
    bridge_nodes: set[str] = set()
    bridge_path_nodes: list[str] = []
    connectivity_extension_nodes: list[str] = []
    unresolved_bridge_endpoints: list[str] = []
    endpoints_outside_pathway = []

    for endpoint in (drug_target, disease_protein):
        if endpoint not in biogrid_graph:
            return None, {
                "status": "skipped_endpoint_missing_in_biogrid",
                "bridge_nodes": [],
                "bridge_path_nodes": [],
                "bridge_extended": False,
                "connectivity_extension_nodes": [],
                "connectivity_extended": False,
                "endpoints_outside_pathway": [],
            }
        if endpoint not in pathway_node_set:
            endpoints_outside_pathway.append(endpoint)
            all_nodes.add(endpoint)

    if include_bridge_nodes_for_extended and experiment_type == "extended" and bridge_needed:
        for endpoint in endpoints_outside_pathway:
            bridge_path = resolve_bridge_path_to_pathway(
                graph=biogrid_graph,
                pathway_nodes=pathway_node_set,
                endpoint=endpoint,
                max_intermediate_nodes=bridge_max_intermediate_nodes,
            )
            if bridge_path:
                bridge_path_nodes.extend(bridge_path)
                bridge_nodes.update(node for node in bridge_path[1:-1] if node not in pathway_node_set)
            else:
                unresolved_bridge_endpoints.append(endpoint)
        all_nodes.update(bridge_nodes)

    if not pathway_node_set:
        return None, {
            "status": "skipped_empty_pathway_after_biogrid_filter",
            "bridge_nodes": [],
            "bridge_path_nodes": bridge_path_nodes,
            "bridge_extended": bool(bridge_path_nodes),
            "connectivity_extension_nodes": [],
            "connectivity_extended": False,
            "endpoints_outside_pathway": endpoints_outside_pathway,
        }

    if experiment_type == "extended" and include_bridge_nodes_for_extended and unresolved_bridge_endpoints:
        return None, {
            "status": "skipped_missing_bridge_nodes",
            "bridge_nodes": sorted(bridge_nodes),
            "bridge_path_nodes": bridge_path_nodes,
            "bridge_extended": bool(bridge_path_nodes),
            "connectivity_extension_nodes": [],
            "connectivity_extended": False,
            "endpoints_outside_pathway": endpoints_outside_pathway,
        }

    subgraph = biogrid_graph.subgraph(all_nodes).copy()
    subgraph.remove_edges_from(nx.selfloop_edges(subgraph))
    if drug_target not in subgraph or disease_protein not in subgraph:
        return None, {
            "status": "skipped_endpoints_not_in_final_subgraph",
            "bridge_nodes": sorted(bridge_nodes),
            "bridge_path_nodes": bridge_path_nodes,
            "bridge_extended": bool(bridge_path_nodes),
            "connectivity_extension_nodes": [],
            "connectivity_extended": False,
            "endpoints_outside_pathway": endpoints_outside_pathway,
        }
    if not nx.has_path(subgraph, drug_target, disease_protein):
        connectivity_extension_nodes = resolve_connectivity_extension_nodes(
            graph=biogrid_graph,
            source=drug_target,
            target=disease_protein,
            existing_nodes=all_nodes,
        )
        if not connectivity_extension_nodes:
            return None, {
                "status": "skipped_disconnected_input_output",
                "bridge_nodes": sorted(bridge_nodes),
                "bridge_path_nodes": bridge_path_nodes,
                "bridge_extended": bool(bridge_path_nodes),
                "connectivity_extension_nodes": [],
                "connectivity_extended": False,
                "endpoints_outside_pathway": endpoints_outside_pathway,
            }
        all_nodes.update(connectivity_extension_nodes)
        subgraph = biogrid_graph.subgraph(all_nodes).copy()
        subgraph.remove_edges_from(nx.selfloop_edges(subgraph))
        if not nx.has_path(subgraph, drug_target, disease_protein):
            return None, {
                "status": "skipped_disconnected_input_output",
                "bridge_nodes": sorted(bridge_nodes),
                "bridge_path_nodes": bridge_path_nodes,
                "bridge_extended": bool(bridge_path_nodes),
                "connectivity_extension_nodes": connectivity_extension_nodes,
                "connectivity_extended": True,
                "endpoints_outside_pathway": endpoints_outside_pathway,
            }

    subgraph_node_count = subgraph.number_of_nodes()
    if subgraph_node_count < min_subgraph_nodes:
        return None, {
            "status": "skipped_too_small",
            "bridge_nodes": sorted(bridge_nodes),
            "bridge_path_nodes": bridge_path_nodes,
            "bridge_extended": bool(bridge_path_nodes),
            "connectivity_extension_nodes": connectivity_extension_nodes,
            "connectivity_extended": bool(connectivity_extension_nodes),
            "endpoints_outside_pathway": endpoints_outside_pathway,
        }
    if subgraph_node_count > max_subgraph_nodes:
        return None, {
            "status": "skipped_too_large",
            "bridge_nodes": sorted(bridge_nodes),
            "bridge_path_nodes": bridge_path_nodes,
            "bridge_extended": bool(bridge_path_nodes),
            "connectivity_extension_nodes": connectivity_extension_nodes,
            "connectivity_extended": bool(connectivity_extension_nodes),
            "endpoints_outside_pathway": endpoints_outside_pathway,
        }

    return subgraph, {
        "status": "built",
        "bridge_nodes": sorted(bridge_nodes),
        "bridge_path_nodes": bridge_path_nodes,
        "bridge_extended": bool(bridge_path_nodes),
        "connectivity_extension_nodes": connectivity_extension_nodes,
        "connectivity_extended": bool(connectivity_extension_nodes),
        "endpoints_outside_pathway": endpoints_outside_pathway,
    }


def annotate_subgraph(
    graph: nx.Graph,
    *,
    drug_target: str,
    disease_protein: str,
    embeddings: Any,
    embedding_dim: int,
    use_embeddings: bool,
) -> dict[str, Any]:
    missing_nodes: list[str] = []
    has_drug_embedding = not use_embeddings
    has_disease_embedding = not use_embeddings

    ordered_nodes = sorted(str(node) for node in graph.nodes())
    for node in ordered_nodes:
        is_input = 1
        if node != drug_target:
            is_input = 0
        is_disease = 1
        if node != disease_protein:
            is_disease = 0

        embedding_found = False
        embedding_vector = np.zeros(embedding_dim, dtype=float)
        if use_embeddings:
            vector = embeddings.get(node) if embeddings is not None else None
            if vector is not None:
                if int(vector.shape[0]) != embedding_dim:
                    raise ValueError(
                        f"Inconsistent embedding dimension for {node}: expected {embedding_dim}, found {vector.shape[0]}"
                    )
                embedding_vector = vector
                embedding_found = True
            else:
                missing_nodes.append(node)
        feature_vector = np.concatenate((np.asarray([is_input, is_disease], dtype=float), embedding_vector), axis=0)
        graph.nodes[node]["uniprot_id"] = node
        graph.nodes[node]["is_input"] = bool(is_input)
        graph.nodes[node]["is_disease"] = bool(is_disease)
        graph.nodes[node]["embedding_found"] = embedding_found if use_embeddings else False
        graph.nodes[node]["features"] = feature_vector.tolist()

        if node == drug_target:
            has_drug_embedding = embedding_found or not use_embeddings
        if node == disease_protein:
            has_disease_embedding = embedding_found or not use_embeddings

    graph.graph["node_order"] = ordered_nodes
    graph.graph["feature_dim"] = int(2 + embedding_dim)
    return {
        "missing_embedding_nodes": missing_nodes,
        "has_drug_target_embedding": bool(has_drug_embedding),
        "has_disease_protein_embedding": bool(has_disease_embedding),
    }


def validate_subgraph(graph: nx.Graph, *, drug_target: str, disease_protein: str, embedding_dim: int) -> None:
    if drug_target not in graph:
        raise ValueError(f"Input node {drug_target} missing from final subgraph.")
    if disease_protein not in graph:
        raise ValueError(f"Output node {disease_protein} missing from final subgraph.")
    expected_dim = 2 + embedding_dim
    for node, payload in graph.nodes(data=True):
        features = payload.get("features")
        if not isinstance(features, list) or len(features) != expected_dim:
            raise ValueError(f"Node {node} has invalid feature vector length: expected {expected_dim}.")


def build_manifest_row(
    *,
    subgraph_id: str,
    row: pd.Series,
    pair_metadata: pd.Series | None,
    subgraph: nx.Graph,
    subgraph_path: Path | None,
    output_dir: Path,
    bridge_nodes: list[str],
    bridge_path_nodes: list[str],
    bridge_extended: bool,
    connectivity_extension_nodes: list[str],
    connectivity_extended: bool,
    embedding_info: dict[str, Any],
    pathway_nodes_count: int,
) -> dict[str, Any]:
    same_component = False
    drug_target = str(row["DrugTarget_UniProt"])
    disease_protein = str(row["DiseaseProtein_UniProt"])
    if drug_target in subgraph and disease_protein in subgraph:
        same_component = nx.has_path(subgraph, drug_target, disease_protein)

    manifest_row = {
        "SubgraphID": subgraph_id,
        "DrugTarget_UniProt": drug_target,
        "DiseaseProtein_UniProt": disease_protein,
        "PathwayID": str(row["PathwayID"]),
        "PathwayName": str(row.get("PathwayName", "")),
        "ExperimentType": str(row.get("ExperimentType", "")),
        "BridgeNeeded": bool(row.get("BridgeNeeded", False)),
        "NumNodes": int(subgraph.number_of_nodes()),
        "NumEdges": int(subgraph.number_of_edges()),
        "SubgraphPath": _format_subgraph_path_for_manifest(subgraph_path, output_dir),
        "Source": str(row.get("Source", "")),
        "EvidenceTier": str(row.get("EvidenceTier", "")),
        "SensitivityLabel": int(row.get("Label", 1)),
        "BridgeExtended": bool(bridge_extended),
        "BridgeNodeCount": int(len(bridge_nodes)),
        "BridgeNodes": "|".join(bridge_nodes),
        "BridgePathNodes": "|".join(bridge_path_nodes),
        "ConnectivityExtended": bool(connectivity_extended),
        "ConnectivityExtensionNodeCount": int(len(connectivity_extension_nodes)),
        "ConnectivityExtensionNodes": "|".join(connectivity_extension_nodes),
        "PathwayNodesInBioGRID": int(pathway_nodes_count),
        "HasDrugTargetEmbedding": bool(embedding_info["has_drug_target_embedding"]),
        "HasDiseaseProteinEmbedding": bool(embedding_info["has_disease_protein_embedding"]),
        "MissingEmbeddingCount": int(len(embedding_info["missing_embedding_nodes"])),
        "MissingEmbeddingNodes": "|".join(embedding_info["missing_embedding_nodes"]),
        "InputOutputConnected": bool(same_component),
    }
    if pair_metadata is not None:
        for column in ["DrugName", "DrugBankID", "DiseaseName", "DiseaseID"]:
            manifest_row[column] = str(pair_metadata.get(column, "")) if column in pair_metadata else ""
    return manifest_row


def build_step04_report(
    *,
    processed_rows: int,
    built_rows: int,
    skipped_rows: dict[str, int],
    manifest_df: pd.DataFrame,
    embedding_dim: int,
    use_embeddings: bool,
    self_loops_rows: int,
    cache_hits: int,
    cache_skip_hits: int,
    self_loop_pathway_rows_excluded: int = 0,
) -> dict[str, Any]:
    report = {
        "pair_pathway_rows_total": int(processed_rows),
        "subgraphs_built": int(built_rows),
        "subgraphs_skipped": int(sum(skipped_rows.values())),
        "cache_hits": int(cache_hits),
        "cache_skip_hits": int(cache_skip_hits),
        "skip_reasons": {key: int(value) for key, value in sorted(skipped_rows.items())},
        "filtered_disconnected_input_output_pairs": int(skipped_rows.get("skipped_disconnected_input_output", 0)),
        "filtered_too_small_subgraphs": int(skipped_rows.get("skipped_too_small", 0)),
        "filtered_too_large_subgraphs": int(skipped_rows.get("skipped_too_large", 0)),
        "self_loop_rows_excluded": int(self_loops_rows),
        "self_loop_pathway_rows_excluded": int(self_loop_pathway_rows_excluded),
        "use_embeddings": bool(use_embeddings),
        "embedding_dim": int(embedding_dim),
    }
    if manifest_df.empty:
        report.update(
            {
                "graph_size_distribution": {"nodes": summarize_numeric([]), "edges": summarize_numeric([])},
                "connected_input_output_pairs": 0,
                "disconnected_input_output_pairs": 0,
                "zero_edge_graphs": 0,
                "graphs_with_missing_embeddings": 0,
                "total_missing_embedding_nodes": 0,
                "graphs_with_bridge_extension": 0,
                "graphs_with_bridge_nodes": 0,
                "bridge_node_distribution": summarize_numeric([]),
                "graphs_with_connectivity_extension": 0,
                "connectivity_extension_distribution": summarize_numeric([]),
                "experiment_type_counts": {},
            }
        )
        return report

    report.update(
        {
            "graph_size_distribution": {
                "nodes": summarize_numeric(manifest_df["NumNodes"].astype(int).tolist()),
                "edges": summarize_numeric(manifest_df["NumEdges"].astype(int).tolist()),
            },
            "connected_input_output_pairs": int(manifest_df["InputOutputConnected"].astype(bool).sum()),
            "disconnected_input_output_pairs": int((~manifest_df["InputOutputConnected"].astype(bool)).sum()),
            "zero_edge_graphs": int((manifest_df["NumEdges"].astype(int) == 0).sum()),
            "graphs_with_missing_embeddings": int((manifest_df["MissingEmbeddingCount"].astype(int) > 0).sum()),
            "total_missing_embedding_nodes": int(manifest_df["MissingEmbeddingCount"].astype(int).sum()),
            "graphs_with_bridge_extension": int(manifest_df["BridgeExtended"].fillna(False).astype(bool).sum()),
            "graphs_with_bridge_nodes": int((manifest_df["BridgeNodeCount"].astype(int) > 0).sum()),
            "bridge_node_distribution": summarize_numeric(manifest_df["BridgeNodeCount"].astype(int).tolist()),
            "graphs_with_connectivity_extension": int(manifest_df["ConnectivityExtended"].fillna(False).astype(bool).sum()),
            "connectivity_extension_distribution": summarize_numeric(
                manifest_df["ConnectivityExtensionNodeCount"].fillna(0).astype(int).tolist()
            ),
            "experiment_type_counts": {
                str(key): int(value)
                for key, value in manifest_df["ExperimentType"].value_counts(dropna=False).to_dict().items()
            },
        }
    )
    return report


def run_step04(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    repo_root = Path(__file__).resolve().parents[2]
    config = _resolve_step04_config(args, repo_root)

    pair_pathway_df = load_tabular_file(Path(args.pair_pathway_mapping))
    positive_pairs_df = load_tabular_file(Path(args.positive_pairs))
    self_loops_df = load_tabular_file(Path(args.positive_pairs_self_loops))
    reactome_df = load_tabular_file(Path(args.reactome_pathways))
    biogrid_graph = read_pickle(Path(args.biogrid_graph))

    self_loop_mask = (
        pair_pathway_df["DrugTarget_UniProt"].astype(str)
        == pair_pathway_df["DiseaseProtein_UniProt"].astype(str)
    )
    self_loop_pathway_rows_excluded = int(self_loop_mask.sum())
    if self_loop_pathway_rows_excluded:
        pair_pathway_df = pair_pathway_df.loc[~self_loop_mask].copy()
        logger.info("Self-loop pathway rows removed from Step04 input: %s", self_loop_pathway_rows_excluded)

    if args.limit is not None:
        pair_pathway_df = pair_pathway_df.head(args.limit).copy()

    pair_pathway_df["BridgeNeeded"] = pair_pathway_df["BridgeNeeded"].map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
    )
    pair_metadata_df = build_pair_metadata(positive_pairs_df)
    biogrid_nodes = {str(node) for node in biogrid_graph.nodes()}
    pathway_to_proteins = build_pathway_to_proteins(reactome_df, biogrid_nodes)

    embeddings: dict[str, np.ndarray] | None = None
    embedding_dim = 0
    embeddings_closer = None
    if config["use_embeddings"]:
        embedding_path = Path(str(config["embedding_file"])).expanduser()
        embeddings, embedding_dim = load_embeddings(embedding_path)
        if hasattr(embeddings, "close"):
            embeddings_closer = embeddings.close
        logger.info("Embeddings: %s | proteins=%s | dim=%s", embedding_path, len(embeddings), embedding_dim)
    else:
        logger.info("Embeddings: disabled")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_variant = resolve_dataset_variant(config, embedding_dim)
    variant_slug = sanitise_variant_for_filename(dataset_variant)
    subgraph_storage = str(getattr(args, "subgraph_storage", "files"))
    if subgraph_storage not in {"files", "shards"}:
        raise ValueError(f"Unsupported subgraph storage mode: {subgraph_storage}")
    shard_size = int(getattr(args, "shard_size", 1000))
    if shard_size < 1:
        raise ValueError("shard_size must be >= 1")
    subgraphs_dir = output_dir / f"{config['subgraphs_dirname']}_{variant_slug}"
    shards_dir = Path(args.shards_dir) if getattr(args, "shards_dir", None) else output_dir / f"subgraph_shards_{variant_slug}"
    if config["save_subgraphs"]:
        if subgraph_storage == "shards":
            shards_dir.mkdir(parents=True, exist_ok=True)
        else:
            subgraphs_dir.mkdir(parents=True, exist_ok=True)
    manifest_suffix = f"{variant_slug}_sharded" if subgraph_storage == "shards" else variant_slug
    manifest_filename = f"subgraph_manifest_{manifest_suffix}.tsv"
    report_filename = f"step04_report_{manifest_suffix}.json"
    skip_cache_filename = f"step04_skip_cache_{variant_slug}.json"
    log_filename = f"step04_run_{variant_slug}.log"
    manifest_path = output_dir / manifest_filename
    report_path = output_dir / report_filename
    skip_cache_path = output_dir / skip_cache_filename
    run_log_dir = Path(args.run_log_dir) if args.run_log_dir else output_dir
    run_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_log_dir / log_filename
    attach_file_logger(log_path)
    _log_step04_header(
        config=config,
        output_dir=output_dir,
        dataset_variant=dataset_variant,
        subgraphs_dir=shards_dir if subgraph_storage == "shards" else subgraphs_dir,
    )
    logger.info("Subgraph storage: %s", subgraph_storage)
    if subgraph_storage == "shards":
        logger.info("Shard size: %s", shard_size)
        logger.info("Shards dir: %s", shards_dir)
    logger.info("Log file: %s", log_path)

    manifest_rows: list[dict[str, Any]] = []
    skipped_rows: dict[str, int] = {}
    cache_hits = 0
    cache_skip_hits = 0
    shard_graphs: dict[str, nx.Graph] = {}
    shard_index = 0
    shard_reports: list[dict[str, Any]] = []

    def flush_shard() -> None:
        nonlocal shard_graphs, shard_index
        if not shard_graphs:
            return
        shard_path = shards_dir / f"subgraph_shard_{shard_index:05d}.pkl"
        shard_bytes = _write_subgraph_shard(
            graphs=shard_graphs,
            shard_path=shard_path,
            source_manifest=manifest_path,
            shard_index=shard_index,
        )
        shard_reports.append(
            {
                "shard_index": int(shard_index),
                "path": str(shard_path),
                "rows": int(len(shard_graphs)),
                "bytes": int(shard_bytes),
            }
        )
        logger.info("Shard written: %s rows=%s size_mb=%.2f", shard_path, len(shard_graphs), shard_bytes / 1_000_000)
        shard_graphs = {}
        shard_index += 1
    manifest_cache = (
        load_manifest_cache(
            manifest_path,
            save_subgraphs=bool(config["save_subgraphs"]),
            min_subgraph_nodes=int(config["min_subgraph_nodes"]),
            max_subgraph_nodes=int(config["max_subgraph_nodes"]),
        )
        if config["enable_cache"]
        else {}
    )
    skip_cache = load_skip_cache(skip_cache_path) if config["enable_cache"] else {}
    if config["enable_cache"]:
        logger.info("Cache loaded: manifest_rows=%s | skipped_rows=%s", len(manifest_cache), len(skip_cache))

    metadata_lookup = {
        (str(row["DrugTarget_UniProt"]), str(row["DiseaseProtein_UniProt"])): row
        for _, row in pair_metadata_df.iterrows()
    }

    _log_step04_inputs(
        pair_pathway_rows=len(pair_pathway_df),
        positive_pairs=len(positive_pairs_df),
        self_loops=len(self_loops_df),
        reactome_rows=len(reactome_df),
        biogrid_nodes=biogrid_graph.number_of_nodes(),
        biogrid_edges=biogrid_graph.number_of_edges(),
        pathways_with_biogrid_overlap=len(pathway_to_proteins),
    )
    logger.info("Self-loops excluded from dataset generation: %s", len(self_loops_df))

    try:
        for index, row in pair_pathway_df.iterrows():
            processed_rows = index + 1
            subgraph_id = f"sg_{index:06d}"

            cached_manifest_row = manifest_cache.get(subgraph_id)
            if cached_manifest_row is not None:
                manifest_rows.append(cached_manifest_row)
                cache_hits += 1
                if processed_rows % STEP04_PROGRESS_EVERY == 0:
                    _log_step04_progress(
                        processed=processed_rows,
                        total=len(pair_pathway_df),
                        built=len(manifest_rows),
                        cached=cache_hits,
                        skipped=sum(skipped_rows.values()),
                    )
                continue

            cached_skip_status = skip_cache.get(subgraph_id)
            if cached_skip_status is not None:
                skipped_rows[cached_skip_status] = skipped_rows.get(cached_skip_status, 0) + 1
                cache_skip_hits += 1
                if processed_rows % STEP04_PROGRESS_EVERY == 0:
                    _log_step04_progress(
                        processed=processed_rows,
                        total=len(pair_pathway_df),
                        built=len(manifest_rows),
                        cached=cache_hits + cache_skip_hits,
                        skipped=sum(skipped_rows.values()),
                    )
                continue

            pathway_id = str(row["PathwayID"])
            drug_target = str(row["DrugTarget_UniProt"])
            disease_protein = str(row["DiseaseProtein_UniProt"])
            pathway_nodes = pathway_to_proteins.get(pathway_id, set())

            subgraph, build_info = build_sample_subgraph(
                row=row,
                biogrid_graph=biogrid_graph,
                pathway_nodes=pathway_nodes,
                include_bridge_nodes_for_extended=bool(config["include_bridge_nodes_for_extended"]),
                min_subgraph_nodes=int(config["min_subgraph_nodes"]),
                max_subgraph_nodes=int(config["max_subgraph_nodes"]),
                bridge_max_intermediate_nodes=int(config["bridge_max_intermediate_nodes"]),
            )
            if subgraph is None:
                skipped_rows[build_info["status"]] = skipped_rows.get(build_info["status"], 0) + 1
                if config["enable_cache"]:
                    skip_cache[subgraph_id] = build_info["status"]
                if processed_rows % STEP04_PROGRESS_EVERY == 0:
                    _log_step04_progress(
                        processed=processed_rows,
                        total=len(pair_pathway_df),
                        built=len(manifest_rows),
                        cached=cache_hits + cache_skip_hits,
                        skipped=sum(skipped_rows.values()),
                    )
                continue

            embedding_info = annotate_subgraph(
                subgraph,
                drug_target=drug_target,
                disease_protein=disease_protein,
                embeddings=embeddings,
                embedding_dim=embedding_dim,
                use_embeddings=bool(config["use_embeddings"]),
            )
            validate_subgraph(
                subgraph,
                drug_target=drug_target,
                disease_protein=disease_protein,
                embedding_dim=embedding_dim,
            )

            subgraph.graph.update(
                {
                    "subgraph_id": subgraph_id,
                    "pathway_id": pathway_id,
                    "pathway_name": str(row.get("PathwayName", "")),
                    "drug_target_uniprot": drug_target,
                    "disease_protein_uniprot": disease_protein,
                    "sensitivity_label": int(row.get("Label", 1)),
                    "experiment_type": str(row.get("ExperimentType", "")),
                    "bridge_needed": bool(row.get("BridgeNeeded", False)),
                    "bridge_nodes": build_info["bridge_nodes"],
                    "bridge_extended": bool(build_info["bridge_extended"]),
                    "bridge_path_nodes": build_info["bridge_path_nodes"],
                    "connectivity_extended": bool(build_info["connectivity_extended"]),
                    "connectivity_extension_nodes": build_info["connectivity_extension_nodes"],
                    "source": str(row.get("Source", "")),
                    "evidence_tier": str(row.get("EvidenceTier", "")),
                }
            )

            subgraph_path: Path | None = None
            if config["save_subgraphs"]:
                if subgraph_storage == "shards":
                    subgraph_path = shards_dir / f"subgraph_shard_{shard_index:05d}.pkl"
                    shard_graphs[subgraph_id] = subgraph
                else:
                    subgraph_path = subgraphs_dir / f"{subgraph_id}.pkl"
                    write_pickle(subgraph, subgraph_path)

            pair_metadata = metadata_lookup.get((drug_target, disease_protein))
            manifest_row = build_manifest_row(
                subgraph_id=subgraph_id,
                row=row,
                pair_metadata=pair_metadata,
                subgraph=subgraph,
                subgraph_path=subgraph_path,
                output_dir=output_dir,
                bridge_nodes=build_info["bridge_nodes"],
                bridge_path_nodes=build_info["bridge_path_nodes"],
                bridge_extended=bool(build_info["bridge_extended"]),
                connectivity_extension_nodes=build_info["connectivity_extension_nodes"],
                connectivity_extended=bool(build_info["connectivity_extended"]),
                embedding_info=embedding_info,
                pathway_nodes_count=len(pathway_nodes),
            )
            if config["save_subgraphs"] and subgraph_storage == "shards" and subgraph_path is not None:
                shard_manifest_path = _format_subgraph_path_for_manifest(subgraph_path, output_dir)
                manifest_row["SubgraphOriginalPath"] = ""
                manifest_row["SubgraphPath"] = shard_manifest_path
                manifest_row["SubgraphStorage"] = "pickle_shard"
                manifest_row["SubgraphShardPath"] = shard_manifest_path
                manifest_row["SubgraphShardKey"] = subgraph_id
                if len(shard_graphs) >= shard_size:
                    flush_shard()
            manifest_rows.append(manifest_row)
            if processed_rows % STEP04_PROGRESS_EVERY == 0:
                _log_step04_progress(
                    processed=processed_rows,
                    total=len(pair_pathway_df),
                    built=len(manifest_rows),
                    cached=cache_hits + cache_skip_hits,
                    skipped=sum(skipped_rows.values()),
                )
    finally:
        if embeddings_closer is not None:
            embeddings_closer()

    if config["save_subgraphs"] and subgraph_storage == "shards":
        flush_shard()

    manifest_df = pd.DataFrame(manifest_rows)
    if not manifest_df.empty:
        manifest_df = manifest_df.sort_values(["SubgraphID"], kind="stable").reset_index(drop=True)
    manifest_df.to_csv(manifest_path, sep="\t", index=False)
    logger.info("Manifest written: %s rows -> %s", len(manifest_df), manifest_path)
    if config["enable_cache"]:
        with skip_cache_path.open("w", encoding="utf-8") as handle:
            json.dump(skip_cache, handle, indent=2, sort_keys=True)
        logger.info("Skip cache written: %s", skip_cache_path)

    report = build_step04_report(
        processed_rows=len(pair_pathway_df),
        built_rows=len(manifest_df),
        skipped_rows=skipped_rows,
        manifest_df=manifest_df,
        embedding_dim=embedding_dim,
        use_embeddings=bool(config["use_embeddings"]),
        self_loops_rows=len(self_loops_df),
        cache_hits=cache_hits,
        cache_skip_hits=cache_skip_hits,
        self_loop_pathway_rows_excluded=self_loop_pathway_rows_excluded,
    )
    report["subgraph_storage"] = subgraph_storage
    if subgraph_storage == "shards":
        report["shard_size"] = int(shard_size)
        report["shards_dir"] = str(shards_dir)
        report["n_shards"] = int(len(shard_reports))
        report["shards"] = shard_reports
        report["total_shard_bytes"] = int(sum(item["bytes"] for item in shard_reports))
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Report written: %s", report_path)
    if config["save_subgraphs"]:
        logger.info("Subgraphs saved: %s", shards_dir if subgraph_storage == "shards" else subgraphs_dir)

    _log_step04_footer(
        built=len(manifest_df),
        skipped=sum(skipped_rows.values()),
        cached=cache_hits + cache_skip_hits,
        manifest_path=manifest_path,
        report_path=report_path,
        skipped_rows=skipped_rows,
    )
