from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from pathlib import Path

import pandas as pd

from api_clients.bulk_downloads import download_file
from api_clients.data_processing import (
    build_ppi_graph,
    build_biogrid_graph,
    build_symbol_to_uniprot_lookup,
    filter_biogrid_human_physical,
    filter_ctd_therapeutic,
    filter_disgenet_curated,
    load_drug_targets_fallback,
    load_tabular_file,
    map_disgenet_gene_ids_to_uniprot,
    parse_drugbank_targets_xml,
    stream_filter_tsv_gz,
)
from api_clients.protein_atlas import ProteinAtlasClient
from api_clients.repodb import normalize_repodb_positives
from api_clients.string_db import StringDBClient, ensure_string_aliases_file, load_string_aliases
from api_clients.uniprot import UniProtMappingClient
from drug_disease_validation.src.protein_resolution import resolve_symbols_with_protein_atlas
from drug_disease_validation.src.utils import load_dotenv_file

LOGGER = logging.getLogger(__name__)

CTD_URL = "https://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz"
DISGENET_URL = "https://www.disgenet.org/static/disgenet_ap1/files/downloads/curated_gene_disease_associations.tsv.gz"
BIOGRID_URL = "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-ALL-LATEST.tab3.zip"


class StaticMappingClient:
    def __init__(self, mapping_df: pd.DataFrame) -> None:
        self.mapping_df = mapping_df

    def map_ids(self, ids: list[str], from_db: str, to_db: str) -> pd.DataFrame:
        _ = (from_db, to_db)
        return self.mapping_df[self.mapping_df["from_id"].astype(str).isin(ids)].copy()


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    LOGGER.info("Wrote %s rows -> %s", len(df), path)


def _write_pickle(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    LOGGER.info("Wrote pickle -> %s", path)


def _write_report(path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(kwargs, handle, indent=2, default=str)
    LOGGER.info("Wrote report -> %s", path)


def _resolve(value: str | None) -> Path | None:
    return Path(value).expanduser().resolve() if value is not None else None


def validate_gzip_file(path: Path) -> None:
    with path.open("rb") as handle:
        magic = handle.read(2)
    if magic != b"\x1f\x8b":
        raise ValueError(
            f"Expected a gzipped file at {path}, but found a non-gzip payload instead."
        )


def build_disgenet_headers(api_key: str | None) -> dict[str, str] | None:
    token = (api_key or "").strip()
    if not token:
        return None
    authorization = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    return {
        "Authorization": authorization,
        "Accept": "application/octet-stream",
        "User-Agent": "digitalhealthlab-sensitive-ppin/0.1",
    }


def format_step01_summary(
    *,
    repodb_rows: int,
    ctd_rows: int | str,
    drug_target_rows: int,
    disgenet_rows: int,
    biogrid_nodes: int,
    biogrid_edges: int,
    overlap: dict[str, int],
    protein_atlas_summary: dict[str, int] | None = None,
    string_summary: dict[str, int | str] | None = None,
) -> str:
    summary = (
        "Step 1 summary:\n"
        f"- repoDB approved rows: {repodb_rows}\n"
        f"- CTD therapeutic rows: {ctd_rows}\n"
        f"- Drugs with targets rows: {drug_target_rows}\n"
        f"- DisGeNET disease-protein rows: {disgenet_rows}\n"
        f"- BioGRID nodes: {biogrid_nodes}\n"
        f"- BioGRID edges: {biogrid_edges}\n"
        f"- Drug targets in BioGRID: {overlap['drug_targets_in_biogrid']}/{overlap['drug_targets_total']}\n"
        f"- Disease proteins in BioGRID: {overlap['disease_proteins_in_biogrid']}/{overlap['disease_proteins_total']}"
    )
    if protein_atlas_summary is not None:
        summary += (
            "\n"
            f"- Protein Atlas total symbols: {protein_atlas_summary['total_symbols']}\n"
            f"- Protein Atlas filtered out: {protein_atlas_summary['filtered_out']}\n"
            f"- Protein Atlas loaded from cache: {protein_atlas_summary['loaded_from_cache']}\n"
            f"- Protein Atlas resolved live: {protein_atlas_summary['resolved_live']}\n"
            f"- Protein Atlas unresolved live: {protein_atlas_summary['unresolved_live']}"
        )
    if string_summary is not None:
        summary += (
            "\n"
            f"- STRING mapped seed proteins: {string_summary['mapped_seed_proteins']}/{string_summary['seed_proteins_total']}\n"
            f"- STRING edges: {string_summary['string_edges']}\n"
            f"- STRING nodes: {string_summary['string_nodes']}\n"
            f"- Union graph nodes: {string_summary['union_nodes']}\n"
            f"- Union graph edges: {string_summary['union_edges']}\n"
            f"- Drug targets in union graph: {string_summary['drug_targets_in_union_graph']}/{string_summary['drug_targets_total']}\n"
            f"- Disease proteins in union graph: {string_summary['disease_proteins_in_union_graph']}/{string_summary['disease_proteins_total']}"
        )
    return summary


def _ensure_file(
    local_path: Path | None,
    url: str,
    cache_dir: Path,
    filename: str,
    refresh: bool,
    headers: dict[str, str] | None = None,
) -> Path:
    expected_gzip = filename.endswith(".gz")

    if local_path is not None and local_path.exists():
        if expected_gzip:
            validate_gzip_file(local_path)
        LOGGER.info("Using local file %s", local_path)
        return local_path
    target = cache_dir / filename
    if target.exists() and not refresh:
        if expected_gzip:
            validate_gzip_file(target)
        LOGGER.info("Using cached download %s", target)
        return target
    downloaded = download_file(url, target, headers=headers)
    if expected_gzip:
        validate_gzip_file(downloaded)
    return downloaded


def _load_mapping_client(mapping_tsv: Path | None):
    if mapping_tsv is not None and mapping_tsv.exists():
        df = load_tabular_file(mapping_tsv)
        missing = {"from_id", "to_id"}.difference(df.columns)
        if missing:
            raise ValueError(f"Mapping TSV missing columns: {sorted(missing)}")
        return StaticMappingClient(df[["from_id", "to_id"]].drop_duplicates())
    return UniProtMappingClient(endpoint="https://rest.uniprot.org")


def prepare_repodb(repodb_source: Path, positive_status: list[str], out_dir: Path) -> pd.DataFrame:
    df = load_tabular_file(repodb_source)
    positives = normalize_repodb_positives(df, positive_status_values=positive_status)
    positives = positives.rename(
        columns={
            "drug_id": "DrugBankID",
            "drug_name": "DrugName",
            "disease_name": "DiseaseName",
            "disease_source_id": "DiseaseID",
            "label": "Label",
        }
    )
    _write_table(positives, out_dir / "repodb_approved.tsv")
    return positives


def prepare_ctd(ctd_source: Path, out_dir: Path) -> pd.DataFrame:
    filtered = stream_filter_tsv_gz(ctd_source, filter_ctd_therapeutic)
    _write_table(filtered, out_dir / "ctd_therapeutic.tsv")
    return filtered


def _normalize_drug_targets_ids(targets_df: pd.DataFrame, repodb_source: Path) -> pd.DataFrame:
    raw = load_tabular_file(repodb_source)
    name_col = next((c for c in raw.columns if c.lower() == "drug_name"), None)
    id_col = next((c for c in raw.columns if c.lower() == "drugbank_id"), None)
    if not name_col or not id_col:
        LOGGER.warning("repodb.csv missing drug_name/drugbank_id -- DrugBankID normalisation skipped.")
        return targets_df
    name_to_dbid = (
        raw[[name_col, id_col]]
        .dropna()
        .drop_duplicates(subset=[name_col])
        .assign(_key=lambda d: d[name_col].str.lower().str.strip())
        .set_index("_key")[id_col]
        .to_dict()
    )
    targets_df = targets_df.copy()
    targets_df["DrugBankID"] = (
        targets_df["DrugName"].str.lower().str.strip().map(name_to_dbid).fillna(targets_df["DrugBankID"])
    )
    mask = targets_df["DrugBankID"].astype(str).str.startswith("DB")
    return targets_df[mask].reset_index(drop=True)


def prepare_drug_targets(
    drugbank_xml: Path | None,
    fallback_targets: Path | None,
    repodb_source: Path | None,
    out_dir: Path,
) -> tuple[pd.DataFrame, str]:
    if drugbank_xml is not None and drugbank_xml.exists():
        targets_df = parse_drugbank_targets_xml(drugbank_xml)
        source_label = "drugbank_xml"
    elif fallback_targets is not None and fallback_targets.exists():
        targets_df = load_drug_targets_fallback(fallback_targets)
        source_label = "fallback_targets"
        LOGGER.warning("DrugBank XML not found -- using fallback %s", fallback_targets)
        if repodb_source is not None and repodb_source.exists():
            targets_df = _normalize_drug_targets_ids(targets_df, repodb_source)
            source_label = "fallback_targets+repodb_id_normalisation"
    else:
        raise FileNotFoundError(
            "Neither DrugBank XML nor fallback drug-target file is available. "
            "Provide --drugbank-xml or --drug-target-fallback with an existing file."
        )
    _write_table(targets_df, out_dir / "drugbank_targets.tsv")
    return targets_df, source_label


def prepare_disgenet(disgenet_source: Path, mapping_client, out_dir: Path) -> pd.DataFrame:
    filtered = stream_filter_tsv_gz(
        disgenet_source,
        lambda chunk: filter_disgenet_curated(chunk, score_min=0.3),
    )
    mapped = map_disgenet_gene_ids_to_uniprot(filtered, mapping_client)
    _write_table(mapped, out_dir / "disgenet_disease_proteins.tsv")
    return mapped


def prepare_biogrid(biogrid_source: Path, mapping_client, out_dir: Path):
    return _prepare_biogrid_internal(biogrid_source, mapping_client, out_dir, skip_symbol_lookup=False)


def _collect_biogrid_missing_symbols(raw_df: pd.DataFrame) -> list[str]:
    required_columns = {
        "Official Symbol Interactor A",
        "Official Symbol Interactor B",
        "SWISS-PROT Accessions Interactor A",
        "SWISS-PROT Accessions Interactor B",
    }
    missing = required_columns.difference(raw_df.columns)
    if missing:
        raise ValueError(f"BioGRID dataframe missing columns for fallback collection: {sorted(missing)}")

    def _extract_identifier(value: object) -> str:
        text = str(value).strip()
        if not text or text == "-" or text.lower() == "nan":
            return ""
        for delimiter in ("|", ";", ","):
            if delimiter in text:
                text = text.split(delimiter)[0].strip()
                break
        return text

    working = raw_df.copy()
    working["A_direct"] = working["SWISS-PROT Accessions Interactor A"].map(_extract_identifier)
    working["B_direct"] = working["SWISS-PROT Accessions Interactor B"].map(_extract_identifier)
    missing_rows = working[(working["A_direct"] == "") | (working["B_direct"] == "")]
    symbols = sorted(
        {
            str(symbol).strip()
            for column in ("Official Symbol Interactor A", "Official Symbol Interactor B")
            for symbol in missing_rows[column]
            if str(symbol).strip() and str(symbol).strip() != "-"
        }
    )
    return symbols


def _prepare_biogrid_internal(
    biogrid_source: Path,
    mapping_client,
    out_dir: Path,
    *,
    skip_symbol_lookup: bool,
    use_protein_atlas_fallback: bool = False,
    enable_secondary_uniprot_fallback: bool = False,
    protein_atlas_cache_file: Path | None = None,
):
    LOGGER.info("Loading BioGRID tabular source from %s", biogrid_source)
    raw_df = load_tabular_file(biogrid_source)
    LOGGER.info("Loaded BioGRID raw table with %d rows and %d columns", len(raw_df), len(raw_df.columns))
    symbol_lookup = None
    protein_atlas_summary: dict[str, int] | None = None
    if use_protein_atlas_fallback:
        cache_path = protein_atlas_cache_file or (out_dir / "biogrid_symbol_to_protein_atlas_cache.tsv")
        LOGGER.info("Building BioGRID symbol->protein mapping via Human Protein Atlas")
        symbols = _collect_biogrid_missing_symbols(raw_df)
        hpa_lookup, hpa_summary = resolve_symbols_with_protein_atlas(
            symbols,
            ProteinAtlasClient(),
            cache_path,
        )
        protein_atlas_summary = hpa_summary
        LOGGER.info(
            "Protein Atlas fallback summary: total=%d, cache=%d, resolved_live=%d, unresolved_live=%d",
            hpa_summary["total_symbols"],
            hpa_summary["loaded_from_cache"],
            hpa_summary["resolved_live"],
            hpa_summary["unresolved_live"],
        )
        symbol_lookup = hpa_lookup or None
        if enable_secondary_uniprot_fallback and not skip_symbol_lookup and not isinstance(mapping_client, StaticMappingClient):
            LOGGER.info("Running secondary UniProt fallback for symbols unresolved by Protein Atlas")
            try:
                secondary_lookup = build_symbol_to_uniprot_lookup(
                    raw_df,
                    mapping_client,
                    cache_tsv=out_dir / "biogrid_symbol_to_uniprot_cache.tsv",
                )
                merged = dict(secondary_lookup)
                merged.update(hpa_lookup)
                symbol_lookup = merged
            except Exception as exc:
                LOGGER.warning("Secondary UniProt fallback failed: %s", exc)
        elif enable_secondary_uniprot_fallback and skip_symbol_lookup:
            LOGGER.info("Secondary UniProt fallback explicitly skipped by flag")
    elif skip_symbol_lookup:
        LOGGER.info("Skipping BioGRID symbol->UniProt fallback lookup due to explicit flag")
    elif not isinstance(mapping_client, StaticMappingClient):
        try:
            LOGGER.info("Building BioGRID symbol->UniProt fallback lookup")
            symbol_lookup = build_symbol_to_uniprot_lookup(
                raw_df,
                mapping_client,
                cache_tsv=out_dir / "biogrid_symbol_to_uniprot_cache.tsv",
            )
        except Exception as exc:
            LOGGER.warning("BioGRID symbol fallback mapping failed: %s", exc)
    else:
        LOGGER.info("Skipping BioGRID symbol fallback lookup because a static mapping client is in use")
    LOGGER.info("Filtering BioGRID to human physical interactions")
    ppi_df = filter_biogrid_human_physical(raw_df, symbol_to_uniprot=symbol_lookup)
    LOGGER.info("Filtered BioGRID PPI table contains %d rows", len(ppi_df))
    LOGGER.info("Building NetworkX graph from filtered BioGRID PPI table")
    graph = build_biogrid_graph(ppi_df)
    LOGGER.info("Built BioGRID graph with %d nodes and %d edges", graph.number_of_nodes(), graph.number_of_edges())
    LOGGER.info("Writing BioGRID processed outputs")
    _write_table(ppi_df, out_dir / "biogrid_human_ppi.tsv")
    _write_pickle(graph, out_dir / "biogrid_graph.pkl")
    return ppi_df, graph, protein_atlas_summary


def prepare_biogrid(
    biogrid_source: Path,
    mapping_client,
    out_dir: Path,
    *,
    skip_symbol_lookup: bool = False,
    use_protein_atlas_fallback: bool = False,
    enable_secondary_uniprot_fallback: bool = False,
    protein_atlas_cache_file: Path | None = None,
):
    return _prepare_biogrid_internal(
        biogrid_source,
        mapping_client,
        out_dir,
        skip_symbol_lookup=skip_symbol_lookup,
        use_protein_atlas_fallback=use_protein_atlas_fallback,
        enable_secondary_uniprot_fallback=enable_secondary_uniprot_fallback,
        protein_atlas_cache_file=protein_atlas_cache_file,
    )


def _normalize_string_seed_ids(seed_ids: set[str]) -> list[str]:
    return sorted({str(item).strip() for item in seed_ids if str(item).strip()})


def _map_uniprot_to_string_ids(
    seed_ids: list[str],
    client: StringDBClient,
    *,
    batch_size: int = 1000,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for start in range(0, len(seed_ids), batch_size):
        batch = seed_ids[start : start + batch_size]
        batch_df = client.map_identifiers(batch, species=9606, limit=1, echo_query=1)
        if not batch_df.empty:
            frames.append(batch_df)
    if not frames:
        return pd.DataFrame()
    mapped = pd.concat(frames, ignore_index=True)
    required = {"queryItem", "stringId", "preferredName"}
    missing = required.difference(mapped.columns)
    if missing:
        raise ValueError(f"STRING identifier mapping missing columns: {sorted(missing)}")
    mapped = mapped.rename(columns={"queryItem": "SeedUniProtID", "preferredName": "PreferredName"})
    mapped["SeedUniProtID"] = mapped["SeedUniProtID"].astype(str).str.strip()
    mapped["stringId"] = mapped["stringId"].astype(str).str.strip()
    return mapped[["SeedUniProtID", "stringId", "PreferredName"]].drop_duplicates(subset=["SeedUniProtID"])


def _load_string_alias_lookup(aliases_path: Path) -> dict[str, str]:
    aliases_df = load_string_aliases(aliases_path)
    alias_lookup = (
        aliases_df[aliases_df["source"].astype(str).str.contains("UniProt", case=False, na=False)]
        .drop_duplicates(subset=["stringId"])
        .set_index("stringId")["alias"]
        .astype(str)
        .to_dict()
    )
    return alias_lookup


def prepare_string_graph(
    *,
    seed_ids: set[str],
    out_dir: Path,
    cache_dir: Path,
    refresh_downloads: bool,
    required_score: int,
    network_type: str,
    partner_limit: int,
    mapping_batch_size: int,
    interaction_batch_size: int,
) -> tuple[pd.DataFrame, object, dict[str, int]]:
    client = StringDBClient()
    normalized_seed_ids = _normalize_string_seed_ids(seed_ids)
    LOGGER.info("Preparing STRING graph from %d seed proteins", len(normalized_seed_ids))

    mapping_df = _map_uniprot_to_string_ids(normalized_seed_ids, client, batch_size=mapping_batch_size)
    _write_table(mapping_df, out_dir / "string_id_mapping.tsv")
    mapped_seed_ids = set(mapping_df["SeedUniProtID"].astype(str))
    string_ids = sorted(set(mapping_df["stringId"].astype(str)))
    LOGGER.info("Mapped %d/%d seed proteins to STRING identifiers", len(mapped_seed_ids), len(normalized_seed_ids))

    alias_lookup = _load_string_alias_lookup(ensure_string_aliases_file(cache_dir, refresh=refresh_downloads))
    string_to_seed = dict(zip(mapping_df["stringId"].astype(str), mapping_df["SeedUniProtID"].astype(str)))
    interaction_frames: list[pd.DataFrame] = []
    for start in range(0, len(string_ids), interaction_batch_size):
        batch = string_ids[start : start + interaction_batch_size]
        partners_df = client.interaction_partners(
            batch,
            species=9606,
            required_score=required_score,
            network_type=network_type,
            limit=partner_limit,
        )
        if not partners_df.empty:
            interaction_frames.append(partners_df)

    if interaction_frames:
        string_raw = pd.concat(interaction_frames, ignore_index=True)
    else:
        string_raw = pd.DataFrame()

    if not string_raw.empty:
        required = {"stringId_A", "stringId_B", "score", "preferredName_A", "preferredName_B"}
        missing = required.difference(string_raw.columns)
        if missing:
            raise ValueError(f"STRING interaction data missing columns: {sorted(missing)}")
        string_raw["ProteinA_UniProt"] = string_raw["stringId_A"].map(string_to_seed).fillna(
            string_raw["stringId_A"].map(alias_lookup)
        )
        string_raw["ProteinB_UniProt"] = string_raw["stringId_B"].map(string_to_seed).fillna(
            string_raw["stringId_B"].map(alias_lookup)
        )
        string_ppi_df = string_raw[
            ["ProteinA_UniProt", "ProteinB_UniProt", "score", "preferredName_A", "preferredName_B"]
        ].copy()
        string_ppi_df = string_ppi_df[
            (string_ppi_df["ProteinA_UniProt"].astype(str) != "")
            & (string_ppi_df["ProteinB_UniProt"].astype(str) != "")
        ]
        string_ppi_df = string_ppi_df[
            string_ppi_df["ProteinA_UniProt"].astype(str).isin(mapped_seed_ids)
            & string_ppi_df["ProteinB_UniProt"].astype(str).isin(mapped_seed_ids)
        ]
        string_ppi_df["Sources"] = "string"
        string_ppi_df["edge_key"] = string_ppi_df.apply(
            lambda row: tuple(sorted((str(row["ProteinA_UniProt"]), str(row["ProteinB_UniProt"])))),
            axis=1,
        )
        string_ppi_df = string_ppi_df.drop_duplicates(subset="edge_key").drop(columns="edge_key").reset_index(drop=True)
    else:
        string_ppi_df = pd.DataFrame(
            columns=["ProteinA_UniProt", "ProteinB_UniProt", "score", "preferredName_A", "preferredName_B", "Sources"]
        )

    string_graph = build_ppi_graph(string_ppi_df, source_column="Sources")
    _write_table(string_ppi_df, out_dir / "string_human_ppi.tsv")
    _write_pickle(string_graph, out_dir / "string_graph.pkl")

    summary = {
        "seed_proteins_total": len(normalized_seed_ids),
        "mapped_seed_proteins": len(mapped_seed_ids),
        "unmapped_seed_proteins": len(normalized_seed_ids) - len(mapped_seed_ids),
        "string_nodes": int(string_graph.number_of_nodes()),
        "string_edges": int(string_graph.number_of_edges()),
        "network_type": network_type,
        "required_score": int(required_score),
    }
    return string_ppi_df, string_graph, summary


def build_union_graph(
    biogrid_ppi_df: pd.DataFrame,
    string_ppi_df: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, object]:
    base = biogrid_ppi_df[["ProteinA_UniProt", "ProteinB_UniProt"]].copy()
    base["InBioGRID"] = 1
    base["InSTRING"] = 0
    if not string_ppi_df.empty:
        extra = string_ppi_df[["ProteinA_UniProt", "ProteinB_UniProt"]].copy()
        extra["InBioGRID"] = 0
        extra["InSTRING"] = 1
        combined = pd.concat([base, extra], ignore_index=True)
    else:
        combined = base
    combined["edge_key"] = combined.apply(
        lambda row: tuple(sorted((str(row["ProteinA_UniProt"]), str(row["ProteinB_UniProt"])))),
        axis=1,
    )
    combined["ProteinA_UniProt"] = combined["edge_key"].map(lambda pair: pair[0])
    combined["ProteinB_UniProt"] = combined["edge_key"].map(lambda pair: pair[1])
    union_df = (
        combined.groupby("edge_key", as_index=False)
        .agg(
            ProteinA_UniProt=("ProteinA_UniProt", "first"),
            ProteinB_UniProt=("ProteinB_UniProt", "first"),
            InBioGRID=("InBioGRID", "max"),
            InSTRING=("InSTRING", "max"),
        )
    )
    union_df["Sources"] = union_df.apply(
        lambda row: "biogrid|string" if row["InBioGRID"] and row["InSTRING"] else "biogrid" if row["InBioGRID"] else "string",
        axis=1,
    )
    union_df = union_df.drop(columns="edge_key")
    union_graph = build_ppi_graph(union_df, source_column="Sources")
    _write_table(union_df, out_dir / "ppi_union_edges.tsv")
    _write_table(union_df[["ProteinA_UniProt", "ProteinB_UniProt", "InBioGRID", "InSTRING", "Sources"]], out_dir / "ppi_edge_sources.tsv")
    _write_pickle(union_graph, out_dir / "ppi_union_graph.pkl")
    return union_df, union_graph


def build_step01_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    _ = default_repo_root
    parser = argparse.ArgumentParser(
        description="Download and prepare source data for Step 1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repodb-file", default="drug_disease_validation/data/raw/repodb.csv")
    parser.add_argument("--repodb-positive-status", nargs="+", default=["approved"])
    parser.add_argument("--ctd-file", default=None)
    parser.add_argument("--ctd-url", default=CTD_URL)
    parser.add_argument("--drugbank-xml", default="drug_disease_validation/data/raw/drugbank_full_database.xml")
    parser.add_argument("--drug-target-fallback", default=None)
    parser.add_argument("--disgenet-file", default=None)
    parser.add_argument("--disgenet-url", default=DISGENET_URL)
    parser.add_argument("--disgenet-mapping-tsv", default=None)
    parser.add_argument("--biogrid-file", default=None)
    parser.add_argument("--biogrid-url", default=BIOGRID_URL)
    parser.add_argument("--build-string-graph", action="store_true")
    parser.add_argument("--string-network-type", default="physical", choices=["physical", "functional"])
    parser.add_argument("--string-required-score", type=int, default=700)
    parser.add_argument("--string-partner-limit", type=int, default=50)
    parser.add_argument("--string-mapping-batch-size", type=int, default=1000)
    parser.add_argument("--string-interaction-batch-size", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        default="drug_disease_validation/data/processed",
    )
    parser.add_argument(
        "--cache-dir",
        default="drug_disease_validation/data/raw",
    )
    parser.add_argument("--refresh-downloads", action="store_true")
    parser.add_argument("--skip-ctd", action="store_true")
    parser.add_argument("--skip-biogrid-uniprot-fallback", action="store_true")
    parser.add_argument("--use-protein-atlas-fallback", action="store_true", default=True)
    parser.add_argument("--enable-secondary-uniprot-fallback", action="store_true")
    parser.add_argument("--protein-atlas-cache-file", default=None)
    parser.add_argument("--logging-level", default="INFO")
    return parser


def run_step01(args: argparse.Namespace) -> None:
    from drug_disease_validation.src.utils import configure_logging

    configure_logging(args.logging_level)
    load_dotenv_file(".env")

    out_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    repodb_source = _resolve(args.repodb_file)
    if repodb_source is None or not repodb_source.exists():
        raise FileNotFoundError(f"repoDB file not found: {repodb_source}")
    repodb_df = prepare_repodb(repodb_source, args.repodb_positive_status, out_dir)

    ctd_df: pd.DataFrame | None = None
    if not args.skip_ctd:
        ctd_source = _ensure_file(
            _resolve(args.ctd_file), args.ctd_url, cache_dir, "CTD_chemicals_diseases.tsv.gz", args.refresh_downloads
        )
        ctd_df = prepare_ctd(ctd_source, out_dir)

    disgenet_api_key = os.environ.get("DISGENET_API_KEY")
    disgenet_headers = build_disgenet_headers(disgenet_api_key)
    if not disgenet_api_key:
        LOGGER.warning("DISGENET_API_KEY not set -- unauthenticated download may fail.")
    try:
        disgenet_source = _ensure_file(
            _resolve(args.disgenet_file),
            args.disgenet_url,
            cache_dir,
            "curated_gene_disease_associations.tsv.gz",
            args.refresh_downloads,
            headers=disgenet_headers,
        )
    except ValueError as exc:
        raise RuntimeError(
            "DisGeNET download is not a valid gzip file. "
            "The server likely returned an HTML page instead of the dataset. "
            "Provide a valid local file with --disgenet-file or configure DISGENET_API_KEY."
        ) from exc

    mapping_client = _load_mapping_client(_resolve(args.disgenet_mapping_tsv))
    drug_targets_df, drug_target_source = prepare_drug_targets(
        _resolve(args.drugbank_xml),
        _resolve(args.drug_target_fallback),
        repodb_source,
        out_dir,
    )
    disgenet_df = prepare_disgenet(disgenet_source, mapping_client, out_dir)

    biogrid_source = _ensure_file(
        _resolve(args.biogrid_file), args.biogrid_url, cache_dir, "BIOGRID-ALL-LATEST.tab3.zip", args.refresh_downloads
    )
    biogrid_ppi_df, biogrid_graph, protein_atlas_summary = prepare_biogrid(
        biogrid_source,
        mapping_client,
        out_dir,
        skip_symbol_lookup=bool(args.skip_biogrid_uniprot_fallback),
        use_protein_atlas_fallback=bool(args.use_protein_atlas_fallback),
        enable_secondary_uniprot_fallback=bool(args.enable_secondary_uniprot_fallback),
        protein_atlas_cache_file=_resolve(args.protein_atlas_cache_file),
    )

    biogrid_nodes = set(biogrid_graph.nodes())
    drug_ids = set(drug_targets_df["TargetUniProtID"].astype(str))
    disease_ids = set(disgenet_df["UniProtID"].astype(str))
    overlap = {
        "drug_targets_in_biogrid": len(drug_ids & biogrid_nodes),
        "drug_targets_total": len(drug_ids),
        "disease_proteins_in_biogrid": len(disease_ids & biogrid_nodes),
        "disease_proteins_total": len(disease_ids),
    }

    string_summary: dict[str, int | str] | None = None
    if args.build_string_graph:
        string_ppi_df, string_graph, string_summary = prepare_string_graph(
            seed_ids=drug_ids | disease_ids,
            out_dir=out_dir,
            cache_dir=cache_dir,
            refresh_downloads=bool(args.refresh_downloads),
            required_score=int(args.string_required_score),
            network_type=str(args.string_network_type),
            partner_limit=int(args.string_partner_limit),
            mapping_batch_size=int(args.string_mapping_batch_size),
            interaction_batch_size=int(args.string_interaction_batch_size),
        )
        _, union_graph = build_union_graph(
            biogrid_ppi_df=biogrid_ppi_df,
            string_ppi_df=string_ppi_df,
            out_dir=out_dir,
        )
        union_nodes = set(union_graph.nodes())
        string_summary.update(
            {
                "union_nodes": int(union_graph.number_of_nodes()),
                "union_edges": int(union_graph.number_of_edges()),
                "drug_targets_in_union_graph": len(drug_ids & union_nodes),
                "drug_targets_total": len(drug_ids),
                "disease_proteins_in_union_graph": len(disease_ids & union_nodes),
                "disease_proteins_total": len(disease_ids),
            }
        )

    _write_report(
        out_dir / "step01_report.json",
        repodb_source=str(repodb_source),
        repodb_positive_status=args.repodb_positive_status,
        repodb_rows=int(len(repodb_df)),
        ctd_rows=int(len(ctd_df)) if ctd_df is not None else "skipped",
        drug_target_source=drug_target_source,
        drug_target_rows=int(len(drug_targets_df)),
        disgenet_rows=int(len(disgenet_df)),
        biogrid_nodes=int(biogrid_graph.number_of_nodes()),
        biogrid_edges=int(biogrid_graph.number_of_edges()),
        overlap=overlap,
        protein_atlas_summary=protein_atlas_summary,
        string_summary=string_summary,
    )
    LOGGER.info(
        "%s",
        format_step01_summary(
            repodb_rows=int(len(repodb_df)),
            ctd_rows=int(len(ctd_df)) if ctd_df is not None else "skipped",
            drug_target_rows=int(len(drug_targets_df)),
            disgenet_rows=int(len(disgenet_df)),
            biogrid_nodes=int(biogrid_graph.number_of_nodes()),
            biogrid_edges=int(biogrid_graph.number_of_edges()),
            overlap=overlap,
            protein_atlas_summary=protein_atlas_summary,
            string_summary=string_summary,
        ),
    )
