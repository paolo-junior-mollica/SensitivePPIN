from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import networkx as nx
import pandas as pd

from api_clients.data_processing import load_tabular_file
from drug_disease_validation.src.utils import configure_logging, read_pickle


logger = logging.getLogger(__name__)


def canonicalize_disease_id(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    if ":" not in text and re.fullmatch(r"C\d+", text):
        return f"UMLS:{text}"
    return text


def normalize_disease_name(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def summarize_series(values: pd.Series) -> dict[str, float | int]:
    if values.empty:
        return {"count": 0, "min": 0, "median": 0, "p90": 0, "max": 0}
    numeric = values.astype(int)
    return {
        "count": int(len(numeric)),
        "min": int(numeric.min()),
        "median": float(numeric.median()),
        "p90": float(numeric.quantile(0.9)),
        "max": int(numeric.max()),
    }


def load_json_report(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else None


def format_metric_table(title: str, rows: list[tuple[str, object]]) -> str:
    label_width = max(len(label) for label, _ in rows)
    value_width = max(len(str(value)) for _, value in rows)
    border = f"+-{'-' * label_width}-+-{'-' * value_width}-+"
    body = "\n".join(
        f"| {label.ljust(label_width)} | {str(value).rjust(value_width)} |"
        for label, value in rows
    )
    return f"{title}\n{border}\n{body}\n{border}"


def build_final_metrics_log(
    *,
    step01_report: dict[str, object] | None,
    step02_report: dict[str, object],
) -> str:
    sections: list[str] = []

    step02_rows = [
        ("input CTD", step02_report["ctd_rows"]),
        ("input repoDB", step02_report["repodb_rows"]),
        ("CTD senza mapping DrugBank", step02_report["unmapped_ctd_rows"]),
        ("malattie risolte via direct_id", step02_report["resolved_by_direct_id"]),
        ("malattie risolte via name_fallback", step02_report["resolved_by_name_fallback"]),
        ("malattie non risolte", step02_report["unmatched_diseases"]),
        ("coppie candidate pre-BioGRID", step02_report["candidate_pairs_before_biogrid"]),
        ("coppie mantenute post-BioGRID", step02_report["candidate_pairs_after_biogrid"]),
        ("self-loops filtrati", step02_report["self_loops"]),
    ]
    sections.append(format_metric_table("Final Step02 Metrics", step02_rows))

    sections.append(
        format_metric_table(
            "Step02 Source Breakdown",
            [
                ("mapping rows ctd", step02_report["source_mapping_rows"]["ctd"]),
                ("mapping rows repodb", step02_report["source_mapping_rows"]["repodb"]),
                ("mapping rows ctd|repodb", step02_report["source_mapping_rows"]["ctd|repodb"]),
                ("positive pairs ctd", step02_report["source_positive_pairs"]["ctd"]),
                ("positive pairs repodb", step02_report["source_positive_pairs"]["repodb"]),
                ("positive pairs ctd|repodb", step02_report["source_positive_pairs"]["ctd|repodb"]),
                ("self-loops ctd", step02_report["source_self_loops"]["ctd"]),
                ("self-loops repodb", step02_report["source_self_loops"]["repodb"]),
                ("self-loops ctd|repodb", step02_report["source_self_loops"]["ctd|repodb"]),
            ],
        )
    )

    if step01_report is not None:
        overlap = step01_report.get("overlap")
        string_summary = step01_report.get("string_summary")
        step01_rows: list[tuple[str, object]] = [
            ("repoDB rows", step01_report.get("repodb_rows", "n/a")),
            ("CTD rows", step01_report.get("ctd_rows", "n/a")),
            ("drug target rows", step01_report.get("drug_target_rows", "n/a")),
            ("DisGeNET rows", step01_report.get("disgenet_rows", "n/a")),
            ("BioGRID nodes", step01_report.get("biogrid_nodes", "n/a")),
            ("BioGRID edges", step01_report.get("biogrid_edges", "n/a")),
        ]
        if isinstance(overlap, dict):
            step01_rows.extend(
                [
                    ("target in BioGRID", overlap.get("drug_targets_in_biogrid", "n/a")),
                    ("target totali", overlap.get("drug_targets_total", "n/a")),
                    ("disease proteins in BioGRID", overlap.get("disease_proteins_in_biogrid", "n/a")),
                    ("disease proteins totali", overlap.get("disease_proteins_total", "n/a")),
                ]
            )
        if isinstance(string_summary, dict):
            step01_rows.extend(
                [
                    ("STRING network type", string_summary.get("network_type", "n/a")),
                    ("STRING required score", string_summary.get("required_score", "n/a")),
                    ("STRING nodes", string_summary.get("string_nodes", "n/a")),
                    ("STRING edges", string_summary.get("string_edges", "n/a")),
                    ("union nodes", string_summary.get("union_nodes", "n/a")),
                    ("union edges", string_summary.get("union_edges", "n/a")),
                ]
            )
        sections.append(format_metric_table("Step01 Context", step01_rows))

    return "\n".join(sections)


def count_sources(frame: pd.DataFrame) -> dict[str, int]:
    base = {"ctd": 0, "repodb": 0, "ctd|repodb": 0}
    if "Source" not in frame.columns:
        return base
    observed = frame["Source"].value_counts(dropna=False).to_dict()
    base.update({str(key): int(value) for key, value in observed.items()})
    return base


def resolve_ctd_disease_matches(
    mapping_df: pd.DataFrame,
    disease_proteins_df: pd.DataFrame,
    enable_name_fallback: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    resolved_df = mapping_df.copy()
    resolved_df["CanonicalDiseaseID"] = resolved_df["DiseaseID"].map(canonicalize_disease_id)
    resolved_df["NormalizedDiseaseName"] = resolved_df["DiseaseName"].map(normalize_disease_name)
    resolved_df["MappedDiseaseID"] = ""
    resolved_df["DiseaseMatchSource"] = "unmatched"

    direct_id_matches = resolved_df["CanonicalDiseaseID"].isin(disease_proteins_df["CanonicalDiseaseID"])
    resolved_df.loc[direct_id_matches, "MappedDiseaseID"] = resolved_df.loc[direct_id_matches, "CanonicalDiseaseID"]
    resolved_df.loc[direct_id_matches, "DiseaseMatchSource"] = "direct_id"

    if enable_name_fallback:
        unresolved_mask = resolved_df["DiseaseMatchSource"] == "unmatched"
        name_matches = resolved_df.loc[unresolved_mask, "NormalizedDiseaseName"].isin(disease_proteins_df["NormalizedDiseaseName"])
        unresolved_index = resolved_df.loc[unresolved_mask].index[name_matches]
        resolved_df.loc[unresolved_index, "DiseaseMatchSource"] = "name_fallback"

    stats = {
        "resolved_by_direct_id": int((resolved_df["DiseaseMatchSource"] == "direct_id").sum()),
        "resolved_by_name_fallback": int((resolved_df["DiseaseMatchSource"] == "name_fallback").sum()),
        "unmatched_diseases": int((resolved_df["DiseaseMatchSource"] == "unmatched").sum()),
    }
    return resolved_df, stats


def build_ctd_to_drugbank_mapping(
    ctd_df: pd.DataFrame,
    drugbank_targets_df: pd.DataFrame,
) -> pd.DataFrame:
    disease_name_column = "DiseaseName" if "DiseaseName" in ctd_df.columns else "disease_name"
    disease_id_column = "DiseaseID" if "DiseaseID" in ctd_df.columns else "disease_id"
    drug_lookup = (
        drugbank_targets_df[["DrugBankID", "DrugName"]]
        .dropna()
        .drop_duplicates()
        .assign(_key=lambda d: d["DrugName"].astype(str).str.lower().str.strip())
    )
    mapping = (
        ctd_df[["ChemicalName", "ChemicalID", disease_name_column, disease_id_column]]
        .dropna()
        .drop_duplicates()
        .assign(_key=lambda d: d["ChemicalName"].astype(str).str.lower().str.strip())
        .merge(drug_lookup[["_key", "DrugBankID", "DrugName"]], on="_key", how="left")
        .drop(columns="_key")
        .rename(
            columns={
                disease_name_column: "DiseaseName",
                disease_id_column: "DiseaseID",
            }
        )
    )
    return mapping


def build_repodb_mapping(repodb_df: pd.DataFrame) -> pd.DataFrame:
    expected_columns = ["DrugBankID", "DrugName", "DiseaseName", "DiseaseID"]
    mapping = (
        repodb_df[expected_columns]
        .dropna(subset=["DrugBankID", "DrugName", "DiseaseName", "DiseaseID"])
        .drop_duplicates()
        .copy()
    )
    mapping["ChemicalName"] = mapping["DrugName"]
    mapping["ChemicalID"] = ""
    return mapping[["ChemicalName", "ChemicalID", "DrugName", "DrugBankID", "DiseaseName", "DiseaseID"]]


def combine_source_mappings(*frames: pd.DataFrame) -> pd.DataFrame:
    non_empty = [frame.copy() for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(
            columns=[
                "ChemicalName",
                "ChemicalID",
                "DrugName",
                "DrugBankID",
                "DiseaseName",
                "DiseaseID",
                "Source",
                "EvidenceTier",
            ]
        )
    combined = pd.concat(non_empty, ignore_index=True)
    group_columns = ["DrugName", "DrugBankID", "DiseaseName", "DiseaseID"]
    aggregated = (
        combined.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            ChemicalName=("ChemicalName", lambda values: next((str(v) for v in values if str(v).strip()), "")),
            ChemicalID=("ChemicalID", lambda values: next((str(v) for v in values if str(v).strip()), "")),
            Source=("Source", lambda values: "|".join(sorted({str(value) for value in values if str(value)}))),
        )
    )
    aggregated["EvidenceTier"] = aggregated["Source"].map(
        lambda value: "high" if "repodb" in str(value).split("|") else "expanded"
    )
    return aggregated[
        ["ChemicalName", "ChemicalID", "DrugName", "DrugBankID", "DiseaseName", "DiseaseID", "Source", "EvidenceTier"]
    ]


def build_positive_pairs(
    *,
    ctd_df: pd.DataFrame,
    repodb_df: pd.DataFrame | None,
    drugbank_targets_df: pd.DataFrame,
    disease_proteins_df: pd.DataFrame,
    biogrid_graph: nx.Graph,
    enable_name_fallback: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, int]]:
    ctd_mapping_df = build_ctd_to_drugbank_mapping(ctd_df, drugbank_targets_df)
    ctd_mapping_df["Source"] = "ctd"
    repodb_mapping_df = pd.DataFrame(columns=ctd_mapping_df.columns)
    if repodb_df is not None and not repodb_df.empty:
        repodb_mapping_df = build_repodb_mapping(repodb_df)
        repodb_mapping_df["Source"] = "repodb"
    mapping_df = combine_source_mappings(ctd_mapping_df, repodb_mapping_df)
    disease_proteins_df = disease_proteins_df.copy()
    disease_proteins_df["CanonicalDiseaseID"] = disease_proteins_df["DiseaseID"].map(canonicalize_disease_id)
    disease_proteins_df["NormalizedDiseaseName"] = disease_proteins_df["DiseaseName"].map(normalize_disease_name)
    mapping_df, disease_match_stats = resolve_ctd_disease_matches(
        mapping_df=mapping_df,
        disease_proteins_df=disease_proteins_df,
        enable_name_fallback=enable_name_fallback,
    )

    rows: list[dict[str, object]] = []
    self_loop_rows: list[dict[str, object]] = []
    biogrid_nodes = set(biogrid_graph.nodes())
    disease_proteins_by_id = {
        key: group.copy()
        for key, group in disease_proteins_df.groupby("CanonicalDiseaseID", dropna=False)
        if key
    }
    disease_proteins_by_name = {
        key: group.copy()
        for key, group in disease_proteins_df.groupby("NormalizedDiseaseName", dropna=False)
        if key
    }
    loss_stats = {
        "ctd_rows_input": int(len(ctd_df)),
        "repodb_rows_input": int(len(repodb_df)) if repodb_df is not None else 0,
        "ctd_rows_total": int(len(mapping_df)),
        "ctd_rows_with_drugbank_mapping": int(mapping_df["DrugBankID"].notna().sum()),
        "ctd_rows_without_drugbank_mapping": int(mapping_df["DrugBankID"].isna().sum()),
        "ctd_rows_with_disease_match": int((mapping_df["DiseaseMatchSource"] != "unmatched").sum()),
        "ctd_rows_without_disease_match": int((mapping_df["DiseaseMatchSource"] == "unmatched").sum()),
        "drug_targets_total_considered": 0,
        "drug_targets_in_biogrid": 0,
        "drug_targets_outside_biogrid": 0,
        "disease_proteins_total_considered": 0,
        "disease_proteins_in_biogrid": 0,
        "disease_proteins_outside_biogrid": 0,
        "candidate_pairs_before_biogrid_filter": 0,
        "candidate_pairs_after_biogrid_filter": 0,
        "self_loops_filtered": 0,
    }

    for association in mapping_df.itertuples(index=False):
        if pd.isna(association.DrugBankID):
            continue

        drug_targets = drugbank_targets_df[drugbank_targets_df["DrugBankID"] == association.DrugBankID]
        if association.DiseaseMatchSource == "name_fallback":
            disease_proteins = disease_proteins_by_name.get(association.NormalizedDiseaseName, disease_proteins_df.iloc[0:0])
        else:
            disease_proteins = disease_proteins_by_id.get(association.MappedDiseaseID, disease_proteins_df.iloc[0:0])

        drug_target_mask = drug_targets["TargetUniProtID"].astype(str).isin(biogrid_nodes)
        disease_protein_mask = disease_proteins["UniProtID"].astype(str).isin(biogrid_nodes)
        loss_stats["drug_targets_total_considered"] += int(len(drug_targets))
        loss_stats["drug_targets_in_biogrid"] += int(drug_target_mask.sum())
        loss_stats["drug_targets_outside_biogrid"] += int(len(drug_targets)) - int(drug_target_mask.sum())
        loss_stats["disease_proteins_total_considered"] += int(len(disease_proteins))
        loss_stats["disease_proteins_in_biogrid"] += int(disease_protein_mask.sum())
        loss_stats["disease_proteins_outside_biogrid"] += int(len(disease_proteins)) - int(disease_protein_mask.sum())
        loss_stats["candidate_pairs_before_biogrid_filter"] += int(len(drug_targets) * len(disease_proteins))

        for target in drug_targets.itertuples(index=False):
            target_id = str(target.TargetUniProtID)
            for protein in disease_proteins.itertuples(index=False):
                disease_id = str(protein.UniProtID)
                if target_id not in biogrid_nodes or disease_id not in biogrid_nodes:
                    continue
                loss_stats["candidate_pairs_after_biogrid_filter"] += 1
                row = {
                    "DrugName": association.DrugName if pd.notna(association.DrugName) else association.ChemicalName,
                    "DrugBankID": association.DrugBankID,
                    "DiseaseName": association.DiseaseName,
                    "DiseaseID": association.DiseaseID,
                    "MatchedDiseaseID": association.MappedDiseaseID,
                    "DiseaseMatchSource": association.DiseaseMatchSource,
                    "Source": association.Source,
                    "EvidenceTier": association.EvidenceTier,
                    "DrugTarget_UniProt": target_id,
                    "DiseaseProtein_UniProt": disease_id,
                    "Label": 1,
                }
                if target_id == disease_id:
                    self_loop_rows.append(row)
                    loss_stats["self_loops_filtered"] += 1
                else:
                    rows.append(row)

    positive_pairs = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    self_loops = pd.DataFrame(self_loop_rows).drop_duplicates().reset_index(drop=True)
    return positive_pairs, self_loops, mapping_df, disease_match_stats, loss_stats


def summarize_step02_outputs(
    *,
    positive_pairs: pd.DataFrame,
    self_loops: pd.DataFrame,
    mapping_df: pd.DataFrame,
    ctd_rows: int,
    repodb_rows: int,
    candidate_pairs_before_biogrid: int,
    candidate_pairs_after_biogrid: int,
    disease_match_stats: dict[str, int],
    loss_stats: dict[str, int],
) -> tuple[dict[str, int], pd.DataFrame, pd.DataFrame]:
    mapped_ctd_rows = int(mapping_df["DrugBankID"].notna().sum()) if "DrugBankID" in mapping_df.columns else 0
    unmapped_ctd_rows = int(len(mapping_df) - mapped_ctd_rows)

    if positive_pairs.empty:
        pairs_per_drug = pd.DataFrame(columns=["DrugBankID", "pair_count"])
        pairs_per_disease = pd.DataFrame(columns=["DiseaseID", "pair_count"])
    else:
        pairs_per_drug = (
            positive_pairs.groupby("DrugBankID", dropna=False)
            .size()
            .reset_index(name="pair_count")
            .sort_values(["pair_count", "DrugBankID"], ascending=[False, True], kind="stable")
            .reset_index(drop=True)
        )
        pairs_per_disease = (
            positive_pairs.groupby("DiseaseID", dropna=False)
            .size()
            .reset_index(name="pair_count")
            .sort_values(["pair_count", "DiseaseID"], ascending=[False, True], kind="stable")
            .reset_index(drop=True)
        )

    report = {
        "ctd_rows": int(ctd_rows),
        "repodb_rows": int(repodb_rows),
        "mapped_ctd_rows": mapped_ctd_rows,
        "unmapped_ctd_rows": unmapped_ctd_rows,
        "candidate_pairs_before_biogrid": int(candidate_pairs_before_biogrid),
        "candidate_pairs_after_biogrid": int(candidate_pairs_after_biogrid),
        "positive_pairs": int(len(positive_pairs)),
        "self_loops": int(len(self_loops)),
        "total_pairs_including_self_loops": int(len(positive_pairs) + len(self_loops)),
        "unique_drugs": int(positive_pairs["DrugBankID"].nunique()) if not positive_pairs.empty else 0,
        "unique_diseases": int(positive_pairs["DiseaseID"].nunique()) if not positive_pairs.empty else 0,
        "unique_drug_targets": int(positive_pairs["DrugTarget_UniProt"].nunique()) if not positive_pairs.empty else 0,
        "unique_disease_proteins": int(positive_pairs["DiseaseProtein_UniProt"].nunique()) if not positive_pairs.empty else 0,
        "pairs_per_drug_summary": summarize_series(pairs_per_drug["pair_count"]) if not pairs_per_drug.empty else summarize_series(pd.Series(dtype="int64")),
        "pairs_per_disease_summary": summarize_series(pairs_per_disease["pair_count"]) if not pairs_per_disease.empty else summarize_series(pd.Series(dtype="int64")),
        "source_mapping_rows": count_sources(mapping_df),
        "source_positive_pairs": count_sources(positive_pairs),
        "source_self_loops": count_sources(self_loops),
        "loss_funnel": loss_stats,
    }
    report.update(disease_match_stats)
    return report, pairs_per_drug, pairs_per_disease


def build_step02_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build positive drug-target/disease-protein pairs from processed Step 1 outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed_dir = default_repo_root / "drug_disease_validation" / "data" / "processed"
    parser.add_argument("--ctd-therapeutic", default=str(processed_dir / "ctd_therapeutic.tsv"))
    parser.add_argument("--repodb-approved", default=str(processed_dir / "repodb_approved.tsv"))
    parser.add_argument("--drugbank-targets", default=str(processed_dir / "drugbank_targets.tsv"))
    parser.add_argument("--disgenet-disease-proteins", default=str(processed_dir / "disgenet_disease_proteins.tsv"))
    parser.add_argument("--biogrid-graph", default=str(processed_dir / "biogrid_graph.pkl"))
    parser.add_argument("--step01-report", default=str(processed_dir / "step01_report.json"))
    parser.add_argument("--output-dir", default=str(processed_dir))
    parser.add_argument("--disable-disease-name-fallback", action="store_true")
    parser.add_argument("--logging-level", default="INFO")
    return parser


def run_step02(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting step02 positive pair building")
    logger.debug("Arguments: %s", vars(args))
    logger.info("Output directory: %s", output_dir)
    logger.info("Disease matching fallback on name: %s", not args.disable_disease_name_fallback)

    ctd_df = load_tabular_file(Path(args.ctd_therapeutic))
    repodb_df = load_tabular_file(Path(args.repodb_approved))
    drugbank_targets_df = load_tabular_file(Path(args.drugbank_targets))
    disease_proteins_df = load_tabular_file(Path(args.disgenet_disease_proteins))
    biogrid_graph = read_pickle(Path(args.biogrid_graph))
    step01_report = load_json_report(Path(args.step01_report))

    logger.info(
        "Loaded inputs: ctd=%s rows, repodb=%s rows, drugbank_targets=%s rows, disease_proteins=%s rows",
        len(ctd_df),
        len(repodb_df),
        len(drugbank_targets_df),
        len(disease_proteins_df),
    )
    logger.info(
        "Loaded BioGRID graph: nodes=%s edges=%s",
        biogrid_graph.number_of_nodes(),
        biogrid_graph.number_of_edges(),
    )

    positive_pairs, self_loops, mapping_df, disease_match_stats, loss_stats = build_positive_pairs(
        ctd_df=ctd_df,
        repodb_df=repodb_df,
        drugbank_targets_df=drugbank_targets_df,
        disease_proteins_df=disease_proteins_df,
        biogrid_graph=biogrid_graph,
        enable_name_fallback=not args.disable_disease_name_fallback,
    )

    logger.info(
        "Built outputs: mapping=%s rows, positive_pairs=%s rows, self_loops=%s rows",
        len(mapping_df),
        len(positive_pairs),
        len(self_loops),
    )
    logger.info(
        "Disease matching summary: direct_id=%s, name_fallback=%s, unmatched=%s",
        disease_match_stats["resolved_by_direct_id"],
        disease_match_stats["resolved_by_name_fallback"],
        disease_match_stats["unmatched_diseases"],
    )
    logger.info(
        "Loss funnel: no_drugbank=%s, no_disease_match=%s, pairs_pre_biogrid=%s, pairs_post_biogrid=%s, self_loops=%s",
        loss_stats["ctd_rows_without_drugbank_mapping"],
        loss_stats["ctd_rows_without_disease_match"],
        loss_stats["candidate_pairs_before_biogrid_filter"],
        loss_stats["candidate_pairs_after_biogrid_filter"],
        loss_stats["self_loops_filtered"],
    )

    positive_pairs.to_csv(output_dir / "positive_pairs.tsv", sep="\t", index=False)
    self_loops.to_csv(output_dir / "positive_pairs_self_loops.tsv", sep="\t", index=False)
    mapping_df.to_csv(output_dir / "ctd_to_drugbank_mapping.tsv", sep="\t", index=False)
    logger.info("Wrote output tables to %s", output_dir)

    report, pairs_per_drug, pairs_per_disease = summarize_step02_outputs(
        positive_pairs=positive_pairs,
        self_loops=self_loops,
        mapping_df=mapping_df,
        ctd_rows=len(ctd_df),
        repodb_rows=len(repodb_df),
        candidate_pairs_before_biogrid=loss_stats["candidate_pairs_before_biogrid_filter"],
        candidate_pairs_after_biogrid=loss_stats["candidate_pairs_after_biogrid_filter"],
        disease_match_stats=disease_match_stats,
        loss_stats=loss_stats,
    )
    pairs_per_drug.to_csv(output_dir / "pairs_per_drug.tsv", sep="\t", index=False)
    pairs_per_disease.to_csv(output_dir / "pairs_per_disease.tsv", sep="\t", index=False)
    with (output_dir / "step02_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Wrote report: %s", output_dir / "step02_report.json")
    logger.info(
        "Final pipeline metrics\n%s",
        build_final_metrics_log(step01_report=step01_report, step02_report=report),
    )
