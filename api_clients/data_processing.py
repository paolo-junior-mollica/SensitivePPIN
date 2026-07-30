"""Data processing utilities for drug-disease source files.

Functions for loading, filtering and mapping tabular source data:
CTD, DisGeNET, DrugBank XML, BioGRID.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import xml.etree.ElementTree as ET
import zipfile
import gzip
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

DRUGBANK_NAMESPACE = {"db": "http://www.drugbank.ca"}
LOGGER = logging.getLogger(__name__)


def load_tabular_file(path: Path, *, comment: str | None = None) -> pd.DataFrame:
    """
    Loads a tabular dataset from various file formats into a pandas DataFrame.

    Handles different file types based on their extensions. It supports standard comma-separated (.csv)
    and tab-separated (.tsv, .txt, .tab, .tab3) files. Furthermore, it natively parses gzipped files (.gz)
    and zip archives (.zip). For zip archives, it automatically searches for and extracts the first valid
    tabular file contained within. Parsing is done with 'low_memory=False' to ensure robust type inference
    on large datasets.

    Args:
        path (Path): The file path or archive to be loaded.
        comment (str | None, keyword-only): An optional character indicating the start of a comment line
                                            (e.g., '#') to be ignored.

    Returns:
        pd.DataFrame: The parsed tabular data.

    Raises:
        ValueError: If a provided zip archive contains no files with recognized tabular extensions.
    """

    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-2:] == [".tsv", ".gz"] or path.suffix.lower() == ".gz":
        return pd.read_csv(path, sep="\t", compression="gzip", comment=comment, low_memory=False)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if name.lower().endswith((".tsv", ".txt", ".tab", ".tab3"))
            ]
            if not candidates:
                raise ValueError(f"No tabular file found in archive {path}")
            with archive.open(candidates[0]) as handle:
                return pd.read_csv(handle, sep="\t", comment=comment, low_memory=False)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt", ".tab", ".tab3"} else ","
    return pd.read_csv(path, sep=sep, comment=comment, low_memory=False)


def stream_filter_tsv_gz(
    path: Path,
    row_filter,
    *,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """
    Reads a gzipped TSV file in chunks, applying a filter function to each chunk.

    Designed to process large compressed datasets without exhausting system memory. Chunks are
    individually filtered and only the surviving rows are retained and concatenated into the
    final DataFrame.

    Args:
        path (Path): Path to the gzipped TSV file.
        row_filter (callable): A function that takes a DataFrame chunk and returns a filtered DataFrame.
        chunksize (int, keyword-only): Number of rows to read per chunk. Defaults to 50,000.

    Returns:
        pd.DataFrame: A consolidated DataFrame containing all rows that passed the filter.
    """

    header_fields: list[str] | None = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        waiting_for_fields_header = False
        for line in handle:
            if line.startswith("# Fields:"):
                header_candidate = line.split(":", 1)[1].strip()
                if header_candidate:
                    header_fields = [field.strip() for field in header_candidate.split("\t")]
                    break
                waiting_for_fields_header = True
                continue

            if waiting_for_fields_header and line.startswith("#"):
                header_candidate = line[1:].strip()
                if header_candidate:
                    header_fields = [field.strip() for field in header_candidate.split("\t")]
                    break

    read_csv_kwargs = {
        "sep": "\t",
        "compression": "gzip",
        "comment": "#",
        "chunksize": chunksize,
        "low_memory": False,
    }
    if header_fields:
        read_csv_kwargs["names"] = header_fields
        read_csv_kwargs["header"] = None

    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, **read_csv_kwargs):
        filtered = row_filter(chunk)
        if not filtered.empty:
            frames.append(filtered)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def filter_ctd_therapeutic(ctd_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the Comparative Toxicogenomics Database (CTD) for therapeutic relationships.

    Retains only the associations where the 'DirectEvidence' column explicitly indicates a 'therapeutic'
    role, standardizing the columns for downstream integration.

    Args:
        ctd_df (pd.DataFrame): The raw CTD dataframe.

    Returns:
        pd.DataFrame: A deduplicated dataframe containing chemical-disease therapeutic pairs.

    Raises:
        ValueError: If required columns are missing from the input dataframe.
    """

    required = {"ChemicalName", "ChemicalID", "DiseaseName", "DiseaseID", "DirectEvidence"}
    missing = required.difference(ctd_df.columns)
    if missing:
        raise ValueError(f"CTD file missing columns: {sorted(missing)}")
    therapeutic = ctd_df[ctd_df["DirectEvidence"].astype(str).str.lower() == "therapeutic"].copy()
    therapeutic = therapeutic[["ChemicalName", "ChemicalID", "DiseaseName", "DiseaseID"]].drop_duplicates()
    return therapeutic.reset_index(drop=True)


def _load_drugbank_root(path: Path) -> ET.Element:
    """
    Parses a DrugBank XML file and returns its root element.

    Can handle both uncompressed '.xml' files and '.zip' archives containing the XML.
    If a zip archive is provided, it automatically extracts and parses the first XML file found.

    Args:
        path (Path): Path to the XML file or ZIP archive.

    Returns:
        ET.Element: The root element of the parsed XML tree.

    Raises:
        ValueError: If no XML file is found inside the provided zip archive.
    """

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            xml_members = [name for name in archive.namelist() if name.lower().endswith(".xml")]
            if not xml_members:
                raise ValueError(f"No XML file found inside {path}")
            with archive.open(xml_members[0]) as handle:
                tree = ET.parse(handle)
                return tree.getroot()
    tree = ET.parse(path)
    return tree.getroot()


def _text(element: ET.Element | None) -> str:
    """
    Safely extracts and strips text from an XML element.

    Args:
        element (ET.Element | None): The XML element to extract text from.

    Returns:
        str: The stripped text content, or an empty string if the element or text is None.
    """

    if element is None or element.text is None:
        return ""
    return element.text.strip()


def parse_drugbank_targets_xml(path: Path) -> pd.DataFrame:
    """
    Parses the official DrugBank XML database to extract drug-target associations.

    Iterates through the XML tree to identify drugs and their respective polypeptide targets.
    Filters the targets to include only human organisms and targets with a 'known action'.

    Args:
        path (Path): Path to the DrugBank XML file or ZIP archive.

    Returns:
        pd.DataFrame: A dataframe linking DrugBankIDs to TargetUniProtIDs and their actions.
    """

    root = _load_drugbank_root(path)
    rows: list[dict[str, str]] = []
    accepted_organisms = {"homo sapiens", "human", "humans"}
    for drug in root.findall("db:drug", DRUGBANK_NAMESPACE):
        primary_id = drug.find('db:drugbank-id[@primary="true"]', DRUGBANK_NAMESPACE)
        drug_id = _text(primary_id)
        drug_name = _text(drug.find("db:name", DRUGBANK_NAMESPACE))
        targets = drug.find("db:targets", DRUGBANK_NAMESPACE)
        if not drug_id or not drug_name or targets is None:
            continue
        for target in targets.findall("db:target", DRUGBANK_NAMESPACE):
            organism = _text(target.find("db:organism", DRUGBANK_NAMESPACE)).lower()
            known_action = _text(target.find("db:known-action", DRUGBANK_NAMESPACE)).lower()
            polypeptide = target.find("db:polypeptide", DRUGBANK_NAMESPACE)
            target_id = polypeptide.get("id") if polypeptide is not None else ""
            actions = [
                _text(action)
                for action in target.findall("db:actions/db:action", DRUGBANK_NAMESPACE)
                if _text(action)
            ]
            if organism not in accepted_organisms:
                continue
            if known_action != "yes":
                continue
            if not target_id:
                continue
            rows.append(
                {
                    "DrugBankID": drug_id,
                    "DrugName": drug_name,
                    "TargetUniProtID": target_id,
                    "Action": "|".join(sorted(set(actions))) if actions else "",
                }
            )
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def load_drug_targets_fallback(path: Path) -> pd.DataFrame:
    """
    Loads and standardizes a tabular fallback file for drug targets.

    Used when the primary DrugBank XML is unavailable. Maps columns related to drug IDs,
    target IDs, and actions, standardizing them into the internal pipeline format.

    Args:
        path (Path): Path to the fallback tabular file.

    Returns:
        pd.DataFrame: A standardized dataframe of drug targets.

    Raises:
        ValueError: If required mapping columns are missing from the fallback file.
    """

    df = load_tabular_file(path)
    required = {"drug_id", "drug_name", "target_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Fallback target file missing columns: {sorted(missing)}")
    action_column = (
        "action_type" if "action_type" in df.columns
        else "mechanism_of_action" if "mechanism_of_action" in df.columns
        else None
    )
    out = pd.DataFrame(
        {
            "DrugBankID": df["drug_id"].astype(str).str.strip(),
            "DrugName": df["drug_name"].astype(str).str.strip(),
            "TargetUniProtID": df["target_id"].astype(str).str.strip(),
            "Action": df[action_column].astype(str).str.strip() if action_column else "",
        }
    )
    return out[(out["DrugBankID"] != "") & (out["TargetUniProtID"] != "")].drop_duplicates().reset_index(drop=True)


def filter_disgenet_curated(disgenet_df: pd.DataFrame, score_min: float = 0.3) -> pd.DataFrame:
    """
    Filters DisGeNET associations for curated entries meeting a confidence threshold.

    Retains rows designated as 'CURATED' and strictly equal to or above the minimum
    confidence score. Also cleans disease IDs by removing 'umls:' prefixes.

    Args:
        disgenet_df (pd.DataFrame): The raw DisGeNET dataframe.
        score_min (float): The minimum acceptable association score. Defaults to 0.3.

    Returns:
        pd.DataFrame: The filtered and standardized DisGeNET dataframe.

    Raises:
        ValueError: If required columns are missing from the input dataframe.
    """

    required = {"diseaseName", "diseaseId", "geneSymbol", "geneId", "score", "source"}
    missing = required.difference(disgenet_df.columns)
    if missing:
        raise ValueError(f"DisGeNET file missing columns: {sorted(missing)}")
    df = disgenet_df.copy()
    scores = pd.to_numeric(df["score"], errors="coerce")
    curated_mask = df["source"].astype(str).str.upper().str.contains("CURATED", regex=False, na=False)
    if curated_mask.sum() == 0:
        curated_mask = pd.Series(True, index=df.index)
    filtered = df[curated_mask & scores.ge(score_min)].copy()
    filtered["Score"] = pd.to_numeric(filtered["score"], errors="coerce")
    disease_ids = (
        filtered["diseaseId"].astype(str).str.strip()
        .str.replace(r"(?i)^umls:", "", regex=True)
    )
    out = pd.DataFrame(
        {
            "DiseaseName": filtered["diseaseName"].astype(str).str.strip(),
            "DiseaseID": disease_ids,
            "GeneSymbol": filtered["geneSymbol"].astype(str).str.strip(),
            "geneId": filtered["geneId"].astype(str).str.strip(),
            "Score": filtered["Score"],
        }
    )
    return out.drop_duplicates().reset_index(drop=True)


def map_disgenet_gene_ids_to_uniprot(
    disgenet_df: pd.DataFrame,
    mapping_client: Any,
    *,
    from_db: str = "GeneID",
    to_db: str = "UniProtKB",
) -> pd.DataFrame:
    """
    Translates Gene IDs in the DisGeNET dataframe to UniProt accessions.

    Utilizes the provided mapping client (static or API) to translate identifiers.
    Rows with multiple UniProt mappings are exploded into separate records.

    Args:
        disgenet_df (pd.DataFrame): Filtered DisGeNET dataframe containing 'geneId'.
        mapping_client (Any): An instantiated client capable of mapping IDs.
        from_db (str, keyword-only): Source database name. Defaults to "GeneID".
        to_db (str, keyword-only): Target database name. Defaults to "UniProtKB".

    Returns:
        pd.DataFrame: A dataframe with added 'UniProtID' mappings.
    """

    gene_ids = sorted({str(value).strip() for value in disgenet_df["geneId"] if str(value).strip()})
    mapping_df = mapping_client.map_ids(gene_ids, from_db=from_db, to_db=to_db)
    if not mapping_df.empty:
        mapping_df = mapping_df.copy()
        mapping_df["from_id"] = mapping_df["from_id"].astype(str)
        mapping_df["to_id"] = mapping_df["to_id"].astype(str)
    lookup = mapping_df.groupby("from_id")["to_id"].agg(list).to_dict()
    rows: list[dict[str, object]] = []
    for row in disgenet_df.itertuples(index=False):
        mapped_ids = lookup.get(str(row.geneId), [])
        for accession in mapped_ids:
            rows.append(
                {
                    "DiseaseName": row.DiseaseName,
                    "DiseaseID": row.DiseaseID,
                    "GeneSymbol": row.GeneSymbol,
                    "UniProtID": str(accession),
                    "Score": float(row.Score),
                }
            )
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def build_symbol_to_uniprot_lookup(
    biogrid_df: pd.DataFrame,
    mapping_client: Any,
    *,
    from_db: str = "Gene_Name",
    to_db: str = "UniProtKB",
    batch_size: int = 500,
    cache_tsv: Path | None = None,
    max_workers: int = 4,
    min_split_batch_size: int = 1,
) -> dict[str, str]:
    """
    Creates a dictionary mapping official gene symbols to UniProt accessions.

    Extracts all unique gene symbols from the BioGRID dataframe and queries the
    mapping client. This acts as a fallback dictionary to resolve BioGRID interactions
    that lack direct SWISS-PROT accessions.

    Args:
        biogrid_df (pd.DataFrame): The BioGRID dataframe containing interactors.
        mapping_client (Any): An instantiated client capable of mapping IDs.
        from_db (str, keyword-only): Source database name. Defaults to "Gene_Name".
        to_db (str, keyword-only): Target database name. Defaults to "UniProtKB".

    Returns:
        dict[str, str]: A dictionary translating gene symbols to UniProt IDs.
    """

    required_columns = {
        "Official Symbol Interactor A",
        "Official Symbol Interactor B",
        "SWISS-PROT Accessions Interactor A",
        "SWISS-PROT Accessions Interactor B",
    }
    missing_columns = required_columns.difference(biogrid_df.columns)
    if missing_columns:
        raise ValueError(f"BioGRID dataframe missing columns for symbol fallback lookup: {sorted(missing_columns)}")

    working = biogrid_df.copy()
    working["A_direct"] = working["SWISS-PROT Accessions Interactor A"].map(_extract_identifier)
    working["B_direct"] = working["SWISS-PROT Accessions Interactor B"].map(_extract_identifier)
    missing_accession_rows = working[(working["A_direct"] == "") | (working["B_direct"] == "")]

    symbols = sorted(
        {
            str(symbol).strip()
            for column in ("Official Symbol Interactor A", "Official Symbol Interactor B")
            for symbol in missing_accession_rows[column]
            if str(symbol).strip() and str(symbol).strip() != "-"
        }
    )
    cached_mapping_df = pd.DataFrame(columns=["from_id", "to_id"])
    if cache_tsv is not None and cache_tsv.exists():
        cached_mapping_df = load_tabular_file(cache_tsv)
        required = {"from_id", "to_id"}
        missing = required.difference(cached_mapping_df.columns)
        if missing:
            raise ValueError(f"Symbol mapping cache missing columns: {sorted(missing)}")
        cached_mapping_df = cached_mapping_df[["from_id", "to_id"]].drop_duplicates()

    cached_symbols = set(cached_mapping_df["from_id"].astype(str)) if not cached_mapping_df.empty else set()
    symbols_to_query = [symbol for symbol in symbols if symbol not in cached_symbols]
    total_batches = (len(symbols_to_query) + batch_size - 1) // batch_size if symbols_to_query else 0
    LOGGER.info(
        "BioGRID fallback summary: %d unique symbols need fallback, %d loaded from cache, %d will be queried via UniProt in %d batches (batch_size=%d, max_workers=%d).",
        len(symbols),
        len(cached_symbols),
        len(symbols_to_query),
        total_batches,
        batch_size,
        max_workers,
    )

    mapping_frames: list[pd.DataFrame] = []
    if not cached_mapping_df.empty:
        mapping_frames.append(cached_mapping_df)
    failed_batches = 0

    batches = [
        (start, symbols_to_query[start : start + batch_size])
        for start in range(0, len(symbols_to_query), batch_size)
    ]

    def _persist_progress() -> None:
        if cache_tsv is not None:
            cache_tsv.parent.mkdir(parents=True, exist_ok=True)
            progressive_df = pd.concat(mapping_frames, ignore_index=True)
            progressive_df = progressive_df[["from_id", "to_id"]].drop_duplicates(
                subset=["from_id"], keep="first"
            )
            progressive_df.to_csv(cache_tsv, sep="\t", index=False)

    def _map_batch(payload: tuple[int, list[str]]):
        start, batch = payload
        batch_df = mapping_client.map_ids(batch, from_db=from_db, to_db=to_db)
        return start, batch, batch_df

    def _map_batch_with_split(start: int, batch: list[str]) -> list[pd.DataFrame]:
        nonlocal failed_batches
        try:
            _, _, batch_df = _map_batch((start, batch))
            return [batch_df] if not batch_df.empty else []
        except Exception as exc:
            failed_batches += 1
            LOGGER.warning(
                "UniProt symbol mapping batch failed for %d symbols starting at index %d: %s",
                len(batch),
                start,
                exc,
            )
            if len(batch) <= max(1, min_split_batch_size):
                return []
            mid = len(batch) // 2
            left = _map_batch_with_split(start, batch[:mid])
            right = _map_batch_with_split(start + mid, batch[mid:])
            return left + right

    if batches:
        worker_count = max(1, min(max_workers, len(batches)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(_map_batch, payload): payload for payload in batches}
            for future in as_completed(future_map):
                start, batch = future_map[future]
                try:
                    _, _, batch_df = future.result()
                except Exception as exc:
                    recovered = _map_batch_with_split(start, batch)
                    if recovered:
                        mapping_frames.extend(recovered)
                        _persist_progress()
                    continue
                if not batch_df.empty:
                    mapping_frames.append(batch_df)
                    _persist_progress()

    if not mapping_frames:
        if failed_batches:
            raise RuntimeError("All UniProt symbol mapping batches failed")
        return {}

    mapping_df = pd.concat(mapping_frames, ignore_index=True)
    mapping_df = mapping_df.drop_duplicates(subset=["from_id"], keep="first")
    lookup = dict(zip(mapping_df["from_id"].astype(str), mapping_df["to_id"].astype(str)))
    LOGGER.info(
        "BioGRID symbol fallback mapping resolved %d/%d symbols across %d batches (%d failed batches, %d loaded from cache).",
        len(lookup),
        len(symbols),
        total_batches,
        failed_batches,
        len(cached_symbols),
    )
    return lookup


def _extract_identifier(value: object) -> str:
    """
    Cleans and extracts the primary identifier from a delimited string.

    BioGRID frequently groups multiple accessions using delimiters like '|', ';', or ','.
    This function splits the string and returns only the first valid identifier.

    Args:
        value (object): The raw identifier string.

    Returns:
        str: The isolated primary identifier, or an empty string if invalid.
    """

    text = str(value).strip()
    if not text or text == "-" or text.lower() == "nan":
        return ""
    for delimiter in ("|", ";", ","):
        if delimiter in text:
            text = text.split(delimiter)[0].strip()
            break
    return text


def filter_biogrid_human_physical(
    biogrid_df: pd.DataFrame,
    symbol_to_uniprot: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Filters the BioGRID dataset for physical interactions between human proteins.

    Restricts edges to Homo Sapiens (taxID: 9606) and physical experimental setups.
    It leverages direct SWISS-PROT accessions when available, falling back to the
    provided symbol lookup dictionary to resolve missing IDs.

    Args:
        biogrid_df (pd.DataFrame): The raw BioGRID tabular dataframe.
        symbol_to_uniprot (dict[str, str] | None): Fallback dictionary for symbol resolution.

    Returns:
        pd.DataFrame: A deduplicated dataframe representing edges between two UniProt IDs.

    Raises:
        ValueError: If required columns are missing from the input dataframe.
    """

    required = {
        "Organism ID Interactor A",
        "Organism ID Interactor B",
        "Experimental System Type",
        "SWISS-PROT Accessions Interactor A",
        "SWISS-PROT Accessions Interactor B",
        "Official Symbol Interactor A",
        "Official Symbol Interactor B",
    }
    missing = required.difference(biogrid_df.columns)
    if missing:
        raise ValueError(f"BioGRID file missing columns: {sorted(missing)}")
    df = biogrid_df.copy()
    organism_a = pd.to_numeric(df["Organism ID Interactor A"], errors="coerce")
    organism_b = pd.to_numeric(df["Organism ID Interactor B"], errors="coerce")
    physical_mask = df["Experimental System Type"].astype(str).str.lower().str.contains("physical", na=False)
    human_mask = organism_a.eq(9606) & organism_b.eq(9606)
    filtered = df[human_mask & physical_mask].copy()

    def resolve_accession(accession_value: object, symbol_value: object) -> str:
        accession = _extract_identifier(accession_value)
        if accession:
            return accession
        symbol = str(symbol_value).strip()
        if symbol_to_uniprot is None or not symbol or symbol == "-":
            return ""
        return symbol_to_uniprot.get(symbol, "")

    filtered["ProteinA_UniProt"] = [
        resolve_accession(a, s)
        for a, s in zip(
            filtered["SWISS-PROT Accessions Interactor A"],
            filtered["Official Symbol Interactor A"],
        )
    ]
    filtered["ProteinB_UniProt"] = [
        resolve_accession(a, s)
        for a, s in zip(
            filtered["SWISS-PROT Accessions Interactor B"],
            filtered["Official Symbol Interactor B"],
        )
    ]
    out = filtered[["ProteinA_UniProt", "ProteinB_UniProt"]].copy()
    out = out[(out["ProteinA_UniProt"] != "") & (out["ProteinB_UniProt"] != "")]
    out = out[out["ProteinA_UniProt"] != out["ProteinB_UniProt"]]
    out["edge_key"] = out.apply(
        lambda row: tuple(sorted((row["ProteinA_UniProt"], row["ProteinB_UniProt"]))),
        axis=1,
    )
    out = out.drop_duplicates(subset="edge_key").drop(columns="edge_key")
    return out.reset_index(drop=True)


def build_biogrid_graph(ppi_df: pd.DataFrame) -> nx.Graph:
    """
    Constructs a NetworkX graph from a protein-protein interaction dataframe.

    Builds an undirected graph using the interacting UniProt IDs as nodes.
    Self-loops (proteins interacting with themselves) are explicitly removed
    to prepare the graph for network topology analysis.

    Args:
        ppi_df (pd.DataFrame): Dataframe containing 'ProteinA_UniProt' and 'ProteinB_UniProt'.

    Returns:
        nx.Graph: The undirected interaction network.
    """

    return build_ppi_graph(ppi_df)


def build_ppi_graph(ppi_df: pd.DataFrame, *, source_column: str | None = None) -> nx.Graph:
    """Build an undirected PPI graph, optionally preserving an edge source attribute."""

    graph = nx.Graph()
    if source_column and source_column in ppi_df.columns:
        for row in ppi_df[["ProteinA_UniProt", "ProteinB_UniProt", source_column]].itertuples(index=False, name=None):
            a, b, source_value = row
            if a == b:
                continue
            graph.add_edge(a, b, source=source_value)
    else:
        graph.add_edges_from(ppi_df[["ProteinA_UniProt", "ProteinB_UniProt"]].itertuples(index=False, name=None))
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph
