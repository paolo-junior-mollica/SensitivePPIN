import argparse
import json
import logging
from pathlib import Path
import pandas as pd

import networkx as nx
import numpy as np

import requests
import time

from api_clients.data_processing import load_tabular_file
from drug_disease_validation.src.utils import configure_logging, load_yaml_file
from drug_disease_validation.src.schemas import CANONICAL_FILENAMES, CANONICAL_COLUMNS

logger = logging.getLogger(__name__)
PATHWAY_SIZE_CACHE = {}
PATHWAY_PROTEINS_CACHE = {}
step06_PROGRESS_EVERY = 5_000

DEFAULT_step06_CONFIG = {
    "max_size_pathway": 50,
    "min_size_pathway": 5,
    "max_try_search": 3,
    "reactome_api_rate_limit_per_sec": 10.0,
    "negative_proportion": 3,
    "random_state": 42,
}

_MIN_INTERVAL_BETWEEN_CALLS = 0.1  # 10 requests/sec max
_LAST_API_CALL_TS = 0.0


def _throttle_api_call() -> None:
    """Ensure minimum interval between Reactome API calls (<=10 req/sec)."""
    global _LAST_API_CALL_TS
    elapsed = time.monotonic() - _LAST_API_CALL_TS
    if elapsed < _MIN_INTERVAL_BETWEEN_CALLS:
        time.sleep(_MIN_INTERVAL_BETWEEN_CALLS - elapsed)
    _LAST_API_CALL_TS = time.monotonic()


def _get_step06_defaults(default_repo_root: Path) -> dict:
    config_path = default_repo_root / "drug_disease_validation" / "config" / "step06.yml"
    config = DEFAULT_step06_CONFIG.copy()
    loaded = load_yaml_file(config_path)
    step06_cfg = loaded.get("step06", {})
    if step06_cfg and not isinstance(step06_cfg, dict):
        raise ValueError(f"Expected mapping under 'step06' in config: {config_path}")
    config.update(step06_cfg)
    return config


def _resolve_step06_config(args: argparse.Namespace, default_repo_root: Path) -> dict:
    config_path = Path(args.config) if args.config else (
        default_repo_root / "drug_disease_validation" / "config" / "step06.yml"
    )
    config = DEFAULT_step06_CONFIG.copy()
    loaded = load_yaml_file(config_path)
    step06_cfg = loaded.get("step06", {})
    if step06_cfg and not isinstance(step06_cfg, dict):
        raise ValueError(f"Expected mapping under 'step06' in config: {config_path}")
    config.update(step06_cfg)

    overrides = {
        "max_size_pathway": args.max_size_pathway,
        "min_size_pathway": args.min_size_pathway,
        "max_try_search": args.max_try_search,
        "reactome_api_rate_limit_per_sec": args.reactome_api_rate_limit_per_sec,
        "negative_proportion": args.negative_proportion,
        "random_state": args.random_state,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    config["config_path"] = config_path
    config["max_size_pathway"] = int(config["max_size_pathway"])
    config["min_size_pathway"] = int(config["min_size_pathway"])
    config["max_try_search"] = int(config["max_try_search"])
    config["reactome_api_rate_limit_per_sec"] = float(config["reactome_api_rate_limit_per_sec"])
    config["negative_proportion"] = int(config["negative_proportion"])
    config["random_state"] = int(config["random_state"])
    if config["reactome_api_rate_limit_per_sec"] <= 0:
        raise ValueError("reactome_api_rate_limit_per_sec must be > 0")
    if config["min_size_pathway"] < 1:
        raise ValueError("min_size_pathway must be >= 1")
    if config["max_size_pathway"] < config["min_size_pathway"]:
        raise ValueError("max_size_pathway must be >= min_size_pathway")
    return config


def _configure_api_rate_limit(rate_limit_per_sec: float) -> None:
    global _MIN_INTERVAL_BETWEEN_CALLS
    _MIN_INTERVAL_BETWEEN_CALLS = 1.0 / rate_limit_per_sec


def _log_step06_header(*, config: dict, output_dir: Path, reactome_path: Path) -> None:
    logger.info("step06 started")
    logger.info(
        "Config: max_size=%s | min_size=%s | max_try=%s | reactome_rate_limit=%s | negative_proportion=%s | random_state=%s",
        config["max_size_pathway"],
        config["min_size_pathway"],
        config["max_try_search"],
        config["reactome_api_rate_limit_per_sec"],
        config["negative_proportion"],
        config["random_state"],
    )
    logger.info("Output: %s", output_dir)
    logger.info("Reactome map: %s", reactome_path)


def _log_step06_inputs(*, positive_pairs: int, unique_positive_pairs: int, full_disease_pool: int, negative_pairs: int, unique_negative_pairs: int, biogrid_rows: int) -> None:
    logger.info(
        "Inputs: positive_pairs=%s | unique_positive_pairs=%s | full_disease_pool=%s | negative_pairs=%s | unique_negative_pairs=%s | biogrid_rows=%s",
        positive_pairs,
        unique_positive_pairs,
        full_disease_pool,
        negative_pairs,
        unique_negative_pairs,
        biogrid_rows,
    )


def _log_step06_progress(*, label: str, processed: int, total: int, results: int) -> None:
    pct = (100.0 * processed / total) if total else 100.0
    logger.info("%s progress: %s/%s pairs (%.1f%%) | pathway_rows=%s", label, processed, total, pct, results)


def _log_step06_footer(*, pair_rows: int, report_path: Path) -> None:
    logger.info("step06 finished")
    logger.info("pair_pathway_mapping rows: %s", pair_rows)
    logger.info("Report: %s", report_path)

def _merge_source_tokens(values) -> str:
    tokens: set[str] = set()
    for raw in values:
        if raw is None:
            continue
        for tok in str(raw).split("|"):
            tok = tok.strip()
            if tok:
                tokens.add(tok)
    return "|".join(sorted(tokens))


def _merge_evidence_tier(values) -> str:
    """`high` wins over `expanded` (step02 semantics: repoDB-backed pairs are high tier)."""
    tiers = {str(v).strip() for v in values if pd.notna(v) and str(v).strip()}

    if "high" in tiers:
        return "high"
    if "expanded" in tiers:
        return "expanded"
    if "none" in tiers:
        return "none"

    return "unknown"


def aggregate_pair_sources(positive_pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse positive_pairs rows to unique (drug_target, disease_protein) pairs,
    merging Source (union) and EvidenceTier (high > expanded) across (drug, disease)
    combinations that share the same protein pair."""
    columns = CANONICAL_COLUMNS["mapping_drug_diseases_uniprot"]
    if positive_pairs_df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS["pair_source_aggregation"])

    frame = positive_pairs_df.copy()
    if "Source" not in frame.columns:
        frame["Source"] = ""
    if "EvidenceTier" not in frame.columns:
        frame["EvidenceTier"] = "expanded"

    return (
        frame.groupby(columns, dropna=False, as_index=False)
        .agg(
            Source=("Source", _merge_source_tokens),
            EvidenceTier=("EvidenceTier", _merge_evidence_tier),
        )
    )


def load_biogrid_graph(biogrid_dataframe : pd.DataFrame) -> nx.Graph:
    """Create a NetworkX graph from BioGRID interactions."""
    G = nx.Graph()
    for row in biogrid_dataframe.itertuples(index=False):
        G.add_edge(row.ProteinA_UniProt, row.ProteinB_UniProt)
    return G

def check_bridge_1_hop(missing_protein: str, pathway_proteins: set, bg_graph: nx.Graph) -> bool:
    """
    Check whether missing_protein is reachable in <= 1 hop 
    from any of the proteins present in the pathway.
    """
    if missing_protein not in bg_graph:
        return False
        
    neighbors = set(bg_graph.neighbors(missing_protein))
    
    if neighbors.intersection(pathway_proteins):
        return True
        
    return False

def get_proteins_in_pathway(pw_id: str, reactome_dict: dict)-> set:
    """
    Returns a set containing all the proteins (UniProt IDs) in a given pathway
    by iterating through the UniProt -> Pathways dictionary.
    """
    if pw_id in PATHWAY_PROTEINS_CACHE:
        return PATHWAY_PROTEINS_CACHE[pw_id]

    proteins_in_pw = set()
    for uniprot, pathways in reactome_dict.items():
        if pw_id in pathways:
            proteins_in_pw.add(uniprot)

    PATHWAY_PROTEINS_CACHE[pw_id] = proteins_in_pw
            
    return proteins_in_pw

def get_pathway_size(pathway_id: str, max_retries : int) -> int:
    """
    Query Reactome to obtain the number of participants (nodes) in a pathway.
    Returns the number of nodes or -1 if an error occurs.
    """
    if pathway_id in PATHWAY_SIZE_CACHE:
        return PATHWAY_SIZE_CACHE[pathway_id]

    url = f"https://reactome.org/ContentService/data/participants/{pathway_id}"
    headers = {"accept": "application/json"}
    
    for _ in range(max_retries):
        try:
            _throttle_api_call()
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                size = 0
                for set in data:
                    for entity in set["refEntities"]:
                        if entity["schemaClass"] == "ReferenceGeneProduct":
                            size += 1
                PATHWAY_SIZE_CACHE[pathway_id] = size
                # La lunghezza della lista rappresenta il numero di entità (nodi) nel pathway
                return size
            else:
                logger.error(f" [!] API Error for {pathway_id}: Status {response.status_code}, Retrying...")
                return -1
        except Exception as e:
            logger.error(f"  [!] Connection error for {pathway_id}: {e}, Retrying...")
    logger.info(f"  [!] {pathway_id} has failed permanently after {max_retries} attempts.")
    return -1
    
def fetch_pathways_from_api(uniprot_id: str, max_retries: int) -> dict:
    """Use the Reactome APIs as a fallback if the local cache fails."""
    url = f"https://reactome.org/ContentService/data/mapping/UniProt/{uniprot_id}/pathways"
    headers = {"accept": "application/json"}
    
    for _ in range(max_retries):
        try:
            _throttle_api_call()
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                logger.info(f"  [API Fallback] Pathways have been found online for {uniprot_id}!")
                return {pw['stId']: pw['displayName'] for pw in data if pw['speciesName'] == "Homo sapiens"}
                
            elif response.status_code == 404:
                return {}
                
            else:
                logger.warning(f"  [API Fallback] Error {response.status_code} for {uniprot_id}. Retrying...")
                
        except Exception as e:
            logger.warning(f"  [API Fallback] Connection error for {uniprot_id}: {e}. Retrying...")
            
        time.sleep(0.1) # Pausa prima del retry
        
    logger.error(f"  [API Fallback] {uniprot_id} has failed permanently after {max_retries} attempts.")
    return {}

def get_pathways_smart(prot_id: str, reactome_dict: dict, max_try_search: int) -> dict:
    """Search the local cache. If nothing is found, use the API and update the cache."""
    if prot_id not in reactome_dict:
        logger.debug(f"Cache miss for {prot_id}. Searching online...")
        pathways = fetch_pathways_from_api(prot_id, max_try_search)
        reactome_dict[prot_id] = pathways
        return pathways
        
    return reactome_dict[prot_id]

def find_shared_pathways(
    drug_prot: str, 
    disease_prot: str, 
    max_size: int, 
    min_size: int,
    reactome_dict: dict, 
    loss_stats: dict[str, int],
    max_try_search: int,
) -> dict:
    """Find the shared pathways"""
    
    pathways_drug = get_pathways_smart(drug_prot, reactome_dict, max_try_search)
    

    if not pathways_drug:
        loss_stats["single_drug_protein_not_found"] += 1

    pathways_disease = get_pathways_smart(disease_prot, reactome_dict, max_try_search)

    if not pathways_disease:
        loss_stats["single_disease_protein_not_found"] += 1
    if not pathways_drug or not pathways_disease:
        return {}
    shared_ids = set(pathways_drug.keys()).intersection(set(pathways_disease.keys()))

    shared_pathways: dict = {}

    for pw_id in shared_ids:
        # Use bulk-file-derived size consistently with Search B (same source
        # avoids A/B divergence caused by dataset drift between API and bulk).
        pathways_size = len(get_proteins_in_pathway(pw_id, reactome_dict))

        if pathways_size > max_size:
            logger.debug("Pathway ID: %s exceeds max size %s. Size: %s.", pw_id, max_size, pathways_size)
            loss_stats["experiment_type"]["pathways_with_exceeds_size_in_reactome"] += 1
        elif pathways_size <= 0:
            logger.debug("Pathway ID: %s not found in bulk reactome map.", pw_id)
        elif pathways_size < min_size:
            logger.debug("Pathway ID: %s too small.", pw_id)
            loss_stats["experiment_type"]["pathways_below_min_size"] += 1
        else:
            shared_pathways[pw_id] = {
                "name": pathways_drug[pw_id],
                "size": pathways_size
            }
    return shared_pathways

def get_pathways_results(
    pairs_df: pd.DataFrame, 
    max_size: int, 
    min_size: int,
    reactome_dict: dict, 
    loss_stats: dict[str, int], 
    max_try_search: int,
    biogrid_graph : nx.Graph
) -> list:
    results = []
    pathway_sizes: list[int] = []
    has_source_cols = {"Source", "EvidenceTier"}.issubset(pairs_df.columns)
    label = "pairs"
    for idx, row in enumerate(pairs_df.itertuples(index=False), start=1):
        drug_p = row.DrugTarget_UniProt
        disease_p = row.DiseaseProtein_UniProt
        source = getattr(row, "Source", "") if has_source_cols else ""
        evidence_tier = getattr(row, "EvidenceTier", "expanded") if has_source_cols else "expanded"

        logger.debug(f"Analysing the pair: Drug({drug_p}) - Disease({disease_p})")

        has_clean = False
        has_extended = False

        shared = find_shared_pathways(
            drug_p,
            disease_p,
            max_size,
            min_size,
            reactome_dict,
            loss_stats,
            max_try_search
        )

        if shared:
            loss_stats["experiment_type"]["total_pathways_searched"] += len(shared)
            loss_stats["experiment_type"]["pathways_found_clean"] += len(shared)
            has_clean = True
            for stId, info in shared.items():
                pathway_sizes.append(int(info["size"]))
                results.append({
                    'PathwayID': stId,
                    'PathwayName': info["name"],
                    'DrugTarget_UniProt': drug_p,
                    'DiseaseProtein_UniProt': disease_p,
                    'NumParticipants': info["size"],
                    'ExperimentType': "clean",
                    'BridgeNeeded': False,
                    'Source': source,
                    'EvidenceTier': evidence_tier,
                    'Label': 0
                })
        else:
            # --- SEARCH B (Extended): union drug ∪ disease pathways ---
            drug_pathways = get_pathways_smart(drug_p, reactome_dict, max_try_search)
            disease_pathways = get_pathways_smart(disease_p, reactome_dict, max_try_search)
            union_pathways = {**drug_pathways, **disease_pathways}
            loss_stats["experiment_type"]["total_pathways_searched"] += len(union_pathways)

            for pw_id, pw_name in union_pathways.items():
                pw_proteins = get_proteins_in_pathway(pw_id, reactome_dict)
                drug_in = drug_p in pw_proteins
                disease_in = disease_p in pw_proteins

                # Skip pathways where both proteins are already present:
                # Search A would have caught them (or they failed the size filter there).
                if drug_in and disease_in:
                    continue

                missing_protein = disease_p if drug_in else drug_p
                is_bridged = check_bridge_1_hop(missing_protein, pw_proteins, biogrid_graph)

                if is_bridged:
                    pw_size = len(pw_proteins)
                    if pw_size > max_size:
                        loss_stats["experiment_type"]["pathways_with_exceeds_size_in_total"] += 1
                        continue
                    if pw_size < min_size:
                        loss_stats["experiment_type"]["pathways_below_min_size"] += 1
                        continue

                    loss_stats["experiment_type"]["pathways_found_bridged"] += 1
                    has_extended = True
                    pathway_sizes.append(int(pw_size))
                    results.append({
                        'PathwayID': pw_id,
                        'PathwayName': pw_name,
                        'DrugTarget_UniProt': drug_p,
                        'DiseaseProtein_UniProt': disease_p,
                        'NumParticipants': pw_size,
                        'ExperimentType': "extended",
                        'BridgeNeeded': True,
                        'Source': source,
                        'EvidenceTier': evidence_tier,
                        'Label': 0
                    })
                else:
                    loss_stats["experiment_type"]["pathways_unbridged"] += 1
                    logger.debug("  -> No bridge for %s via pathway %s.", missing_protein, pw_id)

        # Pair-level bucketing (each pair counted exactly once).
        pair_metrics = loss_stats["pair_level"]
        pair_metrics["pairs_total"] += 1
        outcome = "pairs_with_clean_match" if has_clean else (
            "pairs_with_extended_only" if has_extended else "pairs_with_no_match"
        )
        pair_metrics[outcome] += 1

        # Break-down per Source / EvidenceTier (step02 provenance carried over).
        tier_bucket = loss_stats["by_evidence_tier"].setdefault(
            evidence_tier or "unknown",
            {"pairs_total": 0, "pairs_with_clean_match": 0,
             "pairs_with_extended_only": 0, "pairs_with_no_match": 0},
        )
        tier_bucket["pairs_total"] += 1
        tier_bucket[outcome] += 1

        source_key = source or "unknown"
        source_bucket = loss_stats["by_source"].setdefault(
            source_key,
            {"pairs_total": 0, "pairs_with_clean_match": 0,
             "pairs_with_extended_only": 0, "pairs_with_no_match": 0},
        )
        source_bucket["pairs_total"] += 1
        source_bucket[outcome] += 1
        if idx % step06_PROGRESS_EVERY == 0:
            _log_step06_progress(label=label, processed=idx, total=len(pairs_df), results=len(results))
    loss_stats["pathway_sizes"].extend(pathway_sizes)
    return results

def generate_uniprot2Allpathways_dataframe(path_file : str, col_names : list[str]) -> pd.DataFrame:
    url = "https://reactome.org/download/current/UniProt2Reactome_All_Levels.txt"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        with requests.get(url, stream=True, headers=headers, timeout=30) as r:
            r.raise_for_status()
            
            with open(path_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Download completed: {path_file}")
            
    except Exception as e:
        logger.error(f"Critical error during download: {e}")
        raise
    reactome_db = pd.read_csv(path_file, sep='\t', header=None, names=col_names)
    reactome_human = reactome_db[reactome_db['Species'] == 'Homo sapiens']
    reactome_human.to_csv(path_file, sep='\t', index=False, header=False)
    return reactome_human

def get_reactome_dict(path_file : Path) -> dict:
    col_names = CANONICAL_COLUMNS["reactome_db_col_names"]
    if not path_file.exists():
        reactome_db = generate_uniprot2Allpathways_dataframe(path_file, col_names)
        logger.info(f"Downloaded Reactome Pathways Dataframe. Len={reactome_db.size}")
    else:
        reactome_db = pd.read_csv(path_file, sep='\t', header=None, names=col_names)
        logger.info(f"Loading Reactome Pathways Dataframe. Len={reactome_db.size}")

    reactome_dict = {}
    for uniprot, pw_id, pw_name in zip(reactome_db['UniProtID'], reactome_db['PathwayID'], reactome_db['PathwayName']):
        if uniprot not in reactome_dict:
            reactome_dict[uniprot] = {}
        reactome_dict[uniprot][pw_id] = pw_name
    return reactome_dict

def build_step06_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    config_defaults = _get_step06_defaults(default_repo_root)
    parser = argparse.ArgumentParser(
        description="Mapping protein pairs to Reactome pathways from processed Step 2 outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed_dir = default_repo_root / "drug_disease_validation" / "data" / "processed"
    parser.add_argument(
        "--config",
        default=str(default_repo_root / "drug_disease_validation" / "config" / "step06.yml"),
        help="YAML config file for step06 tunables.",
    )
    parser.add_argument("--positive-pairs", default=str(processed_dir / CANONICAL_FILENAMES["positive_pairs"]))
    parser.add_argument("--biogrid-human-ppi", default=str(processed_dir / CANONICAL_FILENAMES["biogrid_human_ppi"]))
    parser.add_argument("--disgenet-disease-proteins", default=str(processed_dir / CANONICAL_FILENAMES["disgenet_disease_proteins"]))
    parser.add_argument("--ctd-therapeutic", default=str(processed_dir / CANONICAL_FILENAMES["ctd_therapeutic"]))
    parser.add_argument("--reactome-db-pathways-dataframe-file", default=str(processed_dir / CANONICAL_FILENAMES["reactome_uniprot2AllPathways"]))
    parser.add_argument("--max-size-pathway", type=int, default=None, help=f"Override YAML value (current default: {config_defaults['max_size_pathway']}).")
    parser.add_argument("--min-size-pathway", type=int, default=None, help=f"Override YAML value (current default: {config_defaults['min_size_pathway']}).")
    parser.add_argument("--max-try-search", type=int, default=None, help=f"Override YAML value (current default: {config_defaults['max_try_search']}).")
    parser.add_argument("--negative-proportion", type=int, default=None, help=f"Override YAML value (current default: {config_defaults['negative_proportion']}).")
    parser.add_argument("--random-state", type=int, default=None, help=f"Override YAML value (current default: {config_defaults['random_state']}).")
    parser.add_argument(
        "--reactome-api-rate-limit-per-sec",
        type=float,
        default=None,
        help=f"Override YAML value (current default: {config_defaults['reactome_api_rate_limit_per_sec']}).",
    )
    parser.add_argument("--output-dir", default=str(processed_dir))
    parser.add_argument("--logging-level", default="INFO")
    return parser

def build_negative_pairs(
    positive_pairs_df: pd.DataFrame, 
    disease_proteins_df: pd.DataFrame,
    random_state: int, 
    negative_ratio: int, 
    biogrid_graph: nx.Graph
) -> pd.DataFrame:
    """
    Generates 'Open World' negatives based on the complete DisGeNET database.
    It keeps the drug constant and samples disease-protein pairs from the pool,
    with the BioGRID score of the disease protein ranging from 0.5x to 2.0x
    relative to the original positive disease protein.
    """
    if positive_pairs_df.empty:
        return pd.DataFrame(columns=positive_pairs_df.columns)

    # 1. Pre-calcolo dei gradi dal grafo BioGRID per massima efficienza
    deg_dict = dict(biogrid_graph.degree())
    
    def _get_deg(prot_str):
        if pd.isna(prot_str) or str(prot_str).strip() == "": return 0
        return sum(deg_dict.get(p.strip(), 0) for p in str(prot_str).split("|"))
    
    # 2. Costruzione del pool "Open World" da DisGeNET
    # Manteniamo una riga per ogni coppia malattia-proteina.
    pool = disease_proteins_df[["DiseaseID", "DiseaseName", "UniProtID"]].copy()

    pool = pool.rename(columns={
        "UniProtID": "DiseaseProtein_UniProt"
    })

    pool["DiseaseProtein_UniProt"] = pool["DiseaseProtein_UniProt"].astype(str).str.strip()

    pool = pool[
        pool["DiseaseProtein_UniProt"].notna()
        & (pool["DiseaseProtein_UniProt"] != "")
        & (pool["DiseaseProtein_UniProt"].str.lower() != "nan")
    ].drop_duplicates().reset_index(drop=True)

    pool["Degree"] = pool["DiseaseProtein_UniProt"].apply(_get_deg)
    pool = pool[pool["Degree"] > 0].reset_index(drop=True)

    # 3. Lookup dei veri positivi (per non campionare una malattia che il farmaco cura davvero)
    true_pairs_dict = positive_pairs_df.groupby('DrugBankID')['DiseaseID'].apply(set).to_dict()

    rng = np.random.default_rng(random_state)
    negative_rows = []

    logger.info(f"Inizio campionamento negativi da un pool DisGeNET di {len(pool)} malattie valide...")

    # 4. Iterazione esplicita: 1 a N per ogni riga positiva
    for row in positive_pairs_df.itertuples(index=False):
        drug_id = row.DrugBankID
        pos_deg = _get_deg(row.DiseaseProtein_UniProt)
        known_diseases = true_pairs_dict.get(drug_id, set())
        ~pool['DiseaseID'].isin(known_diseases)

        # Filtro fondamentale: 0.5x <= grado <= 2.0x AND malattia non nota come positiva
        mask = (
            (pool['Degree'] >= 0.5 * pos_deg) & 
            (pool['Degree'] <= 2.0 * pos_deg) & 
            (~pool['DiseaseID'].isin(known_diseases))
        )
        candidates = pool[mask]

        if candidates.empty:
            continue

        # Campionamento sicuro
        n_to_sample = min(len(candidates), negative_ratio)
        sampled = candidates.sample(n=n_to_sample, random_state=int(rng.integers(0, 1e6)))

        # Costruzione delle righe negative
        for neg in sampled.itertuples(index=False):
            new_neg = row._asdict()
            new_neg['DiseaseName'] = neg.DiseaseName
            new_neg['DiseaseID'] = neg.DiseaseID
            new_neg['MatchedDiseaseID'] = neg.DiseaseID
            new_neg['DiseaseMatchSource'] = 'DisGeNET_Pool'
            new_neg['DiseaseProtein_UniProt'] = neg.DiseaseProtein_UniProt
            
            # Etichette di controllo
            new_neg['Label'] = 0
            new_neg['Source'] = 'open_world_negative'
            new_neg['EvidenceTier'] = 'none'
            
            negative_rows.append(new_neg)

    if not negative_rows:
        logger.warning("Nessun negativo generato. Controllare i parametri di grado.")
        return pd.DataFrame(columns=positive_pairs_df.columns)

    final_neg_df = pd.DataFrame(negative_rows)
    
    # Riordiniamo le colonne esattamente come nel DF in ingresso
    ordered_cols = list(positive_pairs_df.columns)
    extra_cols = [
        c for c in ["Label", "Source", "EvidenceTier"]
        if c in final_neg_df.columns and c not in ordered_cols
    ]

    final_neg_df = final_neg_df[ordered_cols + extra_cols]
    final_neg_df = final_neg_df.drop_duplicates(
        subset=["DrugBankID", "DrugTarget_UniProt", "DiseaseID", "DiseaseProtein_UniProt"]
    )
    
    logger.info(f"Generate {len(final_neg_df)} negative instances using Degree-Matching (Open World DisGeNET).")
    return final_neg_df

def run_step06(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    config = _resolve_step06_config(args, Path(__file__).resolve().parents[2])
    _configure_api_rate_limit(config["reactome_api_rate_limit_per_sec"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reactome_path = Path(args.reactome_db_pathways_dataframe_file)
    max_try_search = config["max_try_search"]
    max_size_pathway = config["max_size_pathway"]
    min_size_pathway = config["min_size_pathway"]
    negative_proportion = config["negative_proportion"]
    random_state = config["random_state"]
    reactome_dict = get_reactome_dict(reactome_path)

    _log_step06_header(config=config, output_dir=output_dir, reactome_path=reactome_path)
    logger.debug("Arguments: %s", vars(args))
    logger.info("Config file: %s", config["config_path"])

    positive_pairs_df = load_tabular_file(Path(args.positive_pairs))
    biogrid_dataframe = load_tabular_file(Path(args.biogrid_human_ppi))

    full_disease_pool = load_tabular_file(Path(args.disgenet_disease_proteins))

    biogrid_graph = load_biogrid_graph(biogrid_dataframe)
    # Aggregate step02 provenance (Source / EvidenceTier) at the protein-pair level
    # so it can be propagated into pair_pathway_mapping and into the step06 report.
    unique_positive_pairs_df = aggregate_pair_sources(positive_pairs_df)

    negative_pairs_df = build_negative_pairs(
        positive_pairs_df=positive_pairs_df, 
        random_state=random_state, 
        negative_ratio=negative_proportion,
        biogrid_graph=biogrid_graph,
        disease_proteins_df=full_disease_pool,
    )
    unique_negative_pairs_df = aggregate_pair_sources(negative_pairs_df)

    _log_step06_inputs(
        positive_pairs=len(positive_pairs_df),
        unique_positive_pairs=len(unique_positive_pairs_df),
        full_disease_pool=len(full_disease_pool),
        negative_pairs=len(negative_pairs_df),
        unique_negative_pairs=len(unique_negative_pairs_df),
        biogrid_rows=len(biogrid_dataframe),
    )
    
    loss_stats = {
        "positive_pairs_rows_total": int(len(positive_pairs_df)),
        "positive_pairs_cleaned": int(len(unique_positive_pairs_df)),
        "negative_pairs_rows_total": int(len(negative_pairs_df)),
        "negative_pairs_cleaned": int(len(unique_negative_pairs_df)),
        "single_drug_protein_not_found": 0,
        "single_disease_protein_not_found": 0,
        "experiment_type": {
            "total_pathways_searched": 0,
            "pathways_found_clean": 0,
            "pathways_found_bridged": 0,
            "pathways_unbridged": 0,
            "pathways_with_exceeds_size_in_reactome": 0,
            "pathways_with_exceeds_size_in_total": 0,
            "pathways_below_min_size": 0,
        },
        "pair_level": {
            "pairs_total": 0,
            "pairs_with_clean_match": 0,
            "pairs_with_extended_only": 0,
            "pairs_with_no_match": 0,
        },
        "by_evidence_tier": {},
        "by_source": {},
        "pathway_sizes": [],
    }

    results_noloops = get_pathways_results(
        pairs_df=unique_negative_pairs_df,
        max_size=max_size_pathway,
        min_size=min_size_pathway,
        reactome_dict=reactome_dict,
        loss_stats=loss_stats,
        max_try_search=max_try_search,
        biogrid_graph=biogrid_graph
        )
    results_noloops_df = pd.DataFrame(results_noloops)
    logger.info(
        "Results: no_loops=%s",
        len(results_noloops_df),
    )

    dfs_to_concat = [df for df in [results_noloops_df] if not df.empty]

    if dfs_to_concat:
        pair_pathway_mapping = pd.concat(dfs_to_concat, ignore_index=True).drop_duplicates()
    else:
        pair_pathway_mapping = pd.DataFrame(
            columns=[
                "PathwayID",
                "PathwayName",
                "DrugTarget_UniProt",
                "DiseaseProtein_UniProt",
                "NumParticipants",
                "ExperimentType",
                "BridgeNeeded",
                "Source",
                "EvidenceTier",
                "Label",
            ]
        )
    negative_pairs_df.to_csv(output_dir / CANONICAL_FILENAMES["negative_pairs"], sep='\t', index=False)
    logger.info("negative_pairs written: %s", output_dir / CANONICAL_FILENAMES["negative_pairs"])

    pair_pathway_mapping.to_csv(output_dir / CANONICAL_FILENAMES["negative_pair_pathway_mapping"], sep='\t', index=False)
    logger.info("negative_pair_pathway_mapping written: %s", output_dir / CANONICAL_FILENAMES["negative_pair_pathway_mapping"])

    pair_level = loss_stats["pair_level"]
    pairs_total = pair_level["pairs_total"] or 1
    pair_level_pct = {
        "pct_clean": round(100 * pair_level["pairs_with_clean_match"] / pairs_total, 2),
        "pct_extended_only": round(100 * pair_level["pairs_with_extended_only"] / pairs_total, 2),
        "pct_no_match": round(100 * pair_level["pairs_with_no_match"] / pairs_total, 2),
    }

    sizes = loss_stats.pop("pathway_sizes")
    sizes_series = pd.Series(sizes, dtype="int64")
    pathway_size_distribution = (
        {
            "count": int(sizes_series.size),
            "min": int(sizes_series.min()),
            "max": int(sizes_series.max()),
            "mean": float(round(sizes_series.mean(), 2)),
            "median": float(sizes_series.median()),
            "p25": float(sizes_series.quantile(0.25)),
            "p75": float(sizes_series.quantile(0.75)),
            "histogram": sizes_series.value_counts().sort_index().to_dict(),
        }
        if not sizes_series.empty
        else {"count": 0}
    )

    report = {
        "reactome_db": int(len(reactome_dict)),
        "pair_pathway_mapping": int(len(pair_pathway_mapping)),
        "config_path": str(config["config_path"]),
        "pathways_max_size": int(max_size_pathway),
        "pathways_min_size": int(min_size_pathway),
        "negative_pairs_proportion": int(negative_proportion),
        "random_state": int(random_state),
        "max_try_search_uniprot": int(max_try_search),
        "reactome_api_rate_limit_per_sec": config["reactome_api_rate_limit_per_sec"],
        "pair_level_percentages": pair_level_pct,
        "pathway_size_distribution": pathway_size_distribution,
        "loss_funnel": loss_stats,
    }
    report_path = output_dir / "step06_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    _log_step06_footer(pair_rows=len(pair_pathway_mapping), report_path=report_path)
