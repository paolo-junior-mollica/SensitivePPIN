from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import networkx as nx
import pandas as pd

from drug_disease_validation.src.utils import (
    deserialize_nodes,
    ensure_parent_dir,
    normalise_identifier,
    write_pickle,
)

LOGGER = logging.getLogger(__name__)


def expand_positive_pairs_to_protein_pairs(
    pair_df: pd.DataFrame,
    *,
    max_positive_pairs: int | None = None,
    max_drug_targets_per_pair: int | None = None,
    max_disease_genes_per_pair: int | None = None,
) -> pd.DataFrame:
    working = pair_df.copy()
    if max_positive_pairs is not None:
        working = working.head(max_positive_pairs).copy()

    atomic_required = {"DrugTarget_UniProt", "DiseaseProtein_UniProt"}
    if atomic_required.issubset(working.columns):
        out = working[
            [
                column
                for column in [
                    "drug_id",
                    "drug_name",
                    "disease_id",
                    "disease_name",
                    "DrugName",
                    "DrugBankID",
                    "DiseaseName",
                    "DiseaseID",
                    "DrugTarget_UniProt",
                    "DiseaseProtein_UniProt",
                    "Label",
                ]
                if column in working.columns
            ]
        ].drop_duplicates().reset_index(drop=True)
        return out

    required = {"drug_targets", "disease_genes"}
    missing = required.difference(working.columns)
    if missing:
        raise ValueError(f"Positive pair file missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for row in working.itertuples(index=False):
        drug_targets = sorted(deserialize_nodes(row.drug_targets))
        disease_genes = sorted(deserialize_nodes(row.disease_genes))
        if max_drug_targets_per_pair is not None:
            drug_targets = drug_targets[:max_drug_targets_per_pair]
        if max_disease_genes_per_pair is not None:
            disease_genes = disease_genes[:max_disease_genes_per_pair]

        for drug_target in drug_targets:
            for disease_protein in disease_genes:
                rows.append(
                    {
                        "drug_id": getattr(row, "drug_id", ""),
                        "drug_name": getattr(row, "drug_name", ""),
                        "disease_id": getattr(row, "disease_id", ""),
                        "disease_name": getattr(row, "disease_name", ""),
                        "DrugName": getattr(row, "DrugName", getattr(row, "drug_name", "")),
                        "DrugBankID": getattr(row, "DrugBankID", getattr(row, "drug_id", "")),
                        "DiseaseName": getattr(row, "DiseaseName", getattr(row, "disease_name", "")),
                        "DiseaseID": getattr(row, "DiseaseID", getattr(row, "disease_id", "")),
                        "DrugTarget_UniProt": drug_target,
                        "DiseaseProtein_UniProt": disease_protein,
                        "Label": getattr(row, "Label", getattr(row, "label", pd.NA)),
                    }
                )
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


@dataclass(slots=True)
class ReactomeCache:
    pathways_path: Path
    participants_path: Path
    pathways: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    participants: dict[str, dict[str, object]] = field(default_factory=dict)

    @classmethod
    def load(cls, cache_dir: Path) -> "ReactomeCache":
        pathways_path = cache_dir / "reactome_pathways_cache.pkl"
        participants_path = cache_dir / "reactome_participants_cache.pkl"
        pathways = cls._load_pickle(pathways_path)
        participants = cls._load_pickle(participants_path)
        return cls(
            pathways_path=pathways_path,
            participants_path=participants_path,
            pathways=pathways,
            participants=participants,
        )

    @staticmethod
    def _load_pickle(path: Path):
        if not path.exists():
            return {}
        with path.open("rb") as handle:
            return pickle.load(handle)

    def flush(self) -> None:
        ensure_parent_dir(self.pathways_path)
        write_pickle(self.pathways, self.pathways_path)
        write_pickle(self.participants, self.participants_path)


@dataclass(slots=True)
class ReactomeClient:
    base_url: str = "https://reactome.org/ContentService/data"
    min_interval_seconds: float = 0.11
    timeout_seconds: int = 60
    max_retries: int = 4
    retry_backoff_seconds: float = 1.5
    _last_request_time: float = field(default=0.0, init=False)

    def _rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_request_time = time.monotonic()

    def _get_json(self, path: str):
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._rate_limit()
            request = Request(url, headers={"Accept": "application/json"}, method="GET")
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    import json

                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                last_error = exc
                if exc.code == 404:
                    raise
                if 500 <= exc.code < 600 and attempt + 1 < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
                    continue
                raise
            except (RemoteDisconnected, URLError, TimeoutError, ConnectionError, IncompleteRead) as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Reactome request failed without a concrete exception for path {path}")

    def get_pathways_for_protein(self, uniprot_id: str) -> list[dict[str, str]]:
        attempts = [
            f"mapping/UniProt/{quote(uniprot_id)}/pathways",
            f"pathways/low/entity/{quote(uniprot_id)}?species=9606",
        ]
        if not uniprot_id.startswith("UniProt:"):
            attempts.append(f"pathways/low/entity/{quote(f'UniProt:{uniprot_id}')}" + "?species=9606")

        payload = None
        last_error: Exception | None = None
        for path in attempts:
            try:
                payload = self._get_json(path)
                break
            except HTTPError as exc:
                last_error = exc
                if exc.code != 404:
                    raise
                continue

        if payload is None:
            if isinstance(last_error, HTTPError) and last_error.code == 404:
                return []
            if last_error is not None:
                raise last_error
            return []

        if not isinstance(payload, list):
            return []
        rows: list[dict[str, str]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            pathway_id = normalise_identifier(item.get("stId") or item.get("stableIdentifier") or item.get("dbId") or item.get("id"))
            pathway_name = normalise_identifier(item.get("displayName") or item.get("name"))
            species_name = normalise_identifier(item.get("speciesName") or item.get("species"))
            if species_name and "homo sapiens" not in species_name.lower():
                continue
            if pathway_id:
                rows.append({"PathwayID": pathway_id, "PathwayName": pathway_name})
        rows = list({(row["PathwayID"], row["PathwayName"]): row for row in rows}.values())
        return sorted(rows, key=lambda row: (row["PathwayID"], row["PathwayName"]))

    def get_pathway_participants(self, pathway_id: str) -> dict[str, object]:
        payload = self._get_json(f"participants/{quote(pathway_id)}")
        participants = sorted(_extract_reference_gene_products(payload))
        pathway_name = ""
        if isinstance(payload, dict):
            pathway_name = normalise_identifier(payload.get("displayName") or payload.get("name"))
        return {
            "PathwayID": pathway_id,
            "PathwayName": pathway_name,
            "participants": participants,
            "NumParticipants": len(participants),
        }


def _extract_reference_gene_products(payload) -> set[str]:
    identifiers: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return

        schema_class = normalise_identifier(node.get("schemaClass"))
        if schema_class == "ReferenceGeneProduct":
            reference_db = node.get("referenceDatabase")
            database_name = ""
            if isinstance(reference_db, dict):
                database_name = normalise_identifier(reference_db.get("displayName") or reference_db.get("name"))
            accession = normalise_identifier(node.get("identifier") or node.get("accession") or node.get("displayName"))
            if accession.startswith("UniProt:"):
                accession = accession.split(":", 1)[1]
            if accession and (not database_name or "uniprot" in database_name.lower()):
                identifiers.add(accession)

        for value in node.values():
            walk(value)

    walk(payload)
    return identifiers


def _get_cached_pathways(uniprot_id: str, client: ReactomeClient, cache: ReactomeCache) -> list[dict[str, str]]:
    if uniprot_id not in cache.pathways:
        cache.pathways[uniprot_id] = client.get_pathways_for_protein(uniprot_id)
    return cache.pathways[uniprot_id]


def _get_cached_participants(pathway_id: str, client: ReactomeClient, cache: ReactomeCache) -> dict[str, object]:
    if pathway_id not in cache.participants:
        cache.participants[pathway_id] = client.get_pathway_participants(pathway_id)
    return cache.participants[pathway_id]


def _is_size_eligible(num_participants: int, min_size: int, max_size: int) -> bool:
    return min_size <= num_participants <= max_size


def _is_bridgeable(missing_protein: str, participants: set[str], graph: nx.Graph) -> bool:
    if missing_protein in participants:
        return True
    if missing_protein not in graph:
        return False
    return any(neighbor in participants for neighbor in graph.neighbors(missing_protein))


def find_reactome_pair_pathways(
    protein_pairs_df: pd.DataFrame,
    *,
    client: ReactomeClient,
    cache: ReactomeCache,
    biogrid_graph: nx.Graph,
    min_participants: int = 5,
    max_participants: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"DrugTarget_UniProt", "DiseaseProtein_UniProt"}
    missing = required.difference(protein_pairs_df.columns)
    if missing:
        raise ValueError(f"Protein pair file missing columns: {sorted(missing)}")

    output_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for row in protein_pairs_df.itertuples(index=False):
        drug_target = normalise_identifier(row.DrugTarget_UniProt)
        disease_protein = normalise_identifier(row.DiseaseProtein_UniProt)
        target_pathways = _get_cached_pathways(drug_target, client, cache)
        disease_pathways = _get_cached_pathways(disease_protein, client, cache)
        target_lookup = {item["PathwayID"]: item["PathwayName"] for item in target_pathways}
        disease_lookup = {item["PathwayID"]: item["PathwayName"] for item in disease_pathways}
        shared_ids = sorted(set(target_lookup).intersection(disease_lookup))

        pair_clean = 0
        pair_extended = 0

        for pathway_id in shared_ids:
            participant_payload = _get_cached_participants(pathway_id, client, cache)
            participants = set(participant_payload["participants"])
            num_participants = int(participant_payload["NumParticipants"])
            if not _is_size_eligible(num_participants, min_participants, max_participants):
                continue
            if drug_target not in participants or disease_protein not in participants:
                continue
            output_rows.append(
                {
                    "DrugTarget_UniProt": drug_target,
                    "DiseaseProtein_UniProt": disease_protein,
                    "PathwayID": pathway_id,
                    "PathwayName": participant_payload.get("PathwayName") or target_lookup.get(pathway_id) or disease_lookup.get(pathway_id) or "",
                    "NumParticipants": num_participants,
                    "ExperimentType": "clean",
                    "BridgeNeeded": False,
                    "Participants": "|".join(sorted(participants)),
                }
            )
            pair_clean += 1

        if pair_clean == 0:
            union_ids = sorted(set(target_lookup).union(disease_lookup))
            for pathway_id in union_ids:
                participant_payload = _get_cached_participants(pathway_id, client, cache)
                participants = set(participant_payload["participants"])
                num_participants = int(participant_payload["NumParticipants"])
                if not _is_size_eligible(num_participants, min_participants, max_participants):
                    continue

                target_in = drug_target in participants
                disease_in = disease_protein in participants
                if target_in and disease_in:
                    continue
                if not target_in and not disease_in:
                    continue

                missing_protein = disease_protein if target_in else drug_target
                if not _is_bridgeable(missing_protein, participants, biogrid_graph):
                    continue

                output_rows.append(
                    {
                        "DrugTarget_UniProt": drug_target,
                        "DiseaseProtein_UniProt": disease_protein,
                        "PathwayID": pathway_id,
                        "PathwayName": participant_payload.get("PathwayName") or target_lookup.get(pathway_id) or disease_lookup.get(pathway_id) or "",
                        "NumParticipants": num_participants,
                        "ExperimentType": "extended",
                        "BridgeNeeded": True,
                        "Participants": "|".join(sorted(participants)),
                    }
                )
                pair_extended += 1

        summary_rows.append(
            {
                "DrugTarget_UniProt": drug_target,
                "DiseaseProtein_UniProt": disease_protein,
                "num_clean_pathways": pair_clean,
                "num_extended_pathways": pair_extended,
                "has_clean": pair_clean > 0,
                "has_extended": pair_extended > 0,
                "has_any_pathway": pair_clean > 0 or pair_extended > 0,
            }
        )

    mapping_df = pd.DataFrame(output_rows)
    if not mapping_df.empty:
        mapping_df = mapping_df.drop_duplicates(
            subset=[
                "DrugTarget_UniProt",
                "DiseaseProtein_UniProt",
                "PathwayID",
                "ExperimentType",
            ]
        ).sort_values(
            ["DrugTarget_UniProt", "DiseaseProtein_UniProt", "ExperimentType", "PathwayID"],
            kind="stable",
        )
    else:
        mapping_df = pd.DataFrame(
            columns=[
                "DrugTarget_UniProt",
                "DiseaseProtein_UniProt",
                "PathwayID",
                "PathwayName",
                "NumParticipants",
                "ExperimentType",
                "BridgeNeeded",
                "Participants",
            ]
        )

    summary_df = pd.DataFrame(summary_rows).drop_duplicates()
    return mapping_df.reset_index(drop=True), summary_df.reset_index(drop=True)
