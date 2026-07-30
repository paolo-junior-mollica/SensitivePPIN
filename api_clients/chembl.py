from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd

LOGGER = logging.getLogger(__name__)

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"


@dataclass(slots=True)
class ChEMBLClient:
    base_url: str = BASE_URL
    timeout_seconds: int = 60

    def _get_json(self, path: str) -> dict[str, object]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        with urlopen(url, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def search_molecule(self, drug_name: str, limit: int) -> list[dict[str, object]]:
        query = quote(drug_name)
        data = self._get_json(f"molecule/search?q={query}&limit={limit}&format=json")
        return data.get("molecules", [])  # type: ignore[return-value]

    def get_drug_mechanisms(self, molecule_chembl_id: str) -> list[dict[str, object]]:
        data = self._get_json(f"mechanism.json?molecule_chembl_id={quote(molecule_chembl_id)}&limit=1000")
        return data.get("mechanisms", [])  # type: ignore[return-value]

    def get_target(self, target_chembl_id: str) -> dict[str, object]:
        return self._get_json(f"target/{quote(target_chembl_id)}.json")

    def get_activities(self, molecule_chembl_id: str, limit: int) -> list[dict[str, object]]:
        data = self._get_json(
            "activity.json?"
            + f"molecule_chembl_id={quote(molecule_chembl_id)}"
            + f"&limit={int(limit)}"
        )
        return data.get("activities", [])  # type: ignore[return-value]


def _coerce_max_phase(value: object) -> int:
    if value is None or value == "":
        return -1
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return -1


def resolve_repodb_drugs_to_chembl(
    client: ChEMBLClient,
    positives_df: pd.DataFrame,
    limit: int,
    approved_max_phase_min: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for drug_name in sorted(positives_df["drug_name"].astype(str).unique()):
        molecules = client.search_molecule(drug_name, limit=limit)
        if not molecules:
            rows.append(
                {
                    "query_drug_name": drug_name,
                    "resolved": False,
                    "molecule_chembl_id": pd.NA,
                    "chembl_pref_name": pd.NA,
                    "max_phase": pd.NA,
                    "match_rank": pd.NA,
                }
            )
            continue
        chosen = None
        for rank, molecule in enumerate(molecules, start=1):
            max_phase = _coerce_max_phase(molecule.get("max_phase"))
            if max_phase >= approved_max_phase_min:
                chosen = (rank, molecule)
                break
        if chosen is None:
            chosen = (1, molecules[0])
        rank, molecule = chosen
        rows.append(
            {
                "query_drug_name": drug_name,
                "resolved": True,
                "molecule_chembl_id": molecule.get("molecule_chembl_id"),
                "chembl_pref_name": molecule.get("pref_name"),
                "max_phase": _coerce_max_phase(molecule.get("max_phase")),
                "match_rank": rank,
            }
        )
    return pd.DataFrame(rows)


def build_drug_targets_table(
    client: ChEMBLClient,
    resolved_drugs_df: pd.DataFrame,
    target_relation_types: list[str],
    target_organism_exact: str,
    use_activity_fallback: bool = True,
    activity_limit: int = 200,
    activity_assay_types: list[str] | None = None,
    activity_min_pchembl: float = 5.0,
    activity_target_types: list[str] | None = None,
) -> pd.DataFrame:
    allowed_relations = set(target_relation_types)
    allowed_assay_types = set(activity_assay_types or [])
    allowed_target_types = set(activity_target_types or [])
    rows: list[dict[str, object]] = []
    for record in resolved_drugs_df.itertuples(index=False):
        if not bool(record.resolved):
            continue
        rows_before_drug = len(rows)
        mechanisms = client.get_drug_mechanisms(record.molecule_chembl_id)
        for mechanism in mechanisms:
            target_chembl_id = mechanism.get("target_chembl_id")
            if not target_chembl_id:
                continue
            target = client.get_target(str(target_chembl_id))
            target_components = target.get("target_components") or []
            organism = str(target.get("organism") or "")
            if organism and organism.lower() != target_organism_exact.lower():
                continue
            target_relation = mechanism.get("relationship_type")
            if allowed_relations and target_relation is not None and str(target_relation) not in allowed_relations:
                continue
            for component in target_components:
                accessions = component.get("target_component_xrefs") or []
                uniprot_ids = []
                for xref in accessions:
                    src = str(xref.get("xref_src_db") or "").lower()
                    xref_id = xref.get("xref_id")
                    if src == "uniprot" and xref_id:
                        uniprot_ids.append(str(xref_id))
                accession = component.get("accession")
                if accession:
                    uniprot_ids.append(str(accession))
                uniprot_ids = sorted(set(uniprot_ids))
                if not uniprot_ids:
                    continue
                for uniprot_id in uniprot_ids:
                    rows.append(
                        {
                            "drug_name": record.query_drug_name,
                            "drug_id": record.molecule_chembl_id,
                            "target_id": uniprot_id,
                            "target_name": component.get("component_description"),
                            "organism": organism,
                            "target_type": target.get("target_type"),
                            "mechanism_of_action": mechanism.get("mechanism_of_action"),
                            "action_type": mechanism.get("action_type"),
                            "target_relation_type": target_relation,
                        }
                    )
        if len(rows) > rows_before_drug or not use_activity_fallback:
            continue

        activities = client.get_activities(record.molecule_chembl_id, limit=activity_limit)
        for activity in activities:
            assay_type = str(activity.get("assay_type") or "")
            if allowed_assay_types and assay_type not in allowed_assay_types:
                continue
            pchembl_value = activity.get("pchembl_value")
            try:
                if pchembl_value is None or float(pchembl_value) < activity_min_pchembl:
                    continue
            except (TypeError, ValueError):
                continue
            target_chembl_id = activity.get("target_chembl_id")
            if not target_chembl_id:
                continue
            target = client.get_target(str(target_chembl_id))
            organism = str(target.get("organism") or "")
            target_type = str(target.get("target_type") or "")
            if organism and organism.lower() != target_organism_exact.lower():
                continue
            if allowed_target_types and target_type not in allowed_target_types:
                continue
            for component in target.get("target_components") or []:
                accessions = component.get("target_component_xrefs") or []
                uniprot_ids = []
                for xref in accessions:
                    src = str(xref.get("xref_src_db") or "").lower()
                    xref_id = xref.get("xref_id")
                    if src == "uniprot" and xref_id:
                        uniprot_ids.append(str(xref_id))
                accession = component.get("accession")
                if accession:
                    uniprot_ids.append(str(accession))
                uniprot_ids = sorted(set(uniprot_ids))
                for uniprot_id in uniprot_ids:
                    rows.append(
                        {
                            "drug_name": record.query_drug_name,
                            "drug_id": record.molecule_chembl_id,
                            "target_id": uniprot_id,
                            "target_name": component.get("component_description"),
                            "organism": organism,
                            "target_type": target_type,
                            "mechanism_of_action": pd.NA,
                            "action_type": pd.NA,
                            "target_relation_type": "activity_fallback",
                        }
                    )
    return pd.DataFrame(rows).drop_duplicates(subset=["drug_id", "target_id"]).reset_index(drop=True)
