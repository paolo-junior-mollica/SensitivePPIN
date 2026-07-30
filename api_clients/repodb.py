"""RepoDB normalization utilities."""
from __future__ import annotations

import pandas as pd

REPO_DB_ALIASES = {
    "drug_id": ["drug_id", "drugbank_id", "drugbank.id", "drugbank-id"],
    "drug_name": ["drug_name", "drug", "compound_name", "name"],
    "disease_name": ["disease_name", "indication", "disease", "ind_name", "indication_name"],
    "disease_source_id": ["disease_id", "ind_id", "umls_cui", "cui"],
    "status": ["status", "phase", "development_status", "approval_status"],
}


def _resolve_column(columns: list[str], aliases: list[str]) -> str | None:
    normalised = {column.lower(): column for column in columns}
    for alias in aliases:
        if alias.lower() in normalised:
            return normalised[alias.lower()]
    return None


def normalize_repodb_positives(
    repodb_df: pd.DataFrame,
    positive_status_values: list[str],
) -> pd.DataFrame:
    resolved = {
        key: _resolve_column(list(repodb_df.columns), aliases)
        for key, aliases in REPO_DB_ALIASES.items()
    }
    required = ["drug_id", "drug_name", "disease_name"]
    missing = [key for key in required if resolved[key] is None]
    if missing:
        raise ValueError(f"repoDB missing required columns for normalization: {missing}")

    df = repodb_df.copy()
    if resolved["status"] is not None:
        allowed = {value.lower() for value in positive_status_values}
        df = df[df[resolved["status"]].astype(str).str.lower().isin(allowed)]

    out = pd.DataFrame(
        {
            "drug_id": df[resolved["drug_id"]].astype(str).str.strip(),
            "drug_name": df[resolved["drug_name"]].astype(str).str.strip(),
            "disease_name": df[resolved["disease_name"]].astype(str).str.strip(),
        }
    )
    if resolved["disease_source_id"] is not None:
        out["disease_source_id"] = df[resolved["disease_source_id"]].astype(str).str.strip()
    else:
        out["disease_source_id"] = pd.NA
    out["label"] = 1
    out = out[(out["drug_id"] != "") & (out["disease_name"] != "")]
    out = out.drop_duplicates(subset=["drug_id", "disease_name"]).reset_index(drop=True)
    return out
