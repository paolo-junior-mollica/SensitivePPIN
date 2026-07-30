from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


@dataclass(slots=True)
class CtGovClient:
    base_url: str
    timeout_seconds: int = 60

    def fetch_studies(self, condition: str, intervention: str, statuses: list[str], page_size: int) -> list[dict[str, object]]:
        params = {
            "query.cond": condition,
            "query.intr": intervention,
            "pageSize": page_size,
            "format": "json",
        }
        if statuses:
            params["filter.overallStatus"] = ",".join(statuses)
        url = f"{self.base_url}?{urlencode(params)}"
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "ppi-drug-disease-validation/0.1"}, method="GET")
        with urlopen(request, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data.get("studies", [])


def build_labeled_pairs_from_queries(
    client: CtGovClient,
    interventions_df: pd.DataFrame,
    conditions_df: pd.DataFrame,
    positive_statuses: list[str],
    negative_statuses: list[str],
    page_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for drug_name in interventions_df["drug_name"].astype(str).dropna().unique():
        for disease_name in conditions_df["disease_name"].astype(str).dropna().unique():
            positive_studies = client.fetch_studies(disease_name, drug_name, positive_statuses, page_size)
            negative_studies = client.fetch_studies(disease_name, drug_name, negative_statuses, page_size)
            for study in positive_studies:
                rows.append(
                    {
                        "drug_name": drug_name,
                        "disease_name": disease_name,
                        "label": 1,
                        "status": study.get("protocolSection", {}).get("statusModule", {}).get("overallStatus"),
                        "nct_id": study.get("protocolSection", {}).get("identificationModule", {}).get("nctId"),
                    }
                )
            for study in negative_studies:
                rows.append(
                    {
                        "drug_name": drug_name,
                        "disease_name": disease_name,
                        "label": 0,
                        "status": study.get("protocolSection", {}).get("statusModule", {}).get("overallStatus"),
                        "nct_id": study.get("protocolSection", {}).get("identificationModule", {}).get("nctId"),
                    }
                )
    return pd.DataFrame(rows).drop_duplicates(subset=["drug_name", "disease_name", "label", "nct_id"]).reset_index(drop=True)

