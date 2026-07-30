from __future__ import annotations

import json
import time
from dataclasses import dataclass
from urllib.parse import quote
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


@dataclass(slots=True)
class OpenTargetsClient:
    endpoint: str
    timeout_seconds: int = 60
    ols_search_endpoint: str = "https://www.ebi.ac.uk/ols4/api/search"
    max_retries: int = 4
    retry_backoff_seconds: float = 2.0

    def execute(self, query: str, variables: dict[str, object] | None = None) -> dict[str, object]:
        payload = json.dumps({"query": query, "variables": variables or {}}).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            request = Request(
                self.endpoint,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "ppi-drug-disease-validation/0.1",
                    "Connection": "close",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                if "errors" in data:
                    raise RuntimeError(f"Open Targets GraphQL error: {data['errors']}")
                return data.get("data", {})
            except RuntimeError:
                raise
            except (URLError, HTTPError, ConnectionError) as exc:
                last_error = exc
                if isinstance(exc, HTTPError) and 400 <= exc.code < 500 and exc.code != 429:
                    raise
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
                    continue
                break
        raise RuntimeError(f"Open Targets request failed after {self.max_retries} attempts: {last_error}")

    def search_disease(self, disease_name: str, max_hits: int) -> list[dict[str, object]]:
        query = quote(disease_name)
        url = f"{self.ols_search_endpoint}?q={query}&ontology=efo&rows={max_hits}"
        last_error: Exception | None = None
        data = None
        for attempt in range(self.max_retries):
            request = Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "ppi-drug-disease-validation/0.1",
                    "Connection": "close",
                },
                method="GET",
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except (URLError, HTTPError, ConnectionError) as exc:
                last_error = exc
                if isinstance(exc, HTTPError) and 400 <= exc.code < 500 and exc.code != 429:
                    raise
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
                    continue
                break
        if data is None:
            raise RuntimeError(f"OLS disease search failed after {self.max_retries} attempts: {last_error}")

        docs = data.get("response", {}).get("docs", [])
        hits: list[dict[str, object]] = []
        for doc in docs:
            ontology_name = str(doc.get("ontology_name") or "").lower()
            short_form = doc.get("short_form")
            label = doc.get("label")
            if ontology_name != "efo" or not short_form:
                continue
            hits.append(
                {
                    "id": str(short_form).replace(":", "_"),
                    "entity": "disease",
                    "name": label,
                    "description": doc.get("description"),
                }
            )
        return hits[:max_hits]

    def get_disease_associations(self, efo_id: str, page_size: int, score_min: float) -> list[dict[str, object]]:
        query = """
        query diseaseAssociations($efoId: String!, $index: Int!, $size: Int!) {
          disease(efoId: $efoId) {
            associatedTargets(page: {index: $index, size: $size}) {
              count
              rows {
                score
                target {
                  id
                  approvedSymbol
                  approvedName
                }
              }
            }
          }
        }
        """
        index = 0
        rows: list[dict[str, object]] = []
        total = None
        while total is None or index * page_size < total:
            data = self.execute(query, {"efoId": efo_id, "index": index, "size": page_size})
            block = data.get("disease", {}).get("associatedTargets", {})
            total = block.get("count", 0)
            page_rows = block.get("rows", [])
            if not page_rows:
                break
            for row in page_rows:
                score = float(row.get("score") or 0.0)
                if score >= score_min:
                    rows.append(row)
            index += 1
        return rows


def resolve_diseases(client: OpenTargetsClient, disease_names: list[str], max_hits: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for disease_name in disease_names:
        hits = client.search_disease(disease_name, max_hits=max_hits)
        if not hits:
            rows.append(
                {
                    "query_disease_name": disease_name,
                    "resolved": False,
                    "efo_id": pd.NA,
                    "efo_name": pd.NA,
                    "match_rank": pd.NA,
                }
            )
            continue
        top_hit = hits[0]
        rows.append(
            {
                "query_disease_name": disease_name,
                "resolved": True,
                "efo_id": top_hit.get("id"),
                "efo_name": top_hit.get("name"),
                "match_rank": 1,
            }
        )
    return pd.DataFrame(rows)


def resolve_diseases_from_pairs(
    client: OpenTargetsClient,
    pairs_df: pd.DataFrame,
    max_hits: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in pairs_df[["disease_name", "disease_source_id"]].drop_duplicates().itertuples(index=False):
        disease_name = str(record.disease_name)
        disease_source_id = record.disease_source_id
        if pd.notna(disease_source_id) and str(disease_source_id).startswith("EFO_"):
            rows.append(
                {
                    "query_disease_name": disease_name,
                    "resolved": True,
                    "efo_id": str(disease_source_id),
                    "efo_name": disease_name,
                    "match_rank": 0,
                    "resolution_source": "input_disease_source_id",
                }
            )
            continue
        hits = client.search_disease(disease_name, max_hits=max_hits)
        if not hits:
            rows.append(
                {
                    "query_disease_name": disease_name,
                    "resolved": False,
                    "efo_id": pd.NA,
                    "efo_name": pd.NA,
                    "match_rank": pd.NA,
                    "resolution_source": "search",
                }
            )
            continue
        top_hit = hits[0]
        rows.append(
            {
                "query_disease_name": disease_name,
                "resolved": True,
                "efo_id": top_hit.get("id"),
                "efo_name": top_hit.get("name"),
                "match_rank": 1,
                "resolution_source": "search",
            }
        )
    return pd.DataFrame(rows)


def build_disease_gene_table(client: OpenTargetsClient, resolved_df: pd.DataFrame, page_size: int, score_min: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in resolved_df.itertuples(index=False):
        if not bool(record.resolved):
            continue
        associations = client.get_disease_associations(record.efo_id, page_size=page_size, score_min=score_min)
        for association in associations:
            target = association.get("target", {})
            rows.append(
                {
                    "disease_id": record.efo_id,
                    "disease_name": record.efo_name,
                    "gene_id": target.get("id"),
                    "gene_symbol": target.get("approvedSymbol"),
                    "gene_name": target.get("approvedName"),
                    "evidence_score": association.get("score"),
                    "gene_id_type": "ensembl",
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset=["disease_id", "gene_id"]).reset_index(drop=True)
