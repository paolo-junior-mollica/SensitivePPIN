from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from urllib.request import Request, urlopen

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class OpenTargetsClient:
    endpoint: str
    timeout_seconds: int = 60

    def execute(self, query: str, variables: dict[str, object] | None = None) -> dict[str, object]:
        payload = json.dumps({"query": query, "variables": variables or {}}).encode("utf-8")
        request = Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        if "errors" in data:
            raise RuntimeError(f"Open Targets GraphQL error: {data['errors']}")
        return data.get("data", {})

    def search_disease(self, disease_name: str, max_hits: int) -> list[dict[str, object]]:
        query = """
        query searchDisease($queryString: String!) {
          search(queryString: $queryString, entityNames: ["disease"], page: {size: 10}) {
            hits {
              id
              entity
              name
              description
            }
          }
        }
        """
        data = self.execute(query, {"queryString": disease_name})
        hits = data.get("search", {}).get("hits", [])
        return [hit for hit in hits if hit.get("entity") == "disease"][:max_hits]

    def get_disease_associations(
        self,
        efo_id: str,
        page_size: int,
        score_min: float,
    ) -> list[dict[str, object]]:
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


def resolve_diseases(
    client: OpenTargetsClient,
    disease_names: list[str],
    max_hits: int,
) -> pd.DataFrame:
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


def build_disease_gene_table(
    client: OpenTargetsClient,
    resolved_df: pd.DataFrame,
    page_size: int,
    score_min: float,
) -> pd.DataFrame:
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

