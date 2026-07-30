from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


LOGGER = logging.getLogger(__name__)


def _normalize(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ";".join(_normalize(item) for item in value if _normalize(item))
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _first_token(value: str) -> str:
    text = _normalize(value)
    if not text:
        return ""
    for delimiter in (";", ",", "|", " "):
        if delimiter in text:
            text = text.split(delimiter)[0].strip()
            break
    return text


@dataclass(slots=True)
class ProteinAtlasClient:
    base_url: str = "https://www.proteinatlas.org"
    timeout_seconds: int = 60

    def _request_json(self, url: str):
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "digitalhealthlab-sensitive-ppin/0.1",
            },
            method="GET",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def search_gene(self, symbol: str):
        # HPA search pages expose downloadable JSON for the search result set.
        query = quote(symbol)
        attempts = [
            f"{self.base_url}/search/{query}?format=json",
            f"{self.base_url}/search/gene%3A{query}?format=json",
        ]
        last_error: Exception | None = None
        for url in attempts:
            try:
                return self._request_json(url)
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        return None

    def fetch_gene_record(self, symbol: str) -> dict[str, str] | None:
        try:
            payload = self.search_gene(symbol)
        except HTTPError as exc:
            if exc.code in {400, 404}:
                LOGGER.info("Protein Atlas did not resolve symbol %s", symbol)
                return None
            raise
        except URLError:
            raise
        candidates = []
        if isinstance(payload, list):
            candidates = payload
        elif isinstance(payload, dict):
            for key in ("results", "data", "rows", "genes"):
                value = payload.get(key)
                if isinstance(value, list):
                    candidates = value
                    break
            if not candidates:
                candidates = [payload]

        target = symbol.strip().upper()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            gene_symbol = _normalize(
                candidate.get("Gene")
                or candidate.get("gene")
                or candidate.get("Gene name")
                or candidate.get("gene_name")
                or candidate.get("gs")
            )
            synonyms = _normalize(
                candidate.get("Gene synonym")
                or candidate.get("gene_synonym")
                or candidate.get("synonyms")
                or candidate.get("alias")
            )
            uniprot_id = _first_token(
                _normalize(
                    candidate.get("Uniprot")
                    or candidate.get("Uniprot accession")
                    or candidate.get("uniprot")
                    or candidate.get("up")
                    or candidate.get("uniprot_id")
                )
            )

            exact_gene_match = gene_symbol.upper() == target
            synonym_match = target in {item.strip().upper() for item in synonyms.replace(",", ";").split(";") if item.strip()}
            if (exact_gene_match or synonym_match) and uniprot_id:
                return {
                    "from_id": symbol,
                    "resolved_gene_symbol": gene_symbol or symbol,
                    "to_id": uniprot_id,
                    "source": "protein_atlas",
                    "status": "resolved",
                }

        LOGGER.info("Protein Atlas did not resolve symbol %s", symbol)
        return None
