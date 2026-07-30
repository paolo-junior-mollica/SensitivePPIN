from __future__ import annotations

import io
import logging
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from api_clients.bulk_downloads import download_file

LOGGER = logging.getLogger(__name__)

STRING_API_BASE = "https://string-db.org/api"
STRING_DOWNLOAD_ALIASES_URL = "https://string-db.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz"


class StringDBClient:
    def __init__(
        self,
        *,
        api_base: str = STRING_API_BASE,
        caller_identity: str = "digitalhealthlab-sensitive-ppin",
        timeout: int = 120,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.caller_identity = caller_identity
        self.timeout = timeout

    def _post_tsv(self, method: str, params: dict[str, object]) -> pd.DataFrame:
        request_url = f"{self.api_base}/tsv/{method}"
        payload = {**params, "caller_identity": self.caller_identity}
        data = urlencode(payload).encode("utf-8")
        request = Request(
            request_url,
            data=data,
            headers={
                "User-Agent": "digitalhealthlab-sensitive-ppin/0.1",
                "Accept": "text/tab-separated-values",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            text = response.read().decode("utf-8")
        if not text.strip():
            return pd.DataFrame()
        return pd.read_csv(io.StringIO(text), sep="\t")

    def map_identifiers(
        self,
        identifiers: list[str],
        *,
        species: int = 9606,
        limit: int = 1,
        echo_query: int = 1,
    ) -> pd.DataFrame:
        identifiers = [item.strip() for item in identifiers if item and item.strip()]
        if not identifiers:
            return pd.DataFrame()
        return self._post_tsv(
            "get_string_ids",
            {
                "identifiers": "\r".join(identifiers),
                "species": species,
                "limit": limit,
                "echo_query": echo_query,
            },
        )

    def interaction_partners(
        self,
        identifiers: list[str],
        *,
        species: int = 9606,
        required_score: int = 700,
        network_type: str = "physical",
        limit: int = 50,
    ) -> pd.DataFrame:
        identifiers = [item.strip() for item in identifiers if item and item.strip()]
        if not identifiers:
            return pd.DataFrame()
        return self._post_tsv(
            "interaction_partners",
            {
                "identifiers": "\r".join(identifiers),
                "species": species,
                "required_score": required_score,
                "network_type": network_type,
                "limit": limit,
            },
        )


def ensure_string_aliases_file(cache_dir: Path, *, refresh: bool = False) -> Path:
    target = cache_dir / "9606.protein.aliases.v12.0.txt.gz"
    if target.exists() and not refresh:
        LOGGER.info("Using cached STRING aliases file %s", target)
        return target
    return download_file(STRING_DOWNLOAD_ALIASES_URL, target)


def load_string_aliases(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    required = {"#string_protein_id", "alias", "source"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"STRING aliases file missing columns: {sorted(missing)}")
    return df.rename(columns={"#string_protein_id": "stringId"})
