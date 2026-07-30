from __future__ import annotations

import json
import time
from dataclasses import dataclass
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd


@dataclass(slots=True)
class UniProtMappingClient:
    endpoint: str
    poll_seconds: float = 2.0
    max_polls: int = 30

    def _post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        body = urlencode(payload, doseq=True).encode("utf-8")
        request = Request(
            f"{self.endpoint.rstrip('/')}/{path.lstrip('/')}",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get_json(self, path_or_url: str) -> tuple[dict[str, object], object]:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            url = path_or_url
        else:
            url = f"{self.endpoint.rstrip('/')}/{path_or_url.lstrip('/')}"
        request = Request(url, headers={"Accept": "application/json"}, method="GET")
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8")), response.info()

    def _iter_result_pages(self, url: str):
        next_url = url
        while next_url:
            payload, headers = self._get_json(next_url)
            yield payload
            link_header = headers.get("Link")
            next_url = None
            if link_header and 'rel="next"' in link_header:
                for part in link_header.split(","):
                    section = part.strip()
                    if 'rel="next"' in section:
                        next_url = section.split(";")[0].strip()[1:-1]
                        break

    def map_ids(self, ids: list[str], from_db: str, to_db: str) -> pd.DataFrame:
        if not ids:
            return pd.DataFrame(columns=["from_id", "to_id"])
        submission = self._post_json("idmapping/run", {"from": from_db, "to": to_db, "ids": ",".join(ids)})
        job_id = submission["jobId"]
        redirect_url = None
        for _ in range(self.max_polls):
            status, _ = self._get_json(f"idmapping/status/{job_id}")
            redirect_url = status.get("redirectURL")
            if "results" in status or status.get("jobStatus") == "FINISHED" or redirect_url:
                break
            if status.get("jobStatus") in {"RUNNING", "NEW"}:
                time.sleep(self.poll_seconds)
                continue
            raise RuntimeError(f"UniProt mapping failed with status: {status}")
        rows: list[dict[str, str]] = []
        results_url = redirect_url or f"{self.endpoint.rstrip('/')}/idmapping/results/{job_id}?size=500"
        for page in self._iter_result_pages(results_url):
            for row in page.get("results", []):
                source = str(row.get("from"))
                to_value = row.get("to")
                if isinstance(to_value, dict):
                    accession = to_value.get("primaryAccession") or to_value.get("uniProtkbId") or to_value.get("id")
                else:
                    accession = to_value
                if source and accession:
                    rows.append({"from_id": source, "to_id": str(accession)})
        return pd.DataFrame(rows).drop_duplicates()
