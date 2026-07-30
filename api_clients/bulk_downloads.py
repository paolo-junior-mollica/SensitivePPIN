from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

LOGGER = logging.getLogger(__name__)


def download_file(url: str, destination: Path, headers: dict[str, str] | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %s to %s", url, destination)
    request_headers = {
        "User-Agent": "digitalhealthlab-sensitive-ppin/0.1",
        "Accept": "*/*",
    }
    if headers:
        request_headers.update(headers)
    request = Request(url, headers=request_headers)
    with urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                LOGGER.info("Expected download size for %s: %s bytes", destination.name, int(content_length))
            except ValueError:
                LOGGER.info("Expected download size for %s: %s", destination.name, content_length)
        shutil.copyfileobj(response, handle)
    try:
        LOGGER.info("Completed download %s (%s bytes)", destination, destination.stat().st_size)
    except OSError:
        LOGGER.info("Completed download %s", destination)
    return destination


def clone_repository(repo_url: str, destination: Path) -> None:
    if destination.exists():
        LOGGER.info("Repository already present at %s", destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Cloning %s into %s", repo_url, destination)
    subprocess.run(["git", "clone", repo_url, str(destination)], check=True)


def download_zenodo_record(doi_or_record_id: str, destination_dir: Path) -> list[Path]:
    record_id = doi_or_record_id.rsplit(".", 1)[-1]
    api_url = f"https://zenodo.org/api/records/{record_id}"
    LOGGER.info("Resolving Zenodo record %s via %s", doi_or_record_id, api_url)
    request = Request(api_url, headers={"Accept": "application/json"}, method="GET")
    with urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))

    files = payload.get("files", [])
    destination_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for file_record in files:
        filename = file_record.get("key") or file_record.get("filename") or "zenodo_asset"
        links = file_record.get("links", {})
        file_url = links.get("self") or links.get("download") or links.get("content")
        if not file_url:
            continue
        downloaded.append(download_file(str(file_url), destination_dir / str(filename)))
    return downloaded
