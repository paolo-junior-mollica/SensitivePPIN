from __future__ import annotations

import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)
NON_HPA_SYMBOL_PATTERNS = [
    re.compile(r"^CG\d+$", re.IGNORECASE),
    re.compile(r"rik$", re.IGNORECASE),
    re.compile(r"rrna$", re.IGNORECASE),
]

try:
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    _tqdm = None


class _SimpleProgressBar:
    def __init__(self, items: list[str], *, desc: str, unit: str) -> None:
        self._items = items
        self._desc = desc
        self._unit = unit
        self._total = len(items)
        self._count = 0
        self._postfix = ""
        self._render()

    def __iter__(self):
        for item in self._items:
            yield item
            self._count += 1
            self._render()
        self._finish()

    def set_postfix_str(self, postfix: str) -> None:
        self._postfix = postfix
        self._render()

    def _render(self) -> None:
        total = max(1, self._total)
        width = 24
        filled = int(width * self._count / total)
        bar = "#" * filled + "-" * (width - filled)
        message = (
            f"\r{self._desc} [{bar}] {self._count}/{self._total} {self._unit}"
            + (f" | {self._postfix}" if self._postfix else "")
        )
        sys.stderr.write(message)
        sys.stderr.flush()

    def _finish(self) -> None:
        sys.stderr.write("\n")
        sys.stderr.flush()


def _load_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame(columns=["from_id", "resolved_gene_symbol", "to_id", "source", "status", "resolved_at"])
    df = pd.read_csv(cache_path, sep="\t")
    required = {"from_id", "to_id", "source", "status"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Protein Atlas cache missing columns: {sorted(missing)}")
    return df


def _write_cache(df: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, sep="\t", index=False)


def filter_symbols_for_protein_atlas(symbols: list[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    skipped: list[str] = []
    for symbol in [str(symbol).strip() for symbol in symbols if str(symbol).strip()]:
        if any(pattern.search(symbol) for pattern in NON_HPA_SYMBOL_PATTERNS):
            skipped.append(symbol)
        else:
            kept.append(symbol)
    return kept, skipped


def resolve_symbols_with_protein_atlas(symbols: list[str], client, cache_path: Path) -> tuple[dict[str, str], dict[str, int]]:
    cache_df = _load_cache(cache_path)
    cache_df = cache_df.drop_duplicates(subset=["from_id"], keep="first")
    cache_lookup = cache_df.set_index("from_id").to_dict(orient="index") if not cache_df.empty else {}

    lookup: dict[str, str] = {}
    loaded_from_cache = 0
    resolved_live = 0
    unresolved_live = 0

    rows = cache_df.to_dict(orient="records") if not cache_df.empty else []
    all_symbols = sorted({str(symbol).strip() for symbol in symbols if str(symbol).strip()})
    filtered_symbols, skipped_symbols = filter_symbols_for_protein_atlas(all_symbols)

    for symbol in skipped_symbols:
        rows.append(
            {
                "from_id": symbol,
                "resolved_gene_symbol": symbol,
                "to_id": "",
                "source": "protein_atlas",
                "status": "skipped_prefilter",
                "resolved_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    if skipped_symbols:
        written_df = pd.DataFrame(rows).drop_duplicates(subset=["from_id"], keep="last")
        _write_cache(written_df, cache_path)

    if _tqdm is not None:
        progress = _tqdm(
            filtered_symbols,
            desc="ProteinAtlas fallback",
            unit="symbol",
            leave=False,
        )
    else:
        progress = _SimpleProgressBar(
            filtered_symbols,
            desc="ProteinAtlas fallback",
            unit="symbol",
        )
    for symbol in progress:
        cached = cache_lookup.get(symbol)
        if cached is not None:
            if str(cached.get("status")) == "resolved" and str(cached.get("to_id", "")).strip():
                lookup[symbol] = str(cached["to_id"])
            loaded_from_cache += 1
            if hasattr(progress, "set_postfix_str"):
                progress.set_postfix_str(f"cache={loaded_from_cache} resolved={resolved_live} unresolved={unresolved_live}")
            continue

        result = client.fetch_gene_record(symbol)
        resolved_at = datetime.now(timezone.utc).isoformat()
        if result is None:
            rows.append(
                {
                    "from_id": symbol,
                    "resolved_gene_symbol": symbol,
                    "to_id": "",
                    "source": "protein_atlas",
                    "status": "unresolved",
                    "resolved_at": resolved_at,
                }
            )
            unresolved_live += 1
            written_df = pd.DataFrame(rows).drop_duplicates(subset=["from_id"], keep="last")
            _write_cache(written_df, cache_path)
            if hasattr(progress, "set_postfix_str"):
                progress.set_postfix_str(f"cache={loaded_from_cache} resolved={resolved_live} unresolved={unresolved_live}")
            continue

        rows.append(
            {
                "from_id": result["from_id"],
                "resolved_gene_symbol": result["resolved_gene_symbol"],
                "to_id": result["to_id"],
                "source": result["source"],
                "status": result["status"],
                "resolved_at": resolved_at,
            }
        )
        lookup[symbol] = result["to_id"]
        resolved_live += 1
        LOGGER.info("Protein Atlas resolved symbol %s -> %s", symbol, result["to_id"])
        written_df = pd.DataFrame(rows).drop_duplicates(subset=["from_id"], keep="last")
        _write_cache(written_df, cache_path)
        if hasattr(progress, "set_postfix_str"):
            progress.set_postfix_str(f"cache={loaded_from_cache} resolved={resolved_live} unresolved={unresolved_live}")

    written_df = pd.DataFrame(rows).drop_duplicates(subset=["from_id"], keep="last")
    _write_cache(written_df, cache_path)
    LOGGER.info(
        "Protein Atlas resolution summary: total=%d, filtered_out=%d, loaded_from_cache=%d, resolved_live=%d, unresolved_live=%d",
        len(all_symbols),
        len(skipped_symbols),
        loaded_from_cache,
        resolved_live,
        unresolved_live,
    )
    return lookup, {
        "loaded_from_cache": loaded_from_cache,
        "resolved_live": resolved_live,
        "unresolved_live": unresolved_live,
        "filtered_out": len(skipped_symbols),
        "total_symbols": len(all_symbols),
    }
