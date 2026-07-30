from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from drug_disease_validation.src.utils import configure_logging, ensure_parent_dir, read_pickle, write_pickle


logger = logging.getLogger(__name__)

SHARD_FORMAT = "subgraph-shard-v1"
DEFAULT_SHARD_SIZE = 1000


def _resolve_existing_step04_manifest(repo_root: Path) -> Path:
    candidates = [
        repo_root
        / "drug_disease_validation"
        / "data"
        / "processed_step04_external"
        / "subgraph_manifest_io+emb1024.tsv",
        repo_root / "drug_disease_validation" / "data" / "processed" / "subgraph_manifest_io+emb1024.tsv",
        repo_root / "drug_disease_validation" / "data" / "processed" / "subgraph_manifest_io.tsv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _variant_from_manifest(manifest_path: Path) -> str:
    stem = manifest_path.stem
    prefix = "subgraph_manifest_"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    return stem or "subgraphs"


def _default_output_manifest(manifest_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{manifest_path.stem}_sharded.tsv"


def _default_shards_dir(manifest_path: Path, output_dir: Path) -> Path:
    return output_dir / f"subgraph_shards_{_variant_from_manifest(manifest_path)}"


def _resolve_manifest_subgraph_path(raw_path: Any, manifest_path: Path) -> Path:
    text = str(raw_path).strip()
    if not text or text.lower() == "nan":
        raise ValueError("Manifest row has an empty SubgraphPath.")

    path = Path(text)
    if path.is_absolute():
        return path

    manifest_dir = manifest_path.parent
    candidates = [manifest_dir / path, path]
    if path.parent.name and path.name:
        candidates.append(manifest_dir / path.parent.name / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _format_path_for_manifest(path: Path, output_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(output_dir.resolve()))
    except ValueError:
        return str(path)


def _appledouble_sidecar(path: Path) -> Path:
    return path.with_name(f"._{path.name}")


def _load_subgraph(path: Path, subgraph_id: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Subgraph file for {subgraph_id} does not exist: {path}")
    try:
        return read_pickle(path)
    except Exception as exc:
        raise ValueError(f"Could not read subgraph pickle for {subgraph_id}: {path} ({type(exc).__name__}: {exc})") from exc


def _load_shard_graph_keys(shard_path: Path) -> set[str]:
    payload = read_pickle(shard_path)
    if isinstance(payload, dict) and "graphs" in payload and isinstance(payload["graphs"], dict):
        return {str(key) for key in payload["graphs"].keys()}
    if isinstance(payload, dict):
        return {str(key) for key in payload.keys()}
    raise ValueError(f"Existing shard does not contain a graph mapping: {shard_path}")


def build_step04_1_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    default_manifest = _resolve_existing_step04_manifest(default_repo_root)
    default_output_dir = default_manifest.parent
    parser = argparse.ArgumentParser(
        description="Step 4.1: compact Step 4 per-subgraph pickle files into larger shard pickle files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", default=str(default_manifest), help="Step 4 manifest with one pickle per row.")
    parser.add_argument("--output-dir", default=str(default_output_dir), help="Directory for the sharded manifest and report.")
    parser.add_argument(
        "--output-manifest",
        default=None,
        help="Output manifest path. Defaults to <output-dir>/<input-manifest-stem>_sharded.tsv.",
    )
    parser.add_argument("--shards-dir", default=None, help="Directory for shard pickle files.")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE, help="Number of subgraphs per shard.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing shard files and output manifest.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing shard files and continue building the output manifest instead of failing on them.",
    )
    parser.add_argument(
        "--skip-unreadable",
        action="store_true",
        help="Skip missing, empty, or corrupt subgraph pickle files and record them in the report.",
    )
    parser.add_argument(
        "--max-skipped-warnings",
        type=int,
        default=20,
        help="Maximum number of skipped subgraph warnings to print when --skip-unreadable is used.",
    )
    parser.add_argument(
        "--stop-after-consecutive-skips",
        type=int,
        default=0,
        help="Stop early after this many consecutive skipped subgraphs. Use 0 to scan the whole manifest.",
    )
    parser.add_argument(
        "--delete-originals",
        action="store_true",
        help="After all shards and the manifest are written, delete the original per-subgraph .pkl files.",
    )
    parser.add_argument(
        "--delete-appledouble",
        action="store_true",
        help="When deleting originals, also delete matching macOS AppleDouble files named ._<subgraph>.pkl.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def shard_subgraphs(
    *,
    manifest_path: Path,
    output_dir: Path,
    output_manifest_path: Path | None = None,
    shards_dir: Path | None = None,
    shard_size: int = DEFAULT_SHARD_SIZE,
    overwrite: bool = False,
    resume: bool = False,
    skip_unreadable: bool = False,
    max_skipped_warnings: int = 20,
    stop_after_consecutive_skips: int = 0,
    delete_originals: bool = False,
    delete_appledouble: bool = False,
) -> dict[str, Any]:
    if shard_size < 1:
        raise ValueError("shard_size must be >= 1")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_path = output_manifest_path or _default_output_manifest(manifest_path, output_dir)
    shards_dir = shards_dir or _default_shards_dir(manifest_path, output_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    if output_manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Output manifest already exists: {output_manifest_path}. Use --overwrite.")

    manifest = pd.read_csv(manifest_path, sep="\t", low_memory=False)
    if "SubgraphID" not in manifest.columns or "SubgraphPath" not in manifest.columns:
        raise ValueError("Manifest must contain SubgraphID and SubgraphPath columns.")

    sharded_rows: list[dict[str, Any]] = []
    shard_reports: list[dict[str, Any]] = []
    skipped_subgraphs: list[dict[str, Any]] = []
    original_paths: list[Path] = []
    rows = manifest.to_dict(orient="records")
    consecutive_skips = 0
    stopped_early = False

    for shard_index, start in enumerate(range(0, len(rows), shard_size)):
        chunk = rows[start : start + shard_size]
        shard_name = f"subgraph_shard_{shard_index:05d}.pkl"
        shard_path = shards_dir / shard_name
        if shard_path.exists() and resume:
            shard_graph_keys = _load_shard_graph_keys(shard_path)
            shard_bytes = shard_path.stat().st_size
            shard_manifest_path = _format_path_for_manifest(shard_path, output_dir)
            reused_rows = 0
            for offset, row in enumerate(chunk):
                subgraph_id = str(row["SubgraphID"])
                if subgraph_id not in shard_graph_keys:
                    skipped_subgraphs.append(
                        {
                            "row_index": start + offset,
                            "subgraph_id": subgraph_id,
                            "path": str(row.get("SubgraphPath", "")),
                            "reason": "missing_from_existing_shard",
                        }
                    )
                    consecutive_skips += 1
                    if stop_after_consecutive_skips and consecutive_skips >= stop_after_consecutive_skips:
                        logger.warning(
                            "Stopping early after %s consecutive skipped subgraphs at row_index=%s.",
                            consecutive_skips,
                            start + offset,
                        )
                        stopped_early = True
                        break
                    continue
                sharded_row = dict(row)
                sharded_row["SubgraphOriginalPath"] = row["SubgraphPath"]
                sharded_row["SubgraphPath"] = shard_manifest_path
                sharded_row["SubgraphStorage"] = "pickle_shard"
                sharded_row["SubgraphShardPath"] = shard_manifest_path
                sharded_row["SubgraphShardKey"] = subgraph_id
                sharded_rows.append(sharded_row)
                reused_rows += 1
                consecutive_skips = 0
            shard_reports.append(
                {
                    "shard_index": shard_index,
                    "path": str(shard_path),
                    "rows": reused_rows,
                    "bytes": int(shard_bytes),
                    "reused": True,
                }
            )
            logger.info("Shard reused: %s rows=%s size_mb=%.2f", shard_path, reused_rows, shard_bytes / 1_000_000)
            if stopped_early:
                break
            continue
        if shard_path.exists() and not overwrite:
            raise FileExistsError(f"Shard already exists: {shard_path}. Use --overwrite.")

        graphs: dict[str, Any] = {}
        shard_rows: list[dict[str, Any]] = []
        shard_original_paths: list[Path] = []
        for offset, row in enumerate(chunk):
            subgraph_id = str(row["SubgraphID"])
            original_path = _resolve_manifest_subgraph_path(row["SubgraphPath"], manifest_path)
            try:
                graphs[subgraph_id] = _load_subgraph(original_path, subgraph_id)
            except Exception as exc:
                if not skip_unreadable:
                    raise
                skipped = {
                    "row_index": start + offset,
                    "subgraph_id": subgraph_id,
                    "path": str(original_path),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                skipped_subgraphs.append(skipped)
                if len(skipped_subgraphs) <= max_skipped_warnings:
                    logger.warning("Skipped unreadable subgraph: %s", skipped)
                elif len(skipped_subgraphs) == max_skipped_warnings + 1:
                    logger.warning("Further skipped subgraph warnings suppressed.")
                consecutive_skips += 1
                if stop_after_consecutive_skips and consecutive_skips >= stop_after_consecutive_skips:
                    logger.warning(
                        "Stopping early after %s consecutive skipped subgraphs at row_index=%s.",
                        consecutive_skips,
                        start + offset,
                    )
                    stopped_early = True
                    break
                continue
            original_paths.append(original_path)
            shard_rows.append(row)
            shard_original_paths.append(original_path)
            consecutive_skips = 0

        if not graphs:
            logger.info("Shard skipped because it had no readable subgraphs: %s", shard_path)
            if stopped_early:
                break
            continue

        payload = {
            "format": SHARD_FORMAT,
            "metadata": {
                "source_manifest": str(manifest_path),
                "shard_index": shard_index,
                "shard_size": len(graphs),
                "subgraph_ids": list(graphs.keys()),
            },
            "graphs": graphs,
        }
        write_pickle(payload, shard_path)
        shard_bytes = shard_path.stat().st_size
        shard_manifest_path = _format_path_for_manifest(shard_path, output_dir)

        for row, original_path in zip(shard_rows, shard_original_paths):
            sharded_row = dict(row)
            sharded_row["SubgraphOriginalPath"] = row["SubgraphPath"]
            sharded_row["SubgraphPath"] = shard_manifest_path
            sharded_row["SubgraphStorage"] = "pickle_shard"
            sharded_row["SubgraphShardPath"] = shard_manifest_path
            sharded_row["SubgraphShardKey"] = str(row["SubgraphID"])
            sharded_rows.append(sharded_row)

        shard_reports.append(
            {
                "shard_index": shard_index,
                "path": str(shard_path),
                "rows": len(graphs),
                "bytes": int(shard_bytes),
                "reused": False,
            }
        )
        logger.info("Shard written: %s rows=%s size_mb=%.2f", shard_path, len(graphs), shard_bytes / 1_000_000)
        if stopped_early:
            break

    output_manifest_path = ensure_parent_dir(output_manifest_path)
    pd.DataFrame(sharded_rows).to_csv(output_manifest_path, sep="\t", index=False)

    deleted_originals = 0
    deleted_appledouble = 0
    if delete_originals:
        for original_path in sorted(set(original_paths)):
            if original_path.exists():
                original_path.unlink()
                deleted_originals += 1
            if delete_appledouble:
                sidecar = _appledouble_sidecar(original_path)
                if sidecar.exists():
                    sidecar.unlink()
                    deleted_appledouble += 1

    report = {
        "format": SHARD_FORMAT,
        "input_manifest": str(manifest_path),
        "output_manifest": str(output_manifest_path),
        "output_dir": str(output_dir),
        "shards_dir": str(shards_dir),
        "rows": int(len(sharded_rows)),
        "input_manifest_rows": int(len(rows)),
        "skipped_subgraphs": skipped_subgraphs,
        "n_skipped_subgraphs": int(len(skipped_subgraphs)),
        "shard_size": int(shard_size),
        "shards": shard_reports,
        "n_shards": int(len(shard_reports)),
        "total_shard_bytes": int(sum(item["bytes"] for item in shard_reports)),
        "delete_originals": bool(delete_originals),
        "delete_appledouble": bool(delete_appledouble),
        "deleted_original_files": int(deleted_originals),
        "deleted_appledouble_files": int(deleted_appledouble),
        "resume": bool(resume),
        "skip_unreadable": bool(skip_unreadable),
        "stop_after_consecutive_skips": int(stop_after_consecutive_skips),
        "stopped_early": bool(stopped_early),
    }
    report_path = output_manifest_path.with_name(f"{output_manifest_path.stem}_report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report"] = str(report_path)
    logger.info("Sharded manifest written: %s rows=%s", output_manifest_path, len(sharded_rows))
    logger.info("Step 4.1 report written: %s", report_path)
    return report


def run_step04_1(args: argparse.Namespace) -> None:
    configure_logging(getattr(args, "log_level", "INFO"))
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_manifest_path = Path(args.output_manifest) if getattr(args, "output_manifest", None) else None
    shards_dir = Path(args.shards_dir) if getattr(args, "shards_dir", None) else None
    logger.info("Step04.1 started")
    logger.info("Manifest: %s", manifest_path)
    report = shard_subgraphs(
        manifest_path=manifest_path,
        output_dir=output_dir,
        output_manifest_path=output_manifest_path,
        shards_dir=shards_dir,
        shard_size=int(args.shard_size),
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
        skip_unreadable=bool(args.skip_unreadable),
        max_skipped_warnings=int(args.max_skipped_warnings),
        stop_after_consecutive_skips=int(args.stop_after_consecutive_skips),
        delete_originals=bool(args.delete_originals),
        delete_appledouble=bool(args.delete_appledouble),
    )
    logger.info("Step04.1 completed: shards=%s rows=%s", report["n_shards"], report["rows"])
