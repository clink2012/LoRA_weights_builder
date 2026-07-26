from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping
from urllib.parse import quote


class FluxAnalysisPlanError(RuntimeError):
    pass


Analyzer = Callable[[Path, str], Mapping[str, Any]]
TensorReader = Callable[[Path], tuple[bool, int, int]]
LayoutResolver = Callable[[str | None, int], str | None]


def load_plan(path: str | os.PathLike[str]) -> dict[str, Any]:
    plan_path = Path(path).expanduser().resolve(strict=True)
    with plan_path.open("r", encoding="utf-8") as handle:
        plan = json.load(handle)
    if not isinstance(plan, dict):
        raise FluxAnalysisPlanError("Plan JSON must contain an object at the top level")
    return plan


def canonical_plan_bytes(plan: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plan,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def plan_sha256(plan: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_plan_bytes(plan)).hexdigest()


def _normalise_code(value: Any) -> str:
    return str(value or "").strip().upper()


def _open_read_only(path: str | os.PathLike[str]) -> sqlite3.Connection:
    resolved = Path(path).expanduser().resolve(strict=True)
    uri_path = quote(resolved.as_posix(), safe="/:")
    conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _db_file_path(db_root: str, relative: str) -> str:
    root = str(db_root or "").replace("\\", "/").rstrip("/")
    rel = PurePosixPath(relative).as_posix().lstrip("/")
    return f"{root}/{rel}" if root else rel


def _planned_insert_ids(plan: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in plan.get("planned_stable_ids", []):
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("source_type") or "") != "new_metadata_insert":
            continue
        relative = str(raw.get("relative_path") or "").strip()
        stable_id = str(raw.get("planned_stable_id") or "").strip().upper()
        if not relative or not stable_id:
            continue
        key = PurePosixPath(relative).as_posix().casefold()
        if key in result and result[key] != stable_id:
            raise FluxAnalysisPlanError(
                f"Plan contains conflicting stable IDs for {relative}"
            )
        result[key] = stable_id
    return result


def _safe_source_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise FluxAnalysisPlanError(f"Unsafe relative path in plan: {relative}")
    source = root.joinpath(*pure.parts).resolve(strict=True)
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise FluxAnalysisPlanError(
            f"Plan path escapes the approved library root: {relative}"
        ) from exc
    if not source.is_file() or source.suffix.casefold() != ".safetensors":
        raise FluxAnalysisPlanError(f"Flux target is not a safetensors file: {source}")
    return source


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_tensor_reader(path: Path) -> tuple[bool, int, int]:
    from clip_contribution import is_clip_contributor
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as safetensors_file:
        tensor_keys = list(safetensors_file.keys())
    clip_contributor, clip_tensor_count = is_clip_contributor(tensor_keys)
    return bool(clip_contributor), int(clip_tensor_count), len(tensor_keys)


def default_analyzer(path: Path, base_model_code: str) -> Mapping[str, Any]:
    from delta_inspector_engine import inspect_lora

    return inspect_lora(str(path), base_model_code=base_model_code)


def default_layout_resolver(lora_type: str | None, block_count: int) -> str | None:
    from block_layouts import (
        FLUX_FALLBACK_16,
        make_flux_layout,
        normalize_block_layout,
    )

    if block_count <= 0:
        return FLUX_FALLBACK_16

    layout = normalize_block_layout(make_flux_layout(lora_type, block_count))
    if layout is None:
        layout = normalize_block_layout(f"flux_transformer_{block_count}")
    return layout


def build_flux_analysis_plan(
    plan: Mapping[str, Any],
    *,
    library_root: str | os.PathLike[str],
    db_path: str | os.PathLike[str],
    db_path_root: str = "/loras",
    expected_count: int = 3,
    analyzer: Analyzer | None = None,
    tensor_reader: TensorReader | None = None,
    layout_resolver: LayoutResolver | None = None,
) -> dict[str, Any]:
    if plan.get("audit_mode") != "read-only":
        raise FluxAnalysisPlanError(
            "Phase 8.9g accepts only a Phase 8.9d read-only plan"
        )

    safety = plan.get("safety") or {}
    if safety.get("writes_database") is not False:
        raise FluxAnalysisPlanError(
            "Plan safety flags do not describe a read-only planner run"
        )

    blockers = {
        "unresolved_relocations": plan.get("unresolved_relocations") or [],
        "stable_id_groups_exhausted": plan.get("stable_id_groups_exhausted") or [],
        "existing_stable_id_issues": plan.get("existing_stable_id_issues") or [],
    }
    active_blockers = [name for name, values in blockers.items() if values]
    if active_blockers:
        raise FluxAnalysisPlanError(
            "Plan has unresolved blockers: " + ", ".join(sorted(active_blockers))
        )

    candidates = [
        dict(raw)
        for raw in plan.get("new_metadata_insert_candidates", [])
        if isinstance(raw, Mapping)
        and _normalise_code(raw.get("base_model_code")) == "FLX"
    ]
    if len(candidates) != int(expected_count):
        raise FluxAnalysisPlanError(
            f"Expected exactly {expected_count} FLX candidates, found {len(candidates)}"
        )

    ids = _planned_insert_ids(plan)
    root = Path(library_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise FluxAnalysisPlanError(f"Library root is not a directory: {root}")

    analyzer_fn = analyzer or default_analyzer
    tensor_reader_fn = tensor_reader or default_tensor_reader
    layout_resolver_fn = layout_resolver or default_layout_resolver

    conn = _open_read_only(db_path)
    try:
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity.casefold() != "ok":
            raise FluxAnalysisPlanError(
                f"Database integrity check failed: {integrity}"
            )

        results: list[dict[str, Any]] = []
        for candidate in sorted(
            candidates,
            key=lambda item: str(item.get("relative_path") or "").casefold(),
        ):
            if str(candidate.get("source_type") or "") != "new_metadata_insert":
                raise FluxAnalysisPlanError(
                    "FLX candidate is not marked as new_metadata_insert"
                )

            relative = PurePosixPath(
                str(candidate.get("relative_path") or "")
            ).as_posix()
            if not relative or relative == ".":
                raise FluxAnalysisPlanError(
                    "FLX candidate is missing relative_path"
                )

            category_code = _normalise_code(candidate.get("category_code"))
            if not category_code:
                raise FluxAnalysisPlanError(
                    f"FLX candidate has no category code: {relative}"
                )

            stable_id = ids.get(relative.casefold())
            if not stable_id:
                raise FluxAnalysisPlanError(
                    f"No planned stable ID for FLX candidate: {relative}"
                )
            expected_prefix = f"FLX-{category_code}-"
            if not stable_id.startswith(expected_prefix):
                raise FluxAnalysisPlanError(
                    f"Planned stable ID {stable_id} does not match {expected_prefix} for {relative}"
                )

            source = _safe_source_path(root, relative)
            db_file_path = _db_file_path(db_path_root, relative)

            existing_path = conn.execute(
                "SELECT id, stable_id FROM lora WHERE file_path = ?",
                (db_file_path,),
            ).fetchone()
            if existing_path is not None:
                raise FluxAnalysisPlanError(
                    f"Target file_path already exists in DB: {db_file_path}"
                )

            existing_id = conn.execute(
                "SELECT id, file_path FROM lora WHERE stable_id = ?",
                (stable_id,),
            ).fetchone()
            if existing_id is not None:
                raise FluxAnalysisPlanError(
                    f"Planned stable ID already exists in DB: {stable_id}"
                )

            clip_contributor, clip_tensor_count, tensor_key_count = tensor_reader_fn(
                source
            )
            analysis = dict(analyzer_fn(source, "FLX"))
            block_weights = [
                float(value) for value in (analysis.get("block_weights") or [])
            ]
            raw_strengths = [
                float(value)
                for value in (analysis.get("raw_block_strengths") or [])
            ]
            block_layout = layout_resolver_fn(
                analysis.get("lora_type"),
                len(block_weights),
            )

            warnings: list[str] = []
            if block_weights and block_layout is None:
                warnings.append(
                    f"No supported block layout for {len(block_weights)} analysed blocks"
                )
            if raw_strengths and len(raw_strengths) != len(block_weights):
                warnings.append(
                    "raw_block_strengths count does not match block_weights count"
                )

            ready = not warnings
            result = {
                "relative_path": relative,
                "db_file_path": db_file_path,
                "filename": candidate.get("filename") or source.name,
                "planned_stable_id": stable_id,
                "base_model_name": candidate.get("base_model_name"),
                "base_model_code": "FLX",
                "category_name": candidate.get("category_name"),
                "category_code": category_code,
                "source_size_bytes": source.stat().st_size,
                "source_mtime": source.stat().st_mtime,
                "source_sha256": _sha256_file(source),
                "tensor_key_count": int(tensor_key_count),
                "clip_contributor": bool(clip_contributor),
                "clip_tensor_count": int(clip_tensor_count),
                "model_family": analysis.get("model_family"),
                "lora_type": analysis.get("lora_type"),
                "rank": analysis.get("rank"),
                "has_block_weights": bool(block_weights),
                "block_count": len(block_weights),
                "raw_strength_count": len(raw_strengths),
                "block_layout": block_layout,
                "block_weight_min": min(block_weights) if block_weights else None,
                "block_weight_max": max(block_weights) if block_weights else None,
                "block_weight_mean": (
                    sum(block_weights) / len(block_weights)
                    if block_weights
                    else None
                ),
                "warnings": warnings,
                "ready_for_controlled_apply": ready,
            }
            results.append(result)

        ready_count = sum(
            1 for result in results if result["ready_for_controlled_apply"]
        )
        return {
            "phase": "8.9g",
            "mode": "read-only targeted FLX analysis",
            "plan_sha256": plan_sha256(plan),
            "summary": {
                "flux_candidates": len(results),
                "ready_for_controlled_apply": ready_count,
                "blocked_candidates": len(results) - ready_count,
                "total_analysed_block_rows": sum(
                    int(result["block_count"]) for result in results
                ),
            },
            "targets": results,
            "safety": {
                "database_open_mode": "SQLite URI mode=ro plus PRAGMA query_only=ON",
                "writes_database": False,
                "runs_full_indexer": False,
                "discovers_library_files": False,
                "opens_only_plan_listed_safetensors": True,
                "assigns_stable_ids": False,
                "deletes_rows": False,
            },
        }
    finally:
        conn.close()


def print_analysis_plan(result: Mapping[str, Any]) -> None:
    summary = result["summary"]
    print("=== Phase 8.9g targeted FLX analysis ===")
    print(f"Mode                       : {result['mode']}")
    print(f"Plan SHA-256               : {result['plan_sha256']}")
    print(f"FLX candidates             : {summary['flux_candidates']}")
    print(
        "Ready for controlled apply : "
        f"{summary['ready_for_controlled_apply']}"
    )
    print(f"Blocked candidates         : {summary['blocked_candidates']}")
    print(
        "Total analysed block rows  : "
        f"{summary['total_analysed_block_rows']}"
    )
    print()
    for target in result["targets"]:
        print(target["relative_path"])
        print(f"  stable_id       : {target['planned_stable_id']}")
        print(f"  sha256          : {target['source_sha256']}")
        print(
            "  analysis        : "
            f"family={target['model_family']!r}, "
            f"type={target['lora_type']!r}, rank={target['rank']!r}"
        )
        print(
            "  blocks          : "
            f"{target['block_count']} ({target['block_layout']})"
        )
        print(
            "  clip            : "
            f"{target['clip_contributor']} "
            f"({target['clip_tensor_count']} tensors)"
        )
        print(
            "  controlled apply: "
            f"{target['ready_for_controlled_apply']}"
        )
        for warning in target["warnings"]:
            print(f"  warning         : {warning}")
    print()
    print("No database changes were made.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only analysis of the exact FLX insert candidates held out "
            "of the Phase 8.9e metadata reconciliation"
        )
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--db-path-root", default="/loras")
    parser.add_argument("--expected-count", type=int, default=3)
    parser.add_argument("--json")
    args = parser.parse_args()

    plan = load_plan(args.plan)
    result = build_flux_analysis_plan(
        plan,
        library_root=args.root,
        db_path=args.db,
        db_path_root=args.db_path_root,
        expected_count=args.expected_count,
    )
    print_analysis_plan(result)

    if args.json:
        output = Path(args.json).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"JSON analysis written to: {output}")


if __name__ == "__main__":
    main()
