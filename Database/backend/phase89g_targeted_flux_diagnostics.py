from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from phase89g_targeted_flux_analysis import (
    FluxAnalysisPlanError,
    _db_file_path,
    _normalise_code,
    _open_read_only,
    _planned_insert_ids,
    _safe_source_path,
    _sha256_file,
    default_layout_resolver,
    load_plan,
    plan_sha256,
)

TensorInspector = Callable[[Path], Mapping[str, Any]]
Analyzer = Callable[[Path, str], Mapping[str, Any]]


def _key_prefix(key: str) -> str:
    text = str(key or "")
    if "." in text:
        return ".".join(text.split(".")[:3])
    parts = text.split("_")
    return "_".join(parts[:4]) if len(parts) > 1 else text


def default_tensor_inspector(path: Path) -> Mapping[str, Any]:
    from clip_contribution import is_clip_contributor
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as safetensors_file:
        keys = sorted(str(key) for key in safetensors_file.keys())

    clip_contributor, clip_tensor_count = is_clip_contributor(keys)
    prefixes = Counter(_key_prefix(key) for key in keys)
    return {
        "tensor_key_count": len(keys),
        "tensor_key_sample": keys[:25],
        "tensor_key_prefix_counts": [
            {"prefix": prefix, "count": count}
            for prefix, count in prefixes.most_common(20)
        ],
        "clip_contributor": bool(clip_contributor),
        "clip_tensor_count": int(clip_tensor_count),
    }


def default_analyzer(path: Path, base_model_code: str) -> Mapping[str, Any]:
    from delta_inspector_engine import inspect_lora

    return inspect_lora(str(path), base_model_code=base_model_code)


def build_flux_diagnostics(
    plan: Mapping[str, Any],
    *,
    library_root: str | os.PathLike[str],
    db_path: str | os.PathLike[str],
    db_path_root: str = "/loras",
    expected_count: int = 3,
    tensor_inspector: TensorInspector | None = None,
    analyzer: Analyzer | None = None,
) -> dict[str, Any]:
    if plan.get("audit_mode") != "read-only":
        raise FluxAnalysisPlanError(
            "Phase 8.9g diagnostics accept only a Phase 8.9d read-only plan"
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

    tensor_inspector_fn = tensor_inspector or default_tensor_inspector
    analyzer_fn = analyzer or default_analyzer

    conn = _open_read_only(db_path)
    try:
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity.casefold() != "ok":
            raise FluxAnalysisPlanError(
                f"Database integrity check failed: {integrity}"
            )

        targets: list[dict[str, Any]] = []
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
                    f"Planned stable ID {stable_id} does not match "
                    f"{expected_prefix} for {relative}"
                )

            source = _safe_source_path(root, relative)
            db_file_path = _db_file_path(db_path_root, relative)

            if conn.execute(
                "SELECT 1 FROM lora WHERE file_path = ?", (db_file_path,)
            ).fetchone() is not None:
                raise FluxAnalysisPlanError(
                    f"Target file_path already exists in DB: {db_file_path}"
                )
            if conn.execute(
                "SELECT 1 FROM lora WHERE stable_id = ?", (stable_id,)
            ).fetchone() is not None:
                raise FluxAnalysisPlanError(
                    f"Planned stable ID already exists in DB: {stable_id}"
                )

            tensor_info: dict[str, Any] = {}
            tensor_error: dict[str, str] | None = None
            try:
                tensor_info = dict(tensor_inspector_fn(source))
            except Exception as exc:
                tensor_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }

            analysis: dict[str, Any] = {}
            analysis_error: dict[str, str] | None = None
            if tensor_error is None:
                try:
                    analysis = dict(analyzer_fn(source, "FLX"))
                except Exception as exc:
                    analysis_error = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }

            block_weights = [
                float(value) for value in (analysis.get("block_weights") or [])
            ]
            raw_strengths = [
                float(value)
                for value in (analysis.get("raw_block_strengths") or [])
            ]
            block_layout = (
                default_layout_resolver(
                    analysis.get("lora_type"), len(block_weights)
                )
                if analysis_error is None and tensor_error is None
                else None
            )

            warnings: list[str] = []
            if tensor_error is not None:
                warnings.append(
                    "Tensor-key inspection failed; candidate is blocked"
                )
            if analysis_error is not None:
                warnings.append(
                    "Flux/UNet analysis did not recognise this tensor structure"
                )
            if block_weights and block_layout is None:
                warnings.append(
                    f"No supported block layout for {len(block_weights)} analysed blocks"
                )
            if raw_strengths and len(raw_strengths) != len(block_weights):
                warnings.append(
                    "raw_block_strengths count does not match block_weights count"
                )

            ready = not warnings
            targets.append(
                {
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
                    **tensor_info,
                    "tensor_inspection_error": tensor_error,
                    "analysis_error": analysis_error,
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
            )

        ready_count = sum(
            1 for target in targets if target["ready_for_controlled_apply"]
        )
        return {
            "phase": "8.9g-diagnostics",
            "mode": "read-only resilient targeted FLX diagnostics",
            "plan_sha256": plan_sha256(plan),
            "summary": {
                "flux_candidates": len(targets),
                "ready_for_controlled_apply": ready_count,
                "blocked_candidates": len(targets) - ready_count,
                "tensor_inspection_errors": sum(
                    1 for target in targets
                    if target.get("tensor_inspection_error") is not None
                ),
                "analysis_errors": sum(
                    1 for target in targets
                    if target.get("analysis_error") is not None
                ),
                "total_analysed_block_rows": sum(
                    int(target["block_count"]) for target in targets
                ),
            },
            "targets": targets,
            "safety": {
                "database_open_mode": (
                    "SQLite URI mode=ro plus PRAGMA query_only=ON"
                ),
                "writes_database": False,
                "runs_full_indexer": False,
                "discovers_library_files": False,
                "opens_only_plan_listed_safetensors": True,
                "assigns_stable_ids": False,
                "deletes_rows": False,
                "continues_after_per_file_analysis_error": True,
            },
        }
    finally:
        conn.close()


def print_diagnostics(result: Mapping[str, Any]) -> None:
    summary = result["summary"]
    print("=== Phase 8.9g targeted FLX diagnostics ===")
    print(f"Mode                       : {result['mode']}")
    print(f"Plan SHA-256               : {result['plan_sha256']}")
    print(f"FLX candidates             : {summary['flux_candidates']}")
    print(
        "Ready for controlled apply : "
        f"{summary['ready_for_controlled_apply']}"
    )
    print(f"Blocked candidates         : {summary['blocked_candidates']}")
    print(f"Tensor inspection errors   : {summary['tensor_inspection_errors']}")
    print(f"Analysis errors            : {summary['analysis_errors']}")
    print()
    for target in result["targets"]:
        print(target["relative_path"])
        print(f"  stable_id       : {target['planned_stable_id']}")
        print(f"  sha256          : {target['source_sha256']}")
        print(f"  tensor keys     : {target.get('tensor_key_count')}")
        print(
            "  analysis        : "
            f"family={target.get('model_family')!r}, "
            f"type={target.get('lora_type')!r}, "
            f"rank={target.get('rank')!r}"
        )
        print(
            "  blocks          : "
            f"{target['block_count']} ({target.get('block_layout')})"
        )
        print(f"  analysis error  : {target.get('analysis_error')}")
        print(f"  ready           : {target['ready_for_controlled_apply']}")
        for warning in target["warnings"]:
            print(f"  warning         : {warning}")
        print()
    print("No database changes were made.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run resilient read-only diagnostics for the three Phase 8.9g FLX targets"
        )
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--db-path-root", default="/loras")
    parser.add_argument("--expected-count", type=int, default=3)
    parser.add_argument("--json")
    args = parser.parse_args()

    result = build_flux_diagnostics(
        load_plan(args.plan),
        library_root=args.root,
        db_path=args.db,
        db_path_root=args.db_path_root,
        expected_count=args.expected_count,
    )
    print_diagnostics(result)

    if args.json:
        output = Path(args.json).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"JSON diagnostics written to: {output}")


if __name__ == "__main__":
    main()
