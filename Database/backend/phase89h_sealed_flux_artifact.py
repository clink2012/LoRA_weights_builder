from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Callable, Mapping

from phase89g_targeted_flux_analysis import (
    FluxAnalysisPlanError,
    _open_read_only,
    _safe_source_path,
    _sha256_file,
    default_layout_resolver,
    load_plan,
    plan_sha256,
)
from phase89g_targeted_flux_diagnostics import default_tensor_inspector


class SealedArtifactError(RuntimeError):
    pass


Analyzer = Callable[[Path, str], Mapping[str, Any]]
TensorInspector = Callable[[Path], Mapping[str, Any]]


def load_json_object(path: str | os.PathLike[str], label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SealedArtifactError(f"{label} JSON must contain an object")
    return value


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def default_analyzer(path: Path, base_model_code: str) -> Mapping[str, Any]:
    from delta_inspector_engine import inspect_lora

    return inspect_lora(str(path), base_model_code=base_model_code)


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise SealedArtifactError(
            f"{label} mismatch: expected {expected!r}, found {actual!r}"
        )


def _ready_targets(diagnostics: Mapping[str, Any]) -> list[dict[str, Any]]:
    targets = diagnostics.get("targets") or []
    return [
        dict(target)
        for target in targets
        if isinstance(target, Mapping)
        and target.get("ready_for_controlled_apply") is True
    ]


def build_sealed_flux_artifact(
    plan: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    *,
    library_root: str | os.PathLike[str],
    db_path: str | os.PathLike[str],
    expected_stable_id: str = "FLX-PPL-207",
    expected_block_count: int = 57,
    expected_layout: str = "flux_unet_57",
    analyzer: Analyzer | None = None,
    tensor_inspector: TensorInspector | None = None,
) -> dict[str, Any]:
    if plan.get("audit_mode") != "read-only":
        raise SealedArtifactError("Phase 8.9h requires the Phase 8.9d read-only plan")

    expected_plan_sha = plan_sha256(plan)
    _require_equal(
        diagnostics.get("plan_sha256"),
        expected_plan_sha,
        "diagnostics plan_sha256",
    )
    _require_equal(diagnostics.get("phase"), "8.9g-diagnostics", "diagnostics phase")

    ready = _ready_targets(diagnostics)
    if len(ready) != 1:
        raise SealedArtifactError(
            f"Expected exactly one ready diagnostic target, found {len(ready)}"
        )

    target = ready[0]
    stable_id = str(target.get("planned_stable_id") or "").strip().upper()
    _require_equal(stable_id, expected_stable_id.upper(), "ready stable_id")
    _require_equal(target.get("base_model_code"), "FLX", "base_model_code")
    _require_equal(target.get("block_count"), int(expected_block_count), "diagnostic block_count")
    _require_equal(target.get("raw_strength_count"), int(expected_block_count), "diagnostic raw_strength_count")
    _require_equal(target.get("block_layout"), expected_layout, "diagnostic block_layout")
    _require_equal(target.get("warnings"), [], "diagnostic warnings")
    _require_equal(target.get("analysis_error"), None, "diagnostic analysis_error")
    _require_equal(
        target.get("tensor_inspection_error"),
        None,
        "diagnostic tensor_inspection_error",
    )

    relative_path = str(target.get("relative_path") or "").strip()
    if not relative_path:
        raise SealedArtifactError("Ready diagnostic target has no relative_path")

    root = Path(library_root).expanduser().resolve(strict=True)
    source = _safe_source_path(root, relative_path)
    source_sha = _sha256_file(source)
    _require_equal(source_sha, target.get("source_sha256"), "source SHA-256")

    db_file_path = str(target.get("db_file_path") or "").strip()
    if not db_file_path:
        raise SealedArtifactError("Ready diagnostic target has no db_file_path")

    conn = _open_read_only(db_path)
    try:
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity.casefold() != "ok":
            raise SealedArtifactError(
                f"Database integrity check failed: {integrity}"
            )
        if conn.execute(
            "SELECT 1 FROM lora WHERE file_path = ?", (db_file_path,)
        ).fetchone() is not None:
            raise SealedArtifactError(
                f"Target file_path already exists in DB: {db_file_path}"
            )
        if conn.execute(
            "SELECT 1 FROM lora WHERE stable_id = ?", (stable_id,)
        ).fetchone() is not None:
            raise SealedArtifactError(
                f"Planned stable ID already exists in DB: {stable_id}"
            )
    finally:
        conn.close()

    tensor_inspector_fn = tensor_inspector or default_tensor_inspector
    tensor_info = dict(tensor_inspector_fn(source))
    for field in (
        "tensor_key_count",
        "clip_contributor",
        "clip_tensor_count",
    ):
        _require_equal(tensor_info.get(field), target.get(field), f"tensor {field}")

    analyzer_fn = analyzer or default_analyzer
    analysis = dict(analyzer_fn(source, "FLX"))
    block_weights = [
        float(value) for value in (analysis.get("block_weights") or [])
    ]
    raw_strengths = [
        float(value) for value in (analysis.get("raw_block_strengths") or [])
    ]
    if len(block_weights) != int(expected_block_count):
        raise SealedArtifactError(
            f"Expected {expected_block_count} block weights, found {len(block_weights)}"
        )
    if len(raw_strengths) != len(block_weights):
        raise SealedArtifactError(
            "raw_block_strengths count does not match block_weights count"
        )

    layout = default_layout_resolver(analysis.get("lora_type"), len(block_weights))
    _require_equal(layout, expected_layout, "re-analysed block_layout")
    for field in ("model_family", "lora_type", "rank"):
        _require_equal(analysis.get(field), target.get(field), f"analysis {field}")

    payload: dict[str, Any] = {
        "phase": "8.9h",
        "mode": "read-only sealed single-target Flux artifact",
        "plan_sha256": expected_plan_sha,
        "diagnostics_sha256": canonical_sha256(diagnostics),
        "target": {
            "relative_path": relative_path,
            "db_file_path": db_file_path,
            "filename": target.get("filename") or source.name,
            "planned_stable_id": stable_id,
            "base_model_name": target.get("base_model_name"),
            "base_model_code": "FLX",
            "category_name": target.get("category_name"),
            "category_code": target.get("category_code"),
            "source_size_bytes": source.stat().st_size,
            "source_mtime": source.stat().st_mtime,
            "source_sha256": source_sha,
            "tensor_key_count": tensor_info.get("tensor_key_count"),
            "clip_contributor": bool(tensor_info.get("clip_contributor")),
            "clip_tensor_count": int(tensor_info.get("clip_tensor_count") or 0),
            "model_family": analysis.get("model_family"),
            "lora_type": analysis.get("lora_type"),
            "rank": analysis.get("rank"),
            "has_block_weights": True,
            "block_layout": layout,
            "block_count": len(block_weights),
            "block_weights": block_weights,
            "raw_block_strengths": raw_strengths,
        },
        "safety": {
            "database_open_mode": "SQLite URI mode=ro plus PRAGMA query_only=ON",
            "writes_database": False,
            "runs_full_indexer": False,
            "discovers_library_files": False,
            "opens_only_diagnostic_target": True,
            "assigns_stable_ids": False,
            "deletes_rows": False,
            "contains_apply_mode": False,
        },
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    return payload


def verify_artifact_digest(artifact: Mapping[str, Any]) -> str:
    stored = str(artifact.get("artifact_sha256") or "")
    unsigned = dict(artifact)
    unsigned.pop("artifact_sha256", None)
    calculated = canonical_sha256(unsigned)
    if stored != calculated:
        raise SealedArtifactError(
            f"Artifact digest mismatch: stored {stored}, calculated {calculated}"
        )
    return calculated


def print_artifact(artifact: Mapping[str, Any]) -> None:
    target = artifact["target"]
    print("=== Phase 8.9h sealed Flux artifact ===")
    print(f"Mode             : {artifact['mode']}")
    print(f"Plan SHA-256     : {artifact['plan_sha256']}")
    print(f"Diagnostics SHA  : {artifact['diagnostics_sha256']}")
    print(f"Artifact SHA-256 : {artifact['artifact_sha256']}")
    print(f"Stable ID        : {target['planned_stable_id']}")
    print(f"Source SHA-256   : {target['source_sha256']}")
    print(f"Block layout     : {target['block_layout']}")
    print(f"Block rows       : {target['block_count']}")
    print("No database changes were made.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a sealed read-only artifact for the single ready Phase 8.9g FLX target"
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--expected-stable-id", default="FLX-PPL-207")
    parser.add_argument("--expected-block-count", type=int, default=57)
    parser.add_argument("--expected-layout", default="flux_unet_57")
    parser.add_argument("--json")
    args = parser.parse_args()

    artifact = build_sealed_flux_artifact(
        load_plan(args.plan),
        load_json_object(args.diagnostics, "Diagnostics"),
        library_root=args.root,
        db_path=args.db,
        expected_stable_id=args.expected_stable_id,
        expected_block_count=args.expected_block_count,
        expected_layout=args.expected_layout,
    )
    verify_artifact_digest(artifact)
    print_artifact(artifact)

    if args.json:
        output = Path(args.json).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        print(f"JSON artifact written to: {output}")


if __name__ == "__main__":
    main()
