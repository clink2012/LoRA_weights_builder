from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping

from block_layouts import FLUX2_TRANSFORMER_56, expected_block_count_for_layout
from phase89g_targeted_flux_analysis import (
    _open_read_only,
    _safe_source_path,
    _sha256_file,
)


class Flux2LayoutError(RuntimeError):
    pass


EXPECTED_STABLE_ID = "FLX-STL-263"
EXPECTED_SOURCE_SHA256 = (
    "c60c9a5de39da23b3b4f4dca48e3511faa1fe5a4987d4acbb0a04643a9a65be7"
)
EXPECTED_PHASE89J_SHA256 = (
    "7c886a07e87fa36081645d34bd578001420e65a380c6543b17c9b9ee1fb8dc48"
)
EXPECTED_TENSOR_KEY_COUNT = 276
EXPECTED_RANK = 16
DOUBLE_BLOCK_COUNT = 8
SINGLE_BLOCK_COUNT = 48
TOTAL_BLOCK_COUNT = DOUBLE_BLOCK_COUNT + SINGLE_BLOCK_COUNT
EXPECTED_BLOCK_MODULE_COUNT = 128
EXPECTED_BLOCK_TENSOR_COUNT = 256
EXPECTED_GLOBAL_MODULE_COUNT = 10
EXPECTED_GLOBAL_TENSOR_COUNT = 20

EXPECTED_GLOBAL_MODULES = (
    "double_stream_modulation_img.lin",
    "double_stream_modulation_txt.lin",
    "final_layer.linear",
    "guidance_in.in_layer",
    "guidance_in.out_layer",
    "img_in",
    "single_stream_modulation.lin",
    "time_in.in_layer",
    "time_in.out_layer",
    "txt_in",
)

ALLOWED_GLOBAL_ROOTS = frozenset(
    {
        "guidance_in",
        "time_in",
        "double_stream_modulation_img",
        "double_stream_modulation_txt",
        "final_layer",
        "img_in",
        "single_stream_modulation",
        "txt_in",
    }
)

_BLOCK_KEY_RE = re.compile(
    r"^base_model\.model\.(double_blocks|single_blocks)\.(\d+)\."
    r"(.+)\.lora_([ab])\.weight$",
    re.IGNORECASE,
)
_GLOBAL_KEY_RE = re.compile(
    r"^base_model\.model\.(?!double_blocks\.|single_blocks\.)(.+)\."
    r"lora_([ab])\.weight$",
    re.IGNORECASE,
)
_PHASE89J_RANGE_BLOCKER_RE = re.compile(
    r"^Out-of-range single_blocks index (\d+); supported range is 0\.\.37$"
)

TensorAnalyser = Callable[[Path], Mapping[str, Any]]


def load_json_object(path: str | os.PathLike[str], label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Flux2LayoutError(f"{label} JSON must contain an object")
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


def _verify_embedded_digest(
    value: Mapping[str, Any],
    field: str,
    label: str,
) -> str:
    stored = str(value.get(field) or "")
    unsigned = dict(value)
    unsigned.pop(field, None)
    calculated = canonical_sha256(unsigned)
    if stored != calculated:
        raise Flux2LayoutError(
            f"{label} digest mismatch: stored {stored}, calculated {calculated}"
        )
    return calculated


def verify_artifact_digest(artifact: Mapping[str, Any]) -> str:
    return _verify_embedded_digest(
        artifact,
        "artifact_sha256",
        "Phase 8.9k artifact",
    )


def _tensor_norm(value: Any) -> float:
    norm = float(value.norm().item())
    if not math.isfinite(norm):
        raise Flux2LayoutError("Encountered non-finite tensor norm")
    return norm


def _pair_rank(
    module_name: str,
    pair: Mapping[str, Any],
    blockers: list[str],
) -> int | None:
    a = pair.get("a")
    b = pair.get("b")
    if a is None or b is None:
        missing = "A" if a is None else "B"
        blockers.append(
            f"Incomplete LoRA pair for {module_name}: missing lora_{missing}"
        )
        return None
    if int(a.ndim) != 2 or int(b.ndim) != 2:
        blockers.append(
            f"LoRA pair for {module_name} is not two-dimensional: "
            f"A{tuple(a.shape)}, B{tuple(b.shape)}"
        )
        return None
    rank_a = int(a.shape[0])
    rank_b = int(b.shape[1])
    if rank_a <= 0 or rank_b <= 0 or rank_a != rank_b:
        blockers.append(
            f"LoRA rank mismatch for {module_name}: "
            f"A rank {rank_a}, B rank {rank_b}"
        )
        return None
    return rank_a


def _normalise_strengths(raw_strengths: list[float]) -> list[float]:
    maximum = max(raw_strengths, default=0.0)
    if maximum <= 0:
        return [0.0 for _ in raw_strengths]
    return [round(value / maximum, 6) for value in raw_strengths]


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def analyse_flux2_tensor_map(tensors: Mapping[str, Any]) -> dict[str, Any]:
    block_modules: dict[tuple[str, int, str], dict[str, Any]] = {}
    global_modules: dict[str, dict[str, Any]] = {}
    unmatched_keys: list[str] = []

    for raw_name, tensor in sorted(
        tensors.items(),
        key=lambda item: str(item[0]).casefold(),
    ):
        name = str(raw_name)
        block_match = _BLOCK_KEY_RE.match(name)
        if block_match:
            stream = block_match.group(1).lower()
            index = int(block_match.group(2))
            module = block_match.group(3)
            side = block_match.group(4).lower()
            key = (stream, index, module)
            pair = block_modules.setdefault(key, {})
            if side in pair:
                raise Flux2LayoutError(
                    f"Duplicate lora_{side.upper()} tensor for "
                    f"{stream}.{index}.{module}"
                )
            pair[side] = tensor
            continue

        global_match = _GLOBAL_KEY_RE.match(name)
        if global_match:
            module = global_match.group(1)
            root = module.split(".", 1)[0].casefold()
            if root not in ALLOWED_GLOBAL_ROOTS:
                unmatched_keys.append(name)
                continue
            side = global_match.group(2).lower()
            pair = global_modules.setdefault(module, {})
            if side in pair:
                raise Flux2LayoutError(
                    f"Duplicate lora_{side.upper()} tensor for global module {module}"
                )
            pair[side] = tensor
            continue

        unmatched_keys.append(name)

    blockers: list[str] = []
    warnings: list[str] = []
    ranks: set[int] = set()
    double_strengths: dict[int, float] = {}
    single_strengths: dict[int, float] = {}

    for (stream, index, module), pair in sorted(block_modules.items()):
        max_index = (
            DOUBLE_BLOCK_COUNT - 1
            if stream == "double_blocks"
            else SINGLE_BLOCK_COUNT - 1
        )
        if index < 0 or index > max_index:
            _append_unique(
                blockers,
                f"Out-of-range {stream} index {index}; "
                f"supported range is 0..{max_index}",
            )

        module_name = f"{stream}.{index}.{module}"
        rank = _pair_rank(module_name, pair, blockers)
        if rank is not None:
            ranks.add(rank)

        if "a" in pair and "b" in pair:
            strength = _tensor_norm(pair["a"]) + _tensor_norm(pair["b"])
            bucket = (
                double_strengths
                if stream == "double_blocks"
                else single_strengths
            )
            bucket[index] = bucket.get(index, 0.0) + strength

    global_module_names: list[str] = []
    for module, pair in sorted(global_modules.items()):
        global_module_names.append(module)
        rank = _pair_rank(f"global.{module}", pair, blockers)
        if rank is not None:
            ranks.add(rank)
        if "a" in pair:
            _tensor_norm(pair["a"])
        if "b" in pair:
            _tensor_norm(pair["b"])

    unmatched_lora_keys = [
        key for key in unmatched_keys if "lora_" in key.casefold()
    ]
    if unmatched_lora_keys:
        blockers.append(
            f"Found {len(unmatched_lora_keys)} unrecognised LoRA tensor key(s)"
        )

    observed_double = sorted(double_strengths)
    observed_single = sorted(single_strengths)
    expected_double = list(range(DOUBLE_BLOCK_COUNT))
    expected_single = list(range(SINGLE_BLOCK_COUNT))

    missing_double = sorted(set(expected_double) - set(observed_double))
    missing_single = sorted(set(expected_single) - set(observed_single))
    extra_double = sorted(set(observed_double) - set(expected_double))
    extra_single = sorted(set(observed_single) - set(expected_single))

    if missing_double:
        blockers.append(
            f"Missing Flux 2 double block indices: {missing_double}"
        )
    if missing_single:
        blockers.append(
            f"Missing Flux 2 single block indices: {missing_single}"
        )
    if extra_double:
        blockers.append(
            f"Unexpected Flux 2 double block indices: {extra_double}"
        )
    if extra_single:
        blockers.append(
            f"Unexpected Flux 2 single block indices: {extra_single}"
        )

    if global_modules:
        warnings.append(
            "Global projection LoRA tensors are recorded separately and "
            "excluded from per-block strengths"
        )
    if len(ranks) > 1:
        blockers.append(
            "Flux 2 target uses more than one LoRA rank across adapted modules"
        )

    raw_strengths = [
        double_strengths.get(index, 0.0)
        for index in range(DOUBLE_BLOCK_COUNT)
    ]
    raw_strengths.extend(
        single_strengths.get(index, 0.0)
        for index in range(SINGLE_BLOCK_COUNT)
    )
    block_weights = _normalise_strengths(raw_strengths)

    block_tensor_count = sum(len(pair) for pair in block_modules.values())
    global_tensor_count = sum(len(pair) for pair in global_modules.values())

    return {
        "tensor_key_count": len(tensors),
        "model_family": "Flux 2",
        "lora_type": "Flux 2 (PEFT double+single blocks)",
        "rank": next(iter(ranks)) if len(ranks) == 1 else None,
        "rank_values": sorted(ranks),
        "block_layout": FLUX2_TRANSFORMER_56,
        "block_count": len(block_weights),
        "block_weights": block_weights,
        "raw_block_strengths": raw_strengths,
        "observed_double_indices": observed_double,
        "observed_single_indices": observed_single,
        "missing_double_indices": missing_double,
        "missing_single_indices": missing_single,
        "extra_double_indices": extra_double,
        "extra_single_indices": extra_single,
        "block_module_count": len(block_modules),
        "block_tensor_count": block_tensor_count,
        "global_module_count": len(global_modules),
        "global_tensor_count": global_tensor_count,
        "global_modules": global_module_names,
        "unmatched_tensor_count": len(unmatched_keys),
        "unmatched_key_sample": unmatched_keys[:25],
        "warnings": warnings,
        "blockers": blockers,
        "ready_for_sealing": not blockers,
    }


def default_tensor_analyser(path: Path) -> Mapping[str, Any]:
    from safetensors import safe_open

    tensors: dict[str, Any] = {}
    with safe_open(str(path), framework="pt") as handle:
        for key in handle.keys():
            tensors[str(key)] = handle.get_tensor(key)
    return analyse_flux2_tensor_map(tensors)


def _validate_phase89j_report(
    report: Mapping[str, Any],
    expected_report_sha256: str,
) -> dict[str, Any]:
    if report.get("phase") != "8.9j":
        raise Flux2LayoutError("Phase 8.9k requires the Phase 8.9j report")

    verified = _verify_embedded_digest(
        report,
        "analysis_sha256",
        "Phase 8.9j report",
    )
    expected = str(expected_report_sha256 or "").strip().lower()
    if verified != expected:
        raise Flux2LayoutError(
            f"Unexpected Phase 8.9j report digest: expected {expected}, "
            f"found {verified}"
        )

    target = report.get("target")
    if not isinstance(target, Mapping):
        raise Flux2LayoutError("Phase 8.9j report has no target object")
    result = dict(target)

    checks = {
        "planned_stable_id": EXPECTED_STABLE_ID,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "tensor_key_count": EXPECTED_TENSOR_KEY_COUNT,
        "rank": EXPECTED_RANK,
        "rank_values": [EXPECTED_RANK],
        "observed_double_indices": list(range(DOUBLE_BLOCK_COUNT)),
        "observed_single_indices": list(range(SINGLE_BLOCK_COUNT)),
        "block_module_count": EXPECTED_BLOCK_MODULE_COUNT,
        "block_tensor_count": EXPECTED_BLOCK_TENSOR_COUNT,
        "global_module_count": EXPECTED_GLOBAL_MODULE_COUNT,
        "global_tensor_count": EXPECTED_GLOBAL_TENSOR_COUNT,
        "unmatched_tensor_count": 0,
        "ready_for_controlled_apply": False,
    }
    for field, expected_value in checks.items():
        if result.get(field) != expected_value:
            raise Flux2LayoutError(
                f"Phase 8.9j {field} mismatch: expected {expected_value!r}, "
                f"found {result.get(field)!r}"
            )

    global_modules = tuple(result.get("global_module_sample") or [])
    if global_modules != EXPECTED_GLOBAL_MODULES:
        raise Flux2LayoutError(
            "Phase 8.9j global module set no longer matches the observed target"
        )

    blockers = result.get("blockers") or []
    blocker_indices: list[int] = []
    for blocker in blockers:
        match = _PHASE89J_RANGE_BLOCKER_RE.match(str(blocker))
        if not match:
            raise Flux2LayoutError(
                f"Phase 8.9j contains an unexpected blocker: {blocker}"
            )
        blocker_indices.append(int(match.group(1)))
    if sorted(set(blocker_indices)) != list(range(38, 48)):
        raise Flux2LayoutError(
            "Phase 8.9j blockers do not identify single blocks 38..47"
        )

    return result


def _validate_flux2_analysis(analysis: Mapping[str, Any]) -> None:
    exact_checks = {
        "tensor_key_count": EXPECTED_TENSOR_KEY_COUNT,
        "model_family": "Flux 2",
        "lora_type": "Flux 2 (PEFT double+single blocks)",
        "rank": EXPECTED_RANK,
        "rank_values": [EXPECTED_RANK],
        "block_layout": FLUX2_TRANSFORMER_56,
        "block_count": TOTAL_BLOCK_COUNT,
        "observed_double_indices": list(range(DOUBLE_BLOCK_COUNT)),
        "observed_single_indices": list(range(SINGLE_BLOCK_COUNT)),
        "missing_double_indices": [],
        "missing_single_indices": [],
        "extra_double_indices": [],
        "extra_single_indices": [],
        "block_module_count": EXPECTED_BLOCK_MODULE_COUNT,
        "block_tensor_count": EXPECTED_BLOCK_TENSOR_COUNT,
        "global_module_count": EXPECTED_GLOBAL_MODULE_COUNT,
        "global_tensor_count": EXPECTED_GLOBAL_TENSOR_COUNT,
        "global_modules": list(EXPECTED_GLOBAL_MODULES),
        "unmatched_tensor_count": 0,
        "blockers": [],
        "ready_for_sealing": True,
    }
    for field, expected_value in exact_checks.items():
        if analysis.get(field) != expected_value:
            raise Flux2LayoutError(
                f"Flux 2 analysis {field} mismatch: expected {expected_value!r}, "
                f"found {analysis.get(field)!r}"
            )

    block_weights = [float(value) for value in analysis.get("block_weights") or []]
    raw_strengths = [
        float(value) for value in analysis.get("raw_block_strengths") or []
    ]
    if len(block_weights) != TOTAL_BLOCK_COUNT:
        raise Flux2LayoutError(
            f"Expected {TOTAL_BLOCK_COUNT} block weights, found {len(block_weights)}"
        )
    if len(raw_strengths) != TOTAL_BLOCK_COUNT:
        raise Flux2LayoutError(
            f"Expected {TOTAL_BLOCK_COUNT} raw strengths, found {len(raw_strengths)}"
        )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in block_weights):
        raise Flux2LayoutError("Flux 2 block weights must be finite values in 0..1")
    if not all(math.isfinite(value) and value >= 0.0 for value in raw_strengths):
        raise Flux2LayoutError("Flux 2 raw strengths must be finite and non-negative")
    if not block_weights or max(block_weights) != 1.0:
        raise Flux2LayoutError("Flux 2 block weights must contain a strongest block of 1.0")
    if expected_block_count_for_layout(FLUX2_TRANSFORMER_56) != TOTAL_BLOCK_COUNT:
        raise Flux2LayoutError("Flux 2 layout registry does not resolve to 56 blocks")


def build_flux2_sealed_artifact(
    phase89j_report: Mapping[str, Any],
    *,
    library_root: str | os.PathLike[str],
    db_path: str | os.PathLike[str],
    expected_report_sha256: str = EXPECTED_PHASE89J_SHA256,
    expected_stable_id: str = EXPECTED_STABLE_ID,
    expected_source_sha256: str = EXPECTED_SOURCE_SHA256,
    tensor_analyser: TensorAnalyser | None = None,
) -> dict[str, Any]:
    phase89j_target = _validate_phase89j_report(
        phase89j_report,
        expected_report_sha256,
    )

    stable_id = str(phase89j_target.get("planned_stable_id") or "").upper()
    if stable_id != str(expected_stable_id or "").upper():
        raise Flux2LayoutError(
            f"Stable ID mismatch: expected {expected_stable_id}, found {stable_id}"
        )

    relative_path = str(phase89j_target.get("relative_path") or "").strip()
    db_file_path = str(phase89j_target.get("db_file_path") or "").strip()
    if not relative_path or not db_file_path:
        raise Flux2LayoutError("Phase 8.9j target path metadata is incomplete")

    root = Path(library_root).expanduser().resolve(strict=True)
    source = _safe_source_path(root, relative_path)
    source_sha256 = _sha256_file(source)
    expected_source = str(expected_source_sha256 or "").strip().lower()
    if source_sha256 != expected_source:
        raise Flux2LayoutError(
            f"Source SHA-256 mismatch: expected {expected_source}, "
            f"found {source_sha256}"
        )
    if source_sha256 != str(phase89j_target.get("source_sha256") or "").lower():
        raise Flux2LayoutError(
            "Source SHA-256 no longer matches the Phase 8.9j report"
        )

    conn = _open_read_only(db_path)
    try:
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity.casefold() != "ok":
            raise Flux2LayoutError(
                f"Database integrity check failed: {integrity}"
            )
        if conn.execute(
            "SELECT 1 FROM lora WHERE file_path = ?",
            (db_file_path,),
        ).fetchone() is not None:
            raise Flux2LayoutError(
                f"Target file_path already exists in DB: {db_file_path}"
            )
        if conn.execute(
            "SELECT 1 FROM lora WHERE stable_id = ?",
            (stable_id,),
        ).fetchone() is not None:
            raise Flux2LayoutError(
                f"Planned stable ID already exists in DB: {stable_id}"
            )
    finally:
        conn.close()

    analyser = tensor_analyser or default_tensor_analyser
    analysis = dict(analyser(source))
    _validate_flux2_analysis(analysis)

    target = {
        "relative_path": relative_path,
        "db_file_path": db_file_path,
        "filename": phase89j_target.get("filename") or source.name,
        "planned_stable_id": stable_id,
        "base_model_name": "Flux 2",
        "base_model_code": "FLX",
        "category_name": phase89j_target.get("category_name"),
        "category_code": phase89j_target.get("category_code"),
        "source_size_bytes": source.stat().st_size,
        "source_mtime": source.stat().st_mtime,
        "source_sha256": source_sha256,
        "clip_contributor": phase89j_target.get("clip_contributor"),
        "clip_tensor_count": phase89j_target.get("clip_tensor_count"),
        **analysis,
    }

    artifact: dict[str, Any] = {
        "phase": "8.9k",
        "mode": "read-only sealed targeted Flux 2 artifact",
        "phase89j_analysis_sha256": EXPECTED_PHASE89J_SHA256,
        "target": target,
        "summary": {
            "targets_analysed": 1,
            "ready_for_later_controlled_apply": 1,
            "block_rows": TOTAL_BLOCK_COUNT,
            "global_projection_tensors": EXPECTED_GLOBAL_TENSOR_COUNT,
            "damaged_flux_targets_untouched": 1,
        },
        "safety": {
            "database_open_mode": (
                "SQLite URI mode=ro plus PRAGMA query_only=ON"
            ),
            "writes_database": False,
            "creates_backup": False,
            "runs_full_indexer": False,
            "discovers_library_files": False,
            "opens_only_phase89j_target": True,
            "assigns_stable_ids": False,
            "deletes_rows": False,
            "touches_damaged_flux_target": False,
            "contains_apply_mode": False,
        },
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def print_artifact(artifact: Mapping[str, Any]) -> None:
    target = artifact["target"]
    print("=== Phase 8.9k sealed Flux 2 artifact ===")
    print(f"Mode                       : {artifact['mode']}")
    print(f"Artifact SHA-256           : {artifact['artifact_sha256']}")
    print(f"Phase 8.9j SHA-256         : {artifact['phase89j_analysis_sha256']}")
    print(f"Stable ID                  : {target['planned_stable_id']}")
    print(f"Source SHA-256             : {target['source_sha256']}")
    print(f"Tensor keys                : {target['tensor_key_count']}")
    print(f"Model family               : {target['model_family']}")
    print(f"LoRA type                  : {target['lora_type']}")
    print(f"Rank                       : {target['rank']}")
    print(f"Double blocks              : {target['observed_double_indices']}")
    print(f"Single blocks              : {target['observed_single_indices']}")
    print(f"Block modules              : {target['block_module_count']}")
    print(f"Block tensors              : {target['block_tensor_count']}")
    print(f"Global modules             : {target['global_module_count']}")
    print(f"Global tensors             : {target['global_tensor_count']}")
    print(f"Block layout               : {target['block_layout']}")
    print(f"Block rows                 : {target['block_count']}")
    print("Ready for later apply      : True")
    for warning in target["warnings"]:
        print(f"Warning                    : {warning}")
    print("No database changes were made.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Seal the Phase 8.9j target as the distinct Flux 2 "
            "8-double plus 48-single architecture"
        )
    )
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument(
        "--expected-analysis-sha256",
        default=EXPECTED_PHASE89J_SHA256,
    )
    parser.add_argument("--expected-stable-id", default=EXPECTED_STABLE_ID)
    parser.add_argument(
        "--expected-source-sha256",
        default=EXPECTED_SOURCE_SHA256,
    )
    parser.add_argument("--json")
    args = parser.parse_args()

    artifact = build_flux2_sealed_artifact(
        load_json_object(args.analysis, "Phase 8.9j analysis"),
        library_root=args.root,
        db_path=args.db,
        expected_report_sha256=args.expected_analysis_sha256,
        expected_stable_id=args.expected_stable_id,
        expected_source_sha256=args.expected_source_sha256,
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
