from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from phase89g_targeted_flux_analysis import (
    _open_read_only,
    _safe_source_path,
    _sha256_file,
)


class PeftFluxAnalysisError(RuntimeError):
    pass


EXPECTED_STABLE_ID = "FLX-STL-263"
EXPECTED_SOURCE_SHA256 = (
    "c60c9a5de39da23b3b4f4dca48e3511faa1fe5a4987d4acbb0a04643a9a65be7"
)
EXPECTED_LAYOUT = "flux_unet_57"
DOUBLE_BLOCK_COUNT = 19
SINGLE_BLOCK_COUNT = 38
TOTAL_BLOCK_COUNT = DOUBLE_BLOCK_COUNT + SINGLE_BLOCK_COUNT

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

TensorAnalyser = Callable[[Path], Mapping[str, Any]]


def load_json_object(path: str | os.PathLike[str], label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise PeftFluxAnalysisError(f"{label} JSON must contain an object")
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


def verify_analysis_digest(result: Mapping[str, Any]) -> str:
    stored = str(result.get("analysis_sha256") or "")
    unsigned = dict(result)
    unsigned.pop("analysis_sha256", None)
    calculated = canonical_sha256(unsigned)
    if stored != calculated:
        raise PeftFluxAnalysisError(
            f"Analysis digest mismatch: stored {stored}, calculated {calculated}"
        )
    return calculated


def _tensor_norm(value: torch.Tensor) -> float:
    norm = float(value.norm().item())
    if not math.isfinite(norm):
        raise PeftFluxAnalysisError("Encountered non-finite tensor norm")
    return norm


def _pair_rank(
    module_name: str,
    pair: Mapping[str, torch.Tensor],
    blockers: list[str],
) -> int | None:
    a = pair.get("a")
    b = pair.get("b")
    if a is None or b is None:
        missing = "A" if a is None else "B"
        blockers.append(f"Incomplete LoRA pair for {module_name}: missing lora_{missing}")
        return None
    if a.ndim != 2 or b.ndim != 2:
        blockers.append(
            f"LoRA pair for {module_name} is not two-dimensional: "
            f"A{tuple(a.shape)}, B{tuple(b.shape)}"
        )
        return None
    rank_a = int(a.shape[0])
    rank_b = int(b.shape[1])
    if rank_a <= 0 or rank_b <= 0 or rank_a != rank_b:
        blockers.append(
            f"LoRA rank mismatch for {module_name}: A rank {rank_a}, B rank {rank_b}"
        )
        return None
    return rank_a


def _normalise_strengths(raw_strengths: list[float]) -> list[float]:
    maximum = max(raw_strengths, default=0.0)
    if maximum <= 0:
        return [0.0 for _ in raw_strengths]
    return [round(value / maximum, 6) for value in raw_strengths]


def analyse_peft_tensor_map(tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    block_modules: dict[tuple[str, int, str], dict[str, torch.Tensor]] = {}
    global_modules: dict[str, dict[str, torch.Tensor]] = {}
    unmatched_keys: list[str] = []

    for raw_name, tensor in sorted(tensors.items(), key=lambda item: item[0].casefold()):
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
                raise PeftFluxAnalysisError(
                    f"Duplicate lora_{side.upper()} tensor for {stream}.{index}.{module}"
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
                raise PeftFluxAnalysisError(
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
            blockers.append(
                f"Out-of-range {stream} index {index}; supported range is 0..{max_index}"
            )

        module_name = f"{stream}.{index}.{module}"
        rank = _pair_rank(module_name, pair, blockers)
        if rank is not None:
            ranks.add(rank)

        if "a" in pair and "b" in pair:
            strength = _tensor_norm(pair["a"]) + _tensor_norm(pair["b"])
            bucket = (
                double_strengths if stream == "double_blocks" else single_strengths
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
    if not observed_double:
        blockers.append("No base_model.model.double_blocks LoRA pairs were found")
    if not observed_single:
        blockers.append("No base_model.model.single_blocks LoRA pairs were found")

    missing_double = [
        index for index in range(DOUBLE_BLOCK_COUNT) if index not in double_strengths
    ]
    missing_single = [
        index for index in range(SINGLE_BLOCK_COUNT) if index not in single_strengths
    ]
    if missing_double or missing_single:
        warnings.append(
            "Unadapted Flux blocks are represented as zero strength in the 57-block layout"
        )
    if global_modules:
        warnings.append(
            "Global projection LoRA tensors are recorded separately and excluded from per-block strengths"
        )
    if len(ranks) > 1:
        warnings.append("The file uses more than one LoRA rank across adapted modules")

    raw_strengths = [
        double_strengths.get(index, 0.0) for index in range(DOUBLE_BLOCK_COUNT)
    ]
    raw_strengths.extend(
        single_strengths.get(index, 0.0) for index in range(SINGLE_BLOCK_COUNT)
    )
    block_weights = _normalise_strengths(raw_strengths)

    block_tensor_count = sum(len(pair) for pair in block_modules.values())
    global_tensor_count = sum(len(pair) for pair in global_modules.values())
    ready = not blockers

    return {
        "tensor_key_count": len(tensors),
        "model_family": "Flux",
        "lora_type": "Flux (PEFT base_model double+single blocks)",
        "rank": next(iter(ranks)) if len(ranks) == 1 else None,
        "rank_values": sorted(ranks),
        "block_layout": EXPECTED_LAYOUT,
        "block_count": len(block_weights),
        "block_weights": block_weights,
        "raw_block_strengths": raw_strengths,
        "observed_double_indices": observed_double,
        "observed_single_indices": observed_single,
        "missing_double_indices": missing_double,
        "missing_single_indices": missing_single,
        "block_module_count": len(block_modules),
        "block_tensor_count": block_tensor_count,
        "global_module_count": len(global_modules),
        "global_tensor_count": global_tensor_count,
        "global_module_sample": global_module_names[:25],
        "unmatched_tensor_count": len(unmatched_keys),
        "unmatched_key_sample": unmatched_keys[:25],
        "warnings": warnings,
        "blockers": blockers,
        "ready_for_controlled_apply": ready,
    }


def default_tensor_analyser(path: Path) -> Mapping[str, Any]:
    from safetensors import safe_open

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as handle:
        for key in handle.keys():
            tensors[str(key)] = handle.get_tensor(key)
    return analyse_peft_tensor_map(tensors)


def _select_target(
    diagnostics: Mapping[str, Any],
    expected_stable_id: str,
) -> dict[str, Any]:
    if diagnostics.get("phase") != "8.9g-diagnostics":
        raise PeftFluxAnalysisError(
            "Phase 8.9j requires the Phase 8.9g diagnostics report"
        )

    matches = [
        dict(target)
        for target in diagnostics.get("targets", [])
        if isinstance(target, Mapping)
        and str(target.get("planned_stable_id") or "").upper()
        == expected_stable_id.upper()
    ]
    if len(matches) != 1:
        raise PeftFluxAnalysisError(
            f"Expected exactly one diagnostics target for {expected_stable_id}, "
            f"found {len(matches)}"
        )
    target = matches[0]
    if target.get("ready_for_controlled_apply") is not False:
        raise PeftFluxAnalysisError(
            "Target is not in the expected blocked diagnostics state"
        )
    if target.get("tensor_inspection_error") is not None:
        raise PeftFluxAnalysisError("Target tensor header is not readable")
    if target.get("analysis_error") is None:
        raise PeftFluxAnalysisError(
            "Target does not contain the expected unsupported-analysis error"
        )
    return target


def build_targeted_peft_analysis(
    diagnostics: Mapping[str, Any],
    *,
    library_root: str | os.PathLike[str],
    db_path: str | os.PathLike[str],
    expected_stable_id: str = EXPECTED_STABLE_ID,
    expected_source_sha256: str = EXPECTED_SOURCE_SHA256,
    tensor_analyser: TensorAnalyser | None = None,
) -> dict[str, Any]:
    target = _select_target(diagnostics, expected_stable_id)
    relative_path = str(target.get("relative_path") or "").strip()
    if not relative_path:
        raise PeftFluxAnalysisError("Diagnostics target has no relative_path")

    root = Path(library_root).expanduser().resolve(strict=True)
    source = _safe_source_path(root, relative_path)
    source_sha256 = _sha256_file(source)
    expected_source = str(expected_source_sha256 or "").strip().lower()
    if source_sha256 != expected_source:
        raise PeftFluxAnalysisError(
            f"Source SHA-256 mismatch: expected {expected_source}, found {source_sha256}"
        )
    if source_sha256 != str(target.get("source_sha256") or "").lower():
        raise PeftFluxAnalysisError(
            "Source SHA-256 no longer matches Phase 8.9g diagnostics"
        )

    stable_id = str(target.get("planned_stable_id") or "").upper()
    db_file_path = str(target.get("db_file_path") or "").strip()
    if not db_file_path:
        raise PeftFluxAnalysisError("Diagnostics target has no db_file_path")

    conn = _open_read_only(db_path)
    try:
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity.casefold() != "ok":
            raise PeftFluxAnalysisError(
                f"Database integrity check failed: {integrity}"
            )
        if conn.execute(
            "SELECT 1 FROM lora WHERE file_path = ?", (db_file_path,)
        ).fetchone() is not None:
            raise PeftFluxAnalysisError(
                f"Target file_path already exists in DB: {db_file_path}"
            )
        if conn.execute(
            "SELECT 1 FROM lora WHERE stable_id = ?", (stable_id,)
        ).fetchone() is not None:
            raise PeftFluxAnalysisError(
                f"Planned stable ID already exists in DB: {stable_id}"
            )
    finally:
        conn.close()

    analyser = tensor_analyser or default_tensor_analyser
    analysis = dict(analyser(source))
    if int(analysis.get("tensor_key_count") or 0) != int(
        target.get("tensor_key_count") or 0
    ):
        raise PeftFluxAnalysisError(
            "Tensor-key count no longer matches Phase 8.9g diagnostics: "
            f"expected {target.get('tensor_key_count')}, "
            f"found {analysis.get('tensor_key_count')}"
        )
    if int(analysis.get("block_count") or 0) != TOTAL_BLOCK_COUNT:
        raise PeftFluxAnalysisError(
            f"Target analyser returned {analysis.get('block_count')} blocks, "
            f"expected {TOTAL_BLOCK_COUNT}"
        )
    if analysis.get("block_layout") != EXPECTED_LAYOUT:
        raise PeftFluxAnalysisError(
            f"Target analyser returned layout {analysis.get('block_layout')!r}, "
            f"expected {EXPECTED_LAYOUT!r}"
        )

    result: dict[str, Any] = {
        "phase": "8.9j",
        "mode": "read-only targeted PEFT Flux analysis",
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
            "source_sha256": source_sha256,
            "clip_contributor": target.get("clip_contributor"),
            "clip_tensor_count": target.get("clip_tensor_count"),
            **analysis,
        },
        "summary": {
            "targets_analysed": 1,
            "ready_for_controlled_apply": int(
                analysis.get("ready_for_controlled_apply") is True
            ),
            "blocked_targets": int(
                analysis.get("ready_for_controlled_apply") is not True
            ),
            "block_rows": int(analysis.get("block_count") or 0),
            "global_projection_tensors": int(
                analysis.get("global_tensor_count") or 0
            ),
        },
        "safety": {
            "database_open_mode": (
                "SQLite URI mode=ro plus PRAGMA query_only=ON"
            ),
            "writes_database": False,
            "runs_full_indexer": False,
            "discovers_library_files": False,
            "opens_only_diagnostics_target": True,
            "assigns_stable_ids": False,
            "deletes_rows": False,
            "touches_damaged_flux_target": False,
        },
    }
    result["analysis_sha256"] = canonical_sha256(result)
    return result


def print_analysis(result: Mapping[str, Any]) -> None:
    target = result["target"]
    summary = result["summary"]
    print("=== Phase 8.9j targeted PEFT Flux analysis ===")
    print(f"Mode                       : {result['mode']}")
    print(f"Analysis SHA-256           : {result['analysis_sha256']}")
    print(f"Stable ID                  : {target['planned_stable_id']}")
    print(f"Source SHA-256             : {target['source_sha256']}")
    print(f"Tensor keys                : {target['tensor_key_count']}")
    print(f"LoRA type                  : {target['lora_type']}")
    print(f"Rank values                : {target['rank_values']}")
    print(f"Observed double blocks     : {target['observed_double_indices']}")
    print(f"Observed single blocks     : {target['observed_single_indices']}")
    print(f"Block tensors              : {target['block_tensor_count']}")
    print(f"Global projection tensors  : {target['global_tensor_count']}")
    print(f"Unmatched tensors          : {target['unmatched_tensor_count']}")
    print(f"Block layout               : {target['block_layout']}")
    print(f"Block rows                 : {target['block_count']}")
    print(
        "Ready for controlled apply : "
        f"{bool(summary['ready_for_controlled_apply'])}"
    )
    for warning in target["warnings"]:
        print(f"Warning                    : {warning}")
    for blocker in target["blockers"]:
        print(f"Blocker                    : {blocker}")
    print("No database changes were made.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a targeted read-only analyser for the PEFT-style "
            "FLX-STL-263 file"
        )
    )
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--expected-stable-id", default=EXPECTED_STABLE_ID)
    parser.add_argument(
        "--expected-source-sha256", default=EXPECTED_SOURCE_SHA256
    )
    parser.add_argument("--json")
    args = parser.parse_args()

    result = build_targeted_peft_analysis(
        load_json_object(args.diagnostics, "Diagnostics"),
        library_root=args.root,
        db_path=args.db,
        expected_stable_id=args.expected_stable_id,
        expected_source_sha256=args.expected_source_sha256,
    )
    verify_analysis_digest(result)
    print_analysis(result)

    if args.json:
        output = Path(args.json).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        print(f"JSON analysis written to: {output}")


if __name__ == "__main__":
    main()
