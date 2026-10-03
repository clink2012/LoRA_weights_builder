"""Bounded, header-only native LoRA mapping against a pinned FLUX.1 target.

This establishes conditional structural compatibility, not tensor contents,
checkpoint identity, successful Comfy patch application or image quality.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any

from inspire_export import LOADER_SHA256

CONTRACT = json.loads((Path(__file__).parent / "contracts/flux1-dev-native-v1.json").read_text(encoding="utf-8"))
CONTRACT_ID = CONTRACT["contract_id"]
COVERAGE_SOURCE = "statically_resolved_against_pinned_target"
MAX_HEADER_BYTES = 16 * 1024 * 1024
_DTYPES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}


class CoverageError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CoverageError("invalid_header", "Duplicate keys in safetensors header.")
        result[key] = value
    return result


def read_header(path: Path, *, include_metadata: bool = False) -> tuple[dict, dict]:
    """Only the length word and JSON header are read; offsets validate file size."""
    if path.suffix.lower() != ".safetensors":
        raise CoverageError("unsupported_file", "Only safetensors headers are supported by this resolver.")
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise CoverageError("invalid_header", "Missing safetensors length header.")
            length = struct.unpack("<Q", length_bytes)[0]
            if length < 2 or length > MAX_HEADER_BYTES or length > before.st_size - 8:
                raise CoverageError("invalid_header", "Safetensors header length is invalid or exceeds the bounded reader.")
            raw = stream.read(length)
            if len(raw) != length:
                raise CoverageError("invalid_header", "Safetensors header is incomplete.")
            header = json.loads(raw, object_pairs_hook=_unique_object)
            after = os.fstat(stream.fileno())
        current = path.stat()
        identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity(before) != identity(after) or identity(before) != identity(current):
            raise CoverageError("file_changed", "The LoRA changed while its header was being inspected.")
    except FileNotFoundError as exc:
        raise CoverageError("file_missing", "The catalogued LoRA file is no longer present.") from exc
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CoverageError("invalid_header", "The LoRA header could not be read safely.") from exc
    if not isinstance(header, dict):
        raise CoverageError("invalid_header", "Safetensors header must contain a tensor dictionary.")
    if include_metadata and "__metadata__" in header:
        metadata = header["__metadata__"]
        if not isinstance(metadata, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()):
            raise CoverageError("invalid_header", "Safetensors metadata must be a string-to-string dictionary.")
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not tensors:
        raise CoverageError("empty_header", "No tensors were found in this LoRA header.")
    intervals = []
    payload_size = before.st_size - 8 - length
    for key, tensor in tensors.items():
        if not isinstance(tensor, dict) or tensor.get("dtype") not in _DTYPES:
            raise CoverageError("unsupported_tensor", "Only ordinary floating-point LoRA tensors are supported.")
        shape, offsets = tensor.get("shape"), tensor.get("data_offsets")
        if not isinstance(shape, list) or len(shape) > 4 or any(type(d) is not int or d <= 0 or d > 2**31 for d in shape):
            raise CoverageError("invalid_header", "A tensor shape is invalid.")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(v) is not int for v in offsets):
            raise CoverageError("invalid_header", "Tensor offsets are invalid.")
        start, end = offsets
        size = math.prod(shape) * _DTYPES[tensor["dtype"]]
        if start < 0 or end < start or end > payload_size or end - start != size:
            raise CoverageError("invalid_header", "Tensor offsets and shape do not match the file size.")
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise CoverageError("invalid_header", "Tensor payload contains a gap or overlapping ranges.")
        cursor = end
    if cursor != payload_size:
        raise CoverageError("invalid_header", "Tensor payload length does not match the complete file.")
    return (header if include_metadata else tensors), {
        "file_size": before.st_size, "file_mtime_ns": before.st_mtime_ns,
        "file_device": before.st_dev, "file_inode": before.st_ino,
        "header_sha256": hashlib.sha256(length_bytes + raw).hexdigest(),
        "header_bytes_read": 8 + length, "tensor_payload_read": False,
        "tensor_contents_verified": False,
    }


def verify_local_sources(comfy_root: Path | None = None) -> dict[str, str]:
    root = comfy_root or Path(os.environ.get("COMFYUI_ROOT", r"E:\ComfyUI_New\ComfyUI"))
    expected = dict(CONTRACT["source_sha256"])
    expected["custom_nodes/comfyui-inspire-pack/inspire/lora_block_weight.py"] = LOADER_SHA256
    verified = {}
    for relative, digest in expected.items():
        try:
            current = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        except OSError as exc:
            raise CoverageError("runtime_source_unavailable", "The pinned Comfy/Inspire source could not be verified on this machine.") from exc
        if current != digest:
            raise CoverageError("runtime_source_changed", "Comfy or Inspire source changed; this target contract requires review before export.")
        verified[relative] = current
    return verified


def resolve_native_header(tensors: dict[str, Any]) -> dict:
    """Pure mapping: each supported tensor must belong to a complete known pair."""
    grouped: dict[str, dict] = {}
    suffixes = (".lora_up.weight", ".lora_down.weight", ".alpha")
    for key, tensor in tensors.items():
        suffix = next((part for part in suffixes if key.endswith(part)), None)
        if suffix is None:
            raise CoverageError("unsupported_adapter", "This LoRA includes tensor formats outside native up/down pairs and scalar alpha.")
        prefix = key[:-len(suffix)]
        if prefix not in CONTRACT["aliases"]:
            raise CoverageError("unknown_target_module", "A LoRA module is not part of the pinned standard FLUX.1 target.")
        grouped.setdefault(prefix, {})[suffix] = tensor
    keys = []
    base = []
    presence = [0.0] * 57
    for prefix, entries in grouped.items():
        if ".lora_up.weight" not in entries or ".lora_down.weight" not in entries:
            raise CoverageError("incomplete_pair", "A LoRA module is missing an up or down tensor.")
        target = CONTRACT["aliases"][prefix]
        if target in keys:
            raise CoverageError("duplicate_target_alias", "Multiple LoRA aliases target the same module; patch precedence is not supported.")
        expected_out, expected_in = CONTRACT["target_shapes"][target]
        up = entries[".lora_up.weight"]["shape"]
        down = entries[".lora_down.weight"]["shape"]
        if len(up) != 2 or len(down) != 2 or up[1] != down[0] or up[1] <= 0 or up[0] != expected_out or down[1] != expected_in:
            raise CoverageError("target_shape_mismatch", "LoRA pair dimensions do not match the pinned FLUX.1 target module.")
        if ".alpha" in entries and entries[".alpha"]["shape"] not in ([], [1]):
            raise CoverageError("invalid_alpha_shape", "LoRA alpha must contain exactly one scalar value.")
        keys.append(target)
        parts = target.split(".")
        if parts[1] == "double_blocks":
            presence[int(parts[2])] = 1.0
        elif parts[1] == "single_blocks":
            presence[19 + int(parts[2])] = 1.0
        else:
            base.append(target)
    if not keys:
        raise CoverageError("empty_patch_coverage", "No supported LoRA pairs were found.")
    return {
        "resolved_patch_keys": keys, "base_patch_keys": base,
        "block_presence_baseline": presence, "coverage_complete": True,
        "coverage_source": COVERAGE_SOURCE,
        "target_contract_id": CONTRACT_ID, "target_contract_label": CONTRACT["label"],
        "checkpoint_verified": False, "image_quality_verified": False,
        "recommendation_basis": "structural_baseline_unvalidated",
    }


def inspect_native_flux_file(path: Path) -> dict:
    sources = verify_local_sources()
    tensors, identity = read_header(path)
    result = resolve_native_header(tensors)
    result["file_identity"] = identity
    result["source_sha256"] = sources
    return result
