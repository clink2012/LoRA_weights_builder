"""Bounded H3 checkpoint-header observations, separate from floating LoRA reads.

An INT8 matrix's stored dimensions are observable; its dequantization, rotation,
patch application and task identity are not proved by a header. No API or export
capability is enabled here and no tensor payload is read.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import struct

from flux_header_coverage import CoverageError, MAX_HEADER_BYTES, _unique_object

DTYPES = {"I8": 1, "U8": 1, "F16": 2, "BF16": 2, "F32": 4, "F64": 8}


def read_quantization_descriptors(path):
    """Read only bounded U8 JSON descriptors, never matrix or scale payloads.

    Callers must not label this operation header-only. Identity is checked across
    both openings, including after all sparse descriptor reads.
    """
    path = Path(path)
    header, identity = read_checkpoint_header(path)
    selected = {k: v for k, v in header.items() if k.endswith('.comfy_quant')}
    if len(selected) > 256:
        raise CoverageError('descriptor_budget', 'Too many quantization descriptors for this research reader.')
    for tensor in selected.values():
        if tensor.get('dtype') != 'U8' or len(tensor['shape']) != 1 or not 1 <= tensor['shape'][0] <= 1024:
            raise CoverageError('descriptor_budget', 'Quantization descriptor is not bounded U8 JSON.')
    descriptors, bytes_read = {}, 0
    try:
        with path.open('rb') as stream:
            before = os.fstat(stream.fileno())
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                    identity['file_device'], identity['file_inode'], identity['file_size'], identity['file_mtime_ns']):
                raise CoverageError('file_changed', 'Checkpoint changed before descriptor inspection.')
            for key, tensor in selected.items():
                start, end = tensor['data_offsets']
                stream.seek(identity['header_bytes_read'] + start)
                raw = stream.read(end - start)
                if len(raw) != end - start:
                    raise CoverageError('invalid_descriptor', 'Incomplete quantization descriptor.')
                value = json.loads(raw, object_pairs_hook=_unique_object)
                if not isinstance(value, dict):
                    raise CoverageError('invalid_descriptor', 'Quantization descriptor must be a dictionary.')
                descriptors[key.removesuffix('.comfy_quant')] = value
                bytes_read += len(raw)
            after = os.fstat(stream.fileno())
        current = path.stat()
        state = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if state(before) != state(after) or state(before) != state(current):
            raise CoverageError('file_changed', 'Checkpoint changed during descriptor inspection.')
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CoverageError('invalid_descriptor', 'Quantization descriptors could not be read safely.') from exc
    return descriptors, {**identity, 'tensor_payload_read': bytes_read > 0,
                         'descriptor_payload_bytes_read': bytes_read, 'matrix_payload_read': False,
                         'scale_payload_read': False, 'quantization_verified': False, 'rotation_verified': False}


def read_checkpoint_header(path):
    path = Path(path)
    if path.suffix.lower() != ".safetensors":
        raise CoverageError("unsupported_file", "Only safetensors checkpoint headers are supported.")
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            word = stream.read(8)
            if len(word) != 8:
                raise CoverageError("invalid_header", "Missing checkpoint header length.")
            length = struct.unpack("<Q", word)[0]
            if not 2 <= length <= MAX_HEADER_BYTES or length > before.st_size - 8:
                raise CoverageError("invalid_header", "Checkpoint header exceeds the bounded reader or file size.")
            raw = stream.read(length)
            if len(raw) != length:
                raise CoverageError("invalid_header", "Incomplete checkpoint header.")
            header = json.loads(raw, object_pairs_hook=_unique_object)
            after = os.fstat(stream.fileno())
        current = path.stat()
        identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity(before) != identity(after) or identity(before) != identity(current):
            raise CoverageError("file_changed", "Checkpoint changed while its header was being inspected.")
    except FileNotFoundError as exc:
        raise CoverageError("file_missing", "Selected checkpoint is unavailable.") from exc
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CoverageError("invalid_header", "Checkpoint header could not be read safely.") from exc
    if not isinstance(header, dict):
        raise CoverageError("invalid_header", "Checkpoint header must be a tensor dictionary.")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise CoverageError("invalid_header", "Checkpoint metadata must contain strings.")
    intervals = []
    payload_size = before.st_size - 8 - length
    for key, tensor in header.items():
        if key == "__metadata__": continue
        if not isinstance(tensor, dict) or tensor.get("dtype") not in DTYPES:
            raise CoverageError("unsupported_tensor", "Unsupported checkpoint storage dtype.")
        shape, offsets = tensor.get("shape"), tensor.get("data_offsets")
        if (not isinstance(shape, list) or len(shape) > 4
                or any(type(d) is not int or not 1 <= d <= 2**31 for d in shape)):
            raise CoverageError("invalid_header", "Invalid checkpoint tensor shape.")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(v) is not int for v in offsets):
            raise CoverageError("invalid_header", "Invalid checkpoint tensor offsets.")
        start, end = offsets
        if start < 0 or end < start or end > payload_size or end - start != math.prod(shape) * DTYPES[tensor["dtype"]]:
            raise CoverageError("invalid_header", "Checkpoint tensor size disagrees with its storage shape.")
        intervals.append((start, end))
    if not intervals:
        raise CoverageError("empty_header", "Checkpoint has no tensors.")
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise CoverageError("invalid_header", "Checkpoint payload has gaps or overlapping offsets.")
        cursor = end
    if cursor != payload_size:
        raise CoverageError("invalid_header", "Checkpoint payload size disagrees with its header.")
    return header, {"file_size": before.st_size, "file_mtime_ns": before.st_mtime_ns,
                    "file_device": before.st_dev, "file_inode": before.st_ino,
                    "header_sha256": hashlib.sha256(word + raw).hexdigest(), "header_bytes_read": 8 + length,
                    "tensor_payload_read": False, "tensor_contents_verified": False}


def describe_checkpoint(header, contract):
    """Compare every constructor state shape, without claiming loader compatibility."""
    expected = {**contract["target_shapes"], **contract["non_matrix_targets"]}
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    issues, matched, quant_aux = [], 0, 0
    for target, shape in expected.items():
        # This research target is native and unprefixed on disk. Other conversion
        # routes require their own evidence rather than prefix guessing.
        key = target.removeprefix("diffusion_model.")
        if key not in tensors:
            issues.append({"code": "missing_target", "target": target})
        elif tensors[key].get("shape") != shape:
            issues.append({"code": "target_shape_mismatch", "target": target})
        else: matched += 1
    for key, value in tensors.items():
        if "diffusion_model." + key in expected: continue
        module = key.rsplit(".", 1)[0]
        target = "diffusion_model." + module + ".weight"
        if target in contract["target_shapes"] and key.endswith(".weight_scale"):
            rows = contract["target_shapes"][target][0]
            if value.get("dtype") == "F32" and value.get("shape") == [rows, 1]:
                quant_aux += 1
                continue
        if target in contract["target_shapes"] and key.endswith(".comfy_quant"):
            if value.get("dtype") == "U8" and len(value.get("shape", [])) == 1:
                quant_aux += 1
                continue
        issues.append({"code": "unaccounted_checkpoint_tensor", "target": key[:256]})
    metadata = header.get("__metadata__", {})
    if "config" in metadata:
        issues.append({"code": "metadata_config_requires_review", "target": "__metadata__.config"})
    return {"contract_id": contract["contract_id"], "status": "shape_match_candidate" if not issues else "needs_review",
            "constructor_state_shapes_match": not issues, "matched_state_count": matched,
            "expected_state_count": len(expected), "tensor_count": len(tensors), "quant_auxiliary_count": quant_aux,
            "int8_matrix_count": sum(v.get("dtype") == "I8" and len(v.get("shape", [])) == 2 for v in tensors.values()),
            "main_count": contract["main_count"], "refiner_count": contract["refiner_count"],
            "parameters": contract["parameters"], "issue_count": len(issues), "issues": issues[:30],
            "task_identity_verified": False, "quantization_verified": False, "rotation_verified": False,
            "patch_application_verified": False, "measurements_available": False, "export_verified": False}
