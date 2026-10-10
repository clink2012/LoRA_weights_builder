"""Bounded H3-like adapter observations, never target or export verification."""
from collections import Counter
from pathlib import Path
import re

from adapter_identity import PARTS, classify_header
from flux_header_coverage import read_header

OBSERVATION_VERSION = "h3-pair-observations-v1"


def module_group(module):
    # These are observed spellings, not aliases proved against a loaded model.
    native = re.fullmatch(r"(?:(?:diffusion_model|transformer)\.)?(blocks|token_refiner\.blocks)\.(\d+)\.(.+)", module)
    kohya = re.fullmatch(r"lora_unet_(blocks|token_refiner_blocks)_(\d+)_(.+)", module)
    match = native or kohya
    if match:
        group = "main" if match[1] == "blocks" else "refiner"
        raw = match[2]
        if len(raw) > 6 or raw != str(int(raw)) or int(raw) > 100000:
            return "unknown", None
        return group, int(raw)
    if re.fullmatch(r"(?:(?:diffusion_model|transformer)\.)?(video_patch_proj|audio_patch_proj|condition_proj|time_embedder\.(proj_in|proj_out)|final_layer\.(video_out|audio_out|adaln_proj\.linear)|token_refiner\.final_norm)", module):
        return "other", None
    if re.fullmatch(r"lora_unet_(video_patch_proj|audio_patch_proj|condition_proj|time_embedder_(proj_in|proj_out)|final_layer_(video_out|audio_out|adaln_proj_linear)|token_refiner_final_norm)", module):
        return "other", None
    return "unknown", None


def observe_header(header):
    """Classify every supplied tensor; unknowns never become implicit OTHER."""
    identity = classify_header(header, folder_hint="MiniMax-H3")
    modules, issues, unknown_tensors = {}, [], 0
    formats = Counter()
    tensor_count = 0
    for key, tensor in header.items():
        if key == "__metadata__":
            continue
        tensor_count += 1
        part = next(((suffix, fmt, direction) for suffix, fmt, direction in PARTS if key.endswith(suffix)), None)
        if part is None:
            unknown_tensors += 1
            issues.append({"code": "unsupported_tensor", "module": key[:256], "reason": "Tensor is outside ordinary A/B or up/down pairs and scalar alpha."})
            continue
        suffix, fmt, direction = part
        module = key[:-len(suffix)]
        record = modules.setdefault(module, {"parts": {}, "formats": set(), "tensor_count": 0, "duplicate": False})
        record["tensor_count"] += 1
        if direction in record["parts"]:
            record["duplicate"] = True
            issues.append({"code": "duplicate_factor", "module": module[:256], "reason": "Multiple tensor aliases supply the same factor."})
        record["parts"][direction] = tensor
        if fmt != "alpha":
            record["formats"].add(fmt)
            formats[fmt] += 1
    slots, pairs, covered_tensors, unknown_modules = {}, 0, 0, 0
    for module, record in modules.items():
        group, index = module_group(module)
        parts = record["parts"]
        reason = None
        if record["duplicate"]:
            continue  # No ambiguous pair contributes to coverage counts.
        if group == "unknown":
            unknown_modules += 1
            reason = ("unknown_module", "Module cannot be assigned to an observed H3 main/refiner/other group.")
        elif not {"up", "down"}.issubset(parts):
            reason = ("incomplete_pair", "Module is missing an input or output factor.")
        elif len(record["formats"]) != 1:
            reason = ("mixed_pair_format", "A/B and up/down formats must not be mixed in a pair.")
        else:
            up, down = parts["up"].get("shape"), parts["down"].get("shape")
            if (not isinstance(up, list) or not isinstance(down, list) or len(up) != 2 or len(down) != 2
                    or any(type(value) is not int or value <= 0 for value in up + down) or up[1] != down[0]):
                reason = ("invalid_pair_shape", "Only compatible ordinary two-dimensional low-rank factors are observed here.")
            elif "alpha" in parts and parts["alpha"].get("shape") not in ([], [1]):
                reason = ("invalid_alpha_shape", "Alpha must contain one scalar.")
        if reason:
            issues.append({"code": reason[0], "module": module[:256], "reason": reason[1]})
            continue
        slot = slots.setdefault((group, index), {"group": group, "index": index,
            "label": "OTHER" if group == "other" else f"{group.upper()} {index}",
            "pair_count": 0, "ranks": set()})
        slot["pair_count"] += 1
        slot["ranks"].add(parts["down"]["shape"][0])
        pairs += 1
        covered_tensors += record["tensor_count"]
    # Duplicate factors also prevent all-tensor accounting from appearing clear.
    groups = {"main": 0, "refiner": 1, "other": 2}
    ordered = [{**record, "ranks": sorted(record["ranks"])}
               for _, record in sorted(slots.items(), key=lambda entry: (groups[entry[0][0]], entry[0][1] or 0))]
    return {
        "observation_version": OBSERVATION_VERSION,
        "status": "observed" if not issues else "needs_review",
        "slots": ordered, "pair_count": pairs, "tensor_count": tensor_count,
        "accounted_tensor_count": covered_tensors, "unknown_tensor_count": unknown_tensors,
        "unknown_module_count": unknown_modules, "issue_count": len(issues), "issues": issues[:30],
        "factor_formats": sorted(formats), "all_tensors_accounted": not issues and covered_tensors == tensor_count,
        "architecture_verified": False, "target_mapping_verified": False, "export_verified": False,
        "tensor_contents_verified": False, "measurements_available": False,
        "total_model_depth": None, "identity": identity,
        "basis": "Observed tensor names and pair shapes only; sparse indices do not establish total model depth. Counts are not block weights or measured contributions.",
    }


def inspect_file(path):
    header, file_identity = read_header(Path(path), include_metadata=True)
    result = observe_header(header)
    result["file_identity"] = file_identity
    return result
