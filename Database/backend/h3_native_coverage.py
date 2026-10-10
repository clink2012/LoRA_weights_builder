"""Conditional H3 source-target coverage; not a checkpoint or export capability.

Only the captured native constructor defaults are modeled here. Actual task,
pruned, curve, gate-compressed and quantized checkpoint identities are unproved.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from adapter_identity import PARTS
from flux_header_coverage import CoverageError, read_header

CONTRACT = json.loads((Path(__file__).parent / "contracts/h3-native-constructor-defaults-v1.json").read_text(encoding="utf-8"))


def verify_sources(comfy_root):
    for relative, expected in CONTRACT["source_sha256"].items():
        path = Path(comfy_root) / relative
        try: actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc: raise CoverageError("source_unavailable", "Pinned H3 source is unavailable.") from exc
        if actual != expected:
            raise CoverageError("source_changed", "H3 source changed; native target mapping needs review.")
    return dict(CONTRACT["source_sha256"])


def resolve_header(header):
    """Account for every tensor against exact captured aliases and dimensions."""
    modules, issues = {}, []
    tensor_count = sum(key != "__metadata__" for key in header)
    for key, tensor in header.items():
        if key == "__metadata__": continue
        part = next(((suffix, fmt, direction) for suffix, fmt, direction in PARTS if key.endswith(suffix)), None)
        if not part:
            issues.append({"code": "unsupported_tensor", "module": key[:256]})
            continue
        suffix, fmt, direction = part
        module = key[:-len(suffix)]
        record = modules.setdefault(module, {"parts": {}, "formats": set(), "duplicate": False, "count": 0})
        record["duplicate"] |= direction in record["parts"]
        record["parts"][direction] = tensor
        record["count"] += 1
        if fmt != "alpha": record["formats"].add(fmt)
    alias_targets = Counter(CONTRACT["aliases"].get(module) for module in modules)
    mappings, accounted = [], 0
    for module, record in modules.items():
        target, parts = CONTRACT["aliases"].get(module), record["parts"]
        code = None
        if not target: code = "unmapped_target"
        elif alias_targets[target] != 1 or record["duplicate"]: code = "ambiguous_target"
        elif not {"up", "down"}.issubset(parts): code = "incomplete_pair"
        elif len(record["formats"]) != 1: code = "mixed_pair_format"
        else:
            up, down = parts["up"].get("shape"), parts["down"].get("shape")
            out_dim, in_dim = CONTRACT["target_shapes"][target]
            if (not isinstance(up, list) or not isinstance(down, list) or len(up) != 2 or len(down) != 2
                    or any(type(dim) is not int or dim <= 0 for dim in up + down) or up[1] != down[0]):
                code = "invalid_pair_shape"
            elif up[0] != out_dim or down[1] != in_dim: code = "target_shape_mismatch"
            elif "alpha" in parts and parts["alpha"].get("shape") not in ([], [1]): code = "invalid_alpha_shape"
        if code:
            issues.append({"code": code, "module": module[:256]})
            continue
        if target.startswith("diffusion_model.blocks."):
            group, index = "main", int(target.split(".")[2])
        elif target.startswith("diffusion_model.token_refiner.blocks."):
            group, index = "refiner", int(target.split(".")[3])
        else: group, index = "other", None
        mappings.append({"module": module, "target": target, "group": group, "index": index,
                         "rank": parts["down"]["shape"][0], "alpha_present": "alpha" in parts})
        accounted += record["count"]
    return {"contract_id": CONTRACT["contract_id"], "status": "mapped_candidate" if not issues and mappings else "needs_review",
            "mappings": mappings, "tensor_count": tensor_count, "accounted_tensor_count": accounted,
            "all_tensors_mapped": bool(mappings) and not issues and accounted == tensor_count,
            "issue_count": len(issues), "issue_counts": dict(Counter(issue["code"] for issue in issues)), "issues": issues[:30],
            "main_count": CONTRACT["main_count"], "refiner_count": CONTRACT["refiner_count"],
            "source_target_conditional": True, "checkpoint_verified": False, "architecture_verified": False,
            "patch_application_verified": False, "measurements_available": False, "export_verified": False,
            "basis": CONTRACT["scope"]}


def inspect_file(path, comfy_root):
    sources = verify_sources(comfy_root)
    header, identity = read_header(Path(path), include_metadata=True)
    result = resolve_header(header)
    result.update(file_identity=identity, source_sha256=sources)
    return result
