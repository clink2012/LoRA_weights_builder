"""Header observations only. No checkpoint, loader or version compatibility claims."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re

from flux_header_coverage import CoverageError, read_header

IDENTIFICATION_VERSION = "adapter-header-observations-v1"
CLAIM_KEYS = ("ss_base_model_version", "modelspec.architecture", "ss_network_module", "model_version")
PARTS = (
    (".lora_A.weight", "ab", "down"), (".lora_B.weight", "ab", "up"),
    (".lora_A.default.weight", "ab", "down"), (".lora_B.default.weight", "ab", "up"),
    (".lora_down.weight", "up_down", "down"), (".lora_up.weight", "up_down", "up"),
    (".alpha", "alpha", "alpha"),
)


def _family_claim(value):
    value = str(value).casefold().replace("_", "-")
    if "minimax" in value and ("h3" in value or "h-3" in value):
        return "h3"
    if "ltx" in value:
        return "ltx"
    if "flux" in value:
        return "flux"
    return None


def _topology(module):
    """Recognise evidence patterns, never infer total depth from sparse indices."""
    normalized = module.replace("_", ".")
    for pattern, family, group in (
        (r"(?:^|\.)double\.blocks\.(\d+)\.(.+)", "flux", "double"),
        (r"(?:^|\.)single\.blocks\.(\d+)\.(.+)", "flux", "single"),
        (r"(?:^|\.)transformer\.blocks\.(\d+)\.(.+)", "transformer", "transformer"),
        (r"(?:^|\.)token\.refiner\.blocks\.(\d+)\.(.+)", "h3", "refiner"),
        (r"(?:^|\.)blocks\.(\d+)\.(.+)", "blocks", "main"),
    ):
        match = re.search(pattern, normalized)
        if match:
            if len(match[1]) > 6 or int(match[1]) > 100000:
                return "unknown", "other", None, "other"
            index, component = int(match[1]), match[2]
            if family == "transformer":
                # attn1/attn2 also occur in other architectures: label this LTX-like,
                # and retain explicit ambiguity unless AV-specific modules occur.
                if any(part in component for part in ("audio", "attn1", "attn2")) or component.startswith("ff."):
                    family = "ltx"
                else:
                    family = "unknown"
            elif family == "blocks":
                family = "h3" if component.startswith(("attn.qkv.proj", "attn.out.proj", "mlp.fc1", "mlp.fc2", "adaln.")) else "unknown"
            if "video.to.audio" in component or "audio.to.video" in component:
                modality = "cross_stream"
            elif "audio" in component:
                modality = "audio"
            elif family == "ltx" and component.startswith(("attn1.", "attn2.", "ff.")):
                modality = "video"
            else:
                modality = "other"
            return family, group, index, modality
    return "unknown", "other", None, "other"


def classify_header(header: dict, folder_hint: str | None = None) -> dict:
    """Classify a structurally validated header; metadata remains self-declared."""
    if not isinstance(header, dict):
        raise CoverageError("invalid_header", "Identification needs a validated header dictionary.")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        raise CoverageError("invalid_header", "Identification metadata must contain strings.")
    claims = {key: metadata[key][:4096] for key in CLAIM_KEYS if key in metadata}
    families, formats, modules, unknown = Counter(), Counter(), {}, []
    groups = defaultdict(set)
    modalities = Counter()
    av_specific = False
    for key, tensor in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(key, str) or not isinstance(tensor, dict):
            raise CoverageError("invalid_header", "Identification needs tensor descriptors.")
        part = next(((suffix, fmt, direction) for suffix, fmt, direction in PARTS if key.endswith(suffix)), None)
        if part:
            suffix, fmt, direction = part
            module = key[:-len(suffix)]
            formats[fmt] += 1
            record = modules.setdefault(module, {"parts": set(), "formats": set()})
            record["parts"].add(direction)
            if fmt != "alpha":
                record["formats"].add(fmt)
        else:
            module = key
            formats["unsupported"] += 1
            unknown.append(key)
        family, group, index, modality = _topology(module)
        if family != "unknown":
            families[family] += 1
        if index is not None:
            groups[f"{family}.{group}"].add(index)
        modalities[modality] += 1
        av_specific |= family == "ltx" and modality in ("audio", "cross_stream")
    evidence_families = sorted(families)
    reasons = []
    if not evidence_families:
        reasons.append("No supported family pattern observed; metadata does not establish architecture.")
    if len(evidence_families) > 1:
        reasons.append("Multiple family-like key topologies observed; do not assign a single family.")
    if "flux" in families:
        reasons.append("FLUX double/single topology alone does not distinguish FLUX.1 from FLUX.2 or its variants.")
    if "ltx" in families:
        reasons.append("LTX-like topology does not prove LTX-2.3 versus LTX-2.5, checkpoint variant or exact target shape compatibility.")
        if not av_specific:
            reasons.append("Video-only transformer attention names also occur in other architectures; LTX identity remains ambiguous.")
    if "h3" in families:
        reasons.append("H3-like topology does not prove FL2VA versus Ref2VA or compatibility with pruned/quantized checkpoints.")
        reasons.append("Generic block MLP/attention and token-refiner names are not unique to H3; this is a candidate pattern, not family identification.")
    conflicts = []
    declared = sorted({family for value in claims.values() if (family := _family_claim(value))})
    hint_family = _family_claim(folder_hint)
    if len(declared) > 1:
        conflicts.append("Metadata contains conflicting family declarations.")
    for source, candidates in (("Metadata", declared), ("Folder hint", [hint_family] if hint_family else [])):
        if evidence_families and candidates and set(candidates) != set(evidence_families):
            conflicts.append(f"{source} disagrees with observed family-like key topology.")
    incomplete = [module for module, record in modules.items() if not {"up", "down"}.issubset(record["parts"])]
    mixed_modules = [module for module, record in modules.items() if len(record["formats"]) > 1]
    if formats["unsupported"]:
        reasons.append("Unrecognised adapter tensor formats remain visible; they are not silently discarded.")
    reasons.append("Pair completeness counts tensor names only; pair dimensions, alpha values and checkpoint target shapes have not been verified.")
    return {
        "schema_version": IDENTIFICATION_VERSION, "status": "observations_only",
        "family_candidates": evidence_families, "version_status": "unproven",
        "architecture_verified": False, "export_verified": False,
        "tensor_contents_verified": False, "tensor_payload_read": False,
        "tensor_count": sum(formats.values()),
        "observed_groups": {name: {"indices": sorted(indices), "observed_count": len(indices), "total_depth": None}
                            for name, indices in sorted(groups.items())},
        "modality_tensor_counts": dict(sorted(modalities.items())),
        "modality_basis": "key_names_only; H3 packed multimodal blocks are not classified as video-only",
        "unclassified_target_tensor_count": sum(formats.values()) - sum(families.values()),
        "adapter_formats": dict(sorted(formats.items())),
        "mixed_pair_formats": len([fmt for fmt in formats if fmt not in ("alpha", "unsupported")]) > 1,
        "incomplete_pair_count": len(incomplete), "incomplete_pair_examples": incomplete[:20],
        "mixed_module_format_count": len(mixed_modules),
        "mixed_module_format_examples": mixed_modules[:20],
        "unsupported_tensor_examples": unknown[:20],
        "metadata": {"trust": "self_declared", "claims": claims, "claim_families": declared,
                     "claim_values_truncated": any(len(metadata[key]) > 4096 for key in claims)},
        "folder_hint": {"value": folder_hint, "trust": "unverified", "family": hint_family},
        "disagreements": conflicts, "limitations": reasons,
    }


def identify_file(path: Path, *, folder_hint: str | None = None) -> dict:
    header, identity = read_header(Path(path), include_metadata=True)
    result = classify_header(header, folder_hint)
    result["file_identity"] = {**identity, "basis": "header_stat"}
    return result
