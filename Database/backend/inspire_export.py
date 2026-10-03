"""Pure, fail-closed numeric export for one pinned Inspire FLUX.1 loader.

Analysis vectors are not loader vectors. Inspire consumes slots only when the
numeric block index changes, including across the double/single boundary.
Resolved coverage means Comfy's *loaded patch keys*, not raw safetensors keys or
nonzero energy indices. No model loading, database access or inference occurs here.
"""
from __future__ import annotations

import math
import re
from decimal import Decimal
from typing import Any, Sequence

LOADER_SHA256 = "4e6188d8d20e2a13c482d2d7fae6ddb570210c426265605b7760b5b96ebb914e"
ADAPTER_ID = "inspire_flux1_v1"
ARCHITECTURE_LABELS = ["BASE"] + [f"DOUBLE {i}" for i in range(19)] + [f"SINGLE {i}" for i in range(38)]
_BLOCK = re.compile(r"^diffusion_model\.(double_blocks|single_blocks)\.(\d+)\..+$")


def build_inspire_flux1_export(
    block_weights: Sequence[float],
    *,
    base_model_code: str | None,
    block_layout: str | None,
    base_weight: float = 1.0,
    resolved_patch_keys: Sequence[str | tuple] | None = None,
    base_patch_keys: Sequence[str] = (),
    coverage_complete: bool = False,
    coverage_source: str | None = None,
    loader_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Return an explicit capability result; never infer coverage from energies.

    `base_patch_keys` identifies independently resolved non-block patches. Unknown
    keys are rejected even when another key in the file looks like FLUX. BASE is
    explicit and is separate from the outer model/CLIP strengths.
    """
    result: dict[str, Any] = {
        "adapter_id": ADAPTER_ID,
        "loader_source_sha256": LOADER_SHA256,
        "status": "blocked", "reason_code": "", "reason": "",
        "numeric_csv": None, "slot_labels": [], "slot_values": [],
        "architecture_slot_count": 58, "loader_slot_count": 0,
        "architecture_slot_labels": list(ARCHITECTURE_LABELS),
        "architecture_slot_values": [], "coverage_source": coverage_source,
        "recommendation_basis": "heuristic_unvalidated",
    }

    def blocked(code: str, reason: str) -> dict[str, Any]:
        result.update(reason_code=code, reason=reason)
        return result

    if base_model_code != "FLX" or block_layout != "flux_transformer_57":
        return blocked("unsupported_architecture", "This adapter requires an explicit FLUX.1 19-double / 38-single layout.")
    try:
        if any(isinstance(value, bool) for value in [base_weight, *block_weights]):
            raise ValueError("boolean weight")
        values = [float(base_weight), *(float(v) for v in block_weights)]
    except (ValueError, TypeError, OverflowError):
        return blocked("invalid_weights", "Weights must be finite numbers.")
    if len(values) != 58 or not all(math.isfinite(v) for v in values):
        return blocked("invalid_weights", "FLUX.1 requires BASE plus 57 finite block weights.")
    result["architecture_slot_values"] = values
    if resolved_patch_keys is None or not coverage_complete or coverage_source not in {"comfy_resolved_patch_keys", "statically_resolved_against_pinned_target"}:
        return blocked("patch_coverage_unknown", "Resolved Comfy patch coverage is required before this vector can be copied into Inspire.")
    if loader_source_sha256 != LOADER_SHA256:
        return blocked("loader_version_unverified", "The target loader source does not match the tested adapter.")
    if not resolved_patch_keys:
        return blocked("empty_patch_coverage", "No resolved patches were supplied.")

    groups: dict[str, set[int]] = {"double_blocks": set(), "single_blocks": set()}
    base_keys = set(base_patch_keys)
    seen_keys: set[str] = set()
    for raw_key in resolved_patch_keys:
        key = raw_key[0] if isinstance(raw_key, tuple) and raw_key else raw_key
        if not isinstance(key, str):
            return blocked("unknown_patch_key", "Resolved patch keys must be strings or Comfy key tuples.")
        seen_keys.add(key)
        match = _BLOCK.fullmatch(key)
        if match:
            group, raw_index = match.groups()
            index = int(raw_index)
            limit = 19 if group == "double_blocks" else 38
            if index >= limit or raw_index != str(index):
                return blocked("unknown_patch_key", "A resolved block index is outside the verified FLUX.1 mapping.")
            groups[group].add(index)
        elif key not in base_keys:
            return blocked("unknown_patch_key", "An unclassified resolved patch prevents a trustworthy export.")
    if not base_keys.issubset(seen_keys):
        return blocked("invalid_base_coverage", "Declared BASE patches must occur in the resolved patch set.")
    if any(_BLOCK.fullmatch(key) for key in base_keys):
        return blocked("invalid_base_coverage", "Transformer blocks cannot be declared as BASE patches.")

    slots = [values[0]]
    labels = ["BASE"]
    last_index = None
    for group, offset, display in (("double_blocks", 1, "DOUBLE"), ("single_blocks", 20, "SINGLE")):
        for index in sorted(groups[group]):
            value = values[offset + index]
            label = f"{display} {index}"
            if index == last_index:
                if slots[-1] != value:
                    return blocked("group_index_collision", "This Inspire version shares a slot across equal double/single boundary indices; different requested weights cannot be represented.")
                labels[-1] += f" + {label}"
            else:
                slots.append(value)
                labels.append(label)
            last_index = index
    # Installed validate() requires >=12 fields even for a sparse/BASE-only LoRA.
    # These trailing zeros are ignored by this exact resolved patch set.
    while len(slots) < 12:
        labels.append(f"UNUSED PADDING {len(slots)}")
        slots.append(0.0)
    result.update(
        status="ready", reason_code="mapped_to_pinned_loader",
        reason="Numeric slots match the supplied resolved patches and pinned loader; image quality remains unvalidated.",
        # Inspire's numeric validator does not accept exponent notation.
        numeric_csv=",".join(format(Decimal(str(v)), "f") for v in slots),
        slot_labels=labels, slot_values=slots, loader_slot_count=len(slots),
    )
    return result
