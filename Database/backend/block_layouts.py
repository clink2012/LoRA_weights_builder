from __future__ import annotations

import re
from typing import Optional

FLUX_FALLBACK_16 = "flux_fallback_16"
UNET_57 = "unet_57"
FLUX_UNET_57 = "flux_unet_57"
FALLBACK_LAYOUTS = {FLUX_FALLBACK_16, UNET_57, FLUX_UNET_57}

_FLUX_TRANSFORMER_RE = re.compile(r"^flux_transformer_(\d+)$")
_FLUX_DOUBLE_RE = re.compile(r"^flux_double_(\d+)$")
_FLUX_TE_RE = re.compile(r"^flux_te_(\d+)$")
_WAN_UNET_RE = re.compile(r"^wan_unet_(\d+)$")
_WAN_MODE_UNET_RE = re.compile(r"^wan_([a-z0-9]+)_unet_(\d+)$")


def _extract_count(layout: str) -> Optional[int]:
    if layout == FLUX_FALLBACK_16:
        return 16
    if layout in {UNET_57, FLUX_UNET_57}:
        return 57

    for pattern in (
        _FLUX_TRANSFORMER_RE,
        _FLUX_DOUBLE_RE,
        _FLUX_TE_RE,
        _WAN_UNET_RE,
        _WAN_MODE_UNET_RE,
    ):
        match = pattern.match(layout)
        if match:
            value = match.group(match.lastindex or 1)
            return int(value)

    return None


def normalize_block_layout(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None

    value = str(raw).strip().lower()
    if not value:
        return None

    if value in {FLUX_FALLBACK_16, UNET_57, FLUX_UNET_57}:
        return value

    if (
        _FLUX_TRANSFORMER_RE.match(value)
        or _FLUX_DOUBLE_RE.match(value)
        or _FLUX_TE_RE.match(value)
        or _WAN_UNET_RE.match(value)
        or _WAN_MODE_UNET_RE.match(value)
    ):
        return value

    return None


def expected_block_count_for_layout(layout: str) -> Optional[int]:
    normalized = normalize_block_layout(layout)
    if normalized is None:
        return None
    return _extract_count(normalized)


def fallback_block_count_for_layout(layout: Optional[str]) -> Optional[int]:
    normalized = normalize_block_layout(layout)
    if normalized is None or normalized not in FALLBACK_LAYOUTS:
        return None
    return expected_block_count_for_layout(normalized)


def infer_layout_from_block_count(block_count: int) -> Optional[str]:
    if block_count == 57:
        return UNET_57
    if block_count == 16:
        return FLUX_FALLBACK_16
    return None


def make_flux_layout(lora_type: Optional[str], block_count: int) -> Optional[str]:
    if block_count <= 0:
        return None

    lora_type_norm = (lora_type or "").lower()

    if "single_transformer_blocks" in lora_type_norm:
        return f"flux_transformer_{block_count}"
    # Handle inspector label: "Flux (UNet double+single blocks)"
    if "unet double+single blocks" in lora_type_norm or "double+single" in lora_type_norm:
        if block_count == 57:
            return FLUX_UNET_57
        else:
            return None

# Handle legacy/double block wording
    if "unet double_blocks" in lora_type_norm or "double_blocks" in lora_type_norm:
        return f"flux_double_{block_count}"

    if "text-encoder" in lora_type_norm:
        return f"flux_te_{block_count}"

    return None


def _self_test() -> None:
    samples = [
        "flux_fallback_16",
        "UNET_57",
        "flux_transformer_38",
        "flux_double_19",
        "flux_te_24",
        "wan_unet_42",
        "wan_t2v_unet_64",
        "invalid_layout",
    ]

    print("=== block_layouts self-test ===")
    for raw in samples:
        normalized = normalize_block_layout(raw)
        expected = expected_block_count_for_layout(normalized) if normalized else None
        print(f"raw={raw!r} -> normalized={normalized!r}, expected_count={expected!r}")

    print("Infer 57:", infer_layout_from_block_count(57))
    print("Infer 16:", infer_layout_from_block_count(16))
    print("Infer 38:", infer_layout_from_block_count(38))
    print("Flux layout from type:", make_flux_layout("Flux (single_transformer_blocks)", 38))


if __name__ == "__main__":
    _self_test()
