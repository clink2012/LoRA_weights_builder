from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

FLUX_FALLBACK_16 = "flux_fallback_16"
UNET_57 = "unet_57"

_LAYOUT_WITH_SUFFIX_RE = re.compile(r"^(flux_(?:transformer|double|te))_(\d+)$")


@dataclass(frozen=True)
class ParsedLayout:
    layout: str
    expected_block_count: int


def parse_block_layout(layout: Optional[str]) -> Optional[ParsedLayout]:
    """
    Parse a block layout identifier from the supported taxonomy.

    Supported:
    - flux_fallback_16
    - flux_transformer_<N>
    - flux_double_<N>
    - flux_te_<N>
    - unet_57
    """
    if not layout:
        return None

    if layout == FLUX_FALLBACK_16:
        return ParsedLayout(layout=layout, expected_block_count=16)

    if layout == UNET_57:
        return ParsedLayout(layout=layout, expected_block_count=57)

    match = _LAYOUT_WITH_SUFFIX_RE.match(layout)
    if match is None:
        return None

    expected = int(match.group(2))
    return ParsedLayout(layout=layout, expected_block_count=expected)


def normalize_block_layout(layout: Optional[str]) -> Optional[str]:
    parsed = parse_block_layout(layout)
    if parsed is None:
        return None
    return parsed.layout


def expected_block_count_for_layout(layout: Optional[str]) -> Optional[int]:
    parsed = parse_block_layout(layout)
    if parsed is None:
        return None
    return parsed.expected_block_count


def make_flux_layout(lora_type: Optional[str], block_count: int) -> Optional[str]:
    """
    Build a Flux layout string from analysis lora_type + block count.
    """
    if block_count <= 0:
        return None

    normalized = (lora_type or "").strip().lower()
    if "single_transformer_blocks" in normalized:
        return f"flux_transformer_{block_count}"
    if "double+single" in normalized or ("unet" in normalized and "double" in normalized and "single" in normalized):
    return f"flux_double_{block_count}"
    if "double_blocks" in normalized:
        return f"flux_double_{block_count}"
    if "text_encoder" in normalized or "text-encoder" in normalized:
        return f"flux_te_{block_count}"

    return None


def infer_layout_from_block_count(block_count: int) -> Optional[str]:
    """
    Conservative inference helper when only count is known.
    """
    if block_count <= 0:
        return None
    if block_count == 57:
        return UNET_57
    return f"flux_transformer_{block_count}"
