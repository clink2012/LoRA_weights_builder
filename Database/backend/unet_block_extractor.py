from __future__ import annotations

import re
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open

UNET_57_BLOCK_COUNT = 57

# Mapping strategy (57 logical blocks):
#   0..3   : input/time/add embedding stem
#            [conv_in, time_embedding.linear_1, time_embedding.linear_2, add/class embedding]
#   4..23  : down_blocks (4 blocks x 5 slots)
#            slot order: resnet_0, resnet_1, attention_0, attention_1, downsampler_0
#   24..26 : mid_block (resnet_0, attention_0, resnet_1)
#   27..54 : up_blocks (4 blocks x 7 slots)
#            slot order: resnet_0, resnet_1, resnet_2, attention_0, attention_1, attention_2, upsampler_0
#   55..56 : output head [conv_norm_out, conv_out]
#
# This is intentionally deterministic and purely name-based; unknown UNet-style keys fail fast.

_STEM_MATCHERS: List[Tuple[int, re.Pattern[str]]] = [
    (0, re.compile(r"(?:^|[._])(?:lora_)?unet[._]conv_in(?:[._]|$)", re.IGNORECASE)),
    (1, re.compile(r"(?:^|[._])(?:lora_)?unet[._](?:time_embedding|time_embed)[._](?:linear_1|0)(?:[._]|$)", re.IGNORECASE)),
    (2, re.compile(r"(?:^|[._])(?:lora_)?unet[._](?:time_embedding|time_embed)[._](?:linear_2|2)(?:[._]|$)", re.IGNORECASE)),
    (3, re.compile(r"(?:^|[._])(?:lora_)?unet[._](?:add_embedding|class_embedding)(?:[._]|$)", re.IGNORECASE)),
    (55, re.compile(r"(?:^|[._])(?:lora_)?unet[._]conv_norm_out(?:[._]|$)", re.IGNORECASE)),
    (56, re.compile(r"(?:^|[._])(?:lora_)?unet[._]conv_out(?:[._]|$)", re.IGNORECASE)),
]

_DOWN_RE = re.compile(
    r"(?:^|[._])(?:lora_)?unet[._]down_blocks[._](\d+)[._](resnets|attentions|downsamplers)[._](\d+)(?:[._]|$)",
    re.IGNORECASE,
)
_MID_RE = re.compile(
    r"(?:^|[._])(?:lora_)?unet[._]mid_block[._](resnets|attentions)[._](\d+)(?:[._]|$)",
    re.IGNORECASE,
)
_UP_RE = re.compile(
    r"(?:^|[._])(?:lora_)?unet[._]up_blocks[._](\d+)[._](resnets|attentions|upsamplers)[._](\d+)(?:[._]|$)",
    re.IGNORECASE,
)


def _tensor_norm(value: torch.Tensor) -> float:
    return float(value.norm().item())


def _down_index(block: int, family: str, inner: int) -> int:
    if block < 0 or block > 3:
        raise ValueError(f"Unsupported down_blocks index: {block}")
    family = family.lower()
    family_offset = {
        ("resnets", 0): 0,
        ("resnets", 1): 1,
        ("attentions", 0): 2,
        ("attentions", 1): 3,
        ("downsamplers", 0): 4,
    }.get((family, inner))
    if family_offset is None:
        raise ValueError(f"Unsupported down block family/index: {family}[{inner}]")
    return 4 + (block * 5) + family_offset


def _mid_index(family: str, inner: int) -> int:
    family = family.lower()
    mapping = {
        ("resnets", 0): 24,
        ("attentions", 0): 25,
        ("resnets", 1): 26,
    }
    idx = mapping.get((family, inner))
    if idx is None:
        raise ValueError(f"Unsupported mid_block family/index: {family}[{inner}]")
    return idx


def _up_index(block: int, family: str, inner: int) -> int:
    if block < 0 or block > 3:
        raise ValueError(f"Unsupported up_blocks index: {block}")
    family = family.lower()
    family_offset = {
        ("resnets", 0): 0,
        ("resnets", 1): 1,
        ("resnets", 2): 2,
        ("attentions", 0): 3,
        ("attentions", 1): 4,
        ("attentions", 2): 5,
        ("upsamplers", 0): 6,
    }.get((family, inner))
    if family_offset is None:
        raise ValueError(f"Unsupported up block family/index: {family}[{inner}]")
    return 27 + (block * 7) + family_offset


def _match_block_index(key: str) -> int | None:
    for idx, pattern in _STEM_MATCHERS:
        if pattern.search(key):
            return idx

    m = _DOWN_RE.search(key)
    if m:
        return _down_index(int(m.group(1)), m.group(2), int(m.group(3)))

    m = _MID_RE.search(key)
    if m:
        return _mid_index(m.group(1), int(m.group(2)))

    m = _UP_RE.search(key)
    if m:
        return _up_index(int(m.group(1)), m.group(2), int(m.group(3)))

    return None


def _is_unet_candidate_key(key: str) -> bool:
    key_l = key.lower()
    return "unet" in key_l or "down_blocks" in key_l or "up_blocks" in key_l or "mid_block" in key_l


def extract_unet_57_block_strengths(safetensors_path: str) -> Tuple[List[float], List[float]]:
    buckets: Dict[int, float] = {i: 0.0 for i in range(UNET_57_BLOCK_COUNT)}
    saw_unet_key = False

    with safe_open(safetensors_path, framework="pt") as tensor_file:
        for key in tensor_file.keys():
            if not _is_unet_candidate_key(key):
                continue
            saw_unet_key = True

            block_idx = _match_block_index(key)
            if block_idx is None:
                raise ValueError(f"UNet key could not be mapped to 57-block layout: {key}")

            tensor = tensor_file.get_tensor(key)
            buckets[block_idx] += _tensor_norm(tensor)

    if not saw_unet_key:
        raise ValueError("No UNet-style keys found in safetensors file.")

    raw = [buckets[i] for i in range(UNET_57_BLOCK_COUNT)]
    max_value = max(raw) if raw else 0.0
    if max_value <= 0:
        raise ValueError("UNet-style keys were found but all strengths are zero.")

    norm = [round(v / max_value, 6) for v in raw]
    return raw, norm


def extract_unet_57_block_weights(safetensors_path: str) -> List[float]:
    """Extract deterministic, normalized UNet 57-block weights from a safetensors file."""
    _, norm = extract_unet_57_block_strengths(safetensors_path)
    return norm
