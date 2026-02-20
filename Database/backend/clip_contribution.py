from __future__ import annotations

from typing import Iterable, Tuple


CLIP_KEY_SUBSTRINGS = (
    "text_encoder",
    "text_model",
    "text_model_encoder",
    "clip",
    "te1",
    "te2",
    "lora_te1",
    "lora_te2",
)


def is_clip_contributor(keys: Iterable[str]) -> Tuple[bool, int]:
    """
    Determine clip contribution evidence strictly from safetensors key names.

    Returns:
      (clip_contributor, clip_tensor_count)
    """
    clip_tensor_count = 0
    for key in keys:
        key_lower = (key or "").lower()
        if any(token in key_lower for token in CLIP_KEY_SUBSTRINGS):
            clip_tensor_count += 1
    return clip_tensor_count > 0, clip_tensor_count

