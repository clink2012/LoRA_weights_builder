from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


ROUND_DIGITS = 4


@dataclass
class LoRAComposeInput:
    stable_id: str
    base_model_code: Optional[str]
    block_layout: Optional[str]
    block_weights: List[float]


def _round_weights(weights: List[float]) -> List[float]:
    return [round(float(w), ROUND_DIGITS) for w in weights]


def weights_to_csv(weights: List[float]) -> str:
    rounded = _round_weights(weights)
    return ",".join(f"{w:.{ROUND_DIGITS}f}" for w in rounded)


def layout_supports_ab(block_layout: Optional[str]) -> bool:
    if not block_layout:
        return False

    layout = block_layout.lower()
    if layout.startswith("flux_transformer_"):
        return False
    if layout.startswith("flux_te_"):
        return False
    if layout == "flux_fallback_16":
        return False

    return True


def validate_compatibility(loras: List[LoRAComposeInput]) -> Dict[str, Any]:
    reasons: List[str] = []
    if not loras:
        reasons.append("No LoRAs available for validation.")
        return {
            "compatible": False,
            "reasons": reasons,
            "validated_base_model": None,
            "validated_layout": None,
        }

    base_models = {(l.base_model_code or "").upper() for l in loras}
    layouts = {(l.block_layout or "").lower() for l in loras}

    if len(base_models) != 1:
        reasons.append("Selected LoRAs have mismatched base_model_code values.")
    if len(layouts) != 1:
        reasons.append("Selected LoRAs have mismatched block_layout values.")

    compatible = len(reasons) == 0
    if compatible:
        validated_base_model = next(iter(base_models)) or None
        validated_layout = next(iter(layouts)) or None
    else:
        validated_base_model = None
        validated_layout = None

    return {
        "compatible": compatible,
        "reasons": reasons,
        "validated_base_model": validated_base_model,
        "validated_layout": validated_layout,
    }


def combine_weights_weighted_average(
    included_loras: List[LoRAComposeInput],
    per_lora: Dict[str, Dict[str, Any]],
    validated_layout: Optional[str],
) -> Dict[str, Any]:
    warnings: List[str] = []

    if not included_loras:
        return {
            "warnings": ["No LoRAs available to combine."],
            "combined": {
                "strength_model": 1.0,
                "strength_clip": None,
                "A": None,
                "B": None,
                "block_weights": [],
                "block_weights_csv": "",
            },
        }

    expected_len = len(included_loras[0].block_weights)
    for lora in included_loras:
        if len(lora.block_weights) != expected_len:
            raise ValueError("Included LoRAs have different block weight lengths.")

    model_strengths = [
        float(per_lora.get(lora.stable_id, {}).get("strength_model", 1.0))
        for lora in included_loras
    ]
    model_denom = sum(model_strengths)

    if model_denom == 0:
        combined_model = [0.0] * expected_len
        warnings.append("Sum of strength_model values is 0; returned all-zero combined model weights.")
    else:
        combined_model = []
        for idx in range(expected_len):
            numerator = sum(
                lora.block_weights[idx] * model_strengths[pos]
                for pos, lora in enumerate(included_loras)
            )
            combined_model.append(numerator / model_denom)

    clip_contributors: List[tuple[List[float], float, str]] = []
    for lora in included_loras:
        cfg = per_lora.get(lora.stable_id, {})
        affect_clip = bool(cfg.get("affect_clip", True))
        strength_clip = float(cfg.get("strength_clip", 0.0))
        if affect_clip and strength_clip != 0:
            clip_contributors.append((lora.block_weights, strength_clip, lora.stable_id))

    combined_clip: Optional[List[float]] = None
    if clip_contributors:
        clip_denom = sum(c[1] for c in clip_contributors)
        if clip_denom == 0:
            combined_clip = [0.0] * expected_len
            warnings.append("Sum of eligible strength_clip values is 0; returned all-zero clip weights.")
        else:
            combined_clip = []
            for idx in range(expected_len):
                numerator = sum(weights[idx] * strength for weights, strength, _sid in clip_contributors)
                combined_clip.append(numerator / clip_denom)
    else:
        warnings.append("No LoRAs contributed to CLIP combine; combined CLIP weights were returned as null.")

    combined_a: Optional[float] = None
    combined_b: Optional[float] = None
    if layout_supports_ab(validated_layout):
        if model_denom == 0:
            combined_a = 0.0
            combined_b = 0.0
        else:
            combined_a = sum(
                float(per_lora.get(lora.stable_id, {}).get("A", 1.0)) * model_strengths[pos]
                for pos, lora in enumerate(included_loras)
            ) / model_denom
            combined_b = sum(
                float(per_lora.get(lora.stable_id, {}).get("B", 1.0)) * model_strengths[pos]
                for pos, lora in enumerate(included_loras)
            ) / model_denom

    rounded_model = _round_weights(combined_model)

    return {
        "warnings": warnings,
        "combined": {
            "strength_model": 1.0,
            "strength_clip": None if combined_clip is None else 0.0,
            "A": None if combined_a is None else round(combined_a, ROUND_DIGITS),
            "B": None if combined_b is None else round(combined_b, ROUND_DIGITS),
            "block_weights": rounded_model,
            "block_weights_csv": weights_to_csv(rounded_model),
        },
    }
