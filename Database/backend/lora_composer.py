from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


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
    reasons: List[Dict[str, Any]] = []
    if not loras:
        reasons.append(
            {
                "code": "no_loras",
                "detail": "No LoRAs available for validation.",
                "stable_ids": [],
            }
        )
        return {
            "compatible": False,
            "reasons": reasons,
            "validated_base_model": None,
            "validated_layout": None,
        }

    base_models = {(l.base_model_code or "").upper() for l in loras}
    layouts = {(l.block_layout or "").lower() for l in loras}
    stable_ids = [l.stable_id for l in loras]

    if len(base_models) != 1:
        reasons.append(
            {
                "code": "base_model_mismatch",
                "detail": "Selected LoRAs have mismatched base_model_code values.",
                "stable_ids": stable_ids,
            }
        )
    if len(layouts) != 1:
        reasons.append(
            {
                "code": "layout_mismatch",
                "detail": "Selected LoRAs have mismatched block_layout values.",
                "stable_ids": stable_ids,
            }
        )

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


def _combine_by_strength(
    weighted_inputs: List[Tuple[List[float], float]],
    expected_len: int,
) -> Optional[List[float]]:
    if not weighted_inputs:
        return None

    denominator = sum(weight for _weights, weight in weighted_inputs)
    if denominator == 0:
        return [0.0] * expected_len

    combined: List[float] = []
    for idx in range(expected_len):
        numerator = sum(weights[idx] * strength for weights, strength in weighted_inputs)
        combined.append(numerator / denominator)
    return combined


def combine_weights_weighted_average(
    included_loras: List[LoRAComposeInput],
    per_lora: Dict[str, Dict[str, Any]],
    validated_layout: Optional[str],
) -> Dict[str, Any]:
    warnings: List[str] = []

    if not included_loras:
        return {
            "combined_model": [],
            "combined_clip": None,
            "combined_A": None,
            "combined_B": None,
            "strength_model_output": 1.0,
            "strength_clip_output": None,
            "warnings": ["No LoRAs available to combine."],
        }

    expected_len = len(included_loras[0].block_weights)
    for lora in included_loras:
        if len(lora.block_weights) != expected_len:
            raise ValueError("Included LoRAs have different block weight lengths.")

    model_inputs: List[Tuple[List[float], float]] = []
    model_strengths: List[float] = []
    for lora in included_loras:
        strength_model = float(per_lora.get(lora.stable_id, {}).get("strength_model", 1.0))
        model_inputs.append((lora.block_weights, strength_model))
        model_strengths.append(strength_model)

    model_denom = sum(model_strengths)
    if model_denom == 0:
        combined_model = [0.0] * expected_len
        warnings.append(
            "Sum of strength_model values is 0; returned all-zero combined model weights."
        )
    else:
        combined_model = _combine_by_strength(model_inputs, expected_len) or [0.0] * expected_len

    clip_inputs: List[Tuple[List[float], float]] = []
    clip_strengths: List[float] = []
    for lora in included_loras:
        cfg = per_lora.get(lora.stable_id, {})
        affect_clip = bool(cfg.get("affect_clip", True))
        strength_clip = float(cfg.get("strength_clip", 0.0))
        if affect_clip and strength_clip != 0:
            clip_inputs.append((lora.block_weights, strength_clip))
            clip_strengths.append(strength_clip)

    combined_clip: Optional[List[float]] = None
    strength_clip_output: Optional[float] = None
    if not clip_inputs:
        warnings.append("No clip contributors; clip weights omitted.")
    else:
        combined_clip = _combine_by_strength(clip_inputs, expected_len)
        if sum(clip_strengths) == 0:
            warnings.append(
                "Sum of eligible strength_clip values is 0; returned all-zero clip weights."
            )
        # Strength output is aggregate requested clip intensity from contributors.
        strength_clip_output = sum(abs(v) for v in clip_strengths)

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

    return {
        "combined_model": _round_weights(combined_model),
        "combined_clip": None if combined_clip is None else _round_weights(combined_clip),
        "combined_A": None if combined_a is None else round(combined_a, ROUND_DIGITS),
        "combined_B": None if combined_b is None else round(combined_b, ROUND_DIGITS),
        "strength_model_output": 1.0,
        "strength_clip_output": None
        if strength_clip_output is None
        else round(strength_clip_output, ROUND_DIGITS),
        "warnings": warnings,
    }
