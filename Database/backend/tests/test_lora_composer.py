from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lora_composer import (  # noqa: E402
    LoRAComposeInput,
    combine_weights_weighted_average,
    validate_compatibility,
)


def _mk_lora(stable_id, base, layout, weights):
    return LoRAComposeInput(
        stable_id=stable_id,
        base_model_code=base,
        block_layout=layout,
        block_weights=weights,
    )


def test_combine_success_two_loras_same_layout():
    loras = [
        _mk_lora("FLX-AAA-001", "FLX", "flux_transformer_3", [0.2, 0.4, 0.6]),
        _mk_lora("FLX-BBB-002", "FLX", "flux_transformer_3", [0.6, 0.8, 1.0]),
    ]
    validation = validate_compatibility(loras)

    result = combine_weights_weighted_average(
        included_loras=loras,
        per_lora={
            "FLX-AAA-001": {"strength_model": 1.0},
            "FLX-BBB-002": {"strength_model": 3.0},
        },
        validated_layout=validation["validated_layout"],
    )

    assert validation["compatible"] is True
    assert result["combined"]["block_weights"] == [0.5, 0.7, 0.9]
    assert result["combined"]["block_weights_csv"] == "0.5000,0.7000,0.9000"


def test_combine_excludes_fallback_lora():
    selected = [
        _mk_lora("FLX-INC-001", "FLX", "flux_transformer_3", [0.2, 0.4, 0.6]),
    ]
    excluded_loras = ["FLX-EXC-999"]

    validation = validate_compatibility(selected)
    result = combine_weights_weighted_average(
        included_loras=selected,
        per_lora={"FLX-INC-001": {"strength_model": 1.0}},
        validated_layout=validation["validated_layout"],
    )

    assert excluded_loras == ["FLX-EXC-999"]
    assert validation["compatible"] is True
    assert result["combined"]["block_weights"] == [0.2, 0.4, 0.6]


def test_combine_rejects_layout_mismatch():
    loras = [
        _mk_lora("FLX-AAA-001", "FLX", "flux_transformer_3", [0.2, 0.4, 0.6]),
        _mk_lora("FLX-BBB-002", "FLX", "flux_transformer_4", [0.6, 0.8, 1.0]),
    ]
    validation = validate_compatibility(loras)

    assert validation["compatible"] is False
    assert "mismatched block_layout" in validation["reasons"][0] or "mismatched block_layout" in " ".join(validation["reasons"])


def test_combine_rejects_base_model_mismatch():
    loras = [
        _mk_lora("FLX-AAA-001", "FLX", "flux_transformer_3", [0.2, 0.4, 0.6]),
        _mk_lora("SDX-BBB-002", "SDX", "flux_transformer_3", [0.6, 0.8, 1.0]),
    ]
    validation = validate_compatibility(loras)

    assert validation["compatible"] is False
    assert "mismatched base_model_code" in validation["reasons"][0] or "mismatched base_model_code" in " ".join(validation["reasons"])


def test_combine_zero_strength_model_returns_zeros_and_warning():
    loras = [
        _mk_lora("FLX-AAA-001", "FLX", "flux_transformer_3", [0.2, 0.4, 0.6]),
        _mk_lora("FLX-BBB-002", "FLX", "flux_transformer_3", [0.6, 0.8, 1.0]),
    ]

    result = combine_weights_weighted_average(
        included_loras=loras,
        per_lora={
            "FLX-AAA-001": {"strength_model": 0.0},
            "FLX-BBB-002": {"strength_model": 0.0},
        },
        validated_layout="flux_transformer_3",
    )

    assert result["combined"]["block_weights"] == [0.0, 0.0, 0.0]
    assert any("Sum of strength_model values is 0" in w for w in result["warnings"])


def test_combine_clip_toggle_excludes_from_clip_combine():
    loras = [
        _mk_lora("FLX-AAA-001", "FLX", "flux_transformer_3", [0.2, 0.4, 0.6]),
        _mk_lora("FLX-BBB-002", "FLX", "flux_transformer_3", [0.6, 0.8, 1.0]),
    ]

    result = combine_weights_weighted_average(
        included_loras=loras,
        per_lora={
            "FLX-AAA-001": {"strength_model": 1.0, "affect_clip": False, "strength_clip": 1.0},
            "FLX-BBB-002": {"strength_model": 1.0, "affect_clip": True, "strength_clip": 0.0},
        },
        validated_layout="flux_transformer_3",
    )

    assert result["combined"]["strength_clip"] is None
    assert any("No LoRAs contributed to CLIP combine" in w for w in result["warnings"])
