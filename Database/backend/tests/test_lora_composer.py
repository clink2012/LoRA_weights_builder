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
    assert result["combined_model"] == [0.5, 0.7, 0.9]
    assert result["combined_clip"] is None
    assert result["strength_clip_output"] is None


def test_combine_excludes_fallback_lora_inputs_by_using_only_included_loras():
    included_loras = [
        _mk_lora("FLX-INC-001", "FLX", "flux_transformer_3", [0.2, 0.4, 0.6]),
    ]

    result = combine_weights_weighted_average(
        included_loras=included_loras,
        per_lora={
            "FLX-INC-001": {"strength_model": 1.0},
            "FLX-EXC-999": {"strength_model": 999.0},
        },
        validated_layout="flux_transformer_3",
    )

    assert result["combined_model"] == [0.2, 0.4, 0.6]


def test_combine_rejects_layout_mismatch():
    loras = [
        _mk_lora("FLX-AAA-001", "FLX", "flux_transformer_3", [0.2, 0.4, 0.6]),
        _mk_lora("FLX-BBB-002", "FLX", "flux_transformer_4", [0.6, 0.8, 1.0]),
    ]
    validation = validate_compatibility(loras)

    assert validation["compatible"] is False
    assert any(reason["code"] == "layout_mismatch" for reason in validation["reasons"])
    assert validation["reasons"][0]["stable_ids"] == ["FLX-AAA-001", "FLX-BBB-002"]


def test_combine_rejects_base_model_mismatch():
    loras = [
        _mk_lora("FLX-AAA-001", "FLX", "flux_transformer_3", [0.2, 0.4, 0.6]),
        _mk_lora("SDX-BBB-002", "SDX", "flux_transformer_3", [0.6, 0.8, 1.0]),
    ]
    validation = validate_compatibility(loras)

    assert validation["compatible"] is False
    assert any(reason["code"] == "base_model_mismatch" for reason in validation["reasons"])
    assert validation["reasons"][0]["stable_ids"] == ["FLX-AAA-001", "SDX-BBB-002"]


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

    assert result["combined_model"] == [0.0, 0.0, 0.0]
    assert any("Sum of strength_model values is 0" in w for w in result["warnings"])


def test_combine_clip_toggle_excludes_from_clip_combine_when_disabled_or_zero():
    loras = [
        _mk_lora("FLX-AAA-001", "FLX", "flux_transformer_3", [0.2, 0.4, 0.6]),
        _mk_lora("FLX-BBB-002", "FLX", "flux_transformer_3", [0.6, 0.8, 1.0]),
    ]

    result = combine_weights_weighted_average(
        included_loras=loras,
        per_lora={
            "FLX-AAA-001": {
                "strength_model": 1.0,
                "affect_clip": False,
                "strength_clip": 1.0,
            },
            "FLX-BBB-002": {
                "strength_model": 1.0,
                "affect_clip": True,
                "strength_clip": 0.0,
            },
        },
        validated_layout="flux_transformer_3",
    )

    assert result["combined_clip"] is None
    assert result["strength_clip_output"] is None
    assert any("No clip contributors; clip weights omitted." in w for w in result["warnings"])


def test_combine_clip_success_two_contributors_is_deterministic():
    loras = [
        _mk_lora("FLX-AAA-001", "FLX", "flux_transformer_3", [0.2, 0.4, 0.6]),
        _mk_lora("FLX-BBB-002", "FLX", "flux_transformer_3", [0.6, 0.8, 1.0]),
    ]

    result = combine_weights_weighted_average(
        included_loras=loras,
        per_lora={
            "FLX-AAA-001": {
                "strength_model": 1.0,
                "affect_clip": True,
                "strength_clip": 1.0,
            },
            "FLX-BBB-002": {
                "strength_model": 1.0,
                "affect_clip": True,
                "strength_clip": 3.0,
            },
        },
        validated_layout="flux_transformer_3",
    )

    assert result["combined_clip"] == [0.5, 0.7, 0.9]
    assert result["strength_clip_output"] == 4.0
