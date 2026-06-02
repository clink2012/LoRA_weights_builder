from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lora_block_orchestrator import (  # noqa: E402
    LoraBlockOrchestratorInput,
    orchestrate_lora_block_payloads,
)


def _input(
    stable_id: str,
    *,
    role: str = "character",
    block_layout: str = "flux_transformer_3",
    text_encoder_contributor: bool = False,
    affect_text_encoder: bool = True,
    strength_model: float = 1.25,
    strength_text_encoder: float = 0.0,
    weights: list[float] | None = None,
) -> LoraBlockOrchestratorInput:
    return LoraBlockOrchestratorInput(
        stable_id=stable_id,
        filename=f"{stable_id}.safetensors",
        role=role,
        base_model_code="FLX",
        block_layout=block_layout,
        text_encoder_contributor=text_encoder_contributor,
        affect_text_encoder=affect_text_encoder,
        strength_model=strength_model,
        strength_text_encoder=strength_text_encoder,
        block_weights=weights or [0.2, 0.4, 0.6],
    )


def test_orchestrator_returns_one_payload_per_lora_in_order() -> None:
    outputs = orchestrate_lora_block_payloads([
        _input("FLX-AAA-001"),
        _input("FLX-BBB-002", role="style"),
    ])

    assert [output.stable_id for output in outputs] == ["FLX-AAA-001", "FLX-BBB-002"]
    assert [output.role for output in outputs] == ["character", "style"]


def test_orchestrator_preserves_scanned_block_vectors_when_no_softening_applies() -> None:
    outputs = orchestrate_lora_block_payloads([
        _input("FLX-AAA-001", weights=[0.12345, 0.5, 1.0]),
    ])

    assert outputs[0].block_weights == [0.12345, 0.5, 1.0]
    assert outputs[0].block_weights_csv == "0.1235,0.5000,1.0000"
    assert outputs[0].notes


def test_orchestrator_softens_overlapping_same_role_block_vectors() -> None:
    outputs = orchestrate_lora_block_payloads([
        _input("FLX-AAA-001", role="character", weights=[1.0, 1.0, 0.2]),
        _input("FLX-BBB-002", role="character", weights=[1.0, 0.9, 0.1]),
    ])

    by_id = {output.stable_id: output for output in outputs}

    # Stable-id ordering keeps FLX-AAA-001 intact on exact ties and softens the
    # later/overlapping peer. This is deterministic, not divide-by-N scaling.
    assert by_id["FLX-AAA-001"].block_weights == [1.0, 1.0, 0.2]
    assert by_id["FLX-BBB-002"].block_weights != [1.0, 0.9, 0.1]
    assert by_id["FLX-BBB-002"].block_weights[0] < 1.0
    assert by_id["FLX-BBB-002"].block_weights[1] < 0.9
    assert by_id["FLX-BBB-002"].block_weights[2] < 0.1
    assert any("Same-role" in note for note in by_id["FLX-BBB-002"].notes)


def test_orchestrator_does_not_soften_different_roles() -> None:
    outputs = orchestrate_lora_block_payloads([
        _input("FLX-AAA-001", role="character", weights=[1.0, 1.0, 0.2]),
        _input("FLX-BBB-002", role="style", weights=[1.0, 0.9, 0.1]),
    ])

    by_id = {output.stable_id: output for output in outputs}
    assert by_id["FLX-AAA-001"].block_weights == [1.0, 1.0, 0.2]
    assert by_id["FLX-BBB-002"].block_weights == [1.0, 0.9, 0.1]


def test_orchestrator_does_not_soften_different_layouts() -> None:
    outputs = orchestrate_lora_block_payloads([
        _input("FLX-AAA-001", role="character", block_layout="flux_transformer_3", weights=[1.0, 1.0, 0.2]),
        _input("FLX-BBB-002", role="character", block_layout="flux_double_3", weights=[1.0, 0.9, 0.1]),
    ])

    by_id = {output.stable_id: output for output in outputs}
    assert by_id["FLX-AAA-001"].block_weights == [1.0, 1.0, 0.2]
    assert by_id["FLX-BBB-002"].block_weights == [1.0, 0.9, 0.1]


def test_orchestrator_csv_matches_softened_block_vector() -> None:
    outputs = orchestrate_lora_block_payloads([
        _input("FLX-AAA-001", role="character", weights=[1.0, 1.0, 0.2]),
        _input("FLX-BBB-002", role="character", weights=[1.0, 0.9, 0.1]),
    ])

    softened = next(output for output in outputs if output.stable_id == "FLX-BBB-002")
    csv_values = [float(value) for value in softened.block_weights_csv.split(",")]
    assert csv_values == pytest.approx([round(value, 4) for value in softened.block_weights])


def test_orchestrator_disables_text_encoder_when_no_text_encoder_tensors() -> None:
    outputs = orchestrate_lora_block_payloads([
        _input(
            "FLX-AAA-001",
            text_encoder_contributor=False,
            affect_text_encoder=True,
            strength_text_encoder=2.0,
        ),
    ])

    assert outputs[0].text_encoder_contributor is False
    assert outputs[0].affect_text_encoder is False
    assert outputs[0].strength_text_encoder == 0.0


def test_orchestrator_preserves_text_encoder_strength_when_allowed() -> None:
    outputs = orchestrate_lora_block_payloads([
        _input(
            "FLX-AAA-001",
            text_encoder_contributor=True,
            affect_text_encoder=True,
            strength_text_encoder=0.75,
        ),
    ])

    assert outputs[0].text_encoder_contributor is True
    assert outputs[0].affect_text_encoder is True
    assert outputs[0].strength_text_encoder == 0.75
