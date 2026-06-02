from itertools import combinations
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lora_block_orchestrator import (  # noqa: E402
    LoraBlockOrchestratorInput,
    orchestrate_lora_block_payloads,
)
from lora_energy_overlap import (  # noqa: E402
    LoRAEnergyInput,
    OVERLAP_THRESHOLD,
    compute_lora_energy_metrics,
    dot_overlap,
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


def _cosine_overlap(
    left: LoraBlockOrchestratorInput,
    right: LoraBlockOrchestratorInput,
    left_weights: list[float],
    right_weights: list[float],
) -> float:
    left_metrics = compute_lora_energy_metrics(
        LoRAEnergyInput(left.stable_id, left.role, left_weights, left.strength_model)
    )
    right_metrics = compute_lora_energy_metrics(
        LoRAEnergyInput(right.stable_id, right.role, right_weights, right.strength_model)
    )
    return dot_overlap(
        left_metrics.normalized_energy_vector,
        right_metrics.normalized_energy_vector,
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
    left = _input("FLX-AAA-001", role="character", weights=[1.0, 1.0, 0.2])
    right = _input("FLX-BBB-002", role="character", weights=[1.0, 0.9, 0.1])
    initial_overlap = _cosine_overlap(left, right, left.block_weights, right.block_weights)

    outputs = orchestrate_lora_block_payloads([left, right])
    by_id = {output.stable_id: output for output in outputs}

    final_overlap = _cosine_overlap(
        left,
        right,
        by_id["FLX-AAA-001"].block_weights,
        by_id["FLX-BBB-002"].block_weights,
    )

    # Stable-id ordering keeps FLX-AAA-001 intact on exact ties and softens the
    # later/lower-energy overlapping peer. This is deterministic, not divide-by-N
    # scaling, and it must reduce the measured cosine overlap.
    assert initial_overlap > OVERLAP_THRESHOLD
    assert by_id["FLX-AAA-001"].block_weights == [1.0, 1.0, 0.2]
    assert by_id["FLX-BBB-002"].block_weights != [1.0, 0.9, 0.1]
    assert final_overlap < initial_overlap
    assert final_overlap <= OVERLAP_THRESHOLD
    assert any("Same-role" in note for note in by_id["FLX-BBB-002"].notes)


def test_orchestrator_rechecks_triplet_after_later_pair_adjustments() -> None:
    inputs = [
        _input(
            "ID0",
            role="character",
            block_layout="flux_transformer_4",
            strength_model=1.0,
            weights=[0.2597, 4.7248, 2.6514, 2.1151],
        ),
        _input(
            "ID1",
            role="character",
            block_layout="flux_transformer_4",
            strength_model=1.0,
            weights=[0.1184, 5.4682, 2.5122, 1.3267],
        ),
        _input(
            "ID2",
            role="character",
            block_layout="flux_transformer_4",
            strength_model=1.0,
            weights=[0.4105, 1.5778, 3.5730, 1.1398],
        ),
    ]

    assert _cosine_overlap(
        inputs[0],
        inputs[1],
        inputs[0].block_weights,
        inputs[1].block_weights,
    ) > OVERLAP_THRESHOLD

    outputs = orchestrate_lora_block_payloads(inputs)
    by_id = {output.stable_id: output for output in outputs}

    for left, right in combinations(inputs, 2):
        final_overlap = _cosine_overlap(
            left,
            right,
            by_id[left.stable_id].block_weights,
            by_id[right.stable_id].block_weights,
        )
        assert final_overlap <= OVERLAP_THRESHOLD + 1e-6


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
