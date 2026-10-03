from pathlib import Path
import math
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lora_energy_overlap import LoRAEnergyInput, compute_lora_energy_metrics  # noqa: E402
from lora_block_orchestrator import (  # noqa: E402
    LoraBlockOrchestratorInput,
    orchestrate_lora_block_payloads,
)


def test_phase85_energy_vector_uses_l2_normalization_contract() -> None:
    metrics = compute_lora_energy_metrics(
        LoRAEnergyInput(
            stable_id="A",
            role="character",
            block_weights=[1.0, 0.5, 0.0],
            raw_strength_factor=2.0,
        )
    )

    # energy_blocks = [2.0, 1.0, 0.0]
    # L2 norm = sqrt(2^2 + 1^2) = sqrt(5)
    expected = [2.0 / math.sqrt(5.0), 1.0 / math.sqrt(5.0), 0.0]

    assert metrics.normalized_energy_vector == pytest.approx(expected)


def test_phase85_same_role_overlap_can_change_per_lora_block_vectors_contract() -> None:
    # This target is implemented. Exercise the actual orchestrator instead of
    # retaining an artificial expected failure which only compared list copies.
    scanned_a = [1.0, 1.0, 0.2]
    scanned_b = [1.0, 0.9, 0.1]
    inputs = [
        LoraBlockOrchestratorInput(
            stable_id=stable_id,
            filename=f"{stable_id}.safetensors",
            role="character",
            base_model_code="FLX",
            block_layout="flux_transformer_3",
            text_encoder_contributor=False,
            affect_text_encoder=False,
            strength_model=1.0,
            strength_text_encoder=0.0,
            block_weights=weights,
        )
        for stable_id, weights in [("A", scanned_a), ("B", scanned_b)]
    ]
    outputs = orchestrate_lora_block_payloads(inputs)
    recommended_a, recommended_b = [entry.block_weights for entry in outputs]
    assert recommended_a != scanned_a or recommended_b != scanned_b
    assert all(entry.strength_model == 1.0 for entry in outputs)
    assert scanned_a == [1.0, 1.0, 0.2]
    assert scanned_b == [1.0, 0.9, 0.1]
