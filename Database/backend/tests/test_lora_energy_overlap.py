import math

import pytest

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lora_energy_overlap import (  # noqa: E402
    LoRAEnergyInput,
    allocate_strengths_with_role_budget_and_overlap,
    build_overlap_matrix,
    compute_lora_energy_metrics,
)


def test_energy_calculation_and_normalization_are_deterministic():
    metrics = compute_lora_energy_metrics(
        LoRAEnergyInput(
            stable_id="A",
            role="character",
            block_weights=[-1.0, 0.5, 0.0],
            raw_strength_factor=2.0,
        )
    )

    assert metrics.energy_blocks == [2.0, 1.0, 0.0]
    assert metrics.total_energy == 3.0
    expected_l2 = math.sqrt(2.0 * 2.0 + 1.0 * 1.0 + 0.0)
    assert metrics.normalized_energy_vector == pytest.approx(
        [2.0 / expected_l2, 1.0 / expected_l2, 0.0]
    )


def test_overlap_matrix_is_symmetric_and_deterministic():
    m1 = compute_lora_energy_metrics(
        LoRAEnergyInput("A", "style", [1.0, 1.0], 1.0)
    )
    m2 = compute_lora_energy_metrics(
        LoRAEnergyInput("B", "style", [1.0, 3.0], 1.0)
    )

    overlap = build_overlap_matrix([m1, m2])

    assert overlap["A"]["B"] == overlap["B"]["A"]

    # With L2-normalized vectors, dot() is cosine similarity.
    # Self-similarity is 1.0.
    assert overlap["A"]["A"] == pytest.approx(1.0)
    assert overlap["B"]["B"] == pytest.approx(1.0)

    # Cross overlap for [1,1] and [1,3] is:
    # (1*1 + 1*3) / (sqrt(2) * sqrt(10)) = 2/sqrt(5)
    assert overlap["A"]["B"] == pytest.approx(2.0 / math.sqrt(5.0))


def test_role_budget_allocation_applies_caps_before_within_role_distribution():
    inputs = [
        LoRAEnergyInput("CHAR", "character", [1.0, 1.0], 1.0),
        LoRAEnergyInput("STYL", "style", [1.0, 1.0], 1.0),
    ]
    metrics = [compute_lora_energy_metrics(item) for item in inputs]

    allocated = allocate_strengths_with_role_budget_and_overlap(metrics)

    # Total requested abs strength = 2.0
    # Character cap = 0.35 * 2.0 = 0.7
    # Style cap = 0.25 * 2.0 = 0.5
    assert allocated["CHAR"] == 0.7
    assert allocated["STYL"] == 0.5


def test_scaling_stability_for_repeated_identical_inputs():
    inputs = [
        LoRAEnergyInput("A", "clothing", [0.2, 0.8], 1.0),
        LoRAEnergyInput("B", "clothing", [0.2, 0.8], 1.0),
    ]
    metrics = [compute_lora_energy_metrics(item) for item in inputs]

    first = allocate_strengths_with_role_budget_and_overlap(metrics)
    second = allocate_strengths_with_role_budget_and_overlap(metrics)

    assert first == second
    assert first["A"] == first["B"]
