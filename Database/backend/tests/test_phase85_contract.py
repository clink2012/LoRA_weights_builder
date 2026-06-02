from pathlib import Path
import math
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lora_energy_overlap import LoRAEnergyInput, compute_lora_energy_metrics  # noqa: E402


@pytest.mark.xfail(
    reason=(
        "Phase 8.5 target: energy vectors must be L2-normalized so dot-product "
        "overlap is true cosine similarity. Current Phase 8.3 code still uses "
        "sum/total-energy normalization."
    ),
    strict=True,
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


@pytest.mark.xfail(
    reason=(
        "Phase 8.5 target: same-role overlap should be capable of changing "
        "per-LoRA block vectors, not only global strength_model. The current "
        "combine path still returns scanned block vectors unchanged."
    ),
    strict=True,
)
def test_phase85_same_role_overlap_can_change_per_lora_block_vectors_contract() -> None:
    # This is intentionally a target-contract test, not a current implementation
    # test. It describes the required orchestration behaviour before we build the
    # lora_block_orchestrator module.
    scanned_a = [1.0, 1.0, 0.2]
    scanned_b = [1.0, 0.9, 0.1]

    # A future orchestrator should return stack-aware, per-LoRA recommended
    # vectors. For near-identical same-role LoRAs, at least one output vector
    # should be softened/changed to reduce collision.
    recommended_a = list(scanned_a)
    recommended_b = list(scanned_b)

    assert recommended_a != scanned_a or recommended_b != scanned_b
