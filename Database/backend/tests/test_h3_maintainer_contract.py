"""Native H3 maintainer-loader observations remain separate from export readiness."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h3_loader_contract import MAINTAINER_SHA256, maintainer_candidate

FIXTURE = json.loads((Path(__file__).parent / "fixtures/h3_maintainer_contract.json").read_text(encoding="utf-8"))
CASES = {case["name"]: case for case in FIXTURE["cases"]}
MAIN3 = "diffusion_model.blocks.3.attn.qkv_proj.weight"
MAIN49 = "diffusion_model.blocks.49.mlp.fc1.weight"
REFINER = "('diffusion_model.token_refiner.blocks.1.attn.qkv_proj.weight', (0, 0, 1))"
OTHER = "diffusion_model.img_in.weight"


def candidate(**overrides):
    args = dict(main_weights=[1.] * 50, refiner_weights=[1.] * 2,
                main_block_count=50, refiner_block_count=2, other_weight=1.,
                model_strength=1., loader_source_sha256=MAINTAINER_SHA256)
    args.update(overrides)
    return maintainer_candidate(**args)


def test_full_graph_sequence_is_explicit_main_refiner_other():
    result = candidate()
    assert FIXTURE["source_sha256"] == MAINTAINER_SHA256
    assert result["slot_labels"] == [f"MAIN {i}" for i in range(50)] + ["REFINER 0", "REFINER 1", "OTHER"]
    assert result["slot_values"] == [1.] * 53
    assert len(result["block_overrides"].splitlines()) == 52
    assert result["other_strength"] == 1.
    assert result["export_verified"] is False
    assert all(value == 1. for key, value in CASES["neutral"]["applied"].items() if key != "prior_adapter")


def test_sparse_separate_groups_preserve_precision_signed_outer_and_patch_objects():
    case = CASES["signed_sparse_separate_refiner"]
    main = [0.6] * 50
    main[3], main[49] = -0.625, 0.1256789
    result = candidate(main_weights=main, refiner_weights=[0.25, 0.4],
                       other_weight=-0.3, model_strength=-0.8)
    lines = dict(line.split("=", 1) for line in result["block_overrides"].splitlines())
    for key, label in [(MAIN3, "blocks.3"), (MAIN49, "blocks.49"), (REFINER, "refiner.1")]:
        assert case["applied"][key] == pytest.approx(float(lines[label]) * result["strength_model"])
    assert case["applied"][OTHER] == pytest.approx(result["other_strength"] * result["strength_model"])
    assert lines["blocks.49"] == "0.1256789"
    assert len(result["slot_values"]) == 53
    assert case["metadata_attached"] is True


def test_override_replaces_group_multiplier_and_later_rule_wins():
    case = CASES["later_overrides_replace_group"]
    assert case["applied"][MAIN3] == pytest.approx(-0.5 * 0.9)
    assert case["applied"][MAIN49] == pytest.approx(0.6 * 0.9)
    assert case["applied"][REFINER] == pytest.approx(0.9)


def test_zero_behavior_and_chained_model_preservation():
    assert CASES["zero_group_drops_patch"]["applied"] == {"prior_adapter": 0.7}
    bypass = CASES["overall_zero_bypasses_loading"]
    assert bypass["file_load_calls"] == 0
    assert bypass["applied"] == {"prior_adapter": 0.7}
    assert all(case["source_model_unchanged"] for case in FIXTURE["cases"])
    assert all(case.get("applied", {}).get("prior_adapter", 0.7) == 0.7 for case in FIXTURE["cases"])


@pytest.mark.parametrize("name", ["invalid_override_rejected", "invalid_number_rejected", "non_native_model_rejected",
                                  "unaccepted_nonzero_patch_rejected", "empty_resolved_patches_rejected"])
def test_upstream_failures_are_captured_without_success_claim(name):
    assert "error" in CASES[name]
    assert "applied" not in CASES[name]


@pytest.mark.parametrize("overrides", [
    {"main_block_count": True}, {"refiner_block_count": -1}, {"main_block_count": 1001},
    {"main_weights": [1.] * 49}, {"refiner_weights": [1.]},
    {"loader_source_sha256": "changed"}, {"other_weight": 10.1},
    {"model_strength": float("nan")}, {"other_weight": "1.0"}, {"model_strength": True},
])
def test_unverified_counts_partial_sequences_and_invalid_strengths_rejected(overrides):
    with pytest.raises(ValueError):
        candidate(**overrides)
