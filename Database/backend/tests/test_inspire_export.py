"""Independent observed-loader fixtures exercise slot safety, not image quality."""
import json
from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspire_export import ARCHITECTURE_LABELS, LOADER_SHA256, build_inspire_flux1_export

FIXTURE = json.loads((Path(__file__).parent / "fixtures/inspire_flux1_contract.json").read_text(encoding="utf-8"))
CASES = {case["name"]: case for case in FIXTURE["cases"]}
WEIGHTS = [i / 100 for i in range(1, 58)]
BASE = "diffusion_model.img_in.weight"


def export(keys=None, weights=None, **overrides):
    args = dict(
        base_model_code="FLX", block_layout="flux_transformer_57", base_weight=0.25,
        resolved_patch_keys=keys, coverage_complete=True,
        coverage_source="comfy_resolved_patch_keys", loader_source_sha256=LOADER_SHA256,
        base_patch_keys=[BASE] if keys and BASE in keys else [],
    )
    args.update(overrides)
    return build_inspire_flux1_export(WEIGHTS if weights is None else weights, **args)


def test_complete_export_matches_captured_actual_loader_exactly():
    case = CASES["full_flux"]
    result = export(case["patch_keys"])
    assert FIXTURE["source_sha256"] == LOADER_SHA256
    assert result["status"] == "ready"
    assert result["slot_labels"] == ARCHITECTURE_LABELS
    assert result["architecture_slot_values"] == result["slot_values"]
    assert result["loader_slot_count"] == 58
    assert [float(v) for v in result["numeric_csv"].split(",")] == [float(v) for v in case["input_csv"].split(",")]
    for index, key in enumerate(case["patch_keys"][:-1]):
        assert case["applied_weights"][key] == result["slot_values"][index + 1]
    assert case["applied_weights"][BASE] == 0.25


def test_sparse_export_uses_consumed_slots_not_architecture_positions():
    case = CASES["sparse_consumed_vector"]
    result = export(case["patch_keys"])
    assert result["status"] == "ready"
    assert result["slot_labels"][:4] == ["BASE", "DOUBLE 3", "DOUBLE 17", "SINGLE 1"]
    assert result["slot_values"] == [float(v) for v in case["input_csv"].split(",")]
    assert len(result["architecture_slot_values"]) == 58
    assert result["loader_slot_count"] == 12
    assert all(label.startswith("UNUSED PADDING") for label in result["slot_labels"][4:])
    # Independently captured actual-loader result proves prefixing BASE alone fails.
    bad = CASES["sparse_architecture_vector_misapplies"]
    assert bad["applied_weights"] != case["applied_weights"]
    assert case["applied_weights"][case["patch_keys"][0]] == result["slot_values"][1]


def test_boundary_equal_numeric_indices_cannot_express_different_values():
    case = CASES["equal_index_boundary"]
    result = export(case["patch_keys"])
    assert result["reason_code"] == "group_index_collision"
    assert result["numeric_csv"] is None
    assert case["applied_weights"][case["patch_keys"][0]] == case["applied_weights"][case["patch_keys"][1]] == 0.4
    weights = list(WEIGHTS)
    weights[3] = weights[22] = 0.4
    equal = export(case["patch_keys"], weights)
    assert equal["status"] == "ready"
    assert equal["slot_labels"][1] == "DOUBLE 3 + SINGLE 3"
    assert equal["slot_values"][1] == 0.4


def test_repeated_patches_and_comfy_tuple_keys_consume_only_one_block_slot():
    case = CASES["repeated_patch_in_same_block"]
    keys = [(case["patch_keys"][0], (0, 0, 1)), *case["patch_keys"][1:]]
    result = export(keys)
    assert result["status"] == "ready"
    assert result["slot_labels"][1:3] == ["DOUBLE 3", "UNUSED PADDING 2"]
    assert result["slot_values"][1] == 0.04
    assert len(set(case["applied_weights"][key] for key in case["patch_keys"][:-1])) == 1


@pytest.mark.parametrize("overrides,code", [
    ({"coverage_complete": False}, "patch_coverage_unknown"),
    ({"coverage_source": "safetensors_header"}, "patch_coverage_unknown"),
    ({"loader_source_sha256": "changed"}, "loader_version_unverified"),
    ({"base_model_code": "FL2"}, "unsupported_architecture"),
    ({"block_layout": "unet_57"}, "unsupported_architecture"),
])
def test_no_unproven_coverage_or_architecture_claim(overrides, code):
    result = export(CASES["full_flux"]["patch_keys"], **overrides)
    assert result["status"] == "blocked"
    assert result["reason_code"] == code
    assert result["numeric_csv"] is None


def test_empty_and_absent_coverage_are_different_failures():
    assert export(None)["reason_code"] == "patch_coverage_unknown"
    assert export([])["reason_code"] == "empty_patch_coverage"


@pytest.mark.parametrize("key", ["diffusion_model.transformer_blocks.0.weight", "diffusion_model.double_blocks.19.weight", "diffusion_model.single_blocks.38.weight", "diffusion_model.double_blocks.01.weight", "unknown.weight"])
def test_unknown_groups_or_indices_do_not_silently_fall_into_base(key):
    result = export([key])
    assert result["reason_code"] == "unknown_patch_key"
    assert result["numeric_csv"] is None
    captured = CASES["unknown_transformer_uses_base"]
    assert captured["applied_weights"]["diffusion_model.transformer_blocks.0.weight"] == 0.25


def test_declared_base_coverage_must_be_consistent():
    key = CASES["full_flux"]["patch_keys"][0]
    assert export([key], base_patch_keys=[BASE])["reason_code"] == "invalid_base_coverage"
    assert export([key], base_patch_keys=[key])["reason_code"] == "invalid_base_coverage"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "x", True, None])
def test_invalid_weights_never_generate_a_vector(bad):
    values = list(WEIGHTS)
    values[5] = bad
    assert export(CASES["full_flux"]["patch_keys"], values)["reason_code"] == "invalid_weights"


def test_no_truncation_repetition_or_exponent_notation():
    case = CASES["full_flux"]
    values = list(WEIGHTS)
    values[0] = 0.000000000013456789
    result = export(case["patch_keys"], values)
    fields = result["numeric_csv"].split(",")
    assert len(fields) == 58
    assert float(fields[1]) == values[0]
    assert all(re.fullmatch(r"-?\d+(\.\d+)?", item) for item in fields)
    assert CASES["scientific_notation_rejected"]["error"] == "invalid_block_vector"
    short = CASES["short_vector_reuses_last"]
    assert short["applied_weights"][case["patch_keys"][-2]] == 0.11
    assert export(case["patch_keys"], values[:-1])["reason_code"] == "invalid_weights"


def test_base_only_export_respects_minimum_vector_length():
    result = export([BASE])
    assert result["status"] == "ready"
    assert result["slot_values"] == [0.25] + [0.0] * 11
