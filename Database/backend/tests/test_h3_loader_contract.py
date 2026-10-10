"""Independently captured H3 loader observations, not a render acceptance test."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h3_loader_contract import LOADER_SHA256, SLOT_LABELS, candidate_vector

FIXTURE = json.loads((Path(__file__).parent / "fixtures/h3_selective_contract.json").read_text(encoding="utf-8"))
CASES = {case["name"]: case for case in FIXTURE["cases"]}


def vector(weights=None, other=1., sha=LOADER_SHA256):
    return candidate_vector([1.] * 50 if weights is None else weights,
                            other_weight=other, loader_source_sha256=sha)


def assert_update(record, multiplier, outer):
    # Independently multiply the capture's fixed A/B matrices; rank=2, alpha=3.
    expected = [[value * 1.5 * multiplier * outer for value in row]
                for row in [[5., 8.], [8., 10.]]]
    for actual, desired in zip(record["effective_update"], expected):
        assert actual == pytest.approx(desired, rel=1e-14, abs=1e-14)
    if record["retained"]:
        assert record["down_unchanged"] is True
        assert record["alpha_unchanged"] is True


def test_exact_full_sequence_and_other_last_match_upstream_capture():
    case = CASES["full_order"]
    weights = [i / 100 for i in range(50)]
    result = vector(weights, other=0.37)
    assert FIXTURE["source_sha256"] == LOADER_SHA256
    assert len(result["slot_values"]) == 51
    assert result["slot_labels"] == list(SLOT_LABELS)
    assert result["slot_labels"][0] == "MAIN 0"
    assert result["slot_labels"][-1] == "OTHER"
    assert result["candidate_numeric_csv"] == case["input_csv"]
    for i, record in enumerate(case["modules"]):
        assert_update(record, weights[i], case["outer_strength"])
        assert record["retained"] is (i != 0)


def test_sparse_coverage_keeps_architecture_indices_sign_and_outer_strength():
    case = CASES["sparse_signed_and_other"]
    parsed = [float(value) for value in case["input_csv"].split(",")]
    result = vector(parsed[:50], other=parsed[50])
    assert len(result["slot_values"]) == 51  # Never compress to four observed modules.
    for record, multiplier in zip(case["modules"], [-0.625, 0.1256789, -0.25, -0.25]):
        assert_update(record, multiplier, -0.8)
    assert case["applied_model_strength"] == case["applied_clip_strength"] == -0.8
    assert "0.1256789" in result["candidate_numeric_csv"]
    assert "0.1256789" not in case["returned_csv"]  # Rounded output is not persistence authority.


def test_kohya_scaling_is_linear_signed_and_keeps_alpha():
    case = CASES["kohya_signed"]
    assert_update(case["modules"][0], -0.5, 0.9)


def test_zero_omits_whole_pair_and_other_without_calling_comfy():
    case = CASES["zero_removes_main_and_other"]
    assert case["applied_model_strength"] is None
    assert all(not record["retained"] for record in case["modules"])
    for record in case["modules"]:
        assert_update(record, 0., 1.)


def test_short_and_invalid_loader_inputs_are_unsafe_defaults():
    assert_update(CASES["short_50_defaults_other"]["modules"][0], 1., 1.)
    assert_update(CASES["malformed_falls_back_to_preset"]["modules"][0], 1., 1.)
    with pytest.raises(ValueError):
        vector([0.4] * 49)
    with pytest.raises(ValueError):
        vector(["bad,input"] * 50)


def test_unrecognized_target_spellings_require_later_coverage_resolution():
    # Upstream silently drops main block 50 and treats bare blocks as OTHER.
    assert CASES["out_of_range_main_is_dropped"]["modules"][0]["retained"] is False
    assert_update(CASES["bare_blocks_are_other"]["modules"][0], 0.5, 1.)
    assert vector()["export_verified"] is False
    assert vector()["status"] == "characterized_candidate"


def test_required_clip_shared_strength_and_unsupported_formats_are_explicit():
    assert {"clip", "model", "strength", "other_weights", "other_weights_str"}.issubset(FIXTURE["required_inputs"])
    assert all(FIXTURE["unscaled_suffixes"].values())
    # This does not prove DoRA/LoKr/LoHa update equivalence; no support is granted.
    assert vector()["export_verified"] is False


@pytest.mark.parametrize("bad", [True, "1.0", None, float("nan"), float("inf"), -float("inf")])
def test_no_ambiguous_or_nonfinite_weights(bad):
    weights = [1.] * 50
    weights[7] = bad
    with pytest.raises(ValueError):
        vector(weights)
    with pytest.raises(ValueError):
        vector(other=bad)


@pytest.mark.parametrize("count", [0, 4, 49, 51, 58])
def test_no_sparse_or_flux_layout_substitution(count):
    with pytest.raises(ValueError):
        vector([1.] * count)


def test_reject_changed_loader_source():
    with pytest.raises(ValueError, match="characterized version"):
        vector(sha="unverified")
