"""Optional Torch CPU suite. Run explicitly with a Torch-capable analysis Python."""
from dataclasses import replace
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import pytest

torch = pytest.importorskip("torch", reason="Optional effective-update analysis requires a Torch CPU runtime")
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import effective_lora_metrics as metrics


def update(b, a, alpha=None):
    b = torch.as_tensor(b, dtype=torch.float64, device="cpu")
    a = torch.as_tensor(a, dtype=torch.float64, device="cpu")
    return metrics.Update(metrics.tensor_factor(b), metrics.tensor_factor(a), alpha)


def dense(value):
    b = value.up.reader(0, 0, value.up.shape[0])
    a = value.down.reader(1, 0, value.down.shape[1])
    scale = 1.0 if value.alpha is None else value.alpha / a.shape[0]
    return scale * (b @ a)


def inner(left, right=None, budget=None):
    return metrics.update_inner_product(left, left if right is None else right,
                                        work=metrics.Work(budget or metrics.AnalysisBudget()))


@pytest.mark.parametrize("alpha", [None, 0.0, 4.0, -3.0])
def test_gram_norm_matches_dense_effective_update(alpha):
    value = update([[1, -2], [3, 0.4], [-0.5, 2]], [[1, 2, 3, -4], [4, -1, 2, 0.3]], alpha)
    expected = float(torch.square(dense(value)).sum())
    assert inner(value) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_absent_alpha_means_scale_one_not_one_over_rank():
    value = update([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    with_alpha = metrics.Update(value.up, value.down, 1.0)
    assert metrics.alpha_scale(None, 2) == 1.0
    assert inner(with_alpha) == pytest.approx(inner(value) / 4)


def test_general_invertible_gauge_change_preserves_update_norm_and_alignment():
    b = torch.tensor([[1, -2], [3, 0.4], [-0.5, 2]], dtype=torch.float64)
    a = torch.tensor([[1, 2, 3], [4, -1, 2]], dtype=torch.float64)
    r = torch.tensor([[2, 0.75], [-0.3, 1.5]], dtype=torch.float64)
    original = update(b, a, 2.0)
    transformed_b, transformed_a = b @ r, torch.linalg.solve(r, a)
    transformed = update(transformed_b, transformed_a, 2.0)
    assert not math.isclose(float(b.norm() + a.norm()), float(transformed_b.norm() + transformed_a.norm()))
    assert torch.allclose(dense(original), dense(transformed), atol=1e-12)
    assert inner(original) == pytest.approx(inner(transformed), rel=1e-12)
    assert metrics.signed_cosine(inner(original, transformed), inner(original), inner(transformed)) == pytest.approx(1.0)


def test_signed_inner_product_with_different_ranks_matches_dense_oracle():
    left = update([[1, 2], [-3, 4]], [[5, -6, 7], [8, 9, -10]], -2)
    right = update([[2], [3]], [[-4, 5, 6]], 0.25)
    expected = float((dense(left) * dense(right)).sum())
    assert inner(left, right) == pytest.approx(expected, rel=1e-12)
    assert inner(right, left) == pytest.approx(expected, rel=1e-12)


def test_exact_opposites_have_negative_alignment_and_zero_norm_has_no_angle():
    left = update([[1], [2]], [[3, 4]], 1)
    right = update([[-1], [-2]], [[3, 4]], 1)
    zero = update([[0], [0]], [[3, 4]])
    assert metrics.signed_cosine(inner(left, right), inner(left), inner(right)) == pytest.approx(-1.0)
    assert inner(zero) == 0
    assert metrics.signed_cosine(0, 0, inner(left)) is None


def test_negative_alpha_alone_reverses_update_direction():
    positive = update([[1, 2], [-3, 4]], [[5, -6], [7, 8]], 2.0)
    negative = metrics.Update(positive.up, positive.down, -2.0)
    expected_squared_norm = float(torch.square(dense(positive)).sum())
    assert expected_squared_norm > 0
    assert inner(positive) == pytest.approx(expected_squared_norm)
    assert inner(negative) == pytest.approx(expected_squared_norm)
    assert inner(positive, negative) == pytest.approx(-expected_squared_norm)
    assert metrics.signed_cosine(inner(positive, negative), inner(positive), inner(negative)) == pytest.approx(-1.0)


def test_chunk_boundaries_do_not_change_results_and_readers_stay_bounded():
    tensor = torch.arange(21, dtype=torch.float64).reshape(7, 3)
    reads = []

    def read(axis, start, end):
        result = tensor[start:end, :] if axis == 0 else tensor[:, start:end]
        reads.append(result.numel())
        return result

    factor = metrics.Factor(tensor.shape, 8, read)
    budget = replace(metrics.AnalysisBudget(), chunk_elements=6)
    work = metrics.Work(budget)
    assert torch.allclose(metrics.cross_gram(factor, factor, axis=0, work=work), tensor.T @ tensor)
    assert reads == [6, 6, 6, 3]
    assert work.bytes_read == tensor.numel() * 8
    assert work.multiply_adds == 7 * 3 * 3


@pytest.mark.parametrize("budget,code", [
    (replace(metrics.AnalysisBudget(), max_tensor_bytes_read=1), "read_budget_exceeded"),
    (replace(metrics.AnalysisBudget(), max_multiply_adds=1), "work_budget_exceeded"),
    (replace(metrics.AnalysisBudget(), max_rank=1), "rank_budget_exceeded"),
    (replace(metrics.AnalysisBudget(), chunk_elements=1), "chunk_budget_exceeded"),
])
def test_budget_limits_stop_before_unbounded_work(budget, code):
    value = update([[1, 2]], [[3], [4]])
    with pytest.raises(metrics.AnalysisError) as error:
        inner(value, budget=budget)
    assert error.value.code == code


def test_cancellation_and_deadline_are_checked_between_operations():
    value = update([[1]], [[2]])
    with pytest.raises(metrics.AnalysisError) as error:
        metrics.update_inner_product(value, value, work=metrics.Work(metrics.AnalysisBudget(), lambda: True))
    assert error.value.code == "cancelled"
    work = metrics.Work(replace(metrics.AnalysisBudget(), max_seconds=1))
    work.started = time.monotonic() - 2
    with pytest.raises(metrics.AnalysisError) as error:
        metrics.update_inner_product(value, value, work=work)
    assert error.value.code == "time_budget_exceeded"


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_factors_are_rejected_even_when_alpha_is_zero(number):
    with pytest.raises(metrics.AnalysisError) as error:
        inner(update([[number]], [[1]], 0.0))
    assert error.value.code == "nonfinite_tensor"


def test_nonfinite_alpha_shape_mismatch_and_overflow_are_explicit():
    with pytest.raises(metrics.AnalysisError, match="Alpha"):
        update([[1]], [[2]], float("nan"))
    with pytest.raises(metrics.AnalysisError, match="same rank"):
        update([[1, 2]], [[3]])
    with pytest.raises(metrics.AnalysisError) as error:
        inner(update([[1e200]], [[1e200]]))
    assert error.value.code == "numeric_overflow"


def make_file(path, *, block=0, sign=1.0, dtype=None, alpha=None):
    dtype = dtype or torch.float32
    prefix = f"lora_unet_double_blocks_{block}_img_attn_proj"
    tensors = {prefix + ".lora_up.weight": torch.full((3072, 1), sign, dtype=dtype),
               prefix + ".lora_down.weight": torch.ones((1, 3072), dtype=dtype)}
    if alpha is not None:
        tensors[prefix + ".alpha"] = torch.tensor(alpha, dtype=torch.float32)
    save_file(tensors, str(path))
    return path


@pytest.fixture
def source_contract(monkeypatch):
    monkeypatch.setattr(metrics, "verify_local_sources", lambda: {"fixture.py": "0" * 64})


def test_streamed_pair_receipt_matches_known_dense_update_without_allocating_it(tmp_path, source_contract):
    left = make_file(tmp_path / "positive.safetensors", dtype=torch.bfloat16)
    right = make_file(tmp_path / "negative.safetensors", sign=-1, dtype=torch.float16)
    before = [(path.stat().st_size, path.stat().st_mtime_ns) for path in (left, right)]
    result = metrics.analyse_files([left, right])
    assert result["status"] == "complete"
    assert len(result["target_contract_sha256"]) == 64
    assert result["sources"][0]["total_squared_norm"] == 3072**2
    assert result["sources"][1]["block_squared_norms"][1] == 3072**2
    assert result["pairs"][0]["total_inner_product"] == -(3072**2)
    assert result["pairs"][0]["total_signed_cosine"] == -1
    assert result["pairs"][0]["block_signed_cosines"][0] is None
    assert result["tensor_payload_read"] is True
    assert result["semantic_conflict_verified"] is False
    assert result["image_quality_verified"] is False
    assert result["recommendation"] is None
    assert before == [(path.stat().st_size, path.stat().st_mtime_ns) for path in (left, right)]


def test_disjoint_modules_have_zero_inner_product_not_a_quality_claim(tmp_path, source_contract):
    result = metrics.analyse_files([make_file(tmp_path / "a.safetensors", block=0), make_file(tmp_path / "b.safetensors", block=1)])
    pair = result["pairs"][0]
    assert pair["shared_module_count"] == 0
    assert pair["total_inner_product"] == 0
    assert pair["total_signed_cosine"] == 0
    assert all(value is None for value in pair["block_signed_cosines"])


def test_changed_file_during_analysis_invalidates_all_metrics(tmp_path, source_contract):
    path = make_file(tmp_path / "changing.safetensors")
    changed = False

    def change(_progress):
        nonlocal changed
        if not changed:
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10000000))
            changed = True

    with pytest.raises(metrics.AnalysisError) as error:
        metrics.analyse_files([path], progress=change)
    assert error.value.code == "file_changed"


def test_source_deleted_between_header_validation_and_tensor_open_has_explicit_failure(tmp_path, source_contract, monkeypatch):
    path = make_file(tmp_path / "removed.safetensors")
    removed = False

    def runtime_after_header():
        nonlocal removed
        if not removed:
            path.unlink()
            removed = True
        return torch

    monkeypatch.setattr(metrics, "_torch", runtime_after_header)
    with pytest.raises(metrics.AnalysisError) as error:
        metrics.analyse_files([path])
    assert error.value.code == "tensor_read_failed"


def test_unsupported_adapter_never_produces_partial_measurements(tmp_path, source_contract):
    path = tmp_path / "unsupported.safetensors"
    save_file({"lora_unet_double_blocks_0_img_attn_proj.dora_scale": torch.ones(3072)}, str(path))
    with pytest.raises(metrics.AnalysisError) as error:
        metrics.analyse_files([path])
    assert error.value.code == "unsupported_adapter"


def test_duplicate_inputs_and_count_budget_are_explicit(tmp_path, source_contract):
    path = make_file(tmp_path / "one.safetensors")
    with pytest.raises(metrics.AnalysisError) as error:
        metrics.analyse_files([path, path])
    assert error.value.code == "duplicate_source"
    with pytest.raises(metrics.AnalysisError) as error:
        metrics.analyse_files([])
    assert error.value.code == "lora_budget_exceeded"


def test_cli_saves_failure_receipt_without_partial_metrics(tmp_path):
    script = Path(__file__).resolve().parents[3] / "tools/analyse_effective_lora.py"
    # tests lives Database/backend/tests; repository root is parents[3].
    spec = importlib.util.spec_from_file_location("effective_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    output = tmp_path / "failed.json"
    assert cli.main([str(tmp_path / "missing.safetensors"), "--output", str(output), "--max-seconds", "-1"]) == 2
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "failed"
    assert receipt["reason_code"] == "invalid_budget"
    assert receipt["metrics"] is None
