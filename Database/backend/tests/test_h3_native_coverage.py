"""Independent captured constructor/alias fixture remains conditional evidence."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flux_header_coverage import CoverageError
from h3_native_coverage import CONTRACT, resolve_header, verify_sources


def pair(module, out_dim=21504, in_dim=5376, rank=4, ab=True):
    up, down = ("lora_B", "lora_A") if ab else ("lora_up", "lora_down")
    return {f"{module}.{up}.weight": {"shape": [out_dim, rank]}, f"{module}.{down}.weight": {"shape": [rank, in_dim]}}


def test_captured_native_target_has_complete_groups_and_actual_shapes():
    assert CONTRACT["main_count"] == 50 and CONTRACT["refiner_count"] == 2
    assert len(CONTRACT["target_shapes"]) == 266 and len(CONTRACT["aliases"]) == 798
    assert CONTRACT["parameters"]["attention_heads"] == 56
    assert CONTRACT["target_shapes"]["diffusion_model.blocks.49.attn.out_proj.weight"] == [5376, 7168]
    assert CONTRACT["target_shapes"]["diffusion_model.token_refiner.blocks.1.mlp.fc1.weight"] == [28672, 5376]
    assert CONTRACT["target_shapes"]["diffusion_model.final_layer.video_out.weight"] == [96, 5376]
    assert "diffusion_model.rope.inv_freq" in CONTRACT["non_matrix_targets"]


@pytest.mark.parametrize("module", ["blocks.0.attn.qkv_proj", "diffusion_model.blocks.0.attn.qkv_proj", "lora_unet_blocks_0_attn_qkv_proj"])
def test_only_actual_captured_aliases_resolve_to_the_same_native_target(module):
    result = resolve_header(pair(module))
    assert result["all_tensors_mapped"] and result["status"] == "mapped_candidate"
    assert result["mappings"][0]["target"] == "diffusion_model.blocks.0.attn.qkv_proj.weight"
    assert result["main_count"] == 50  # Target source, never sparse highest-index inference.
    assert result["source_target_conditional"]
    for flag in ("checkpoint_verified", "architecture_verified", "patch_application_verified", "measurements_available", "export_verified"):
        assert result[flag] is False


def test_refiner_and_other_are_resolved_target_groups_and_alpha_is_not_measured():
    tensors = {**pair("token_refiner.blocks.1.mlp.fc1", 28672, 5376, ab=False),
               **pair("final_layer.audio_out", 32, 5376), "final_layer.audio_out.alpha": {"shape": []}}
    result = resolve_header(tensors)
    assert result["all_tensors_mapped"] and result["accounted_tensor_count"] == 5
    assert [(entry["group"], entry["index"]) for entry in result["mappings"]] == [("refiner", 1), ("other", None)]
    assert result["mappings"][1]["alpha_present"] and "alpha_value" not in result["mappings"][1]


@pytest.mark.parametrize("tensors,code", [
    (pair("transformer.blocks.0.attn.qkv_proj"), "unmapped_target"),
    (pair("blocks.50.attn.qkv_proj"), "unmapped_target"),
    (pair("blocks.0.attn.to_gate_compress"), "unmapped_target"),
    (pair("blocks.0.attn.qkv_proj", 16128, 5376), "target_shape_mismatch"),
    (pair("blocks.0.adaln_proj.linear", 96768, 64), "target_shape_mismatch"),
    ({**pair("blocks.0.attn.qkv_proj"), **pair("diffusion_model.blocks.0.attn.qkv_proj")}, "ambiguous_target"),
    ({**pair("blocks.0.attn.qkv_proj"), "blocks.0.attn.qkv_proj.lora_B.default.weight": {"shape": [21504, 4]}}, "ambiguous_target"),
    ({"blocks.0.attn.qkv_proj.lokr_w1": {"shape": [2, 2]}}, "unsupported_tensor"),
    ({"blocks.0.attn.qkv_proj.lora_A.weight": {"shape": [4, 5376]}}, "incomplete_pair"),
    ({"blocks.0.attn.qkv_proj.lora_A.weight": {"shape": [4, 5376]}, "blocks.0.attn.qkv_proj.lora_up.weight": {"shape": [21504, 4]}}, "mixed_pair_format"),
    (pair("blocks.0.attn.qkv_proj", rank=True), "invalid_pair_shape"),
    ({**pair("blocks.0.attn.qkv_proj"), "blocks.0.attn.qkv_proj.alpha": {"shape": [2]}}, "invalid_alpha_shape"),
])
def test_variants_and_unknowns_cannot_silently_become_native_defaults(tensors, code):
    result = resolve_header(tensors)
    assert not result["all_tensors_mapped"] and result["status"] == "needs_review"
    assert result["issue_counts"][code] >= 1 and result["mappings"] == []


def test_empty_or_extra_unknown_tensor_cannot_approve_a_candidate():
    assert not resolve_header({})["all_tensors_mapped"]
    result = resolve_header({**pair("blocks.0.attn.qkv_proj"), "unrecognized": {"shape": [4]}})
    assert len(result["mappings"]) == 1 and not result["all_tensors_mapped"]
    assert result["accounted_tensor_count"] == 2 and result["tensor_count"] == 3


def test_source_pins_fail_closed_without_importing_or_installing_comfy(tmp_path):
    with pytest.raises(CoverageError, match="unavailable"): verify_sources(tmp_path)
    for relative in CONTRACT["source_sha256"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("changed source")
    with pytest.raises(CoverageError, match="changed"): verify_sources(tmp_path)


def test_capture_rejects_drift_before_executing_any_source(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    from capture_h3_native_target import PINNED, capture
    assert PINNED == CONTRACT["source_sha256"]
    for relative in PINNED:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raise RuntimeError('must never execute this source')")
    with pytest.raises(ValueError, match="Pinned source changed"): capture(tmp_path)
