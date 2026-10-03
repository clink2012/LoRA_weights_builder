import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flux_header_coverage as coverage
import lora_api_server as api

PREFIX = "lora_unet_double_blocks_0_img_attn_proj"


def pair(prefix=PREFIX, up=(3072, 4), down=(4, 3072), alpha=()):
    result = {prefix + ".lora_up.weight": {"dtype": "F16", "shape": list(up)},
              prefix + ".lora_down.weight": {"dtype": "F16", "shape": list(down)}}
    if alpha is not None:
        result[prefix + ".alpha"] = {"dtype": "F32", "shape": list(alpha)}
    return result


def write_header(path, tensors, *, mutate=None):
    cursor = 0
    for tensor in tensors.values():
        size = 4 if tensor["dtype"] == "F32" else 2
        for dimension in tensor["shape"]:
            size *= dimension
        tensor["data_offsets"] = [cursor, cursor + size]
        cursor += size
    if mutate:
        mutate(tensors)
    data = json.dumps(tensors).encode()
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(data)))
        stream.write(data)
        stream.truncate(8 + len(data) + cursor)
    return 8 + len(data)


def test_contract_shapes_are_observed_from_real_symbolic_constructors():
    target = coverage.CONTRACT
    assert target["parameters"]["depth"] == 19
    assert target["parameters"]["depth_single_blocks"] == 38
    assert target["target_shapes"]["diffusion_model.double_blocks.0.img_attn.qkv.weight"] == [9216, 3072]
    assert target["target_shapes"]["diffusion_model.single_blocks.37.linear1.weight"] == [21504, 3072]
    assert target["target_shapes"]["diffusion_model.single_blocks.37.linear2.weight"] == [3072, 15360]
    assert target["target_shapes"]["diffusion_model.img_in.weight"] == [3072, 64]
    assert target["aliases"][PREFIX] == "diffusion_model.double_blocks.0.img_attn.proj.weight"
    observed = target["native_adapter_observations"]
    assert observed[0]["loaded_keys"] == [PREFIX + ".lora_down.weight", PREFIX + ".lora_up.weight"]
    assert observed[1]["error"] == "missing_pair_key"


def test_native_pair_header_inspection_reads_no_payload(tmp_path, monkeypatch):
    path = tmp_path / "native.safetensors"
    header_bytes = write_header(path, pair())
    monkeypatch.setattr(coverage, "verify_local_sources", lambda: {"fixture": "pinned"})
    result = coverage.inspect_native_flux_file(path)
    assert result["block_presence_baseline"] == [1.0] + [0.0] * 56
    assert result["coverage_source"] == "statically_resolved_against_pinned_target"
    assert result["checkpoint_verified"] is False
    assert result["file_identity"]["header_bytes_read"] == header_bytes
    assert result["file_identity"]["tensor_payload_read"] is False
    assert result["file_identity"]["tensor_contents_verified"] is False


def test_complete_and_sparse_blocks_have_canonical_presence_vectors():
    tensors = {}
    for index in range(19):
        tensors.update(pair(f"lora_unet_double_blocks_{index}_img_attn_proj", alpha=None))
    for index in range(38):
        tensors.update(pair(f"diffusion_model.single_blocks.{index}.linear2", down=(4, 15360)))
    result = coverage.resolve_native_header(tensors)
    assert result["block_presence_baseline"] == [1.0] * 57
    assert len(result["resolved_patch_keys"]) == 57
    sparse = coverage.resolve_native_header(pair("lora_unet_single_blocks_37_linear1", up=(21504, 4)))
    assert sparse["block_presence_baseline"] == [0.0] * 56 + [1.0]


def test_base_pair_is_explicitly_classified():
    result = coverage.resolve_native_header(pair("lora_unet_img_in", down=(4, 64)))
    assert result["base_patch_keys"] == ["diffusion_model.img_in.weight"]
    assert result["block_presence_baseline"] == [0.0] * 57


@pytest.mark.parametrize("tensors,code", [
    ({PREFIX + ".lora_up.weight": {"shape": [3072, 4]}}, "incomplete_pair"),
    (pair(up=(2048, 4)), "target_shape_mismatch"),
    (pair(down=(8, 3072)), "target_shape_mismatch"),
    (pair(down=(4, 3072, 1, 1)), "target_shape_mismatch"),
    (pair(alpha=(2,)), "invalid_alpha_shape"),
    (pair("lora_unet_double_blocks_19_img_attn_proj"), "unknown_target_module"),
    (pair("lora_unet_transformer_blocks_0_proj"), "unknown_target_module"),
    ({"anything.lora_mid.weight": {"shape": [4, 4]}}, "unsupported_adapter"),
])
def test_unknown_mismatched_or_incomplete_formats_are_blocked(tensors, code):
    with pytest.raises(coverage.CoverageError) as error:
        coverage.resolve_native_header(tensors)
    assert error.value.code == code


def test_two_aliases_cannot_silently_overwrite_a_target():
    tensors = pair()
    tensors.update(pair("diffusion_model.double_blocks.0.img_attn.proj"))
    with pytest.raises(coverage.CoverageError, match="Multiple LoRA aliases"):
        coverage.resolve_native_header(tensors)


@pytest.mark.parametrize("mutate", [
    lambda t: t[PREFIX + ".alpha"].update(data_offsets=[0, 4]),
    lambda t: t[PREFIX + ".alpha"].update(dtype="I64"),
    lambda t: t[PREFIX + ".alpha"].update(shape=[True]),
    lambda t: t[PREFIX + ".alpha"].update(shape=[-1]),
])
def test_invalid_header_offsets_shapes_and_types(tmp_path, mutate):
    path = tmp_path / "bad.safetensors"
    write_header(path, pair(), mutate=mutate)
    with pytest.raises(coverage.CoverageError):
        coverage.read_header(path)


def test_oversized_or_duplicate_header_never_reads_tensor_data(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", coverage.MAX_HEADER_BYTES + 1))
    with pytest.raises(coverage.CoverageError):
        coverage.read_header(path)
    duplicate = b'{"x":{},"x":{}}'
    path.write_bytes(struct.pack("<Q", len(duplicate)) + duplicate)
    with pytest.raises(coverage.CoverageError, match="Duplicate"):
        coverage.read_header(path)


def test_missing_file_and_unverified_runtime_are_explicit(tmp_path):
    with pytest.raises(coverage.CoverageError) as error:
        coverage.read_header(tmp_path / "missing.safetensors")
    assert error.value.code == "file_missing"
    with pytest.raises(coverage.CoverageError) as error:
        coverage.verify_local_sources(tmp_path)
    assert error.value.code == "runtime_source_unavailable"


def test_atomic_path_replacement_is_detected_even_when_open_handle_is_unchanged(tmp_path, monkeypatch):
    path = tmp_path / "replaced.safetensors"
    write_header(path, pair())
    original = Path.stat

    def replacement_stat(self, *args, **kwargs):
        actual = original(self, *args, **kwargs)
        if self == path:
            return SimpleNamespace(st_dev=actual.st_dev, st_ino=actual.st_ino + 1,
                                   st_size=actual.st_size, st_mtime_ns=actual.st_mtime_ns)
        return actual

    monkeypatch.setattr(Path, "stat", replacement_stat)
    with pytest.raises(coverage.CoverageError) as error:
        coverage.read_header(path)
    assert error.value.code == "file_changed"


def test_prepare_route_ignores_stale_cached_layout_and_preserves_db(tmp_path, monkeypatch):
    path = tmp_path / "native.safetensors"
    write_header(path, pair())
    db = tmp_path / "catalog.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE lora(stable_id TEXT, filename TEXT, file_path TEXT, base_model_code TEXT, block_layout TEXT)")
    conn.execute("INSERT INTO lora VALUES('real','native.safetensors',?,'FLX','flux_fallback_16')", (str(path),))
    conn.execute("INSERT INTO lora VALUES('old','gone.safetensors',?,'FLX','unet_57')", (str(tmp_path / "gone.safetensors"),))
    conn.commit()
    conn.close()
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    monkeypatch.setattr(api, "DB_PATH", db)
    monkeypatch.setattr(coverage, "verify_local_sources", lambda: {"fixture": "pinned"})
    # Deliberately no lifespan: this route itself must never invoke legacy backfills.
    client = TestClient(api.app)
    result = client.post("/api/lora/prepare-blocks", json={"stable_ids": ["old", "real"], "target_contract_id": coverage.CONTRACT_ID})
    assert result.status_code == 200
    body = result.json()
    assert body["compatible"] is False
    assert body["requested_loras"] == ["old", "real"]
    assert body["included_loras"] == ["real"]
    assert body["excluded_loras"][0]["reason_code"] == "file_missing"
    node = body["node_payloads"][0]
    assert node["block_weights"] == [1.0] + [0.0] * 56
    assert node["loader_export"]["status"] == "ready"
    assert node["loader_export"]["recommendation_basis"] == "structural_baseline_unvalidated"
    assert node["loader_export"]["checkpoint_verified"] is False
    assert len(node["loader_export"]["architecture_slot_values"]) == 58
    single = client.post("/api/lora/prepare-blocks", json={"stable_ids": ["real"], "target_contract_id": coverage.CONTRACT_ID})
    assert single.json()["compatible"] is True
    assert client.post("/api/lora/prepare-blocks", json={"stable_ids": ["real"], "target_contract_id": "flux2"}).status_code == 422
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
