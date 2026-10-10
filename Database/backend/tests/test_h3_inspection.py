"""Observed coverage remains separate from target verification and weights."""
import json
import math
from pathlib import Path
import sqlite3
import struct
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catalogue_refresh import CatalogueService
from h3_adapter_observations import inspect_file, module_group, observe_header
import h3_inspection_router as router_module


def pair(module, fmt="ab", rank=2):
    down, up = ("lora_A", "lora_B") if fmt == "ab" else ("lora_down", "lora_up")
    return {f"{module}.{down}.weight": {"shape": [rank, 4]},
            f"{module}.{up}.weight": {"shape": [6, rank]}}


def write_header(path, tensors):
    header, offset = {}, 0
    for key, tensor in tensors.items():
        size = math.prod(tensor["shape"]) * 4
        header[key] = {**tensor, "dtype": "F32", "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(raw)))
        stream.write(raw)
        stream.truncate(8 + len(raw) + offset)


@pytest.mark.parametrize("module,expected", [
    ("blocks.49.attn.qkv_proj", ("main", 49)),
    ("transformer.blocks.100.mlp.fc1", ("main", 100)),
    ("diffusion_model.token_refiner.blocks.1.mlp.fc2", ("refiner", 1)),
    ("lora_unet_token_refiner_blocks_0_attn_qkv_proj", ("refiner", 0)),
    ("lora_unet_blocks_2_mlp_fc1", ("main", 2)),
    ("final_layer.video_out", ("other", None)),
    ("lora_unet_time_embedder_proj_in", ("other", None)),
    ("surprise.new_layer", ("unknown", None)),
    ("blocks.01.mlp.fc1", ("unknown", None)),
    ("blocks.100001.mlp.fc1", ("unknown", None)),
])
def test_group_observations_are_explicit_not_total_depth(module, expected):
    assert module_group(module) == expected


def test_counts_ranks_and_accounting_do_not_claim_weights_or_depth():
    result = observe_header({**pair("blocks.49.attn.qkv_proj"),
                             **pair("blocks.49.mlp.fc1", rank=4),
                             **pair("token_refiner.blocks.1.mlp.fc2", "up_down"),
                             **pair("video_patch_proj")})
    assert result["slots"] == [
        {"group": "main", "index": 49, "label": "MAIN 49", "pair_count": 2, "ranks": [2, 4]},
        {"group": "refiner", "index": 1, "label": "REFINER 1", "pair_count": 1, "ranks": [2]},
        {"group": "other", "index": None, "label": "OTHER", "pair_count": 1, "ranks": [2]}]
    assert result["all_tensors_accounted"] and result["pair_count"] == 4
    assert result["accounted_tensor_count"] == result["tensor_count"] == 8
    assert result["total_model_depth"] is None
    for claim in ("architecture_verified", "target_mapping_verified", "export_verified", "tensor_contents_verified", "measurements_available"):
        assert result[claim] is False
    assert "values" not in result and "multipliers" not in result


@pytest.mark.parametrize("tensors,code", [
    ({"unrecognized": {"shape": [2]}}, "unsupported_tensor"),
    (pair("new.module"), "unknown_module"),
    ({"blocks.0.mlp.fc1.lora_A.weight": {"shape": [2, 4]}}, "incomplete_pair"),
    ({"blocks.0.mlp.fc1.lora_A.weight": {"shape": [2, 4]}, "blocks.0.mlp.fc1.lora_up.weight": {"shape": [6, 2]}}, "mixed_pair_format"),
    ({**pair("blocks.0.mlp.fc1"), "blocks.0.mlp.fc1.alpha": {"shape": [2]}}, "invalid_alpha_shape"),
    ({**pair("blocks.0.mlp.fc1"), "blocks.0.mlp.fc1.lora_B.weight": {"shape": [6, 3]}}, "invalid_pair_shape"),
    ({**pair("blocks.0.mlp.fc1"), "blocks.0.mlp.fc1.lora_A.default.weight": {"shape": [2, 4]}}, "duplicate_factor"),
])
def test_bad_pairs_never_contribute_to_graph_or_implicit_other(tensors, code):
    result = observe_header(tensors)
    assert result["status"] == "needs_review"
    assert result["pair_count"] == result["accounted_tensor_count"] == 0
    assert not result["slots"] and not result["all_tensors_accounted"]
    assert code in {issue["code"] for issue in result["issues"]}


def test_read_is_header_only_and_source_unchanged(tmp_path):
    path = tmp_path / "adapter.safetensors"
    write_header(path, pair("blocks.0.mlp.fc1"))
    before = path.read_bytes()
    result = inspect_file(path)
    assert not result["file_identity"]["tensor_payload_read"]
    assert result["file_identity"]["header_bytes_read"] == 8 + struct.unpack("<Q", before[:8])[0]
    assert path.read_bytes() == before


@pytest.fixture
def api(tmp_path):
    root = tmp_path / "library"
    root.mkdir()
    database = tmp_path / "catalogue.db"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE lora (stable_id TEXT PRIMARY KEY, filename TEXT, file_path TEXT, base_model_code TEXT)")
        conn.execute("CREATE TABLE personal_history (saved TEXT)")
        conn.execute("INSERT INTO personal_history VALUES ('preserve me')")
    service = CatalogueService(database, root)
    app = FastAPI()
    app.include_router(router_module.create_h3_inspection_router(service))
    def add(path, sid="H3-test", family="MH3"):
        with sqlite3.connect(database) as conn:
            conn.execute("INSERT INTO lora VALUES (?,?,?,?)", (sid, path.name, str(path), family))
    return TestClient(app), root, database, add


def test_api_read_only_fresh_each_time_and_stable_id_only(api):
    client, root, database, add = api
    path = root / "adapter.safetensors"
    write_header(path, pair("blocks.0.mlp.fc1")); add(path)
    db_before = database.read_bytes()
    first = client.get("/api/h3-inspection/H3-test").json()
    assert first["slots"][0]["index"] == 0
    write_header(path, pair("token_refiner.blocks.1.mlp.fc1"))
    second = client.get("/api/h3-inspection/H3-test").json()
    assert second["slots"][0]["group"] == "refiner"
    assert first["file_identity"]["header_sha256"] != second["file_identity"]["header_sha256"]
    assert database.read_bytes() == db_before
    assert client.get("/api/h3-inspection/unknown").status_code == 404
    add(path, sid="FLX-test", family="FLX")
    assert client.get("/api/h3-inspection/FLX-test").status_code == 422
    assert client.post("/api/h3-inspection/H3-test", json={"path": str(path)}).status_code == 405


@pytest.mark.parametrize("kind", ["missing", "outside", "invalid"])
def test_unavailable_paths_are_actionable_without_source_or_db_changes(api, kind):
    client, root, database, add = api
    path = (root.parent if kind == "outside" else root) / "adapter.safetensors"
    if kind != "missing": path.write_bytes(b"invalid header")
    add(path); before = database.read_bytes()
    result = client.get("/api/h3-inspection/H3-test").json()
    assert result["status"] == "unavailable" and result["slots"] == []
    assert result["export_verified"] is False
    assert database.read_bytes() == before


def test_changed_during_inspection_never_returns_stale_coverage(api, monkeypatch):
    client, root, _, add = api
    path = root / "adapter.safetensors"
    write_header(path, pair("blocks.0.mlp.fc1")); add(path)
    original = router_module.inspect_file
    def change(path):
        result = original(path)
        write_header(path, pair("blocks.1.mlp.fc1", rank=3))
        return result
    monkeypatch.setattr(router_module, "inspect_file", change)
    result = client.get("/api/h3-inspection/H3-test").json()
    assert result["status"] == "unavailable" and result["reason_code"] == "file_changed"


def test_linked_file_is_not_followed(api):
    client, root, _, add = api
    outside = root.parent / "outside.safetensors"
    write_header(outside, pair("blocks.0.mlp.fc1"))
    link = root / "link.safetensors"
    try: link.symlink_to(outside)
    except OSError: pytest.skip("Symlinks require permission on this host")
    add(link)
    result = client.get("/api/h3-inspection/H3-test").json()
    assert result["status"] == "unavailable" and result["reason_code"] == "outside_library"
