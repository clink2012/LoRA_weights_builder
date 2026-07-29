from __future__ import annotations

import hashlib
import sqlite3
import sys
import types
from pathlib import Path

import pytest


class _FakeScalar:
    def __init__(self, value: float) -> None:
        self._value = float(value)

    def item(self) -> float:
        return self._value


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], norm_value: float = 1.0) -> None:
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self._norm_value = float(norm_value)

    def norm(self) -> _FakeScalar:
        return _FakeScalar(self._norm_value)


fake_torch = types.ModuleType("torch")
fake_torch.Tensor = FakeTensor
sys.modules.setdefault("torch", fake_torch)

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from phase89j_targeted_peft_flux_analysis import (
    EXPECTED_SOURCE_SHA256,
    EXPECTED_STABLE_ID,
    PeftFluxAnalysisError,
    analyse_peft_tensor_map,
    build_targeted_peft_analysis,
    verify_analysis_digest,
)


RELATIVE_PATH = "FLUX/02 - Styles/aidmaMJ61Flux.2v0.5.safetensors"
DB_FILE_PATH = f"/loras/{RELATIVE_PATH}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(root: Path, payload: bytes = b"phase89j-target") -> Path:
    path = root / Path(*RELATIVE_PATH.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _db(path: Path, *, collision: bool = False) -> Path:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE lora (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL UNIQUE,
            stable_id TEXT
        )
        """
    )
    if collision:
        conn.execute(
            "INSERT INTO lora (file_path, stable_id) VALUES (?, ?)",
            ("/loras/existing.safetensors", EXPECTED_STABLE_ID),
        )
    conn.commit()
    conn.close()
    return path


def _diagnostics(source_sha: str) -> dict[str, object]:
    return {
        "phase": "8.9g-diagnostics",
        "plan_sha256": "plan-digest",
        "summary": {
            "flux_candidates": 3,
            "ready_for_controlled_apply": 1,
            "blocked_candidates": 2,
        },
        "targets": [
            {
                "relative_path": "FLUX/01 - People/ready.safetensors",
                "planned_stable_id": "FLX-PPL-207",
                "ready_for_controlled_apply": True,
            },
            {
                "relative_path": RELATIVE_PATH,
                "db_file_path": DB_FILE_PATH,
                "filename": Path(RELATIVE_PATH).name,
                "planned_stable_id": EXPECTED_STABLE_ID,
                "base_model_name": "Flux",
                "base_model_code": "FLX",
                "category_name": "Styles",
                "category_code": "STL",
                "source_sha256": source_sha,
                "tensor_key_count": 12,
                "clip_contributor": False,
                "clip_tensor_count": 0,
                "tensor_inspection_error": None,
                "analysis_error": {
                    "type": "ValueError",
                    "message": "No UNet-style keys found in safetensors file.",
                },
                "ready_for_controlled_apply": False,
            },
            {
                "relative_path": "FLUX/05 - Body/broken.safetensors",
                "planned_stable_id": "FLX-BDY-071",
                "tensor_inspection_error": {"type": "SafetensorError"},
                "ready_for_controlled_apply": False,
            },
        ],
        "safety": {"writes_database": False},
    }


def _pair(prefix: str, rank: int = 2) -> dict[str, FakeTensor]:
    return {
        f"{prefix}.lora_A.weight": FakeTensor((rank, 4)),
        f"{prefix}.lora_B.weight": FakeTensor((6, rank)),
    }


def _valid_tensors() -> dict[str, FakeTensor]:
    tensors: dict[str, FakeTensor] = {}
    tensors.update(
        _pair("base_model.model.double_blocks.0.img_attn.proj")
    )
    tensors.update(
        _pair("base_model.model.double_blocks.18.txt_attn.qkv")
    )
    tensors.update(
        _pair("base_model.model.single_blocks.0.linear1")
    )
    tensors.update(
        _pair("base_model.model.single_blocks.37.linear2")
    )
    tensors.update(_pair("base_model.model.time_in.in_layer"))
    tensors.update(_pair("base_model.model.final_layer.linear"))
    return tensors


def test_valid_peft_namespace_maps_to_57_blocks_and_preserves_db(tmp_path: Path) -> None:
    root = tmp_path / "loras"
    source = _source(root)
    db = _db(tmp_path / "lora.db")
    diagnostics = _diagnostics(_sha256(source))
    before = db.read_bytes()

    result = build_targeted_peft_analysis(
        diagnostics,
        library_root=root,
        db_path=db,
        expected_source_sha256=_sha256(source),
        tensor_analyser=lambda _: analyse_peft_tensor_map(_valid_tensors()),
    )

    target = result["target"]
    assert result["phase"] == "8.9j"
    assert target["planned_stable_id"] == EXPECTED_STABLE_ID
    assert target["block_layout"] == "flux_unet_57"
    assert target["block_count"] == 57
    assert len(target["block_weights"]) == 57
    assert len(target["raw_block_strengths"]) == 57
    assert target["observed_double_indices"] == [0, 18]
    assert target["observed_single_indices"] == [0, 37]
    assert target["rank"] == 2
    assert target["rank_values"] == [2]
    assert target["global_tensor_count"] == 4
    assert target["ready_for_controlled_apply"] is True
    assert target["blockers"] == []
    assert verify_analysis_digest(result) == result["analysis_sha256"]
    assert db.read_bytes() == before


def test_incomplete_lora_pair_is_blocked() -> None:
    tensors = _valid_tensors()
    tensors.pop(
        "base_model.model.single_blocks.37.linear2.lora_B.weight"
    )

    result = analyse_peft_tensor_map(tensors)

    assert result["ready_for_controlled_apply"] is False
    assert any("Incomplete LoRA pair" in item for item in result["blockers"])


def test_out_of_range_block_index_is_blocked() -> None:
    tensors = _valid_tensors()
    tensors.update(
        _pair("base_model.model.double_blocks.19.img_attn.proj")
    )

    result = analyse_peft_tensor_map(tensors)

    assert result["ready_for_controlled_apply"] is False
    assert any("Out-of-range double_blocks index 19" in item for item in result["blockers"])


def test_unrecognised_lora_namespace_is_blocked() -> None:
    tensors = _valid_tensors()
    tensors.update(_pair("base_model.model.weird_blocks.0.linear"))

    result = analyse_peft_tensor_map(tensors)

    assert result["ready_for_controlled_apply"] is False
    assert any("unrecognised LoRA tensor" in item for item in result["blockers"])


def test_source_hash_drift_is_rejected_before_tensor_analysis(tmp_path: Path) -> None:
    root = tmp_path / "loras"
    source = _source(root)
    db = _db(tmp_path / "lora.db")
    diagnostics = _diagnostics(_sha256(source))
    source.write_bytes(b"changed-after-diagnostics")

    called = False

    def analyser(_: Path) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    with pytest.raises(PeftFluxAnalysisError, match="Source SHA-256 mismatch"):
        build_targeted_peft_analysis(
            diagnostics,
            library_root=root,
            db_path=db,
            expected_source_sha256="0" * 64,
            tensor_analyser=analyser,
        )

    assert called is False


def test_existing_stable_id_collision_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "loras"
    source = _source(root)
    db = _db(tmp_path / "lora.db", collision=True)
    diagnostics = _diagnostics(_sha256(source))

    with pytest.raises(PeftFluxAnalysisError, match="Planned stable ID already exists"):
        build_targeted_peft_analysis(
            diagnostics,
            library_root=root,
            db_path=db,
            expected_source_sha256=_sha256(source),
            tensor_analyser=lambda _: analyse_peft_tensor_map(_valid_tensors()),
        )
