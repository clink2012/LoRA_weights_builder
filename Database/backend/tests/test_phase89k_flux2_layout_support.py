from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import phase89k_flux2_layout_support as phase89k
from block_layouts import (
    FLUX2_TRANSFORMER_56,
    expected_block_count_for_layout,
    infer_layout_from_block_count,
    make_flux_layout,
    normalize_block_layout,
)
from phase89k_flux2_layout_support import (
    EXPECTED_GLOBAL_MODULES,
    EXPECTED_SOURCE_SHA256,
    EXPECTED_STABLE_ID,
    Flux2LayoutError,
    analyse_flux2_tensor_map,
    build_flux2_sealed_artifact,
    canonical_sha256,
    verify_artifact_digest,
)


RELATIVE_PATH = "FLUX/02 - Styles/aidmaMJ61Flux.2v0.5.safetensors"
DB_FILE_PATH = f"/loras/{RELATIVE_PATH}"


class _Scalar:
    def __init__(self, value: float) -> None:
        self._value = float(value)

    def item(self) -> float:
        return self._value


class _Tensor:
    def __init__(self, shape: tuple[int, int], norm: float) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self._norm = float(norm)

    def norm(self) -> _Scalar:
        return _Scalar(self._norm)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(root: Path, payload: bytes = b"phase89k-flux2-target") -> Path:
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


def _pair(prefix: str, seed: float) -> dict[str, _Tensor]:
    return {
        f"{prefix}.lora_A.weight": _Tensor((16, 4), seed),
        f"{prefix}.lora_B.weight": _Tensor((6, 16), seed + 0.5),
    }


def _valid_tensors() -> dict[str, _Tensor]:
    tensors: dict[str, _Tensor] = {}
    seed = 1.0

    for index in range(8):
        modules = ["img_attn.proj", "txt_attn.qkv", "img_mlp.2"]
        for module in modules:
            tensors.update(
                _pair(
                    f"base_model.model.double_blocks.{index}.{module}",
                    seed,
                )
            )
            seed += 1.0

    for index in range(48):
        modules = ["linear1", "linear2"]
        if index < 8:
            modules.append("modulation.lin")
        for module in modules:
            tensors.update(
                _pair(
                    f"base_model.model.single_blocks.{index}.{module}",
                    seed,
                )
            )
            seed += 1.0

    for module in EXPECTED_GLOBAL_MODULES:
        tensors.update(_pair(f"base_model.model.{module}", seed))
        seed += 1.0

    return tensors


def _phase89j_report(source_sha: str) -> dict[str, object]:
    blockers = []
    for index in range(38, 48):
        blocker = (
            f"Out-of-range single_blocks index {index}; "
            "supported range is 0..37"
        )
        blockers.extend([blocker, blocker])

    report: dict[str, object] = {
        "phase": "8.9j",
        "mode": "read-only targeted PEFT Flux analysis",
        "diagnostics_sha256": "diagnostics-digest",
        "target": {
            "relative_path": RELATIVE_PATH,
            "db_file_path": DB_FILE_PATH,
            "filename": Path(RELATIVE_PATH).name,
            "planned_stable_id": EXPECTED_STABLE_ID,
            "base_model_name": "Flux",
            "base_model_code": "FLX",
            "category_name": "Styles",
            "category_code": "STL",
            "source_sha256": source_sha,
            "clip_contributor": False,
            "clip_tensor_count": 0,
            "tensor_key_count": 276,
            "rank": 16,
            "rank_values": [16],
            "observed_double_indices": list(range(8)),
            "observed_single_indices": list(range(48)),
            "block_module_count": 128,
            "block_tensor_count": 256,
            "global_module_count": 10,
            "global_tensor_count": 20,
            "global_module_sample": list(EXPECTED_GLOBAL_MODULES),
            "unmatched_tensor_count": 0,
            "ready_for_controlled_apply": False,
            "blockers": blockers,
        },
        "summary": {
            "targets_analysed": 1,
            "ready_for_controlled_apply": 0,
            "blocked_targets": 1,
        },
        "safety": {"writes_database": False},
    }
    report["analysis_sha256"] = canonical_sha256(report)
    return report


def _mock_live_source_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        phase89k,
        "_sha256_file",
        lambda _: EXPECTED_SOURCE_SHA256,
    )


def test_layout_registry_supports_flux2_56() -> None:
    assert normalize_block_layout("FLUX2_TRANSFORMER_56") == FLUX2_TRANSFORMER_56
    assert expected_block_count_for_layout(FLUX2_TRANSFORMER_56) == 56
    assert infer_layout_from_block_count(56) == FLUX2_TRANSFORMER_56
    assert (
        make_flux_layout("Flux 2 (PEFT double+single blocks)", 56)
        == FLUX2_TRANSFORMER_56
    )


def test_valid_flux2_tensor_map_is_exactly_8_plus_48() -> None:
    result = analyse_flux2_tensor_map(_valid_tensors())

    assert result["tensor_key_count"] == 276
    assert result["model_family"] == "Flux 2"
    assert result["rank"] == 16
    assert result["rank_values"] == [16]
    assert result["block_layout"] == FLUX2_TRANSFORMER_56
    assert result["block_count"] == 56
    assert len(result["block_weights"]) == 56
    assert len(result["raw_block_strengths"]) == 56
    assert result["observed_double_indices"] == list(range(8))
    assert result["observed_single_indices"] == list(range(48))
    assert result["block_module_count"] == 128
    assert result["block_tensor_count"] == 256
    assert result["global_modules"] == list(EXPECTED_GLOBAL_MODULES)
    assert result["global_tensor_count"] == 20
    assert result["unmatched_tensor_count"] == 0
    assert result["blockers"] == []
    assert result["ready_for_sealing"] is True
    assert max(result["block_weights"]) == 1.0


def test_incomplete_pair_blocks_flux2_sealing() -> None:
    tensors = _valid_tensors()
    tensors.pop(
        "base_model.model.single_blocks.47.linear2.lora_B.weight"
    )

    result = analyse_flux2_tensor_map(tensors)

    assert result["ready_for_sealing"] is False
    assert any("Incomplete LoRA pair" in item for item in result["blockers"])


def test_missing_architecture_block_is_rejected() -> None:
    tensors = _valid_tensors()
    for key in list(tensors):
        if key.startswith("base_model.model.double_blocks.7."):
            tensors.pop(key)

    result = analyse_flux2_tensor_map(tensors)

    assert result["ready_for_sealing"] is False
    assert result["missing_double_indices"] == [7]
    assert any("Missing Flux 2 double block indices" in item for item in result["blockers"])


def test_builds_sealed_artifact_without_changing_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "loras"
    _source(root)
    db = _db(tmp_path / "lora.db")
    report = _phase89j_report(EXPECTED_SOURCE_SHA256)
    before = db.read_bytes()
    _mock_live_source_hash(monkeypatch)

    artifact = build_flux2_sealed_artifact(
        report,
        library_root=root,
        db_path=db,
        expected_report_sha256=str(report["analysis_sha256"]),
        expected_source_sha256=EXPECTED_SOURCE_SHA256,
        tensor_analyser=lambda _: analyse_flux2_tensor_map(_valid_tensors()),
    )

    target = artifact["target"]
    assert artifact["phase"] == "8.9k"
    assert target["planned_stable_id"] == EXPECTED_STABLE_ID
    assert target["model_family"] == "Flux 2"
    assert target["block_layout"] == FLUX2_TRANSFORMER_56
    assert target["block_count"] == 56
    assert len(target["block_weights"]) == 56
    assert len(target["raw_block_strengths"]) == 56
    assert artifact["safety"]["writes_database"] is False
    assert artifact["safety"]["contains_apply_mode"] is False
    assert verify_artifact_digest(artifact) == artifact["artifact_sha256"]
    assert db.read_bytes() == before


def test_unexpected_phase89j_blocker_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "loras"
    _source(root)
    db = _db(tmp_path / "lora.db")
    report = _phase89j_report(EXPECTED_SOURCE_SHA256)
    target = dict(report["target"])
    target["blockers"] = ["Unexpected tensor namespace"]
    report["target"] = target
    report.pop("analysis_sha256")
    report["analysis_sha256"] = canonical_sha256(report)
    _mock_live_source_hash(monkeypatch)

    with pytest.raises(Flux2LayoutError, match="unexpected blocker"):
        build_flux2_sealed_artifact(
            report,
            library_root=root,
            db_path=db,
            expected_report_sha256=str(report["analysis_sha256"]),
            expected_source_sha256=EXPECTED_SOURCE_SHA256,
            tensor_analyser=lambda _: analyse_flux2_tensor_map(_valid_tensors()),
        )


def test_existing_stable_id_collision_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "loras"
    _source(root)
    db = _db(tmp_path / "lora.db", collision=True)
    report = _phase89j_report(EXPECTED_SOURCE_SHA256)
    _mock_live_source_hash(monkeypatch)

    with pytest.raises(Flux2LayoutError, match="Planned stable ID already exists"):
        build_flux2_sealed_artifact(
            report,
            library_root=root,
            db_path=db,
            expected_report_sha256=str(report["analysis_sha256"]),
            expected_source_sha256=EXPECTED_SOURCE_SHA256,
            tensor_analyser=lambda _: analyse_flux2_tensor_map(_valid_tensors()),
        )
