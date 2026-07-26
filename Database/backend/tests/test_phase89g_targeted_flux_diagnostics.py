from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from phase89g_targeted_flux_analysis import FluxAnalysisPlanError  # noqa: E402
from phase89g_targeted_flux_diagnostics import build_flux_diagnostics  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _create_db(path: Path) -> None:
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
    conn.commit()
    conn.close()


def _plan(count: int = 3) -> dict:
    candidates = []
    planned = []
    for index in range(1, count + 1):
        relative = f"FLUX/03 - Utils/target-{index}.safetensors"
        candidates.append(
            {
                "source_type": "new_metadata_insert",
                "relative_path": relative,
                "filename": f"target-{index}.safetensors",
                "base_model_name": "Flux",
                "base_model_code": "FLX",
                "category_name": "Utils",
                "category_code": "UTL",
            }
        )
        planned.append(
            {
                "source_type": "new_metadata_insert",
                "relative_path": relative,
                "planned_stable_id": f"FLX-UTL-{index:03d}",
            }
        )
    return {
        "audit_mode": "read-only",
        "safety": {"writes_database": False},
        "unresolved_relocations": [],
        "stable_id_groups_exhausted": [],
        "existing_stable_id_issues": [],
        "new_metadata_insert_candidates": candidates,
        "planned_stable_ids": planned,
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "loras"
    db = tmp_path / "lora_master.db"
    for index in range(1, 4):
        path = root / "FLUX" / "03 - Utils" / f"target-{index}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"target-{index}".encode("utf-8"))
    _create_db(db)
    return root, db


def _tensor_inspector(path: Path) -> dict:
    return {
        "tensor_key_count": 4,
        "tensor_key_sample": [f"sample.{path.stem}.key"],
        "tensor_key_prefix_counts": [{"prefix": "sample", "count": 4}],
        "clip_contributor": False,
        "clip_tensor_count": 0,
    }


def test_analysis_error_is_captured_and_other_targets_continue(tmp_path: Path) -> None:
    root, db = _fixture(tmp_path)

    def analyzer(path: Path, base_model_code: str) -> dict:
        assert base_model_code == "FLX"
        if path.name == "target-2.safetensors":
            raise ValueError("No recognised Flux-style structures found")
        return {
            "model_family": "flux",
            "lora_type": "unknown-no-blocks",
            "rank": None,
            "block_weights": [],
            "raw_block_strengths": [],
        }

    before = _sha256(db)
    result = build_flux_diagnostics(
        _plan(),
        library_root=root,
        db_path=db,
        tensor_inspector=_tensor_inspector,
        analyzer=analyzer,
    )
    after = _sha256(db)

    assert before == after
    assert result["summary"] == {
        "flux_candidates": 3,
        "ready_for_controlled_apply": 2,
        "blocked_candidates": 1,
        "tensor_inspection_errors": 0,
        "analysis_errors": 1,
        "total_analysed_block_rows": 0,
    }
    assert len(result["targets"]) == 3

    by_name = {target["filename"]: target for target in result["targets"]}
    failed = by_name["target-2.safetensors"]
    assert failed["ready_for_controlled_apply"] is False
    assert failed["analysis_error"]["type"] == "ValueError"
    assert "No recognised" in failed["analysis_error"]["message"]
    assert failed["tensor_key_sample"] == ["sample.target-2.key"]

    assert by_name["target-1.safetensors"]["ready_for_controlled_apply"] is True
    assert by_name["target-3.safetensors"]["ready_for_controlled_apply"] is True


def test_tensor_error_is_captured_without_calling_analyzer(tmp_path: Path) -> None:
    root, db = _fixture(tmp_path)
    analyzer_calls: list[str] = []

    def tensor_inspector(path: Path) -> dict:
        if path.name == "target-1.safetensors":
            raise RuntimeError("broken safetensors header")
        return _tensor_inspector(path)

    def analyzer(path: Path, base_model_code: str) -> dict:
        analyzer_calls.append(path.name)
        return {
            "model_family": "flux",
            "lora_type": "unknown-no-blocks",
            "rank": None,
            "block_weights": [],
            "raw_block_strengths": [],
        }

    result = build_flux_diagnostics(
        _plan(),
        library_root=root,
        db_path=db,
        tensor_inspector=tensor_inspector,
        analyzer=analyzer,
    )

    assert result["summary"]["tensor_inspection_errors"] == 1
    assert result["summary"]["blocked_candidates"] == 1
    assert "target-1.safetensors" not in analyzer_calls
    assert set(analyzer_calls) == {
        "target-2.safetensors",
        "target-3.safetensors",
    }


def test_candidate_count_guard_remains_strict(tmp_path: Path) -> None:
    root, db = _fixture(tmp_path)

    with pytest.raises(FluxAnalysisPlanError, match="Expected exactly 3"):
        build_flux_diagnostics(
            _plan(count=2),
            library_root=root,
            db_path=db,
            tensor_inspector=_tensor_inspector,
            analyzer=lambda path, code: {},
        )
