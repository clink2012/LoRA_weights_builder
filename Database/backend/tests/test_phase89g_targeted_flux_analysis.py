from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3
import sys

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from phase89g_targeted_flux_analysis import (  # noqa: E402
    FluxAnalysisPlanError,
    build_flux_analysis_plan,
)


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


def _touch(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _candidate(relative: str, stable_id: str, category_code: str) -> tuple[dict, dict]:
    filename = Path(relative).name
    item = {
        "source_type": "new_metadata_insert",
        "relative_path": relative,
        "filename": filename,
        "base_model_name": "Flux",
        "base_model_code": "FLX",
        "category_name": {
            "PPL": "People",
            "STL": "Styles",
            "UTL": "Utils",
        }[category_code],
        "category_code": category_code,
    }
    planned = {
        "source_type": "new_metadata_insert",
        "relative_path": relative,
        "planned_stable_id": stable_id,
    }
    return item, planned


def _plan() -> dict:
    pairs = [
        _candidate(
            "FLUX/01 - People/alpha.safetensors",
            "FLX-PPL-101",
            "PPL",
        ),
        _candidate(
            "FLUX/02 - Styles/beta.safetensors",
            "FLX-STL-102",
            "STL",
        ),
        _candidate(
            "FLUX/03 - Utils/gamma.safetensors",
            "FLX-UTL-103",
            "UTL",
        ),
    ]
    return {
        "audit_mode": "read-only",
        "safety": {
            "writes_database": False,
            "runs_indexer": False,
        },
        "unresolved_relocations": [],
        "stable_id_groups_exhausted": [],
        "existing_stable_id_issues": [],
        "new_metadata_insert_candidates": [pair[0] for pair in pairs],
        "planned_stable_ids": [pair[1] for pair in pairs],
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "loras"
    db = tmp_path / "lora_master.db"
    _touch(root / "FLUX" / "01 - People" / "alpha.safetensors", b"alpha")
    _touch(root / "FLUX" / "02 - Styles" / "beta.safetensors", b"beta")
    _touch(root / "FLUX" / "03 - Utils" / "gamma.safetensors", b"gamma")
    _create_db(db)
    return root, db


def _analyzer(path: Path, base_model_code: str) -> dict:
    assert base_model_code == "FLX"
    if path.name == "alpha.safetensors":
        return {
            "model_family": "flux",
            "lora_type": "transformer",
            "rank": 16,
            "block_weights": [0.1, 0.2],
            "raw_block_strengths": [1.0, 2.0],
        }
    if path.name == "beta.safetensors":
        return {
            "model_family": "flux",
            "lora_type": "transformer",
            "rank": 32,
            "block_weights": [],
            "raw_block_strengths": [],
        }
    return {
        "model_family": "flux",
        "lora_type": "transformer",
        "rank": 8,
        "block_weights": [0.3],
        "raw_block_strengths": [3.0],
    }


def _tensor_reader(path: Path) -> tuple[bool, int, int]:
    return path.name != "beta.safetensors", 2, 10


def _layout_resolver(_lora_type: str | None, count: int) -> str | None:
    return "flux_fallback_16" if count == 0 else f"flux_transformer_{count}"


def test_read_only_plan_analyses_only_three_flux_targets(tmp_path: Path) -> None:
    root, db = _fixture(tmp_path)
    before = hashlib.sha256(db.read_bytes()).hexdigest()

    result = build_flux_analysis_plan(
        _plan(),
        library_root=root,
        db_path=db,
        analyzer=_analyzer,
        tensor_reader=_tensor_reader,
        layout_resolver=_layout_resolver,
    )

    after = hashlib.sha256(db.read_bytes()).hexdigest()
    assert before == after
    assert result["summary"] == {
        "flux_candidates": 3,
        "ready_for_controlled_apply": 3,
        "blocked_candidates": 0,
        "total_analysed_block_rows": 3,
    }
    assert [item["planned_stable_id"] for item in result["targets"]] == [
        "FLX-PPL-101",
        "FLX-STL-102",
        "FLX-UTL-103",
    ]
    assert result["targets"][0]["source_sha256"] == hashlib.sha256(b"alpha").hexdigest()
    assert result["targets"][1]["has_block_weights"] is False
    assert result["targets"][1]["block_layout"] == "flux_fallback_16"
    assert result["safety"]["writes_database"] is False
    assert result["safety"]["discovers_library_files"] is False


def test_plan_refuses_unexpected_flux_candidate_count(tmp_path: Path) -> None:
    root, db = _fixture(tmp_path)
    plan = _plan()
    plan["new_metadata_insert_candidates"].pop()

    with pytest.raises(
        FluxAnalysisPlanError,
        match="Expected exactly 3 FLX candidates, found 2",
    ):
        build_flux_analysis_plan(
            plan,
            library_root=root,
            db_path=db,
            analyzer=_analyzer,
            tensor_reader=_tensor_reader,
            layout_resolver=_layout_resolver,
        )


def test_plan_refuses_existing_path_or_stable_id(tmp_path: Path) -> None:
    root, db = _fixture(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO lora (file_path, stable_id) VALUES (?, ?)",
        ("/loras/FLUX/01 - People/alpha.safetensors", "OLD-PPL-001"),
    )
    conn.commit()
    conn.close()

    with pytest.raises(
        FluxAnalysisPlanError,
        match="Target file_path already exists in DB",
    ):
        build_flux_analysis_plan(
            _plan(),
            library_root=root,
            db_path=db,
            analyzer=_analyzer,
            tensor_reader=_tensor_reader,
            layout_resolver=_layout_resolver,
        )


def test_unsupported_layout_marks_candidate_blocked(tmp_path: Path) -> None:
    root, db = _fixture(tmp_path)

    def layout_resolver(_lora_type: str | None, count: int) -> str | None:
        return None if count == 2 else _layout_resolver(_lora_type, count)

    result = build_flux_analysis_plan(
        _plan(),
        library_root=root,
        db_path=db,
        analyzer=_analyzer,
        tensor_reader=_tensor_reader,
        layout_resolver=layout_resolver,
    )

    assert result["summary"]["ready_for_controlled_apply"] == 2
    assert result["summary"]["blocked_candidates"] == 1
    blocked = [item for item in result["targets"] if not item["ready_for_controlled_apply"]]
    assert len(blocked) == 1
    assert "No supported block layout" in blocked[0]["warnings"][0]
