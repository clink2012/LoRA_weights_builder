from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from phase89g_targeted_flux_analysis import plan_sha256
from phase89h_sealed_flux_artifact import (
    SealedArtifactError,
    build_sealed_flux_artifact,
    verify_artifact_digest,
)


STABLE_ID = "FLX-PPL-207"
RELATIVE_PATH = "FLUX/01 - People/ang3l4wh1t3-f1.safetensors"
DB_FILE_PATH = f"/loras/{RELATIVE_PATH}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE lora (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL UNIQUE,
            stable_id TEXT
        );
        """
    )
    conn.commit()
    conn.close()
    return path


def _plan() -> dict[str, object]:
    return {
        "audit_mode": "read-only",
        "safety": {"writes_database": False},
        "new_metadata_insert_candidates": [],
    }


def _diagnostics(plan: dict[str, object], source_sha: str) -> dict[str, object]:
    return {
        "phase": "8.9g-diagnostics",
        "plan_sha256": plan_sha256(plan),
        "summary": {
            "flux_candidates": 3,
            "ready_for_controlled_apply": 1,
            "blocked_candidates": 2,
        },
        "targets": [
            {
                "relative_path": RELATIVE_PATH,
                "db_file_path": DB_FILE_PATH,
                "filename": "ang3l4wh1t3-f1.safetensors",
                "planned_stable_id": STABLE_ID,
                "base_model_name": "Flux",
                "base_model_code": "FLX",
                "category_name": "People",
                "category_code": "PPL",
                "source_sha256": source_sha,
                "tensor_key_count": 912,
                "clip_contributor": False,
                "clip_tensor_count": 0,
                "tensor_inspection_error": None,
                "analysis_error": None,
                "model_family": "Flux",
                "lora_type": "Flux (UNet double+single blocks)",
                "rank": None,
                "block_count": 57,
                "raw_strength_count": 57,
                "block_layout": "flux_unet_57",
                "warnings": [],
                "ready_for_controlled_apply": True,
            },
            {
                "relative_path": "FLUX/02 - Styles/blocked.safetensors",
                "planned_stable_id": "FLX-STL-263",
                "ready_for_controlled_apply": False,
            },
            {
                "relative_path": "FLUX/05 - Body/broken.safetensors",
                "planned_stable_id": "FLX-BDY-071",
                "ready_for_controlled_apply": False,
            },
        ],
        "safety": {"writes_database": False},
    }


def _source(root: Path) -> Path:
    source = root / Path(*RELATIVE_PATH.split("/"))
    source.parent.mkdir(parents=True)
    source.write_bytes(b"sealed-fixture")
    return source


def _tensor_inspector(_: Path) -> dict[str, object]:
    return {
        "tensor_key_count": 912,
        "clip_contributor": False,
        "clip_tensor_count": 0,
        "tensor_key_sample": [],
        "tensor_key_prefix_counts": [],
    }


def _analyzer(_: Path, base_model_code: str) -> dict[str, object]:
    assert base_model_code == "FLX"
    return {
        "model_family": "Flux",
        "lora_type": "Flux (UNet double+single blocks)",
        "rank": None,
        "block_weights": [index / 56 for index in range(57)],
        "raw_block_strengths": [float(index + 1) for index in range(57)],
    }


def test_builds_sealed_artifact_without_changing_db(tmp_path: Path) -> None:
    root = tmp_path / "loras"
    source = _source(root)
    db = _db(tmp_path / "lora.db")
    plan = _plan()
    diagnostics = _diagnostics(plan, _sha256(source))
    db_before = db.read_bytes()

    artifact = build_sealed_flux_artifact(
        plan,
        diagnostics,
        library_root=root,
        db_path=db,
        analyzer=_analyzer,
        tensor_inspector=_tensor_inspector,
    )

    assert artifact["phase"] == "8.9h"
    assert artifact["target"]["planned_stable_id"] == STABLE_ID
    assert artifact["target"]["block_count"] == 57
    assert len(artifact["target"]["block_weights"]) == 57
    assert len(artifact["target"]["raw_block_strengths"]) == 57
    assert verify_artifact_digest(artifact) == artifact["artifact_sha256"]
    assert db.read_bytes() == db_before


def test_rejects_source_hash_change(tmp_path: Path) -> None:
    root = tmp_path / "loras"
    source = _source(root)
    db = _db(tmp_path / "lora.db")
    plan = _plan()
    diagnostics = _diagnostics(plan, _sha256(source))
    source.write_bytes(b"changed-after-diagnostics")

    with pytest.raises(SealedArtifactError, match="source SHA-256 mismatch"):
        build_sealed_flux_artifact(
            plan,
            diagnostics,
            library_root=root,
            db_path=db,
            analyzer=_analyzer,
            tensor_inspector=_tensor_inspector,
        )


def test_rejects_existing_stable_id(tmp_path: Path) -> None:
    root = tmp_path / "loras"
    source = _source(root)
    db = _db(tmp_path / "lora.db")
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO lora (file_path, stable_id) VALUES (?, ?)",
        ("/loras/existing.safetensors", STABLE_ID),
    )
    conn.commit()
    conn.close()
    plan = _plan()
    diagnostics = _diagnostics(plan, _sha256(source))

    with pytest.raises(SealedArtifactError, match="Planned stable ID already exists"):
        build_sealed_flux_artifact(
            plan,
            diagnostics,
            library_root=root,
            db_path=db,
            analyzer=_analyzer,
            tensor_inspector=_tensor_inspector,
        )


def test_requires_exactly_one_ready_target(tmp_path: Path) -> None:
    root = tmp_path / "loras"
    source = _source(root)
    db = _db(tmp_path / "lora.db")
    plan = _plan()
    diagnostics = _diagnostics(plan, _sha256(source))
    second = dict(diagnostics["targets"][1])
    second["ready_for_controlled_apply"] = True
    diagnostics["targets"][1] = second

    with pytest.raises(SealedArtifactError, match="exactly one ready"):
        build_sealed_flux_artifact(
            plan,
            diagnostics,
            library_root=root,
            db_path=db,
            analyzer=_analyzer,
            tensor_inspector=_tensor_inspector,
        )
