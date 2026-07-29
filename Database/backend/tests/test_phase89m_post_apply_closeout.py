from __future__ import annotations

import hashlib
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from phase89k_flux2_layout_support import (
    EXPECTED_GLOBAL_MODULES,
    canonical_sha256,
)
from phase89m_post_apply_closeout import (
    Phase89mVerificationError,
    verify_phase89m_closeout,
    verify_report_digest,
)


STABLE_ID = "FLX-STL-263"
DAMAGED_ID = "FLX-BDY-071"
RELATIVE_PATH = "FLUX/02 - Styles/aidmaMJ61Flux.2v0.5.safetensors"
DB_FILE_PATH = f"/loras/{RELATIVE_PATH}"


@dataclass
class Fixture:
    root: Path
    source: Path
    artifact_path: Path
    artifact: dict[str, object]
    current_db: Path
    backup_db: Path
    target_lora_id: int


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _create_schema(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE lora (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL UNIQUE,
            filename TEXT NOT NULL,
            base_model_name TEXT,
            base_model_code TEXT,
            category_name TEXT,
            category_code TEXT,
            model_family TEXT,
            lora_type TEXT,
            rank INTEGER,
            has_block_weights INTEGER NOT NULL DEFAULT 0,
            block_layout TEXT,
            clip_contributor INTEGER NOT NULL DEFAULT 0,
            clip_tensor_count INTEGER NOT NULL DEFAULT -1,
            last_modified REAL NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            stable_id TEXT
        );
        CREATE TABLE lora_block_weights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lora_id INTEGER NOT NULL,
            stable_id TEXT,
            block_index INTEGER NOT NULL,
            weight REAL NOT NULL,
            raw_strength REAL,
            FOREIGN KEY (lora_id) REFERENCES lora(id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()
    conn.close()


def _insert_existing(
    conn: sqlite3.Connection,
    *,
    file_path: str,
    stable_id: str,
    with_blocks: bool,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO lora (
            file_path, filename,
            base_model_name, base_model_code,
            category_name, category_code,
            model_family, lora_type, rank,
            has_block_weights, block_layout,
            clip_contributor, clip_tensor_count,
            last_modified, created_at, updated_at,
            stable_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            file_path,
            Path(file_path).name,
            "Flux",
            "FLX",
            "People",
            "PPL",
            "Flux",
            "Flux (UNet double+single blocks)" if with_blocks else None,
            None,
            1 if with_blocks else 0,
            "flux_unet_57" if with_blocks else None,
            0,
            0,
            100.0,
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
            stable_id,
        ),
    )
    lora_id = int(cursor.lastrowid)
    if with_blocks:
        conn.executemany(
            """
            INSERT INTO lora_block_weights (
                lora_id, stable_id, block_index, weight, raw_strength
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (lora_id, stable_id, 0, 0.25, 1.0),
                (lora_id, stable_id, 1, 0.5, 2.0),
            ],
        )
    return lora_id


def _artifact(source: Path) -> dict[str, object]:
    source_sha = _sha256(source)
    weights = [round(0.75 + (0.25 * index / 55), 6) for index in range(56)]
    weights[-1] = 1.0
    raw = [float(index + 1) for index in range(56)]

    artifact: dict[str, object] = {
        "phase": "8.9k",
        "mode": "read-only sealed targeted Flux 2 artifact",
        "phase89j_analysis_sha256": "j" * 64,
        "target": {
            "relative_path": RELATIVE_PATH,
            "db_file_path": DB_FILE_PATH,
            "filename": source.name,
            "planned_stable_id": STABLE_ID,
            "base_model_name": "Flux 2",
            "base_model_code": "FLX",
            "category_name": "Styles",
            "category_code": "STL",
            "source_size_bytes": source.stat().st_size,
            "source_mtime": source.stat().st_mtime,
            "source_sha256": source_sha,
            "clip_contributor": False,
            "clip_tensor_count": 0,
            "tensor_key_count": 276,
            "model_family": "Flux 2",
            "lora_type": "Flux 2 (PEFT double+single blocks)",
            "rank": 16,
            "rank_values": [16],
            "block_layout": "flux2_transformer_56",
            "block_count": 56,
            "block_weights": weights,
            "raw_block_strengths": raw,
            "observed_double_indices": list(range(8)),
            "observed_single_indices": list(range(48)),
            "missing_double_indices": [],
            "missing_single_indices": [],
            "extra_double_indices": [],
            "extra_single_indices": [],
            "block_module_count": 128,
            "block_tensor_count": 256,
            "global_module_count": 10,
            "global_tensor_count": 20,
            "global_modules": list(EXPECTED_GLOBAL_MODULES),
            "unmatched_tensor_count": 0,
            "unmatched_key_sample": [],
            "warnings": [
                "Global projection LoRA tensors are recorded separately and "
                "excluded from per-block strengths"
            ],
            "blockers": [],
            "ready_for_sealing": True,
        },
        "summary": {
            "targets_analysed": 1,
            "ready_for_later_controlled_apply": 1,
            "block_rows": 56,
            "global_projection_tensors": 20,
            "damaged_flux_targets_untouched": 1,
        },
        "safety": {
            "database_open_mode": "SQLite URI mode=ro plus PRAGMA query_only=ON",
            "writes_database": False,
            "creates_backup": False,
            "runs_full_indexer": False,
            "discovers_library_files": False,
            "opens_only_phase89j_target": True,
            "assigns_stable_ids": False,
            "deletes_rows": False,
            "touches_damaged_flux_target": False,
            "contains_apply_mode": False,
        },
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def _build_fixture(
    tmp_path: Path,
    *,
    duplicate_backup_ids: bool = False,
    damaged_in_backup: bool = False,
) -> Fixture:
    root = tmp_path / "loras"
    source = root / Path(*RELATIVE_PATH.split("/"))
    source.parent.mkdir(parents=True)
    source.write_bytes(b"phase89m-flux2-source")

    artifact = _artifact(source)
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(
        __import__("json").dumps(artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    backup_db = tmp_path / "backup.db"
    current_db = tmp_path / "current.db"
    _create_schema(backup_db)

    conn = sqlite3.connect(backup_db)
    first_id = "FLX-PPL-001"
    second_id = first_id if duplicate_backup_ids else "FLX-PPL-002"
    _insert_existing(
        conn,
        file_path="/loras/FLUX/01 - People/one.safetensors",
        stable_id=first_id,
        with_blocks=False,
    )
    _insert_existing(
        conn,
        file_path="/loras/FLUX/01 - People/two.safetensors",
        stable_id=second_id,
        with_blocks=True,
    )
    if damaged_in_backup:
        _insert_existing(
            conn,
            file_path="/loras/FLUX/05 - Body/damaged.safetensors",
            stable_id=DAMAGED_ID,
            with_blocks=False,
        )
    conn.commit()
    conn.close()

    shutil.copy2(backup_db, current_db)

    target = artifact["target"]
    assert isinstance(target, dict)
    conn = sqlite3.connect(current_db)
    cursor = conn.execute(
        """
        INSERT INTO lora (
            file_path, filename,
            base_model_name, base_model_code,
            category_name, category_code,
            model_family, lora_type, rank,
            has_block_weights, block_layout,
            clip_contributor, clip_tensor_count,
            last_modified, created_at, updated_at,
            stable_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            target["db_file_path"],
            target["filename"],
            "Flux 2",
            "FLX",
            "Styles",
            "STL",
            "Flux 2",
            "Flux 2 (PEFT double+single blocks)",
            16,
            1,
            "flux2_transformer_56",
            0,
            0,
            float(target["source_mtime"]),
            "2026-07-29T22:18:23+00:00",
            "2026-07-29T22:18:23+00:00",
            STABLE_ID,
        ),
    )
    target_lora_id = int(cursor.lastrowid)
    conn.executemany(
        """
        INSERT INTO lora_block_weights (
            lora_id, stable_id, block_index, weight, raw_strength
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                target_lora_id,
                STABLE_ID,
                index,
                float(target["block_weights"][index]),
                float(target["raw_block_strengths"][index]),
            )
            for index in range(56)
        ],
    )
    conn.commit()
    conn.close()

    return Fixture(
        root=root,
        source=source,
        artifact_path=artifact_path,
        artifact=artifact,
        current_db=current_db,
        backup_db=backup_db,
        target_lora_id=target_lora_id,
    )


def _counts(path: Path) -> tuple[int, int]:
    conn = sqlite3.connect(path)
    try:
        return (
            int(conn.execute("SELECT COUNT(*) FROM lora").fetchone()[0]),
            int(
                conn.execute(
                    "SELECT COUNT(*) FROM lora_block_weights"
                ).fetchone()[0]
            ),
        )
    finally:
        conn.close()


def _verify(fixture: Fixture, **overrides: object) -> dict[str, object]:
    current_loras, current_blocks = _counts(fixture.current_db)
    backup_loras, backup_blocks = _counts(fixture.backup_db)
    artifact_sha = str(fixture.artifact["artifact_sha256"])
    source_sha = str(fixture.artifact["target"]["source_sha256"])

    values: dict[str, object] = {
        "artifact_path": fixture.artifact_path,
        "current_db_path": fixture.current_db,
        "backup_db_path": fixture.backup_db,
        "library_root": fixture.root,
        "expected_artifact_sha256": artifact_sha,
        "expected_current_db_sha256": _sha256(fixture.current_db),
        "expected_backup_db_sha256": _sha256(fixture.backup_db),
        "expected_backup_name": fixture.backup_db.name,
        "expected_source_sha256": source_sha,
        "expected_target_lora_id": fixture.target_lora_id,
        "expected_current_lora_rows": current_loras,
        "expected_backup_lora_rows": backup_loras,
        "expected_current_block_rows": current_blocks,
        "expected_backup_block_rows": backup_blocks,
    }
    values.update(overrides)
    return verify_phase89m_closeout(**values)


def test_complete_closeout_preserves_every_preexisting_row(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    current_before = fixture.current_db.read_bytes()
    backup_before = fixture.backup_db.read_bytes()

    report = _verify(fixture)

    assert report["status"] == "verified"
    assert report["database"]["preserved_preexisting_lora_rows"] == 2
    assert report["database"]["preserved_preexisting_block_rows"] == 2
    assert report["target"]["verified_block_rows"] == 56
    assert report["quarantine"]["status"] == "absent and untouched"
    assert verify_report_digest(report) == report["verification_sha256"]
    assert fixture.current_db.read_bytes() == current_before
    assert fixture.backup_db.read_bytes() == backup_before


def test_changed_preexisting_lora_row_is_rejected(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    conn = sqlite3.connect(fixture.current_db)
    conn.execute("UPDATE lora SET filename = 'changed.safetensors' WHERE id = 1")
    conn.commit()
    conn.close()

    with pytest.raises(
        Phase89mVerificationError,
        match="pre-existing lora row changed",
    ):
        _verify(fixture)


def test_changed_preexisting_block_row_is_rejected(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    conn = sqlite3.connect(fixture.current_db)
    conn.execute("UPDATE lora_block_weights SET weight = 0.9 WHERE id = 1")
    conn.commit()
    conn.close()

    with pytest.raises(
        Phase89mVerificationError,
        match="pre-existing lora_block_weights row changed",
    ):
        _verify(fixture)


def test_missing_new_block_is_rejected(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    conn = sqlite3.connect(fixture.current_db)
    conn.execute(
        "DELETE FROM lora_block_weights WHERE lora_id = ? AND block_index = 55",
        (fixture.target_lora_id,),
    )
    conn.commit()
    conn.close()

    with pytest.raises(Phase89mVerificationError, match="block row delta"):
        _verify(fixture)


def test_duplicate_stable_ids_in_preserved_database_are_rejected(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path, duplicate_backup_ids=True)

    with pytest.raises(Phase89mVerificationError, match="duplicate stable IDs"):
        _verify(fixture)


def test_damaged_target_leak_is_rejected(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path, damaged_in_backup=True)

    with pytest.raises(
        Phase89mVerificationError,
        match="damaged target current rows",
    ):
        _verify(fixture)


def test_source_drift_is_rejected(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    fixture.source.write_bytes(b"source-drift-after-apply")

    with pytest.raises(Phase89mVerificationError, match="source SHA-256"):
        _verify(fixture)


def test_wrong_current_database_digest_is_rejected(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)

    with pytest.raises(Phase89mVerificationError, match="current DB SHA-256"):
        _verify(fixture, expected_current_db_sha256="0" * 64)
