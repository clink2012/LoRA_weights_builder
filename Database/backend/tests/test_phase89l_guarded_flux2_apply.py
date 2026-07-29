from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import phase89l_guarded_flux2_apply as phase89l
from phase89k_flux2_layout_support import (
    EXPECTED_GLOBAL_MODULES,
    canonical_sha256,
)


STABLE_ID = "FLX-STL-263"
DAMAGED_ID = "FLX-BDY-071"
RELATIVE_PATH = "FLUX/02 - Styles/aidmaMJ61Flux.2v0.5.safetensors"
DB_FILE_PATH = f"/loras/{RELATIVE_PATH}"


def _db(path: Path) -> Path:
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
    return path


def _source(root: Path) -> Path:
    source = root / Path(*RELATIVE_PATH.split("/"))
    source.parent.mkdir(parents=True)
    source.write_bytes(b"phase89l-flux2-source-fixture")
    return source


def _artifact(source: Path) -> dict[str, object]:
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    weights = [(index + 1) / 56 for index in range(56)]
    raw = [float(index + 1) for index in range(56)]

    artifact: dict[str, object] = {
        "phase": "8.9k",
        "mode": "read-only sealed targeted Flux 2 artifact",
        "phase89j_analysis_sha256": phase89l.EXPECTED_PHASE89J_SHA256,
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
                "Global projection LoRA tensors are recorded separately and excluded from per-block strengths"
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


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, object]]:
    root = tmp_path / "loras"
    source = _source(root)
    db = _db(tmp_path / "lora.db")
    artifact = _artifact(source)

    monkeypatch.setattr(
        phase89l,
        "EXPECTED_SOURCE_SHA256",
        str(artifact["target"]["source_sha256"]),
    )
    monkeypatch.setattr(
        phase89l,
        "EXPECTED_ARTIFACT_SHA256",
        str(artifact["artifact_sha256"]),
    )
    return root, db, artifact


def _insert_existing(conn: sqlite3.Connection, stable_id: str, file_path: str) -> None:
    conn.execute(
        """
        INSERT INTO lora (
            file_path, filename, base_model_code, category_code,
            has_block_weights, clip_contributor, clip_tensor_count,
            last_modified, created_at, updated_at, stable_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            file_path,
            Path(file_path).name,
            "FLX",
            "STL",
            0,
            0,
            -1,
            0.0,
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
            stable_id,
        ),
    )


def test_dry_run_preview_changes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, db, artifact = _fixture(tmp_path, monkeypatch)
    before = db.read_bytes()

    preview = phase89l.build_preview(
        artifact,
        db_path=db,
        library_root=root,
    )

    assert preview.artifact_sha256 == artifact["artifact_sha256"]
    assert preview.target["planned_stable_id"] == STABLE_ID
    assert preview.target["model_family"] == "Flux 2"
    assert preview.summary()["block_rows_to_insert"] == 56
    assert preview.summary()["global_projection_tensors"] == 20
    assert db.read_bytes() == before


def test_apply_inserts_one_lora_and_56_blocks_with_verified_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, db, artifact = _fixture(tmp_path, monkeypatch)
    backup_dir = tmp_path / "backups"
    preview = phase89l.build_preview(
        artifact,
        db_path=db,
        library_root=root,
    )

    result = phase89l.apply_artifact(
        preview,
        db_path=db,
        library_root=root,
        backup_dir=backup_dir,
        expected_artifact_sha256=str(artifact["artifact_sha256"]),
    )

    backup = Path(result["backup_path"])
    assert backup.is_file()
    assert ".phase89l." in backup.name
    assert result["lora_rows_inserted"] == 1
    assert result["block_rows_inserted"] == 56
    assert result["global_projection_tensors_preserved_in_artifact"] == 20
    assert result["damaged_candidate_untouched"] == 1

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM lora WHERE stable_id = ?",
        (STABLE_ID,),
    ).fetchone()
    assert row is not None
    assert row["file_path"] == DB_FILE_PATH
    assert row["base_model_name"] == "Flux 2"
    assert row["model_family"] == "Flux 2"
    assert row["lora_type"] == "Flux 2 (PEFT double+single blocks)"
    assert row["rank"] == 16
    assert row["has_block_weights"] == 1
    assert row["block_layout"] == "flux2_transformer_56"

    blocks = conn.execute(
        """
        SELECT *
        FROM lora_block_weights
        WHERE lora_id = ?
        ORDER BY block_index
        """,
        (row["id"],),
    ).fetchall()
    damaged = conn.execute(
        "SELECT COUNT(1) FROM lora WHERE stable_id = ?",
        (DAMAGED_ID,),
    ).fetchone()[0]
    conn.close()

    assert len(blocks) == 56
    assert [block["block_index"] for block in blocks] == list(range(56))
    assert all(block["stable_id"] == STABLE_ID for block in blocks)
    assert damaged == 0


def test_apply_rejects_wrong_digest_before_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, db, artifact = _fixture(tmp_path, monkeypatch)
    backup_dir = tmp_path / "backups"
    preview = phase89l.build_preview(
        artifact,
        db_path=db,
        library_root=root,
    )

    with pytest.raises(phase89l.GuardedFlux2ApplyError, match="Artifact digest mismatch"):
        phase89l.apply_artifact(
            preview,
            db_path=db,
            library_root=root,
            backup_dir=backup_dir,
            expected_artifact_sha256="0" * 64,
        )

    assert not backup_dir.exists()


def test_apply_rejects_source_drift_before_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, db, artifact = _fixture(tmp_path, monkeypatch)
    backup_dir = tmp_path / "backups"
    preview = phase89l.build_preview(
        artifact,
        db_path=db,
        library_root=root,
    )
    preview.source_path.write_bytes(b"source-changed-after-seal")

    with pytest.raises(
        phase89l.GuardedFlux2ApplyError,
        match="source SHA-256 mismatch",
    ):
        phase89l.apply_artifact(
            preview,
            db_path=db,
            library_root=root,
            backup_dir=backup_dir,
            expected_artifact_sha256=str(artifact["artifact_sha256"]),
        )

    assert not backup_dir.exists()


def test_preview_rejects_stable_id_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, db, artifact = _fixture(tmp_path, monkeypatch)
    conn = sqlite3.connect(db)
    _insert_existing(conn, STABLE_ID, "/loras/existing.safetensors")
    conn.commit()
    conn.close()

    with pytest.raises(
        phase89l.GuardedFlux2ApplyError,
        match="Planned stable ID already exists",
    ):
        phase89l.build_preview(
            artifact,
            db_path=db,
            library_root=root,
        )


def test_preview_rejects_damaged_target_presence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, db, artifact = _fixture(tmp_path, monkeypatch)
    conn = sqlite3.connect(db)
    _insert_existing(conn, DAMAGED_ID, "/loras/damaged.safetensors")
    conn.commit()
    conn.close()

    with pytest.raises(
        phase89l.GuardedFlux2ApplyError,
        match="Damaged quarantined target unexpectedly exists",
    ):
        phase89l.build_preview(
            artifact,
            db_path=db,
            library_root=root,
        )


def test_preview_rejects_layout_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, db, artifact = _fixture(tmp_path, monkeypatch)
    target = dict(artifact["target"])
    target["block_layout"] = "flux_unet_57"
    artifact["target"] = target
    artifact.pop("artifact_sha256")
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    monkeypatch.setattr(
        phase89l,
        "EXPECTED_ARTIFACT_SHA256",
        str(artifact["artifact_sha256"]),
    )

    with pytest.raises(
        phase89l.GuardedFlux2ApplyError,
        match="artifact target block_layout mismatch",
    ):
        phase89l.build_preview(
            artifact,
            db_path=db,
            library_root=root,
        )


def test_mid_transaction_block_failure_rolls_back_lora_insert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, db, artifact = _fixture(tmp_path, monkeypatch)
    backup_dir = tmp_path / "backups"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TRIGGER fail_phase89l_block
        BEFORE INSERT ON lora_block_weights
        WHEN NEW.block_index = 10
        BEGIN
            SELECT RAISE(ABORT, 'forced Flux 2 block failure');
        END;
        """
    )
    conn.commit()
    conn.close()

    preview = phase89l.build_preview(
        artifact,
        db_path=db,
        library_root=root,
    )

    with pytest.raises(sqlite3.IntegrityError, match="forced Flux 2 block failure"):
        phase89l.apply_artifact(
            preview,
            db_path=db,
            library_root=root,
            backup_dir=backup_dir,
            expected_artifact_sha256=str(artifact["artifact_sha256"]),
        )

    assert list(backup_dir.glob("*.db"))
    conn = sqlite3.connect(db)
    lora_count = conn.execute("SELECT COUNT(1) FROM lora").fetchone()[0]
    block_count = conn.execute(
        "SELECT COUNT(1) FROM lora_block_weights"
    ).fetchone()[0]
    conn.close()
    assert lora_count == 0
    assert block_count == 0
