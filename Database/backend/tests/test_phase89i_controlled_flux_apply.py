from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from phase89h_sealed_flux_artifact import canonical_sha256
from phase89i_controlled_flux_apply import (
    ControlledFluxApplyError,
    apply_artifact,
    build_preview,
)


STABLE_ID = "FLX-PPL-207"
RELATIVE_PATH = "FLUX/01 - People/ang3l4wh1t3-f1.safetensors"
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
    source.write_bytes(b"phase89i-source-fixture")
    return source


def _artifact(source: Path) -> dict[str, object]:
    import hashlib

    weights = [index / 56 for index in range(57)]
    raw = [float(index + 1) for index in range(57)]
    payload: dict[str, object] = {
        "phase": "8.9h",
        "mode": "read-only sealed single-target Flux artifact",
        "plan_sha256": "p" * 64,
        "diagnostics_sha256": "d" * 64,
        "target": {
            "relative_path": RELATIVE_PATH,
            "db_file_path": DB_FILE_PATH,
            "filename": source.name,
            "planned_stable_id": STABLE_ID,
            "base_model_name": "Flux",
            "base_model_code": "FLX",
            "category_name": "People",
            "category_code": "PPL",
            "source_size_bytes": source.stat().st_size,
            "source_mtime": source.stat().st_mtime,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "tensor_key_count": 912,
            "clip_contributor": False,
            "clip_tensor_count": 0,
            "model_family": "Flux",
            "lora_type": "Flux (UNet double+single blocks)",
            "rank": None,
            "has_block_weights": True,
            "block_layout": "flux_unet_57",
            "block_count": 57,
            "block_weights": weights,
            "raw_block_strengths": raw,
        },
        "safety": {
            "database_open_mode": "SQLite URI mode=ro plus PRAGMA query_only=ON",
            "writes_database": False,
            "runs_full_indexer": False,
            "discovers_library_files": False,
            "opens_only_diagnostic_target": True,
            "assigns_stable_ids": False,
            "deletes_rows": False,
            "contains_apply_mode": False,
        },
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    return payload


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    root = tmp_path / "loras"
    source = _source(root)
    db = _db(tmp_path / "lora.db")
    return root, db, _artifact(source)


def test_dry_run_preview_changes_nothing(tmp_path: Path) -> None:
    root, db, artifact = _fixture(tmp_path)
    before = db.read_bytes()

    preview = build_preview(artifact, db_path=db, library_root=root)

    assert preview.artifact_sha256 == artifact["artifact_sha256"]
    assert preview.target["planned_stable_id"] == STABLE_ID
    assert preview.summary()["block_rows_to_insert"] == 57
    assert db.read_bytes() == before


def test_apply_inserts_one_lora_and_57_blocks_with_verified_backup(tmp_path: Path) -> None:
    root, db, artifact = _fixture(tmp_path)
    backup_dir = tmp_path / "backups"
    preview = build_preview(artifact, db_path=db, library_root=root)

    result = apply_artifact(
        preview,
        db_path=db,
        library_root=root,
        backup_dir=backup_dir,
        expected_artifact_sha256=str(artifact["artifact_sha256"]),
    )

    backup = Path(result["backup_path"])
    assert backup.is_file()
    assert result["lora_rows_inserted"] == 1
    assert result["block_rows_inserted"] == 57
    assert result["blocked_candidates_untouched"] == 2

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM lora WHERE stable_id = ?", (STABLE_ID,)).fetchone()
    assert row is not None
    assert row["file_path"] == DB_FILE_PATH
    assert row["has_block_weights"] == 1
    assert row["block_layout"] == "flux_unet_57"
    blocks = conn.execute(
        "SELECT * FROM lora_block_weights WHERE lora_id = ? ORDER BY block_index",
        (row["id"],),
    ).fetchall()
    conn.close()

    assert len(blocks) == 57
    assert [block["block_index"] for block in blocks] == list(range(57))
    assert all(block["stable_id"] == STABLE_ID for block in blocks)


def test_apply_rejects_wrong_digest_before_backup(tmp_path: Path) -> None:
    root, db, artifact = _fixture(tmp_path)
    backup_dir = tmp_path / "backups"
    preview = build_preview(artifact, db_path=db, library_root=root)

    with pytest.raises(ControlledFluxApplyError, match="Artifact digest mismatch"):
        apply_artifact(
            preview,
            db_path=db,
            library_root=root,
            backup_dir=backup_dir,
            expected_artifact_sha256="0" * 64,
        )

    assert not backup_dir.exists()


def test_apply_rejects_source_drift_before_backup(tmp_path: Path) -> None:
    root, db, artifact = _fixture(tmp_path)
    backup_dir = tmp_path / "backups"
    preview = build_preview(artifact, db_path=db, library_root=root)
    preview.source_path.write_bytes(b"source-changed-after-seal")

    with pytest.raises(ControlledFluxApplyError, match="source SHA-256 mismatch"):
        apply_artifact(
            preview,
            db_path=db,
            library_root=root,
            backup_dir=backup_dir,
            expected_artifact_sha256=str(artifact["artifact_sha256"]),
        )

    assert not backup_dir.exists()


def test_preview_rejects_stable_id_collision(tmp_path: Path) -> None:
    root, db, artifact = _fixture(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute(
        """
        INSERT INTO lora (
            file_path, filename, base_model_code, category_code,
            has_block_weights, clip_contributor, clip_tensor_count,
            last_modified, created_at, updated_at, stable_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "/loras/existing.safetensors",
            "existing.safetensors",
            "FLX",
            "PPL",
            0,
            0,
            -1,
            0.0,
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
            STABLE_ID,
        ),
    )
    conn.commit()
    conn.close()

    with pytest.raises(ControlledFluxApplyError, match="Planned stable ID already exists"):
        build_preview(artifact, db_path=db, library_root=root)


def test_mid_transaction_block_failure_rolls_back_lora_insert(tmp_path: Path) -> None:
    root, db, artifact = _fixture(tmp_path)
    backup_dir = tmp_path / "backups"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TRIGGER fail_phase89i_block
        BEFORE INSERT ON lora_block_weights
        WHEN NEW.block_index = 10
        BEGIN
            SELECT RAISE(ABORT, 'forced block failure');
        END;
        """
    )
    conn.commit()
    conn.close()

    preview = build_preview(artifact, db_path=db, library_root=root)

    with pytest.raises(sqlite3.IntegrityError, match="forced block failure"):
        apply_artifact(
            preview,
            db_path=db,
            library_root=root,
            backup_dir=backup_dir,
            expected_artifact_sha256=str(artifact["artifact_sha256"]),
        )

    assert list(backup_dir.glob("*.db"))
    conn = sqlite3.connect(db)
    lora_count = conn.execute("SELECT COUNT(1) FROM lora").fetchone()[0]
    block_count = conn.execute("SELECT COUNT(1) FROM lora_block_weights").fetchone()[0]
    conn.close()
    assert lora_count == 0
    assert block_count == 0
