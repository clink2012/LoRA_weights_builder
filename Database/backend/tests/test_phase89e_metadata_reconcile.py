from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from phase89e_metadata_reconcile import (  # noqa: E402
    ReconcileError,
    apply_preview,
    build_execution_preview,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")


def _create_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
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
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO lora (
            id, file_path, filename,
            base_model_name, base_model_code,
            category_name, category_code,
            model_family, lora_type, rank,
            has_block_weights, block_layout,
            clip_contributor, clip_tensor_count,
            last_modified, created_at, updated_at, stable_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                1,
                "/loras/Flux.2-Klein/03 - Utils/existing-klein.safetensors",
                "existing-klein.safetensors",
                "Flux.2-Klein",
                None,
                "Utils",
                "UTL",
                None,
                None,
                None,
                0,
                None,
                0,
                -1,
                1.0,
                "2026-07-26T00:00:00+00:00",
                "2026-07-26T00:00:00+00:00",
                None,
            ),
            (
                2,
                "/loras/WAN2.1/T2V/04 - Action/existing-wan.safetensors",
                "existing-wan.safetensors",
                "WAN2.1",
                "W21",
                "04 - Action",
                "ACT",
                None,
                None,
                None,
                0,
                None,
                0,
                -1,
                1.0,
                "2026-07-26T00:00:00+00:00",
                "2026-07-26T00:00:00+00:00",
                "W21-ACT-001",
            ),
        ],
    )
    conn.commit()
    conn.close()


def _plan() -> dict:
    return {
        "audit_mode": "read-only",
        "summary": {
            "untouched_stale_current_family_rows": 668,
            "untouched_legacy_unmounted_rows": 114,
        },
        "safety": {
            "writes_database": False,
            "runs_indexer": False,
            "assigns_stable_ids": False,
        },
        "unresolved_relocations": [],
        "stable_id_groups_exhausted": [],
        "existing_stable_id_issues": [],
        "same_family_relocations": [{"row_id": 10}, {"row_id": 11}, {"row_id": 12}],
        "cross_family_reclassifications": [{"row_id": row_id} for row_id in range(20, 40)],
        "new_metadata_insert_candidates": [
            {
                "source_type": "new_metadata_insert",
                "relative_path": "Flux.2-Klein/03 - Utils/new-klein.safetensors",
                "filename": "new-klein.safetensors",
                "base_model_name": "Flux.2-Klein",
                "base_model_code": "F2K",
                "category_name": "Utils",
                "category_code": "UTL",
            },
            {
                "source_type": "new_metadata_insert",
                "relative_path": "WAN2.2/I2V/05 - Body/new-wan.safetensors",
                "filename": "new-wan.safetensors",
                "base_model_name": "WAN2.2",
                "base_model_code": "W22",
                "category_name": "Body",
                "category_code": "BDY",
            },
            {
                "source_type": "new_metadata_insert",
                "relative_path": "FLUX/01 - People/needs-scanner.safetensors",
                "filename": "needs-scanner.safetensors",
                "base_model_name": "Flux",
                "base_model_code": "FLX",
                "category_name": "People",
                "category_code": "PPL",
            },
        ],
        "mounted_metadata_backfill_candidates": [
            {
                "row_id": 1,
                "file_path": "/loras/Flux.2-Klein/03 - Utils/existing-klein.safetensors",
                "parsed_base_model_code": "F2K",
                "changed_fields": {"base_model_code": {"from": None, "to": "F2K"}},
            },
            {
                "row_id": 2,
                "file_path": "/loras/WAN2.1/T2V/04 - Action/existing-wan.safetensors",
                "parsed_base_model_code": "W21",
                "changed_fields": {"category_name": {"from": "04 - Action", "to": "Action"}},
            },
        ],
        "mounted_existing_rows_missing_stable_id": [
            {
                "source_type": "existing_mounted_row_missing_id",
                "row_id": 1,
                "file_path": "/loras/Flux.2-Klein/03 - Utils/existing-klein.safetensors",
                "relative_path": "Flux.2-Klein/03 - Utils/existing-klein.safetensors",
                "filename": "existing-klein.safetensors",
                "base_model_code": "F2K",
                "category_code": "UTL",
            }
        ],
        "planned_stable_ids": [
            {
                "source_type": "new_metadata_insert",
                "relative_path": "Flux.2-Klein/03 - Utils/new-klein.safetensors",
                "planned_stable_id": "F2K-UTL-002",
            },
            {
                "source_type": "new_metadata_insert",
                "relative_path": "WAN2.2/I2V/05 - Body/new-wan.safetensors",
                "planned_stable_id": "W22-BDY-001",
            },
            {
                "source_type": "new_metadata_insert",
                "relative_path": "FLUX/01 - People/needs-scanner.safetensors",
                "planned_stable_id": "FLX-PPL-001",
            },
            {
                "source_type": "existing_mounted_row_missing_id",
                "row_id": 1,
                "planned_stable_id": "F2K-UTL-001",
            },
        ],
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "loras"
    db = tmp_path / "lora_master.db"
    backups = tmp_path / "backups"
    _touch(root / "Flux.2-Klein" / "03 - Utils" / "new-klein.safetensors")
    _touch(root / "WAN2.2" / "I2V" / "05 - Body" / "new-wan.safetensors")
    _touch(root / "FLUX" / "01 - People" / "needs-scanner.safetensors")
    _create_db(db)
    return root, db, backups


def test_preview_excludes_scanned_families_and_all_relocations() -> None:
    preview = build_execution_preview(_plan())

    assert len(preview.inserts) == 2
    assert {item["base_model_code"] for item in preview.inserts} == {"F2K", "W22"}
    assert len(preview.excluded_scanned_inserts) == 1
    assert preview.excluded_scanned_inserts[0]["base_model_code"] == "FLX"
    assert preview.excluded_relocations == 23
    assert len(preview.backfills) == 2
    assert len(preview.id_assignments) == 1
    assert preview.untouched_stale_rows == 668
    assert preview.untouched_legacy_rows == 114


def test_wrong_digest_refuses_apply_before_backup(tmp_path: Path) -> None:
    root, db, backups = _fixture(tmp_path)
    preview = build_execution_preview(_plan())

    with pytest.raises(ReconcileError, match="Plan digest mismatch"):
        apply_preview(
            preview,
            db_path=db,
            library_root=root,
            db_path_root="/loras",
            backup_dir=backups,
            expected_plan_sha256="0" * 64,
        )

    assert not backups.exists()


def test_apply_creates_backup_and_only_applies_approved_scope(tmp_path: Path) -> None:
    root, db, backups = _fixture(tmp_path)
    preview = build_execution_preview(_plan())

    result = apply_preview(
        preview,
        db_path=db,
        library_root=root,
        db_path_root="/loras",
        backup_dir=backups,
        expected_plan_sha256=preview.plan_sha256,
    )

    assert result["metadata_inserts"] == 2
    assert result["metadata_backfills"] == 2
    assert result["existing_id_assignments"] == 1
    assert result["excluded_scanned_inserts"] == 1
    assert result["excluded_relocations"] == 23
    assert Path(result["backup_path"]).is_file()

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = [dict(row) for row in conn.execute("SELECT * FROM lora ORDER BY id")]
    conn.close()

    by_filename = {row["filename"]: row for row in rows}
    assert by_filename["existing-klein.safetensors"]["base_model_code"] == "F2K"
    assert by_filename["existing-klein.safetensors"]["stable_id"] == "F2K-UTL-001"
    assert by_filename["existing-wan.safetensors"]["category_name"] == "Action"
    assert by_filename["new-klein.safetensors"]["stable_id"] == "F2K-UTL-002"
    assert by_filename["new-klein.safetensors"]["clip_tensor_count"] == -1
    assert by_filename["new-wan.safetensors"]["stable_id"] == "W22-BDY-001"
    assert "needs-scanner.safetensors" not in by_filename
    assert len(rows) == 4


def test_compare_and_swap_guard_rolls_back_transaction(tmp_path: Path) -> None:
    root, db, backups = _fixture(tmp_path)
    preview = build_execution_preview(_plan())

    conn = sqlite3.connect(db)
    conn.execute("UPDATE lora SET category_name = 'Changed elsewhere' WHERE id = 2")
    conn.commit()
    before = conn.execute("SELECT id, base_model_code, category_name, stable_id FROM lora ORDER BY id").fetchall()
    conn.close()

    with pytest.raises(ReconcileError, match="changed since planning"):
        apply_preview(
            preview,
            db_path=db,
            library_root=root,
            db_path_root="/loras",
            backup_dir=backups,
            expected_plan_sha256=preview.plan_sha256,
        )

    conn = sqlite3.connect(db)
    after = conn.execute("SELECT id, base_model_code, category_name, stable_id FROM lora ORDER BY id").fetchall()
    conn.close()
    assert after == before
