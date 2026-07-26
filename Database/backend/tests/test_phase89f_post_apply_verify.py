from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from phase89e_metadata_reconcile import apply_preview, build_execution_preview  # noqa: E402
from phase89f_post_apply_verify import VerificationError, verify_post_apply  # noqa: E402


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")


def _create_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
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
        );
        CREATE TABLE lora_block_weights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lora_id INTEGER NOT NULL,
            stable_id TEXT,
            block_index INTEGER NOT NULL,
            weight REAL NOT NULL,
            raw_strength REAL
        );
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
            (
                3,
                "/loras/Z-Image/05 - Body/excluded-zim.safetensors",
                "excluded-zim.safetensors",
                "Z-Image",
                None,
                "Body",
                "BDY",
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
                "UNK-BDY-001",
            ),
            (
                4,
                "/loras/WAN2.2/I2V/04 - Action/relocated.safetensors",
                "relocated.safetensors",
                "WAN2.2",
                "W22",
                "Action",
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
                "W22-ACT-001",
            ),
        ],
    )
    conn.commit()
    conn.close()


def _plan() -> dict:
    return {
        "audit_mode": "read-only",
        "summary": {
            "untouched_stale_current_family_rows": 10,
            "untouched_legacy_unmounted_rows": 2,
        },
        "safety": {"writes_database": False, "runs_indexer": False, "assigns_stable_ids": False},
        "unresolved_relocations": [],
        "stable_id_groups_exhausted": [],
        "existing_stable_id_issues": [],
        "same_family_relocations": [
            {
                "row_id": 4,
                "old_path": "/loras/WAN2.2/I2V/04 - Action/relocated.safetensors",
                "new_path": "WAN2.2/I2V/05 - Body/relocated.safetensors",
            }
        ],
        "cross_family_reclassifications": [],
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
                "stable_id": None,
                "file_path": "/loras/Flux.2-Klein/03 - Utils/existing-klein.safetensors",
                "parsed_base_model_code": "F2K",
                "changed_fields": {"base_model_code": {"from": None, "to": "F2K"}},
            },
            {
                "row_id": 2,
                "stable_id": "W21-ACT-001",
                "file_path": "/loras/WAN2.1/T2V/04 - Action/existing-wan.safetensors",
                "parsed_base_model_code": "W21",
                "changed_fields": {"category_name": {"from": "04 - Action", "to": "Action"}},
            },
            {
                "row_id": 3,
                "stable_id": "UNK-BDY-001",
                "file_path": "/loras/Z-Image/05 - Body/excluded-zim.safetensors",
                "parsed_base_model_code": "ZIM",
                "changed_fields": {"base_model_code": {"from": None, "to": "ZIM"}},
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


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    root = tmp_path / "loras"
    db = tmp_path / "lora_master.db"
    backup_dir = tmp_path / "backups"
    plan_path = tmp_path / "plan.json"
    _touch(root / "Flux.2-Klein" / "03 - Utils" / "new-klein.safetensors")
    _touch(root / "FLUX" / "01 - People" / "needs-scanner.safetensors")
    _create_db(db)
    plan = _plan()
    plan_path.write_text(__import__("json").dumps(plan), encoding="utf-8")
    preview = build_execution_preview(plan)
    result = apply_preview(
        preview,
        db_path=db,
        library_root=root,
        db_path_root="/loras",
        backup_dir=backup_dir,
        expected_plan_sha256=preview.plan_sha256,
    )
    return plan_path, db, Path(result["backup_path"]), plan


def test_verifier_confirms_exact_apply_scope(tmp_path: Path) -> None:
    plan_path, db, backup, _ = _fixture(tmp_path)

    result = verify_post_apply(
        plan_path=plan_path,
        current_db_path=db,
        backup_db_path=backup,
        db_path_root="/loras",
    )

    assert result["status"] == "verified"
    assert result["row_delta"] == 1
    assert result["verified_metadata_inserts"] == 1
    assert result["verified_metadata_backfills"] == 2
    assert result["verified_existing_id_assignments"] == 1
    assert result["verified_excluded_id_prefix_backfills"] == 1
    assert result["verified_excluded_relocations"] == 1
    assert result["duplicate_stable_ids"] == 0
    assert result["block_weight_row_delta"] == 0


def test_verifier_detects_change_to_excluded_relocation(tmp_path: Path) -> None:
    plan_path, db, backup, _ = _fixture(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE lora SET category_name = 'Body' WHERE id = 4")
    conn.commit()
    conn.close()

    with pytest.raises(VerificationError, match="excluded relocation row 4 category_name"):
        verify_post_apply(
            plan_path=plan_path,
            current_db_path=db,
            backup_db_path=backup,
            db_path_root="/loras",
        )


def test_verifier_detects_deleted_original_row(tmp_path: Path) -> None:
    plan_path, db, backup, _ = _fixture(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM lora WHERE id = 4")
    conn.commit()
    conn.close()

    with pytest.raises(VerificationError, match="Original DB rows were deleted"):
        verify_post_apply(
            plan_path=plan_path,
            current_db_path=db,
            backup_db_path=backup,
            db_path_root="/loras",
        )
