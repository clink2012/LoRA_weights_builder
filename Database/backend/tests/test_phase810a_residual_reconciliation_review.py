from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from phase810a_residual_reconciliation_review import (
    ResidualReviewError,
    build_residual_review,
    verify_review_digest,
)
from phase89e_metadata_reconcile import plan_sha256
from phase89k_flux2_layout_support import canonical_sha256


META_RELATIVE = "Flux.2-Klein/01 - People/meta.safetensors"
PPL_RELATIVE = "FLUX/01 - People/ang3l4wh1t3-f1.safetensors"
STL_RELATIVE = "FLUX/02 - Styles/aidmaMJ61Flux.2v0.5.safetensors"
BDY_RELATIVE = "FLUX/05 - Body/Eye_Detail_Flux_Lora_-_Inpainting-421d.safetensors"
SAME_OLD = "/legacy/WAN2.2/01 - Action/move.safetensors"
SAME_NEW = "WAN2.2/02 - Body/move.safetensors"
CROSS_OLD = "/legacy/WAN2.2/01 - People/person.safetensors"
CROSS_NEW = "WAN2.1/01 - People/person.safetensors"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE lora (
            id INTEGER PRIMARY KEY,
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
            last_modified REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT '2026-01-01T00:00:00+00:00',
            updated_at TEXT NOT NULL DEFAULT '2026-01-01T00:00:00+00:00',
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


def _insert_lora(
    conn: sqlite3.Connection,
    *,
    row_id: int,
    file_path: str,
    stable_id: str,
    base_code: str,
    category_name: str,
    category_code: str,
    has_blocks: int = 0,
    block_layout: str | None = None,
    model_family: str | None = None,
    lora_type: str | None = None,
    rank: int | None = None,
    clip_tensor_count: int = -1,
) -> None:
    conn.execute(
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
        (
            row_id,
            file_path,
            Path(file_path.replace("\\", "/")).name,
            model_family or base_code,
            base_code,
            category_name,
            category_code,
            model_family,
            lora_type,
            rank,
            has_blocks,
            block_layout,
            0,
            clip_tensor_count,
            1.0,
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
            stable_id,
        ),
    )


def _insert_blocks(
    conn: sqlite3.Connection,
    *,
    lora_id: int,
    stable_id: str,
    count: int,
) -> None:
    conn.executemany(
        """
        INSERT INTO lora_block_weights (
            lora_id, stable_id, block_index, weight, raw_strength
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [
            (lora_id, stable_id, index, (index + 1) / count, float(index + 1))
            for index in range(count)
        ],
    )


def _database(path: Path) -> Path:
    conn = sqlite3.connect(path)
    _schema(conn)

    _insert_lora(
        conn,
        row_id=1,
        file_path=f"/loras/{META_RELATIVE}",
        stable_id="F2K-PPL-001",
        base_code="F2K",
        category_name="People",
        category_code="PPL",
        clip_tensor_count=-1,
    )
    _insert_lora(
        conn,
        row_id=2,
        file_path="/loras/Z-Image/01 - Action/backfill.safetensors",
        stable_id="ZIM-ACT-001",
        base_code="ZIM",
        category_name="Action",
        category_code="ACT",
    )
    _insert_lora(
        conn,
        row_id=3,
        file_path="/loras/Z-Image/05 - Body/conflict.safetensors",
        stable_id="UNK-BDY-001",
        base_code="UNK",
        category_name="Body",
        category_code="BDY",
    )
    _insert_lora(
        conn,
        row_id=4,
        file_path="/loras/LTXV2/01 - People/id.safetensors",
        stable_id="LTX-PPL-001",
        base_code="LTX",
        category_name="People",
        category_code="PPL",
    )
    _insert_lora(
        conn,
        row_id=5,
        file_path=SAME_OLD,
        stable_id="W22-ACT-001",
        base_code="W22",
        category_name="Action",
        category_code="ACT",
    )
    _insert_lora(
        conn,
        row_id=6,
        file_path=CROSS_OLD,
        stable_id="W22-PPL-001",
        base_code="W22",
        category_name="People",
        category_code="PPL",
    )
    _insert_lora(
        conn,
        row_id=7,
        file_path=f"/loras/{PPL_RELATIVE}",
        stable_id="FLX-PPL-207",
        base_code="FLX",
        category_name="People",
        category_code="PPL",
        has_blocks=1,
        block_layout="flux_unet_57",
        model_family="Flux",
        lora_type="Flux (UNet double+single blocks)",
        clip_tensor_count=0,
    )
    _insert_blocks(conn, lora_id=7, stable_id="FLX-PPL-207", count=57)
    _insert_lora(
        conn,
        row_id=8,
        file_path=f"/loras/{STL_RELATIVE}",
        stable_id="FLX-STL-263",
        base_code="FLX",
        category_name="Styles",
        category_code="STL",
        has_blocks=1,
        block_layout="flux2_transformer_56",
        model_family="Flux 2",
        lora_type="Flux 2 (PEFT double+single blocks)",
        rank=16,
        clip_tensor_count=0,
    )
    _insert_blocks(conn, lora_id=8, stable_id="FLX-STL-263", count=56)

    conn.commit()
    conn.close()
    return path


def _plan() -> dict[str, object]:
    scanned = [
        {
            "relative_path": PPL_RELATIVE,
            "filename": Path(PPL_RELATIVE).name,
            "base_model_name": "Flux",
            "base_model_code": "FLX",
            "category_name": "People",
            "category_code": "PPL",
            "source_type": "new_metadata_insert",
        },
        {
            "relative_path": STL_RELATIVE,
            "filename": Path(STL_RELATIVE).name,
            "base_model_name": "Flux",
            "base_model_code": "FLX",
            "category_name": "Styles",
            "category_code": "STL",
            "source_type": "new_metadata_insert",
        },
        {
            "relative_path": BDY_RELATIVE,
            "filename": Path(BDY_RELATIVE).name,
            "base_model_name": "Flux",
            "base_model_code": "FLX",
            "category_name": "Body",
            "category_code": "BDY",
            "source_type": "new_metadata_insert",
        },
    ]
    metadata_insert = {
        "relative_path": META_RELATIVE,
        "filename": Path(META_RELATIVE).name,
        "base_model_name": "Flux.2-Klein",
        "base_model_code": "F2K",
        "category_name": "People",
        "category_code": "PPL",
        "source_type": "new_metadata_insert",
    }
    return {
        "audit_mode": "read-only",
        "summary": {
            "untouched_stale_current_family_rows": 2,
            "untouched_legacy_unmounted_rows": 1,
        },
        "new_metadata_insert_candidates": [metadata_insert, *scanned],
        "mounted_metadata_backfill_candidates": [
            {
                "row_id": 2,
                "stable_id": "ZIM-ACT-001",
                "file_path": "/loras/Z-Image/01 - Action/backfill.safetensors",
                "relative_path": "Z-Image/01 - Action/backfill.safetensors",
                "changed_fields": {
                    "category_name": {"from": None, "to": "Action"}
                },
                "parsed_base_model_code": "ZIM",
                "parsed_category_code": "ACT",
            },
            {
                "row_id": 3,
                "stable_id": "UNK-BDY-001",
                "file_path": "/loras/Z-Image/05 - Body/conflict.safetensors",
                "relative_path": "Z-Image/05 - Body/conflict.safetensors",
                "changed_fields": {
                    "base_model_code": {"from": "UNK", "to": "ZIM"}
                },
                "parsed_base_model_code": "ZIM",
                "parsed_category_code": "BDY",
            },
        ],
        "mounted_existing_rows_missing_stable_id": [
            {
                "source_type": "existing_mounted_row_missing_id",
                "row_id": 4,
                "file_path": "/loras/LTXV2/01 - People/id.safetensors",
                "relative_path": "LTXV2/01 - People/id.safetensors",
                "filename": "id.safetensors",
                "base_model_code": "LTX",
                "category_code": "PPL",
            }
        ],
        "planned_stable_ids": [
            {**metadata_insert, "planned_stable_id": "F2K-PPL-001"},
            {**scanned[0], "planned_stable_id": "FLX-PPL-207"},
            {**scanned[1], "planned_stable_id": "FLX-STL-263"},
            {**scanned[2], "planned_stable_id": "FLX-BDY-071"},
            {
                "source_type": "existing_mounted_row_missing_id",
                "row_id": 4,
                "file_path": "/loras/LTXV2/01 - People/id.safetensors",
                "relative_path": "LTXV2/01 - People/id.safetensors",
                "filename": "id.safetensors",
                "base_model_code": "LTX",
                "category_code": "PPL",
                "planned_stable_id": "LTX-PPL-001",
            },
        ],
        "same_family_relocations": [
            {
                "row_id": 5,
                "stable_id": "W22-ACT-001",
                "old_path": SAME_OLD,
                "new_path": SAME_NEW,
                "from_family": "WAN2.2",
                "to_family": "WAN2.2",
                "new_base_model_code": "W22",
                "new_category_code": "BDY",
                "identity_evidence": "unique case-insensitive exact filename only",
                "requires_content_hash_verification": True,
                "review_class": "same-family path relocation",
                "stable_id_policy": "preserve existing stable_id",
            }
        ],
        "cross_family_reclassifications": [
            {
                "row_id": 6,
                "stable_id": "W22-PPL-001",
                "old_path": CROSS_OLD,
                "new_path": CROSS_NEW,
                "from_family": "WAN2.2",
                "to_family": "WAN2.1",
                "new_base_model_code": "W21",
                "new_category_code": "PPL",
                "identity_evidence": "unique case-insensitive exact filename only",
                "requires_content_hash_verification": True,
                "review_class": "cross-family reclassification",
                "stable_id_policy": "manual decision required",
            }
        ],
        "unresolved_relocations": [],
        "stable_id_groups_exhausted": [],
        "existing_stable_id_issues": [],
        "safety": {
            "writes_database": False,
            "runs_indexer": False,
        },
    }


def _phase89m_report(db_sha: str, lora_rows: int, block_rows: int) -> dict[str, object]:
    report: dict[str, object] = {
        "phase": "8.9m",
        "mode": "read-only post-apply verification and closeout",
        "status": "verified",
        "current_db_sha256": db_sha,
        "database": {
            "current_lora_rows": lora_rows,
            "current_block_rows": block_rows,
            "duplicate_stable_ids": 0,
            "orphan_block_rows": 0,
        },
        "quarantine": {
            "stable_id": "FLX-BDY-071",
            "current_rows": 0,
            "status": "absent and untouched",
        },
    }
    report["verification_sha256"] = canonical_sha256(report)
    return report


def _run(
    db: Path,
    plan: dict[str, object],
    *,
    lora_rows: int = 8,
    block_rows: int = 113,
) -> dict[str, object]:
    report = _phase89m_report(_sha256(db), lora_rows, block_rows)
    return build_residual_review(
        plan,
        report,
        db_path=db,
        expected_plan_sha256=plan_sha256(plan),
        expected_phase89m_sha256=str(report["verification_sha256"]),
        expected_db_sha256=_sha256(db),
        expected_lora_rows=lora_rows,
        expected_block_rows=block_rows,
        expected_metadata_inserts=1,
        expected_metadata_backfills=1,
        expected_id_assignments=1,
        expected_scanned_candidates=3,
        expected_prefix_conflicts=1,
        expected_same_family_relocations=1,
        expected_cross_family_relocations=1,
        expected_stale_hold_rows=2,
        expected_legacy_hold_rows=1,
    )


def test_builds_verified_residual_review_without_changing_db(tmp_path: Path) -> None:
    db = _database(tmp_path / "lora.db")
    plan = _plan()
    before = db.read_bytes()

    result = _run(db, plan)

    assert result["status"] == "verified"
    assert result["completed_work_verified"] == {
        "verified_metadata_inserts": 1,
        "verified_metadata_backfills": 1,
        "verified_existing_id_assignments": 1,
        "completed_scanned_targets": 2,
    }
    assert result["residual_counts"] == {
        "damaged_scanned_candidates": 1,
        "id_prefix_conflicts": 1,
        "same_family_relocations": 1,
        "cross_family_reclassifications": 1,
        "stale_current_family_rows_declared_hold": 2,
        "legacy_unmounted_rows_declared_hold": 1,
    }
    assert verify_review_digest(result) == result["review_sha256"]
    assert db.read_bytes() == before


def test_rejects_wrong_plan_digest(tmp_path: Path) -> None:
    db = _database(tmp_path / "lora.db")
    plan = _plan()
    report = _phase89m_report(_sha256(db), 8, 113)

    with pytest.raises(ResidualReviewError, match="Phase 8.9d plan SHA-256"):
        build_residual_review(
            plan,
            report,
            db_path=db,
            expected_plan_sha256="0" * 64,
            expected_phase89m_sha256=str(report["verification_sha256"]),
            expected_db_sha256=_sha256(db),
        )


def test_rejects_tampered_phase89m_report(tmp_path: Path) -> None:
    db = _database(tmp_path / "lora.db")
    plan = _plan()
    report = _phase89m_report(_sha256(db), 8, 113)
    report["status"] = "tampered"

    with pytest.raises(Exception, match="digest mismatch"):
        build_residual_review(
            plan,
            report,
            db_path=db,
            expected_plan_sha256=plan_sha256(plan),
            expected_phase89m_sha256=str(report["verification_sha256"]),
            expected_db_sha256=_sha256(db),
        )


def test_rejects_completed_metadata_drift(tmp_path: Path) -> None:
    db = _database(tmp_path / "lora.db")
    conn = sqlite3.connect(db)
    conn.execute("UPDATE lora SET category_code = 'BAD' WHERE id = 1")
    conn.commit()
    conn.close()

    with pytest.raises(ResidualReviewError, match="F2K-PPL-001 category_code"):
        _run(db, _plan())


def test_rejects_prefix_conflict_row_change(tmp_path: Path) -> None:
    db = _database(tmp_path / "lora.db")
    conn = sqlite3.connect(db)
    conn.execute("UPDATE lora SET base_model_code = 'ZIM' WHERE id = 3")
    conn.commit()
    conn.close()

    with pytest.raises(ResidualReviewError, match="unchanged base_model_code"):
        _run(db, _plan())


def test_rejects_relocation_destination_collision(tmp_path: Path) -> None:
    db = _database(tmp_path / "lora.db")
    conn = sqlite3.connect(db)
    _insert_lora(
        conn,
        row_id=9,
        file_path=f"/loras/{SAME_NEW}",
        stable_id="W22-BDY-999",
        base_code="W22",
        category_name="Body",
        category_code="BDY",
    )
    conn.commit()
    conn.close()

    with pytest.raises(ResidualReviewError, match="new-path DB collisions"):
        _run(db, _plan(), lora_rows=9)


def test_rejects_damaged_target_database_leak(tmp_path: Path) -> None:
    db = _database(tmp_path / "lora.db")
    conn = sqlite3.connect(db)
    _insert_lora(
        conn,
        row_id=9,
        file_path=f"/loras/{BDY_RELATIVE}",
        stable_id="FLX-BDY-071",
        base_code="FLX",
        category_name="Body",
        category_code="BDY",
    )
    conn.commit()
    conn.close()

    with pytest.raises(ResidualReviewError, match="FLX-BDY-071 quarantine rows"):
        _run(db, _plan(), lora_rows=9)


def test_review_digest_detects_tampering(tmp_path: Path) -> None:
    db = _database(tmp_path / "lora.db")
    result = _run(db, _plan())
    result["status"] = "changed"

    with pytest.raises(ResidualReviewError, match="residual review digest mismatch"):
        verify_review_digest(result)
