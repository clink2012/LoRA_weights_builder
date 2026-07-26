from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from phase89d_index_plan import build_index_plan  # noqa: E402


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")


def _create_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE lora (
            id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL,
            filename TEXT NOT NULL,
            base_model_name TEXT,
            base_model_code TEXT,
            category_name TEXT,
            category_code TEXT,
            stable_id TEXT,
            has_block_weights INTEGER NOT NULL DEFAULT 0,
            block_layout TEXT,
            model_family TEXT,
            lora_type TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def _seed_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "loras"
    db = tmp_path / "lora_master.db"

    _touch(root / "WAN2.1" / "T2V" / "04 - Action" / "moved.safetensors")
    (root / "WAN2.2").mkdir(parents=True, exist_ok=True)
    _touch(root / "Z-Image" / "05 - Body" / "same-family.safetensors")
    _touch(root / "LTXV2" / "02 - Styles" / "existing-ltx.safetensors")
    _touch(root / "LTXV2" / "02 - Styles" / "new-ltx.safetensors")
    _touch(root / "Flux.2-Klein" / "03 - Utils" / "existing-klein.safetensors")
    _touch(root / "FutureModel" / "03 - Utils" / "unmapped.safetensors")

    _create_db(db)
    conn = sqlite3.connect(db)
    conn.executemany(
        """
        INSERT INTO lora (
            id, file_path, filename, base_model_name, base_model_code,
            category_name, category_code, stable_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                1,
                r"E:\models\loras\WAN2.2\T2V\04 - Action\moved.safetensors",
                "moved.safetensors",
                "WAN2.2",
                "W22",
                "Action",
                "ACT",
                "W22-ACT-001",
            ),
            (
                2,
                r"E:\models\loras\Z-Image\04 - Action\same-family.safetensors",
                "same-family.safetensors",
                "Z-Image",
                "ZIM",
                "Action",
                "ACT",
                "ZIM-ACT-001",
            ),
            (
                3,
                str(root / "Flux.2-Klein" / "03 - Utils" / "existing-klein.safetensors"),
                "existing-klein.safetensors",
                "Flux.2-Klein",
                None,
                "Utils",
                "UTL",
                None,
            ),
            (
                4,
                str(root / "LTXV2" / "02 - Styles" / "existing-ltx.safetensors"),
                "existing-ltx.safetensors",
                "LTXV2",
                "LTX",
                "Styles",
                "STL",
                "LTX-STL-001",
            ),
        ],
    )
    conn.commit()
    conn.close()
    return root, db


def test_plan_separates_relocations_inserts_and_cross_family_policy(tmp_path: Path) -> None:
    root, db = _seed_fixture(tmp_path)

    plan = build_index_plan(root_dir=root, db_path=db)
    summary = plan["summary"]

    assert plan["audit_mode"] == "read-only"
    assert summary["same_family_relocation_candidates"] == 1
    assert summary["cross_family_reclassification_candidates"] == 1
    assert summary["new_metadata_insert_candidates"] == 1
    assert summary["mounted_metadata_backfill_candidates"] == 1
    assert summary["unparseable_missing_files"] == 1
    assert summary["mounted_existing_rows_missing_stable_id"] == 1
    assert summary["planned_stable_ids"] == 2
    assert summary["untouched_stale_current_family_rows"] == 0

    cross_family = plan["cross_family_reclassifications"][0]
    assert cross_family["stable_id"] == "W22-ACT-001"
    assert cross_family["new_base_model_code"] == "W21"
    assert "manual decision required" in cross_family["stable_id_policy"]

    same_family = plan["same_family_relocations"][0]
    assert same_family["stable_id"] == "ZIM-ACT-001"
    assert same_family["stable_id_policy"] == "preserve existing stable_id"

    backfill = plan["mounted_metadata_backfill_candidates"][0]
    assert backfill["row_id"] == 3
    assert backfill["parsed_base_model_code"] == "F2K"
    assert backfill["changed_fields"]["base_model_code"] == {"from": None, "to": "F2K"}

    planned_ids = {
        (item["source_type"], item["filename"]): item["planned_stable_id"]
        for item in plan["planned_stable_ids"]
    }
    assert planned_ids[("existing_mounted_row_missing_id", "existing-klein.safetensors")] == "F2K-UTL-001"
    assert planned_ids[("new_metadata_insert", "new-ltx.safetensors")] == "LTX-STL-002"


def test_plan_does_not_modify_database(tmp_path: Path) -> None:
    root, db = _seed_fixture(tmp_path)

    with sqlite3.connect(db) as conn:
        before = conn.execute(
            "SELECT id, file_path, base_model_code, stable_id FROM lora ORDER BY id"
        ).fetchall()

    plan = build_index_plan(root_dir=root, db_path=db)
    assert plan["safety"]["writes_database"] is False
    assert plan["safety"]["assigns_stable_ids"] is False

    with sqlite3.connect(db) as conn:
        after = conn.execute(
            "SELECT id, file_path, base_model_code, stable_id FROM lora ORDER BY id"
        ).fetchall()
    assert after == before
