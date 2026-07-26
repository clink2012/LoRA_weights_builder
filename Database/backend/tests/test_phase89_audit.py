from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase89_audit import open_read_only_db, run_audit


def _create_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE lora (
            id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL,
            filename TEXT,
            base_model_name TEXT,
            base_model_code TEXT,
            stable_id TEXT,
            has_block_weights INTEGER NOT NULL DEFAULT 0,
            block_layout TEXT,
            model_family TEXT,
            lora_type TEXT
        );
        CREATE TABLE lora_block_weights (
            id INTEGER PRIMARY KEY,
            lora_id INTEGER NOT NULL,
            block_index INTEGER NOT NULL,
            weight REAL NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")


def test_audit_matches_windows_db_paths_to_posix_mount(tmp_path: Path) -> None:
    root = tmp_path / "loras"
    db = tmp_path / "lora_master.db"

    _touch(root / "FLUX" / "01 - People" / "one.safetensors")
    _touch(root / "SDXL" / "02 - Styles" / "two.safetensors")
    _touch(root / "Z-Image" / "03 - Utils" / "new.safetensors")
    _touch(root / "recipes" / "ignored.safetensors")
    _touch(root / "LoRA_Manager_Images" / "ignored-image.safetensors")

    _create_db(db)
    conn = sqlite3.connect(db)
    conn.executemany(
        """
        INSERT INTO lora (
            id, file_path, filename, base_model_name, base_model_code,
            stable_id, has_block_weights, block_layout
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (1, r"E:\models\loras\FLUX\01 - People\one.safetensors", "one.safetensors", "Flux", "FLX", "FLX-PPL-001", 1, "flux_transformer_38"),
            (2, r"E:\models\loras\SDXL\02 - Styles\two.safetensors", "two.safetensors", "SDXL", "SDX", None, 0, None),
            (3, r"E:\models\loras\SDXL\02 - Styles\deleted.safetensors", "deleted.safetensors", "SDXL", "SDX", "SDX-STL-001", 0, None),
            (4, r"E:\models\loras\LoRA_Manager_Images\ignored-image.safetensors", "ignored-image.safetensors", None, None, None, 0, None),
        ],
    )
    conn.execute(
        "INSERT INTO lora_block_weights (lora_id, block_index, weight) VALUES (1, 0, 1.0)"
    )
    conn.commit()
    conn.close()

    report = run_audit(root_dir=root, db_path=db)

    assert report["audit_mode"] == "read-only"
    assert report["top_level_folders"] == ["FLUX", "SDXL", "Z-Image"]
    assert report["summary"]["mounted_safetensors"] == 3
    assert report["summary"]["db_rows"] == 4
    assert report["summary"]["with_stable_id"] == 2
    assert report["summary"]["without_stable_id"] == 2
    assert report["summary"]["stale_db_rows"] == 1
    assert report["summary"]["mounted_files_missing_from_db"] == 1
    assert report["summary"]["ignored_db_rows"] == 1

    matrix = {row["folder"]: row for row in report["support_matrix"]}
    assert matrix["FLUX"]["support_status"] == "scanned"
    assert matrix["FLUX"]["classifications"] == {"scanned": 1}
    assert matrix["SDXL"]["support_status"] == "metadata-only"
    assert matrix["SDXL"]["stale_db_rows"] == 1
    assert matrix["Z-Image"]["support_status"] == "unindexed"
    assert matrix["Z-Image"]["missing_from_db"] == 1
    assert report["mounted_files_missing_from_db"]["all"] == ["Z-Image/03 - Utils/new.safetensors"]
    assert report["stale_db_paths"]["count"] == 1
    json.dumps(report)


def test_open_read_only_db_rejects_writes(tmp_path: Path) -> None:
    db = tmp_path / "read_only.db"
    _create_db(db)

    conn = open_read_only_db(db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO lora (id, file_path) VALUES (1, 'x')")
    finally:
        conn.close()


def test_flag_and_block_row_inconsistencies_are_explicit(tmp_path: Path) -> None:
    root = tmp_path / "loras"
    db = tmp_path / "lora_master.db"
    _touch(root / "FLUX" / "03 - Utils" / "flag-only.safetensors")
    _touch(root / "FLUX" / "03 - Utils" / "rows-only.safetensors")
    _create_db(db)

    conn = sqlite3.connect(db)
    conn.executemany(
        """
        INSERT INTO lora (id, file_path, filename, base_model_name, base_model_code, stable_id, has_block_weights, block_layout)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (1, r"E:\models\loras\FLUX\03 - Utils\flag-only.safetensors", "flag-only.safetensors", "Flux", "FLX", "FLX-UTL-001", 1, "flux_transformer_38"),
            (2, r"E:\models\loras\FLUX\03 - Utils\rows-only.safetensors", "rows-only.safetensors", "Flux", "FLX", "FLX-UTL-002", 0, "flux_transformer_38"),
        ],
    )
    conn.execute("INSERT INTO lora_block_weights (lora_id, block_index, weight) VALUES (2, 0, 0.5)")
    conn.commit()
    conn.close()

    report = run_audit(root_dir=root, db_path=db)
    assert report["summary"]["classifications"] == {
        "blocks_without_flag": 1,
        "flagged_missing_blocks": 1,
    }
    assert report["support_matrix"][0]["support_status"] == "inconsistent"
