from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from phase89g_targeted_flux_analysis import _safe_source_path, _sha256_file
from phase89k_flux2_layout_support import (
    EXPECTED_GLOBAL_MODULES,
    canonical_sha256,
    load_json_object,
    verify_artifact_digest,
)


class Phase89mVerificationError(RuntimeError):
    pass


EXPECTED_ARTIFACT_SHA256 = (
    "adad9f9c3eb65bf0b2c0774e3c0c1508c43603ed3774c804f5ef49123d6a48df"
)
EXPECTED_CURRENT_DB_SHA256 = (
    "6526505261ed62c79c433217161716e6d0bb9b286fb266867f9e6c87b1fa2357"
)
EXPECTED_BACKUP_DB_SHA256 = (
    "d732b2739cd2df278b104e325d17481413152d89776d8f1abdc59637bc86c79e"
)
EXPECTED_BACKUP_NAME = (
    "lora_master.phase89l.20260729T221823Z.adad9f9c3eb6.db"
)
EXPECTED_SOURCE_SHA256 = (
    "c60c9a5de39da23b3b4f4dca48e3511faa1fe5a4987d4acbb0a04643a9a65be7"
)
EXPECTED_STABLE_ID = "FLX-STL-263"
DAMAGED_STABLE_ID = "FLX-BDY-071"
EXPECTED_LAYOUT = "flux2_transformer_56"
EXPECTED_TARGET_LORA_ID = 2834
EXPECTED_CURRENT_LORA_ROWS = 2834
EXPECTED_BACKUP_LORA_ROWS = 2833
EXPECTED_CURRENT_BLOCK_ROWS = 4348
EXPECTED_BACKUP_BLOCK_ROWS = 4292
EXPECTED_BLOCK_COUNT = 56
EXPECTED_TENSOR_KEY_COUNT = 276
EXPECTED_RANK = 16
EXPECTED_BLOCK_MODULE_COUNT = 128
EXPECTED_BLOCK_TENSOR_COUNT = 256
EXPECTED_GLOBAL_MODULE_COUNT = 10
EXPECTED_GLOBAL_TENSOR_COUNT = 20


def _file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_read_only(path: str | os.PathLike[str]) -> sqlite3.Connection:
    resolved = Path(path).expanduser().resolve(strict=True)
    uri_path = quote(resolved.as_posix(), safe="/:")
    conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise Phase89mVerificationError(
            f"{label} mismatch: expected {expected!r}, found {actual!r}"
        )


def _integrity_check(conn: sqlite3.Connection, label: str) -> str:
    row = conn.execute("PRAGMA integrity_check").fetchone()
    result = str(row[0] if row else "")
    if result.casefold() != "ok":
        raise Phase89mVerificationError(
            f"{label} integrity_check failed: {result or 'no result'}"
        )
    return result


def _validate_numeric_vector(
    values: Any,
    *,
    label: str,
    count: int,
    unit_range: bool,
) -> list[float]:
    if not isinstance(values, list):
        raise Phase89mVerificationError(f"{label} must be a JSON list")
    if len(values) != count:
        raise Phase89mVerificationError(
            f"{label} count mismatch: expected {count}, found {len(values)}"
        )

    result: list[float] = []
    for index, raw in enumerate(values):
        value = float(raw)
        if not math.isfinite(value):
            raise Phase89mVerificationError(
                f"{label}[{index}] is not finite: {raw!r}"
            )
        if unit_range and not 0.0 <= value <= 1.0:
            raise Phase89mVerificationError(
                f"{label}[{index}] is outside [0, 1]: {value}"
            )
        if not unit_range and value < 0.0:
            raise Phase89mVerificationError(
                f"{label}[{index}] is negative: {value}"
            )
        result.append(value)
    return result


def _validate_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_artifact_sha256: str,
    expected_source_sha256: str,
) -> tuple[str, dict[str, Any]]:
    _require_equal(artifact.get("phase"), "8.9k", "artifact phase")
    _require_equal(
        artifact.get("mode"),
        "read-only sealed targeted Flux 2 artifact",
        "artifact mode",
    )

    digest = verify_artifact_digest(artifact)
    _require_equal(digest, expected_artifact_sha256, "artifact SHA-256")

    summary = artifact.get("summary") or {}
    _require_equal(summary.get("targets_analysed"), 1, "targets_analysed")
    _require_equal(
        summary.get("ready_for_later_controlled_apply"),
        1,
        "ready_for_later_controlled_apply",
    )
    _require_equal(summary.get("block_rows"), EXPECTED_BLOCK_COUNT, "block_rows")
    _require_equal(
        summary.get("global_projection_tensors"),
        EXPECTED_GLOBAL_TENSOR_COUNT,
        "global_projection_tensors",
    )
    _require_equal(
        summary.get("damaged_flux_targets_untouched"),
        1,
        "damaged_flux_targets_untouched",
    )

    safety = artifact.get("safety") or {}
    for field in (
        "writes_database",
        "creates_backup",
        "contains_apply_mode",
        "runs_full_indexer",
        "discovers_library_files",
        "assigns_stable_ids",
        "deletes_rows",
        "touches_damaged_flux_target",
    ):
        _require_equal(safety.get(field), False, f"artifact safety {field}")

    target = dict(artifact.get("target") or {})
    exact_checks = {
        "planned_stable_id": EXPECTED_STABLE_ID,
        "base_model_name": "Flux 2",
        "base_model_code": "FLX",
        "category_name": "Styles",
        "category_code": "STL",
        "source_sha256": expected_source_sha256,
        "tensor_key_count": EXPECTED_TENSOR_KEY_COUNT,
        "model_family": "Flux 2",
        "lora_type": "Flux 2 (PEFT double+single blocks)",
        "rank": EXPECTED_RANK,
        "rank_values": [EXPECTED_RANK],
        "block_layout": EXPECTED_LAYOUT,
        "block_count": EXPECTED_BLOCK_COUNT,
        "observed_double_indices": list(range(8)),
        "observed_single_indices": list(range(48)),
        "missing_double_indices": [],
        "missing_single_indices": [],
        "extra_double_indices": [],
        "extra_single_indices": [],
        "block_module_count": EXPECTED_BLOCK_MODULE_COUNT,
        "block_tensor_count": EXPECTED_BLOCK_TENSOR_COUNT,
        "global_module_count": EXPECTED_GLOBAL_MODULE_COUNT,
        "global_tensor_count": EXPECTED_GLOBAL_TENSOR_COUNT,
        "global_modules": list(EXPECTED_GLOBAL_MODULES),
        "unmatched_tensor_count": 0,
        "blockers": [],
        "ready_for_sealing": True,
    }
    for field, expected in exact_checks.items():
        actual = target.get(field)
        if field == "planned_stable_id":
            actual = str(actual or "").strip().upper()
        if field == "source_sha256":
            actual = str(actual or "").strip().lower()
        _require_equal(actual, expected, f"artifact target {field}")

    for field in (
        "relative_path",
        "db_file_path",
        "filename",
        "source_size_bytes",
        "source_mtime",
    ):
        if target.get(field) in (None, ""):
            raise Phase89mVerificationError(
                f"artifact target is missing required field {field}"
            )

    target["block_weights"] = _validate_numeric_vector(
        target.get("block_weights"),
        label="block_weights",
        count=EXPECTED_BLOCK_COUNT,
        unit_range=True,
    )
    target["raw_block_strengths"] = _validate_numeric_vector(
        target.get("raw_block_strengths"),
        label="raw_block_strengths",
        count=EXPECTED_BLOCK_COUNT,
        unit_range=False,
    )
    if max(target["block_weights"], default=0.0) != 1.0:
        raise Phase89mVerificationError(
            "block_weights must contain a strongest value of exactly 1.0"
        )
    if any(value <= 0.0 for value in target["block_weights"]):
        raise Phase89mVerificationError(
            "all 56 Flux 2 block weights must be greater than zero"
        )
    return digest, target


def _verify_source(
    library_root: str | os.PathLike[str],
    target: Mapping[str, Any],
    *,
    expected_source_sha256: str,
) -> Path:
    root = Path(library_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise Phase89mVerificationError(
            f"LoRA library root is not a directory: {root}"
        )
    source = _safe_source_path(root, str(target["relative_path"]))
    _require_equal(_sha256_file(source), expected_source_sha256, "source SHA-256")
    _require_equal(
        source.stat().st_size,
        int(target["source_size_bytes"]),
        "source size",
    )
    expected_mtime = float(target["source_mtime"])
    if abs(source.stat().st_mtime - expected_mtime) > 1e-6:
        raise Phase89mVerificationError(
            "source mtime mismatch: "
            f"expected {expected_mtime!r}, found {source.stat().st_mtime!r}"
        )
    return source


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    columns = [str(row[1]) for row in rows]
    if not columns:
        raise Phase89mVerificationError(f"missing table or columns: {table}")
    return columns


def _verify_backup_rows_preserved(
    backup: sqlite3.Connection,
    current: sqlite3.Connection,
    table: str,
) -> int:
    backup_columns = _table_columns(backup, table)
    current_columns = _table_columns(current, table)
    _require_equal(current_columns, backup_columns, f"{table} column list")

    backup_rows = backup.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
    current_rows = current.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
    current_by_id = {int(row["id"]): row for row in current_rows}

    for before in backup_rows:
        row_id = int(before["id"])
        after = current_by_id.get(row_id)
        if after is None:
            raise Phase89mVerificationError(
                f"pre-existing {table} row was deleted: id={row_id}"
            )
        for column in backup_columns:
            if after[column] != before[column]:
                raise Phase89mVerificationError(
                    f"pre-existing {table} row changed: "
                    f"id={row_id}, column={column}, "
                    f"before={before[column]!r}, after={after[column]!r}"
                )
    return len(backup_rows)


def _duplicate_stable_ids(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in conn.execute(
            """
            SELECT stable_id, COUNT(*) AS count
            FROM lora
            WHERE stable_id IS NOT NULL
              AND TRIM(stable_id) <> ''
            GROUP BY stable_id
            HAVING COUNT(*) > 1
            ORDER BY stable_id
            """
        )
    ]


def _verify_target_row(
    current: sqlite3.Connection,
    backup: sqlite3.Connection,
    target: Mapping[str, Any],
    *,
    expected_target_lora_id: int,
) -> tuple[int, int]:
    backup_count = int(
        backup.execute(
            "SELECT COUNT(*) FROM lora WHERE stable_id = ? OR file_path = ?",
            (EXPECTED_STABLE_ID, target["db_file_path"]),
        ).fetchone()[0]
    )
    _require_equal(backup_count, 0, "backup target row count")

    rows = current.execute(
        "SELECT * FROM lora WHERE stable_id = ? OR file_path = ?",
        (EXPECTED_STABLE_ID, target["db_file_path"]),
    ).fetchall()
    _require_equal(len(rows), 1, "current target row count")
    row = rows[0]
    lora_id = int(row["id"])
    _require_equal(lora_id, expected_target_lora_id, "target lora id")

    exact_fields = {
        "file_path": target["db_file_path"],
        "filename": target["filename"],
        "base_model_name": "Flux 2",
        "base_model_code": "FLX",
        "category_name": "Styles",
        "category_code": "STL",
        "model_family": "Flux 2",
        "lora_type": "Flux 2 (PEFT double+single blocks)",
        "rank": EXPECTED_RANK,
        "has_block_weights": 1,
        "block_layout": EXPECTED_LAYOUT,
        "clip_contributor": 0,
        "clip_tensor_count": 0,
        "stable_id": EXPECTED_STABLE_ID,
    }
    for field, expected in exact_fields.items():
        _require_equal(row[field], expected, f"target lora {field}")

    _require_equal(
        float(row["last_modified"]),
        float(target["source_mtime"]),
        "target lora last_modified",
    )
    if not str(row["created_at"] or "").strip():
        raise Phase89mVerificationError("target lora created_at is empty")
    _require_equal(row["updated_at"], row["created_at"], "target timestamps")

    blocks = current.execute(
        """
        SELECT *
        FROM lora_block_weights
        WHERE lora_id = ?
        ORDER BY block_index
        """,
        (lora_id,),
    ).fetchall()
    _require_equal(len(blocks), EXPECTED_BLOCK_COUNT, "target block row count")

    for index, block in enumerate(blocks):
        _require_equal(block["stable_id"], EXPECTED_STABLE_ID, f"block {index} stable_id")
        _require_equal(int(block["block_index"]), index, f"block {index} index")
        if not math.isclose(
            float(block["weight"]),
            float(target["block_weights"][index]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise Phase89mVerificationError(
                f"block {index} weight mismatch"
            )
        if not math.isclose(
            float(block["raw_strength"]),
            float(target["raw_block_strengths"][index]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise Phase89mVerificationError(
                f"block {index} raw_strength mismatch"
            )
    return lora_id, len(blocks)


def verify_phase89m_closeout(
    *,
    artifact_path: str | os.PathLike[str],
    current_db_path: str | os.PathLike[str],
    backup_db_path: str | os.PathLike[str],
    library_root: str | os.PathLike[str],
    expected_artifact_sha256: str = EXPECTED_ARTIFACT_SHA256,
    expected_current_db_sha256: str = EXPECTED_CURRENT_DB_SHA256,
    expected_backup_db_sha256: str = EXPECTED_BACKUP_DB_SHA256,
    expected_backup_name: str = EXPECTED_BACKUP_NAME,
    expected_source_sha256: str = EXPECTED_SOURCE_SHA256,
    expected_target_lora_id: int = EXPECTED_TARGET_LORA_ID,
    expected_current_lora_rows: int = EXPECTED_CURRENT_LORA_ROWS,
    expected_backup_lora_rows: int = EXPECTED_BACKUP_LORA_ROWS,
    expected_current_block_rows: int = EXPECTED_CURRENT_BLOCK_ROWS,
    expected_backup_block_rows: int = EXPECTED_BACKUP_BLOCK_ROWS,
) -> dict[str, Any]:
    artifact_file = Path(artifact_path).expanduser().resolve(strict=True)
    current_file = Path(current_db_path).expanduser().resolve(strict=True)
    backup_file = Path(backup_db_path).expanduser().resolve(strict=True)

    _require_equal(backup_file.name, expected_backup_name, "backup filename")

    current_sha256 = _file_sha256(current_file)
    backup_sha256 = _file_sha256(backup_file)
    _require_equal(current_sha256, expected_current_db_sha256, "current DB SHA-256")
    _require_equal(backup_sha256, expected_backup_db_sha256, "backup DB SHA-256")

    artifact = load_json_object(artifact_file, "Phase 8.9k artifact")
    artifact_sha256, target = _validate_artifact(
        artifact,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_source_sha256=expected_source_sha256,
    )
    source = _verify_source(
        library_root,
        target,
        expected_source_sha256=expected_source_sha256,
    )

    current = _open_read_only(current_file)
    backup = _open_read_only(backup_file)
    try:
        current_integrity = _integrity_check(current, "current DB")
        backup_integrity = _integrity_check(backup, "backup DB")

        current_lora_rows = int(
            current.execute("SELECT COUNT(*) FROM lora").fetchone()[0]
        )
        backup_lora_rows = int(
            backup.execute("SELECT COUNT(*) FROM lora").fetchone()[0]
        )
        current_block_rows = int(
            current.execute("SELECT COUNT(*) FROM lora_block_weights").fetchone()[0]
        )
        backup_block_rows = int(
            backup.execute("SELECT COUNT(*) FROM lora_block_weights").fetchone()[0]
        )

        _require_equal(
            current_lora_rows,
            expected_current_lora_rows,
            "current lora row count",
        )
        _require_equal(
            backup_lora_rows,
            expected_backup_lora_rows,
            "backup lora row count",
        )
        _require_equal(
            current_block_rows,
            expected_current_block_rows,
            "current block row count",
        )
        _require_equal(
            backup_block_rows,
            expected_backup_block_rows,
            "backup block row count",
        )
        _require_equal(current_lora_rows - backup_lora_rows, 1, "lora row delta")
        _require_equal(
            current_block_rows - backup_block_rows,
            EXPECTED_BLOCK_COUNT,
            "block row delta",
        )

        preserved_lora_rows = _verify_backup_rows_preserved(
            backup,
            current,
            "lora",
        )
        preserved_block_rows = _verify_backup_rows_preserved(
            backup,
            current,
            "lora_block_weights",
        )

        target_lora_id, target_block_rows = _verify_target_row(
            current,
            backup,
            target,
            expected_target_lora_id=expected_target_lora_id,
        )

        current_ids = {int(row[0]) for row in current.execute("SELECT id FROM lora")}
        backup_ids = {int(row[0]) for row in backup.execute("SELECT id FROM lora")}
        _require_equal(
            current_ids - backup_ids,
            {target_lora_id},
            "new lora id set",
        )

        duplicates = _duplicate_stable_ids(current)
        if duplicates:
            raise Phase89mVerificationError(
                f"duplicate stable IDs detected: {duplicates[:20]}"
            )

        with_stable_id = int(
            current.execute(
                """
                SELECT COUNT(*)
                FROM lora
                WHERE stable_id IS NOT NULL
                  AND TRIM(stable_id) <> ''
                """
            ).fetchone()[0]
        )
        _require_equal(
            with_stable_id,
            expected_current_lora_rows,
            "rows with stable IDs",
        )

        damaged_current = int(
            current.execute(
                "SELECT COUNT(*) FROM lora WHERE stable_id = ?",
                (DAMAGED_STABLE_ID,),
            ).fetchone()[0]
        )
        damaged_backup = int(
            backup.execute(
                "SELECT COUNT(*) FROM lora WHERE stable_id = ?",
                (DAMAGED_STABLE_ID,),
            ).fetchone()[0]
        )
        _require_equal(damaged_current, 0, "damaged target current rows")
        _require_equal(damaged_backup, 0, "damaged target backup rows")

        orphan_blocks = int(
            current.execute(
                """
                SELECT COUNT(*)
                FROM lora_block_weights AS bw
                LEFT JOIN lora AS l ON l.id = bw.lora_id
                WHERE l.id IS NULL
                """
            ).fetchone()[0]
        )
        _require_equal(orphan_blocks, 0, "orphan block rows")
    finally:
        current.close()
        backup.close()

    result: dict[str, Any] = {
        "phase": "8.9m",
        "mode": "read-only post-apply verification and closeout",
        "status": "verified",
        "artifact_sha256": artifact_sha256,
        "current_db_sha256": current_sha256,
        "backup_db_sha256": backup_sha256,
        "backup_path": str(backup_file),
        "source_sha256": expected_source_sha256,
        "source_path": str(source),
        "database": {
            "current_integrity": current_integrity,
            "backup_integrity": backup_integrity,
            "current_lora_rows": current_lora_rows,
            "backup_lora_rows": backup_lora_rows,
            "lora_row_delta": current_lora_rows - backup_lora_rows,
            "current_block_rows": current_block_rows,
            "backup_block_rows": backup_block_rows,
            "block_row_delta": current_block_rows - backup_block_rows,
            "rows_with_stable_ids": with_stable_id,
            "duplicate_stable_ids": 0,
            "orphan_block_rows": orphan_blocks,
            "preserved_preexisting_lora_rows": preserved_lora_rows,
            "preserved_preexisting_block_rows": preserved_block_rows,
        },
        "target": {
            "stable_id": EXPECTED_STABLE_ID,
            "lora_id": target_lora_id,
            "model_family": "Flux 2",
            "block_layout": EXPECTED_LAYOUT,
            "verified_block_rows": target_block_rows,
            "global_projection_tensors_preserved_in_artifact": (
                EXPECTED_GLOBAL_TENSOR_COUNT
            ),
        },
        "quarantine": {
            "stable_id": DAMAGED_STABLE_ID,
            "current_rows": damaged_current,
            "backup_rows": damaged_backup,
            "status": "absent and untouched",
        },
        "safety": {
            "database_open_mode": "SQLite URI mode=ro plus PRAGMA query_only=ON",
            "writes_database": False,
            "creates_backup": False,
            "runs_full_indexer": False,
            "runs_reindex": False,
            "discovers_library_files": False,
            "opens_only_approved_source": True,
            "touches_damaged_flux_target": False,
        },
    }
    result["verification_sha256"] = canonical_sha256(result)
    return result


def verify_report_digest(report: Mapping[str, Any]) -> str:
    stored = str(report.get("verification_sha256") or "")
    unsigned = dict(report)
    unsigned.pop("verification_sha256", None)
    calculated = canonical_sha256(unsigned)
    if stored != calculated:
        raise Phase89mVerificationError(
            f"verification report digest mismatch: "
            f"stored {stored}, calculated {calculated}"
        )
    return calculated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only Phase 8.9m post-apply verification and closeout"
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--backup", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--json")
    args = parser.parse_args()

    result = verify_phase89m_closeout(
        artifact_path=args.artifact,
        current_db_path=args.db,
        backup_db_path=args.backup,
        library_root=args.root,
    )
    verify_report_digest(result)

    print("=== Phase 8.9m post-apply verification and closeout ===")
    print(f"Status                       : {result['status']}")
    print(f"Verification SHA-256          : {result['verification_sha256']}")
    print(f"Current DB SHA-256            : {result['current_db_sha256']}")
    print(f"Backup DB SHA-256             : {result['backup_db_sha256']}")
    print(f"Current LoRA rows             : {result['database']['current_lora_rows']}")
    print(f"Backup LoRA rows              : {result['database']['backup_lora_rows']}")
    print(f"Current block rows            : {result['database']['current_block_rows']}")
    print(f"Backup block rows             : {result['database']['backup_block_rows']}")
    print(
        "Preserved existing LoRAs      : "
        f"{result['database']['preserved_preexisting_lora_rows']}"
    )
    print(
        "Preserved existing blocks     : "
        f"{result['database']['preserved_preexisting_block_rows']}"
    )
    print(f"Verified target ID            : {result['target']['stable_id']}")
    print(f"Verified target row ID        : {result['target']['lora_id']}")
    print(f"Verified target blocks        : {result['target']['verified_block_rows']}")
    print(f"Remaining quarantine          : {result['quarantine']['stable_id']}")
    print(f"Quarantine status             : {result['quarantine']['status']}")
    print("No database changes were made.")

    if args.json:
        output = Path(args.json).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        print(f"JSON verification written to: {output}")


if __name__ == "__main__":
    main()
