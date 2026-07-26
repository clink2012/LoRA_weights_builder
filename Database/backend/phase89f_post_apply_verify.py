from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from phase89e_metadata_reconcile import build_execution_preview, load_plan


class VerificationError(RuntimeError):
    pass


def _open_read_only(path: str | os.PathLike[str]) -> sqlite3.Connection:
    resolved = Path(path).expanduser().resolve(strict=True)
    uri_path = quote(resolved.as_posix(), safe="/:")
    conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _normalise(value: Any) -> str:
    return str(value or "").strip()


def _db_file_path(db_root: str, relative: str) -> str:
    root = str(db_root or "").replace("\\", "/").rstrip("/")
    rel = PurePosixPath(relative).as_posix().lstrip("/")
    return f"{root}/{rel}" if root else rel


def _integrity_check(conn: sqlite3.Connection, label: str) -> str:
    row = conn.execute("PRAGMA integrity_check").fetchone()
    result = str(row[0] if row else "")
    if result.lower() != "ok":
        raise VerificationError(f"{label} integrity_check failed: {result or 'no result'}")
    return result


def _row_by_id(conn: sqlite3.Connection, row_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM lora WHERE id = ?", (row_id,)).fetchone()
    if row is None:
        raise VerificationError(f"Expected lora row is missing: {row_id}")
    return row


def _assert_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise VerificationError(f"{message}: expected {expected!r}, found {actual!r}")


def verify_post_apply(
    *,
    plan_path: str | os.PathLike[str],
    current_db_path: str | os.PathLike[str],
    backup_db_path: str | os.PathLike[str],
    db_path_root: str = "/loras",
) -> dict[str, Any]:
    plan = load_plan(plan_path)
    preview = build_execution_preview(plan)

    current = _open_read_only(current_db_path)
    backup = _open_read_only(backup_db_path)
    try:
        _integrity_check(current, "current DB")
        _integrity_check(backup, "backup DB")

        current_count = int(current.execute("SELECT COUNT(1) FROM lora").fetchone()[0])
        backup_count = int(backup.execute("SELECT COUNT(1) FROM lora").fetchone()[0])

        backup_ids = {int(row[0]) for row in backup.execute("SELECT id FROM lora")}
        current_ids = {int(row[0]) for row in current.execute("SELECT id FROM lora")}
        missing_ids = sorted(backup_ids - current_ids)
        if missing_ids:
            raise VerificationError(f"Original DB rows were deleted: {missing_ids[:20]}")

        expected_delta = len(preview.inserts)
        _assert_equal(current_count - backup_count, expected_delta, "lora row-count delta")

        verified_inserts = 0
        for item in preview.inserts:
            stable_id = str(item["planned_stable_id"]).upper()
            relative = str(item["relative_path"])
            expected_path = _db_file_path(db_path_root, relative)
            row = current.execute("SELECT * FROM lora WHERE stable_id = ?", (stable_id,)).fetchone()
            if row is None:
                raise VerificationError(f"Inserted row is missing: {stable_id}")
            if backup.execute("SELECT 1 FROM lora WHERE stable_id = ?", (stable_id,)).fetchone() is not None:
                raise VerificationError(f"Inserted stable ID already existed in backup: {stable_id}")
            _assert_equal(row["file_path"], expected_path, f"insert {stable_id} file_path")
            _assert_equal(row["filename"], item.get("filename") or PurePosixPath(relative).name, f"insert {stable_id} filename")
            _assert_equal(row["base_model_code"], str(item.get("base_model_code") or "").upper(), f"insert {stable_id} base_model_code")
            _assert_equal(row["category_code"], str(item.get("category_code") or "").upper(), f"insert {stable_id} category_code")
            _assert_equal(int(row["has_block_weights"] or 0), 0, f"insert {stable_id} has_block_weights")
            _assert_equal(row["block_layout"], None, f"insert {stable_id} block_layout")
            _assert_equal(int(row["clip_tensor_count"]), -1, f"insert {stable_id} clip_tensor_count")
            if current.execute("SELECT 1 FROM lora_block_weights WHERE lora_id = ?", (row["id"],)).fetchone() is not None:
                raise VerificationError(f"Metadata-only insert unexpectedly has block rows: {stable_id}")
            verified_inserts += 1

        verified_backfills = 0
        for item in preview.backfills:
            row_id = int(item["row_id"])
            before = _row_by_id(backup, row_id)
            after = _row_by_id(current, row_id)
            _assert_equal(after["file_path"], before["file_path"], f"backfill row {row_id} file_path")
            for field, transition in (item.get("changed_fields") or {}).items():
                _assert_equal(before[field], transition.get("from"), f"backfill row {row_id} backup {field}")
                _assert_equal(after[field], transition.get("to"), f"backfill row {row_id} current {field}")
            verified_backfills += 1

        verified_ids = 0
        for item in preview.id_assignments:
            row_id = int(item["row_id"])
            before = _row_by_id(backup, row_id)
            after = _row_by_id(current, row_id)
            if _normalise(before["stable_id"]):
                raise VerificationError(f"Backup row {row_id} already had stable_id {before['stable_id']}")
            _assert_equal(after["stable_id"], str(item["planned_stable_id"]).upper(), f"row {row_id} stable_id")
            verified_ids += 1

        verified_excluded_prefix = 0
        for item in preview.excluded_id_prefix_backfills:
            row_id = int(item["row_id"])
            before = _row_by_id(backup, row_id)
            after = _row_by_id(current, row_id)
            for field in ("file_path", "base_model_code", "category_name", "category_code", "stable_id"):
                _assert_equal(after[field], before[field], f"excluded prefix row {row_id} {field}")
            verified_excluded_prefix += 1

        relocation_items = [
            *(plan.get("same_family_relocations") or []),
            *(plan.get("cross_family_reclassifications") or []),
        ]
        verified_relocations = 0
        for item in relocation_items:
            row_id = int(item["row_id"])
            before = _row_by_id(backup, row_id)
            after = _row_by_id(current, row_id)
            for field in ("file_path", "base_model_name", "base_model_code", "category_name", "category_code", "stable_id"):
                _assert_equal(after[field], before[field], f"excluded relocation row {row_id} {field}")
            verified_relocations += 1

        duplicate_stable_ids = [
            dict(row)
            for row in current.execute(
                """
                SELECT stable_id, COUNT(1) AS count
                FROM lora
                WHERE stable_id IS NOT NULL AND TRIM(stable_id) <> ''
                GROUP BY stable_id
                HAVING COUNT(1) > 1
                ORDER BY stable_id
                """
            )
        ]
        if duplicate_stable_ids:
            raise VerificationError(f"Duplicate stable IDs detected: {duplicate_stable_ids[:20]}")

        backup_block_rows = int(backup.execute("SELECT COUNT(1) FROM lora_block_weights").fetchone()[0])
        current_block_rows = int(current.execute("SELECT COUNT(1) FROM lora_block_weights").fetchone()[0])
        _assert_equal(current_block_rows, backup_block_rows, "lora_block_weights row count")

        return {
            "status": "verified",
            "plan_sha256": preview.plan_sha256,
            "backup_lora_rows": backup_count,
            "current_lora_rows": current_count,
            "row_delta": current_count - backup_count,
            "verified_metadata_inserts": verified_inserts,
            "verified_metadata_backfills": verified_backfills,
            "verified_existing_id_assignments": verified_ids,
            "verified_excluded_id_prefix_backfills": verified_excluded_prefix,
            "verified_excluded_relocations": verified_relocations,
            "duplicate_stable_ids": 0,
            "block_weight_row_delta": current_block_rows - backup_block_rows,
            "untouched_stale_rows_declared": preview.untouched_stale_rows,
            "untouched_legacy_rows_declared": preview.untouched_legacy_rows,
        }
    finally:
        current.close()
        backup.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a Phase 8.9e apply against its backup and plan")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--backup", required=True)
    parser.add_argument("--db-path-root", default="/loras")
    parser.add_argument("--json")
    args = parser.parse_args()

    result = verify_post_apply(
        plan_path=args.plan,
        current_db_path=args.db,
        backup_db_path=args.backup,
        db_path_root=args.db_path_root,
    )
    print("=== Phase 8.9f post-apply verification ===")
    for key, value in result.items():
        print(f"{key}: {value}")
    if args.json:
        output = Path(args.json).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"JSON verification written to: {output}")


if __name__ == "__main__":
    main()
