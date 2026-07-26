from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

ALLOWED_INSERT_BASE_CODES = frozenset({"F2K", "ILL", "LTX", "PNY", "SDX", "W22", "ZIM"})
EXCLUDED_SCANNED_BASE_CODES = frozenset({"FLX", "FLK"})
ALLOWED_BACKFILL_FIELDS = frozenset({"base_model_code", "category_name"})
REQUIRED_LORA_COLUMNS = frozenset(
    {
        "id",
        "file_path",
        "filename",
        "base_model_name",
        "base_model_code",
        "category_name",
        "category_code",
        "model_family",
        "lora_type",
        "rank",
        "has_block_weights",
        "block_layout",
        "clip_contributor",
        "clip_tensor_count",
        "last_modified",
        "created_at",
        "updated_at",
        "stable_id",
    }
)


class ReconcileError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionPreview:
    plan_sha256: str
    inserts: tuple[dict[str, Any], ...]
    backfills: tuple[dict[str, Any], ...]
    id_assignments: tuple[dict[str, Any], ...]
    excluded_scanned_inserts: tuple[dict[str, Any], ...]
    excluded_relocations: int
    untouched_stale_rows: int
    untouched_legacy_rows: int

    def summary(self) -> dict[str, int | str]:
        return {
            "plan_sha256": self.plan_sha256,
            "metadata_inserts": len(self.inserts),
            "metadata_backfills": len(self.backfills),
            "existing_id_assignments": len(self.id_assignments),
            "excluded_scanned_inserts": len(self.excluded_scanned_inserts),
            "excluded_relocations": self.excluded_relocations,
            "untouched_stale_rows": self.untouched_stale_rows,
            "untouched_legacy_rows": self.untouched_legacy_rows,
        }


def load_plan(path: str | os.PathLike[str]) -> dict[str, Any]:
    plan_path = Path(path).expanduser().resolve(strict=True)
    with plan_path.open("r", encoding="utf-8") as handle:
        plan = json.load(handle)
    if not isinstance(plan, dict):
        raise ReconcileError("Plan JSON must contain an object at the top level")
    return plan


def canonical_plan_bytes(plan: Mapping[str, Any]) -> bytes:
    return json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def plan_sha256(plan: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_plan_bytes(plan)).hexdigest()


def _normalise_code(value: Any) -> str:
    return str(value or "").strip().upper()


def _planned_id_lookup(plan: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for item in plan.get("planned_stable_ids", []):
        if not isinstance(item, Mapping):
            continue
        source_type = str(item.get("source_type") or "")
        stable_id = str(item.get("planned_stable_id") or "").strip().upper()
        if not source_type or not stable_id:
            continue
        if source_type == "new_metadata_insert":
            key = str(item.get("relative_path") or "").casefold()
        elif source_type == "existing_mounted_row_missing_id":
            key = str(item.get("row_id") or "")
        else:
            continue
        lookup[(source_type, key)] = stable_id
    return lookup


def build_execution_preview(plan: Mapping[str, Any]) -> ExecutionPreview:
    if plan.get("audit_mode") != "read-only":
        raise ReconcileError("Phase 8.9e accepts only a Phase 8.9d read-only plan")

    safety = plan.get("safety") or {}
    if safety.get("writes_database") is not False or safety.get("runs_indexer") is not False:
        raise ReconcileError("Plan safety flags do not describe a read-only planner run")

    blockers = {
        "unresolved_relocations": plan.get("unresolved_relocations") or [],
        "stable_id_groups_exhausted": plan.get("stable_id_groups_exhausted") or [],
        "existing_stable_id_issues": plan.get("existing_stable_id_issues") or [],
    }
    active_blockers = {name: values for name, values in blockers.items() if values}
    if active_blockers:
        raise ReconcileError(f"Plan has unresolved blockers: {', '.join(sorted(active_blockers))}")

    ids = _planned_id_lookup(plan)
    inserts: list[dict[str, Any]] = []
    excluded_scanned: list[dict[str, Any]] = []
    for raw in plan.get("new_metadata_insert_candidates", []):
        item = dict(raw)
        code = _normalise_code(item.get("base_model_code"))
        if code in EXCLUDED_SCANNED_BASE_CODES:
            excluded_scanned.append(item)
            continue
        if code not in ALLOWED_INSERT_BASE_CODES:
            raise ReconcileError(f"Insert candidate uses unapproved base-model code: {code or 'NULL'}")
        relative = str(item.get("relative_path") or "")
        stable_id = ids.get(("new_metadata_insert", relative.casefold()))
        if not stable_id:
            raise ReconcileError(f"No planned stable ID for insert candidate: {relative}")
        item["planned_stable_id"] = stable_id
        inserts.append(item)

    backfills: list[dict[str, Any]] = []
    for raw in plan.get("mounted_metadata_backfill_candidates", []):
        item = dict(raw)
        changed_fields = item.get("changed_fields") or {}
        unexpected = sorted(set(changed_fields) - ALLOWED_BACKFILL_FIELDS)
        if unexpected:
            raise ReconcileError(
                f"Backfill row {item.get('row_id')} contains unapproved field(s): {', '.join(unexpected)}"
            )
        if changed_fields:
            backfills.append(item)

    id_assignments: list[dict[str, Any]] = []
    for raw in plan.get("mounted_existing_rows_missing_stable_id", []):
        item = dict(raw)
        row_key = str(item.get("row_id") or "")
        stable_id = ids.get(("existing_mounted_row_missing_id", row_key))
        if not stable_id:
            raise ReconcileError(f"No planned stable ID for existing row {row_key}")
        item["planned_stable_id"] = stable_id
        id_assignments.append(item)

    summary = plan.get("summary") or {}
    excluded_relocations = len(plan.get("same_family_relocations") or []) + len(
        plan.get("cross_family_reclassifications") or []
    )

    return ExecutionPreview(
        plan_sha256=plan_sha256(plan),
        inserts=tuple(inserts),
        backfills=tuple(backfills),
        id_assignments=tuple(id_assignments),
        excluded_scanned_inserts=tuple(excluded_scanned),
        excluded_relocations=excluded_relocations,
        untouched_stale_rows=int(summary.get("untouched_stale_current_family_rows") or 0),
        untouched_legacy_rows=int(summary.get("untouched_legacy_unmounted_rows") or 0),
    )


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _check_schema(conn: sqlite3.Connection) -> None:
    columns = _table_columns(conn, "lora")
    missing = sorted(REQUIRED_LORA_COLUMNS - columns)
    if missing:
        raise ReconcileError(f"lora table is missing required column(s): {', '.join(missing)}")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _create_backup(db_path: Path, backup_dir: Path, digest: str) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"{db_path.stem}.phase89e.{stamp}.{digest[:12]}.db"
    if backup_path.exists():
        raise ReconcileError(f"Backup path already exists: {backup_path}")

    source = sqlite3.connect(db_path)
    try:
        destination = sqlite3.connect(backup_path)
        try:
            source.backup(destination)
        finally:
            destination.close()
    finally:
        source.close()

    if not backup_path.is_file() or backup_path.stat().st_size == 0:
        raise ReconcileError("SQLite backup was not created correctly")
    return backup_path


def _same_db_value(actual: Any, expected: Any) -> bool:
    if actual is None and expected in (None, ""):
        return True
    return actual == expected


def _assert_stable_id_available(conn: sqlite3.Connection, stable_id: str, *, except_row_id: int | None = None) -> None:
    if except_row_id is None:
        row = conn.execute("SELECT id FROM lora WHERE stable_id = ?", (stable_id,)).fetchone()
    else:
        row = conn.execute(
            "SELECT id FROM lora WHERE stable_id = ? AND id <> ?",
            (stable_id, except_row_id),
        ).fetchone()
    if row is not None:
        raise ReconcileError(f"Stable ID is already in use: {stable_id}")


def _apply_backfills(conn: sqlite3.Connection, items: Iterable[Mapping[str, Any]], now: str) -> int:
    count = 0
    for item in items:
        row_id = int(item["row_id"])
        row = conn.execute("SELECT * FROM lora WHERE id = ?", (row_id,)).fetchone()
        if row is None:
            raise ReconcileError(f"Backfill row no longer exists: {row_id}")
        if str(row["file_path"]) != str(item.get("file_path")):
            raise ReconcileError(f"Backfill row {row_id} file_path changed since planning")

        assignments: list[str] = []
        values: list[Any] = []
        for field, transition in (item.get("changed_fields") or {}).items():
            if field not in ALLOWED_BACKFILL_FIELDS:
                raise ReconcileError(f"Backfill field is not approved: {field}")
            expected_old = transition.get("from")
            new_value = transition.get("to")
            if not _same_db_value(row[field], expected_old):
                raise ReconcileError(
                    f"Backfill row {row_id} field {field} changed since planning: "
                    f"expected {expected_old!r}, found {row[field]!r}"
                )
            assignments.append(f"{field} = ?")
            values.append(new_value)

        if not assignments:
            continue
        assignments.append("updated_at = ?")
        values.extend([now, row_id])
        conn.execute(f"UPDATE lora SET {', '.join(assignments)} WHERE id = ?", values)
        count += 1
    return count


def _apply_existing_ids(conn: sqlite3.Connection, items: Iterable[Mapping[str, Any]], now: str) -> int:
    count = 0
    for item in items:
        row_id = int(item["row_id"])
        planned_id = str(item["planned_stable_id"]).upper()
        row = conn.execute(
            "SELECT id, file_path, base_model_code, category_code, stable_id FROM lora WHERE id = ?",
            (row_id,),
        ).fetchone()
        if row is None:
            raise ReconcileError(f"ID-assignment row no longer exists: {row_id}")
        if str(row["file_path"]) != str(item.get("file_path")):
            raise ReconcileError(f"ID-assignment row {row_id} file_path changed since planning")
        if str(row["stable_id"] or "").strip():
            raise ReconcileError(f"ID-assignment row {row_id} already has stable_id {row['stable_id']}")
        if _normalise_code(row["base_model_code"]) != _normalise_code(item.get("base_model_code")):
            raise ReconcileError(f"ID-assignment row {row_id} base_model_code does not match the plan")
        if _normalise_code(row["category_code"]) != _normalise_code(item.get("category_code")):
            raise ReconcileError(f"ID-assignment row {row_id} category_code does not match the plan")
        _assert_stable_id_available(conn, planned_id, except_row_id=row_id)
        conn.execute(
            "UPDATE lora SET stable_id = ?, updated_at = ? WHERE id = ?",
            (planned_id, now, row_id),
        )
        count += 1
    return count


def _db_file_path(db_root: str, relative: str) -> str:
    root = str(db_root or "").replace("\\", "/").rstrip("/")
    rel = PurePosixPath(relative).as_posix().lstrip("/")
    return f"{root}/{rel}" if root else rel


def _apply_inserts(
    conn: sqlite3.Connection,
    items: Iterable[Mapping[str, Any]],
    *,
    library_root: Path,
    db_path_root: str,
    now: str,
) -> int:
    count = 0
    for item in items:
        relative = PurePosixPath(str(item["relative_path"])).as_posix()
        source_path = library_root.joinpath(*PurePosixPath(relative).parts)
        if not source_path.is_file():
            raise ReconcileError(f"Insert source file is missing: {source_path}")

        file_path = _db_file_path(db_path_root, relative)
        if conn.execute("SELECT id FROM lora WHERE file_path = ?", (file_path,)).fetchone() is not None:
            raise ReconcileError(f"Insert file_path already exists in DB: {file_path}")

        stable_id = str(item["planned_stable_id"]).upper()
        _assert_stable_id_available(conn, stable_id)
        stat = source_path.stat()
        conn.execute(
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
                item.get("filename") or PurePosixPath(relative).name,
                item.get("base_model_name"),
                _normalise_code(item.get("base_model_code")),
                item.get("category_name"),
                _normalise_code(item.get("category_code")),
                None,
                None,
                None,
                0,
                None,
                0,
                -1,
                float(stat.st_mtime),
                now,
                now,
                stable_id,
            ),
        )
        count += 1
    return count


def apply_preview(
    preview: ExecutionPreview,
    *,
    db_path: str | os.PathLike[str],
    library_root: str | os.PathLike[str],
    db_path_root: str,
    backup_dir: str | os.PathLike[str],
    expected_plan_sha256: str,
) -> dict[str, Any]:
    expected = str(expected_plan_sha256 or "").strip().lower()
    if expected != preview.plan_sha256:
        raise ReconcileError(
            f"Plan digest mismatch: expected argument {expected or 'EMPTY'}, actual {preview.plan_sha256}"
        )

    db = Path(db_path).expanduser().resolve(strict=True)
    root = Path(library_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ReconcileError(f"LoRA library root is not a directory: {root}")

    backup_path = _create_backup(db, Path(backup_dir).expanduser().resolve(), preview.plan_sha256)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")

    try:
        _check_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        now = _timestamp()
        backfilled = _apply_backfills(conn, preview.backfills, now)
        ids_assigned = _apply_existing_ids(conn, preview.id_assignments, now)
        inserted = _apply_inserts(
            conn,
            preview.inserts,
            library_root=root,
            db_path_root=db_path_root,
            now=now,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return {
        "backup_path": str(backup_path),
        "metadata_inserts": inserted,
        "metadata_backfills": backfilled,
        "existing_id_assignments": ids_assigned,
        "excluded_scanned_inserts": len(preview.excluded_scanned_inserts),
        "excluded_relocations": preview.excluded_relocations,
    }


def print_preview(preview: ExecutionPreview) -> None:
    summary = preview.summary()
    print("=== Phase 8.9e controlled metadata reconciliation ===")
    print("Mode                         : dry-run")
    print(f"Plan SHA-256                 : {summary['plan_sha256']}")
    print(f"Metadata inserts             : {summary['metadata_inserts']}")
    print(f"Metadata backfills           : {summary['metadata_backfills']}")
    print(f"Existing ID assignments      : {summary['existing_id_assignments']}")
    print(f"Excluded FLX/FLK inserts     : {summary['excluded_scanned_inserts']}")
    print(f"Excluded relocations         : {summary['excluded_relocations']}")
    print(f"Untouched stale rows         : {summary['untouched_stale_rows']}")
    print(f"Untouched legacy rows        : {summary['untouched_legacy_rows']}")
    print()
    print("No database changes were made.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guarded Phase 8.9e metadata reconciliation")
    parser.add_argument("--plan", required=True, help="Phase 8.9d JSON plan")
    parser.add_argument("--db", required=True, help="SQLite database path")
    parser.add_argument("--root", required=True, help="Mounted LoRA library root")
    parser.add_argument("--db-path-root", default="/loras", help="Path prefix stored in SQLite for new rows")
    parser.add_argument("--apply", action="store_true", help="Apply the approved metadata-only scope")
    parser.add_argument("--expected-plan-sha256", help="Required with --apply")
    parser.add_argument("--backup-dir", help="Required with --apply")
    return parser


def main() -> int:
    args = _parser().parse_args()
    plan = load_plan(args.plan)
    preview = build_execution_preview(plan)
    if not args.apply:
        print_preview(preview)
        return 0

    if not args.expected_plan_sha256:
        raise SystemExit("--expected-plan-sha256 is required with --apply")
    if not args.backup_dir:
        raise SystemExit("--backup-dir is required with --apply")

    result = apply_preview(
        preview,
        db_path=args.db,
        library_root=args.root,
        db_path_root=args.db_path_root,
        backup_dir=args.backup_dir,
        expected_plan_sha256=args.expected_plan_sha256,
    )
    print("=== Phase 8.9e apply complete ===")
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
