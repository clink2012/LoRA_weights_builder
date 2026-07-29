from __future__ import annotations

import argparse
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from phase89g_targeted_flux_analysis import (
    _open_read_only,
    _safe_source_path,
    _sha256_file,
)
from phase89k_flux2_layout_support import (
    EXPECTED_GLOBAL_MODULES,
    load_json_object,
    verify_artifact_digest,
)


class GuardedFlux2ApplyError(RuntimeError):
    pass


EXPECTED_ARTIFACT_SHA256 = (
    "adad9f9c3eb65bf0b2c0774e3c0c1508c43603ed3774c804f5ef49123d6a48df"
)
EXPECTED_PHASE89J_SHA256 = (
    "7c886a07e87fa36081645d34bd578001420e65a380c6543b17c9b9ee1fb8dc48"
)
EXPECTED_SOURCE_SHA256 = (
    "c60c9a5de39da23b3b4f4dca48e3511faa1fe5a4987d4acbb0a04643a9a65be7"
)
EXPECTED_STABLE_ID = "FLX-STL-263"
DAMAGED_STABLE_ID = "FLX-BDY-071"
EXPECTED_LAYOUT = "flux2_transformer_56"
EXPECTED_BLOCK_COUNT = 56
EXPECTED_TENSOR_KEY_COUNT = 276
EXPECTED_RANK = 16
EXPECTED_BLOCK_MODULE_COUNT = 128
EXPECTED_BLOCK_TENSOR_COUNT = 256
EXPECTED_GLOBAL_MODULE_COUNT = 10
EXPECTED_GLOBAL_TENSOR_COUNT = 20

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
REQUIRED_BLOCK_COLUMNS = frozenset(
    {
        "id",
        "lora_id",
        "stable_id",
        "block_index",
        "weight",
        "raw_strength",
    }
)


@dataclass(frozen=True)
class ApplyPreview:
    artifact_sha256: str
    source_path: Path
    target: dict[str, Any]
    lora_rows_before: int
    block_rows_before: int

    def summary(self) -> dict[str, Any]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "stable_id": self.target["planned_stable_id"],
            "db_file_path": self.target["db_file_path"],
            "source_sha256": self.target["source_sha256"],
            "model_family": self.target["model_family"],
            "rank": self.target["rank"],
            "block_layout": self.target["block_layout"],
            "lora_rows_to_insert": 1,
            "block_rows_to_insert": len(self.target["block_weights"]),
            "global_projection_tensors": self.target["global_tensor_count"],
            "lora_rows_before": self.lora_rows_before,
            "block_rows_before": self.block_rows_before,
        }


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _check_schema(conn: sqlite3.Connection) -> None:
    lora_missing = sorted(REQUIRED_LORA_COLUMNS - _table_columns(conn, "lora"))
    if lora_missing:
        raise GuardedFlux2ApplyError(
            "lora table is missing required column(s): " + ", ".join(lora_missing)
        )

    block_missing = sorted(
        REQUIRED_BLOCK_COLUMNS - _table_columns(conn, "lora_block_weights")
    )
    if block_missing:
        raise GuardedFlux2ApplyError(
            "lora_block_weights table is missing required column(s): "
            + ", ".join(block_missing)
        )


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GuardedFlux2ApplyError(
            f"{label} mismatch: expected {expected!r}, found {actual!r}"
        )


def _validate_numeric_vector(
    values: Any,
    *,
    label: str,
    count: int,
    unit_range: bool,
) -> list[float]:
    if not isinstance(values, list):
        raise GuardedFlux2ApplyError(f"{label} must be a JSON list")
    if len(values) != count:
        raise GuardedFlux2ApplyError(
            f"{label} count mismatch: expected {count}, found {len(values)}"
        )

    result: list[float] = []
    for index, raw in enumerate(values):
        value = float(raw)
        if not math.isfinite(value):
            raise GuardedFlux2ApplyError(
                f"{label}[{index}] is not finite: {raw!r}"
            )
        if unit_range and not 0.0 <= value <= 1.0:
            raise GuardedFlux2ApplyError(
                f"{label}[{index}] is outside [0, 1]: {value}"
            )
        if not unit_range and value < 0.0:
            raise GuardedFlux2ApplyError(
                f"{label}[{index}] is negative: {value}"
            )
        result.append(value)
    return result


def _validate_artifact(
    artifact: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    _require_equal(artifact.get("phase"), "8.9k", "artifact phase")
    _require_equal(
        artifact.get("mode"),
        "read-only sealed targeted Flux 2 artifact",
        "artifact mode",
    )

    digest = verify_artifact_digest(artifact)
    _require_equal(digest, EXPECTED_ARTIFACT_SHA256, "sealed artifact SHA-256")
    _require_equal(
        artifact.get("phase89j_analysis_sha256"),
        EXPECTED_PHASE89J_SHA256,
        "Phase 8.9j analysis SHA-256",
    )

    summary = artifact.get("summary") or {}
    _require_equal(summary.get("targets_analysed"), 1, "artifact targets_analysed")
    _require_equal(
        summary.get("ready_for_later_controlled_apply"),
        1,
        "artifact ready_for_later_controlled_apply",
    )
    _require_equal(summary.get("block_rows"), EXPECTED_BLOCK_COUNT, "artifact block_rows")
    _require_equal(
        summary.get("global_projection_tensors"),
        EXPECTED_GLOBAL_TENSOR_COUNT,
        "artifact global_projection_tensors",
    )
    _require_equal(
        summary.get("damaged_flux_targets_untouched"),
        1,
        "artifact damaged_flux_targets_untouched",
    )

    safety = artifact.get("safety") or {}
    _require_equal(safety.get("writes_database"), False, "artifact writes_database")
    _require_equal(safety.get("creates_backup"), False, "artifact creates_backup")
    _require_equal(safety.get("contains_apply_mode"), False, "artifact contains_apply_mode")
    _require_equal(safety.get("runs_full_indexer"), False, "artifact runs_full_indexer")
    _require_equal(safety.get("discovers_library_files"), False, "artifact discovers_library_files")
    _require_equal(safety.get("assigns_stable_ids"), False, "artifact assigns_stable_ids")
    _require_equal(safety.get("deletes_rows"), False, "artifact deletes_rows")
    _require_equal(
        safety.get("touches_damaged_flux_target"),
        False,
        "artifact touches_damaged_flux_target",
    )

    target = dict(artifact.get("target") or {})
    stable_id = str(target.get("planned_stable_id") or "").strip().upper()
    source_sha256 = str(target.get("source_sha256") or "").strip().lower()

    exact_target_checks = {
        "planned_stable_id": EXPECTED_STABLE_ID,
        "base_model_name": "Flux 2",
        "base_model_code": "FLX",
        "category_name": "Styles",
        "category_code": "STL",
        "source_sha256": EXPECTED_SOURCE_SHA256,
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
    for field, expected in exact_target_checks.items():
        actual = stable_id if field == "planned_stable_id" else target.get(field)
        if field == "source_sha256":
            actual = source_sha256
        _require_equal(actual, expected, f"artifact target {field}")

    relative_path = str(target.get("relative_path") or "").strip()
    db_file_path = str(target.get("db_file_path") or "").strip()
    filename = str(target.get("filename") or "").strip()
    if not relative_path or not db_file_path or not filename:
        raise GuardedFlux2ApplyError(
            "Artifact target is missing relative_path, db_file_path or filename"
        )

    block_weights = _validate_numeric_vector(
        target.get("block_weights"),
        label="block_weights",
        count=EXPECTED_BLOCK_COUNT,
        unit_range=True,
    )
    raw_strengths = _validate_numeric_vector(
        target.get("raw_block_strengths"),
        label="raw_block_strengths",
        count=EXPECTED_BLOCK_COUNT,
        unit_range=False,
    )
    if max(block_weights, default=0.0) != 1.0:
        raise GuardedFlux2ApplyError(
            "block_weights must contain a strongest block of exactly 1.0"
        )
    if any(value <= 0.0 for value in block_weights):
        raise GuardedFlux2ApplyError(
            "all 56 Flux 2 block weights must be greater than zero"
        )

    target["planned_stable_id"] = stable_id
    target["source_sha256"] = source_sha256
    target["block_weights"] = block_weights
    target["raw_block_strengths"] = raw_strengths
    target["has_block_weights"] = True
    return digest, target


def _validate_source(root: Path, target: Mapping[str, Any]) -> Path:
    source = _safe_source_path(root, str(target["relative_path"]))
    _require_equal(_sha256_file(source), target["source_sha256"], "source SHA-256")
    _require_equal(
        source.stat().st_size,
        int(target["source_size_bytes"]),
        "source size",
    )
    expected_mtime = float(target["source_mtime"])
    if abs(source.stat().st_mtime - expected_mtime) > 1e-6:
        raise GuardedFlux2ApplyError(
            "source mtime mismatch: "
            f"expected {expected_mtime!r}, found {source.stat().st_mtime!r}"
        )
    return source


def _assert_target_absent(
    conn: sqlite3.Connection,
    target: Mapping[str, Any],
) -> None:
    if conn.execute(
        "SELECT id FROM lora WHERE file_path = ?",
        (target["db_file_path"],),
    ).fetchone() is not None:
        raise GuardedFlux2ApplyError(
            f"Target file_path already exists in DB: {target['db_file_path']}"
        )
    if conn.execute(
        "SELECT id FROM lora WHERE stable_id = ?",
        (target["planned_stable_id"],),
    ).fetchone() is not None:
        raise GuardedFlux2ApplyError(
            f"Planned stable ID already exists in DB: {target['planned_stable_id']}"
        )


def _assert_damaged_target_absent(conn: sqlite3.Connection) -> None:
    if conn.execute(
        "SELECT id FROM lora WHERE stable_id = ?",
        (DAMAGED_STABLE_ID,),
    ).fetchone() is not None:
        raise GuardedFlux2ApplyError(
            f"Damaged quarantined target unexpectedly exists in DB: {DAMAGED_STABLE_ID}"
        )


def _duplicate_stable_id_count(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute(
            """
            SELECT COUNT(1)
            FROM (
                SELECT stable_id
                FROM lora
                WHERE stable_id IS NOT NULL
                  AND TRIM(stable_id) <> ''
                GROUP BY stable_id
                HAVING COUNT(1) > 1
            )
            """
        ).fetchone()[0]
    )


def build_preview(
    artifact: Mapping[str, Any],
    *,
    db_path: str | os.PathLike[str],
    library_root: str | os.PathLike[str],
) -> ApplyPreview:
    digest, target = _validate_artifact(artifact)

    root = Path(library_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise GuardedFlux2ApplyError(
            f"LoRA library root is not a directory: {root}"
        )
    source = _validate_source(root, target)

    conn = _open_read_only(db_path)
    try:
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity.casefold() != "ok":
            raise GuardedFlux2ApplyError(
                f"Database integrity check failed: {integrity}"
            )
        _check_schema(conn)
        _assert_target_absent(conn, target)
        _assert_damaged_target_absent(conn)
        _require_equal(
            _duplicate_stable_id_count(conn),
            0,
            "pre-apply duplicate stable ID count",
        )
        lora_rows = int(conn.execute("SELECT COUNT(1) FROM lora").fetchone()[0])
        block_rows = int(
            conn.execute("SELECT COUNT(1) FROM lora_block_weights").fetchone()[0]
        )
    finally:
        conn.close()

    return ApplyPreview(
        artifact_sha256=digest,
        source_path=source,
        target=target,
        lora_rows_before=lora_rows,
        block_rows_before=block_rows,
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _create_verified_backup(
    db_path: Path,
    backup_dir: Path,
    artifact_digest: str,
    *,
    expected_lora_rows: int,
    expected_block_rows: int,
) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / (
        f"{db_path.stem}.phase89l.{stamp}.{artifact_digest[:12]}.db"
    )
    if backup_path.exists():
        raise GuardedFlux2ApplyError(f"Backup path already exists: {backup_path}")

    source = _open_read_only(db_path)
    try:
        destination = sqlite3.connect(backup_path)
        try:
            source.backup(destination)
        finally:
            destination.close()
    finally:
        source.close()

    if not backup_path.is_file() or backup_path.stat().st_size == 0:
        raise GuardedFlux2ApplyError("SQLite backup was not created correctly")

    check = _open_read_only(backup_path)
    try:
        integrity = str(check.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity.casefold() != "ok":
            raise GuardedFlux2ApplyError(
                f"Backup integrity check failed: {integrity}"
            )
        _check_schema(check)
        _require_equal(
            int(check.execute("SELECT COUNT(1) FROM lora").fetchone()[0]),
            expected_lora_rows,
            "backup lora row count",
        )
        _require_equal(
            int(
                check.execute(
                    "SELECT COUNT(1) FROM lora_block_weights"
                ).fetchone()[0]
            ),
            expected_block_rows,
            "backup block row count",
        )
        _require_equal(
            _duplicate_stable_id_count(check),
            0,
            "backup duplicate stable ID count",
        )
        _assert_damaged_target_absent(check)
    finally:
        check.close()
    return backup_path


def _insert_target(
    conn: sqlite3.Connection,
    target: Mapping[str, Any],
    *,
    now: str,
) -> int:
    cursor = conn.execute(
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
            target["db_file_path"],
            target["filename"],
            target["base_model_name"],
            target["base_model_code"],
            target["category_name"],
            target["category_code"],
            target["model_family"],
            target["lora_type"],
            target["rank"],
            1,
            target["block_layout"],
            1 if target.get("clip_contributor") else 0,
            int(target.get("clip_tensor_count") or 0),
            float(target["source_mtime"]),
            now,
            now,
            target["planned_stable_id"],
        ),
    )
    lora_id = int(cursor.lastrowid)

    rows = [
        (
            lora_id,
            target["planned_stable_id"],
            index,
            float(weight),
            float(target["raw_block_strengths"][index]),
        )
        for index, weight in enumerate(target["block_weights"])
    ]
    conn.executemany(
        """
        INSERT INTO lora_block_weights (
            lora_id, stable_id, block_index, weight, raw_strength
        ) VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    return lora_id


def _verify_inserted_rows(
    conn: sqlite3.Connection,
    *,
    lora_id: int,
    target: Mapping[str, Any],
) -> None:
    row = conn.execute("SELECT * FROM lora WHERE id = ?", (lora_id,)).fetchone()
    if row is None:
        raise GuardedFlux2ApplyError(
            "Inserted Flux 2 lora row is missing inside transaction"
        )

    expected_fields = {
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
        "clip_contributor": 1 if target.get("clip_contributor") else 0,
        "clip_tensor_count": int(target.get("clip_tensor_count") or 0),
        "stable_id": EXPECTED_STABLE_ID,
    }
    for field, expected in expected_fields.items():
        _require_equal(row[field], expected, f"inserted lora {field}")

    _require_equal(
        float(row["last_modified"]),
        float(target["source_mtime"]),
        "inserted lora last_modified",
    )

    block_rows = conn.execute(
        """
        SELECT stable_id, block_index, weight, raw_strength
        FROM lora_block_weights
        WHERE lora_id = ?
        ORDER BY block_index
        """,
        (lora_id,),
    ).fetchall()
    _require_equal(
        len(block_rows),
        EXPECTED_BLOCK_COUNT,
        "inserted block row count",
    )

    for index, block_row in enumerate(block_rows):
        _require_equal(
            block_row["stable_id"],
            EXPECTED_STABLE_ID,
            f"block {index} stable_id",
        )
        _require_equal(int(block_row["block_index"]), index, f"block {index} index")
        _require_equal(
            float(block_row["weight"]),
            float(target["block_weights"][index]),
            f"block {index} weight",
        )
        _require_equal(
            float(block_row["raw_strength"]),
            float(target["raw_block_strengths"][index]),
            f"block {index} raw_strength",
        )


def apply_artifact(
    preview: ApplyPreview,
    *,
    db_path: str | os.PathLike[str],
    library_root: str | os.PathLike[str],
    backup_dir: str | os.PathLike[str],
    expected_artifact_sha256: str,
) -> dict[str, Any]:
    expected = str(expected_artifact_sha256 or "").strip().lower()
    if expected != preview.artifact_sha256:
        raise GuardedFlux2ApplyError(
            "Artifact digest mismatch: "
            f"expected argument {expected or 'EMPTY'}, actual {preview.artifact_sha256}"
        )

    db = Path(db_path).expanduser().resolve(strict=True)
    root = Path(library_root).expanduser().resolve(strict=True)
    _validate_source(root, preview.target)

    backup_path = _create_verified_backup(
        db,
        Path(backup_dir).expanduser().resolve(),
        preview.artifact_sha256,
        expected_lora_rows=preview.lora_rows_before,
        expected_block_rows=preview.block_rows_before,
    )

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")

    try:
        _check_schema(conn)
        conn.execute("BEGIN IMMEDIATE")

        _require_equal(
            int(conn.execute("SELECT COUNT(1) FROM lora").fetchone()[0]),
            preview.lora_rows_before,
            "current lora row count",
        )
        _require_equal(
            int(
                conn.execute(
                    "SELECT COUNT(1) FROM lora_block_weights"
                ).fetchone()[0]
            ),
            preview.block_rows_before,
            "current block row count",
        )
        _require_equal(
            _duplicate_stable_id_count(conn),
            0,
            "current duplicate stable ID count",
        )
        _assert_target_absent(conn, preview.target)
        _assert_damaged_target_absent(conn)
        _validate_source(root, preview.target)

        lora_id = _insert_target(conn, preview.target, now=_timestamp())
        _verify_inserted_rows(conn, lora_id=lora_id, target=preview.target)

        _require_equal(
            int(conn.execute("SELECT COUNT(1) FROM lora").fetchone()[0]),
            preview.lora_rows_before + 1,
            "post-insert lora row count",
        )
        _require_equal(
            int(
                conn.execute(
                    "SELECT COUNT(1) FROM lora_block_weights"
                ).fetchone()[0]
            ),
            preview.block_rows_before + EXPECTED_BLOCK_COUNT,
            "post-insert block row count",
        )
        _require_equal(
            _duplicate_stable_id_count(conn),
            0,
            "post-insert duplicate stable ID count",
        )
        _assert_damaged_target_absent(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    check = _open_read_only(db)
    try:
        integrity = str(check.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity.casefold() != "ok":
            raise GuardedFlux2ApplyError(
                "Post-apply database integrity check failed: "
                f"{integrity}; backup={backup_path}"
            )
        _require_equal(
            _duplicate_stable_id_count(check),
            0,
            "post-apply duplicate stable ID count",
        )
        _assert_damaged_target_absent(check)
        rows = check.execute(
            """
            SELECT id
            FROM lora
            WHERE stable_id = ? AND file_path = ?
            """,
            (EXPECTED_STABLE_ID, preview.target["db_file_path"]),
        ).fetchall()
        _require_equal(len(rows), 1, "post-apply exact target row count")
        final_lora_id = int(rows[0]["id"])
        final_block_count = int(
            check.execute(
                "SELECT COUNT(1) FROM lora_block_weights WHERE lora_id = ?",
                (final_lora_id,),
            ).fetchone()[0]
        )
        _require_equal(
            final_block_count,
            EXPECTED_BLOCK_COUNT,
            "post-apply target block count",
        )
        _verify_inserted_rows(
            check,
            lora_id=final_lora_id,
            target=preview.target,
        )
    finally:
        check.close()

    return {
        "artifact_sha256": preview.artifact_sha256,
        "backup_path": str(backup_path),
        "stable_id": EXPECTED_STABLE_ID,
        "lora_id": final_lora_id,
        "model_family": "Flux 2",
        "block_layout": EXPECTED_LAYOUT,
        "lora_rows_inserted": 1,
        "block_rows_inserted": final_block_count,
        "global_projection_tensors_preserved_in_artifact": EXPECTED_GLOBAL_TENSOR_COUNT,
        "damaged_candidate_untouched": 1,
    }


def print_preview(preview: ApplyPreview) -> None:
    summary = preview.summary()
    print("=== Phase 8.9l guarded single Flux 2 apply ===")
    print("Mode                         : dry-run")
    print(f"Artifact SHA-256              : {summary['artifact_sha256']}")
    print(f"Stable ID                     : {summary['stable_id']}")
    print(f"DB file path                  : {summary['db_file_path']}")
    print(f"Source SHA-256                : {summary['source_sha256']}")
    print(f"Model family                  : {summary['model_family']}")
    print(f"Rank                          : {summary['rank']}")
    print(f"Block layout                  : {summary['block_layout']}")
    print(f"LoRA rows to insert           : {summary['lora_rows_to_insert']}")
    print(f"Block rows to insert          : {summary['block_rows_to_insert']}")
    print(f"Global tensors in artifact    : {summary['global_projection_tensors']}")
    print(f"LoRA rows before              : {summary['lora_rows_before']}")
    print(f"Block rows before             : {summary['block_rows_before']}")
    print(f"Damaged target {DAMAGED_STABLE_ID} : absent and untouched")
    print()
    print("No database changes were made.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guarded Phase 8.9l one-row Flux 2 apply"
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--expected-artifact-sha256",
        help="Exact approved Phase 8.9k artifact SHA-256; required with --apply",
    )
    parser.add_argument("--backup-dir", help="Required with --apply")
    return parser


def main() -> int:
    args = _parser().parse_args()
    artifact = load_json_object(args.artifact, "Artifact")
    preview = build_preview(
        artifact,
        db_path=args.db,
        library_root=args.root,
    )

    if not args.apply:
        print_preview(preview)
        return 0

    if not args.expected_artifact_sha256:
        raise SystemExit("--expected-artifact-sha256 is required with --apply")
    if not args.backup_dir:
        raise SystemExit("--backup-dir is required with --apply")

    result = apply_artifact(
        preview,
        db_path=args.db,
        library_root=args.root,
        backup_dir=args.backup_dir,
        expected_artifact_sha256=args.expected_artifact_sha256,
    )
    print("=== Phase 8.9l apply complete ===")
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
