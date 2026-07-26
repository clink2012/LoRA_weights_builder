from __future__ import annotations

import argparse
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from phase89g_targeted_flux_analysis import _open_read_only, _safe_source_path, _sha256_file
from phase89h_sealed_flux_artifact import (
    load_json_object,
    verify_artifact_digest,
)


class ControlledFluxApplyError(RuntimeError):
    pass


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
            "block_layout": self.target["block_layout"],
            "lora_rows_to_insert": 1,
            "block_rows_to_insert": len(self.target["block_weights"]),
            "lora_rows_before": self.lora_rows_before,
            "block_rows_before": self.block_rows_before,
        }


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _check_schema(conn: sqlite3.Connection) -> None:
    lora_missing = sorted(REQUIRED_LORA_COLUMNS - _table_columns(conn, "lora"))
    if lora_missing:
        raise ControlledFluxApplyError(
            "lora table is missing required column(s): " + ", ".join(lora_missing)
        )

    block_missing = sorted(
        REQUIRED_BLOCK_COLUMNS - _table_columns(conn, "lora_block_weights")
    )
    if block_missing:
        raise ControlledFluxApplyError(
            "lora_block_weights table is missing required column(s): "
            + ", ".join(block_missing)
        )


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ControlledFluxApplyError(
            f"{label} mismatch: expected {expected!r}, found {actual!r}"
        )


def _validate_numeric_vector(values: Any, *, label: str, count: int, unit_range: bool) -> list[float]:
    if not isinstance(values, list):
        raise ControlledFluxApplyError(f"{label} must be a JSON list")
    if len(values) != count:
        raise ControlledFluxApplyError(
            f"{label} count mismatch: expected {count}, found {len(values)}"
        )

    result: list[float] = []
    for index, raw in enumerate(values):
        value = float(raw)
        if not math.isfinite(value):
            raise ControlledFluxApplyError(
                f"{label}[{index}] is not finite: {raw!r}"
            )
        if unit_range and not 0.0 <= value <= 1.0:
            raise ControlledFluxApplyError(
                f"{label}[{index}] is outside [0, 1]: {value}"
            )
        result.append(value)
    return result


def _validate_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_stable_id: str,
    expected_block_count: int,
    expected_layout: str,
) -> tuple[str, dict[str, Any]]:
    _require_equal(artifact.get("phase"), "8.9h", "artifact phase")
    digest = verify_artifact_digest(artifact)

    safety = artifact.get("safety") or {}
    _require_equal(safety.get("writes_database"), False, "artifact writes_database")
    _require_equal(safety.get("contains_apply_mode"), False, "artifact contains_apply_mode")
    _require_equal(safety.get("runs_full_indexer"), False, "artifact runs_full_indexer")

    target = dict(artifact.get("target") or {})
    stable_id = str(target.get("planned_stable_id") or "").strip().upper()
    _require_equal(stable_id, expected_stable_id.upper(), "planned stable_id")
    _require_equal(target.get("base_model_code"), "FLX", "base_model_code")
    _require_equal(target.get("category_code"), "PPL", "category_code")
    _require_equal(target.get("has_block_weights"), True, "has_block_weights")
    _require_equal(target.get("block_layout"), expected_layout, "block_layout")
    _require_equal(target.get("block_count"), expected_block_count, "block_count")

    relative_path = str(target.get("relative_path") or "").strip()
    db_file_path = str(target.get("db_file_path") or "").strip()
    source_sha256 = str(target.get("source_sha256") or "").strip().lower()
    if not relative_path or not db_file_path or len(source_sha256) != 64:
        raise ControlledFluxApplyError(
            "Artifact target is missing relative_path, db_file_path or source_sha256"
        )

    block_weights = _validate_numeric_vector(
        target.get("block_weights"),
        label="block_weights",
        count=expected_block_count,
        unit_range=True,
    )
    raw_strengths = _validate_numeric_vector(
        target.get("raw_block_strengths"),
        label="raw_block_strengths",
        count=expected_block_count,
        unit_range=False,
    )

    target["planned_stable_id"] = stable_id
    target["source_sha256"] = source_sha256
    target["block_weights"] = block_weights
    target["raw_block_strengths"] = raw_strengths
    return digest, target


def _validate_source(root: Path, target: Mapping[str, Any]) -> Path:
    source = _safe_source_path(root, str(target["relative_path"]))
    _require_equal(_sha256_file(source), target["source_sha256"], "source SHA-256")
    _require_equal(source.stat().st_size, int(target["source_size_bytes"]), "source size")
    expected_mtime = float(target["source_mtime"])
    if abs(source.stat().st_mtime - expected_mtime) > 1e-6:
        raise ControlledFluxApplyError(
            "source mtime mismatch: "
            f"expected {expected_mtime!r}, found {source.stat().st_mtime!r}"
        )
    return source


def _assert_target_absent(conn: sqlite3.Connection, target: Mapping[str, Any]) -> None:
    if conn.execute(
        "SELECT id FROM lora WHERE file_path = ?", (target["db_file_path"],)
    ).fetchone() is not None:
        raise ControlledFluxApplyError(
            f"Target file_path already exists in DB: {target['db_file_path']}"
        )
    if conn.execute(
        "SELECT id FROM lora WHERE stable_id = ?", (target["planned_stable_id"],)
    ).fetchone() is not None:
        raise ControlledFluxApplyError(
            f"Planned stable ID already exists in DB: {target['planned_stable_id']}"
        )


def build_preview(
    artifact: Mapping[str, Any],
    *,
    db_path: str | os.PathLike[str],
    library_root: str | os.PathLike[str],
    expected_stable_id: str = "FLX-PPL-207",
    expected_block_count: int = 57,
    expected_layout: str = "flux_unet_57",
) -> ApplyPreview:
    digest, target = _validate_artifact(
        artifact,
        expected_stable_id=expected_stable_id,
        expected_block_count=expected_block_count,
        expected_layout=expected_layout,
    )

    root = Path(library_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ControlledFluxApplyError(f"LoRA library root is not a directory: {root}")
    source = _validate_source(root, target)

    conn = _open_read_only(db_path)
    try:
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity.casefold() != "ok":
            raise ControlledFluxApplyError(
                f"Database integrity check failed: {integrity}"
            )
        _check_schema(conn)
        _assert_target_absent(conn, target)
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
        f"{db_path.stem}.phase89i.{stamp}.{artifact_digest[:12]}.db"
    )
    if backup_path.exists():
        raise ControlledFluxApplyError(f"Backup path already exists: {backup_path}")

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
        raise ControlledFluxApplyError("SQLite backup was not created correctly")

    check = _open_read_only(backup_path)
    try:
        integrity = str(check.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity.casefold() != "ok":
            raise ControlledFluxApplyError(
                f"Backup integrity check failed: {integrity}"
            )
        _require_equal(
            int(check.execute("SELECT COUNT(1) FROM lora").fetchone()[0]),
            expected_lora_rows,
            "backup lora row count",
        )
        _require_equal(
            int(
                check.execute("SELECT COUNT(1) FROM lora_block_weights").fetchone()[0]
            ),
            expected_block_rows,
            "backup block row count",
        )
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
            target.get("base_model_name"),
            target["base_model_code"],
            target.get("category_name"),
            target["category_code"],
            target.get("model_family"),
            target.get("lora_type"),
            target.get("rank"),
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
        raise ControlledFluxApplyError("Inserted lora row is missing inside transaction")

    expected_fields = {
        "file_path": target["db_file_path"],
        "filename": target["filename"],
        "base_model_name": target.get("base_model_name"),
        "base_model_code": target["base_model_code"],
        "category_name": target.get("category_name"),
        "category_code": target["category_code"],
        "model_family": target.get("model_family"),
        "lora_type": target.get("lora_type"),
        "rank": target.get("rank"),
        "has_block_weights": 1,
        "block_layout": target["block_layout"],
        "clip_contributor": 1 if target.get("clip_contributor") else 0,
        "clip_tensor_count": int(target.get("clip_tensor_count") or 0),
        "stable_id": target["planned_stable_id"],
    }
    for field, expected in expected_fields.items():
        _require_equal(row[field], expected, f"inserted lora {field}")

    block_rows = conn.execute(
        """
        SELECT stable_id, block_index, weight, raw_strength
        FROM lora_block_weights
        WHERE lora_id = ?
        ORDER BY block_index
        """,
        (lora_id,),
    ).fetchall()
    _require_equal(len(block_rows), len(target["block_weights"]), "inserted block row count")

    for index, block_row in enumerate(block_rows):
        _require_equal(block_row["stable_id"], target["planned_stable_id"], f"block {index} stable_id")
        _require_equal(int(block_row["block_index"]), index, f"block {index} index")
        _require_equal(float(block_row["weight"]), float(target["block_weights"][index]), f"block {index} weight")
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
        raise ControlledFluxApplyError(
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
                conn.execute("SELECT COUNT(1) FROM lora_block_weights").fetchone()[0]
            ),
            preview.block_rows_before,
            "current block row count",
        )
        _assert_target_absent(conn, preview.target)
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
                conn.execute("SELECT COUNT(1) FROM lora_block_weights").fetchone()[0]
            ),
            preview.block_rows_before + len(preview.target["block_weights"]),
            "post-insert block row count",
        )
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
            raise ControlledFluxApplyError(
                f"Post-apply database integrity check failed: {integrity}; backup={backup_path}"
            )
        row = check.execute(
            "SELECT id FROM lora WHERE stable_id = ? AND file_path = ?",
            (preview.target["planned_stable_id"], preview.target["db_file_path"]),
        ).fetchone()
        if row is None:
            raise ControlledFluxApplyError(
                f"Post-apply target verification failed; backup={backup_path}"
            )
        final_lora_id = int(row["id"])
        final_block_count = int(
            check.execute(
                "SELECT COUNT(1) FROM lora_block_weights WHERE lora_id = ?",
                (final_lora_id,),
            ).fetchone()[0]
        )
        _require_equal(
            final_block_count,
            len(preview.target["block_weights"]),
            "post-apply target block count",
        )
    finally:
        check.close()

    return {
        "artifact_sha256": preview.artifact_sha256,
        "backup_path": str(backup_path),
        "stable_id": preview.target["planned_stable_id"],
        "lora_id": final_lora_id,
        "lora_rows_inserted": 1,
        "block_rows_inserted": final_block_count,
        "blocked_candidates_untouched": 2,
    }


def print_preview(preview: ApplyPreview) -> None:
    summary = preview.summary()
    print("=== Phase 8.9i controlled single Flux apply ===")
    print("Mode                 : dry-run")
    print(f"Artifact SHA-256      : {summary['artifact_sha256']}")
    print(f"Stable ID             : {summary['stable_id']}")
    print(f"DB file path          : {summary['db_file_path']}")
    print(f"Source SHA-256        : {summary['source_sha256']}")
    print(f"Block layout          : {summary['block_layout']}")
    print(f"LoRA rows to insert   : {summary['lora_rows_to_insert']}")
    print(f"Block rows to insert  : {summary['block_rows_to_insert']}")
    print(f"LoRA rows before      : {summary['lora_rows_before']}")
    print(f"Block rows before     : {summary['block_rows_before']}")
    print("Blocked FLX targets   : 2 (untouched)")
    print()
    print("No database changes were made.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guarded Phase 8.9i one-row Flux apply"
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-stable-id", default="FLX-PPL-207")
    parser.add_argument("--expected-block-count", type=int, default=57)
    parser.add_argument("--expected-layout", default="flux_unet_57")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expected-artifact-sha256", help="Required with --apply")
    parser.add_argument("--backup-dir", help="Required with --apply")
    return parser


def main() -> int:
    args = _parser().parse_args()
    artifact = load_json_object(args.artifact, "Artifact")
    preview = build_preview(
        artifact,
        db_path=args.db,
        library_root=args.root,
        expected_stable_id=args.expected_stable_id,
        expected_block_count=args.expected_block_count,
        expected_layout=args.expected_layout,
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
    print("=== Phase 8.9i apply complete ===")
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
