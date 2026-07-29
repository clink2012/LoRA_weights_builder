from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import quote

from phase89e_metadata_reconcile import (
    build_execution_preview,
    load_plan,
    plan_sha256,
)
from phase89k_flux2_layout_support import canonical_sha256, load_json_object
from phase89m_post_apply_closeout import verify_report_digest as verify_phase89m_digest


class ResidualReviewError(RuntimeError):
    pass


EXPECTED_PHASE89D_PLAN_SHA256 = (
    "e93d23901e4f05b0f250a0574c8662700be3428854c104f62146058e7ba6c7f2"
)
EXPECTED_PHASE89M_VERIFICATION_SHA256 = (
    "132805103fa858d7954245a309e81bcaa23c4062190d60252fc15a41fb655da7"
)
EXPECTED_CURRENT_DB_SHA256 = (
    "6526505261ed62c79c433217161716e6d0bb9b286fb266867f9e6c87b1fa2357"
)
EXPECTED_LORA_ROWS = 2834
EXPECTED_BLOCK_ROWS = 4348
EXPECTED_METADATA_INSERTS = 308
EXPECTED_METADATA_BACKFILLS = 49
EXPECTED_ID_ASSIGNMENTS = 2
EXPECTED_SCANNED_CANDIDATES = 3
EXPECTED_PREFIX_CONFLICTS = 30
EXPECTED_SAME_FAMILY_RELOCATIONS = 3
EXPECTED_CROSS_FAMILY_RELOCATIONS = 20
EXPECTED_STALE_HOLD_ROWS = 668
EXPECTED_LEGACY_HOLD_ROWS = 114

COMPLETED_SCANNED_TARGETS = {
    "FLX-PPL-207": {
        "status": "completed in Phase 8.9i",
        "block_layout": "flux_unet_57",
        "block_rows": 57,
    },
    "FLX-STL-263": {
        "status": "completed in Phase 8.9l",
        "block_layout": "flux2_transformer_56",
        "block_rows": 56,
    },
}
DAMAGED_SCANNED_TARGET = "FLX-BDY-071"


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
        raise ResidualReviewError(
            f"{label} mismatch: expected {expected!r}, found {actual!r}"
        )


def _normalise_path(value: Any) -> str:
    return "/".join(
        part
        for part in str(value or "").strip().replace("\\", "/").split("/")
        if part
    ).casefold()


def _db_file_path(relative: str, root: str = "/loras") -> str:
    rel = PurePosixPath(relative).as_posix().lstrip("/")
    prefix = str(root or "").replace("\\", "/").rstrip("/")
    return f"{prefix}/{rel}" if prefix else rel


def _matches_relative(file_path: Any, relative: str) -> bool:
    current = _normalise_path(file_path)
    expected = _normalise_path(relative)
    return current == expected or current.endswith(f"/{expected}")


def _planned_insert_ids(plan: Mapping[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for raw in plan.get("planned_stable_ids") or []:
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("source_type") or "") != "new_metadata_insert":
            continue
        relative = str(raw.get("relative_path") or "").strip()
        stable_id = str(raw.get("planned_stable_id") or "").strip().upper()
        if relative and stable_id:
            lookup[relative.casefold()] = stable_id
    return lookup


def _verify_phase89m_report(
    report: Mapping[str, Any],
    *,
    expected_digest: str,
    expected_db_sha256: str,
    expected_lora_rows: int,
    expected_block_rows: int,
) -> str:
    digest = verify_phase89m_digest(report)
    _require_equal(digest, expected_digest, "Phase 8.9m verification SHA-256")
    _require_equal(report.get("phase"), "8.9m", "Phase 8.9m report phase")
    _require_equal(report.get("status"), "verified", "Phase 8.9m report status")
    _require_equal(
        report.get("current_db_sha256"),
        expected_db_sha256,
        "Phase 8.9m current DB SHA-256",
    )
    database = report.get("database") or {}
    _require_equal(
        database.get("current_lora_rows"),
        expected_lora_rows,
        "Phase 8.9m current LoRA rows",
    )
    _require_equal(
        database.get("current_block_rows"),
        expected_block_rows,
        "Phase 8.9m current block rows",
    )
    _require_equal(database.get("duplicate_stable_ids"), 0, "Phase 8.9m duplicates")
    _require_equal(database.get("orphan_block_rows"), 0, "Phase 8.9m orphan blocks")
    quarantine = report.get("quarantine") or {}
    _require_equal(
        quarantine.get("stable_id"),
        DAMAGED_SCANNED_TARGET,
        "Phase 8.9m quarantine stable ID",
    )
    _require_equal(quarantine.get("current_rows"), 0, "Phase 8.9m quarantine rows")
    return digest


def _verify_database_state(
    conn: sqlite3.Connection,
    *,
    expected_lora_rows: int,
    expected_block_rows: int,
) -> dict[str, int | str]:
    integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
    if integrity.casefold() != "ok":
        raise ResidualReviewError(f"database integrity_check failed: {integrity}")

    lora_rows = int(conn.execute("SELECT COUNT(*) FROM lora").fetchone()[0])
    block_rows = int(
        conn.execute("SELECT COUNT(*) FROM lora_block_weights").fetchone()[0]
    )
    with_ids = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM lora
            WHERE stable_id IS NOT NULL
              AND TRIM(stable_id) <> ''
            """
        ).fetchone()[0]
    )
    duplicates = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT stable_id
                FROM lora
                WHERE stable_id IS NOT NULL
                  AND TRIM(stable_id) <> ''
                GROUP BY stable_id
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
    )
    orphan_blocks = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM lora_block_weights AS bw
            LEFT JOIN lora AS l ON l.id = bw.lora_id
            WHERE l.id IS NULL
            """
        ).fetchone()[0]
    )

    _require_equal(lora_rows, expected_lora_rows, "current LoRA row count")
    _require_equal(block_rows, expected_block_rows, "current block row count")
    _require_equal(with_ids, expected_lora_rows, "rows with stable IDs")
    _require_equal(duplicates, 0, "duplicate stable IDs")
    _require_equal(orphan_blocks, 0, "orphan block rows")

    return {
        "integrity": integrity,
        "lora_rows": lora_rows,
        "block_rows": block_rows,
        "rows_with_stable_ids": with_ids,
        "duplicate_stable_ids": duplicates,
        "orphan_block_rows": orphan_blocks,
    }


def _single_row_by_stable_id(
    conn: sqlite3.Connection,
    stable_id: str,
) -> sqlite3.Row:
    rows = conn.execute(
        "SELECT * FROM lora WHERE stable_id = ?",
        (stable_id,),
    ).fetchall()
    _require_equal(len(rows), 1, f"row count for {stable_id}")
    return rows[0]


def _verify_completed_phase89e(
    conn: sqlite3.Connection,
    preview: Any,
    *,
    expected_metadata_inserts: int,
    expected_metadata_backfills: int,
    expected_id_assignments: int,
) -> dict[str, int]:
    _require_equal(
        len(preview.inserts),
        expected_metadata_inserts,
        "Phase 8.9e metadata insert count",
    )
    _require_equal(
        len(preview.backfills),
        expected_metadata_backfills,
        "Phase 8.9e metadata backfill count",
    )
    _require_equal(
        len(preview.id_assignments),
        expected_id_assignments,
        "Phase 8.9e ID assignment count",
    )

    verified_inserts = 0
    for item in preview.inserts:
        stable_id = str(item["planned_stable_id"]).upper()
        row = _single_row_by_stable_id(conn, stable_id)
        _require_equal(
            row["file_path"],
            _db_file_path(str(item["relative_path"])),
            f"{stable_id} file_path",
        )
        _require_equal(
            str(row["base_model_code"] or "").upper(),
            str(item.get("base_model_code") or "").upper(),
            f"{stable_id} base_model_code",
        )
        _require_equal(
            str(row["category_code"] or "").upper(),
            str(item.get("category_code") or "").upper(),
            f"{stable_id} category_code",
        )
        _require_equal(int(row["has_block_weights"] or 0), 0, f"{stable_id} block flag")
        _require_equal(row["block_layout"], None, f"{stable_id} block layout")
        block_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM lora_block_weights WHERE lora_id = ?",
                (int(row["id"]),),
            ).fetchone()[0]
        )
        _require_equal(block_count, 0, f"{stable_id} block rows")
        verified_inserts += 1

    verified_backfills = 0
    for item in preview.backfills:
        row_id = int(item["row_id"])
        row = conn.execute("SELECT * FROM lora WHERE id = ?", (row_id,)).fetchone()
        if row is None:
            raise ResidualReviewError(f"Phase 8.9e backfill row is missing: {row_id}")
        _require_equal(row["file_path"], item.get("file_path"), f"backfill {row_id} path")
        for field, transition in (item.get("changed_fields") or {}).items():
            _require_equal(
                row[field],
                transition.get("to"),
                f"backfill {row_id} {field}",
            )
        verified_backfills += 1

    verified_ids = 0
    for item in preview.id_assignments:
        row_id = int(item["row_id"])
        row = conn.execute("SELECT * FROM lora WHERE id = ?", (row_id,)).fetchone()
        if row is None:
            raise ResidualReviewError(f"Phase 8.9e ID row is missing: {row_id}")
        _require_equal(
            str(row["stable_id"] or "").upper(),
            str(item["planned_stable_id"]).upper(),
            f"ID assignment row {row_id}",
        )
        verified_ids += 1

    return {
        "verified_metadata_inserts": verified_inserts,
        "verified_metadata_backfills": verified_backfills,
        "verified_existing_id_assignments": verified_ids,
    }


def _classify_scanned_candidates(
    conn: sqlite3.Connection,
    plan: Mapping[str, Any],
    excluded_scanned: Any,
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    _require_equal(len(excluded_scanned), expected_count, "scanned candidate count")
    id_lookup = _planned_insert_ids(plan)
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for raw in excluded_scanned:
        item = dict(raw)
        relative = str(item.get("relative_path") or "").strip()
        stable_id = id_lookup.get(relative.casefold())
        if not stable_id:
            raise ResidualReviewError(
                f"scanned candidate has no planned stable ID: {relative}"
            )
        seen.add(stable_id)
        expected_path = _db_file_path(relative)

        if stable_id in COMPLETED_SCANNED_TARGETS:
            policy = COMPLETED_SCANNED_TARGETS[stable_id]
            row = _single_row_by_stable_id(conn, stable_id)
            _require_equal(row["file_path"], expected_path, f"{stable_id} file_path")
            _require_equal(
                row["block_layout"],
                policy["block_layout"],
                f"{stable_id} block layout",
            )
            block_rows = int(
                conn.execute(
                    "SELECT COUNT(*) FROM lora_block_weights WHERE lora_id = ?",
                    (int(row["id"]),),
                ).fetchone()[0]
            )
            _require_equal(
                block_rows,
                int(policy["block_rows"]),
                f"{stable_id} block row count",
            )
            results.append(
                {
                    "stable_id": stable_id,
                    "relative_path": relative,
                    "status": policy["status"],
                    "database_rows": 1,
                    "block_layout": policy["block_layout"],
                    "block_rows": block_rows,
                    "residual_action": "none",
                }
            )
            continue

        if stable_id == DAMAGED_SCANNED_TARGET:
            db_rows = int(
                conn.execute(
                    "SELECT COUNT(*) FROM lora WHERE stable_id = ? OR file_path = ?",
                    (stable_id, expected_path),
                ).fetchone()[0]
            )
            _require_equal(db_rows, 0, f"{stable_id} quarantine rows")
            results.append(
                {
                    "stable_id": stable_id,
                    "relative_path": relative,
                    "status": "quarantined damaged source",
                    "database_rows": 0,
                    "block_rows": 0,
                    "residual_action": "replace source file before any new analysis",
                }
            )
            continue

        raise ResidualReviewError(f"unexpected scanned candidate stable ID: {stable_id}")

    expected_ids = set(COMPLETED_SCANNED_TARGETS) | {DAMAGED_SCANNED_TARGET}
    _require_equal(seen, expected_ids, "scanned candidate stable ID set")
    return sorted(results, key=lambda item: str(item["stable_id"]))


def _verify_prefix_conflicts(
    conn: sqlite3.Connection,
    items: Any,
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    _require_equal(len(items), expected_count, "ID-prefix conflict count")
    results: list[dict[str, Any]] = []
    for raw in items:
        item = dict(raw)
        row_id = int(item["row_id"])
        row = conn.execute("SELECT * FROM lora WHERE id = ?", (row_id,)).fetchone()
        if row is None:
            raise ResidualReviewError(f"ID-prefix conflict row is missing: {row_id}")
        _require_equal(row["file_path"], item.get("file_path"), f"conflict {row_id} path")
        _require_equal(
            str(row["stable_id"] or ""),
            str(item.get("stable_id") or ""),
            f"conflict {row_id} stable ID",
        )
        for field, transition in (item.get("changed_fields") or {}).items():
            _require_equal(
                row[field],
                transition.get("from"),
                f"conflict {row_id} unchanged {field}",
            )
        results.append(
            {
                "row_id": row_id,
                "stable_id": item.get("stable_id"),
                "file_path": item.get("file_path"),
                "changed_fields": item.get("changed_fields") or {},
                "exclusion_reason": item.get("exclusion_reason"),
                "status": "unchanged; manual stable-ID continuity policy required",
            }
        )
    return results


def _verify_relocations(
    conn: sqlite3.Connection,
    items: Any,
    *,
    expected_count: int,
    review_class: str,
) -> list[dict[str, Any]]:
    _require_equal(len(items), expected_count, f"{review_class} count")
    all_paths = [
        (int(row["id"]), str(row["file_path"] or ""))
        for row in conn.execute("SELECT id, file_path FROM lora")
    ]
    results: list[dict[str, Any]] = []

    for raw in items:
        item = dict(raw)
        row_id = int(item["row_id"])
        row = conn.execute("SELECT * FROM lora WHERE id = ?", (row_id,)).fetchone()
        if row is None:
            raise ResidualReviewError(f"relocation row is missing: {row_id}")
        _require_equal(row["file_path"], item.get("old_path"), f"relocation {row_id} old path")
        _require_equal(
            str(row["stable_id"] or ""),
            str(item.get("stable_id") or ""),
            f"relocation {row_id} stable ID",
        )

        new_path = str(item.get("new_path") or "")
        collisions = [
            other_id
            for other_id, file_path in all_paths
            if other_id != row_id and _matches_relative(file_path, new_path)
        ]
        _require_equal(collisions, [], f"relocation {row_id} new-path DB collisions")

        results.append(
            {
                **item,
                "status": "unchanged; no relocation applied",
                "new_path_database_rows": 0,
                "next_requirement": (
                    "targeted identity evidence review"
                    if review_class == "same-family relocation"
                    else "manual family and stable-ID policy plus identity evidence"
                ),
            }
        )
    return results


def build_residual_review(
    plan: Mapping[str, Any],
    phase89m_report: Mapping[str, Any],
    *,
    db_path: str | os.PathLike[str],
    expected_plan_sha256: str = EXPECTED_PHASE89D_PLAN_SHA256,
    expected_phase89m_sha256: str = EXPECTED_PHASE89M_VERIFICATION_SHA256,
    expected_db_sha256: str = EXPECTED_CURRENT_DB_SHA256,
    expected_lora_rows: int = EXPECTED_LORA_ROWS,
    expected_block_rows: int = EXPECTED_BLOCK_ROWS,
    expected_metadata_inserts: int = EXPECTED_METADATA_INSERTS,
    expected_metadata_backfills: int = EXPECTED_METADATA_BACKFILLS,
    expected_id_assignments: int = EXPECTED_ID_ASSIGNMENTS,
    expected_scanned_candidates: int = EXPECTED_SCANNED_CANDIDATES,
    expected_prefix_conflicts: int = EXPECTED_PREFIX_CONFLICTS,
    expected_same_family_relocations: int = EXPECTED_SAME_FAMILY_RELOCATIONS,
    expected_cross_family_relocations: int = EXPECTED_CROSS_FAMILY_RELOCATIONS,
    expected_stale_hold_rows: int = EXPECTED_STALE_HOLD_ROWS,
    expected_legacy_hold_rows: int = EXPECTED_LEGACY_HOLD_ROWS,
) -> dict[str, Any]:
    actual_plan_sha = plan_sha256(plan)
    _require_equal(actual_plan_sha, expected_plan_sha256, "Phase 8.9d plan SHA-256")
    preview = build_execution_preview(plan)
    _require_equal(preview.plan_sha256, actual_plan_sha, "execution preview plan SHA-256")

    phase89m_sha = _verify_phase89m_report(
        phase89m_report,
        expected_digest=expected_phase89m_sha256,
        expected_db_sha256=expected_db_sha256,
        expected_lora_rows=expected_lora_rows,
        expected_block_rows=expected_block_rows,
    )

    db_file = Path(db_path).expanduser().resolve(strict=True)
    actual_db_sha = _file_sha256(db_file)
    _require_equal(actual_db_sha, expected_db_sha256, "current DB SHA-256")

    summary = plan.get("summary") or {}
    _require_equal(
        int(summary.get("untouched_stale_current_family_rows") or 0),
        expected_stale_hold_rows,
        "carried-forward stale hold count",
    )
    _require_equal(
        int(summary.get("untouched_legacy_unmounted_rows") or 0),
        expected_legacy_hold_rows,
        "carried-forward legacy hold count",
    )

    conn = _open_read_only(db_file)
    try:
        database = _verify_database_state(
            conn,
            expected_lora_rows=expected_lora_rows,
            expected_block_rows=expected_block_rows,
        )
        completed_metadata = _verify_completed_phase89e(
            conn,
            preview,
            expected_metadata_inserts=expected_metadata_inserts,
            expected_metadata_backfills=expected_metadata_backfills,
            expected_id_assignments=expected_id_assignments,
        )
        scanned = _classify_scanned_candidates(
            conn,
            plan,
            preview.excluded_scanned_inserts,
            expected_count=expected_scanned_candidates,
        )
        prefix_conflicts = _verify_prefix_conflicts(
            conn,
            preview.excluded_id_prefix_backfills,
            expected_count=expected_prefix_conflicts,
        )
        same_family = _verify_relocations(
            conn,
            plan.get("same_family_relocations") or [],
            expected_count=expected_same_family_relocations,
            review_class="same-family relocation",
        )
        cross_family = _verify_relocations(
            conn,
            plan.get("cross_family_reclassifications") or [],
            expected_count=expected_cross_family_relocations,
            review_class="cross-family reclassification",
        )
    finally:
        conn.close()

    damaged_items = [
        item for item in scanned if item["stable_id"] == DAMAGED_SCANNED_TARGET
    ]
    _require_equal(len(damaged_items), 1, "damaged scanned residual count")

    report: dict[str, Any] = {
        "phase": "8.10a",
        "mode": "read-only residual reconciliation review",
        "status": "verified",
        "phase89d_plan_sha256": actual_plan_sha,
        "phase89m_verification_sha256": phase89m_sha,
        "current_db_sha256": actual_db_sha,
        "database": database,
        "completed_work_verified": {
            **completed_metadata,
            "completed_scanned_targets": len(COMPLETED_SCANNED_TARGETS),
        },
        "scanned_candidates": scanned,
        "residual": {
            "damaged_scanned_candidates": damaged_items,
            "id_prefix_conflicts": prefix_conflicts,
            "same_family_relocations": same_family,
            "cross_family_reclassifications": cross_family,
            "stale_current_family_rows_declared_hold": expected_stale_hold_rows,
            "legacy_unmounted_rows_declared_hold": expected_legacy_hold_rows,
        },
        "residual_counts": {
            "damaged_scanned_candidates": len(damaged_items),
            "id_prefix_conflicts": len(prefix_conflicts),
            "same_family_relocations": len(same_family),
            "cross_family_reclassifications": len(cross_family),
            "stale_current_family_rows_declared_hold": expected_stale_hold_rows,
            "legacy_unmounted_rows_declared_hold": expected_legacy_hold_rows,
        },
        "recommended_next_slices": {
            "damaged_scanned_candidate": (
                "wait for a clean replacement before targeted analysis"
            ),
            "id_prefix_conflicts": (
                "decide whether stable-ID continuity or folder-derived family metadata takes precedence"
            ),
            "same_family_relocations": (
                "perform a targeted identity-evidence review; do not relocate from filename evidence alone"
            ),
            "cross_family_reclassifications": (
                "make a manual WAN family/stable-ID policy decision before any path update"
            ),
            "stale_and_legacy_holds": (
                "retain unchanged until a fresh library audit is explicitly approved"
            ),
        },
        "evidence_limits": {
            "library_enumerated": False,
            "safetensors_opened": False,
            "stale_and_legacy_counts_are_fresh": False,
            "stale_and_legacy_basis": (
                "carried forward from the exact Phase 8.9d plan; not re-observed in Phase 8.10a"
            ),
            "relocation_identity_proven": False,
            "relocation_identity_basis": (
                "the Phase 8.9d candidates remain filename-only advisory matches"
            ),
        },
        "safety": {
            "database_open_mode": "SQLite URI mode=ro plus PRAGMA query_only=ON",
            "writes_database": False,
            "creates_backup": False,
            "runs_indexer": False,
            "runs_full_scan": False,
            "enumerates_library": False,
            "opens_safetensors": False,
            "assigns_stable_ids": False,
            "relocates_rows": False,
            "deletes_rows": False,
        },
    }
    report["review_sha256"] = canonical_sha256(report)
    return report


def verify_review_digest(report: Mapping[str, Any]) -> str:
    stored = str(report.get("review_sha256") or "")
    unsigned = dict(report)
    unsigned.pop("review_sha256", None)
    calculated = canonical_sha256(unsigned)
    if stored != calculated:
        raise ResidualReviewError(
            f"residual review digest mismatch: stored {stored}, calculated {calculated}"
        )
    return calculated


def print_review(report: Mapping[str, Any]) -> None:
    completed = report["completed_work_verified"]
    residual = report["residual_counts"]
    print("=== Phase 8.10a residual reconciliation review ===")
    print(f"Status                              : {report['status']}")
    print(f"Review SHA-256                       : {report['review_sha256']}")
    print(f"Current DB SHA-256                   : {report['current_db_sha256']}")
    print(f"Verified metadata inserts            : {completed['verified_metadata_inserts']}")
    print(f"Verified metadata backfills          : {completed['verified_metadata_backfills']}")
    print(f"Verified existing ID assignments     : {completed['verified_existing_id_assignments']}")
    print(f"Completed scanned targets            : {completed['completed_scanned_targets']}")
    print(f"Damaged scanned candidates           : {residual['damaged_scanned_candidates']}")
    print(f"ID-prefix conflicts                  : {residual['id_prefix_conflicts']}")
    print(f"Same-family relocation reviews       : {residual['same_family_relocations']}")
    print(f"Cross-family reclassification reviews: {residual['cross_family_reclassifications']}")
    print(f"Carried stale-row hold count          : {residual['stale_current_family_rows_declared_hold']}")
    print(f"Carried legacy-row hold count         : {residual['legacy_unmounted_rows_declared_hold']}")
    print("No database changes were made.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only Phase 8.10a residual reconciliation review"
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--phase89m-report", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--json")
    args = parser.parse_args()

    report = build_residual_review(
        load_plan(args.plan),
        load_json_object(args.phase89m_report, "Phase 8.9m report"),
        db_path=args.db,
    )
    verify_review_digest(report)
    print_review(report)

    if args.json:
        output = Path(args.json).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        print(f"JSON residual review written to: {output}")


if __name__ == "__main__":
    main()
