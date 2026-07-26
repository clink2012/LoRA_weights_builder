from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from lora_path_parser import parse_base_and_category
from phase89_audit import (
    DEFAULT_IGNORED_FOLDERS,
    canonical_db_relative_path,
    discover_library,
    open_read_only_db,
    run_audit,
)
from phase89_relocation_audit import analyse_relocations

_STABLE_ID_RE = re.compile(r"^(?P<prefix>[A-Z0-9]{3}-[A-Z0-9]{3})-(?P<num>[0-9]{3})$")


def _normalise_path(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


def _path_key(value: Any) -> str:
    return "/".join(part for part in _normalise_path(value).split("/") if part).casefold()


def _relative_full_path(root_dir: str | Path, relative: str) -> str:
    return str(Path(root_dir).joinpath(*PurePosixPath(relative).parts))


def _parse_relative_metadata(relative: str, root_dir: str | Path) -> dict[str, Any]:
    full_path = _relative_full_path(root_dir, relative)
    base_name, base_code, category_name, category_code = parse_base_and_category(
        full_path,
        str(root_dir),
    )
    return {
        "relative_path": relative,
        "filename": PurePosixPath(relative).name,
        "base_model_name": base_name,
        "base_model_code": base_code,
        "category_name": category_name,
        "category_code": category_code,
    }


def _stable_id_parts(stable_id: Any) -> tuple[str, int] | None:
    value = str(stable_id or "").strip().upper()
    match = _STABLE_ID_RE.match(value)
    if not match:
        return None
    return match.group("prefix"), int(match.group("num"))


def _db_rows(db_path: str | Path) -> list[dict[str, Any]]:
    conn = open_read_only_db(db_path)
    try:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(lora)")}
        required = {"id", "file_path", "filename", "base_model_code", "category_code", "stable_id"}
        missing = sorted(required - columns)
        if missing:
            raise RuntimeError(f"lora table is missing required column(s): {', '.join(missing)}")
        return [
            dict(row)
            for row in conn.execute(
                """
                SELECT id, file_path, filename, base_model_name, base_model_code,
                       category_name, category_code, stable_id
                FROM lora
                ORDER BY id
                """
            )
        ]
    finally:
        conn.close()


def _allocate_stable_ids(
    rows: Iterable[Mapping[str, Any]],
    pending: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    used_by_prefix: dict[str, set[int]] = defaultdict(set)
    duplicate_counts: Counter[str] = Counter()
    invalid_existing: list[str] = []

    for row in rows:
        stable_id = str(row.get("stable_id") or "").strip().upper()
        if not stable_id:
            continue
        duplicate_counts[stable_id] += 1
        parsed = _stable_id_parts(stable_id)
        if parsed is None:
            invalid_existing.append(stable_id)
            continue
        prefix, number = parsed
        used_by_prefix[prefix].add(number)

    duplicate_ids = [stable_id for stable_id, count in duplicate_counts.items() if count > 1]
    planned: list[dict[str, Any]] = []
    exhausted: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for item in pending:
        base_code = str(item.get("base_model_code") or "").strip().upper()
        category_code = str(item.get("category_code") or "").strip().upper()
        if not base_code or not category_code:
            continue
        grouped[(base_code, category_code)].append(item)

    for (base_code, category_code), items in sorted(grouped.items()):
        prefix = f"{base_code}-{category_code}"
        used = used_by_prefix[prefix]
        next_candidate = 1
        for item in sorted(
            items,
            key=lambda value: (
                str(value.get("filename") or "").casefold(),
                str(value.get("relative_path") or value.get("file_path") or "").casefold(),
                str(value.get("source_type") or ""),
            ),
        ):
            while next_candidate in used and next_candidate <= 999:
                next_candidate += 1
            if next_candidate > 999:
                exhausted.append({**item, "stable_id_prefix": prefix})
                continue
            stable_id = f"{prefix}-{next_candidate:03d}"
            planned.append({**item, "planned_stable_id": stable_id})
            used.add(next_candidate)
            next_candidate += 1

    return planned, exhausted, sorted(set(duplicate_ids + invalid_existing))


def build_index_plan(
    *,
    root_dir: str | Path,
    db_path: str | Path,
    ignored_folders: Iterable[str] = DEFAULT_IGNORED_FOLDERS,
    sample_limit: int = 20,
) -> dict[str, Any]:
    root = Path(root_dir).expanduser().resolve(strict=True)
    audit = run_audit(
        root_dir=root,
        db_path=db_path,
        ignored_folders=ignored_folders,
        sample_limit=sample_limit,
    )
    relocation_report = analyse_relocations(audit)
    library = discover_library(root, ignored_folders)
    mounted_keys = set(library["mounted_files"])
    top_level_folders = [*library["top_level_folders"], *library["ignored_folders_present"]]
    rows = _db_rows(db_path)

    rows_by_raw_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_raw_path[_path_key(row.get("file_path"))].append(row)

    relocation_new_paths: set[str] = set()
    same_family_relocations: list[dict[str, Any]] = []
    cross_family_reclassifications: list[dict[str, Any]] = []
    unresolved_relocations: list[dict[str, Any]] = []
    pending_ids: list[dict[str, Any]] = []

    for match in relocation_report["unique_matches"]:
        old_path = match["old_path"]
        new_path = match["new_path"]
        relocation_new_paths.add(_path_key(new_path))
        matching_rows = rows_by_raw_path.get(_path_key(old_path), [])
        if len(matching_rows) != 1:
            unresolved_relocations.append(
                {
                    **match,
                    "reason": f"expected one DB row for old path, found {len(matching_rows)}",
                }
            )
            continue

        row = matching_rows[0]
        metadata = _parse_relative_metadata(new_path, root)
        stable_id = str(row.get("stable_id") or "").strip() or None
        item = {
            "row_id": row["id"],
            "stable_id": stable_id,
            "old_path": old_path,
            "new_path": new_path,
            "from_family": match["from_family"],
            "to_family": match["to_family"],
            "new_base_model_code": metadata["base_model_code"],
            "new_category_code": metadata["category_code"],
            "identity_evidence": "unique case-insensitive exact filename only",
            "requires_content_hash_verification": True,
        }

        if match["from_family"].casefold() == match["to_family"].casefold():
            item["review_class"] = "same-family path relocation"
            item["stable_id_policy"] = "preserve existing stable_id"
            same_family_relocations.append(item)
        else:
            item["review_class"] = "cross-family reclassification"
            item["stable_id_policy"] = (
                "manual decision required: preserving the ID keeps references stable but may leave an old family prefix"
            )
            cross_family_reclassifications.append(item)

        if not stable_id and metadata["base_model_code"] and metadata["category_code"]:
            pending_ids.append(
                {
                    "source_type": "relocation_row_missing_id",
                    "row_id": row["id"],
                    "file_path": old_path,
                    "relative_path": new_path,
                    "filename": metadata["filename"],
                    "base_model_code": metadata["base_model_code"],
                    "category_code": metadata["category_code"],
                }
            )

    new_insert_candidates: list[dict[str, Any]] = []
    unparseable_missing_files: list[dict[str, Any]] = []
    for relative in audit["mounted_files_missing_from_db"]["all"]:
        if _path_key(relative) in relocation_new_paths:
            continue
        metadata = _parse_relative_metadata(relative, root)
        if metadata["base_model_code"] and metadata["category_code"]:
            candidate = {
                **metadata,
                "source_type": "new_metadata_insert",
                "support_scope": "metadata-only unless the existing indexer explicitly supports this family",
            }
            new_insert_candidates.append(candidate)
            pending_ids.append(candidate)
        else:
            unparseable_missing_files.append(
                {
                    **metadata,
                    "reason": "missing registered base-model code or recognised category code",
                }
            )

    mounted_metadata_backfill_candidates: list[dict[str, Any]] = []
    mounted_existing_rows_missing_ids: list[dict[str, Any]] = []
    for row in rows:
        canonical = canonical_db_relative_path(
            str(row.get("file_path") or ""),
            root_dir=str(root),
            top_level_folders=top_level_folders,
        )
        if canonical is None or canonical not in mounted_keys:
            continue

        relative = library["mounted_files"][canonical]
        parsed = _parse_relative_metadata(relative, root)
        parsed_base_code = str(parsed.get("base_model_code") or "").strip().upper()
        parsed_category_code = str(parsed.get("category_code") or "").strip().upper()
        db_base_code = str(row.get("base_model_code") or "").strip().upper()
        db_category_code = str(row.get("category_code") or "").strip().upper()

        changed_fields: dict[str, dict[str, Any]] = {}
        comparisons = (
            ("base_model_name", row.get("base_model_name"), parsed.get("base_model_name")),
            ("base_model_code", row.get("base_model_code"), parsed.get("base_model_code")),
            ("category_name", row.get("category_name"), parsed.get("category_name")),
            ("category_code", row.get("category_code"), parsed.get("category_code")),
        )
        for field, old_value, new_value in comparisons:
            old_text = str(old_value or "").strip()
            new_text = str(new_value or "").strip()
            if new_text and old_text != new_text:
                changed_fields[field] = {"from": old_value, "to": new_value}

        if changed_fields:
            mounted_metadata_backfill_candidates.append(
                {
                    "row_id": row["id"],
                    "stable_id": str(row.get("stable_id") or "").strip() or None,
                    "file_path": row["file_path"],
                    "relative_path": relative,
                    "changed_fields": changed_fields,
                    "parsed_base_model_code": parsed_base_code or None,
                    "parsed_category_code": parsed_category_code or None,
                    "review_class": "mounted metadata backfill",
                }
            )

        if str(row.get("stable_id") or "").strip():
            continue
        if not parsed_base_code or not parsed_category_code:
            continue
        item = {
            "source_type": "existing_mounted_row_missing_id",
            "row_id": row["id"],
            "file_path": row["file_path"],
            "relative_path": relative,
            "filename": row.get("filename") or PurePosixPath(canonical).name,
            "base_model_code": parsed_base_code,
            "category_code": parsed_category_code,
            "metadata_source": "registry-backed path parser",
            "current_base_model_code": db_base_code or None,
            "current_category_code": db_category_code or None,
        }
        mounted_existing_rows_missing_ids.append(item)
        pending_ids.append(item)

    planned_ids, exhausted_id_groups, existing_id_issues = _allocate_stable_ids(rows, pending_ids)

    inserts_by_family = Counter(
        str(item.get("base_model_code") or "UNKNOWN") for item in new_insert_candidates
    )
    backfills_by_family = Counter(
        str(item.get("parsed_base_model_code") or "UNKNOWN")
        for item in mounted_metadata_backfill_candidates
    )
    plan = {
        "audit_mode": "read-only",
        "identity_limit": (
            "Relocation candidates use filename equality only. Old file bytes are unavailable, so content identity is not proven."
        ),
        "summary": {
            "mounted_safetensors": audit["summary"]["mounted_safetensors"],
            "db_rows": audit["summary"]["db_rows"],
            "same_family_relocation_candidates": len(same_family_relocations),
            "cross_family_reclassification_candidates": len(cross_family_reclassifications),
            "unresolved_relocation_candidates": len(unresolved_relocations),
            "new_metadata_insert_candidates": len(new_insert_candidates),
            "mounted_metadata_backfill_candidates": len(mounted_metadata_backfill_candidates),
            "unparseable_missing_files": len(unparseable_missing_files),
            "mounted_existing_rows_missing_stable_id": len(mounted_existing_rows_missing_ids),
            "planned_stable_ids": len(planned_ids),
            "stable_id_groups_exhausted": len(exhausted_id_groups),
            "existing_stable_id_issues": len(existing_id_issues),
            "untouched_stale_current_family_rows": (
                audit["stale_db_paths"]["count"] - len(relocation_report["unique_matches"])
            ),
            "untouched_legacy_unmounted_rows": audit["unresolved_db_paths"]["count"],
        },
        "new_inserts_by_base_code": dict(sorted(inserts_by_family.items())),
        "metadata_backfills_by_base_code": dict(sorted(backfills_by_family.items())),
        "same_family_relocations": same_family_relocations,
        "cross_family_reclassifications": cross_family_reclassifications,
        "unresolved_relocations": unresolved_relocations,
        "new_metadata_insert_candidates": new_insert_candidates,
        "mounted_metadata_backfill_candidates": mounted_metadata_backfill_candidates,
        "unparseable_missing_files": unparseable_missing_files,
        "mounted_existing_rows_missing_stable_id": mounted_existing_rows_missing_ids,
        "planned_stable_ids": planned_ids,
        "stable_id_groups_exhausted": exhausted_id_groups,
        "existing_stable_id_issues": existing_id_issues,
        "safety": {
            "database_open_mode": "SQLite URI mode=ro plus PRAGMA query_only=ON",
            "opens_safetensors": False,
            "writes_database": False,
            "runs_indexer": False,
            "assigns_stable_ids": False,
            "deletes_stale_rows": False,
        },
    }
    return plan


def print_plan(plan: Mapping[str, Any], sample_limit: int = 20) -> None:
    summary = plan["summary"]
    print("=== Phase 8.9d controlled indexing plan ===")
    print(f"Mode                                  : {plan['audit_mode']}")
    print(f"Mounted .safetensors                  : {summary['mounted_safetensors']}")
    print(f"DB rows                               : {summary['db_rows']}")
    print(f"Same-family relocation candidates     : {summary['same_family_relocation_candidates']}")
    print(f"Cross-family reclassification reviews : {summary['cross_family_reclassification_candidates']}")
    print(f"New metadata insert candidates        : {summary['new_metadata_insert_candidates']}")
    print(f"Mounted metadata backfill candidates  : {summary['mounted_metadata_backfill_candidates']}")
    print(f"Unparseable missing files             : {summary['unparseable_missing_files']}")
    print(f"Existing mounted rows missing IDs     : {summary['mounted_existing_rows_missing_stable_id']}")
    print(f"Planned stable IDs                    : {summary['planned_stable_ids']}")
    print(f"Untouched stale current-family rows   : {summary['untouched_stale_current_family_rows']}")
    print(f"Untouched legacy/unmounted rows       : {summary['untouched_legacy_unmounted_rows']}")

    print("\n=== New inserts by base code ===")
    for code, count in plan["new_inserts_by_base_code"].items():
        print(f"{count:4}  {code}")

    print("\n=== Metadata backfills by base code ===")
    for code, count in plan["metadata_backfills_by_base_code"].items():
        print(f"{count:4}  {code}")

    print("\n=== Cross-family review sample ===")
    for item in plan["cross_family_reclassifications"][: max(sample_limit, 0)]:
        print(f"{item['old_path']} -> {item['new_path']}")
        print(f"  stable_id: {item.get('stable_id') or '(missing)'}")
        print(f"  policy: {item['stable_id_policy']}")

    print("\nNo database changes were made.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a read-only Phase 8.9d plan for controlled LoRA index reconciliation."
    )
    parser.add_argument("--root", required=True, help="Mounted LoRA root")
    parser.add_argument("--db", required=True, help="SQLite database path")
    parser.add_argument("--json", dest="json_path", help="Optional output path for the full plan")
    parser.add_argument("--sample-limit", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_index_plan(
        root_dir=args.root,
        db_path=args.db,
        sample_limit=args.sample_limit,
    )
    print_plan(plan, sample_limit=args.sample_limit)
    if args.json_path:
        output = Path(args.json_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
        print(f"JSON plan written to: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
