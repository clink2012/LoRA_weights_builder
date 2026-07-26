from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

DEFAULT_LORA_ROOT = os.environ.get("LORA_ROOT", r"E:\models\loras")
DEFAULT_DB_PATH = os.environ.get("LORA_DB_PATH", r"E:\LoRA Project\Database\lora_master.db")
DEFAULT_IGNORED_FOLDERS = ("LoRA_Manager_Images", "recipes")


def _normalise_separators(value: str) -> str:
    value = str(value or "").strip().replace("\\", "/")
    return re.sub(r"/+", "/", value)


def _casefold_path(value: str) -> str:
    return _normalise_separators(value).strip("/").casefold()


def _path_parts(value: str) -> list[str]:
    normalised = _normalise_separators(value)
    return [part for part in normalised.split("/") if part not in ("", ".")]


def _top_level_lookup(folder_names: Iterable[str]) -> dict[str, str]:
    return {name.casefold(): name for name in folder_names}


def canonical_db_relative_path(
    file_path: str,
    *,
    root_dir: str,
    top_level_folders: Sequence[str],
) -> str | None:
    """Map a DB path to a case-insensitive path relative to the mounted root.

    Production DB rows may contain Windows paths while the live mount is POSIX.
    First try to strip the configured root. If that cannot work because the DB
    was created on a different host, anchor the path at a known top-level folder.
    """
    raw = _normalise_separators(file_path)
    if not raw:
        return None

    root = _normalise_separators(root_dir).rstrip("/")
    raw_fold = raw.casefold()
    root_fold = root.casefold()
    if raw_fold == root_fold:
        return ""
    if root and raw_fold.startswith(root_fold + "/"):
        return _casefold_path(raw[len(root) + 1 :])

    lookup = _top_level_lookup(top_level_folders)
    parts = _path_parts(raw)
    for index, part in enumerate(parts):
        canonical_top = lookup.get(part.casefold())
        if canonical_top is None:
            continue
        relative_parts = [canonical_top, *parts[index + 1 :]]
        return _casefold_path("/".join(relative_parts))

    return None


def open_read_only_db(db_path: str | os.PathLike[str]) -> sqlite3.Connection:
    resolved = Path(db_path).expanduser().resolve(strict=True)
    uri_path = quote(resolved.as_posix(), safe="/:")
    conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON;")
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table});")}
    except sqlite3.DatabaseError:
        return set()


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def discover_library(
    root_dir: str | os.PathLike[str],
    ignored_folders: Iterable[str],
) -> dict[str, Any]:
    root = Path(root_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"LoRA root is not a directory: {root}")

    ignored = {name.casefold() for name in ignored_folders}
    top_level_folders: list[str] = []
    ignored_present: list[str] = []
    mounted_files: dict[str, str] = {}
    top_level_file_counts: Counter[str] = Counter()
    root_level_safetensors: list[str] = []

    for child in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if child.is_dir():
            if child.name.casefold() in ignored:
                ignored_present.append(child.name)
                continue
            top_level_folders.append(child.name)
        elif child.is_file() and child.suffix.casefold() == ".safetensors":
            relative = child.relative_to(root).as_posix()
            canonical = _casefold_path(relative)
            mounted_files[canonical] = relative
            root_level_safetensors.append(relative)
            top_level_file_counts["(root)"] += 1

    for folder_name in top_level_folders:
        folder = root / folder_name
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames[:] = sorted(dirnames, key=str.casefold)
            for filename in sorted(filenames, key=str.casefold):
                if not filename.casefold().endswith(".safetensors"):
                    continue
                path = Path(dirpath) / filename
                relative = path.relative_to(root).as_posix()
                canonical = _casefold_path(relative)
                mounted_files[canonical] = relative
                top_level_file_counts[folder_name] += 1

    return {
        "root": str(root),
        "top_level_folders": top_level_folders,
        "ignored_folders_present": ignored_present,
        "mounted_files": mounted_files,
        "top_level_file_counts": dict(top_level_file_counts),
        "root_level_safetensors": root_level_safetensors,
    }


def _load_block_counts(conn: sqlite3.Connection) -> dict[int, int]:
    columns = _table_columns(conn, "lora_block_weights")
    if "lora_id" not in columns:
        return {}
    return {
        int(row["lora_id"]): int(row["block_count"] or 0)
        for row in conn.execute(
            """
            SELECT lora_id, COUNT(1) AS block_count
            FROM lora_block_weights
            GROUP BY lora_id
            """
        )
    }


def _classification(row: Mapping[str, Any], block_count: int) -> str:
    has_flag = bool(_safe_int(row.get("has_block_weights")))
    layout = str(row.get("block_layout") or "").strip()
    if block_count > 0 and has_flag:
        return "scanned"
    if has_flag and block_count == 0:
        return "flagged_missing_blocks"
    if block_count > 0 and not has_flag:
        return "blocks_without_flag"
    if layout:
        return "fallback_only"
    return "metadata_only"


def _support_status(counts: Mapping[str, int], db_rows: int) -> str:
    if db_rows == 0:
        return "unindexed"
    scanned = counts.get("scanned", 0)
    fallback = counts.get("fallback_only", 0)
    metadata = counts.get("metadata_only", 0)
    inconsistent = counts.get("flagged_missing_blocks", 0) + counts.get("blocks_without_flag", 0)
    if inconsistent:
        return "inconsistent"
    if scanned and scanned == db_rows:
        return "scanned"
    if scanned:
        return "mixed"
    if fallback and fallback == db_rows:
        return "fallback-only"
    if metadata and metadata == db_rows:
        return "metadata-only"
    if fallback or metadata:
        return "mixed-unscanned"
    return "unknown"


def _display_list(values: Iterable[str], limit: int) -> dict[str, Any]:
    ordered = sorted(values, key=str.casefold)
    return {"count": len(ordered), "sample": ordered[:limit], "all": ordered}


def run_audit(
    *,
    root_dir: str | os.PathLike[str],
    db_path: str | os.PathLike[str],
    ignored_folders: Iterable[str] = DEFAULT_IGNORED_FOLDERS,
    sample_limit: int = 20,
) -> dict[str, Any]:
    library = discover_library(root_dir, ignored_folders)
    ignored = {name.casefold() for name in ignored_folders}
    comparison_folders = [*library["top_level_folders"], *library["ignored_folders_present"]]
    mounted_files: dict[str, str] = library["mounted_files"]
    mounted_set = set(mounted_files)

    conn = open_read_only_db(db_path)
    try:
        lora_columns = _table_columns(conn, "lora")
        required = {"id", "file_path"}
        missing_required = sorted(required - lora_columns)
        if missing_required:
            raise RuntimeError(f"lora table is missing required column(s): {', '.join(missing_required)}")

        select_columns = [
            name
            for name in (
                "id",
                "file_path",
                "filename",
                "base_model_name",
                "base_model_code",
                "stable_id",
                "has_block_weights",
                "block_layout",
                "model_family",
                "lora_type",
            )
            if name in lora_columns
        ]
        rows = [dict(row) for row in conn.execute(f"SELECT {', '.join(select_columns)} FROM lora ORDER BY id")]
        block_counts = _load_block_counts(conn)
    finally:
        conn.close()

    db_by_canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unresolved_db_rows: list[str] = []
    ignored_db_rows: list[str] = []
    group_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    folder_stats: dict[str, dict[str, Any]] = {}
    top_lookup = _top_level_lookup(library["top_level_folders"])

    for folder in library["top_level_folders"]:
        folder_stats[folder] = {
            "folder": folder,
            "mounted_files": _safe_int(library["top_level_file_counts"].get(folder)),
            "db_rows": 0,
            "with_stable_id": 0,
            "without_stable_id": 0,
            "classifications": Counter(),
            "observed_base_model_names": set(),
            "observed_base_model_codes": set(),
            "stale_db_rows": 0,
            "missing_from_db": 0,
        }

    overall_classifications: Counter[str] = Counter()
    with_stable_id = 0

    for row in rows:
        row_id = _safe_int(row.get("id"))
        block_count = block_counts.get(row_id, 0)
        classification = _classification(row, block_count)
        overall_classifications[classification] += 1
        stable_id = str(row.get("stable_id") or "").strip()
        if stable_id:
            with_stable_id += 1

        name = str(row.get("base_model_name") or "NULL")
        code = str(row.get("base_model_code") or "NULL")
        group_counts[(name, code)]["rows"] += 1
        group_counts[(name, code)][classification] += 1
        group_counts[(name, code)]["with_stable_id" if stable_id else "without_stable_id"] += 1

        canonical = canonical_db_relative_path(
            str(row.get("file_path") or ""),
            root_dir=str(root_dir),
            top_level_folders=comparison_folders,
        )
        if canonical is None:
            unresolved_db_rows.append(str(row.get("file_path") or ""))
            continue

        top_part = canonical.split("/", 1)[0]
        if top_part in ignored:
            ignored_db_rows.append(str(row.get("file_path") or ""))
            continue

        db_by_canonical[canonical].append(row)
        folder_name = top_lookup.get(top_part)
        if folder_name is None:
            continue
        stats = folder_stats[folder_name]
        stats["db_rows"] += 1
        stats["with_stable_id" if stable_id else "without_stable_id"] += 1
        stats["classifications"][classification] += 1
        if row.get("base_model_name"):
            stats["observed_base_model_names"].add(str(row["base_model_name"]))
        if row.get("base_model_code"):
            stats["observed_base_model_codes"].add(str(row["base_model_code"]))

    db_set = set(db_by_canonical)
    missing_from_db_keys = mounted_set - db_set
    stale_db_keys = db_set - mounted_set
    duplicate_db_keys = {key for key, matching_rows in db_by_canonical.items() if len(matching_rows) > 1}

    for canonical in missing_from_db_keys:
        top_part = canonical.split("/", 1)[0]
        folder_name = top_lookup.get(top_part)
        if folder_name:
            folder_stats[folder_name]["missing_from_db"] += 1
    for canonical in stale_db_keys:
        top_part = canonical.split("/", 1)[0]
        folder_name = top_lookup.get(top_part)
        if folder_name:
            folder_stats[folder_name]["stale_db_rows"] += len(db_by_canonical[canonical])

    matrix: list[dict[str, Any]] = []
    for folder in library["top_level_folders"]:
        stats = folder_stats[folder]
        classifications = dict(sorted(stats["classifications"].items()))
        matrix.append(
            {
                "folder": folder,
                "mounted_files": stats["mounted_files"],
                "db_rows": stats["db_rows"],
                "with_stable_id": stats["with_stable_id"],
                "without_stable_id": stats["without_stable_id"],
                "classifications": classifications,
                "support_status": _support_status(classifications, stats["db_rows"]),
                "observed_base_model_names": sorted(stats["observed_base_model_names"], key=str.casefold),
                "observed_base_model_codes": sorted(stats["observed_base_model_codes"], key=str.casefold),
                "stale_db_rows": stats["stale_db_rows"],
                "missing_from_db": stats["missing_from_db"],
            }
        )

    base_model_groups = []
    for (name, code), counts in sorted(group_counts.items(), key=lambda item: (item[0][1], item[0][0])):
        base_model_groups.append(
            {
                "base_model_name": name,
                "base_model_code": code,
                **dict(sorted(counts.items())),
            }
        )

    missing_display = [mounted_files[key] for key in missing_from_db_keys]
    stale_display = [db_by_canonical[key][0].get("file_path") or key for key in stale_db_keys]
    duplicate_display = [
        {
            "canonical_path": key,
            "db_row_ids": [_safe_int(row.get("id")) for row in db_by_canonical[key]],
        }
        for key in sorted(duplicate_db_keys)
    ]

    return {
        "audit_mode": "read-only",
        "comparison_mode": "canonical relative path (Windows/POSIX neutral)",
        "root_dir": str(Path(root_dir).expanduser().resolve()),
        "db_path": str(Path(db_path).expanduser().resolve()),
        "ignored_folders": sorted(set(ignored_folders), key=str.casefold),
        "top_level_folders": library["top_level_folders"],
        "ignored_folders_present": library["ignored_folders_present"],
        "summary": {
            "mounted_safetensors": len(mounted_set),
            "db_rows": len(rows),
            "with_stable_id": with_stable_id,
            "without_stable_id": len(rows) - with_stable_id,
            "classifications": dict(sorted(overall_classifications.items())),
            "stale_db_rows": sum(len(db_by_canonical[key]) for key in stale_db_keys),
            "mounted_files_missing_from_db": len(missing_from_db_keys),
            "unresolved_db_paths": len(unresolved_db_rows),
            "ignored_db_rows": len(ignored_db_rows),
            "duplicate_canonical_db_paths": len(duplicate_db_keys),
        },
        "base_model_groups": base_model_groups,
        "support_matrix": matrix,
        "stale_db_paths": _display_list(stale_display, sample_limit),
        "mounted_files_missing_from_db": _display_list(missing_display, sample_limit),
        "unresolved_db_paths": _display_list(unresolved_db_rows, sample_limit),
        "ignored_db_paths": _display_list(ignored_db_rows, sample_limit),
        "duplicate_canonical_db_paths": {
            "count": len(duplicate_display),
            "sample": duplicate_display[:sample_limit],
            "all": duplicate_display,
        },
        "root_level_safetensors": library["root_level_safetensors"],
    }


def _print_matrix(report: Mapping[str, Any]) -> None:
    print("\n=== Model ecosystem support matrix ===")
    header = f"{'Folder':<22} {'Mounted':>7} {'DB':>7} {'IDs':>7} {'Scanned':>8} {'Fallback':>9} {'Metadata':>9} {'Missing':>8} {'Stale':>7}  Status"
    print(header)
    print("-" * len(header))
    for row in report["support_matrix"]:
        counts = row["classifications"]
        print(
            f"{row['folder']:<22} "
            f"{row['mounted_files']:>7} "
            f"{row['db_rows']:>7} "
            f"{row['with_stable_id']:>7} "
            f"{counts.get('scanned', 0):>8} "
            f"{counts.get('fallback_only', 0):>9} "
            f"{counts.get('metadata_only', 0):>9} "
            f"{row['missing_from_db']:>8} "
            f"{row['stale_db_rows']:>7}  "
            f"{row['support_status']}"
        )


def print_report(report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    print("=== Phase 8.9 read-only audit ===")
    print(f"LoRA root : {report['root_dir']}")
    print(f"Database  : {report['db_path']}")
    print(f"Mode      : {report['audit_mode']}")
    print(f"Compare   : {report['comparison_mode']}")
    print()
    print(f"Mounted .safetensors          : {summary['mounted_safetensors']}")
    print(f"DB rows                      : {summary['db_rows']}")
    print(f"With stable_id               : {summary['with_stable_id']}")
    print(f"Without stable_id            : {summary['without_stable_id']}")
    print(f"DB rows missing on mount     : {summary['stale_db_rows']}")
    print(f"Mounted files missing in DB  : {summary['mounted_files_missing_from_db']}")
    print(f"Unresolved DB paths          : {summary['unresolved_db_paths']}")
    print(f"Ignored DB rows              : {summary['ignored_db_rows']}")
    print(f"Duplicate canonical DB paths : {summary['duplicate_canonical_db_paths']}")
    print("Classifications              : " + json.dumps(summary["classifications"], sort_keys=True))
    _print_matrix(report)

    for key, title in (
        ("stale_db_paths", "DB rows whose mounted file no longer exists"),
        ("mounted_files_missing_from_db", "Mounted files missing from DB"),
        ("unresolved_db_paths", "DB paths that could not be matched to a mounted family"),
    ):
        section = report[key]
        print(f"\n=== {title} ({section['count']}) ===")
        for value in section["sample"]:
            print(f"  {value}")
        if section["count"] > len(section["sample"]):
            print(f"  ... {section['count'] - len(section['sample'])} more in JSON output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only Phase 8.9 audit of mounted LoRA files and the current SQLite index."
    )
    parser.add_argument("--root", default=DEFAULT_LORA_ROOT, help="Mounted LoRA library root")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Current lora_master.db path")
    parser.add_argument(
        "--ignore",
        action="append",
        default=None,
        help="Top-level folder to ignore; repeat for multiple folders",
    )
    parser.add_argument("--json", dest="json_path", help="Optional path for the full JSON report")
    parser.add_argument("--sample-limit", type=int, default=20, help="Console sample size per discrepancy list")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ignored = args.ignore if args.ignore is not None else list(DEFAULT_IGNORED_FOLDERS)
    report = run_audit(
        root_dir=args.root,
        db_path=args.db,
        ignored_folders=ignored,
        sample_limit=max(args.sample_limit, 0),
    )
    print_report(report)
    if args.json_path:
        output = Path(args.json_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nJSON report written to: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
