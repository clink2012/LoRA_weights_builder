from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_AUDIT_JSON = "/home/clink/docker/lora_builder/data/phase89_audit.json"


def _normalise_path(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


def _filename_key(path: Any) -> str:
    return PurePosixPath(_normalise_path(path)).name.casefold()


def _mounted_family(path: Any) -> str:
    parts = [part for part in _normalise_path(path).split("/") if part]
    return parts[0] if parts else "(root)"


def _db_family(path: Any) -> str:
    parts = [part for part in _normalise_path(path).split("/") if part]
    for index, part in enumerate(parts):
        if part.casefold() == "loras" and index + 1 < len(parts):
            return parts[index + 1]
    return "(unresolved)"


def _as_all_paths(report: Mapping[str, Any], key: str) -> list[str]:
    section = report.get(key) or {}
    values = section.get("all") if isinstance(section, Mapping) else []
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def analyse_relocations(report: Mapping[str, Any]) -> dict[str, Any]:
    """Compare stale DB paths with mounted files absent from the DB by filename.

    This is deliberately conservative. A relocation candidate is only considered
    unique when exactly one stale DB path and exactly one missing mounted path have
    the same case-insensitive filename. File contents are never opened.
    """
    stale_paths = _as_all_paths(report, "stale_db_paths")
    missing_paths = _as_all_paths(report, "mounted_files_missing_from_db")
    unresolved_paths = _as_all_paths(report, "unresolved_db_paths")

    stale_by_filename: dict[str, list[str]] = defaultdict(list)
    missing_by_filename: dict[str, list[str]] = defaultdict(list)

    for path in stale_paths:
        stale_by_filename[_filename_key(path)].append(path)
    for path in missing_paths:
        missing_by_filename[_filename_key(path)].append(path)

    unique_matches: list[dict[str, str]] = []
    ambiguous_matches: list[dict[str, Any]] = []

    shared_filenames = sorted(set(stale_by_filename) & set(missing_by_filename))
    for filename_key in shared_filenames:
        old_paths = sorted(stale_by_filename[filename_key], key=str.casefold)
        new_paths = sorted(missing_by_filename[filename_key], key=str.casefold)
        if len(old_paths) == 1 and len(new_paths) == 1:
            old_path = old_paths[0]
            new_path = new_paths[0]
            unique_matches.append(
                {
                    "filename": PurePosixPath(_normalise_path(new_path)).name,
                    "old_path": old_path,
                    "new_path": new_path,
                    "from_family": _db_family(old_path),
                    "to_family": _mounted_family(new_path),
                }
            )
        else:
            ambiguous_matches.append(
                {
                    "filename": filename_key,
                    "old_paths": old_paths,
                    "new_paths": new_paths,
                }
            )

    transition_counts = Counter(
        (item["from_family"], item["to_family"]) for item in unique_matches
    )
    transitions = [
        {"from_family": old, "to_family": new, "count": count}
        for (old, new), count in sorted(
            transition_counts.items(),
            key=lambda item: (-item[1], item[0][0].casefold(), item[0][1].casefold()),
        )
    ]

    legacy_counts = Counter(_db_family(path) for path in unresolved_paths)
    legacy_families = [
        {"family": family, "count": count}
        for family, count in sorted(
            legacy_counts.items(), key=lambda item: (-item[1], item[0].casefold())
        )
    ]

    matched_old_paths = {item["old_path"] for item in unique_matches}
    matched_new_paths = {item["new_path"] for item in unique_matches}

    return {
        "audit_mode": "read-only",
        "matching_rule": "case-insensitive exact filename; one stale path to one missing mounted path",
        "summary": {
            "stale_current_family_rows": len(stale_paths),
            "mounted_files_missing_from_db": len(missing_paths),
            "unique_filename_matches": len(unique_matches),
            "ambiguous_filename_matches": len(ambiguous_matches),
            "unmatched_stale_rows": len(stale_paths) - len(matched_old_paths),
            "unmatched_missing_files": len(missing_paths) - len(matched_new_paths),
            "legacy_unmounted_rows": len(unresolved_paths),
        },
        "transitions": transitions,
        "unique_matches": unique_matches,
        "ambiguous_matches": ambiguous_matches,
        "legacy_families": legacy_families,
    }


def print_report(report: Mapping[str, Any], sample_limit: int = 30) -> None:
    summary = report["summary"]
    print("=== Phase 8.9 relocation audit ===")
    print(f"Mode                         : {report['audit_mode']}")
    print(f"Matching rule                : {report['matching_rule']}")
    print(f"Stale current-family DB rows : {summary['stale_current_family_rows']}")
    print(f"Mounted files missing in DB  : {summary['mounted_files_missing_from_db']}")
    print(f"Unique filename matches      : {summary['unique_filename_matches']}")
    print(f"Ambiguous filename matches   : {summary['ambiguous_filename_matches']}")
    print(f"Unmatched stale DB rows      : {summary['unmatched_stale_rows']}")
    print(f"Unmatched mounted files      : {summary['unmatched_missing_files']}")
    print(f"Legacy/unmounted DB rows     : {summary['legacy_unmounted_rows']}")

    print("\n=== Likely family/folder transitions ===")
    for item in report["transitions"]:
        print(f"{item['count']:4}  {item['from_family']} -> {item['to_family']}")

    print("\n=== Sample unique relocation candidates ===")
    for item in report["unique_matches"][: max(sample_limit, 0)]:
        print(f"OLD: {item['old_path']}")
        print(f"NEW: {item['new_path']}")
        print()

    print("=== Legacy/unmounted DB families ===")
    for item in report["legacy_families"]:
        print(f"{item['count']:4}  {item['family']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only exact-filename relocation analysis for a Phase 8.9 audit JSON report."
    )
    parser.add_argument(
        "--audit-json",
        default=DEFAULT_AUDIT_JSON,
        help="JSON produced by phase89_audit.py",
    )
    parser.add_argument("--json", dest="json_path", help="Optional output path for relocation JSON")
    parser.add_argument("--sample-limit", type=int, default=30, help="Console relocation sample size")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.audit_json).expanduser().resolve(strict=True)
    audit_report = json.loads(source.read_text(encoding="utf-8"))
    report = analyse_relocations(audit_report)
    print_report(report, sample_limit=args.sample_limit)

    if args.json_path:
        output = Path(args.json_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nJSON report written to: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
