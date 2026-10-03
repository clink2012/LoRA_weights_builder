"""Snapshot local application state; restore only to a new, separate directory."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import stat
import subprocess


PROJECT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_database(path: Path) -> dict:
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        integrity = [row[0] for row in conn.execute("PRAGMA integrity_check")]
        if integrity != ["ok"]:
            raise ValueError("SQLite integrity check failed.")
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]
        counts = {name: conn.execute('SELECT COUNT(*) FROM "' + name.replace('"', '""') + '"').fetchone()[0]
                  for name in tables}
    return {"integrity": "ok", "table_counts": counts}


def selected_database(project: Path, database: Path | None) -> Path:
    candidate = project / "Database" / "lora_master.db" if database is None else database
    if not candidate.is_absolute():
        candidate = project / candidate
    # Check the original components before resolve() can hide symlinks or junctions.
    for component in (candidate, *candidate.parents):
        try:
            info = component.lstat()
        except FileNotFoundError:
            raise ValueError("Expected a regular existing application database.") from None
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Application database paths must not use links or junctions.")
    source = candidate.resolve()
    if not source.is_relative_to(project) or not source.is_file():
        raise ValueError("Expected a regular existing application database within the project.")
    return source


def snapshot(project: Path, destination: Path, database: Path | None = None) -> dict:
    project, destination = project.resolve(), destination.resolve()
    source = selected_database(project, database)
    destination.mkdir(parents=True, exist_ok=False)
    target = destination / "Database" / "lora_master.db"
    target.parent.mkdir()
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original:
        with closing(sqlite3.connect(target)) as backup:
            original.backup(backup)
    database = inspect_database(target)
    candidates = list((project / "Database" / "backend" / "profiles").rglob("*.json"))
    questionnaire = project / ".local" / "gui-questionnaire"
    if (questionnaire / "answers.json").is_file():
        candidates.append(questionnaire / "answers.json")
    candidates.extend((questionnaire / "submissions").glob("*.json"))
    for path in candidates:
        if path.is_symlink() or not path.resolve().is_relative_to(project):
            raise ValueError("State files must remain within the project.")
        output = destination / path.relative_to(project)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output)
    revision = subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=False)
    files = {p.relative_to(destination).as_posix(): digest(p)
             for p in destination.rglob("*") if p.is_file()}
    manifest = {"format": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "source_revision": revision.stdout.strip() if revision.returncode == 0 else None,
                "source_database": str(source),
                "source_database_relative": source.relative_to(project).as_posix(),
                "database": database, "files": files,
                "scope": "SQLite, external profile JSONs and questionnaire responses; not model binaries or a source/environment backup"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify(snapshot_path: Path) -> dict:
    root = snapshot_path.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported snapshot manifest.")
    for relative, expected in manifest["files"].items():
        path = root / relative
        if Path(relative).is_absolute() or not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError("Snapshot path escapes its directory.")
        if not path.is_file() or digest(path) != expected:
            raise ValueError("Snapshot file is missing or has changed: " + relative)
    database_path = "Database/lora_master.db"
    if database_path not in manifest["files"]:
        raise ValueError("Snapshot must contain the application database.")
    if inspect_database(root / database_path) != manifest["database"]:
        raise ValueError("Snapshot database does not match its receipt.")
    return manifest


def restore(snapshot_path: Path, destination: Path) -> dict:
    manifest = verify(snapshot_path)
    destination = destination.resolve()
    # Refuse existing locations, including the application, rather than overlaying data.
    destination.mkdir(parents=True, exist_ok=False)
    for relative in manifest["files"]:
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot_path / relative, output)
    shutil.copy2(snapshot_path / "manifest.json", destination / "manifest.json")
    verify(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("snapshot")
    create.add_argument("destination", type=Path)
    create.add_argument("--project", type=Path, default=PROJECT)
    create.add_argument("--database", type=Path,
                        help="Existing database within the project; relative paths use the project root. Defaults to Database/lora_master.db.")
    check = sub.add_parser("verify")
    check.add_argument("snapshot", type=Path)
    recover = sub.add_parser("restore")
    recover.add_argument("snapshot", type=Path)
    recover.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.command == "snapshot":
        result = snapshot(args.project, args.destination, args.database)
    elif args.command == "restore":
        result = restore(args.snapshot, args.destination)
    else:
        result = verify(args.snapshot)
    print(json.dumps({"status": "verified", "database": result["database"],
                      "source_database": result.get("source_database"),
                      "file_count": len(result["files"])}, indent=2))


if __name__ == "__main__":
    main()
