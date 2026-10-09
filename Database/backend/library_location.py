"""Per-database library location; changes apply together at the next startup."""
import os
from pathlib import Path
import stat

from catalogue_refresh import CatalogueError, reparse


def initialise_location_schema(conn):
    with conn:
        conn.execute('CREATE TABLE IF NOT EXISTS lora_library_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)')


def selected_location(conn, fallback):
    row = conn.execute("SELECT value FROM lora_library_settings WHERE key='root'").fetchone()
    return Path(row[0]).absolute() if row else Path(fallback).absolute()


def validate_location(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 4096 or '\0' in value:
        raise CatalogueError('invalid_root', 'Enter the full path to an existing LoRA folder.')
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        raise CatalogueError('invalid_root', 'Use an absolute folder path.')
    path = Path(os.path.abspath(path))
    try:
        for item in (path, *path.parents):
            info = item.lstat()
            if reparse(info):
                raise CatalogueError('invalid_root', 'Choose a real folder without symbolic links or junctions.')
        if not stat.S_ISDIR(path.lstat().st_mode):
            raise CatalogueError('invalid_root', 'The library path must be a folder.')
        with os.scandir(path) as entries:
            next(entries, None)
    except OSError as exc:
        raise CatalogueError('root_unavailable', 'This folder cannot be read. Check the path and access before saving it.', 422) from exc
    return path
