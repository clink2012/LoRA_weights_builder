"""Path/stat inventory only: preserve catalogue identity and all saved history."""
from __future__ import annotations

from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import time
from uuid import uuid4

from lora_path_parser import parse_base_and_category
from analysis_job_service import WorkerLease, JobError

EXTENSION = '.safetensors'
IGNORED = {'lora_manager_images', 'recipes'}
MAX_FILES = 10000
MAX_DIRECTORIES = 10000
MAX_ENTRIES = 100000
MAX_SECONDS = 60
ROLE_HINTS = {'PPL': 'character', 'CHT': 'character', 'BDY': 'character', 'STL': 'style',
              'CLT': 'clothing', 'ACT': 'pose', 'UTL': 'utility', 'BLD': 'environment', 'NAT': 'environment'}
HISTORY_TABLES = ('lora', 'lora_block_weights', 'lora_clink_overrides', 'lora_user_profiles',
                  'lora_profile_versions', 'lora_profile_selections')


class CatalogueError(ValueError):
    def __init__(self, code, message, status=422):
        self.code, self.status = code, status
        super().__init__(message)


def path_key(path):
    return os.path.abspath(os.path.normpath(str(path))).replace('\\', '/').casefold()


def file_identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def directory_identity(value):
    return (value.st_dev, value.st_ino, value.st_mtime_ns)


def reparse(value):
    return stat.S_ISLNK(value.st_mode) or bool(getattr(value, 'st_file_attributes', 0) & 0x400)


def initialise_catalogue_schema(conn):
    if conn.in_transaction:
        raise CatalogueError('active_transaction', 'Catalogue schema setup requires its own transaction.')
    with conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS lora_catalogue_presence (
            lora_id INTEGER PRIMARY KEY, path_key TEXT NOT NULL, presence TEXT NOT NULL,
            file_identity_json TEXT, last_seen_scan_id TEXT, last_checked_scan_id TEXT NOT NULL,
            observed_metadata_json TEXT NOT NULL,
            FOREIGN KEY(lora_id) REFERENCES lora(id)
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS lora_catalogue_scans (
            scan_id TEXT PRIMARY KEY, completed_at TEXT NOT NULL, root TEXT NOT NULL,
            receipt_json TEXT NOT NULL
        )''')


def _deadline(started):
    if time.monotonic() - started > MAX_SECONDS:
        raise CatalogueError('inventory_budget', 'Library discovery exceeded its time limit; no catalogue changes were applied.', 409)


def discover(root, started):
    """Do not follow links or junctions; any inaccessible directory aborts discovery."""
    root = Path(root)
    root_stat = root.lstat()
    if reparse(root_stat) or not stat.S_ISDIR(root_stat.st_mode):
        raise CatalogueError('invalid_root', 'The configured LoRA root must be a real local directory.', 503)
    files, directories = {}, {}
    # Exclusions are configuration, even when an auxiliary folder is absent.
    excluded = [path_key(root / name) for name in sorted(IGNORED)]
    entries_seen = 0
    pending = [root]
    while pending:
        _deadline(started)
        directory = pending.pop()
        before = directory.lstat()
        if reparse(before) or not stat.S_ISDIR(before.st_mode):
            raise CatalogueError('inventory_changed', 'A library directory changed during discovery. Retry refresh.', 409)
        directories[str(directory)] = directory_identity(before)
        if len(directories) > MAX_DIRECTORIES:
            raise CatalogueError('inventory_budget', 'The library exceeds the directory limit; no changes were applied.', 409)
        with os.scandir(directory) as iterator:
            entries = []
            for entry in iterator:
                _deadline(started)
                entries_seen += 1
                if entries_seen > MAX_ENTRIES:
                    raise CatalogueError('inventory_budget', 'The library exceeds the directory-entry limit; no changes were applied.', 409)
                entries.append(entry)
            entries.sort(key=lambda entry: entry.name.casefold())
        for entry in entries:
            _deadline(started)
            path = Path(entry.path)
            # Windows DirEntry.stat can report zero device/inode values; use
            # the same no-follow stat operation as the pre-commit check.
            info = path.lstat()
            if reparse(info):
                excluded.append(path_key(path))
                continue
            if stat.S_ISDIR(info.st_mode):
                if directory == root and entry.name.casefold() in IGNORED:
                    continue
                else:
                    pending.append(path)
            elif stat.S_ISREG(info.st_mode) and path.suffix.casefold() == EXTENSION:
                key = path_key(path)
                if key in files:
                    raise CatalogueError('ambiguous_path', 'Two library paths have the same Windows catalogue identity.', 409)
                files[key] = {'path': str(path), 'identity': file_identity(info)}
                if len(files) > MAX_FILES:
                    raise CatalogueError('inventory_budget', 'The library exceeds the file limit; no changes were applied.', 409)
        if directory_identity(directory.lstat()) != directory_identity(before):
            raise CatalogueError('inventory_changed', 'A library directory changed during discovery. Retry refresh.', 409)
    return {'files': files, 'directories': directories, 'excluded': sorted(excluded)}


def verify_inventory(inventory, started):
    for path, identity in inventory['directories'].items():
        _deadline(started)
        current = Path(path).lstat()
        if reparse(current) or directory_identity(current) != identity:
            raise CatalogueError('inventory_changed', 'A library directory changed before the refresh could be saved.', 409)
    for record in inventory['files'].values():
        _deadline(started)
        current = Path(record['path']).lstat()
        if reparse(current) or file_identity(current) != record['identity']:
            raise CatalogueError('inventory_changed', 'A LoRA changed before the refresh could be saved.', 409)


def _metadata(path, root):
    family, code, category, category_code = parse_base_and_category(str(path), str(root))
    return {'base_model_name': family, 'base_model_code': code, 'category_name': category,
            'category_code': category_code, 'role_hint': ROLE_HINTS.get(category_code, 'other'),
            'basis': 'folder_hint', 'architecture_verified': False}


def _reserved_ids(conn):
    used = set()
    for table in HISTORY_TABLES:
        columns = {row[1] for row in conn.execute(f'PRAGMA table_info({table})')}
        if 'stable_id' in columns:
            used.update(str(row[0]).casefold() for row in conn.execute(f'SELECT stable_id FROM {table} WHERE stable_id IS NOT NULL') if row[0])
    return used


def _allocate(metadata, used):
    prefix = f"{metadata['base_model_code'] or 'UNK'}-{metadata['category_code'] or 'UNK'}"
    pattern = re.compile(re.escape(prefix.casefold()) + r'-(\d+)$')
    maximum = max((int(match.group(1)) for sid in used if (match := pattern.fullmatch(sid))), default=0)
    result = f'{prefix}-{maximum + 1:03d}'
    used.add(result.casefold())
    return result


def _scope(key, root_key, excluded):
    return (key.endswith(EXTENSION) and key.startswith(root_key + '/')
            and not any(key == prefix or key.startswith(prefix + '/') for prefix in excluded))


class CatalogueService:
    def __init__(self, database, root):
        self.database = Path(database).absolute()
        self.root = Path(root).absolute()

    def connection(self):
        conn = sqlite3.connect(self.database.as_uri() + '?mode=rw', uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def freshness(self):
        """Read-only, bounded path/stat comparison against the latest saved inventory.

        Never open model payloads, allocate IDs or mutate presence/history. A
        changing/inaccessible tree cannot receive a reassuring current result.
        """
        started = time.monotonic()
        try:
            first = discover(self.root, started)
            inventory = discover(self.root, started)
            if first != inventory:
                raise CatalogueError('inventory_changed', 'The library changed during the check. Retry the folder check.', 409)
            with closing(self.connection()) as conn:
                conn.execute('BEGIN')
                latest = conn.execute('SELECT scan_id,root FROM lora_catalogue_scans ORDER BY rowid DESC LIMIT 1').fetchone()
                rows = conn.execute('SELECT path_key,presence,file_identity_json FROM lora_catalogue_presence WHERE last_checked_scan_id=?',
                                    (latest['scan_id'],)).fetchall() if latest else []
                known = {row['path_key']: row for row in rows}
                current = inventory['files']
                added = sum(key not in known for key in current)
                returned = sum(key in current and row['presence'] != 'current' for key, row in known.items())
                removed = sum(key not in current and row['presence'] == 'current' for key, row in known.items())
                changed = sum(row['presence'] == 'current' and key in current and
                              tuple(json.loads(row['file_identity_json'] or '[]')) != current[key]['identity']
                              for key, row in known.items())
                root_changed = bool(latest and path_key(latest['root']) != path_key(self.root))
                verify_inventory(inventory, started)
                conn.rollback()
            # A concurrent catalogue refresh cannot lend its old identity to
            # this result. The caller retries once the refresh finishes.
            with closing(self.connection()) as conn:
                after = conn.execute('SELECT scan_id FROM lora_catalogue_scans ORDER BY rowid DESC LIMIT 1').fetchone()
            if (after['scan_id'] if after else None) != (latest['scan_id'] if latest else None):
                raise CatalogueError('scan_superseded', 'The saved inventory changed during the folder check. Retry after the refresh.', 409)
            outdated = not latest or root_changed or any((added, returned, removed, changed))
            return {'status': 'not_scanned' if not latest else 'outdated' if outdated else 'current',
                    'root': str(self.root), 'catalogue_scan_id': latest['scan_id'] if latest else None,
                    'checked_at': datetime.now(timezone.utc).isoformat(), 'root_changed': root_changed,
                    'counts': {'discovered': len(current), 'added': added, 'returned': returned,
                               'removed': removed, 'changed': changed},
                    'basis': 'path_and_file_stat_only', 'tensor_payload_read': False,
                    'architecture_verified': False, 'export_verified': False}
        except CatalogueError:
            raise
        except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
            raise CatalogueError('freshness_unavailable', 'The complete library and saved inventory could not be compared. Retry the folder check.', 503) from exc

    def refresh(self, *, cancelled=None):
        started = time.monotonic()
        try:
            lease = WorkerLease(self.database.with_suffix('.catalogue-refresh.lock'))
        except JobError as exc:
            raise CatalogueError('refresh_busy', 'A library refresh is already running.', 409) from exc
        try:
            if cancelled and cancelled():
                raise CatalogueError('scan_cancelled', 'Library refresh was cancelled before discovery.', 409)
            first = discover(self.root, started)
            if cancelled and cancelled():
                raise CatalogueError('scan_cancelled', 'Library refresh was cancelled; no changes were applied.', 409)
            inventory = discover(self.root, started)
            if first != inventory:
                raise CatalogueError('inventory_changed', 'The library changed during discovery. Retry refresh; no changes were applied.', 409)
            with closing(self.connection()) as conn:
                conn.execute('BEGIN IMMEDIATE')
                try:
                    result = self._apply(conn, inventory, started)
                    if cancelled and cancelled():
                        raise CatalogueError('scan_cancelled', 'Library refresh was cancelled; no changes were applied.', 409)
                    conn.commit()
                    return result
                except BaseException:
                    conn.rollback()
                    raise
        except CatalogueError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise CatalogueError('refresh_unavailable', 'The complete local library or catalogue could not be read. No refresh changes were applied.', 503) from exc
        finally:
            lease.close()

    def _apply(self, conn, inventory, started):
        rows = [dict(row) for row in conn.execute('SELECT * FROM lora ORDER BY id')]
        identities = [str(row['stable_id']).casefold() for row in rows if row.get('stable_id')]
        if any(count > 1 for count in Counter(identities).values()):
            raise CatalogueError('duplicate_identity', 'The catalogue contains duplicate stable IDs; resolve these before refreshing.', 409)
        by_path = {}
        for row in rows:
            key = path_key(row['file_path'])
            if key in by_path:
                raise CatalogueError('duplicate_path', 'The catalogue contains duplicate Windows paths; resolve these before refreshing.', 409)
            by_path[key] = row
        used, scan_id = _reserved_ids(conn), uuid4().hex
        timestamp = datetime.now(timezone.utc).isoformat()
        counts = {'discovered': len(inventory['files']), 'added': 0, 'assigned_existing_ids': 0,
                  'present': 0, 'missing': 0, 'out_of_scope': 0}
        for key, item in sorted(inventory['files'].items()):
            _deadline(started)
            metadata = _metadata(item['path'], self.root)
            row = by_path.get(key)
            if row is None:
                sid = _allocate(metadata, used)
                fields = {name: metadata[name] for name in ('base_model_name', 'base_model_code', 'category_name', 'category_code')}
                fields.update(file_path=item['path'], filename=Path(item['path']).name, stable_id=sid,
                              model_family=None, lora_type=None, rank=None, has_block_weights=0,
                              block_layout=None, clip_contributor=0, clip_tensor_count=-1,
                              last_modified=item['identity'][3] / 1e9, created_at=timestamp, updated_at=timestamp)
                columns = ','.join(fields)
                cursor = conn.execute(f"INSERT INTO lora ({columns}) VALUES ({','.join('?' for _ in fields)})", tuple(fields.values()))
                row = dict(fields, id=cursor.lastrowid)
                rows.append(row)
                by_path[key] = row
                counts['added'] += 1
            elif not row.get('stable_id'):
                row['stable_id'] = _allocate(metadata, used)
                conn.execute('UPDATE lora SET stable_id=? WHERE id=?', (row['stable_id'], row['id']))
                counts['assigned_existing_ids'] += 1
        root_key = path_key(self.root)
        for row in rows:
            _deadline(started)
            key = path_key(row['file_path'])
            item = inventory['files'].get(key)
            presence = 'current' if item else 'missing' if _scope(key, root_key, inventory['excluded']) else 'out_of_scope'
            counts['present' if presence == 'current' else presence] += 1
            metadata = _metadata(row['file_path'], self.root) if _scope(key, root_key, inventory['excluded']) else {'basis': 'out_of_scope', 'architecture_verified': False}
            conn.execute('''INSERT INTO lora_catalogue_presence
                (lora_id,path_key,presence,file_identity_json,last_seen_scan_id,last_checked_scan_id,observed_metadata_json)
                VALUES(?,?,?,?,?,?,?) ON CONFLICT(lora_id) DO UPDATE SET
                path_key=excluded.path_key,presence=excluded.presence,file_identity_json=excluded.file_identity_json,
                last_seen_scan_id=COALESCE(excluded.last_seen_scan_id,lora_catalogue_presence.last_seen_scan_id),
                last_checked_scan_id=excluded.last_checked_scan_id,observed_metadata_json=excluded.observed_metadata_json''',
                (row['id'], key, presence, json.dumps(item['identity']) if item else None,
                 scan_id if item else None, scan_id, json.dumps(metadata)))
        # Verify again after all SQL work and immediately before committing. Any
        # path/stat mismatch rolls back new rows, IDs and presence changes together.
        verify_inventory(inventory, started)
        result = {'scan_id': scan_id, 'status': 'complete', 'completed_at': timestamp, 'root': str(self.root),
                  'extension': EXTENSION, 'counts': counts, 'metadata_basis': 'folder_and_file_stat_only',
                  'architecture_verified': False, 'tensor_payload_read': False,
                  'inventory_sha256': hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
                  'elapsed_seconds': round(time.monotonic() - started, 3)}
        conn.execute('INSERT INTO lora_catalogue_scans VALUES(?,?,?,?)', (scan_id, timestamp, str(self.root), json.dumps(result)))
        return result

    def search(self, *, presence='current', base=None, category=None, search=None, limit=50, offset=0):
        if presence not in ('current', 'missing', 'all'):
            raise CatalogueError('invalid_presence', 'Choose current, missing or all library entries.')
        if type(limit) is not int or not 1 <= limit <= 5000 or type(offset) is not int or offset < 0:
            raise CatalogueError('invalid_page', 'The library page range is invalid.')
        where, parameters = [], []
        if presence != 'all':
            where.append('p.presence=?')
            parameters.append(presence)
        for column, value in (('base_model_code', base), ('category_code', category)):
            if value and value.upper() != 'ALL':
                where.append(f'l.{column}=?')
                parameters.append(value.upper())
        if search and search.strip():
            where.append('LOWER(l.filename) LIKE ?')
            parameters.append('%' + search.strip().lower() + '%')
        joined = ' FROM lora l LEFT JOIN lora_catalogue_presence p ON p.lora_id=l.id'
        clause = ' WHERE ' + ' AND '.join(where) if where else ''
        with closing(self.connection()) as conn:
            total = conn.execute('SELECT COUNT(*)' + joined + clause, parameters).fetchone()[0]
            rows = conn.execute('SELECT l.*, p.presence, p.last_seen_scan_id, p.last_checked_scan_id, p.observed_metadata_json'
                                + joined + clause + ' ORDER BY l.filename COLLATE NOCASE,l.id LIMIT ? OFFSET ?', parameters + [limit, offset]).fetchall()
            scanned = conn.execute('SELECT COUNT(*) FROM lora_catalogue_scans').fetchone()[0] > 0
        results = []
        for row in rows:
            item = dict(row)
            metadata = json.loads(item.pop('observed_metadata_json') or '{}')
            item.update(presence=item['presence'] or 'unchecked', metadata_provenance=metadata,
                        role=metadata.get('role_hint', ROLE_HINTS.get(item.get('category_code'), 'other')),
                        role_source='folder_hint', architecture_verified=False, legacy_analysis_unverified=True)
            item['clip_contributor'] = bool(item.get('clip_contributor'))
            results.append(item)
        return {'results': results, 'count': len(results), 'total': total, 'limit': limit, 'offset': offset,
                'presence': presence, 'catalogue_status': 'refreshed' if scanned else 'not_refreshed', 'extension': EXTENSION}
