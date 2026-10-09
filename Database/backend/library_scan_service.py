"""Background path inventory and bounded header observations; never tensor analysis."""
from __future__ import annotations

from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import stat
import struct
import threading
import time
from uuid import uuid4

from adapter_identity import IDENTIFICATION_VERSION, identify_file
from analysis_job_service import WorkerLease, JobError
from catalogue_refresh import CatalogueError, file_identity, reparse
from flux_header_coverage import CoverageError, MAX_HEADER_BYTES
from library_location import selected_location, validate_location

MAX_AUDIT_BYTES = 512 * 1024 * 1024
MAX_AUDIT_SECONDS = 90
MAX_AUDIT_FILES = 5000


def now():
    return datetime.now(timezone.utc).isoformat()


def initialise_library_scan_schema(conn):
    if conn.in_transaction:
        raise CatalogueError('active_transaction', 'Library scan schema setup requires its own transaction.')
    with conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS lora_library_scan_jobs (
            job_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, state_json TEXT NOT NULL)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS lora_library_observations (
            scan_id TEXT NOT NULL, lora_id INTEGER NOT NULL, job_id TEXT NOT NULL,
            identity_json TEXT NOT NULL, observed_at TEXT NOT NULL, observation_json TEXT NOT NULL,
            PRIMARY KEY(scan_id,lora_id))''')


def _empty():
    return {'job_id': None, 'status': 'idle', 'phase': 'finished', 'catalogue_scan_id': None,
            'scanned': 0, 'total': 0, 'remaining': 0, 'issue_count': 0,
            'unchecked_family_count': 0, 'header_bytes_accounted': 0, 'cancel_requested': False,
            'reason_code': None, 'message': 'The library has not been scanned in this session.',
            'catalogue': None, 'architecture_verified': False, 'export_verified': False,
            'tensor_payload_read': False}


def _observations(result):
    issues = [{'code': 'family_disagreement', 'severity': 'potential_issue', 'message': message}
              for message in result['disagreements']]
    if result['incomplete_pair_count']:
        issues.append({'code': 'incomplete_pair_names', 'severity': 'potential_issue',
                       'message': 'Some recognised LoRA pair names are incomplete; inspect this adapter before use.'})
    if result['mixed_module_format_count']:
        issues.append({'code': 'mixed_module_formats', 'severity': 'potential_issue',
                       'message': 'Some modules contain more than one pair format; loader support needs checking.'})
    if result['adapter_formats'].get('unsupported'):
        issues.append({'code': 'unsupported_adapter_format', 'severity': 'not_checked',
                       'message': 'Some tensor formats are not understood by this header checker; this is not a corruption finding.'})
    return {'status': 'observed', 'identification_version': IDENTIFICATION_VERSION,
            'issues': issues, 'family_checked': bool(result['family_candidates']),
            'family_check_status': 'family_patterns_observed' if result['family_candidates'] else 'not_checked_for_family',
            'identification': result, 'architecture_verified': False, 'export_verified': False}


class LibraryScanService:
    def __init__(self, catalogue, *, inspector=identify_file, max_bytes=MAX_AUDIT_BYTES,
                 max_seconds=MAX_AUDIT_SECONDS, max_files=MAX_AUDIT_FILES):
        self.catalogue = catalogue
        self.inspector = inspector
        self.max_bytes, self.max_seconds, self.max_files = max_bytes, max_seconds, max_files
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._thread = None
        self._state = _empty()
        self._started = False
        self._closed = False

    def connection(self):
        return self.catalogue.connection()

    def startup(self):
        with self._lock:
            if self._started:
                return self.status()
            self._started = True
        try:
            return self.start()
        except CatalogueError as exc:
            if exc.code != 'scan_busy':
                raise
            # Another process may already own the shared scan lease. Keep the
            # app available and manual retry enabled; do not claim its results.
            with self._lock:
                self._state.update(status='failed', phase='finished', reason_code='scan_busy',
                                   message='Another app process is scanning this library. Refresh here after it finishes.')
            return self.status()

    def status(self):
        with self._lock:
            return deepcopy(self._state)

    def freshness(self):
        with self._lock:
            if self._thread and self._thread.is_alive() and self._state['phase'] == 'catalogue':
                raise CatalogueError('scan_busy', 'Inventory refresh is running. The folder comparison will be available afterwards.', 409)
        return self.catalogue.freshness()

    def location(self):
        with closing(self.connection()) as conn:
            selected = selected_location(conn, self.catalogue.root)
        return {'active_root': str(self.catalogue.root), 'selected_root': str(selected),
                'restart_required': selected != self.catalogue.root,
                'history_preserved': True, 'automatic_relocation': False}

    def choose_location(self, value):
        selected = validate_location(value)
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise CatalogueError('scan_busy', 'Wait for background library checks to finish before choosing another folder.', 409)
            with closing(self.connection()) as conn, conn:
                conn.execute("INSERT INTO lora_library_settings VALUES('root',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(selected),))
        return self.location()

    def _publish(self, job_id, **updates):
        with self._lock:
            if self._state['job_id'] != job_id:
                return False
            self._state.update(updates)
            with closing(self.connection()) as conn, conn:
                conn.execute('UPDATE lora_library_scan_jobs SET state_json=? WHERE job_id=?',
                             (json.dumps(self._state), job_id))
            return True

    def start(self, *, resume=False):
        with self._lock:
            if self._closed:
                raise CatalogueError('scan_shutdown', 'The library scanner is shutting down.', 503)
            if self._thread and self._thread.is_alive():
                return self.status()
            try:
                lease = WorkerLease(self.catalogue.database.with_suffix('.library-scan.lock'))
            except JobError as exc:
                raise CatalogueError('scan_busy', 'Another app process is scanning this library.', 409) from exc
            try:
                scan_id = None
                if resume:
                    with closing(self.connection()) as conn:
                        row = conn.execute('SELECT scan_id FROM lora_catalogue_scans ORDER BY rowid DESC LIMIT 1').fetchone()
                    if row is None:
                        raise CatalogueError('nothing_to_resume', 'Refresh the library before continuing header checks.', 409)
                    scan_id = row['scan_id']
                self._cancel = threading.Event()
                self._state = dict(_empty(), job_id=uuid4().hex, status='running', phase='headers' if resume else 'catalogue',
                                   catalogue_scan_id=scan_id, created_at=now(), message='Checking current library files.')
                with closing(self.connection()) as conn, conn:
                    conn.execute('INSERT INTO lora_library_scan_jobs VALUES(?,?,?)',
                                 (self._state['job_id'], self._state['created_at'], json.dumps(self._state)))
                self._thread = threading.Thread(target=self._run, args=(self._state['job_id'], lease, scan_id),
                                                name='lora-library-scan', daemon=True)
                self._thread.start()
            except BaseException:
                lease.close()
                raise
            return self.status()

    def cancel(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                self._cancel.set()
                self._state['cancel_requested'] = True
                self._state['message'] = ('Stopping after the current bounded inventory or header read. '
                                          'Unchecked files are not marked clear.')
            return self.status()

    def shutdown(self):
        with self._lock:
            self._closed = True
        self.cancel()
        if self._thread:
            self._thread.join(timeout=1)

    def _current_rows(self, scan_id):
        with closing(self.connection()) as conn:
            return [dict(row) for row in conn.execute('''SELECT l.id,l.stable_id,l.filename,l.file_path,
                p.file_identity_json,p.observed_metadata_json FROM lora l JOIN lora_catalogue_presence p ON p.lora_id=l.id
                WHERE p.presence='current' AND p.last_checked_scan_id=? ORDER BY l.id''', (scan_id,))]

    def _assert_scan(self, scan_id):
        with closing(self.connection()) as conn:
            latest = conn.execute('SELECT scan_id FROM lora_catalogue_scans ORDER BY rowid DESC LIMIT 1').fetchone()
        if latest is None or latest['scan_id'] != scan_id:
            raise CatalogueError('scan_superseded', 'A newer library inventory replaced this header scan.', 409)

    def _safe_stat(self, path):
        root = self.catalogue.root
        path = Path(path)
        path.relative_to(root)
        for item in (path, *path.parents):
            info = item.lstat()
            if reparse(info):
                raise CoverageError('file_changed', 'A file path now uses a link or junction; refresh the library.')
            if item == root:
                break
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise CoverageError('file_changed', 'The current entry is no longer a regular file.')
        return file_identity(info)

    def _check(self, row, remaining_bytes):
        path = Path(row['file_path'])
        expected = tuple(json.loads(row['file_identity_json']))
        if self._safe_stat(path) != expected:
            raise CoverageError('file_changed', 'The file changed after discovery; refresh before inspecting it.')
        # Only the eight-byte length prefix is read before reserving the header budget.
        with path.open('rb') as stream:
            if file_identity(os.fstat(stream.fileno())) != expected:
                raise CoverageError('file_changed', 'The file changed before its header was read.')
            prefix = stream.read(8)
        length = struct.unpack('<Q', prefix)[0] if len(prefix) == 8 else 0
        if length > MAX_HEADER_BYTES:
            return {'status': 'not_checked', 'identification_version': IDENTIFICATION_VERSION, 'family_checked': False,
                    'issues': [{'code': 'header_budget', 'severity': 'not_checked',
                                'message': 'This header exceeds the per-file inspection limit; it has not been checked.'}],
                    'architecture_verified': False, 'export_verified': False}, 8
        cost = 16 + length  # include both prefix reads, even if the reader rejects the file
        if cost > remaining_bytes:
            raise CatalogueError('audit_budget', 'The header byte budget was reached; continue checks for remaining files.', 409)
        hint = path.relative_to(self.catalogue.root).parts
        try:
            result = self.inspector(path, folder_hint=hint[0] if len(hint) > 1 else None)
            identity = result['file_identity']
            actual = tuple(identity[key] for key in ('file_device', 'file_inode', 'file_size', 'file_mtime_ns'))
            if actual != expected or self._safe_stat(path) != expected:
                raise CoverageError('file_changed', 'The file changed while its header was inspected; refresh and retry.')
        except (CoverageError, OSError, ValueError) as exc:
            return self._error(getattr(exc, 'code', 'header_unreadable'), 'Header could not be checked: ' + str(exc)), cost
        return _observations(result), cost

    def _save(self, job_id, scan_id, row, observation):
        # A separate metadata refresh or source replacement cannot receive old observations.
        with self._lock:
            if self._cancel.is_set() or self._state['job_id'] != job_id:
                return False
            with closing(self.connection()) as conn:
                conn.execute('BEGIN IMMEDIATE')
                current = conn.execute('SELECT * FROM lora_catalogue_presence WHERE lora_id=?', (row['id'],)).fetchone()
                latest = conn.execute('SELECT scan_id FROM lora_catalogue_scans ORDER BY rowid DESC LIMIT 1').fetchone()
                if (not current or not latest or latest['scan_id'] != scan_id or current['presence'] != 'current'
                        or current['last_checked_scan_id'] != scan_id or current['file_identity_json'] != row['file_identity_json']):
                    conn.rollback()
                    raise CatalogueError('scan_superseded', 'A newer library inventory replaced this header scan.', 409)
                # Error observations are also tied to the inventory identity. A changed
                # source is recorded as stale, never as an observation of its new contents.
                try:
                    fresh = self._safe_stat(row['file_path']) == tuple(json.loads(row['file_identity_json']))
                except (OSError, ValueError):
                    fresh = False
                observation['source_current_at_capture'] = fresh
                if not fresh:
                    observation = self._error('file_changed', 'File changed or disappeared after discovery; refresh the library.')
                    observation['source_current_at_capture'] = False
                conn.execute('''INSERT INTO lora_library_observations VALUES(?,?,?,?,?,?)
                    ON CONFLICT(scan_id,lora_id) DO UPDATE SET job_id=excluded.job_id,
                    identity_json=excluded.identity_json,observed_at=excluded.observed_at,observation_json=excluded.observation_json''',
                             (scan_id, row['id'], job_id, row['file_identity_json'], now(), json.dumps(observation)))
                conn.commit()
                return observation

    @staticmethod
    def _error(code, message):
        unsupported = code in ('unsupported_tensor', 'unsupported_file', 'header_budget')
        return {'status': 'not_checked' if unsupported else 'unavailable',
                'identification_version': IDENTIFICATION_VERSION, 'family_checked': False,
                'issues': [{'code': code, 'severity': 'not_checked' if unsupported else 'potential_issue',
                            'message': message}], 'architecture_verified': False, 'export_verified': False}

    def _run(self, job_id, lease, scan_id):
        try:
            if scan_id is None:
                receipt = self.catalogue.refresh(cancelled=self._cancel.is_set)
                scan_id = receipt['scan_id']
                self._publish(job_id, catalogue_scan_id=scan_id, catalogue=receipt, phase='headers',
                              message='Library list refreshed. Checking headers for potential issues.')
            self._assert_scan(scan_id)
            rows = self._current_rows(scan_id)
            with closing(self.connection()) as conn:
                saved = {row['lora_id']: json.loads(row['observation_json']) for row in conn.execute(
                    'SELECT lora_id,observation_json FROM lora_library_observations WHERE scan_id=?', (scan_id,))}
            # Resuming a batch rechecks cheap path/stat identity before reusing
            # completed observations. Changed sources need a fresh warning.
            for row in rows:
                if row['id'] not in saved:
                    continue
                try:
                    current = self._safe_stat(row['file_path']) == tuple(json.loads(row['file_identity_json']))
                except (OSError, ValueError):
                    current = False
                if not current:
                    saved.pop(row['id'])
            pending = [row for row in rows if row['id'] not in saved]
            scanned, issues, unchecked = len(saved), sum(bool(o['issues']) for o in saved.values()), sum(not o['family_checked'] for o in saved.values())
            self._publish(job_id, total=len(rows), scanned=scanned, remaining=len(pending), issue_count=issues,
                          unchecked_family_count=unchecked)
            started, byte_count, processed = time.monotonic(), 0, 0
            for row in pending:
                if self._cancel.is_set():
                    break
                if processed >= self.max_files or time.monotonic() - started >= self.max_seconds or byte_count + 8 > self.max_bytes:
                    raise CatalogueError('audit_budget', 'This header-check batch reached its limit; continue to check remaining files.', 409)
                try:
                    observation, cost = self._check(row, self.max_bytes - byte_count)
                except (CoverageError, OSError, ValueError) as exc:
                    if isinstance(exc, CatalogueError):
                        raise
                    code = getattr(exc, 'code', 'header_unreadable')
                    observation = self._error(code, 'Header could not be checked: ' + str(exc))
                    cost = 8  # conservative allowance when the preliminary file check fails
                saved_observation = self._save(job_id, scan_id, row, observation)
                if not saved_observation:
                    break
                byte_count += cost
                processed += 1
                scanned += 1
                issues += bool(saved_observation['issues'])
                unchecked += not saved_observation['family_checked']
                self._publish(job_id, scanned=scanned, remaining=len(rows)-scanned, issue_count=issues,
                              unchecked_family_count=unchecked, header_bytes_accounted=byte_count)
            if self._cancel.is_set():
                self._publish(job_id, status='cancelled', phase='finished', reason_code='cancelled',
                              message='Scan stopped. Remaining files have not been checked.')
            else:
                self._assert_scan(scan_id)
                self._publish(job_id, status='complete', phase='finished', completed_at=now(),
                              message='Header checks finished. Observations do not establish compatibility or image quality.')
        except CatalogueError as exc:
            self._publish(job_id, status='partial' if exc.code == 'audit_budget' else 'cancelled' if exc.code == 'scan_cancelled' else 'failed', phase='finished',
                          reason_code=exc.code, message=str(exc))
        except Exception:
            try:
                self._publish(job_id, status='failed', phase='finished', reason_code='scan_unavailable',
                              message='Library checking could not finish; unchecked files are not marked clear.')
            except sqlite3.Error:
                with self._lock:
                    self._state.update(status='failed', phase='finished', reason_code='scan_persistence_failed',
                                       message='Scan results could not be saved. No complete scan is claimed.')
        finally:
            lease.close()

    def issues(self, *, stable_id=None, limit=50, offset=0):
        if not 1 <= limit <= 500 or offset < 0:
            raise CatalogueError('invalid_page', 'Invalid scan observation page.')
        with closing(self.connection()) as conn:
            latest = conn.execute('SELECT scan_id FROM lora_catalogue_scans ORDER BY rowid DESC LIMIT 1').fetchone()
            if latest is None:
                return {'results': [], 'total': 0, 'catalogue_scan_id': None, 'scope': 'current_inventory_only'}
            where = ' AND l.stable_id=?' if stable_id else ''
            parameters = [latest['scan_id']] + ([stable_id] if stable_id else [])
            records = conn.execute('''SELECT l.stable_id,l.filename,l.file_path,o.*,p.presence,p.file_identity_json
                FROM lora_library_observations o JOIN lora l ON l.id=o.lora_id
                JOIN lora_catalogue_presence p ON p.lora_id=o.lora_id
                WHERE o.scan_id=? AND p.last_checked_scan_id=o.scan_id''' + where + ' ORDER BY l.id', parameters).fetchall()
        results = []
        for record in records:
            observation = json.loads(record['observation_json'])
            if not observation['issues']:
                continue
            try:
                fresh = (record['presence'] == 'current' and record['file_identity_json'] == record['identity_json']
                         and self._safe_stat(record['file_path']) == tuple(json.loads(record['identity_json'])))
            except (ValueError, OSError):
                fresh = False
            fresh = fresh and observation.get('source_current_at_capture', False)
            results.append({'stable_id': record['stable_id'], 'filename': record['filename'],
                            'catalogue_scan_id': record['scan_id'], 'observed_at': record['observed_at'],
                            'source_status': 'current' if fresh else 'stale',
                            'requires_revalidation': not fresh, 'observation': observation})
        return {'results': results[offset:offset+limit], 'total': len(results), 'limit': limit, 'offset': offset,
                'catalogue_scan_id': latest['scan_id'], 'scope': 'current_inventory_only',
                'architecture_verified': False, 'export_verified': False}
