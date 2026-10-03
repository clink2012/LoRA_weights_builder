"""Bounded current-catalogue loader preflight. This is not visual compatibility."""
from __future__ import annotations

from collections import Counter, OrderedDict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import struct
import threading
import time

from catalogue_refresh import ROLE_HINTS
from flux_header_coverage import CONTRACT, CONTRACT_ID, CoverageError, MAX_HEADER_BYTES, verify_local_sources

MAX_CANDIDATES = 5000
MAX_TOTAL_HEADER_BYTES = 256 * 1024 * 1024
MAX_SECONDS = 25
MAX_CACHE_ENTRIES = 2048
_GLOBAL_ERRORS = {'runtime_source_unavailable', 'runtime_source_changed'}
_UNKNOWN_ERRORS = {'file_missing', 'invalid_header', 'file_changed', 'unsupported_tensor', 'empty_header', 'unreadable_file'}


class PreflightError(ValueError):
    def __init__(self, code, message, status=409, **context):
        self.code, self.status, self.context = code, status, context
        super().__init__(message)


def _stat_identity(path):
    try:
        value = path.lstat()
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_mode,
                getattr(value, 'st_file_attributes', 0))
    except FileNotFoundError:
        return None


def _outcome(status, code, reason):
    return {'status': status, 'reason_code': code, 'reason': reason}


class CompatibilityService:
    def __init__(self, database, root, evaluate, *, source_verifier=verify_local_sources):
        self.database, self.root = Path(database), Path(root)
        self.evaluate, self.source_verifier = evaluate, source_verifier
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.epoch = None

    def _connection(self):
        conn = sqlite3.connect(self.database.resolve().as_uri() + '?mode=ro', uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _scan(conn):
        row = conn.execute('SELECT scan_id FROM lora_catalogue_scans ORDER BY completed_at DESC,rowid DESC LIMIT 1').fetchone()
        return row[0] if row else None

    def _check_budget(self, started):
        if time.monotonic() - started > MAX_SECONDS:
            raise PreflightError('preflight_budget', 'Compatibility checking exceeded its time budget. Narrow the library filters and retry; no partial list was returned.')

    def query(self, *, reference_stable_id, target_contract_id, view='eligible', base=None, category=None,
              search=None, limit=50, offset=0):
        if target_contract_id != CONTRACT_ID:
            raise PreflightError('unsupported_target', 'This preflight supports only the pinned FLUX.1 target.', 422)
        if view not in ('eligible', 'excluded', 'all') or type(limit) is not int or not 1 <= limit <= 500 or type(offset) is not int or offset < 0:
            raise PreflightError('invalid_query', 'The compatibility view or page is invalid.', 422)
        if not self.lock.acquire(blocking=False):
            raise PreflightError('preflight_busy', 'A library compatibility check is already running. Retry shortly.')
        try:
            return self._query(reference_stable_id, target_contract_id, view, base, category, search, limit, offset)
        except CoverageError as exc:
            raise PreflightError(exc.code, str(exc), 503) from exc
        except (sqlite3.Error, OSError) as exc:
            raise PreflightError('preflight_unavailable', 'The complete current catalogue or local files could not be checked. No partial list was returned.', 503) from exc
        finally:
            self.lock.release()

    def _query(self, reference, target, view, base, category, search, limit, offset):
        started = time.monotonic()
        sources = self.source_verifier()
        pins = hashlib.sha256(json.dumps({'contract': CONTRACT, 'sources': sources}, sort_keys=True).encode()).hexdigest()
        with closing(self._connection()) as conn:
            conn.execute('BEGIN')
            scan = self._scan(conn)
            if not scan:
                raise PreflightError('catalogue_not_refreshed', 'Refresh the library before checking compatible files.')
            joined = ' FROM lora l JOIN lora_catalogue_presence p ON p.lora_id=l.id '
            ref_rows = conn.execute('SELECT l.*,p.presence,p.observed_metadata_json' + joined + 'WHERE l.stable_id=?', (reference,)).fetchall()
            if len(ref_rows) != 1 or ref_rows[0]['presence'] != 'current':
                raise PreflightError('reference_not_current', 'The first selected LoRA is not a unique current library entry. Refresh the library or choose another first LoRA.')
            where, args = ["p.presence='current'"], []
            for column, value in (('base_model_code', base), ('category_code', category)):
                if value and value.upper() != 'ALL':
                    where.append('l.' + column + '=?')
                    args.append(value.upper())
            if search and search.strip():
                where.append('LOWER(l.filename) LIKE ?')
                args.append('%' + search.strip().lower() + '%')
            rows = conn.execute('SELECT l.*,p.presence,p.observed_metadata_json' + joined + 'WHERE ' + ' AND '.join(where)
                                + ' ORDER BY l.filename COLLATE NOCASE,l.id LIMIT ?', args + [MAX_CANDIDATES + 1]).fetchall()
        if len(rows) > MAX_CANDIDATES:
            raise PreflightError('preflight_budget', 'Too many candidates for one bounded compatibility check. Narrow the library filters; no partial list was returned.')
        ids = [row['stable_id'] for row in rows]
        if any(not sid for sid in ids) or len(set(ids)) != len(ids):
            raise PreflightError('ambiguous_catalogue', 'The current catalogue has missing or duplicate stable IDs. Refresh or repair it before checking compatibility.')
        epoch = (scan, pins)
        if epoch != self.epoch:
            self.cache.clear()
            self.epoch = epoch
        root = Path(os.path.abspath(self.root))
        directories = {}

        def directory_current(directory, *, required=False):
            info = directory.lstat()
            identity = (info.st_dev, info.st_ino, info.st_mode, getattr(info, 'st_file_attributes', 0))
            if str(directory) in directories and directories[str(directory)] != identity:
                raise PreflightError('files_changed', 'A library directory changed during compatibility checking. Retry; no partial list was returned.')
            ordinary = stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and not identity[3] & 0x400
            if not ordinary and required:
                raise PreflightError('preflight_unavailable', 'The configured LoRA root must be an ordinary local directory without links or junctions.', 503)
            if ordinary:
                directories[str(directory)] = identity
            return ordinary

        # Check lexical ancestors before any resolution can follow a replaced root.
        for directory in reversed((root, *root.parents)):
            directory_current(directory, required=True)
        reads, hits, states = 0, 0, {}

        def inspect(row):
            nonlocal reads, hits
            self._check_budget(started)
            path = Path(os.path.abspath(row['file_path']))
            # Paths come only from the catalogue; stay in the configured model root.
            try:
                path.relative_to(root)
                for directory in reversed(path.parents):
                    if not directory_current(directory):
                        return _outcome('unknown', 'unsupported_file_path', 'The catalogue path now uses a link, junction, or non-directory parent. Refresh the library.')
                resolved_path = path
            except PreflightError:
                raise
            except ValueError:
                return _outcome('excluded', 'outside_library', 'The file path is outside the current local LoRA library.')
            except OSError:
                self.cache.clear()
                return _outcome('unknown', 'unreadable_file', 'This file cannot currently be read. Check its local access and retry.')
            try:
                state = _stat_identity(path)
            except OSError:
                self.cache.clear()
                return _outcome('unknown', 'unreadable_file', 'This file cannot currently be read. Check its local access and retry.')
            if str(path) in states and states[str(path)] != state:
                raise PreflightError('files_changed', 'A candidate changed during compatibility checking. Retry; no partial list was returned.')
            states[str(path)] = state
            if state is None:
                return _outcome('unknown', 'file_missing', 'The file is no longer available. Refresh the library.')
            if not stat.S_ISREG(state[4]) or stat.S_ISLNK(state[4]) or state[5] & 0x400:
                return _outcome('unknown', 'unsupported_file_path', 'The catalogue path is not an ordinary local file.')
            if row['base_model_code'] != 'FLX':
                return _outcome('excluded', 'unsupported_family', 'This target currently supports FLUX.1 catalogue entries only; other families need their own loader contract.')
            cache_key = (epoch, str(resolved_path), state, row['base_model_code'])
            if reads + 8 > MAX_TOTAL_HEADER_BYTES:
                raise PreflightError('preflight_budget', 'Compatibility checking exceeded its aggregate header-read budget. Narrow the library filters; no partial list was returned.')
            try:
                # Even cached outcomes require the file to remain readable now.
                with path.open('rb') as stream:
                    word = stream.read(8)
            except OSError:
                self.cache.pop(cache_key, None)
                states.pop(str(path), None)
                return _outcome('unknown', 'unreadable_file', 'This file cannot currently be read. Check its local access and retry.')
            reads += len(word)
            if cache_key in self.cache:
                hits += 1
                value = self.cache.pop(cache_key)
                self.cache[cache_key] = value
                return dict(value)
            length = struct.unpack('<Q', word)[0] if len(word) == 8 else 0
            if len(word) != 8 or length < 2 or length > MAX_HEADER_BYTES or length > state[2] - 8:
                value = _outcome('unknown', 'invalid_header', 'The safetensors header length is invalid or exceeds the bounded reader.')
            else:
                # Reserve the exact header read before invoking the existing fresh resolver.
                if reads + 8 + length > MAX_TOTAL_HEADER_BYTES:
                    raise PreflightError('preflight_budget', 'Compatibility checking exceeded its aggregate header-read budget. Narrow the library filters; no partial list was returned.')
                reads += 8 + length
                try:
                    node, _ = self.evaluate(dict(row))
                    export = node['loader_export']
                    value = (_outcome('eligible', 'structural_loader_ready', 'Passes the current FLUX.1 header and loader checks. This does not predict visual compatibility.')
                             if export.get('status') == 'ready' else _outcome('excluded', export.get('reason_code', 'loader_blocked'), export.get('reason', 'The loader cannot represent this adapter.')))
                except CoverageError as exc:
                    if exc.code in _GLOBAL_ERRORS:
                        raise
                    if exc.code == 'invalid_header' and isinstance(exc.__cause__, OSError):
                        states.pop(str(path), None)
                        return _outcome('unknown', 'unreadable_file', 'This file cannot currently be read. Check its local access and retry.')
                    if exc.code == 'file_changed':
                        raise PreflightError('files_changed', 'A candidate changed during compatibility checking. Retry; no partial list was returned.') from exc
                    value = _outcome('unknown' if exc.code in _UNKNOWN_ERRORS else 'excluded', exc.code, str(exc))
                except OSError:
                    states.pop(str(path), None)
                    return _outcome('unknown', 'unreadable_file', 'This file cannot currently be read. Check its local access and retry.')
            try:
                final_state = _stat_identity(path)
            except OSError:
                states.pop(str(path), None)
                return _outcome('unknown', 'unreadable_file', 'This file cannot currently be read. Check its local access and retry.')
            if final_state != state:
                raise PreflightError('files_changed', 'A candidate changed during compatibility checking. Retry; no partial list was returned.')
            self.cache[cache_key] = dict(value)
            while len(self.cache) > MAX_CACHE_ENTRIES:
                self.cache.popitem(last=False)
            return value

        ref_outcome = inspect(ref_rows[0])
        reference_result = {'stable_id': reference, **ref_outcome}
        if ref_outcome['status'] != 'eligible':
            raise PreflightError('reference_not_supported', 'The first selected LoRA does not pass the current target checks. Choose another first LoRA or inspect the reason.', reference=reference_result)
        outcomes = []
        for row in rows:
            result = inspect(row)
            if row['stable_id'] == reference and result['status'] != 'eligible':
                raise PreflightError('reference_not_supported', 'The first selected LoRA stopped passing the current target checks. Retry or choose another first LoRA.', reference={'stable_id': reference, **result})
            item = dict(row)
            metadata = json.loads(item.pop('observed_metadata_json') or '{}')
            item.update(compatibility=result, metadata_provenance=metadata, role=metadata.get('role_hint', ROLE_HINTS.get(item.get('category_code'), 'other')),
                        role_source='folder_hint', architecture_verified=False, legacy_analysis_unverified=True)
            item['clip_contributor'] = bool(item.get('clip_contributor'))
            outcomes.append(item)
        # The reference may not match the current search filters. Recheck its
        # current read access regardless, before returning any candidate list.
        final_reference = inspect(ref_rows[0])
        if final_reference['status'] != 'eligible':
            raise PreflightError('reference_not_supported', 'The first selected LoRA stopped passing the current target checks. Retry or choose another first LoRA.', reference={'stable_id': reference, **final_reference})
        # Cached entries remain conditional on their current stat and runtime pins.
        for path, state in states.items():
            self._check_budget(started)
            try:
                current_state = _stat_identity(Path(path))
            except OSError:
                self.cache.clear()
                failure = _outcome('unknown', 'unreadable_file', 'This file cannot currently be read. Check its local access and retry.')
                if path == ref_rows[0]['file_path']:
                    raise PreflightError('reference_not_supported', 'The first selected LoRA became unreadable. Retry or choose another first LoRA.', reference={'stable_id': reference, **failure})
                for item in outcomes:
                    if item['file_path'] == path:
                        item['compatibility'] = dict(failure)
                continue
            if current_state != state:
                raise PreflightError('files_changed', 'The library changed during compatibility checking. Retry; no partial list was returned.')
        for directory in list(directories):
            self._check_budget(started)
            directory_current(Path(directory), required=True)
        if self.source_verifier() != sources:
            raise PreflightError('runtime_source_changed', 'The loader sources changed during compatibility checking.', 503)
        with closing(self._connection()) as conn:
            if self._scan(conn) != scan:
                raise PreflightError('catalogue_changed', 'The library was refreshed during compatibility checking. Retry to use the new inventory.')
        self._check_budget(started)
        counts = Counter(row['compatibility']['status'] for row in outcomes)
        filtered = [row for row in outcomes if view == 'all' or (row['compatibility']['status'] == 'eligible') == (view == 'eligible')]
        page = filtered[offset:offset + limit]
        reasons = Counter(row['compatibility']['reason_code'] for row in outcomes if row['compatibility']['status'] != 'eligible')
        return {'results': page, 'count': len(page), 'total': len(filtered), 'limit': limit, 'offset': offset,
                'counts': {name: counts[name] for name in ('eligible', 'excluded', 'unknown')}, 'reason_counts': dict(reasons),
                'view': view, 'presence': 'current', 'catalogue_status': 'refreshed', 'reference': reference_result,
                'target_contract_id': target, 'scope': 'structural_loader_checks_only',
                'freshness': {'checked_at': datetime.now(timezone.utc).isoformat(), 'scan_id': scan,
                              'basis': 'current_file_stat_and_pinned_contract', 'contract_fingerprint': pins,
                              'cache_used': hits > 0, 'cache_hits': hits, 'header_bytes_read': reads,
                              'tensor_payload_read': False},
                'limitations': ['Conditional on the pinned standard FLUX.1 target; actual checkpoint and tensor contents are unverified.',
                                'Passing these checks does not establish visual compatibility. Final export always requires fresh preparation.']}
