"""One bounded CPU analysis worker; immutable profile selections are the authority.

The normal API process never imports Torch. Receipts live under the project local
data directory and contain measurements, not new profiles or balancing advice.
"""
from __future__ import annotations

from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid

from composition_versions import preparation_digest
from analysis_process import launch, stop


class JobError(ValueError):
    def __init__(self, code, message, status=422):
        self.code, self.status = code, status
        super().__init__(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, allow_nan=False, separators=(',', ':'))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    with path.open('rb') as stream:
        raw = stream.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise JobError('receipt_too_large', 'The analysis receipt exceeds its size limit.')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise JobError('invalid_receipt', 'The analysis receipt must be an object.')
    return value


class WorkerLease:
    """OS releases this lock on process exit; multiple API instances cannot overlap."""
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open('a+b')
        self.stream.seek(0, 2)
        if not self.stream.tell():
            self.stream.write(b'0')
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise JobError('analysis_busy', 'Another local analysis worker is already running.', 409) from exc

    def close(self):
        self.stream.close()


class ProcessRunner:
    """Fixed executable/worker, no shell, bounded shutdown even for a stuck tensor call."""
    def __init__(self, executable, worker, timeout=210, poll_seconds=.1):
        self.executable, self.worker = Path(executable), Path(worker)
        self.timeout, self.poll_seconds = timeout, poll_seconds

    def available(self):
        return self.executable.is_file() and self.worker.is_file()

    def fingerprint(self):
        if not self.available():
            raise JobError('analysis_runtime_unavailable', 'The optional project CPU analysis environment is unavailable.', 503)
        root = self.worker.parent.parent
        paths = [self.worker, root / 'tools/analyse_effective_lora.py', root / 'Database/backend/effective_lora_metrics.py',
                 root / 'Database/backend/flux_header_coverage.py',
                 root / 'Database/backend/requirements-analysis.txt']
        stat = self.executable.stat()
        return {'python': str(self.executable), 'python_size': stat.st_size,
                'python_mtime_ns': stat.st_mtime_ns,
                'code_sha256': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}

    def __call__(self, manifest, output, cancel):
        process = launch([str(self.executable), '-B', str(self.worker),
                          '--manifest', str(manifest), '--output', str(output)])
        started = time.monotonic()
        try:
            while process.poll() is None:
                if cancel.is_set():
                    raise JobError('cancelled', 'Analysis cancelled; no partial measurements are available.')
                if time.monotonic() - started >= self.timeout:
                    raise JobError('analysis_timeout', 'The analysis worker exceeded its hard time limit.')
                cancel.wait(self.poll_seconds)
            if not output.is_file():
                raise JobError('worker_failed', 'The analysis worker exited without a complete receipt. Check the optional CPU environment.')
            receipt = read_json(output)
            if process.returncode != 0 or receipt.get('status') != 'complete':
                raise JobError(receipt.get('reason_code', 'worker_failed'), receipt.get('reason', 'The analysis worker failed.'))
            return receipt
        finally:
            try:
                if process.poll() is None:
                    stop(process)
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        stop(process)
                        process.wait(timeout=2)
            finally:
                # Closing the Windows job is the final tree-kill guarantee,
                # including when poll/terminate/wait themselves fail. On POSIX,
                # descendants may remain after their group leader has exited.
                if hasattr(process, 'close'):
                    process.close()
                elif os.name != 'nt':
                    try:
                        stop(process)
                    except ProcessLookupError:
                        pass


class AnalysisJobService:
    def __init__(self, connection_factory, preparation_resolver, *, directory=None, runner=None):
        root = Path(__file__).resolve().parents[2]
        self.directory = Path(directory) if directory else root / '.local/analysis-jobs'
        self.runner = runner or ProcessRunner(root / '.venv-analysis/Scripts/python.exe', root / 'tools/analysis_job_worker.py')
        self.connection_factory, self.preparation_resolver = connection_factory, preparation_resolver
        self.lock = threading.RLock()
        self.jobs, self.events = {}, {}
        self.active = None
        self.thread = None
        self.lease = None
        self.closed = False

    def _prepare(self, request):
        with closing(self.connection_factory()) as conn:
            prepared = self.preparation_resolver(conn, request['entries'], request['target_contract_id'])
            paths = []
            for entry in request['entries']:
                row = conn.execute('SELECT file_path FROM lora WHERE stable_id=?', (entry['stable_id'],)).fetchone()
                if row is None:
                    raise JobError('selection_changed', 'A selected LoRA is no longer in the catalogue.', 409)
                paths.append(str(Path(row[0]).resolve()))
        nodes = prepared.get('node_payloads', [])
        if (not prepared.get('compatible') or prepared.get('excluded_loras')
                or [(n.get('stable_id'), n.get('profile_version_id')) for n in nodes]
                != [(e['stable_id'], e['profile_version_id']) for e in request['entries']]
                or any(n.get('loader_export', {}).get('status') != 'ready' for n in nodes)
                or prepared.get('target_contract_id') != request['target_contract_id']
                or preparation_digest(prepared) != request['expected_preparation_digest']):
            raise JobError('selection_changed', 'Saved profiles, source files or target changed. Prepare the selection again.', 409)
        return prepared, paths

    def start(self, request):
        required = {'entries', 'target_contract_id', 'expected_preparation_digest'}
        if not isinstance(request, dict) or set(request) != required:
            raise JobError('invalid_request', 'Only saved profile entries, target and preparation digest are accepted.')
        entries = request['entries']
        if (not isinstance(entries, list) or not 1 <= len(entries) <= 8
                or any(not isinstance(e, dict) or set(e) != {'stable_id', 'profile_version_id'}
                       or any(not isinstance(v, str) or not v.strip() or len(v) > 256 for v in e.values()) for e in entries)
                or len({e['stable_id'] for e in entries}) != len(entries)
                or not isinstance(request['target_contract_id'], str)
                or not isinstance(request['expected_preparation_digest'], str)
                or len(request['expected_preparation_digest']) != 64):
            raise JobError('invalid_request', 'Choose one to eight unique LoRAs with saved profile versions and a current preparation digest.')
        request = deepcopy(request)
        with self.lock:
            if self.closed:
                raise JobError('service_stopping', 'Analysis is shutting down.', 503)
            if self.active:
                raise JobError('analysis_busy', 'One analysis is already running. Wait or cancel it first.', 409)
            if not self.runner.available():
                raise JobError('analysis_runtime_unavailable', 'The optional project CPU analysis environment is unavailable.', 503)
            self.lease = WorkerLease(self.directory / 'worker.lock')
            try:
                prepared, paths = self._prepare(request)
                runtime = self.runner.fingerprint()
                job_id = uuid.uuid4().hex
                job = {'job_id': job_id, 'status': 'queued', 'entries': request['entries'],
                       'target_contract_id': request['target_contract_id'],
                       'preparation_digest': request['expected_preparation_digest'], 'created_at': now(),
                       'finished_at': None, 'reason_code': None, 'reason': None, 'metrics': None,
                       'worker_binding': runtime}
                folder = self.directory / job_id
                folder.mkdir(parents=True, exist_ok=False)
                write_json(folder / 'job.json', job)
                self.jobs[job_id], self.events[job_id] = job, threading.Event()
                self.active = job_id
                self.thread = threading.Thread(target=self._run, args=(job_id, request, prepared, paths, runtime), daemon=True)
                self.thread.start()
                return deepcopy(job)
            except BaseException:
                self.active = None
                self.lease.close()
                self.lease = None
                raise

    def _run(self, job_id, request, prepared, paths, runtime):
        folder = self.directory / job_id
        cancel = self.events[job_id]
        completed_receipt = None
        try:
            if cancel.is_set():
                raise JobError('cancelled', 'Analysis cancelled.')
            fresh, fresh_paths = self._prepare(request)
            if fresh_paths != paths or self.runner.fingerprint() != runtime:
                raise JobError('selection_changed', 'The analysis source or worker changed.', 409)
            with self.lock:
                self.jobs[job_id]['status'] = 'running'
                write_json(folder / 'job.json', self.jobs[job_id])
            write_json(folder / 'input.json', {'paths': paths})
            receipt = self.runner(folder / 'input.json', folder / 'worker.json', cancel)
            if cancel.is_set():
                raise JobError('cancelled', 'Analysis cancelled.')
            fresh, fresh_paths = self._prepare(request)
            if fresh_paths != paths or self.runner.fingerprint() != runtime:
                raise JobError('selection_changed', 'The analysis source or worker changed.', 409)
            self._verify_receipt(receipt, prepared, paths)
            with self.lock:
                if cancel.is_set():
                    raise JobError('cancelled', 'Analysis cancelled.')
                completed_receipt = receipt
        except JobError as exc:
            state = 'cancelled' if exc.code == 'cancelled' else 'stale' if exc.status == 409 else 'failed'
            with self.lock:
                self.jobs[job_id].update(status=state, reason_code=exc.code, reason=str(exc), metrics=None)
        except Exception:
            with self.lock:
                self.jobs[job_id].update(status='failed', reason_code='analysis_failed',
                                         reason='Analysis could not complete. No partial measurements were accepted.', metrics=None)
        finally:
            with self.lock:
                if completed_receipt is not None:
                    if cancel.is_set():
                        self.jobs[job_id].update(status='cancelled', reason_code='cancelled',
                                                 reason='Analysis cancelled.', metrics=None)
                    else:
                        self.jobs[job_id].update(status='complete', metrics=completed_receipt)
                self.jobs[job_id]['finished_at'] = now()
                try:
                    write_json(folder / 'job.json', self.jobs[job_id])
                except (OSError, ValueError, TypeError):
                    self.jobs[job_id].update(status='failed', reason_code='receipt_persistence_failed',
                                             reason='The analysis receipt could not be saved. No current measurements were accepted.', metrics=None)
                    try:
                        write_json(folder / 'job.json', self.jobs[job_id])
                    except (OSError, ValueError, TypeError):
                        pass
                finally:
                    self.active = None
                    if self.lease:
                        self.lease.close()
                        self.lease = None

    @staticmethod
    def _verify_receipt(receipt, prepared, paths):
        nodes = prepared['node_payloads']
        if receipt.get('status') != 'complete' or len(receipt.get('sources', [])) != len(nodes):
            raise JobError('invalid_receipt', 'The worker did not provide a complete selection receipt.')
        for index, (source, node, path) in enumerate(zip(receipt['sources'], nodes, paths)):
            binding = node['profile_default_binding']
            identity = source['identity']
            if (source['source_index'] != index or str(Path(source['path']).resolve()) != path
                    or identity['header_sha256'] != binding['source_identity']['header_sha256']
                    or identity['file_size'] != binding['source_identity']['size_bytes']
                    or identity['file_mtime_ns'] != binding['source_identity']['mtime_ns']
                    or receipt['target_contract_id'] != binding['target_contract']['id']
                    or receipt['target_contract_sha256'] != binding['target_contract']['sha256']
                    or receipt['source_sha256'] != binding['loader_adapter']['source_sha256']
                    or receipt['slot_labels'] != [slot['label'] for slot in binding['slots']]):
                raise JobError('receipt_binding_changed', 'Worker measurements do not match the selected source and target.', 409)

    def get(self, job_id):
        if not isinstance(job_id, str) or len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
            raise JobError('job_not_found', 'Analysis job not found.', 404)
        with self.lock:
            if job_id in self.jobs:
                job = deepcopy(self.jobs[job_id])
            else:
                path = self.directory / job_id / 'job.json'
                if not path.is_file():
                    raise JobError('job_not_found', 'Analysis job not found.', 404)
                job = read_json(path)
                if job['status'] in ('queued', 'running'):
                    job.update(status='failed', reason_code='analysis_interrupted',
                               reason='The previous app process ended before analysis completed.', metrics=None)
        if job['status'] == 'complete':
            try:
                self._prepare({'entries': job['entries'], 'target_contract_id': job['target_contract_id'],
                               'expected_preparation_digest': job['preparation_digest']})
                if self.runner.fingerprint() != job['worker_binding']:
                    raise JobError('selection_changed', 'The analysis worker changed.', 409)
            except Exception:
                # Preserve the original disk receipt as historical evidence, but
                # do not expose it as current measurements for a changed source.
                job.update(status='stale', reason_code='selection_changed',
                           reason='The saved analysis no longer matches the current files, profiles or worker.', metrics=None)
        return job

    def cancel(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job['status'] in ('queued', 'running'):
                self.events[job_id].set()
            return job

    def revalidate(self, job_id):
        """Trusted boundary for future proposals; never consume historical metrics directly."""
        job = self.get(job_id)
        if job['status'] != 'complete':
            raise JobError('analysis_not_current', 'Complete current measurements are required.', 409)
        prepared, paths = self._prepare({'entries': job['entries'], 'target_contract_id': job['target_contract_id'],
                                        'expected_preparation_digest': job['preparation_digest']})
        self._verify_receipt(job['metrics'], prepared, paths)
        if self.runner.fingerprint() != job['worker_binding']:
            raise JobError('analysis_not_current', 'The analysis worker changed.', 409)
        return {'job': job, 'preparation': prepared}

    def shutdown(self):
        with self.lock:
            self.closed = True
            if self.active:
                self.events[self.active].set()
            thread = self.thread
        if thread:
            thread.join(timeout=6)
