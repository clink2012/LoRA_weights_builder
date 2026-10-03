"""Worker ownership, selection freshness and HTTP authority without Torch imports."""
from copy import deepcopy
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_job_router import create_analysis_job_router
from analysis_job_service import AnalysisJobService, JobError, ProcessRunner, read_json
import analysis_job_service
from composition_versions import preparation_digest


class Runner:
    def __init__(self, receipt):
        self.receipt = receipt
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.present, self.version = True, 'v1'
        self.error = None

    def available(self):
        return self.present

    def fingerprint(self):
        return {'version': self.version}

    def __call__(self, manifest, output, cancel):
        self.started.set()
        while not self.release.wait(.01):
            if cancel.is_set():
                raise JobError('cancelled', 'Stopped')
        if self.error:
            raise self.error
        return deepcopy(self.receipt)


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / 'lora.safetensors'
    path.write_bytes(b'fixture')
    database = tmp_path / 'catalogue.db'
    with sqlite3.connect(database) as conn:
        conn.execute('CREATE TABLE lora(stable_id TEXT, file_path TEXT)')
        conn.execute('INSERT INTO lora VALUES (?,?)', ('one', str(path)))
    identity = {'basis': 'header_stat', 'header_sha256': 'a' * 64, 'size_bytes': 7, 'mtime_ns': 42}
    binding = {'source_identity': identity, 'target_contract': {'id': 'target', 'sha256': 'b' * 64},
               'loader_adapter': {'source_sha256': {'loader': 'c' * 64}}, 'slots': [{'label': 'BASE'}]}
    prepared = {'compatible': True, 'excluded_loras': [], 'target_contract_id': 'target',
                'node_payloads': [{'stable_id': 'one', 'profile_version_id': 'version',
                                   'loader_export': {'status': 'ready'}, 'profile_default_binding': binding}]}
    receipt = {'status': 'complete', 'target_contract_id': 'target', 'target_contract_sha256': 'b' * 64,
               'source_sha256': {'loader': 'c' * 64}, 'slot_labels': ['BASE'],
               'sources': [{'source_index': 0, 'path': str(path),
                            'identity': {'header_sha256': 'a' * 64, 'file_size': 7, 'file_mtime_ns': 42}}]}
    request = {'entries': [{'stable_id': 'one', 'profile_version_id': 'version'}],
               'target_contract_id': 'target', 'expected_preparation_digest': preparation_digest(prepared)}
    runner = Runner(receipt)
    factory = lambda: sqlite3.connect(database)
    resolver = lambda conn, entries, target: deepcopy(prepared)
    service = AnalysisJobService(factory, resolver, directory=tmp_path / 'jobs', runner=runner)
    app = FastAPI()
    app.include_router(create_analysis_job_router(service))
    yield service, runner, prepared, request, TestClient(app)
    service.shutdown()


def finished(service, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if service.jobs[job_id]['finished_at']:
            return service.get(job_id)
        time.sleep(.01)
    pytest.fail('The bounded worker did not finish')


def test_complete_exact_request_receipt_persistence_and_revalidation(setup):
    service, runner, _, request, client = setup
    response = client.post('/api/analysis-jobs', json=request)
    assert response.status_code == 202
    job = finished(service, response.json()['job_id'])
    assert job['status'] == 'complete' and job['metrics'] == runner.receipt
    assert service.revalidate(job['job_id'])['job']['preparation_digest'] == request['expected_preparation_digest']
    assert read_json(service.directory / job['job_id'] / 'job.json')['metrics'] == runner.receipt
    reopened = AnalysisJobService(service.connection_factory, service.preparation_resolver,
                                  directory=service.directory, runner=runner)
    assert reopened.get(job['job_id']) == job


@pytest.mark.parametrize('change', [lambda r: r.update(paths=['C:/arbitrary']),
                                   lambda r: r['entries'][0].update(file_path='x'),
                                   lambda r: r['entries'][0].update(profile_version_id=''),
                                   lambda r: r['entries'].append(dict(r['entries'][0])),
                                   lambda r: r.update(entries=[]),
                                   lambda r: r.update(expected_preparation_digest='x')])
def test_request_rejects_paths_unsaved_duplicate_and_invalid_input(setup, change):
    service, _, _, request, client = setup
    change(request)
    assert client.post('/api/analysis-jobs', json=request).status_code == 422
    assert not service.jobs


def test_unavailable_runtime_is_actionable_and_no_job_started(setup):
    service, runner, _, request, client = setup
    runner.present = False
    result = client.post('/api/analysis-jobs', json=request)
    assert result.status_code == 503
    assert result.json()['detail']['reason_code'] == 'analysis_runtime_unavailable'
    assert not service.jobs


def test_changed_digest_before_launch_rejected_and_lease_released(setup):
    service, _, prepared, request, client = setup
    prepared['new_value'] = True
    assert client.post('/api/analysis-jobs', json=request).status_code == 409
    assert service.active is None and service.lease is None


def test_single_worker_across_service_instances_and_cancel(setup):
    service, runner, _, request, client = setup
    runner.release.clear()
    job = service.start(request)
    assert runner.started.wait(2)
    assert client.post('/api/analysis-jobs', json=request).status_code == 409
    other = AnalysisJobService(service.connection_factory, service.preparation_resolver,
                               directory=service.directory, runner=runner)
    with pytest.raises(JobError, match='already running'):
        other.start(request)
    assert client.post(f"/api/analysis-jobs/{job['job_id']}/cancel", json={}).status_code == 200
    assert finished(service, job['job_id'])['status'] == 'cancelled'
    runner.release.set()
    assert finished(other, other.start(request)['job_id'])['status'] == 'complete'


@pytest.mark.parametrize('changed', ['profile', 'worker', 'receipt'])
def test_changed_binding_during_analysis_never_surfaces_partial_metrics(setup, changed):
    service, runner, prepared, request, _ = setup
    runner.release.clear()
    job = service.start(request)
    assert runner.started.wait(2)
    if changed == 'profile':
        prepared['node_payloads'][0]['profile_version_id'] = 'different'
    elif changed == 'worker':
        runner.version = 'v2'
    else:
        runner.receipt['sources'][0]['identity']['header_sha256'] = 'd' * 64
    runner.release.set()
    result = finished(service, job['job_id'])
    assert result['status'] == 'stale' and result['metrics'] is None


def test_completed_receipt_becomes_stale_without_rewriting_history(setup):
    service, _, prepared, request, _ = setup
    job = finished(service, service.start(request)['job_id'])
    original = (service.directory / job['job_id'] / 'job.json').read_bytes()
    prepared['changed'] = True
    current = service.get(job['job_id'])
    assert current['status'] == 'stale' and current['metrics'] is None
    with pytest.raises(JobError, match='current measurements'):
        service.revalidate(job['job_id'])
    assert (service.directory / job['job_id'] / 'job.json').read_bytes() == original


def test_worker_failure_and_restart_interruption_are_explicit(setup):
    service, runner, _, request, _ = setup
    runner.error = JobError('read_budget_exceeded', 'Read budget exhausted')
    result = finished(service, service.start(request)['job_id'])
    assert result['status'] == 'failed' and result['reason_code'] == 'read_budget_exceeded'
    path = service.directory / result['job_id'] / 'job.json'
    path.write_text(json.dumps(dict(result, status='running')), encoding='utf-8')
    service.jobs.clear()
    assert service.get(result['job_id'])['reason_code'] == 'analysis_interrupted'


@pytest.mark.parametrize('persistent_failure', [False, True])
def test_receipt_write_failure_never_grants_in_memory_completion(setup, monkeypatch, persistent_failure):
    service, _, _, request, _ = setup
    original_write = analysis_job_service.write_json
    failed = False
    def write(path, value):
        nonlocal failed
        if path.name == 'job.json' and (value.get('status') == 'complete' or (failed and persistent_failure)):
            failed = True
            raise OSError('Injected receipt storage failure')
        return original_write(path, value)
    monkeypatch.setattr(analysis_job_service, 'write_json', write)
    result = finished(service, service.start(request)['job_id'])
    assert result['status'] == 'failed' and result['metrics'] is None
    assert result['reason_code'] == 'receipt_persistence_failed'
    assert service.active is None and service.lease is None
    with pytest.raises(JobError, match='current measurements'):
        service.revalidate(result['job_id'])


def test_unknown_ids_and_cancel_payload_rejected(setup):
    _, _, _, _, client = setup
    assert client.get('/api/analysis-jobs/not-a-job').status_code == 404
    assert client.post('/api/analysis-jobs/' + 'a' * 32 + '/cancel', json={'path': 'x'}).status_code == 422


def test_receipt_reader_bounded_and_requires_object(tmp_path):
    path = tmp_path / 'receipt.json'
    path.write_bytes(b' ' * (16 * 1024 * 1024 + 1))
    with pytest.raises(JobError, match='size limit'):
        read_json(path)
    path.write_text('[]')
    with pytest.raises(JobError, match='object'):
        read_json(path)


@pytest.mark.parametrize('failure_at', ['poll', 'terminate', 'wait'])
def test_owned_job_is_always_closed_when_cleanup_operations_fail(tmp_path, monkeypatch, failure_at):
    class Process:
        closed = False
        polls = 0
        def poll(self):
            self.polls += 1
            if failure_at == 'poll' and self.polls > 1:
                raise OSError('Injected poll failure')
            return None
        def wait(self, timeout):
            if failure_at == 'wait':
                raise OSError('Injected wait failure')
            return 1
        def close(self):
            self.closed = True
    process = Process()
    def stop(_process):
        if failure_at == 'terminate':
            raise OSError('Injected terminate failure')
    monkeypatch.setattr(analysis_job_service, 'launch', lambda args: process)
    monkeypatch.setattr(analysis_job_service, 'stop', stop)
    event = threading.Event()
    event.set()
    with pytest.raises(OSError, match='Injected'):
        ProcessRunner('unused', 'unused')(tmp_path / 'input', tmp_path / 'output', event)
    assert process.closed


@pytest.mark.skipif(os.name != 'nt', reason='Windows job ownership test')
@pytest.mark.parametrize('mode', ['cancel', 'timeout'])
def test_windows_real_venv_redirector_and_descendant_stop(tmp_path, mode):
    """Real Windows processes; retain handles so PID reuse cannot fake the check."""
    executable = Path(__file__).resolve().parents[3] / '.venv-analysis/Scripts/python.exe'
    if not executable.is_file():
        pytest.skip('Optional project runtime is not installed')
    worker = tmp_path / 'sleep_worker.py'
    pid_file = tmp_path / 'pids.json'
    worker.write_text('import os,sys,subprocess,time,json\nfrom pathlib import Path\n'
                      'child=subprocess.Popen([sys.executable,"-B","-c","import time;time.sleep(60)"])\n'
                      f'Path({str(pid_file)!r}).write_text(json.dumps([os.getpid(),child.pid]))\n'
                      'time.sleep(60)\n', encoding='utf-8')
    runner = ProcessRunner(executable, worker, timeout=1.5, poll_seconds=.01)
    event, errors = threading.Event(), []
    def run():
        try:
            runner(tmp_path / 'unused.json', tmp_path / 'absent.json', event)
        except JobError as exc:
            errors.append(exc.code)
    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 1
    while not pid_file.is_file() and time.monotonic() < deadline:
        time.sleep(.01)
    assert pid_file.is_file(), 'Worker failed to reach the descendant fixture'
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handles = [kernel.OpenProcess(0x100000, False, pid) for pid in json.loads(pid_file.read_text())]
    assert all(handles)
    try:
        if mode == 'cancel':
            event.set()
        thread.join(5)
        assert not thread.is_alive()
        assert errors == ['cancelled' if mode == 'cancel' else 'analysis_timeout']
        assert all(kernel.WaitForSingleObject(handle, 2000) == 0 for handle in handles)
    finally:
        event.set()
        thread.join(5)
        for handle in handles:
            kernel.CloseHandle(handle)
