"""Exact receipt lookup remains read-only and cannot become historical authority."""
from copy import deepcopy
import json

import pytest

from analysis_job_service import AnalysisJobService
from test_analysis_jobs import setup, finished


def complete(service, request):
    job = service.start(request)
    assert finished(service, job['job_id'])['status'] == 'complete'
    return job['job_id']


def test_reopen_recovers_exact_receipt_without_running_worker_or_changing_history(setup):
    service, runner, _, request, client = setup
    assert client.post('/api/analysis-jobs/resolve', json=request).json() == {'status': 'not_found', 'job': None}
    job_id = complete(service, request)
    receipt_path = service.directory / job_id / 'job.json'
    before = receipt_path.read_bytes()
    reopened = AnalysisJobService(service.connection_factory, service.preparation_resolver, directory=service.directory, runner=runner)
    runner.started.clear()
    result = reopened.resolve(request)
    assert result['status'] == 'reused' and result['job']['job_id'] == job_id
    assert not runner.started.is_set() and reopened.active is None
    assert receipt_path.read_bytes() == before


def test_new_measurement_updates_pointer_but_keeps_both_receipts(setup):
    service, _, _, request, _ = setup
    first = complete(service, request)
    second = complete(service, request)
    assert first != second and service.resolve(request)['job']['job_id'] == second
    assert (service.directory / first / 'job.json').is_file()


def test_worker_change_blocks_reuse_without_deleting_original_receipt(setup):
    service, runner, _, request, _ = setup
    job_id = complete(service, request)
    original = (service.directory / job_id / 'job.json').read_bytes()
    runner.version = 'changed'
    assert service.resolve(request)['status'] == 'not_found'
    assert (service.directory / job_id / 'job.json').read_bytes() == original


def test_source_or_profile_change_fails_fresh_preparation_even_with_lookup(setup):
    service, _, prepared, request, client = setup
    complete(service, request)
    prepared['node_payloads'][0]['profile_default_binding']['source_identity']['header_sha256'] = 'd' * 64
    response = client.post('/api/analysis-jobs/resolve', json=request)
    assert response.status_code == 409
    assert response.json()['detail']['reason_code'] == 'selection_changed'


@pytest.mark.parametrize('index', [{'job_id': '../../other'}, {'path': 'another receipt'}, {'job_id': 'f' * 32}])
def test_untrusted_or_missing_lookup_never_selects_arbitrary_paths(setup, index):
    service, _, _, request, _ = setup
    complete(service, request)
    pointer = service.directory / 'lookup' / (service._lookup_key(request) + '.json')
    pointer.write_text(json.dumps(index))
    assert service.resolve(request)['status'] == 'not_found'


def test_strict_lookup_request_rejects_extra_paths_and_unsaved_values(setup):
    _, _, _, request, client = setup
    assert client.post('/api/analysis-jobs/resolve', json={**request, 'paths': ['/elsewhere']}).status_code == 422
    unsaved = deepcopy(request)
    unsaved['entries'][0]['values'] = [1]
    assert client.post('/api/analysis-jobs/resolve', json=unsaved).status_code == 422
