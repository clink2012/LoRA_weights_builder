from concurrent.futures import ThreadPoolExecutor
import sqlite3
from threading import Barrier

import pytest

from computed_baselines import initialise_schema, resolve_baseline
from profile_versions import ProfileValidationError


def preview(value=1):
    return {'status': 'experimental_preview', 'values': [9007199254740993, value], 'calibrated': False}


def test_automatic_reuse_does_not_recompute_or_round_saved_values(tmp_path):
    conn = sqlite3.connect(tmp_path / 'cache.db')
    first = resolve_baseline(conn, {'context': 'one'}, preview)
    second = resolve_baseline(conn, {'context': 'one'}, lambda: pytest.fail('must reuse'))
    assert first['reused'] is False and second['reused'] is True
    assert first['baseline_id'] == second['baseline_id']
    assert second['preview']['values'][0] == 9007199254740993
    assert conn.execute('SELECT COUNT(*) FROM lora_computed_baselines').fetchone()[0] == 1
    conn.close()


def test_manual_recompute_adds_history_and_latest_is_reused(tmp_path):
    conn = sqlite3.connect(tmp_path / 'cache.db')
    first = resolve_baseline(conn, {'context': 'one'}, preview)
    second = resolve_baseline(conn, {'context': 'one'}, lambda: preview(.8), force=True)
    assert second['baseline_id'] != first['baseline_id']
    third = resolve_baseline(conn, {'context': 'one'}, lambda: pytest.fail('must reuse'))
    assert third['baseline_id'] == second['baseline_id']
    assert conn.execute('SELECT COUNT(*) FROM lora_computed_baselines').fetchone()[0] == 2
    for statement, args in (('UPDATE lora_computed_baselines SET preview_json=?', ('changed',)),
                            ('DELETE FROM lora_computed_baselines WHERE context_key=?', (first['context_key'],))):
        with pytest.raises(sqlite3.IntegrityError, match='immutable'):
            conn.execute(statement, args)
        conn.rollback()
    conn.close()


def test_context_changes_separate_baselines_and_failed_compute_retains_history(tmp_path):
    conn = sqlite3.connect(tmp_path / 'cache.db')
    first = resolve_baseline(conn, {'context': 'one'}, preview)
    second = resolve_baseline(conn, {'context': 'two'}, preview)
    assert first['context_key'] != second['context_key']
    with pytest.raises(ProfileValidationError):
        resolve_baseline(conn, {'context': 'one'}, lambda: {'status': 'partial'}, force=True)
    assert resolve_baseline(conn, {'context': 'one'}, preview)['baseline_id'] == first['baseline_id']
    assert conn.execute('SELECT COUNT(*) FROM lora_computed_baselines').fetchone()[0] == 2
    conn.close()


def test_concurrent_requests_store_one_matching_baseline(tmp_path):
    path = tmp_path / 'cache.db'
    with sqlite3.connect(path) as conn:
        initialise_schema(conn)
    barrier = Barrier(2)
    def worker():
        conn = sqlite3.connect(path, timeout=10)
        try:
            def calculate():
                barrier.wait(timeout=10)
                return preview()
            return resolve_baseline(conn, {'context': 'one'}, calculate)
        finally:
            conn.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker) for _ in range(2)]
        results = [future.result() for future in futures]
    assert results[0]['baseline_id'] == results[1]['baseline_id']
    assert sorted(r['reused'] for r in results) == [False, True]


@pytest.mark.parametrize('force', [1, 'yes', None])
def test_recompute_requires_explicit_boolean(tmp_path, force):
    with sqlite3.connect(tmp_path / 'cache.db') as conn:
        with pytest.raises(ProfileValidationError):
            resolve_baseline(conn, {'context': 'one'}, preview, force=force)
