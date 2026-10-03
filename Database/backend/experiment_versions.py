"""Atomic experiment history. This module is an internal service, not an API.

Only trusted server callbacks may produce plans or loader preparations. A future
router must accept identifiers, names and digests only, never client snapshots.
Private no-commit store helpers are intentionally used inside one transaction;
calling public profile/composition save methods here would split that boundary.
"""
from datetime import datetime, timezone
import hashlib
import json
import re
from uuid import uuid4

from composition_versions import PreparationChangedError, _insert_composition, get_composition
from profile_versions import (
    ProfileNotFoundError, ProfileValidationError, _insert, _json, _lineage,
    _query, _text, _transaction, get_version, validate_snapshot,
)


def _digest(value):
    try:
        return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProfileValidationError('Experiment data must be finite JSON values') from exc


def experiment_plan_digest(plan):
    if not isinstance(plan, dict):
        raise ProfileValidationError('A server-owned experiment plan is required')
    return _digest({key: value for key, value in plan.items() if key != 'proposal_digest'})


def initialise_experiment_schema(conn):
    with _transaction(conn):
        conn.execute('''CREATE TABLE IF NOT EXISTS lora_experiment_versions (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT NOT NULL UNIQUE,
            idempotency_key TEXT NOT NULL UNIQUE,
            request_digest TEXT NOT NULL,
            job_id TEXT NOT NULL,
            composition_version_id TEXT NOT NULL,
            receipt_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )''')
        for operation in ('UPDATE', 'DELETE'):
            conn.execute(f'''CREATE TRIGGER IF NOT EXISTS experiment_versions_no_{operation.lower()}
                BEFORE {operation} ON lora_experiment_versions
                BEGIN SELECT RAISE(ABORT, 'Experiment history is immutable'); END''')


def _decode(conn, row):
    result = {key: value for key, value in row.items() if key not in ('receipt_json', 'idempotency_key', 'request_digest')}
    result['receipt'] = json.loads(row['receipt_json'])
    result['composition'] = get_composition(conn, row['composition_version_id'])
    result['requires_revalidation'] = True
    result['status'] = 'saved'
    return result


def get_experiment(conn, experiment_id):
    rows = _query(conn, 'SELECT * FROM lora_experiment_versions WHERE experiment_id=?',
                  (_text(experiment_id, 'experiment_id'),))
    if not rows:
        raise ProfileNotFoundError('Experiment does not exist')
    return _decode(conn, rows[0])


def _replay(conn, key, request_digest):
    rows = _query(conn, 'SELECT * FROM lora_experiment_versions WHERE idempotency_key=?', (key,))
    if not rows:
        return None
    if rows[0]['request_digest'] != request_digest:
        raise PreparationChangedError('This save identifier already belongs to a different experiment request')
    return _decode(conn, rows[0])


def _validate_plan(plan, job_id, expected_digest):
    required = {'job_id', 'engine_version', 'policy_version', 'target_contract_id',
                'metrics_receipt', 'metrics_receipt_sha256', 'entries', 'proposal_digest',
                'input_preparation_digest', 'policy_preview'}
    if not isinstance(plan, dict) or set(plan) != required or plan['job_id'] != job_id:
        raise ProfileValidationError('The server experiment plan is incomplete or belongs to a different job')
    if experiment_plan_digest(plan) != expected_digest or plan['proposal_digest'] != expected_digest:
        raise PreparationChangedError('The experiment changed since its preview; review a fresh proposal before saving')
    for key in ('engine_version', 'policy_version', 'target_contract_id'):
        _text(plan[key], key)
    if not isinstance(plan['input_preparation_digest'], str) or not re.fullmatch('[0-9a-f]{64}', plan['input_preparation_digest']):
        raise ProfileValidationError('The pinned input preparation digest is required')
    metrics = plan['metrics_receipt']
    if (not isinstance(metrics, dict) or metrics.get('status') != 'complete'
            or metrics.get('engine_version') != plan['engine_version']
            or metrics.get('target_contract_id') != plan['target_contract_id']
            or _digest(metrics) != plan['metrics_receipt_sha256']):
        raise ProfileValidationError('A complete matching server metrics receipt is required')
    entries = plan['entries']
    if not isinstance(entries, list) or not 2 <= len(entries) <= 8:
        raise ProfileValidationError('An experiment requires two to eight ordered pinned LoRAs')
    preview = plan['policy_preview']
    if (not isinstance(preview, dict) or preview.get('policy_version') != plan['policy_version']
            or preview.get('status') != 'experimental_preview' or preview.get('calibrated') is not False
            or not isinstance(preview.get('entries'), list) or len(preview['entries']) != len(entries)
            or not isinstance(preview.get('constants'), dict) or not isinstance(preview.get('changes'), list)
            or not isinstance(preview.get('blocks'), list)):
        raise ProfileValidationError('The complete matching policy preview must be retained')
    identities = set()
    for entry, proposed in zip(entries, preview['entries']):
        if not isinstance(entry, dict) or set(entry) != {'stable_id', 'parent_version_id', 'values', 'settings', 'ab', 'priority'}:
            raise ProfileValidationError('Every proposed snapshot requires an explicit saved parent and priority')
        sid = _text(entry['stable_id'], 'stable_id')
        _text(entry['parent_version_id'], 'parent_version_id')
        if sid in identities or type(entry['priority']) is not int or entry['priority'] not in (0, 1, 2):
            raise ProfileValidationError('Experiment LoRAs must be unique with explicit valid priorities')
        if (not isinstance(proposed, dict) or proposed.get('stable_id') != sid
                or proposed.get('profile_version_id') != entry['parent_version_id']
                or proposed.get('priority') != entry['priority'] or proposed.get('values') != entry['values']
                or not isinstance(proposed.get('before_values'), list)):
            raise ProfileValidationError('Proposed snapshots must exactly match the ordered policy preview')
        identities.add(sid)


def save_experiment(conn, *, job_id, expected_proposal_digest, name, idempotency_key,
                    experiment_resolver, preparation_resolver, parent_version_id=None):
    """Save changed children, full recipe and provenance together or save nothing.

    experiment_resolver(conn, job_id) must fresh-check the server-owned job,
    measurements, selection and proposal; it runs before the write transaction.
    preparation_resolver(conn, ordered_refs, target) MUST use this SAME connection,
    freshly validate current bindings and export inserted snapshots. It runs inside
    the transaction: full tensor analysis must never happen in that callback.
    Exact idempotent retries return historical data and do not grant Copy authority.
    """
    job_id, name = _text(job_id, 'job_id'), _text(name, 'Experiment name')
    if name.casefold() == 'default':
        raise ProfileValidationError('Default is reserved; choose an experiment name')
    key = _text(idempotency_key, 'idempotency_key')
    if not isinstance(expected_proposal_digest, str) or not re.fullmatch('[0-9a-f]{64}', expected_proposal_digest):
        raise ProfileValidationError('A current server proposal digest is required')
    if parent_version_id is not None:
        parent_version_id = _text(parent_version_id, 'parent_version_id')
    request_digest = _digest({'job_id': job_id, 'proposal_digest': expected_proposal_digest,
                             'name': name, 'parent_version_id': parent_version_id})
    replay = _replay(conn, key, request_digest)
    if replay is not None:
        return replay
    plan = experiment_resolver(conn, job_id)
    _validate_plan(plan, job_id, expected_proposal_digest)
    # Detach the audited input from mutable callback-owned objects before writes.
    plan = json.loads(_json(plan))
    with _transaction(conn):
        replay = _replay(conn, key, request_digest)
        if replay is not None:
            return replay
        parent_recipe = get_composition(conn, parent_version_id) if parent_version_id else None
        experiment_id, refs, created = str(uuid4()), [], []
        provenance = {'method': 'gentle_balance_experiment', 'experiment_id': experiment_id,
                      'job_id': job_id, 'engine_version': plan['engine_version'], 'policy_version': plan['policy_version'],
                      'metrics_receipt_sha256': plan['metrics_receipt_sha256'], 'proposal_digest': expected_proposal_digest,
                      'calibrated': False, 'image_quality_verified': False}
        for entry, proposed in zip(plan['entries'], plan['policy_preview']['entries']):
            parent = get_version(conn, entry['stable_id'], entry['parent_version_id'])
            _lineage(conn, entry['stable_id'], parent['default_id'], parent['version_id'], parent['binding'])
            if parent['binding']['target_contract']['id'] != plan['target_contract_id']:
                raise ProfileValidationError('The proposed target differs from its saved profile lineage')
            snapshot = validate_snapshot(parent['binding'], entry['values'], entry['settings'], entry['ab'])
            # Gentle balancing changes blocks only. Roles and global/CLIP settings
            # must stay those of the reviewed pinned parent, not an implicit edit.
            if snapshot['settings'] != parent['settings']:
                raise ProfileValidationError('An experiment must preserve its saved supporting settings')
            if proposed['before_values'] != parent['values']:
                raise ProfileValidationError('The policy preview must begin with the exact saved parent values')
            version = parent
            changed_values = snapshot['values'] != parent['values']
            if not changed_values and snapshot['ab'] != parent['ab']:
                raise ProfileValidationError('An unchanged experiment entry must preserve its saved A/B metadata')
            if changed_values:
                version = _insert(conn, stable_id=entry['stable_id'], default_id=parent['default_id'],
                                  parent_id=parent['version_id'], kind='personal', name=name,
                                  binding=parent['binding'], snapshot=snapshot,
                                  provenance={**provenance, 'priority': entry['priority'], 'parent_ab': parent['ab'],
                                              'ab_changed': snapshot['ab'] != parent['ab']})
                created.append(version['version_id'])
            refs.append({'stable_id': entry['stable_id'], 'profile_version_id': version['version_id']})
        if not created:
            return {'status': 'no_changes', 'job_id': job_id, 'proposal_digest': expected_proposal_digest,
                    'reason': 'This experiment proposes no changes; no profile or recipe history was added.',
                    'requires_revalidation': True}
        prepared = preparation_resolver(conn, refs, plan['target_contract_id'])
        composition = _insert_composition(conn, name=name, entries=refs, target_contract_id=plan['target_contract_id'],
                                          prepared=prepared, parent=parent_recipe)
        receipt = {'plan': plan, 'created_profile_version_ids': created, 'provenance': provenance}
        conn.execute('''INSERT INTO lora_experiment_versions
            (experiment_id,idempotency_key,request_digest,job_id,composition_version_id,receipt_json,created_at)
            VALUES(?,?,?,?,?,?,?)''',
            (experiment_id, key, request_digest, job_id, composition['version_id'], _json(receipt), datetime.now(timezone.utc).isoformat()))
        return get_experiment(conn, experiment_id)
