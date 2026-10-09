"""Append-only preferred recipe choices, separate from immutable saved values.

Recall compares fresh server-owned source/loader bindings. Historical recipes
are restored as selections only; their saved exports never become Copy authority.
"""
from datetime import datetime, timezone
import hashlib
import re

from composition_versions import (
    PreparationChangedError, get_composition, prepare_composition,
)
from profile_versions import (
    ProfileValidationError, _json, _query, _text, _transaction, capture_default,
    get_version, validate_binding,
)


def initialise_preference_schema(conn):
    with _transaction(conn):
        conn.execute('''CREATE TABLE IF NOT EXISTS lora_composition_preferences (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            selection_key TEXT NOT NULL,
            context_key TEXT,
            version_id TEXT,
            selected_at TEXT NOT NULL
        )''')
        conn.execute('CREATE INDEX IF NOT EXISTS composition_preference_selection ON lora_composition_preferences(selection_key,sequence)')
        for operation in ('UPDATE', 'DELETE'):
            conn.execute(f'''CREATE TRIGGER IF NOT EXISTS composition_preferences_no_{operation.lower()}
                BEFORE {operation} ON lora_composition_preferences
                BEGIN SELECT RAISE(ABORT, 'Composition preferences are append-only'); END''')


def selection(stable_ids, target_contract_id):
    if not isinstance(stable_ids, list) or not 1 <= len(stable_ids) <= 32:
        raise ProfileValidationError('Choose 1 to 32 ordered LoRAs')
    ids = [_text(value, 'stable_id') for value in stable_ids]
    if len(set(ids)) != len(ids):
        raise ProfileValidationError('Each LoRA may appear only once in a combination')
    target = _text(target_contract_id, 'target_contract_id')
    key = hashlib.sha256(_json({'stable_ids': ids, 'target_contract_id': target}).encode()).hexdigest()
    return ids, target, key


def fresh_context(conn, ids, target, default_resolver):
    roots = [default_resolver(conn, sid) for sid in ids]
    bindings = [validate_binding(root['binding']) for root in roots]
    if any(binding['target_contract']['id'] != target for binding in bindings):
        raise ProfileValidationError('The selected target does not match the current loader binding')
    # Settings and edited multipliers belong to the preferred recipe itself.
    # A user's personal role correction must not prevent its own recall.
    key = hashlib.sha256(_json({'stable_ids': ids, 'bindings': bindings}).encode()).hexdigest()
    return roots, bindings, key


def resolve_preference(conn, *, stable_ids, target_contract_id, default_resolver, preparation_resolver):
    ids, target, key = selection(stable_ids, target_contract_id)
    rows = _query(conn, 'SELECT * FROM lora_composition_preferences WHERE selection_key=? ORDER BY sequence DESC LIMIT 1', (key,))
    if not rows or rows[0]['version_id'] is None:
        return {'status': 'none', 'stable_ids': ids, 'target_contract_id': target, 'recipe': None}
    row = rows[0]
    recipe = get_composition(conn, row['version_id'])
    _, bindings, context = fresh_context(conn, ids, target, default_resolver)
    profiles = [get_version(conn, entry['stable_id'], entry['profile_version_id']) for entry in recipe['entries']]
    matches = (recipe['target_contract_id'] == target
               and [entry['stable_id'] for entry in recipe['entries']] == ids
               and [profile['binding'] for profile in profiles] == bindings
               and row['context_key'] == context)
    if not matches:
        return {'status': 'needs_review', 'stable_ids': ids, 'target_contract_id': target,
                'recipe': None, 'retained_version_id': recipe['version_id'],
                'reason': 'The source or loader binding changed. Your preferred recipe is retained in history; review it against the current files.'}
    # Fresh preparation checks current loader coverage and the pinned versions.
    # It is deliberately not returned as a copyable export by this endpoint.
    prepare_composition(conn, recipe['version_id'], preparation_resolver=preparation_resolver)
    return {'status': 'preferred', 'stable_ids': ids, 'target_contract_id': target, 'recipe': recipe}


def choose_preference(conn, *, version_id, expected_preparation_digest, default_resolver, preparation_resolver):
    if not isinstance(expected_preparation_digest, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_preparation_digest):
        raise ProfileValidationError('Prepare and review the recipe before making it preferred')
    recipe = get_composition(conn, version_id)
    ids, target, key = selection([entry['stable_id'] for entry in recipe['entries']], recipe['target_contract_id'])
    prepared = prepare_composition(conn, version_id, preparation_resolver=preparation_resolver)
    if prepared['preparation_digest'] != expected_preparation_digest:
        raise PreparationChangedError('Preparation changed. Prepare the current recipe again before making it preferred.')
    _, bindings, context = fresh_context(conn, ids, target, default_resolver)
    if [get_version(conn, entry['stable_id'], entry['profile_version_id'])['binding'] for entry in recipe['entries']] != bindings:
        raise PreparationChangedError('The recipe belongs to an earlier source or loader binding')
    with _transaction(conn):
        _append(conn, key, context, recipe['version_id'])
    return {'status': 'preferred', 'stable_ids': ids, 'target_contract_id': target, 'recipe': recipe}


def _append(conn, key, context, version_id):
    conn.execute('INSERT INTO lora_composition_preferences(selection_key,context_key,version_id,selected_at) VALUES(?,?,?,?)',
                 (key, context, version_id, datetime.now(timezone.utc).isoformat()))


def restore_originals(conn, *, stable_ids, target_contract_id, default_resolver):
    ids, target, key = selection(stable_ids, target_contract_id)
    roots, _, _ = fresh_context(conn, ids, target, default_resolver)
    # Capture each immutable Default explicitly. If a later capture fails, no
    # preference changes and no prior personal values are deleted.
    entries = [{'stable_id': sid, 'profile_version_id': capture_default(conn, stable_id=sid, **root)['version_id']}
               for sid, root in zip(ids, roots)]
    with _transaction(conn):
        _append(conn, key, None, None)
    return {'status': 'originals', 'stable_ids': ids, 'target_contract_id': target,
            'entries': entries, 'requires_revalidation': True}
