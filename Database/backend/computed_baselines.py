"""Automatic immutable computed baselines, separate from personal variants.

The caller must fresh-check source/profile/loader authority before lookup.
A stored preview is data, never permission to copy a historical export.
"""
from datetime import datetime, timezone
import hashlib
import json
from uuid import uuid4

from profile_versions import ProfileValidationError, _json, _query, _transaction


def initialise_schema(conn):
    with _transaction(conn):
        conn.execute('''CREATE TABLE IF NOT EXISTS lora_computed_baselines (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            baseline_id TEXT NOT NULL UNIQUE,
            context_key TEXT NOT NULL,
            preview_json TEXT NOT NULL,
            preview_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL
        )''')
        conn.execute('CREATE INDEX IF NOT EXISTS computed_baseline_context ON lora_computed_baselines(context_key,sequence)')
        for operation in ('UPDATE', 'DELETE'):
            conn.execute(f'''CREATE TRIGGER IF NOT EXISTS computed_baselines_no_{operation.lower()}
                BEFORE {operation} ON lora_computed_baselines
                BEGIN SELECT RAISE(ABORT, 'Computed baselines are immutable'); END''')


def context_key(context):
    if not isinstance(context, dict) or not context:
        raise ProfileValidationError('A complete server-owned calculation context is required')
    return hashlib.sha256(_json(context).encode()).hexdigest()


def _latest(conn, key):
    rows = _query(conn, 'SELECT * FROM lora_computed_baselines WHERE context_key=? ORDER BY sequence DESC LIMIT 1', (key,))
    if not rows:
        return None
    row = rows[0]
    if hashlib.sha256(row['preview_json'].encode()).hexdigest() != row['preview_sha256']:
        raise ProfileValidationError('The stored computed baseline failed its data integrity check')
    try:
        preview = json.loads(row['preview_json'])
    except (TypeError, ValueError) as exc:
        raise ProfileValidationError('The stored computed baseline is unreadable') from exc
    if not isinstance(preview, dict) or preview.get('status') != 'experimental_preview':
        raise ProfileValidationError('The stored computed baseline is incomplete')
    return {'baseline_id': row['baseline_id'], 'context_key': key, 'created_at': row['created_at'], 'preview': preview}


def resolve_baseline(conn, context, compute, *, force=False):
    if type(force) is not bool:
        raise ProfileValidationError('Recompute must be an explicit boolean')
    key = context_key(context)
    initialise_schema(conn)
    existing = _latest(conn, key)
    if existing is not None and not force:
        return {**existing, 'reused': True}
    preview = compute()
    if not isinstance(preview, dict) or preview.get('status') != 'experimental_preview':
        raise ProfileValidationError('Only complete server-computed previews can be retained')
    encoded = _json(preview)
    with _transaction(conn):
        # Concurrent identical requests share the first committed baseline.
        existing = _latest(conn, key)
        if existing is not None and not force:
            return {**existing, 'reused': True}
        conn.execute('INSERT INTO lora_computed_baselines(baseline_id,context_key,preview_json,preview_sha256,created_at) VALUES(?,?,?,?,?)',
                     (str(uuid4()), key, encoded, hashlib.sha256(encoded.encode()).hexdigest(), datetime.now(timezone.utc).isoformat()))
        return {**_latest(conn, key), 'reused': False}
