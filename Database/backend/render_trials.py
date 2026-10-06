"""Immutable visual trials: saved server recipes, declared settings and owner judgement.

PNG association and generation settings are owner supplied, never execution proof.
All evidence is stored in SQLite so the existing full-database backup includes it.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import struct
from uuid import uuid4
import zlib

from composition_versions import PreparationChangedError, get_composition
from profile_versions import ProfileNotFoundError, ProfileValidationError, _json, _query, _text, _transaction

MAX_PNG = 16 * 1024 * 1024
MAX_METADATA = 1024 * 1024


def initialise_render_trial_schema(conn):
    with _transaction(conn):
        conn.execute('''CREATE TABLE IF NOT EXISTS lora_render_trials (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT, trial_id TEXT UNIQUE NOT NULL,
            idempotency_key TEXT UNIQUE NOT NULL, request_digest TEXT NOT NULL,
            name TEXT NOT NULL, composition_version_id TEXT NOT NULL,
            baseline_trial_id TEXT, receipt_json TEXT NOT NULL, created_at TEXT NOT NULL)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS lora_render_evidence (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT, evidence_id TEXT UNIQUE NOT NULL,
            trial_id TEXT NOT NULL, filename TEXT NOT NULL, sha256 TEXT NOT NULL,
            metadata_json TEXT NOT NULL, png BLOB NOT NULL, created_at TEXT NOT NULL,
            UNIQUE(trial_id, sha256))''')
        conn.execute('''CREATE TABLE IF NOT EXISTS lora_render_assessments (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT, assessment_id TEXT UNIQUE NOT NULL,
            trial_id TEXT NOT NULL, parent_assessment_id TEXT,
            idempotency_key TEXT UNIQUE NOT NULL, request_digest TEXT NOT NULL,
            assessment_json TEXT NOT NULL, created_at TEXT NOT NULL)''')
        for table in ('lora_render_trials', 'lora_render_evidence', 'lora_render_assessments'):
            conn.execute(f'CREATE INDEX IF NOT EXISTS {table}_trial ON {table}(trial_id, sequence)')
            for op in ('UPDATE', 'DELETE'):
                conn.execute(f'''CREATE TRIGGER IF NOT EXISTS {table}_no_{op.lower()}
                    BEFORE {op} ON {table} BEGIN SELECT RAISE(ABORT, 'Render history is immutable'); END''')


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _fields(value, required):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ProfileValidationError('Missing or unexpected render-trial fields')


def validate_generation(value):
    _fields(value, ('checkpoint', 'checkpoint_sha256', 'positive_prompt', 'negative_prompt',
                    'seed', 'width', 'height', 'steps', 'sampler', 'scheduler', 'guidance', 'denoise', 'stage'))
    result = {key: _text(value[key], key, 500) for key in ('checkpoint', 'sampler', 'scheduler')}
    for key in ('positive_prompt', 'negative_prompt'):
        if not isinstance(value[key], str) or len(value[key]) > 20000:
            raise ProfileValidationError('Prompts must be text of at most 20000 characters')
        result[key] = value[key]  # Preserve whitespace exactly.
    if not result['positive_prompt'].strip():
        raise ProfileValidationError('The positive prompt is required')
    seed = value['seed']
    if not isinstance(seed, str) or not re.fullmatch(r'0|[1-9][0-9]{0,19}', seed) or int(seed) > 2**64 - 1:
        raise ProfileValidationError('Use a fixed unsigned 64-bit seed as decimal text')
    result['seed'] = seed
    for key, limit in (('width', 8192), ('height', 8192), ('steps', 1000)):
        if type(value[key]) is not int or not 1 <= value[key] <= limit:
            raise ProfileValidationError(f'{key} is outside the supported trial range')
        result[key] = value[key]
    if result['width'] * result['height'] > 50_000_000:
        raise ProfileValidationError('Trial dimensions exceed 50 million pixels')
    for key, limit in (('guidance', 100), ('denoise', 1)):
        number = value[key]
        if type(number) not in (int, float) or not 0 <= number <= limit or not math.isfinite(number):
            raise ProfileValidationError(f'{key} must be a finite number between 0 and {limit}')
        result[key] = number
    sha = value['checkpoint_sha256']
    if sha is not None and (not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha)):
        raise ProfileValidationError('Checkpoint SHA256 must be lowercase hexadecimal or null')
    result['checkpoint_sha256'] = sha
    if value['stage'] not in ('first_pass', 'final'):
        raise ProfileValidationError('Choose first_pass or final')
    result['stage'] = value['stage']
    return result


def get_trial(conn, trial_id):
    rows = _query(conn, 'SELECT * FROM lora_render_trials WHERE trial_id=?', (_text(trial_id, 'trial_id'),))
    if not rows:
        raise ProfileNotFoundError('Render trial does not exist')
    row = rows[0]
    result = {k: v for k, v in row.items() if k not in ('receipt_json', 'idempotency_key', 'request_digest')}
    result['receipt'] = json.loads(row['receipt_json'])
    result['evidence'] = _query(conn, 'SELECT evidence_id,filename,sha256,metadata_json,created_at FROM lora_render_evidence WHERE trial_id=? ORDER BY sequence', (trial_id,))
    for evidence in result['evidence']:
        evidence['metadata'] = json.loads(evidence.pop('metadata_json'))
    result['assessments'] = _query(conn, 'SELECT assessment_id,parent_assessment_id,assessment_json,created_at FROM lora_render_assessments WHERE trial_id=? ORDER BY sequence', (trial_id,))
    for assessment in result['assessments']:
        assessment['assessment'] = json.loads(assessment.pop('assessment_json'))
    result['requires_revalidation'] = True
    result['generation_verified'] = False
    return result


def list_trials(conn, limit=50, offset=0):
    return _query(conn, '''SELECT trial_id,name,composition_version_id,baseline_trial_id,created_at
        FROM lora_render_trials ORDER BY sequence DESC LIMIT ? OFFSET ?''', (limit, offset))


def create_trial(conn, *, name, composition_version_id, generation, criteria, baseline_trial_id, idempotency_key):
    name, key = _text(name, 'Trial name'), _text(idempotency_key, 'idempotency_key')
    composition_version_id = _text(composition_version_id, 'composition_version_id')
    generation = validate_generation(generation)
    criteria = _text(criteria, 'Intended contributions', 2000)
    baseline_trial_id = _text(baseline_trial_id, 'baseline_trial_id') if baseline_trial_id is not None else None
    digest = _digest(dict(name=name, composition_version_id=composition_version_id, generation=generation, criteria=criteria, baseline_trial_id=baseline_trial_id))
    with _transaction(conn):
        prior = _query(conn, 'SELECT trial_id,request_digest FROM lora_render_trials WHERE idempotency_key=?', (key,))
        if prior:
            if prior[0]['request_digest'] != digest:
                raise PreparationChangedError('This save identifier belongs to a different trial')
            return get_trial(conn, prior[0]['trial_id'])
        recipe = get_composition(conn, composition_version_id)
        if baseline_trial_id is not None:
            baseline = get_trial(conn, baseline_trial_id)
            if baseline['receipt']['declared_generation'] != generation:
                raise ProfileValidationError('A controlled comparison must retain its baseline generation settings')
        receipt = {'composition': recipe, 'declared_generation': generation, 'criteria': criteria,
                   'settings_basis': 'owner_declared', 'image_recipe_match_verified': False,
                   'image_quality_verified': False, 'calibrated_policy': False}
        trial_id = str(uuid4())
        conn.execute('''INSERT INTO lora_render_trials
            (trial_id,idempotency_key,request_digest,name,composition_version_id,baseline_trial_id,receipt_json,created_at)
            VALUES(?,?,?,?,?,?,?,?)''', (trial_id, key, digest, name, composition_version_id, baseline_trial_id, _json(receipt), _now()))
        return get_trial(conn, trial_id)


def inspect_png(data):
    """Bounded chunk/CRC inspection; never execute metadata or claim pixel validation."""
    if not isinstance(data, bytes) or len(data) > MAX_PNG or not data.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ProfileValidationError('Choose a PNG no larger than 16 MiB')
    position, dimensions, has_data, ended, metadata = 8, None, False, False, {}
    text_size = 0
    while position + 12 <= len(data):
        size = struct.unpack('>I', data[position:position + 4])[0]
        kind = data[position + 4:position + 8]
        end = position + size + 12
        if end > len(data):
            raise ProfileValidationError('PNG contains a truncated chunk')
        body = data[position + 8:end - 4]
        crc = struct.unpack('>I', data[end - 4:end])[0]
        if zlib.crc32(kind + body) & 0xffffffff != crc:
            raise ProfileValidationError('PNG chunk checksum failed')
        if dimensions is None:
            if kind != b'IHDR' or size != 13:
                raise ProfileValidationError('PNG requires an initial IHDR chunk')
            dimensions = list(struct.unpack('>II', body[:8]))
            if not all(0 < d <= 8192 for d in dimensions) or dimensions[0] * dimensions[1] > 50_000_000:
                raise ProfileValidationError('PNG dimensions exceed the trial limits')
        elif kind == b'IHDR':
            raise ProfileValidationError('PNG contains a repeated IHDR chunk')
        if kind == b'IDAT':
            has_data = True
        if kind == b'tEXt':
            text_size += size
            if text_size > MAX_METADATA:
                raise ProfileValidationError('PNG text metadata exceeds 1 MiB')
            key, separator, text = body.partition(b'\0')
            if separator and key in (b'prompt', b'workflow'):
                label = key.decode('ascii')
                if label in metadata:
                    raise ProfileValidationError('PNG contains duplicate generation metadata')
                try:
                    value = json.loads(text.decode('utf-8'), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
                    _json(value)
                except (ValueError, UnicodeError, RecursionError, OverflowError):
                    raise ProfileValidationError('PNG generation metadata is not finite JSON') from None
                metadata[label] = value
        position = end
        if kind == b'IEND':
            if size != 0 or position != len(data):
                raise ProfileValidationError('PNG has invalid trailing data')
            ended = True
            break
    if not ended or not has_data or dimensions is None:
        raise ProfileValidationError('PNG is incomplete')
    return {'dimensions': dimensions, 'embedded_untrusted_metadata': metadata,
            'metadata_formats_read': ['tEXt'], 'image_recipe_match_verified': False,
            'pixel_decode_verified': False}


def add_evidence(conn, trial_id, filename, data):
    filename = _text(filename, 'Filename', 250)
    if any(c in filename for c in ('/', '\\', '\0')):
        raise ProfileValidationError('Use a filename without a directory path')
    metadata = inspect_png(data)
    sha = hashlib.sha256(data).hexdigest()
    with _transaction(conn):
        get_trial(conn, trial_id)
        prior = _query(conn, 'SELECT evidence_id FROM lora_render_evidence WHERE trial_id=? AND sha256=?', (trial_id, sha))
        if not prior:
            conn.execute('''INSERT INTO lora_render_evidence
                (evidence_id,trial_id,filename,sha256,metadata_json,png,created_at) VALUES(?,?,?,?,?,?,?)''',
                (str(uuid4()), trial_id, filename, sha, _json(metadata), data, _now()))
        return get_trial(conn, trial_id)


def evidence_bytes(conn, trial_id, evidence_id):
    rows = conn.execute('SELECT png FROM lora_render_evidence WHERE trial_id=? AND evidence_id=?', (trial_id, evidence_id)).fetchall()
    if not rows:
        raise ProfileNotFoundError('Render evidence does not exist in this trial')
    return rows[0][0]


def assess_trial(conn, trial_id, *, assessment, expected_assessment_id, idempotency_key):
    _fields(assessment, ('identity', 'effect', 'colour', 'outcome', 'notes', 'regressions'))
    for field, choices in {
        'identity': ('retained', 'changed', 'uncertain', 'not_assessed', 'not_applicable'),
        'effect': ('recovered', 'partial', 'absent', 'not_assessed'),
        'colour': ('correct', 'partial', 'absent', 'not_assessed', 'not_applicable'),
        'outcome': ('accepted', 'partial', 'no_improvement', 'worse', 'unresolved'),
    }.items():
        if assessment[field] not in choices:
            raise ProfileValidationError(f'Choose a supported {field} assessment')
    for field in ('notes', 'regressions'):
        if not isinstance(assessment[field], str) or len(assessment[field]) > 4000:
            raise ProfileValidationError('Assessment notes must be text of at most 4000 characters')
    if assessment['outcome'] == 'accepted' and (assessment['identity'] not in ('retained', 'not_applicable') or assessment['effect'] != 'recovered' or assessment['colour'] not in ('correct', 'not_applicable') or assessment['regressions'].strip()):
        raise ProfileValidationError('An accepted result requires retained contributions and no recorded regressions')
    key = _text(idempotency_key, 'idempotency_key')
    if expected_assessment_id is not None:
        expected_assessment_id = _text(expected_assessment_id, 'expected_assessment_id')
    digest = _digest(dict(trial_id=trial_id, assessment=assessment, expected_assessment_id=expected_assessment_id))
    with _transaction(conn):
        trial = get_trial(conn, trial_id)
        prior = _query(conn, 'SELECT trial_id,request_digest FROM lora_render_assessments WHERE idempotency_key=?', (key,))
        if prior:
            if prior[0]['request_digest'] != digest:
                raise PreparationChangedError('This save identifier belongs to a different assessment')
            return get_trial(conn, trial_id)
        current = trial['assessments'][-1]['assessment_id'] if trial['assessments'] else None
        if current != expected_assessment_id:
            raise PreparationChangedError('A newer assessment exists; reload it before adding a correction')
        if not trial['evidence']:
            raise ProfileValidationError('Attach the rendered PNG before assessing this trial')
        conn.execute('''INSERT INTO lora_render_assessments
            (assessment_id,trial_id,parent_assessment_id,idempotency_key,request_digest,assessment_json,created_at)
            VALUES(?,?,?,?,?,?,?)''', (str(uuid4()), trial_id, current, key, digest, _json(assessment), _now()))
        return get_trial(conn, trial_id)
