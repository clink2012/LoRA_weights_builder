"""Explicit, additive storage for immutable profile snapshots.

These functions never infer a model layout from a vector length, migrate legacy
rows in place, or initialise tables during reads. Call initialise_schema on a
verified database copy before integrating it into an application migration.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import sqlite3
from typing import Any
from uuid import uuid4


class ProfileValidationError(ValueError):
    pass


class ProfileNotFoundError(LookupError):
    pass


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: Any, label: str, limit: int = 200) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ProfileValidationError(f"{label} must be non-empty text of at most {limit} characters")
    return value.strip()


def _number(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileValidationError(f"{label} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ProfileValidationError(f"{label} must be a finite number")
    return value


def validate_binding(binding: Any) -> dict:
    required = {"architecture", "slots", "source_identity", "engine_version", "policy_version", "target_contract", "loader_adapter"}
    if not isinstance(binding, dict) or set(binding) != required:
        raise ProfileValidationError("Binding requires architecture, slots, source_identity, engine_version, policy_version, target_contract and loader_adapter")
    result = {key: _text(binding[key], key) for key in ("architecture", "engine_version", "policy_version")}
    target, loader = binding["target_contract"], binding["loader_adapter"]
    if not isinstance(target, dict) or set(target) != {"id", "sha256"}:
        raise ProfileValidationError("Target contract requires id and content SHA-256")
    if not isinstance(loader, dict) or set(loader) != {"id", "source_sha256"}:
        raise ProfileValidationError("Loader adapter requires id and named source SHA-256 digests")
    if not isinstance(loader["source_sha256"], dict) or not loader["source_sha256"]:
        raise ProfileValidationError("Loader adapter requires named source SHA-256 digests")
    digests = [("target", target["sha256"]), *loader["source_sha256"].items()]
    for name, digest in digests:
        _text(name, "Source name", 500)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ProfileValidationError("Contract/source fingerprints must be SHA-256 digests")
    result["target_contract"] = {"id": _text(target["id"], "Target contract id"), "sha256": target["sha256"].lower()}
    result["loader_adapter"] = {"id": _text(loader["id"], "Loader adapter id"),
                                "source_sha256": {key: value.lower() for key, value in loader["source_sha256"].items()}}
    identity = binding["source_identity"]
    if not isinstance(identity, dict):
        raise ProfileValidationError("Source identity requires an explicit fingerprint basis")
    identity = dict(identity)
    if identity.get("basis") == "file_sha256" and set(identity) == {"basis", "sha256"}:
        digest_key = "sha256"
    elif identity.get("basis") == "header_stat" and set(identity) == {"basis", "header_sha256", "size_bytes", "mtime_ns"}:
        digest_key = "header_sha256"
        for key in ("size_bytes", "mtime_ns"):
            if type(identity[key]) is not int or identity[key] < 0:
                raise ProfileValidationError("Source size/mtime must be non-negative integers")
    else:
        raise ProfileValidationError("Source identity must describe full-file SHA-256 or header/stat evidence explicitly")
    if not isinstance(identity[digest_key], str) or not re.fullmatch(r"[0-9a-fA-F]{64}", identity[digest_key]):
        raise ProfileValidationError("Source fingerprint must contain a SHA-256 digest")
    identity[digest_key] = identity[digest_key].lower()
    result["source_identity"] = identity
    slots = binding["slots"]
    if not isinstance(slots, list) or not 1 <= len(slots) <= 4096:
        raise ProfileValidationError("Binding must contain 1 to 4096 ordered slots")
    result["slots"] = []
    labels = set()
    for slot in slots:
        if not isinstance(slot, dict) or set(slot) != {"group", "label"}:
            raise ProfileValidationError("Each slot requires a group and a unique label")
        parsed = {key: _text(slot[key], f"slot {key}", 100) for key in slot}
        if parsed["label"] in labels:
            raise ProfileValidationError("Slot labels must be unique across groups")
        labels.add(parsed["label"])
        result["slots"].append(parsed)
    return result


def validate_snapshot(binding: dict, values: Any, settings: Any, ab: Any = None) -> dict:
    if not isinstance(values, list) or len(values) != len(binding["slots"]):
        raise ProfileValidationError("The value count must match every ordered binding slot")
    values = [_number(value, "Block value") for value in values]
    required = {"role", "strength_model", "strength_clip", "affect_clip"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ProfileValidationError("Settings require role, strength_model, strength_clip and affect_clip")
    settings = dict(settings)
    settings["role"] = _text(settings["role"], "role", 100)
    settings["strength_model"] = _number(settings["strength_model"], "strength_model")
    if settings["strength_clip"] is not None:
        settings["strength_clip"] = _number(settings["strength_clip"], "strength_clip")
    if not isinstance(settings["affect_clip"], bool):
        raise ProfileValidationError("affect_clip must be true or false")
    if settings["affect_clip"] and settings["strength_clip"] is None:
        raise ProfileValidationError("Enabled CLIP settings require a numeric strength_clip")
    if ab is None:
        ab = {}
    if not isinstance(ab, dict) or not set(ab).issubset({"A", "B"}):
        raise ProfileValidationError("Experiments may define A and/or B only")
    slot_values = dict(zip((slot["label"] for slot in binding["slots"]), values))
    parsed_ab, used_slots = {}, set()
    for symbol, experiment in ab.items():
        if not isinstance(experiment, dict) or set(experiment) != {"slot_labels", "min", "max", "value", "basis"}:
            raise ProfileValidationError(f"{symbol} requires slot_labels, min, max, value and basis")
        minimum, maximum, value = [_number(experiment[key], f"{symbol}.{key}") for key in ("min", "max", "value")]
        if not minimum <= value <= maximum:
            raise ProfileValidationError(f"{symbol} value must lie within its ordered min/max bounds")
        labels = experiment["slot_labels"]
        if not isinstance(labels, list) or not labels or not all(isinstance(label, str) for label in labels):
            raise ProfileValidationError(f"{symbol} requires named target slots")
        if len(set(labels)) != len(labels) or set(labels) & used_slots or not set(labels).issubset(slot_values):
            raise ProfileValidationError("A/B slots must exist and must not be repeated or shared")
        if any(slot_values[label] != value for label in labels):
            raise ProfileValidationError("Resolved A/B values must match the saved numeric vector")
        used_slots.update(labels)
        parsed_ab[symbol] = {"slot_labels": labels, "min": minimum, "max": maximum,
                             "value": value, "basis": _text(experiment["basis"], "Experiment basis", 1000)}
    return {"values": values, "settings": settings, "ab": parsed_ab}


@contextmanager
def _transaction(conn: sqlite3.Connection):
    if conn.in_transaction:
        raise ProfileValidationError("Profile operations require a connection without an active transaction")
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def initialise_schema(conn: sqlite3.Connection) -> None:
    """Add only new tables and immutability triggers; never alter legacy tables."""
    statements = [
        """CREATE TABLE IF NOT EXISTS lora_profile_versions (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            version_id TEXT NOT NULL UNIQUE,
            stable_id TEXT NOT NULL,
            default_id TEXT NOT NULL,
            parent_id TEXT,
            kind TEXT NOT NULL CHECK(kind IN ('default', 'personal', 'legacy')),
            name TEXT NOT NULL,
            binding_json TEXT NOT NULL,
            snapshot_json TEXT NOT NULL,
            provenance_json TEXT NOT NULL,
            import_key TEXT UNIQUE,
            created_at TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS lora_profile_selections (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            stable_id TEXT NOT NULL,
            default_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            selected_at TEXT NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS profile_versions_lineage ON lora_profile_versions(stable_id, default_id, sequence)",
        "CREATE INDEX IF NOT EXISTS profile_selections_lineage ON lora_profile_selections(stable_id, default_id, sequence)",
    ]
    for table in ("lora_profile_versions", "lora_profile_selections"):
        for operation in ("UPDATE", "DELETE"):
            statements.append(f"""CREATE TRIGGER IF NOT EXISTS {table}_no_{operation.lower()}
                BEFORE {operation} ON {table}
                BEGIN SELECT RAISE(ABORT, 'Profile history is immutable'); END""")
    with _transaction(conn):
        for statement in statements:
            conn.execute(statement)


def _query(conn: sqlite3.Connection, sql: str, parameters: tuple = ()) -> list[dict]:
    cursor = conn.execute(sql, parameters)
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def _decode(row: dict) -> dict:
    row = dict(row)
    row["binding"] = json.loads(row.pop("binding_json"))
    row.update(json.loads(row.pop("snapshot_json")))
    row["provenance"] = json.loads(row.pop("provenance_json"))
    row.pop("import_key")
    return row


def get_version(conn: sqlite3.Connection, stable_id: str, version_id: str) -> dict:
    stable_id, version_id = _text(stable_id, "stable_id"), _text(version_id, "version_id")
    rows = _query(conn, "SELECT * FROM lora_profile_versions WHERE stable_id=? AND version_id=?", (stable_id, version_id))
    if not rows:
        raise ProfileNotFoundError("Profile version does not exist for this LoRA")
    return _decode(rows[0])


def _insert(conn, *, stable_id, default_id, parent_id, kind, name, binding, snapshot, provenance, import_key=None):
    version_id = str(uuid4())
    conn.execute("""INSERT INTO lora_profile_versions
        (version_id,stable_id,default_id,parent_id,kind,name,binding_json,snapshot_json,provenance_json,import_key,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (version_id, stable_id, default_id or version_id, parent_id, kind, name, _json(binding),
         _json(snapshot), _json(provenance), import_key, datetime.now(timezone.utc).isoformat()))
    return get_version(conn, stable_id, version_id)


def capture_default(conn, *, stable_id, binding, values, settings, ab=None) -> dict:
    """Explicit capture; returning an existing identical root is idempotent."""
    stable_id = _text(stable_id, "stable_id")
    binding = validate_binding(binding)
    snapshot = validate_snapshot(binding, values, settings, ab)
    with _transaction(conn):
        matches = _query(conn, "SELECT * FROM lora_profile_versions WHERE stable_id=? AND kind='default' AND binding_json=?",
                         (stable_id, _json(binding)))
        if matches:
            root = _decode(matches[0])
            if any(root[key] != snapshot[key] for key in snapshot):
                raise ProfileValidationError("This exact binding already has an immutable Default; save a personal revision")
            return root
        return _insert(conn, stable_id=stable_id, default_id=None, parent_id=None, kind="default", name="Default",
                       binding=binding, snapshot=snapshot, provenance={"method": "explicit_default_capture"})


def _lineage(conn, stable_id, default_id, parent_id, binding):
    root = get_version(conn, stable_id, default_id)
    parent = get_version(conn, stable_id, parent_id)
    if root["kind"] != "default" or root["default_id"] != default_id or parent["default_id"] != default_id:
        raise ProfileValidationError("Parent and Default must belong to the same LoRA and Default lineage")
    if validate_binding(binding) != root["binding"] or parent["binding"] != root["binding"]:
        raise ProfileValidationError("Binding changed; capture a separate verified Default instead of mixing lineages")
    return root


def create_revision(conn, *, stable_id, default_id, parent_id, name, binding, values, settings, ab=None) -> dict:
    stable_id = _text(stable_id, "stable_id")
    default_id, parent_id = _text(default_id, "default_id"), _text(parent_id, "parent_id")
    name = _text(name, "Revision name")
    if name.casefold() == "default":
        raise ProfileValidationError("Default is reserved; choose a personal revision name")
    binding = validate_binding(binding)
    snapshot = validate_snapshot(binding, values, settings, ab)
    with _transaction(conn):
        _lineage(conn, stable_id, default_id, parent_id, binding)
        return _insert(conn, stable_id=stable_id, default_id=default_id, parent_id=parent_id,
                       kind="personal", name=name, binding=binding, snapshot=snapshot, provenance={"method": "manual_revision"})


def list_versions(conn, stable_id: str, default_id: str | None = None) -> list[dict]:
    stable_id = _text(stable_id, "stable_id")
    sql, parameters = "SELECT * FROM lora_profile_versions WHERE stable_id=?", (stable_id,)
    if default_id is not None:
        default_id = _text(default_id, "default_id")
        sql, parameters = sql + " AND default_id=?", parameters + (default_id,)
    return [_decode(row) for row in _query(conn, sql + " ORDER BY sequence", parameters)]


def select_version(conn, *, stable_id: str, default_id: str, version_id: str) -> dict:
    stable_id = _text(stable_id, "stable_id")
    default_id, version_id = _text(default_id, "default_id"), _text(version_id, "version_id")
    with _transaction(conn):
        version = get_version(conn, stable_id, version_id)
        _lineage(conn, stable_id, default_id, version_id, version["binding"])
        conn.execute("INSERT INTO lora_profile_selections(stable_id,default_id,version_id,selected_at) VALUES(?,?,?,?)",
                     (stable_id, default_id, version_id, datetime.now(timezone.utc).isoformat()))
        return version


def get_selection(conn, stable_id: str, default_id: str) -> dict:
    stable_id, default_id = _text(stable_id, "stable_id"), _text(default_id, "default_id")
    root = get_version(conn, stable_id, default_id)
    if root["kind"] != "default":
        raise ProfileValidationError("Selection requires a Default lineage")
    rows = _query(conn, "SELECT version_id FROM lora_profile_selections WHERE stable_id=? AND default_id=? ORDER BY sequence DESC LIMIT 1",
                  (stable_id, default_id))
    return get_version(conn, stable_id, rows[0]["version_id"]) if rows else root


def import_legacy_profiles(conn, *, stable_id: str, default_id: str, confirm_mapping: bool = False) -> dict:
    """Explicitly attach recognised legacy vectors to a verified ordered Default.

    Older rows contain no fingerprint or settings. Values/name are preserved;
    inherited Default settings are explicitly recorded as supplied context, not
    invented historic evidence. Unknown shapes stay exclusively in legacy data.
    """
    if confirm_mapping is not True:
        raise ProfileValidationError("Legacy import requires explicit confirmation of the source/layout mapping")
    stable_id, default_id = _text(stable_id, "stable_id"), _text(default_id, "default_id")
    root = get_version(conn, stable_id, default_id)
    if root["kind"] != "default":
        raise ProfileValidationError("Legacy import requires a Default lineage")
    tables = _query(conn, "SELECT name FROM sqlite_master WHERE type='table' AND name='lora_user_profiles'")
    if not tables:
        return {"imported": [], "existing": [], "retained": [], "reason": "No legacy profile table"}
    columns = {row["name"] for row in _query(conn, "PRAGMA table_info(lora_user_profiles)")}
    if not {"id", "stable_id", "profile_name", "block_weights"}.issubset(columns):
        return {"imported": [], "existing": [], "retained": [], "reason": "Unrecognised legacy schema retained untouched"}
    result = {"imported": [], "existing": [], "retained": []}
    with _transaction(conn):
        rows = _query(conn, "SELECT * FROM lora_user_profiles WHERE stable_id=? ORDER BY id", (stable_id,))
        for row in rows:
            try:
                name = _text(row["profile_name"], "Legacy profile name")
                values = json.loads(row["block_weights"])
                snapshot = validate_snapshot(root["binding"], values, root["settings"])
                row_digest = hashlib.sha256(_json(row).encode()).hexdigest()
            except (ProfileValidationError, ValueError, TypeError) as exc:
                result["retained"].append({"legacy_id": row["id"], "reason": str(exc)})
                continue
            key = f"lora_user_profiles:{row['id']}:{row_digest}"
            existing = _query(conn, "SELECT * FROM lora_profile_versions WHERE import_key=?", (key,))
            if existing:
                if existing[0]["default_id"] != default_id:
                    raise ProfileValidationError("This legacy snapshot is already bound to a different Default")
                result["existing"].append(existing[0]["version_id"])
                continue
            version = _insert(conn, stable_id=stable_id, default_id=default_id, parent_id=default_id, kind="legacy",
                              name=name, binding=root["binding"], snapshot=snapshot, import_key=key,
                              provenance={"method": "explicit_legacy_import", "legacy_id": row["id"],
                                          "legacy_sha256": row_digest, "context": "Source/layout mapping confirmed at import; settings inherited from Default"})
            result["imported"].append(version["version_id"])
    return result
