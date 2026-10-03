"""Immutable ordered compositions; historical exports require fresh preparation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import re
import sqlite3
from uuid import uuid4

from profile_versions import (
    ProfileNotFoundError, ProfileValidationError, _json, _query, _text,
    _transaction, get_version,
)


class PreparationChangedError(ProfileValidationError):
    pass


def preparation_digest(prepared: dict) -> str:
    """Digest all server preparation fields except the digest itself."""
    if not isinstance(prepared, dict):
        raise ProfileValidationError("Preparation must be a server result object")
    try:
        serialized = _json({key: value for key, value in prepared.items() if key != "preparation_digest"})
    except (TypeError, ValueError) as exc:
        raise ProfileValidationError("Preparation contains non-finite or unsupported values") from exc
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_entries(entries) -> list[dict]:
    if not isinstance(entries, list) or not 1 <= len(entries) <= 32:
        raise ProfileValidationError("A composition requires 1 to 32 ordered LoRAs")
    parsed = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"stable_id", "profile_version_id"}:
            raise ProfileValidationError("Each entry requires stable_id and an explicit profile_version_id, including Default")
        parsed.append({key: _text(entry[key], key) for key in entry})
    if len({entry["stable_id"] for entry in parsed}) != len(parsed):
        raise ProfileValidationError("A LoRA may appear only once in an ordered composition")
    return parsed


def initialise_composition_schema(conn: sqlite3.Connection) -> None:
    with _transaction(conn):
        conn.execute("""CREATE TABLE IF NOT EXISTS lora_composition_versions (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            version_id TEXT NOT NULL UNIQUE,
            composition_id TEXT NOT NULL,
            parent_version_id TEXT,
            name TEXT NOT NULL,
            target_contract_id TEXT NOT NULL,
            entries_json TEXT NOT NULL,
            snapshot_json TEXT NOT NULL,
            preparation_digest TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS composition_history ON lora_composition_versions(composition_id,sequence)")
        for operation in ("UPDATE", "DELETE"):
            conn.execute(f"""CREATE TRIGGER IF NOT EXISTS composition_versions_no_{operation.lower()}
                BEFORE {operation} ON lora_composition_versions
                BEGIN SELECT RAISE(ABORT, 'Composition history is immutable'); END""")


def _decode(row):
    result = dict(row)
    result["entries"] = json.loads(result.pop("entries_json"))
    # Explicit nesting prevents the normal Studio reader from mistaking a
    # historical receipt for current node_payloads/ready export authority.
    result["historical_snapshot"] = json.loads(result.pop("snapshot_json"))
    result["requires_revalidation"] = True
    return result


def get_composition(conn, version_id: str) -> dict:
    version_id = _text(version_id, "version_id")
    rows = _query(conn, "SELECT * FROM lora_composition_versions WHERE version_id=?", (version_id,))
    if not rows:
        raise ProfileNotFoundError("Composition version does not exist")
    return _decode(rows[0])


def list_compositions(conn, composition_id: str | None = None) -> list[dict]:
    sql = "SELECT sequence,version_id,composition_id,parent_version_id,name,target_contract_id,entries_json,created_at FROM lora_composition_versions"
    parameters = ()
    if composition_id is not None:
        sql += " WHERE composition_id=?"
        parameters = (_text(composition_id, "composition_id"),)
    rows = _query(conn, sql + " ORDER BY sequence DESC", parameters)
    for row in rows:
        row["entry_count"] = len(json.loads(row.pop("entries_json")))
        row["requires_revalidation"] = True
    return rows


def _validate_prepared(conn, entries, target_contract_id, prepared):
    expected_ids = [entry["stable_id"] for entry in entries]
    if (not isinstance(prepared, dict) or prepared.get("compatible") is not True
            or prepared.get("target_contract_id") != target_contract_id
            or prepared.get("requested_loras") != expected_ids
            or prepared.get("included_loras") != expected_ids
            or prepared.get("excluded_loras") != []):
        raise ProfileValidationError("The complete ordered composition must be freshly prepared for the chosen target")
    nodes = prepared.get("node_payloads")
    if not isinstance(nodes, list) or len(nodes) != len(entries):
        raise ProfileValidationError("Every composition entry requires its own server-prepared node")
    for entry, node in zip(entries, nodes):
        version = get_version(conn, entry["stable_id"], entry["profile_version_id"])
        if (not isinstance(node, dict) or node.get("stable_id") != entry["stable_id"]
                or node.get("profile_version_id") != entry["profile_version_id"]):
            raise ProfileValidationError("Prepared nodes must preserve ordered LoRA and profile version identities")
        export = node.get("loader_export")
        if not isinstance(export, dict) or export.get("status") != "ready" or export.get("target_contract_id") != target_contract_id:
            raise ProfileValidationError("All composition entries must have a ready current loader mapping")
        if version["binding"]["target_contract"]["id"] != target_contract_id:
            raise ProfileValidationError("Profile target differs from the composition target")
        if export.get("architecture_slot_values") != version["values"] or export.get("architecture_slot_labels") != [slot["label"] for slot in version["binding"]["slots"]]:
            raise ProfileValidationError("Prepared canonical values must match the pinned immutable profile")
        for setting in ("strength_model", "strength_clip", "affect_clip", "role"):
            if node.get(setting) != version["settings"][setting]:
                raise ProfileValidationError("Prepared supporting settings differ from the pinned profile")
        values, labels, csv = export.get("slot_values"), export.get("slot_labels"), export.get("numeric_csv")
        if not isinstance(values, list) or not values or not isinstance(labels, list) or len(labels) != len(values) or not isinstance(csv, str):
            raise ProfileValidationError("Prepared loader vector is incomplete")
        fields = csv.split(",")
        if (len(fields) != len(values) or export.get("loader_slot_count") != len(values)
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
                or any(not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", field) for field in fields)
                or [float(field) for field in fields] != values):
            raise ProfileValidationError("Prepared loader CSV must exactly match its numeric slots")
    digest = preparation_digest(prepared)
    if prepared.get("preparation_digest", digest) != digest:
        raise ProfileValidationError("Server preparation digest is inconsistent")
    return digest


def save_composition(conn, *, name, entries, target_contract_id, expected_preparation_digest,
                     preparation_resolver, parent_version_id=None) -> dict:
    """Re-prepare on the server; never accept a client's claimed CSV/snapshot.

    The resolver runs outside the short SQLite write transaction, then immutable
    profile references and exact results are validated inside it. A changed
    preparation returns a conflict rather than saving unseen values silently.
    """
    name = _text(name, "Composition name")
    entries = validate_entries(entries)
    target_contract_id = _text(target_contract_id, "target_contract_id")
    if not isinstance(expected_preparation_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_preparation_digest):
        raise ProfileValidationError("A current server preparation digest is required before saving")
    parent = get_composition(conn, parent_version_id) if parent_version_id is not None else None
    # Resolve the explicit version IDs before doing source inspection work.
    for entry in entries:
        get_version(conn, entry["stable_id"], entry["profile_version_id"])
    prepared = preparation_resolver(conn, entries, target_contract_id)
    with _transaction(conn):
        digest = _validate_prepared(conn, entries, target_contract_id, prepared)
        if digest != expected_preparation_digest:
            raise PreparationChangedError("Preparation changed since it was displayed. Prepare and review the composition again before saving.")
        version_id = str(uuid4())
        conn.execute("""INSERT INTO lora_composition_versions
            (version_id,composition_id,parent_version_id,name,target_contract_id,entries_json,snapshot_json,preparation_digest,created_at)
            VALUES(?,?,?,?,?,?,?,?,?)""",
            (version_id, parent["composition_id"] if parent else version_id,
             parent["version_id"] if parent else None, name, target_contract_id, _json(entries),
             _json(prepared), digest, datetime.now(timezone.utc).isoformat()))
        return get_composition(conn, version_id)


def prepare_composition(conn, version_id, *, preparation_resolver) -> dict:
    """Return only a fresh preparation; saved ready flags are never reused."""
    recipe = get_composition(conn, version_id)
    prepared = preparation_resolver(conn, recipe["entries"], recipe["target_contract_id"])
    digest = _validate_prepared(conn, recipe["entries"], recipe["target_contract_id"], prepared)
    return {**prepared, "preparation_digest": digest}
