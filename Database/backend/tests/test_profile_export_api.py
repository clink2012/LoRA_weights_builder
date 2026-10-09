"""HTTP integration: immutable profiles are freshly bound before numeric export."""
import json
import os
from pathlib import Path
import sqlite3
import struct
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flux_header_coverage as coverage
import lora_api_server as api
from profile_versions import initialise_schema
from composition_versions import initialise_composition_schema
from profile_versions import _insert


@pytest.fixture
def profile_client(tmp_path, monkeypatch):
    file = tmp_path / "native.safetensors"
    prefix = "lora_unet_double_blocks_0_img_attn_proj"
    header = {
        prefix + ".lora_up.weight": {"dtype": "F16", "shape": [3072, 4], "data_offsets": [0, 24576]},
        prefix + ".lora_down.weight": {"dtype": "F16", "shape": [4, 3072], "data_offsets": [24576, 49152]},
    }
    raw = json.dumps(header).encode()
    with file.open("wb") as stream:
        stream.write(struct.pack("<Q", len(raw)) + raw)
        stream.truncate(8 + len(raw) + 49152)
    db = tmp_path / "profiles.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE lora(stable_id TEXT, filename TEXT, file_path TEXT, base_model_code TEXT)")
    conn.executemany("INSERT INTO lora VALUES(?,?,?,'FLX')", [(sid, file.name, str(file)) for sid in ("first", "second")])
    conn.commit()
    initialise_schema(conn)
    initialise_composition_schema(conn)
    conn.close()
    monkeypatch.setattr(api, "DB_PATH", db)
    monkeypatch.setattr(coverage, "verify_local_sources", lambda: {"fixture.py": "0" * 64})
    return TestClient(api.app), file, db


def capture(client, sid="first"):
    response = client.post(f"/api/profile-versions/{sid}/defaults", json={})
    assert response.status_code == 200, response.text
    return response.json()


def test_preferred_recipe_recall_checks_actual_current_header_and_stat(profile_client):
    from composition_preferences import initialise_preference_schema
    client, file, database = profile_client
    conn = sqlite3.connect(database)
    initialise_preference_schema(conn)
    conn.close()
    root = capture(client)
    prepared = prepare(client, 'first', root['version_id']).json()
    recipe = client.post('/api/composition-versions', json={
        'name': 'Recall actual native file', 'target_contract_id': coverage.CONTRACT_ID,
        'entries': [{'stable_id': 'first', 'profile_version_id': root['version_id']}],
        'expected_preparation_digest': prepared['preparation_digest'],
    }).json()
    preference = client.post('/api/composition-preferences/choose', json={
        'version_id': recipe['version_id'], 'expected_preparation_digest': prepared['preparation_digest'],
    })
    assert preference.status_code == 200, preference.text
    body = {'stable_ids': ['first'], 'target_contract_id': coverage.CONTRACT_ID}
    recall = client.post('/api/composition-preferences/resolve', json=body)
    assert recall.status_code == 200, recall.text
    assert recall.json()['status'] == 'preferred'
    assert 'node_payloads' not in recall.json()
    stat = file.stat()
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10000000))
    changed = client.post('/api/composition-preferences/resolve', json=body)
    assert changed.status_code == 200, changed.text
    assert changed.json()['status'] == 'needs_review'
    assert changed.json()['recipe'] is None
    assert client.get('/api/composition-versions/'+recipe['version_id']).json()['entries'] == recipe['entries']


def prepare(client, sid, version):
    return client.post("/api/lora/prepare-blocks", json={
        "stable_ids": [sid], "target_contract_id": coverage.CONTRACT_ID,
        "profile_version_ids": {sid: version},
    })


def test_preparation_inside_transaction_sees_unsaved_children_without_committing(profile_client):
    client, _, database = profile_client
    root = capture(client)
    values = list(root['values'])
    values[1] = .42
    conn = sqlite3.connect(database)
    try:
        conn.execute('BEGIN IMMEDIATE')
        child = _insert(conn, stable_id='first', default_id=root['version_id'], parent_id=root['version_id'],
                        kind='personal', name='Atomic child', binding=root['binding'],
                        snapshot={'values': values, 'settings': root['settings'], 'ab': {}},
                        provenance={'method': 'gentle_balance_experiment', 'experiment_id': 'test-experiment'})
        prepared = api.resolve_composition_preparation(conn, [{'stable_id': 'first', 'profile_version_id': child['version_id']}], coverage.CONTRACT_ID)
        assert prepared['compatible'] is True
        assert prepared['node_payloads'][0]['loader_export']['architecture_slot_values'] == values
        assert prepared['node_payloads'][0]['loader_export']['recommendation_basis'] == 'experimental_parameter_policy_unvalidated'
        assert prepared['node_payloads'][0]['profile_provenance']['experiment_id'] == 'test-experiment'
        assert conn.in_transaction
        assert prepare(client, 'first', child['version_id']).json()['compatible'] is False
        conn.rollback()
        assert prepare(client, 'first', child['version_id']).json()['compatible'] is False
    finally:
        conn.close()


def test_fresh_default_then_personal_revision_exports_exact_values_and_settings(profile_client):
    client, _, _ = profile_client
    root = capture(client)
    assert len(root["values"]) == 58
    assert root["values"][:3] == [1, 1, 0]
    assert root["binding"]["source_identity"]["basis"] == "header_stat"
    assert root["binding"]["target_contract"]["id"] == coverage.CONTRACT_ID
    values = list(root["values"])
    values[0], values[1] = 0.7, 0.345678912345
    settings = dict(root["settings"], role="clothing", strength_model=0.8754)
    revision = client.post("/api/profile-versions/first/revisions", json={
        "default_id": root["version_id"], "parent_id": root["version_id"], "name": "Gentle clothing",
        "values": values, "settings": settings,
        "ab": {"A": {"slot_labels": ["DOUBLE 0"], "min": 0.1, "max": 0.7,
                     "value": values[1], "basis": "Personal experiment; no quality validation"}},
    })
    assert revision.status_code == 200, revision.text
    version = revision.json()
    prepared = prepare(client, "first", version["version_id"]).json()
    assert prepared["compatible"] is True
    node = prepared["node_payloads"][0]
    assert node["profile_version_id"] == version["version_id"]
    assert node["profile_name"] == "Gentle clothing"
    assert node["role"] == "clothing"
    assert node["strength_model"] == 0.8754
    assert node["loader_export"]["architecture_slot_values"] == values
    assert [float(x) for x in node["loader_export"]["numeric_csv"].split(",")][:2] == values[:2]
    assert node["loader_export"]["recommendation_basis"] == "manual_variant_unvalidated"
    assert node["ab"]["A"]["value"] == values[1]
    assert node["A"] == values[1]
    assert node["B"] == 1.0
    # Returning to Default restores the original vector without mutating history.
    restored = prepare(client, "first", root["version_id"]).json()["node_payloads"][0]
    assert restored["loader_export"]["architecture_slot_values"] == root["values"]
    history = client.get("/api/profile-versions/first").json()["versions"]
    assert len(history) == 2
    assert capture(client)["version_id"] == root["version_id"]


def test_selected_profile_cannot_cross_lora_or_be_silently_replaced(profile_client):
    client, _, _ = profile_client
    root = capture(client)
    result = prepare(client, "second", root["version_id"]).json()
    assert result["compatible"] is False
    assert result["node_payloads"] == []
    assert result["excluded_loras"][0]["reason_code"] == "profile_unavailable"
    blank = prepare(client, "first", " ").json()
    assert blank["excluded_loras"][0]["reason_code"] == "profile_unavailable"
    unexpected = client.post("/api/lora/prepare-blocks", json={
        "stable_ids": ["first"], "target_contract_id": coverage.CONTRACT_ID,
        "profile_version_ids": {"second": root["version_id"]},
    })
    assert unexpected.status_code == 400


def test_changed_source_requires_new_default_and_retains_old_history(profile_client):
    client, file, _ = profile_client
    root = capture(client)
    stat = file.stat()
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10000000))
    stale = prepare(client, "first", root["version_id"]).json()
    assert stale["compatible"] is False
    assert stale["excluded_loras"][0]["reason_code"] == "profile_binding_changed"
    fresh = capture(client)
    assert fresh["version_id"] != root["version_id"]
    assert prepare(client, "first", fresh["version_id"]).json()["compatible"] is True
    assert len(client.get("/api/profile-versions/first").json()["versions"]) == 2


def test_default_capture_rejects_browser_fingerprints_and_wrong_values(profile_client):
    client, _, _ = profile_client
    assert client.post("/api/profile-versions/first/defaults", json={"values": [1]}).status_code == 422
    root = capture(client)
    failed = client.post("/api/profile-versions/first/revisions", json={
        "default_id": root["version_id"], "parent_id": root["version_id"], "name": "Missing BASE",
        "values": root["values"][1:], "settings": root["settings"],
    })
    assert failed.status_code == 422
    assert len(client.get("/api/profile-versions/first").json()["versions"]) == 1


def test_model_only_header_does_not_export_clip_enabled_revision(profile_client):
    client, _, _ = profile_client
    root = capture(client)
    revision = client.post("/api/profile-versions/first/revisions", json={
        "default_id": root["version_id"], "parent_id": root["version_id"], "name": "CLIP experiment",
        "values": root["values"], "settings": dict(root["settings"], affect_clip=True, strength_clip=0.5),
    }).json()
    prepared = prepare(client, "first", revision["version_id"]).json()
    assert prepared["excluded_loras"][0]["reason_code"] == "unsupported_clip_profile"


def test_complete_composition_roundtrip_preserves_order_and_revalidates(profile_client):
    client, file, _ = profile_client
    first, second = capture(client, "first"), capture(client, "second")
    entries = [{"stable_id": "second", "profile_version_id": second["version_id"]},
               {"stable_id": "first", "profile_version_id": first["version_id"]}]
    request = {"stable_ids": [entry["stable_id"] for entry in entries],
               "target_contract_id": coverage.CONTRACT_ID,
               "profile_version_ids": {entry["stable_id"]: entry["profile_version_id"] for entry in entries}}
    prepared = client.post("/api/lora/prepare-blocks", json=request).json()
    assert prepared["compatible"] is True
    assert client.post("/api/lora/prepare-blocks", json=request).json()["preparation_digest"] == prepared["preparation_digest"]
    saved_response = client.post("/api/composition-versions", json={
        "name": "Two LoRA comparison", "entries": entries,
        "target_contract_id": coverage.CONTRACT_ID,
        "expected_preparation_digest": prepared["preparation_digest"],
    })
    assert saved_response.status_code == 200, saved_response.text
    saved = saved_response.json()
    assert saved["entries"] == entries
    assert saved["historical_snapshot"] == prepared
    assert saved["requires_revalidation"] is True
    refreshed = client.post(f"/api/composition-versions/{saved['version_id']}/prepare", json={})
    assert refreshed.status_code == 200
    assert refreshed.json() == prepared
    # A stale source leaves the historical receipt readable but cannot regain Copy.
    stat = file.stat()
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10000000))
    stale = client.post(f"/api/composition-versions/{saved['version_id']}/prepare", json={})
    assert stale.status_code == 422
    old = client.get(f"/api/composition-versions/{saved['version_id']}").json()
    assert old["historical_snapshot"] == prepared
    assert old["requires_revalidation"] is True


def test_composition_never_saves_changed_or_client_supplied_exports(profile_client):
    client, _, _ = profile_client
    root = capture(client)
    entries = [{"stable_id": "first", "profile_version_id": root["version_id"]}]
    body = {"name": "Stale receipt", "entries": entries, "target_contract_id": coverage.CONTRACT_ID,
            "expected_preparation_digest": "0" * 64}
    assert client.post("/api/composition-versions", json=body).status_code == 409
    assert client.get("/api/composition-versions").json()["versions"] == []
    assert client.post("/api/composition-versions", json={**body, "snapshot": {"numeric_csv": "fake"}}).status_code == 422
    assert client.post("/api/composition-versions", json={**body, "entries": entries * 2}).status_code == 422
