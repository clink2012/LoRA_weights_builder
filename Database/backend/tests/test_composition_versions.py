from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import sqlite3

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from composition_versions import (
    PreparationChangedError, get_composition, initialise_composition_schema,
    list_compositions, preparation_digest, prepare_composition, save_composition,
)
from composition_version_router import create_composition_version_router
from profile_versions import ProfileValidationError, capture_default, get_version, initialise_schema


TARGET = "flux1-dev-native-v1"


def add_default(conn, sid, role):
    return capture_default(conn, stable_id=sid,
        binding={"architecture": "test-architecture", "source_identity": {"basis": "file_sha256", "sha256": "a" * 64},
                 "engine_version": "test-1", "policy_version": "unvalidated-test-policy",
                 "target_contract": {"id": TARGET, "sha256": "b" * 64},
                 "loader_adapter": {"id": "test-adapter", "source_sha256": {"loader.py": "c" * 64}},
                 "slots": [{"group": "base", "label": "BASE"}, {"group": "double", "label": "DOUBLE 0"}]},
        values=[1.0, -0.1234567890123456],
        settings={"role": role, "strength_model": 0.8765432109876543, "strength_clip": None, "affect_clip": False})


@pytest.fixture
def state(tmp_path):
    path = tmp_path / "composition.sqlite"
    conn = sqlite3.connect(path)
    initialise_schema(conn)
    initialise_composition_schema(conn)
    roots = [add_default(conn, "A", "person"), add_default(conn, "B", "clothing")]
    entries = [{"stable_id": root["stable_id"], "profile_version_id": root["version_id"]} for root in roots]
    yield conn, path, entries
    conn.close()


def resolve(conn, entries, target):
    """Synthetic server preparation; these tests cover persistence, not loaders."""
    nodes = []
    for entry in entries:
        version = get_version(conn, entry["stable_id"], entry["profile_version_id"])
        labels = [slot["label"] for slot in version["binding"]["slots"]]
        nodes.append({**entry, **version["settings"], "loader_export": {
            "target_contract_id": target, "status": "ready", "numeric_csv": "1.0,-0.1234567890123456",
            "architecture_slot_labels": labels, "architecture_slot_values": version["values"],
            "architecture_slot_count": 2, "slot_labels": labels, "slot_values": version["values"], "loader_slot_count": 2,
            "checkpoint_verified": False, "image_quality_verified": False,
        }})
    result = {"target_contract_id": target, "compatible": True,
              "requested_loras": [entry["stable_id"] for entry in entries],
              "included_loras": [entry["stable_id"] for entry in entries],
              "excluded_loras": [], "node_payloads": nodes, "warnings": ["Synthetic unvalidated fixture"]}
    result["preparation_digest"] = preparation_digest(result)
    return result


def save(conn, initial_entries, **overrides):
    args = {"name": "Portrait and coat", "entries": initial_entries, "target_contract_id": TARGET,
            "expected_preparation_digest": preparation_digest(resolve(conn, initial_entries, TARGET)),
            "preparation_resolver": resolve}
    args.update(overrides)
    return save_composition(conn, **args)


def test_ordered_exact_snapshot_round_trip_and_immutable_history(state):
    conn, path, entries = state
    expected = resolve(conn, entries, TARGET)
    root = save(conn, entries)
    assert root["entries"] == entries
    assert root["historical_snapshot"] == expected
    assert root["requires_revalidation"] is True
    assert "node_payloads" not in root
    child = save(conn, list(reversed(entries)), name="Reverse loader order", parent_version_id=root["version_id"])
    assert child["composition_id"] == root["composition_id"]
    assert child["parent_version_id"] == root["version_id"]
    reopened = sqlite3.connect(path)
    try:
        assert get_composition(reopened, root["version_id"]) == root
        assert get_composition(reopened, child["version_id"])["entries"] == list(reversed(entries))
        assert [row["version_id"] for row in list_compositions(reopened)] == [child["version_id"], root["version_id"]]
        for sql in ("UPDATE lora_composition_versions SET name='overwrite'", "DELETE FROM lora_composition_versions"):
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                reopened.execute(sql)
            reopened.rollback()
    finally:
        reopened.close()


def test_changed_fresh_preparation_is_conflict_and_never_saved(state):
    conn, _, entries = state
    calls = []

    def changed(conn, entries, target):
        calls.append(entries)
        result = resolve(conn, entries, target)
        result["warnings"] = ["Source evidence changed since the visible preparation"]
        result["preparation_digest"] = preparation_digest(result)
        return result

    with pytest.raises(PreparationChangedError, match="changed since"):
        save(conn, entries, preparation_resolver=changed)
    assert calls == [entries]
    assert list_compositions(conn) == []


@pytest.mark.parametrize("mode", ["order", "wrong_version", "blocked", "excluded", "values", "settings", "csv", "target", "digest"])
def test_incomplete_or_inconsistent_server_results_cannot_be_saved(state, mode):
    conn, _, entries = state

    def altered(conn, entries, target):
        data = resolve(conn, entries, target)
        first = data["node_payloads"][0]
        if mode == "order":
            data["node_payloads"].reverse()
        elif mode == "wrong_version":
            first["profile_version_id"] = entries[1]["profile_version_id"]
        elif mode == "blocked":
            first["loader_export"]["status"] = "blocked"
        elif mode == "excluded":
            data["excluded_loras"] = [{"stable_id": "A"}]
        elif mode == "values":
            first["loader_export"]["architecture_slot_values"] = [1, 0]
        elif mode == "settings":
            first["strength_model"] = 9
        elif mode == "csv":
            first["loader_export"]["numeric_csv"] = "1.0,-0.12"
        elif mode == "target":
            first["loader_export"]["target_contract_id"] = "different-target"
        data["preparation_digest"] = "a" * 64 if mode == "digest" else preparation_digest(data)
        return data

    prepared = altered(conn, entries, TARGET)
    with pytest.raises(ProfileValidationError):
        save(conn, entries, preparation_resolver=altered, expected_preparation_digest=preparation_digest(prepared))
    assert list_compositions(conn) == []


def test_reading_old_recipe_cannot_bypass_fresh_restore_validation(state):
    conn, _, entries = state
    recipe = save(conn, entries)
    assert recipe["historical_snapshot"]["node_payloads"][0]["loader_export"]["status"] == "ready"
    calls = []

    def no_longer_ready(conn, entries, target):
        calls.append(entries)
        data = resolve(conn, entries, target)
        data["compatible"] = False
        return data

    with pytest.raises(ProfileValidationError, match="freshly prepared"):
        prepare_composition(conn, recipe["version_id"], preparation_resolver=no_longer_ready)
    assert calls == [entries]
    assert get_composition(conn, recipe["version_id"]) == recipe
    fresh = prepare_composition(conn, recipe["version_id"], preparation_resolver=resolve)
    assert fresh["preparation_digest"] == preparation_digest(fresh)


def test_entries_require_explicit_unique_profile_versions(state):
    conn, _, entries = state
    for invalid in ([], entries * 2, [{"stable_id": "A"}], [{"stable_id": "A", "profile_version_id": ""}],
                    [{"stable_id": "A", "profile_version_id": entries[0]["profile_version_id"], "numeric_csv": "1,2"}]):
        with pytest.raises(ProfileValidationError):
            save(conn, entries, entries=invalid)


def test_simultaneous_children_preserve_both_versions_without_overwriting_parent(state):
    conn, path, entries = state
    parent = save(conn, entries)

    def create(number):
        connection = sqlite3.connect(path)
        try:
            return save(connection, entries, name=f"Child {number}", parent_version_id=parent["version_id"])
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        children = list(executor.map(create, range(2)))
    assert len({child["version_id"] for child in children}) == 2
    assert all(child["composition_id"] == parent["composition_id"] for child in children)
    assert len(list_compositions(conn, parent["composition_id"])) == 3
    assert get_composition(conn, parent["version_id"]) == parent


def test_router_rejects_client_snapshots_and_returns_409_for_changed_result(state):
    conn, path, entries = state
    latest = resolve(conn, entries, TARGET)
    resolver_result = deepcopy(latest)

    def resolver(conn, entries, target):
        return deepcopy(resolver_result)

    app = FastAPI()
    app.include_router(create_composition_version_router(lambda: sqlite3.connect(path), resolver))
    body = {"name": "Saved through API", "entries": entries, "target_contract_id": TARGET,
            "expected_preparation_digest": latest["preparation_digest"]}
    with TestClient(app) as client:
        base = "/api/composition-versions"
        assert client.post(base, json={**body, "snapshot": latest}).status_code == 422
        response = client.post(base, json=body)
        assert response.status_code == 200
        recipe = response.json()
        assert client.get(base).json()["versions"][0]["version_id"] == recipe["version_id"]
        assert client.get(base + "/" + recipe["version_id"]).json()["requires_revalidation"] is True
        assert client.post(base + "/" + recipe["version_id"] + "/prepare", json={}).json() == latest
        resolver_result["warnings"] = ["changed"]
        resolver_result["preparation_digest"] = preparation_digest(resolver_result)
        assert client.post(base, json=body).status_code == 409
        assert len(client.get(base).json()["versions"]) == 1
