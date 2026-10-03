from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from profile_versions import (
    ProfileNotFoundError, ProfileValidationError, capture_default, create_revision,
    get_selection, get_version, import_legacy_profiles, initialise_schema,
    list_versions, select_version,
)
from profile_version_router import create_profile_version_router


@pytest.fixture
def connection(tmp_path):
    conn = sqlite3.connect(tmp_path / "profiles.sqlite")
    initialise_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def payload():
    return {
        "stable_id": "FLX-PPL-001",
        "binding": {
            "architecture": "flux.1",
            "source_identity": {"basis": "header_stat", "header_sha256": "a" * 64, "size_bytes": 2048, "mtime_ns": 123456789},
            "engine_version": "test-engine-1",
            "policy_version": "test-heuristic-1",
            "target_contract": {"id": "flux1-dev-native-v1", "sha256": "b" * 64},
            "loader_adapter": {"id": "inspire-test-adapter", "source_sha256": {"inspire.py": "c" * 64, "comfy.py": "d" * 64}},
            "slots": [{"group": "base", "label": "BASE"}, {"group": "double", "label": "DOUBLE_0"},
                      {"group": "single", "label": "SINGLE_0"}],
        },
        "values": [1, -0.12567890123456789, 0.3333333333333333],
        "settings": {"role": "person", "strength_model": 0.987654321, "strength_clip": None, "affect_clip": False},
    }


def revision(root, **changes):
    result = {key: deepcopy(root[key]) for key in ("stable_id", "binding", "values", "settings", "ab")}
    result.update(default_id=root["default_id"], parent_id=root["version_id"], name="My revision")
    result.update(changes)
    return result


def test_default_is_explicit_idempotent_and_cannot_be_changed(connection, payload):
    assert list_versions(connection, payload["stable_id"]) == []
    root = capture_default(connection, **payload)
    assert capture_default(connection, **payload) == root
    altered = deepcopy(payload)
    altered["values"][0] = 0
    with pytest.raises(ProfileValidationError, match="immutable Default"):
        capture_default(connection, **altered)
    assert len(list_versions(connection, payload["stable_id"])) == 1
    for sql in ("UPDATE lora_profile_versions SET name='oops'", "DELETE FROM lora_profile_versions"):
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(sql)
        connection.rollback()
    assert get_version(connection, root["stable_id"], root["version_id"]) == root


def test_revision_round_trip_and_return_to_default_preserve_history(connection, payload):
    root = capture_default(connection, **payload)
    first = create_revision(connection, **revision(root, values=[1, -0.25, 0.4567890123456789]))
    second = create_revision(connection, **revision(first, values=[1, -0.75, 0.4567890123456789]))
    assert get_version(connection, root["stable_id"], first["version_id"]) == first
    assert first["values"][2] == 0.4567890123456789
    assert get_selection(connection, root["stable_id"], root["default_id"]) == root
    for target in (second, root, first):
        assert select_version(connection, stable_id=root["stable_id"], default_id=root["default_id"],
                              version_id=target["version_id"]) == target
        assert get_selection(connection, root["stable_id"], root["default_id"]) == target
    assert list_versions(connection, root["stable_id"]) == [root, first, second]
    assert connection.execute("SELECT COUNT(*) FROM lora_profile_selections").fetchone()[0] == 3
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        connection.execute("DELETE FROM lora_profile_selections")
    connection.rollback()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), True, "0.5", 10 ** 1000])
def test_nonfinite_or_non_numeric_values_are_rejected(connection, payload, bad):
    payload["values"][1] = bad
    with pytest.raises(ProfileValidationError, match="finite number"):
        capture_default(connection, **payload)
    assert list_versions(connection, payload["stable_id"]) == []


def test_wrong_shape_and_identity_binding_are_rejected(connection, payload):
    root = capture_default(connection, **payload)
    with pytest.raises(ProfileValidationError, match="value count"):
        create_revision(connection, **revision(root, values=[1]))
    for key, value in (("architecture", "flux.2"), ("source_identity", {"basis": "file_sha256", "sha256": "b" * 64}),
                       ("engine_version", "different"), ("policy_version", "different"),
                       ("target_contract", {"id": "new-contract", "sha256": "f" * 64}),
                       ("loader_adapter", {"id": "changed-loader", "source_sha256": {"inspire.py": "f" * 64}})):
        binding = deepcopy(root["binding"])
        binding[key] = value
        with pytest.raises(ProfileValidationError, match="Binding changed"):
            create_revision(connection, **revision(root, binding=binding))
    with pytest.raises(ProfileNotFoundError):
        create_revision(connection, **revision(root, stable_id="FLX-OTHER"))
    second_payload = deepcopy(payload)
    second_payload["binding"]["source_identity"]["header_sha256"] = "b" * 64
    second = capture_default(connection, **second_payload)
    with pytest.raises(ProfileValidationError, match="lineage"):
        create_revision(connection, **revision(root, parent_id=second["version_id"]))
    with pytest.raises(ProfileValidationError, match="lineage"):
        select_version(connection, stable_id=root["stable_id"], default_id=root["version_id"], version_id=second["version_id"])


def test_ordered_group_labels_and_source_digest_are_required(connection, payload):
    for binding in (
        {**payload["binding"], "source_identity": {"basis": "file_sha256", "sha256": "filename-is-not-evidence"}},
        {**payload["binding"], "source_identity": {"basis": "unknown", "sha256": "a" * 64}},
        {**payload["binding"], "source_identity": {"basis": "header_stat", "header_sha256": "a" * 64, "size_bytes": -1, "mtime_ns": 0}},
        {**payload["binding"], "slots": []},
        {**payload["binding"], "slots": [{"group": "double", "label": "duplicate"}] * 3},
        {**payload["binding"], "target_contract": {"id": "bad", "sha256": "not-a-digest"},
         "loader_adapter": {"id": "loader", "source_sha256": {"target": "a" * 64}}},
    ):
        with pytest.raises(ProfileValidationError):
            capture_default(connection, **{**payload, "binding": binding})


def test_ab_bounds_and_resolved_vector_round_trip(connection, payload):
    payload["ab"] = {"A": {"slot_labels": ["DOUBLE_0"], "min": -1.0, "max": 0.5,
                            "value": payload["values"][1], "basis": "Manual trial bounds; not image validation"}}
    root = capture_default(connection, **payload)
    assert root["ab"] == payload["ab"]
    for change in ({"min": 1}, {"max": -2}, {"value": 0}, {"min": float("nan")},
                   {"slot_labels": ["missing"]}, {"slot_labels": ["DOUBLE_0", "DOUBLE_0"]}):
        ab = deepcopy(root["ab"])
        ab["A"].update(change)
        with pytest.raises(ProfileValidationError):
            create_revision(connection, **revision(root, ab=ab))
    with pytest.raises(ProfileValidationError, match="shared"):
        create_revision(connection, **revision(root, ab={"A": root["ab"]["A"], "B": root["ab"]["A"]}))


def test_legacy_import_is_explicit_idempotent_and_preserves_original_rows(connection, payload):
    connection.execute("CREATE TABLE lora_user_profiles(id INTEGER PRIMARY KEY, stable_id TEXT, profile_name TEXT, block_weights TEXT)")
    valid = json.dumps([0.7654321098765432, -0.25, 0])
    connection.executemany("INSERT INTO lora_user_profiles VALUES (?,?,?,?)", [
        (1, payload["stable_id"], "Earlier personal edit", valid),
        (2, payload["stable_id"], "Unknown old shape", "[1,2]"),
        (3, payload["stable_id"], "Corrupt old data", "not-json"),
    ])
    connection.commit()
    before = connection.execute("SELECT * FROM lora_user_profiles").fetchall()
    root = capture_default(connection, **payload)
    with pytest.raises(ProfileValidationError, match="explicit confirmation"):
        import_legacy_profiles(connection, stable_id=root["stable_id"], default_id=root["version_id"])
    result = import_legacy_profiles(connection, stable_id=root["stable_id"], default_id=root["version_id"], confirm_mapping=True)
    assert len(result["imported"]) == 1
    assert [item["legacy_id"] for item in result["retained"]] == [2, 3]
    imported = get_version(connection, root["stable_id"], result["imported"][0])
    assert imported["values"] == json.loads(valid)
    assert imported["name"] == "Earlier personal edit"
    assert "inherited from Default" in imported["provenance"]["context"]
    repeated = import_legacy_profiles(connection, stable_id=root["stable_id"], default_id=root["version_id"], confirm_mapping=True)
    assert repeated["imported"] == []
    assert repeated["existing"] == result["imported"]
    assert connection.execute("SELECT * FROM lora_user_profiles").fetchall() == before


def test_unrecognised_legacy_table_is_not_migrated(connection, payload):
    connection.execute("CREATE TABLE lora_user_profiles(id INTEGER, mystery TEXT)")
    connection.execute("INSERT INTO lora_user_profiles VALUES(1,'keep me')")
    connection.commit()
    root = capture_default(connection, **payload)
    result = import_legacy_profiles(connection, stable_id=root["stable_id"], default_id=root["version_id"], confirm_mapping=True)
    assert "Unrecognised" in result["reason"]
    assert connection.execute("SELECT * FROM lora_user_profiles").fetchall() == [(1, "keep me")]


def test_schema_initialisation_is_additive_and_idempotent(tmp_path):
    source = sqlite3.connect(tmp_path / "old.sqlite")
    source.execute("CREATE TABLE profiles(id INTEGER PRIMARY KEY, data TEXT)")
    source.execute("INSERT INTO profiles VALUES(1,'original')")
    source.commit()
    copied = sqlite3.connect(tmp_path / "copied.sqlite")
    source.backup(copied)
    initialise_schema(copied)
    initialise_schema(copied)
    assert copied.execute("SELECT * FROM profiles").fetchall() == [(1, "original")]
    assert source.execute("SELECT name FROM sqlite_master WHERE name LIKE 'lora_profile_%'").fetchall() == []
    source.close()
    copied.close()


def test_router_uses_trusted_default_resolver_and_never_initialises_on_read(tmp_path, payload):
    path = tmp_path / "router.sqlite"
    calls = []

    def connect():
        return sqlite3.connect(path)

    def resolve(conn, stable_id):
        calls.append(stable_id)
        return {key: value for key, value in payload.items() if key != "stable_id"}

    app = FastAPI()
    app.include_router(create_profile_version_router(connect, resolve))
    with TestClient(app) as client:
        base = "/api/profile-versions/FLX-PPL-001"
        assert client.get(base).status_code == 503
        with connect() as conn:
            assert conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
            initialise_schema(conn)
        assert client.post(base + "/defaults", json={"binding": payload["binding"]}).status_code == 422
        assert calls == []
        root = client.post(base + "/defaults", json={}).json()
        assert root["values"] == payload["values"]
        assert calls == [payload["stable_id"]]
        data = revision(root, values=[1, 0.1234567890123456, 0])
        for key in ("stable_id", "binding"):
            data.pop(key)
        response = client.post(base + "/revisions", json=data)
        assert response.status_code == 200
        saved = response.json()
        assert saved["values"] == data["values"]
        assert client.get(base).json()["versions"] == [root, saved]
        assert client.get(base + "/versions/missing").status_code == 404
        assert client.post(base + "/selection", json={"default_id": root["version_id"], "version_id": saved["version_id"]}).status_code == 200
        assert client.get(base + "/selection", params={"default_id": root["version_id"]}).json() == saved
        assert client.post(base + "/selection", json={"default_id": [], "version_id": saved["version_id"]}).status_code == 422


def test_concurrent_identical_default_captures_share_one_root(tmp_path, payload):
    path = tmp_path / "concurrent.sqlite"
    conn = sqlite3.connect(path)
    initialise_schema(conn)
    conn.close()

    def capture(_):
        conn = sqlite3.connect(path)
        try:
            return capture_default(conn, **payload)["version_id"]
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=4) as executor:
        ids = list(executor.map(capture, range(8)))
    assert len(set(ids)) == 1


def test_persisted_target_and_header_stat_binding_cannot_drift_on_revision(connection, payload):
    root = capture_default(connection, **payload)
    personal = create_revision(connection, **revision(root, values=[1, -0.1234567890123456, 0]))
    path = connection.execute("PRAGMA database_list").fetchone()[2]
    reopened = sqlite3.connect(path)
    try:
        assert get_version(reopened, root["stable_id"], root["version_id"]) == root
        assert get_version(reopened, root["stable_id"], personal["version_id"]) == personal
        for identity_change in ({"header_sha256": "f" * 64}, {"mtime_ns": 999}, {"size_bytes": 4096}):
            binding = deepcopy(root["binding"])
            binding["source_identity"].update(identity_change)
            with pytest.raises(ProfileValidationError, match="Binding changed"):
                create_revision(reopened, **revision(personal, binding=binding))
        binding = deepcopy(root["binding"])
        binding["loader_adapter"]["source_sha256"]["comfy.py"] = "f" * 64
        with pytest.raises(ProfileValidationError, match="Binding changed"):
            create_revision(reopened, **revision(personal, binding=binding))
        assert list_versions(reopened, root["stable_id"]) == [root, personal]
    finally:
        reopened.close()
