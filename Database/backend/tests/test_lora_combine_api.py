from pathlib import Path
import json
import sqlite3
import sys

from fastapi.testclient import TestClient
import pytest
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))

sys.modules.setdefault("delta_inspector_engine", types.SimpleNamespace(inspect_lora=lambda *args, **kwargs: None))
import lora_api_server  # noqa: E402


def _init_test_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE lora (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stable_id TEXT UNIQUE,
            filename TEXT,
            base_model_code TEXT,
            block_layout TEXT,
            has_block_weights INTEGER
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE lora_block_weights (
            stable_id TEXT,
            block_index INTEGER,
            weight REAL
        );
        """
    )
    conn.commit()
    conn.close()


def _insert_lora(
    conn: sqlite3.Connection,
    *,
    stable_id: str,
    filename: str,
    base_model_code: str,
    block_layout: str,
    has_block_weights: int,
) -> None:
    conn.execute(
        """
        INSERT INTO lora (stable_id, filename, base_model_code, block_layout, has_block_weights)
        VALUES (?, ?, ?, ?, ?);
        """,
        (stable_id, filename, base_model_code, block_layout, has_block_weights),
    )


def _insert_weights(conn: sqlite3.Connection, stable_id: str, weights: list[float]) -> None:
    conn.executemany(
        """
        INSERT INTO lora_block_weights (stable_id, block_index, weight)
        VALUES (?, ?, ?);
        """,
        [(stable_id, idx, weight) for idx, weight in enumerate(weights)],
    )


def _csv_to_floats(csv_weights: str) -> list[float]:
    return [float(v) for v in csv_weights.split(",")]


def _make_combine_body_for_tests(client: TestClient) -> dict:
    combine_response = client.post(
        "/api/lora/combine",
        json={
            "stable_ids": ["FLX-REAL-001", "FLX-REAL-002"],
            "per_lora": {
                "FLX-REAL-001": {"strength_model": 0.7, "strength_clip": 0.0, "affect_clip": False},
                "FLX-REAL-002": {"strength_model": 1.3, "strength_clip": 0.0, "affect_clip": False},
            },
        },
    )
    assert combine_response.status_code == 200
    return combine_response.json()


@pytest.fixture
def client_with_temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "combine_test.sqlite"
    _init_test_db(db_path)
    monkeypatch.setattr(lora_api_server, "DB_PATH", db_path)
    monkeypatch.setattr(lora_api_server, "_schema_migrations_done", False)
    with TestClient(lora_api_server.app) as client:
        yield client, db_path


def test_combine_only_fallback_loras_returns_400_with_policy_reason(client_with_temp_db):
    client, db_path = client_with_temp_db
    conn = sqlite3.connect(db_path)
    _insert_lora(
        conn,
        stable_id="FLX-FALL-001",
        filename="fallback_one.safetensors",
        base_model_code="FLX",
        block_layout="flux_transformer_3",
        has_block_weights=0,
    )
    conn.commit()
    conn.close()

    response = client.post("/api/lora/combine", json={"stable_ids": ["FLX-FALL-001"], "per_lora": {}})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["included_loras"] == []
    assert detail["excluded_loras"][0]["reason_code"] == "fallback_excluded"
    assert detail["reasons"][0]["code"] == "all_loras_excluded"
    assert any("fallback LoRAs are not allowed in /api/lora/combine" in warning for warning in detail["warnings"])


def test_combine_mixed_real_and_fallback_returns_200_and_excludes_fallback(client_with_temp_db):
    client, db_path = client_with_temp_db
    conn = sqlite3.connect(db_path)
    _insert_lora(
        conn,
        stable_id="FLX-REAL-001",
        filename="real.safetensors",
        base_model_code="FLX",
        block_layout="flux_transformer_3",
        has_block_weights=1,
    )
    _insert_weights(conn, "FLX-REAL-001", [0.2, 0.4, 0.6])
    _insert_lora(
        conn,
        stable_id="FLX-FALL-001",
        filename="fallback_one.safetensors",
        base_model_code="FLX",
        block_layout="flux_transformer_3",
        has_block_weights=0,
    )
    conn.commit()
    conn.close()

    response = client.post(
        "/api/lora/combine",
        json={"stable_ids": ["FLX-REAL-001", "FLX-FALL-001"], "per_lora": {}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["included_loras"] == ["FLX-REAL-001"]
    assert body["excluded_loras"][0]["reason_code"] == "fallback_excluded"
    assert any("Excluded 1 fallback LoRA(s)" in warning for warning in body["warnings"])


def test_combine_base_model_mismatch_uses_structured_reason_objects(client_with_temp_db):
    client, db_path = client_with_temp_db
    conn = sqlite3.connect(db_path)
    _insert_lora(
        conn,
        stable_id="FLX-REAL-001",
        filename="real_a.safetensors",
        base_model_code="FLX",
        block_layout="flux_transformer_3",
        has_block_weights=1,
    )
    _insert_weights(conn, "FLX-REAL-001", [0.2, 0.4, 0.6])
    _insert_lora(
        conn,
        stable_id="SDX-REAL-002",
        filename="real_b.safetensors",
        base_model_code="SDX",
        block_layout="flux_transformer_3",
        has_block_weights=1,
    )
    _insert_weights(conn, "SDX-REAL-002", [0.6, 0.8, 1.0])
    conn.commit()
    conn.close()

    response = client.post(
        "/api/lora/combine",
        json={"stable_ids": ["FLX-REAL-001", "SDX-REAL-002"], "per_lora": {}},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["compatible"] is False
    assert any(reason["code"] == "base_model_mismatch" for reason in detail["reasons"])
    mismatch_reason = next(reason for reason in detail["reasons"] if reason["code"] == "base_model_mismatch")
    assert mismatch_reason["stable_ids"] == ["FLX-REAL-001", "SDX-REAL-002"]


def test_combine_response_includes_aliases_and_csv_consistency_for_model_and_clip(client_with_temp_db):
    client, db_path = client_with_temp_db
    conn = sqlite3.connect(db_path)
    _insert_lora(
        conn,
        stable_id="FLX-REAL-001",
        filename="real_a.safetensors",
        base_model_code="FLX",
        block_layout="flux_transformer_3",
        has_block_weights=1,
    )
    _insert_weights(conn, "FLX-REAL-001", [0.2, 0.4, 0.6])
    _insert_lora(
        conn,
        stable_id="FLX-REAL-002",
        filename="real_b.safetensors",
        base_model_code="FLX",
        block_layout="flux_transformer_3",
        has_block_weights=1,
    )
    _insert_weights(conn, "FLX-REAL-002", [0.6, 0.8, 1.0])
    conn.commit()
    conn.close()

    response = client.post(
        "/api/lora/combine",
        json={
            "stable_ids": ["FLX-REAL-001", "FLX-REAL-002"],
            "per_lora": {
                "FLX-REAL-001": {"strength_clip": 1.0, "affect_clip": True},
                "FLX-REAL-002": {"strength_clip": 3.0, "affect_clip": True},
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    combined = body["combined"]

    assert body["response_schema_version"] == "7.1"
    assert isinstance(body["node_payloads"], list)
    assert len(body["node_payloads"]) == len(body["included_loras"])
    for payload in body["node_payloads"]:
        assert {
            "stable_id",
            "filename",
            "base_model_code",
            "block_layout",
            "strength_model",
            "strength_clip",
            "A",
            "B",
            "block_weights",
            "block_weights_csv",
        } <= set(payload.keys())

    # Per-LoRA contract: each node payload contains THAT LoRA's own block weights.
    expected_by_id = {
        "FLX-REAL-001": [0.2, 0.4, 0.6],
        "FLX-REAL-002": [0.6, 0.8, 1.0],
    }
    for payload in body["node_payloads"]:
        sid = payload["stable_id"]
        assert sid in expected_by_id
        assert payload["block_weights"] == expected_by_id[sid]
        assert _csv_to_floats(payload["block_weights_csv"]) == expected_by_id[sid]

    assert body["excluded_loras"] == []
    assert body["reasons"] == []
    assert isinstance(body["warnings"], list)

    assert combined["combined_strength_model"] == combined["strength_model"]
    assert combined["combined_strength_clip"] == combined["strength_clip"]
    assert combined["combined_A"] == combined["A"]
    assert combined["combined_B"] == combined["B"]

    assert combined["block_weights"] == combined["block_weights_model"]
    assert combined["block_weights_csv"] == combined["block_weights_model_csv"]
    assert _csv_to_floats(combined["block_weights_model_csv"]) == combined["block_weights_model"]
    assert _csv_to_floats(combined["block_weights_clip_csv"]) == combined["block_weights_clip"]


def test_combine_response_clip_keys_present_and_null_without_clip_contributors(client_with_temp_db):
    client, db_path = client_with_temp_db
    conn = sqlite3.connect(db_path)
    _insert_lora(
        conn,
        stable_id="FLX-REAL-001",
        filename="real.safetensors",
        base_model_code="FLX",
        block_layout="flux_transformer_3",
        has_block_weights=1,
    )
    _insert_weights(conn, "FLX-REAL-001", [0.2, 0.4, 0.6])
    conn.commit()
    conn.close()

    response = client.post(
        "/api/lora/combine",
        json={"stable_ids": ["FLX-REAL-001"], "per_lora": {}},
    )

    assert response.status_code == 200
    body = response.json()
    combined = body["combined"]

    assert body["response_schema_version"] == "7.1"
    assert isinstance(body["node_payloads"], list)
    assert len(body["node_payloads"]) == len(body["included_loras"])

    # Clip is omitted (null) when there are no clip contributors
    assert all(node["strength_clip"] is None for node in body["node_payloads"])

    # Per-LoRA contract: node payload block weights are per-LoRA (not shared combined).
    assert body["node_payloads"][0]["block_weights"] == [0.2, 0.4, 0.6]
    assert _csv_to_floats(body["node_payloads"][0]["block_weights_csv"]) == [0.2, 0.4, 0.6]

    assert combined["block_weights_model_csv"] is not None
    assert combined["block_weights_clip"] is None
    assert combined["block_weights_clip_csv"] is None
    assert body["excluded_loras"] == []
    assert body["reasons"] == []
    assert isinstance(body["warnings"], list)
    assert any("No clip contributors" in warning for warning in body["warnings"])


def test_save_combined_profile_persists_verbatim_combined_payload_with_canonical_csvs(client_with_temp_db):
    client, db_path = client_with_temp_db
    conn = sqlite3.connect(db_path)
    _insert_lora(
        conn,
        stable_id="FLX-REAL-001",
        filename="real_a.safetensors",
        base_model_code="FLX",
        block_layout="flux_transformer_3",
        has_block_weights=1,
    )
    _insert_weights(conn, "FLX-REAL-001", [0.2, 0.4, 0.6])
    _insert_lora(
        conn,
        stable_id="FLX-REAL-002",
        filename="real_b.safetensors",
        base_model_code="FLX",
        block_layout="flux_transformer_3",
        has_block_weights=1,
    )
    _insert_weights(conn, "FLX-REAL-002", [0.6, 0.8, 1.0])
    conn.commit()
    conn.close()

    combine_input = {
        "stable_ids": ["FLX-REAL-001", "FLX-REAL-002"],
        "per_lora": {
            "FLX-REAL-001": {"strength_model": 0.5, "strength_clip": 0.0, "affect_clip": False},
            "FLX-REAL-002": {"strength_model": 1.5, "strength_clip": 0.0, "affect_clip": False},
        },
    }
    combine_response = client.post("/api/lora/combine", json=combine_input)
    assert combine_response.status_code == 200
    combine_body = combine_response.json()

    save_response = client.post(
        "/api/lora/combined-profile",
        json={
            "profile_name": "My Combined Test",
            "recipe": {
                "stable_ids": combine_input["stable_ids"],
                "per_lora": combine_input["per_lora"],
            },
            "combine_response": combine_body,
        },
    )

    assert save_response.status_code == 201
    saved = save_response.json()
    assert isinstance(saved["id"], int)
    assert saved["profile_name"] == "My Combined Test"
    assert saved["response_schema_version"] == combine_body["response_schema_version"]
    assert saved["validated_base_model"] == combine_body["validated_base_model"]
    assert saved["validated_layout"] == combine_body["validated_layout"]

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT combined_payload_json, response_schema_version
        FROM lora_combined_profiles
        WHERE id = ?
        """,
        (saved["id"],),
    )
    row = cur.fetchone()
    conn.close()

    assert row is not None
    combined_payload = json.loads(row[0])
    assert isinstance(combined_payload, dict)

    # Phase 6.2 contract: persist the combine response VERBATIM (no mutation, no recompute)
    assert combined_payload == combine_body

    # CSV determinism is guaranteed by /api/lora/combine output (and should remain consistent with list values)
    combined = combined_payload["combined"]
    assert combined["block_weights_model_csv"] == lora_api_server.weights_to_csv(combined["block_weights_model"])

    clip_weights = combined["block_weights_clip"]
    if clip_weights is None:
        assert combined["block_weights_clip_csv"] is None
    else:
        assert combined["block_weights_clip_csv"] == lora_api_server.weights_to_csv(clip_weights)

    assert combined["block_weights_csv"] == combined["block_weights_model_csv"]
    assert row[1] == combine_body["response_schema_version"]


def test_combined_profile_list_and_load_endpoints(client_with_temp_db):
    client, db_path = client_with_temp_db
    conn = sqlite3.connect(db_path)
    _insert_lora(
        conn,
        stable_id="FLX-REAL-001",
        filename="real_a.safetensors",
        base_model_code="FLX",
        block_layout="flux_transformer_3",
        has_block_weights=1,
    )
    _insert_weights(conn, "FLX-REAL-001", [0.2, 0.4, 0.6])
    _insert_lora(
        conn,
        stable_id="FLX-REAL-002",
        filename="real_b.safetensors",
        base_model_code="FLX",
        block_layout="flux_transformer_3",
        has_block_weights=1,
    )
    _insert_weights(conn, "FLX-REAL-002", [0.6, 0.8, 1.0])
    conn.commit()
    conn.close()

    combine_body = _make_combine_body_for_tests(client)

    save_alpha = client.post(
        "/api/lora/combined-profile",
        json={
            "profile_name": "Alpha",
            "recipe": {"stable_ids": ["FLX-REAL-001", "FLX-REAL-002"], "per_lora": {}},
            "combine_response": combine_body,
        },
    )
    assert save_alpha.status_code == 201
    alpha_id = save_alpha.json()["id"]

    save_latest = client.post(
        "/api/lora/combined-profile",
        json={
            "profile_name": "Shared Name",
            "recipe": {"stable_ids": ["FLX-REAL-001"], "per_lora": {}},
            "combine_response": combine_body,
        },
    )
    assert save_latest.status_code == 201
    latest_id = save_latest.json()["id"]

    save_older = client.post(
        "/api/lora/combined-profile",
        json={
            "profile_name": "Shared Name",
            "recipe": {"stable_ids": ["FLX-REAL-002"], "per_lora": {}},
            "combine_response": combine_body,
        },
    )
    assert save_older.status_code == 201
    older_id = save_older.json()["id"]

    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE lora_combined_profiles SET created_at = ?, updated_at = ? WHERE id = ?;",
        ("2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", alpha_id),
    )
    conn.execute(
        "UPDATE lora_combined_profiles SET created_at = ?, updated_at = ? WHERE id = ?;",
        ("2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z", latest_id),
    )
    conn.execute(
        "UPDATE lora_combined_profiles SET created_at = ?, updated_at = ? WHERE id = ?;",
        ("2024-01-04T00:00:00Z", "2024-01-02T00:00:00Z", older_id),
    )
    conn.commit()
    conn.close()

    list_response = client.get("/api/lora/combined-profiles")
    assert list_response.status_code == 200
    profiles = list_response.json()["profiles"]
    assert [p["id"] for p in profiles] == [latest_id, older_id, alpha_id]

    first = profiles[0]
    assert set(first.keys()) == {
        "id",
        "profile_name",
        "validated_base_model",
        "validated_layout",
        "response_schema_version",
        "created_at",
        "updated_at",
    }

    by_id_response = client.get(f"/api/lora/combined-profile/{alpha_id}")
    assert by_id_response.status_code == 200
    by_id = by_id_response.json()
    assert by_id["id"] == alpha_id
    assert by_id["combine_response"] == combine_body
    assert by_id["recipe"] == {"stable_ids": ["FLX-REAL-001", "FLX-REAL-002"], "per_lora": {}}

    by_name_response = client.get("/api/lora/combined-profile/by-name/Shared Name")
    assert by_name_response.status_code == 200
    by_name = by_name_response.json()
    assert by_name["id"] == latest_id
    assert by_name["profile_name"] == "Shared Name"

    missing_id_response = client.get("/api/lora/combined-profile/999999")
    assert missing_id_response.status_code == 404
    assert "not found" in missing_id_response.json()["detail"].lower()

    missing_name_response = client.get("/api/lora/combined-profile/by-name/does-not-exist")
    assert missing_name_response.status_code == 404
    assert "not found" in missing_name_response.json()["detail"].lower()
