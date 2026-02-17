from pathlib import Path
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

    assert combined["block_weights_model_csv"] is not None
    assert combined["block_weights_clip"] is None
    assert combined["block_weights_clip_csv"] is None
    assert body["excluded_loras"] == []
    assert body["reasons"] == []
    assert isinstance(body["warnings"], list)
