from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
import threading
import time

import os
from datetime import datetime, timezone

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

from inspire_export import ADAPTER_ID, LOADER_SHA256, build_inspire_flux1_export
from flux_header_coverage import CONTRACT, CONTRACT_ID, CoverageError, inspect_native_flux_file
from profile_version_router import create_profile_version_router
from composition_version_router import create_composition_version_router
from composition_versions import initialise_composition_schema, preparation_digest
from lora_thumbnail import ThumbnailError, read_thumbnail
from profile_versions import (
    ProfileNotFoundError, ProfileValidationError, get_version,
    initialise_schema as initialise_profile_schema, validate_binding, validate_snapshot,
)
from model_family_router import router as model_family_router
from block_layouts import (
    FLUX_FALLBACK_16,
    expected_block_count_for_layout,
    fallback_block_count_for_layout,
    infer_layout_from_block_count,
    make_flux_layout,
    normalize_block_layout,
)
from lora_composer import (
    LoRAComposeInput,
    combine_weights_weighted_average,
    validate_compatibility,
    weights_to_csv,
)
from lora_energy_overlap import (
    LoRAEnergyInput,
    allocate_strengths_with_role_budget_and_overlap,
    compute_lora_energy_metrics,
)
from lora_role_policy import (
    build_role_recommendation_notes,
    build_role_strength_recommendation,
    get_role_policy,
)
from lora_block_orchestrator import (
    LoraBlockOrchestratorInput,
    orchestrate_lora_block_payloads,
)

# ----------------------------------------------------------------------
# Paths & basic config
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Main SQLite DB (same path as your indexer/inspector scripts)
DB_PATH = Path(os.environ.get("LORA_DB_PATH", str(BASE_DIR.parent / "lora_master.db")))


def inspect_lora(*args, **kwargs):
    """Ordinary catalogue/combine requests do not need the tensor runtime."""
    try:
        from delta_inspector_engine import inspect_lora as implementation
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"LoRA inspection is unavailable: install the analysis dependencies ({exc.name}) in this app's Python environment.",
        ) from exc
    return implementation(*args, **kwargs)


def index_all_loras():
    try:
        from lora_indexer import main as implementation
        import lora_indexer
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"LoRA indexing is unavailable: install the analysis dependencies ({exc.name}) in this app's Python environment.",
        ) from exc
    # Honour an isolated database override throughout an explicit index operation.
    from model_family_integration import apply_model_family_registry
    apply_model_family_registry(lora_indexer)
    lora_indexer.DB_PATH = str(DB_PATH)
    if os.environ.get("LORA_ROOT"):
        lora_indexer.LORA_ROOT = os.environ["LORA_ROOT"]
    return implementation()


def assign_stable_ids():
    import lora_id_assigner
    lora_id_assigner.DB_PATH = str(DB_PATH)
    return lora_id_assigner.main()

# Add future self-healing columns here, e.g. {"new_column": "INTEGER DEFAULT 0"}.
REQUIRED_LORA_COLUMNS = {
    "block_layout": "TEXT",
    "clip_contributor": "INTEGER NOT NULL DEFAULT 0",
    "clip_tensor_count": "INTEGER NOT NULL DEFAULT 0",
}
REQUIRED_LORA_BLOCK_WEIGHTS_COLUMNS = {
    "stable_id": "TEXT",
}


_schema_migrations_lock = threading.Lock()
_schema_migrations_done = False


def ensure_safe_schema_migrations(conn: sqlite3.Connection) -> None:
    """
    Safe, idempotent migration for schema drift between indexer and API versions.

    Some older DBs were created before `lora.block_layout` existed, but newer API
    queries may reference it. We add the column if missing so startup/requests do
    not crash on legacy databases.
    """
    global _schema_migrations_done

    if _schema_migrations_done:
        return

    with _schema_migrations_lock:
        if _schema_migrations_done:
            return

        cur = conn.cursor()
        cur.execute("PRAGMA table_info(lora)")
        columns = {row[1] for row in cur.fetchall()}

        for column_name, column_definition in REQUIRED_LORA_COLUMNS.items():
            if column_name in columns:
                continue
            try:
                cur.execute(
                    f"ALTER TABLE lora ADD COLUMN {column_name} {column_definition};"
                )
                conn.commit()
                columns.add(column_name)
            except sqlite3.OperationalError as exc:
                # Concurrent requests/workers can race on startup:
                # both observe the missing column, one ALTER succeeds and the
                # loser sees "duplicate column name". Treat that loser as success.
                if "duplicate column name" in str(exc).lower():
                    conn.rollback()
                    columns.add(column_name)
                else:
                    raise

        cur.execute("PRAGMA table_info(lora_block_weights)")
        bw_columns = {row[1] for row in cur.fetchall()}
        for column_name, column_definition in REQUIRED_LORA_BLOCK_WEIGHTS_COLUMNS.items():
            if column_name in bw_columns:
                continue
            try:
                cur.execute(
                    f"ALTER TABLE lora_block_weights ADD COLUMN {column_name} {column_definition};"
                )
                conn.commit()
                bw_columns.add(column_name)
            except sqlite3.OperationalError as exc:
                if "duplicate column name" in str(exc).lower():
                    conn.rollback()
                    bw_columns.add(column_name)
                else:
                    raise

        # Ensure lora_user_profiles table exists (Phase 5.1)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS lora_user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lora_id INTEGER NOT NULL,
                stable_id TEXT,
                profile_name TEXT NOT NULL,
                block_weights TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (lora_id) REFERENCES lora(id) ON DELETE CASCADE
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS lora_combined_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_name TEXT NOT NULL,
                recipe_json TEXT NOT NULL,
                combined_payload_json TEXT NOT NULL,
                validated_base_model TEXT NOT NULL,
                validated_layout TEXT NOT NULL,
                included_loras_json TEXT NOT NULL,
                excluded_loras_json TEXT NOT NULL,
                warnings_json TEXT NOT NULL,
                reasons_json TEXT NOT NULL,
                response_schema_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.commit()

        _schema_migrations_done = True


def get_db_connection() -> sqlite3.Connection:
    """
    Open a SQLite connection with Row factory enabled.

    We open a fresh connection per request â€“ totally fine for your usage.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_safe_schema_migrations(conn)
    return conn


def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def derive_role_from_path(file_path: str) -> str:
    path = (file_path or "").replace("\\", "/")
    segments = path.split("/")
    category_to_role = {
        "01 - People": "character",
        "06 - Characters": "character",
        "05 - Body": "character",
        "02 - Styles": "style",
        "08 - Clothing": "clothing",
        "04 - Action": "pose",
        "03 - Utils": "utility",
        "10 - Buildings": "environment",
        "11 - Nature": "environment",
        "07 - Machines_Vehicles": "other",
        "09 - Animals": "other",
    }
    for segment in segments:
        if segment in category_to_role:
            return category_to_role[segment]
    return "other"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- Index status tracking (Phase 5.1: rescan progress) ---
_index_status_lock = threading.Lock()
_index_status: Dict[str, Any] = {
    "indexing": False,
    "last_scan": None,
    "total_loras": 0,
    "with_blocks": 0,
    "duration_last_scan_sec": None,
}


# ----------------------------------------------------------------------
# Block layout validation helpers
# ----------------------------------------------------------------------

def _should_force_flux_fallback_layout(base_model_code: Optional[str], has_blocks: bool) -> bool:
    """
    For Flux/Flux-Krea where has_block_weights is false, we always want
    a stable layout for the UI (16 neutral blocks).
    """
    if has_blocks:
        return False
    code = (base_model_code or "").upper()
    return code in ("FLX", "FLK")


def validate_block_layout_for_search_row(row: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    """
    Validate/normalize block_layout for /api/lora/search rows.

    We do NOT mutate the DB here. We only ensure the response is consistent.
    """
    warnings: List[str] = []

    base_code = (row.get("base_model_code") or "").upper() or None
    has_blocks = bool(row.get("has_block_weights"))

    raw_layout = row.get("block_layout")
    layout = normalize_block_layout(raw_layout)

    # If Flux/FLK and no blocks, force the UI-friendly fallback layout
    if _should_force_flux_fallback_layout(base_code, has_blocks):
        if layout != FLUX_FALLBACK_16:
            if layout is None and raw_layout:
                warnings.append(f"Invalid block_layout '{raw_layout}' normalized to fallback.")
            layout = FLUX_FALLBACK_16

    # If non-Flux and layout is invalid, just null it out
    if raw_layout and layout is None:
        warnings.append(f"Invalid block_layout '{raw_layout}' normalized to null.")

    return layout, warnings


def validate_blocks_response(
    *,
    stable_id: str,
    base_model_code: Optional[str],
    has_blocks: bool,
    lora_type: Optional[str],
    block_layout: Optional[str],
    blocks: List[Dict[str, Any]],
    fallback: bool,
) -> Tuple[Optional[str], List[Dict[str, Any]], List[str]]:
    """
    Validate/normalize block_layout + blocks list for /api/lora/{stable_id}/blocks.

    Returns:
      (final_block_layout, final_blocks, warnings)

    Strategy:
    - Always normalize block_layout against VALID_BLOCK_LAYOUTS
    - For Flux/FLK with no blocks, enforce flux_fallback_16
    - If layout is missing but blocks count matches a known layout, infer it
    - If layout is present and block count mismatches, warn (don't crash)
    """
    warnings: List[str] = []

    base_code = (base_model_code or "").upper() or None
    layout = normalize_block_layout(block_layout)
    if block_layout and layout is None:
        warnings.append(f"Invalid block_layout '{block_layout}' normalized to null.")

    # Enforce Flux fallback layout for the "no blocks" case
    if _should_force_flux_fallback_layout(base_code, has_blocks):
        if layout != FLUX_FALLBACK_16:
            layout = FLUX_FALLBACK_16

    # If we have blocks but no layout, try infer
    if not fallback and blocks and layout is None:
        inferred = infer_layout_from_block_count(len(blocks))
        if inferred:
            layout = inferred
        else:
            warnings.append(
                f"block_layout is null and block count {len(blocks)} does not match a known layout."
            )

    # If we have a layout and blocks, validate count
    if blocks and layout:
        expected = expected_block_count_for_layout(layout)
        if expected is not None and len(blocks) != expected:
            warnings.append(
                f"block_layout '{layout}' expects {expected} blocks but response has {len(blocks)}."
            )

    # Validate basic shape of blocks payload (indices, weights)
    if blocks:
        # Ensure sorted by block_index for UI stability
        try:
            blocks_sorted = sorted(blocks, key=lambda b: int(b.get("block_index") or 0))
        except Exception:
            blocks_sorted = blocks
            warnings.append("Could not sort blocks by block_index (unexpected block_index values).")

        # Check contiguous indices (non-fatal)
        indices: List[int] = []
        try:
            indices = [int(b.get("block_index") or 0) for b in blocks_sorted]
            if indices:
                expected_indices = list(range(min(indices), min(indices) + len(indices)))
                if indices != expected_indices:
                    warnings.append("block_index values are not contiguous; UI may display gaps.")
        except Exception:
            warnings.append("Could not validate block_index contiguity (non-integer indices).")

        # Validate weight range (non-fatal)
        for b in blocks_sorted:
            w = b.get("weight")
            if w is None:
                continue
            try:
                wf = float(w)
                if wf < 0.0 or wf > 1.0:
                    warnings.append("One or more block weights fall outside [0,1].")
                    break
            except Exception:
                warnings.append("One or more block weights are non-numeric.")
                break

        blocks = blocks_sorted

    return layout, blocks, warnings


# ----------------------------------------------------------------------
# FastAPI application
# ----------------------------------------------------------------------

def get_index_summary() -> dict:
    """
    Quick summary of what's in the DB, for UI display after a rescan.

    - total: all LoRAs in the DB
    - with_blocks: LoRAs that have block weights (has_block_weights = 1)
    - no_blocks: LoRAs with no block weights (has_block_weights = 0)
    - with_stable_id: LoRAs that have a stable_id
    """

    summary = {
        "total": 0,
        "with_blocks": 0,
        "no_blocks": 0,
        "with_stable_id": 0,
    }

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        # Total LoRAs
        cur.execute("SELECT COUNT(1) FROM lora")
        summary["total"] = int(cur.fetchone()[0] or 0)

        # With block weights
        cur.execute("SELECT COUNT(1) FROM lora WHERE has_block_weights = 1")
        summary["with_blocks"] = int(cur.fetchone()[0] or 0)

        # Without block weights
        cur.execute("SELECT COUNT(1) FROM lora WHERE has_block_weights = 0")
        summary["no_blocks"] = int(cur.fetchone()[0] or 0)

        # With stable_id
        cur.execute("SELECT COUNT(1) FROM lora WHERE stable_id IS NOT NULL")
        summary["with_stable_id"] = int(cur.fetchone()[0] or 0)

    except Exception as e:
        print(f"[index_summary] ERROR: {e}")
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass

    return summary




def _backfill_flux_layouts(conn: sqlite3.Connection) -> int:
    """Ensure Flux rows always have a normalized block_layout."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, base_model_code, has_block_weights, lora_type, block_layout
        FROM lora
        WHERE UPPER(COALESCE(base_model_code, '')) IN ('FLX', 'FLK');
        """
    )
    rows = cur.fetchall()

    updates = 0
    for row in rows:
        lora_id = row["id"]
        has_blocks = bool(row["has_block_weights"])
        raw_layout = row["block_layout"]
        current_layout = normalize_block_layout(raw_layout)

        new_layout: Optional[str] = current_layout

        if not has_blocks:
            new_layout = FLUX_FALLBACK_16
        elif current_layout is None:
            cur.execute("SELECT COUNT(1) AS cnt FROM lora_block_weights WHERE lora_id = ?", (lora_id,))
            count = int(cur.fetchone()["cnt"] or 0)
            if count > 0:
                new_layout = normalize_block_layout(make_flux_layout(row["lora_type"], count))
                if new_layout is None:
                    new_layout = normalize_block_layout(f"flux_transformer_{count}")
                if new_layout is None:
                    new_layout = infer_layout_from_block_count(count)

        if new_layout != raw_layout:
            cur.execute("UPDATE lora SET block_layout = ? WHERE id = ?", (new_layout, lora_id))
            updates += 1

    if updates:
        conn.commit()

    return updates


def _is_unet57_candidate_row(row: sqlite3.Row) -> bool:
    layout = normalize_block_layout(row["block_layout"])
    if layout in ("unet_57", "flux_unet_57"):
        return True
    lora_type = (row["lora_type"] or "").lower()
    return "unet" in lora_type and "57" in lora_type


def _persist_analysis_for_lora(conn: sqlite3.Connection, row: sqlite3.Row) -> Dict[str, Any]:
    lora_id = row["id"]
    stable_id = row["stable_id"]
    file_path = row["file_path"]
    base_model_code = (row["base_model_code"] or "").upper() or None

    if not file_path or not os.path.isfile(file_path):
        raise FileNotFoundError(f"LoRA file not found on disk: {file_path}")

    analysis = inspect_lora(file_path, base_model_code=base_model_code)
    block_weights = analysis.get("block_weights") or []
    raw_strengths = analysis.get("raw_block_strengths") or []
    has_blocks = bool(block_weights)

    if not has_blocks:
        block_layout = FLUX_FALLBACK_16 if (base_model_code or "") in ("FLX", "FLK") else None
    else:
        block_layout = normalize_block_layout(make_flux_layout(analysis.get("lora_type"), len(block_weights)))
        if block_layout is None:
            block_layout = infer_layout_from_block_count(len(block_weights))

    now_iso = datetime.utcnow().isoformat(timespec="seconds")
    mtime = os.path.getmtime(file_path)
    cur = conn.cursor()

    cur.execute("BEGIN")
    try:
        cur.execute(
            """
            UPDATE lora
            SET
                model_family = ?,
                lora_type = ?,
                rank = ?,
                has_block_weights = ?,
                block_layout = ?,
                last_modified = ?,
                updated_at = ?
            WHERE id = ?;
            """,
            (
                analysis.get("model_family"),
                analysis.get("lora_type"),
                analysis.get("rank"),
                1 if has_blocks else 0,
                block_layout,
                mtime,
                now_iso,
                lora_id,
            ),
        )

        cur.execute("DELETE FROM lora_block_weights WHERE lora_id = ?", (lora_id,))
        if has_blocks:
            for idx, (w, r) in enumerate(zip(block_weights, raw_strengths)):
                cur.execute(
                    """
                    INSERT INTO lora_block_weights
                    (lora_id, stable_id, block_index, weight, raw_strength)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (lora_id, stable_id, idx, float(w), float(r) if r is not None else None),
                )
        cur.execute("COMMIT")
    except Exception:
        cur.execute("ROLLBACK")
        raise

    return {
        "stable_id": stable_id,
        "has_block_weights": has_blocks,
        "block_count": len(block_weights),
        "block_layout": block_layout,
    }


def on_startup_backfills() -> None:
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = get_db_connection()
        updated = _backfill_flux_layouts(conn)
        if updated:
            print(f"[startup] Backfilled normalized Flux block_layout for {updated} row(s).")
    except Exception as exc:
        print(f"[startup] block_layout backfill skipped due to error: {exc}")
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

app = FastAPI(
    title="LoRA Master API",
    version="0.2",
    description="Backend API for LoRA Master (DB-backed).",
)

app.add_event_handler("startup", on_startup_backfills)
app.include_router(model_family_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev â€“ you can tighten this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoRACombineSettings(BaseModel):
    strength_model: float = 1.0
    strength_clip: float = 0.0
    affect_clip: bool = True
    A: Optional[float] = None
    B: Optional[float] = None


class LoRACombineRequest(BaseModel):
    stable_ids: List[str] = Field(default_factory=list)
    per_lora: Dict[str, LoRACombineSettings] = Field(default_factory=dict)


class PrepareBlocksRequest(BaseModel):
    stable_ids: List[str] = Field(min_length=1, max_length=32)
    target_contract_id: Literal["flux1-dev-native-v1"]
    profile_version_ids: Dict[str, str] = Field(default_factory=dict)


class CombinedProfileSaveRequest(BaseModel):
    profile_name: str
    recipe: Dict[str, Any]
    combine_response: Dict[str, Any]


def _parse_json_column(raw_json: Any, *, profile_id: int, field_name: str) -> Any:
    try:
        return json.loads(raw_json)
    except (TypeError, json.JSONDecodeError):
        raise HTTPException(
            status_code=500,
            detail=f"Stored combined profile {profile_id} has invalid JSON in '{field_name}'.",
        )


def _combined_profile_row_to_response(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "profile_name": row["profile_name"],
        "recipe": _parse_json_column(row["recipe_json"], profile_id=row["id"], field_name="recipe_json"),
        "combine_response": _parse_json_column(
            row["combined_payload_json"],
            profile_id=row["id"],
            field_name="combined_payload_json",
        ),
        "validated_base_model": row["validated_base_model"],
        "validated_layout": row["validated_layout"],
        "included_loras": _parse_json_column(
            row["included_loras_json"], profile_id=row["id"], field_name="included_loras_json"
        ),
        "excluded_loras": _parse_json_column(
            row["excluded_loras_json"], profile_id=row["id"], field_name="excluded_loras_json"
        ),
        "warnings": _parse_json_column(row["warnings_json"], profile_id=row["id"], field_name="warnings_json"),
        "reasons": _parse_json_column(row["reasons_json"], profile_id=row["id"], field_name="reasons_json"),
        "response_schema_version": row["response_schema_version"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


FALLBACK_EXCLUDED_REASON_CODE = "fallback_excluded"
FALLBACK_EXCLUDED_REASON_DETAIL = (
    "Excluded by policy: LoRA has no scanned block weights (fallback). "
    "Fallback LoRAs are not allowed in /api/lora/combine."
)


def _make_excluded_lora_entry(
    *,
    stable_id: str,
    filename: Optional[str],
    role: Optional[str],
    reason_code: str,
    reason_detail: str,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "stable_id": stable_id,
        "reason_code": reason_code,
        "reason_detail": reason_detail,
    }
    if filename:
        entry["filename"] = filename
    if role:
        entry["role"] = role
    return entry


def _build_combined_response_payload(compose_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a stable combined payload for /api/lora/combine.

    Aliases:
    - combined_strength_model/combined_strength_clip mirror strength_model/strength_clip.
    - combined_A/combined_B mirror A/B.
    - block_weights + block_weights_csv remain backward-compatible aliases for
      MODEL weights only (same values as block_weights_model/_csv).
    """
    combined_model = compose_result["combined_model"]
    combined_clip = compose_result["combined_clip"]
    block_weights_model_csv = weights_to_csv(combined_model)
    block_weights_clip_csv = None if combined_clip is None else weights_to_csv(combined_clip)

    return {
        "strength_model": compose_result["strength_model_output"],
        "strength_clip": compose_result["strength_clip_output"],
        "combined_strength_model": compose_result["strength_model_output"],
        "combined_strength_clip": compose_result["strength_clip_output"],
        "A": compose_result["combined_A"],
        "B": compose_result["combined_B"],
        "combined_A": compose_result["combined_A"],
        "combined_B": compose_result["combined_B"],
        "block_weights_model": combined_model,
        "block_weights_model_csv": block_weights_model_csv,
        "block_weights_clip": combined_clip,
        "block_weights_clip_csv": block_weights_clip_csv,
        "block_weights": combined_model,
        "block_weights_csv": block_weights_model_csv,
    }


def _build_node_payloads(
    *,
    included_loras: List[LoRAComposeInput],
    rows_by_sid: Dict[str, sqlite3.Row],
    per_lora_cfg: Dict[str, Dict[str, Any]],
    compose_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build per-LoRA node payloads for ComfyUI.

    Product rule: ComfyUI requires one LoRA Loader (Block Weight) node per LoRA.
    Therefore `node_payloads` must contain PER-LoRA settings and PER-LoRA block
    weights (not a synthetic merged LoRA).

    Phase 8.5:
    - Payload construction now passes through lora_block_orchestrator.
    - Public API field names stay backward-compatible (`clip_*` names remain).
    - The legacy `combined` payload is deliberately not changed here.
    - The skeleton preserves scanned per-LoRA block vectors for now.
    """

    clip_unavailable = compose_result["combined_clip"] is None

    orchestrator_inputs: List[LoraBlockOrchestratorInput] = []
    cfg_by_sid: Dict[str, Dict[str, Any]] = {}
    row_by_sid: Dict[str, Optional[sqlite3.Row]] = {}

    for lora in included_loras:
        stable_id = lora.stable_id
        row = rows_by_sid.get(stable_id)
        cfg = per_lora_cfg.get(stable_id, {})

        clip_contributor = bool(row["clip_contributor"]) if row else False
        affect_clip = bool(cfg.get("affect_clip", True))
        if not clip_contributor:
            affect_clip = False

        orchestrator_inputs.append(
            LoraBlockOrchestratorInput(
                stable_id=stable_id,
                filename=row["filename"] if row else None,
                role=derive_role_from_path(row["file_path"]) if row else "other",
                base_model_code=row["base_model_code"] if row else None,
                block_layout=normalize_block_layout(row["block_layout"]) if row else None,
                text_encoder_contributor=clip_contributor,
                affect_text_encoder=affect_clip,
                strength_model=float(cfg.get("strength_model", 1.0)),
                strength_text_encoder=float(cfg.get("strength_clip", 0.0)),
                block_weights=list(lora.block_weights or []),
            )
        )
        cfg_by_sid[stable_id] = cfg
        row_by_sid[stable_id] = row

    orchestrated = orchestrate_lora_block_payloads(orchestrator_inputs)

    node_payloads: List[Dict[str, Any]] = []
    for payload in orchestrated:
        stable_id = payload.stable_id
        cfg = cfg_by_sid.get(stable_id, {})
        row = row_by_sid.get(stable_id)

        # Preserve existing public clip behaviour:
        # - non-clip contributors output 0.0
        # - clip contributors with no clip contribution in the combined result output null
        # - otherwise use the orchestrated text-encoder strength
        if not payload.affect_text_encoder:
            strength_clip: Optional[float] = 0.0
        elif clip_unavailable:
            strength_clip = None
        else:
            strength_clip = payload.strength_text_encoder

        a_val = cfg.get("A")
        b_val = cfg.get("B")
        a_out: Optional[float] = None if a_val is None else float(a_val)
        b_out: Optional[float] = None if b_val is None else float(b_val)

        role_policy = get_role_policy(payload.role)
        role_recommendation_notes = list(build_role_recommendation_notes(payload.role))
        role_strength_recommendation = build_role_strength_recommendation(
            payload.role,
            requested_model_strength=float(cfg.get("_requested_model_strength", payload.strength_model)),
            overlap_corrected_model_strength=payload.strength_model,
            requested_clip_strength=float(cfg.get("_requested_clip_strength", cfg.get("strength_clip", 0.0))),
            clip_contributor=payload.text_encoder_contributor,
        ).to_payload()
        node_payloads.append(
            {
                "stable_id": stable_id,
                "filename": payload.filename,
                "role": payload.role,
                "base_model_code": payload.base_model_code,
                "block_layout": payload.block_layout,
                "clip_contributor": payload.text_encoder_contributor,
                "affect_clip": payload.affect_text_encoder,
                "strength_model": payload.strength_model,
                "strength_clip": strength_clip,
                "A": a_out,
                "B": b_out,
                "block_weights": payload.block_weights,
                # Historical energy strings are not valid Inspire slot mappings.
                "block_weights_csv": None,
                "analysis_block_weights_csv": payload.block_weights_csv,
                "loader_export": build_inspire_flux1_export(
                    payload.block_weights,
                    base_model_code=payload.base_model_code,
                    block_layout=payload.block_layout,
                ),
                "orchestration_notes": payload.notes + role_recommendation_notes,
                "role_recommendation_notes": role_recommendation_notes,
                "role_strength_recommendation": role_strength_recommendation,
                "role_policy": {
                    "priority": role_policy.priority,
                    "intent_label": role_policy.intent_label,
                    "default_model_strength": role_policy.default_model_strength,
                    "default_clip_strength": role_policy.default_clip_strength,
                    "protects_identity": role_policy.protects_identity,
                    "preserves_composition": role_policy.preserves_composition,
                    "treat_as_flavour": role_policy.treat_as_flavour,
                },
            }
        )

    return node_payloads



@app.post("/api/lora/reindex_all")
async def api_reindex_all():
    """Retired: catalogue refresh must never invoke legacy tensor reindexing."""
    raise HTTPException(status_code=410, detail={
        "reason_code": "legacy_reindex_retired",
        "reason": "Use Refresh library (/api/catalogue/refresh) to discover current files. Optional CPU measurements are a separate explicit action.",
    })


# ----------------------------------------------------------------------
# Health check
# ----------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Basic health check + a quick DB summary.
    """
    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB open failed: {e}")

    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS cnt FROM lora;")
        total = cur.fetchone()["cnt"]

        cur.execute(
            "SELECT COUNT(*) AS cnt FROM lora WHERE stable_id IS NOT NULL;"
        )
        with_id = cur.fetchone()["cnt"]

        return {
            "status": "ok",
            "db_path": str(DB_PATH),
            "total_loras": total,
            "with_stable_id": with_id,
        }
    finally:
        conn.close()



@app.get("/api/lora/{stable_id}/thumbnail")
def api_lora_thumbnail(stable_id: str):
    # A thumbnail read must not create a database or trigger lazy migrations.
    try:
        conn = sqlite3.connect(Path(DB_PATH).resolve().as_uri() + "?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT file_path FROM lora WHERE stable_id=?", (stable_id,)).fetchone()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail="The local catalogue is unavailable.") from exc
    if row is None:
        raise HTTPException(status_code=404, detail="No local thumbnail is available.")
    try:
        content, mime = read_thumbnail(row[0], Path(os.environ.get("LORA_ROOT", r"E:\models\loras")))
    except ThumbnailError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return Response(content, media_type=mime, headers={
        "X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=300",
        "ETag": '"' + hashlib.sha256(content).hexdigest() + '"',
    })


def prepare_native_flux_node(row):
    """Return a fresh neutral node plus internal coverage for trusted profile use."""
    if row["base_model_code"] != "FLX":
        raise CoverageError("unsupported_family", "This first target contract supports FLUX.1 catalogue entries only.")
    if not row["file_path"]:
        raise CoverageError("file_missing", "This catalogue entry has no current local file path.")
    coverage = inspect_native_flux_file(Path(row["file_path"]))
    weights = coverage["block_presence_baseline"]
    export = build_inspire_flux1_export(
        weights, base_model_code="FLX", block_layout="flux_transformer_57",
        base_weight=1.0, resolved_patch_keys=coverage["resolved_patch_keys"],
        base_patch_keys=coverage["base_patch_keys"], coverage_complete=True,
        coverage_source=coverage["coverage_source"], loader_source_sha256=LOADER_SHA256,
    )
    export.update({key: coverage[key] for key in (
        "target_contract_id", "target_contract_label", "checkpoint_verified",
        "image_quality_verified", "recommendation_basis", "file_identity", "source_sha256",
    )})
    if export["status"] == "ready":
        export["reason"] = "Conditional on the standard FLUX.1 dev target: current header pairs map to pinned module shapes. Present blocks start at 1, absent blocks at 0. This is an unbalanced manual baseline; actual checkpoint, tensor contents and image quality are not verified."
    return {
        "stable_id": row["stable_id"], "filename": row["filename"],
        "role": derive_role_from_path(row["file_path"]), "base_model_code": "FLX",
        "block_layout": "flux_transformer_57", "block_weights": weights,
        "block_weights_csv": export["numeric_csv"], "analysis_block_weights_csv": None,
        "strength_model": 1.0, "strength_clip": 0.0,
        "clip_contributor": False, "affect_clip": False, "A": 1.0, "B": 1.0,
        "loader_export": export, "orchestration_notes": [export["reason"]],
    }, coverage


def _profile_binding(node, coverage):
    identity = coverage["file_identity"]
    return validate_binding({
        "architecture": "flux.1",
        "slots": [{"group": label.split(" ")[0].lower(), "label": label}
                  for label in node["loader_export"]["architecture_slot_labels"]],
        "source_identity": {"basis": "header_stat", "header_sha256": identity["header_sha256"],
                            "size_bytes": identity["file_size"], "mtime_ns": identity["file_mtime_ns"]},
        "engine_version": "flux_header_coverage_v1",
        "policy_version": "structural_baseline_unvalidated",
        "target_contract": {"id": CONTRACT_ID, "sha256": hashlib.sha256(
            json.dumps(CONTRACT, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()},
        "loader_adapter": {"id": ADAPTER_ID, "source_sha256": coverage["source_sha256"]},
    })


def profile_connection():
    """History reads never create a database or run implicit schema migrations."""
    conn = sqlite3.connect(Path(DB_PATH).resolve().as_uri() + "?mode=rw", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def resolve_profile_default(conn, stable_id):
    row = conn.execute("SELECT stable_id, filename, file_path, base_model_code FROM lora WHERE stable_id=?", (stable_id,)).fetchone()
    if row is None:
        raise ProfileNotFoundError("The LoRA is not in the catalogue")
    try:
        node, coverage = prepare_native_flux_node(row)
    except CoverageError as exc:
        raise ProfileValidationError(str(exc)) from exc
    if node["loader_export"]["status"] != "ready":
        raise ProfileValidationError(node["loader_export"]["reason"])
    return {
        "binding": _profile_binding(node, coverage),
        "values": node["loader_export"]["architecture_slot_values"],
        "settings": {key: node[key] for key in ("role", "strength_model", "strength_clip", "affect_clip")},
        "ab": {},
    }


def initialise_profile_history():
    conn = profile_connection()
    try:
        initialise_profile_schema(conn)
        initialise_composition_schema(conn)
        from composition_preferences import initialise_preference_schema
        initialise_preference_schema(conn)
        from experiment_versions import initialise_experiment_schema
        initialise_experiment_schema(conn)
        from render_trials import initialise_render_trial_schema
        initialise_render_trial_schema(conn)
        from catalogue_refresh import initialise_catalogue_schema
        initialise_catalogue_schema(conn)
    finally:
        conn.close()


app.add_event_handler("startup", initialise_profile_history)
app.include_router(create_profile_version_router(profile_connection, resolve_profile_default))


def _apply_profile_version(node, coverage, version_id, connection=None):
    """Re-export a saved immutable snapshot only against its fresh exact binding."""
    conn = connection if connection is not None else sqlite3.connect(Path(DB_PATH).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        version = get_version(conn, node["stable_id"], version_id)
        root = get_version(conn, node["stable_id"], version["default_id"])
    except (ProfileNotFoundError, ProfileValidationError) as exc:
        raise CoverageError("profile_unavailable", str(exc)) from exc
    except sqlite3.OperationalError as exc:
        raise CoverageError("profile_history_unavailable", "Saved profile history is unavailable in this database.") from exc
    finally:
        if connection is None:
            conn.close()
    binding = _profile_binding(node, coverage)
    if (version["binding"] != binding or root["binding"] != binding
            or root["kind"] != "default" or root["version_id"] != root["default_id"]):
        raise CoverageError("profile_binding_changed", "This saved variant belongs to a different source, target or engine version. Capture a new Default; the older history is preserved.")
    try:
        snapshot = validate_snapshot(binding, version["values"], version["settings"], version["ab"])
    except ProfileValidationError as exc:
        raise CoverageError("invalid_profile_snapshot", str(exc)) from exc
    settings, values = snapshot["settings"], snapshot["values"]
    if settings["affect_clip"]:
        raise CoverageError("unsupported_clip_profile", "This native FLUX contract contains model patches only; a CLIP-enabled variant cannot be exported through it.")
    export = build_inspire_flux1_export(
        values[1:], base_weight=values[0], base_model_code="FLX", block_layout="flux_transformer_57",
        resolved_patch_keys=coverage["resolved_patch_keys"], base_patch_keys=coverage["base_patch_keys"],
        coverage_complete=True, coverage_source=coverage["coverage_source"], loader_source_sha256=LOADER_SHA256,
    )
    metadata = {key: node["loader_export"][key] for key in (
        "target_contract_id", "target_contract_label", "checkpoint_verified", "image_quality_verified",
        "file_identity", "source_sha256",
    )}
    export.update(metadata)
    export["recommendation_basis"] = "structural_baseline_unvalidated" if version["kind"] == "default" else "manual_variant_unvalidated"
    if version.get("provenance", {}).get("method") == "gentle_balance_experiment":
        export["recommendation_basis"] = "experimental_parameter_policy_unvalidated"
    if export["status"] == "ready":
        export["reason"] = "Saved numeric profile mapped against the current header and conditional standard FLUX.1 dev target. Actual checkpoint, tensor contents and image quality remain unverified."
    node.update(settings)
    for symbol in ("A", "B"):
        node[symbol] = snapshot["ab"].get(symbol, {}).get("value", 1.0)
    node.update(profile_version_id=version["version_id"], profile_name=version["name"],
                profile_default_id=version["default_id"], block_weights=values[1:],
                profile_provenance=version.get("provenance", {}),
                block_weights_csv=export["numeric_csv"], loader_export=export,
                ab=snapshot["ab"], orchestration_notes=[export["reason"]])
    return node


@app.post("/api/lora/prepare-blocks")
def api_prepare_blocks(body: PrepareBlocksRequest):
    return _prepare_blocks(body)


def _prepare_blocks(body: PrepareBlocksRequest, connection=None):
    """Prepare conditional numeric slots from current headers, without DB writes.

    This intentionally bypasses old energy/layout caches and balancing heuristics.
    A neutral structural baseline is a manual starting point, not a prediction.
    """
    stable_ids = list(dict.fromkeys(sid.strip() for sid in body.stable_ids if sid.strip()))
    if not stable_ids:
        raise HTTPException(status_code=400, detail="Choose at least one LoRA.")
    if set(body.profile_version_ids) - set(stable_ids):
        raise HTTPException(status_code=400, detail="Profile choices must belong to the requested LoRAs.")
    # Read-only connection avoids even the legacy lazy schema migration path.
    try:
        conn = connection if connection is not None else sqlite3.connect(Path(DB_PATH).resolve().as_uri() + "?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.row_factory = sqlite3.Row
        placeholders = ",".join("?" for _ in stable_ids)
        rows = cursor.execute(f"SELECT stable_id, filename, file_path, base_model_code FROM lora WHERE stable_id IN ({placeholders})", stable_ids).fetchall()
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail="The local catalogue is unavailable or needs its standard schema restored.") from exc
    finally:
        if "cursor" in locals():
            cursor.close()
        if connection is None and "conn" in locals():
            conn.close()
    rows_by_sid = {row["stable_id"]: row for row in rows}
    nodes, excluded = [], []
    for sid in stable_ids:
        row = rows_by_sid.get(sid)
        try:
            if row is None:
                raise CoverageError("missing_lora", "The requested LoRA is not in the catalogue.")
            node, _coverage = prepare_native_flux_node(row)
            node["profile_default_binding"] = _profile_binding(node, _coverage)
            node["profile_version_id"] = None
            node["profile_name"] = "Unsaved Default"
            if sid in body.profile_version_ids:
                node = _apply_profile_version(node, _coverage, body.profile_version_ids[sid], connection=connection)
            nodes.append(node)
        except CoverageError as exc:
            excluded.append({"stable_id": sid, "filename": row["filename"] if row else None,
                             "reason_code": exc.code, "reason_detail": str(exc)})
    ready_ids = [node["stable_id"] for node in nodes if node["loader_export"]["status"] == "ready"]
    result = {
        "engine_kind": "structural_baseline", "target_contract_id": CONTRACT_ID,
        "compatible": len(ready_ids) == len(stable_ids),
        "requested_loras": stable_ids, "included_loras": ready_ids,
        "excluded_loras": excluded, "node_payloads": nodes,
        "validated_base_model": "FLX", "validated_layout": "flux_transformer_57",
        "warnings": ["Structural baseline only: block interactions have not been balanced or validated in generated images."] + (["Some selected LoRAs could not be prepared; the complete selection is not ready."] if len(ready_ids) != len(stable_ids) else []),
        "reasons": excluded,
    }
    result["preparation_digest"] = preparation_digest(result)
    return result


def resolve_composition_preparation(conn, entries, target_contract_id):
    if target_contract_id != CONTRACT_ID:
        raise ProfileValidationError("This target contract is not supported by the current preparation engine")
    try:
        request = PrepareBlocksRequest(
            stable_ids=[entry["stable_id"] for entry in entries],
            target_contract_id=target_contract_id,
            profile_version_ids={entry["stable_id"]: entry["profile_version_id"] for entry in entries},
        )
    except ValueError as exc:
        raise ProfileValidationError("The composition entries are not a valid preparation request") from exc
    return _prepare_blocks(request, connection=conn)


app.include_router(create_composition_version_router(profile_connection, resolve_composition_preparation))
from composition_preference_router import create_composition_preference_router
app.include_router(create_composition_preference_router(profile_connection, resolve_profile_default, resolve_composition_preparation))

# Optional CPU work runs only on an explicit job request, outside the API runtime.
from analysis_job_service import AnalysisJobService
from analysis_job_router import create_analysis_job_router
analysis_jobs = AnalysisJobService(profile_connection, resolve_composition_preparation)
app.include_router(create_analysis_job_router(analysis_jobs))
app.add_event_handler("shutdown", analysis_jobs.shutdown)
from experiment_router import create_experiment_router
app.include_router(create_experiment_router(profile_connection, analysis_jobs, resolve_composition_preparation))
from render_trial_router import create_render_trial_router
app.include_router(create_render_trial_router(profile_connection))
from catalogue_refresh import CatalogueService
from catalogue_router import create_catalogue_router
catalogue_service = CatalogueService(DB_PATH, os.environ.get("LORA_ROOT", r"E:\models\loras"))
app.include_router(create_catalogue_router(catalogue_service))
from compatibility_preflight import CompatibilityService
from compatibility_router import create_compatibility_router
compatibility_service = CompatibilityService(DB_PATH, os.environ.get("LORA_ROOT", r"E:\models\loras"), prepare_native_flux_node)
app.include_router(create_compatibility_router(compatibility_service))
from library_scan_service import LibraryScanService
from library_scan_router import install_library_scan


def create_selected_library_scanner():
    # Resolve only at startup, after the launcher/test has selected its database.
    return LibraryScanService(CatalogueService(
        DB_PATH, os.environ.get("LORA_ROOT", r"E:\models\loras")))


install_library_scan(app, create_selected_library_scanner)


@app.post("/api/lora/combine")
def api_lora_combine(body: LoRACombineRequest):
    stable_ids = [sid.strip() for sid in body.stable_ids if sid and sid.strip()]
    if not stable_ids:
        raise HTTPException(status_code=400, detail="stable_ids must contain at least one stable_id.")

    deduped_stable_ids = list(dict.fromkeys(stable_ids))

    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB open failed: {e}")

    try:
        placeholders = ",".join("?" for _ in deduped_stable_ids)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(lora)")
        lora_columns = {row[1] for row in cur.fetchall()}
        has_file_path_column = "file_path" in lora_columns
        file_path_select = "file_path" if has_file_path_column else "NULL AS file_path"
        cur.execute(
            f"""
            SELECT id, stable_id, filename, {file_path_select}, base_model_code, block_layout, has_block_weights
                   , clip_contributor
            FROM lora
            WHERE stable_id IN ({placeholders});
            """,
            deduped_stable_ids,
        )
        lora_rows = cur.fetchall()

        rows_by_sid = {row["stable_id"]: row for row in lora_rows}
        missing_ids = [sid for sid in deduped_stable_ids if sid not in rows_by_sid]

        excluded_loras: List[Dict[str, Any]] = []
        warnings: List[str] = []
        included_loras: List[LoRAComposeInput] = []
        fallback_excluded_ids: List[str] = []

        for stable_id in deduped_stable_ids:
            if stable_id in missing_ids:
                excluded_loras.append(
                    _make_excluded_lora_entry(
                        stable_id=stable_id,
                        filename=None,
                        role=None,
                        reason_code="missing_lora",
                        reason_detail="Excluded because the requested LoRA was not found.",
                    )
                )
                warnings.append(f"LoRA {stable_id} was not found and was excluded from combination.")
                continue

            row = rows_by_sid[stable_id]
            cur.execute(
                """
                SELECT block_index, weight
                FROM lora_block_weights
                WHERE stable_id = ?
                ORDER BY block_index ASC;
                """,
                (stable_id,),
            )
            bw_rows = cur.fetchall()
            has_rows = len(bw_rows) > 0
            has_flag = bool(row["has_block_weights"])

            if not has_rows:
                fallback_excluded_ids.append(stable_id)
                excluded_loras.append(
                    _make_excluded_lora_entry(
                        stable_id=stable_id,
                        filename=row["filename"],
                        role=derive_role_from_path((row["file_path"] or "")),
                        reason_code=FALLBACK_EXCLUDED_REASON_CODE,
                        reason_detail=FALLBACK_EXCLUDED_REASON_DETAIL,
                    )
                )
                if has_flag:
                    warnings.append(
                        f"LoRA {stable_id} indicates block weights in metadata but has no scanned rows; excluded by fallback policy."
                    )
                continue

            included_loras.append(
                LoRAComposeInput(
                    stable_id=stable_id,
                    base_model_code=row["base_model_code"],
                    block_layout=normalize_block_layout(row["block_layout"]),
                    block_weights=[float(r["weight"]) for r in bw_rows],
                )
            )

        if fallback_excluded_ids:
            warnings.append(
                f"Excluded {len(fallback_excluded_ids)} fallback LoRA(s): fallback LoRAs are not allowed in /api/lora/combine."
            )

        if not included_loras:
            raise HTTPException(
                status_code=400,
                detail={
                    "compatible": False,
                    "validated_base_model": None,
                    "validated_layout": None,
                    "included_loras": [],
                    "excluded_loras": excluded_loras,
                    "reasons": [
                        {
                            "code": "all_loras_excluded",
                            "detail": "All requested LoRAs were excluded by policy because they have no scanned block weights.",
                            "stable_ids": fallback_excluded_ids,
                        }
                    ],
                    "warnings": warnings,
                },
            )

        validation = validate_compatibility(included_loras)
        if not validation["compatible"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "compatible": False,
                    "validated_base_model": validation["validated_base_model"],
                    "validated_layout": validation["validated_layout"],
                    "included_loras": [l.stable_id for l in included_loras],
                    "excluded_loras": excluded_loras,
                    "reasons": validation["reasons"],
                    "warnings": warnings,
                },
            )

        per_lora_cfg: Dict[str, Dict[str, Any]] = {}
        for stable_id, cfg in body.per_lora.items():
            if hasattr(cfg, "model_dump"):
                per_lora_cfg[stable_id] = cfg.model_dump(exclude_none=True)
            else:
                per_lora_cfg[stable_id] = cfg.dict(exclude_none=True)

        energy_inputs: List[LoRAEnergyInput] = []
        for lora in included_loras:
            row = rows_by_sid[lora.stable_id]
            cfg = per_lora_cfg.setdefault(lora.stable_id, {})

            file_path_value = row["file_path"] if "file_path" in row.keys() else None
            if has_file_path_column and (file_path_value is None or str(file_path_value).strip() == ""):
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"LoRA {lora.stable_id} is missing file_path; folder-derived role is required for deterministic combine."
                    ),
                )
            role = derive_role_from_path(file_path_value or "")
            raw_strength_model = float(cfg.get("strength_model", 1.0))
            cfg["_requested_model_strength"] = raw_strength_model
            cfg["_requested_clip_strength"] = float(cfg.get("strength_clip", 0.0))
            energy_inputs.append(
                LoRAEnergyInput(
                    stable_id=lora.stable_id,
                    role=role,
                    block_weights=lora.block_weights,
                    raw_strength_factor=raw_strength_model,
                )
            )

        corrected_strengths = allocate_strengths_with_role_budget_and_overlap(
            [compute_lora_energy_metrics(entry) for entry in energy_inputs]
        )

        for lora in included_loras:
            cfg = per_lora_cfg.setdefault(lora.stable_id, {})
            corrected_strength_model = float(corrected_strengths.get(lora.stable_id, 0.0))
            cfg["strength_model"] = corrected_strength_model

            # IMPORTANT (Phase 8.3 contract + tests):
            # - We enforce clip OFF for non-clip contributors below.
            # - We do NOT scale strength_clip by the model correction ratio for clip contributors.
            #   User-tuned strength_clip remains user-tuned when clip is allowed.

        clip_enforced_warnings: List[str] = []
        for lora in included_loras:
            row = rows_by_sid[lora.stable_id]
            cfg = per_lora_cfg.setdefault(lora.stable_id, {})
            if bool(row["clip_contributor"]):
                continue
            requested_affect_clip = bool(cfg.get("affect_clip", True))
            requested_strength_clip = float(cfg.get("strength_clip", 0.0))
            cfg["affect_clip"] = False
            cfg["strength_clip"] = 0.0
            if requested_affect_clip or requested_strength_clip != 0.0:
                clip_enforced_warnings.append(
                    f"LoRA {lora.stable_id} is not a clip contributor; clip was ignored for this LoRA."
                )

        compose_result = combine_weights_weighted_average(
            included_loras=included_loras,
            per_lora=per_lora_cfg,
            validated_layout=validation["validated_layout"],
        )

        return {
            "response_schema_version": "7.1",
            "compatible": True,
            "validated_base_model": validation["validated_base_model"],
            "validated_layout": validation["validated_layout"],
            "included_loras": [l.stable_id for l in included_loras],
            "excluded_loras": excluded_loras,
            "reasons": [],
            "warnings": warnings + clip_enforced_warnings + compose_result["warnings"],
            "combined": _build_combined_response_payload(compose_result),
            "node_payloads": _build_node_payloads(
                included_loras=included_loras,
                rows_by_sid=rows_by_sid,
                per_lora_cfg=per_lora_cfg,
                compose_result=compose_result,
            ),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        conn.close()


@app.post("/api/lora/combined-profile", status_code=201)
def api_lora_combined_profile_create(body: CombinedProfileSaveRequest):
    profile_name = body.profile_name.strip()
    if not profile_name:
        raise HTTPException(status_code=400, detail="profile_name is required and must be non-empty.")

    combine_response = body.combine_response
    required_keys = {
        "compatible",
        "validated_base_model",
        "validated_layout",
        "included_loras",
        "excluded_loras",
        "reasons",
        "warnings",
        "combined",
        "response_schema_version",
    }
    missing_keys = sorted(required_keys - set(combine_response.keys()))
    if missing_keys:
        raise HTTPException(status_code=400, detail=f"combine_response missing required keys: {missing_keys}")

    if combine_response["compatible"] is not True:
        raise HTTPException(status_code=400, detail="combine_response.compatible must be true to save.")

    if not isinstance(combine_response["included_loras"], list):
        raise HTTPException(status_code=400, detail="combine_response.included_loras must be a list.")
    if not isinstance(combine_response["excluded_loras"], list):
        raise HTTPException(status_code=400, detail="combine_response.excluded_loras must be a list.")
    if not isinstance(combine_response["reasons"], list):
        raise HTTPException(status_code=400, detail="combine_response.reasons must be a list.")
    if not isinstance(combine_response["warnings"], list):
        raise HTTPException(status_code=400, detail="combine_response.warnings must be a list.")
    if not isinstance(combine_response["combined"], dict):
        raise HTTPException(status_code=400, detail="combine_response.combined must be an object.")
    if not isinstance(combine_response["response_schema_version"], str):
        raise HTTPException(status_code=400, detail="combine_response.response_schema_version must be a string.")

    # Store the combine response VERBATIM (Phase 6.2 contract):
    # - Do NOT canonicalize or recompute any CSV fields here.
    # - Do NOT rebuild the combined payload from list values.
    # The save endpoint is persistence-only.
    combined_payload = combine_response
    now = _now_iso()

    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB open failed: {e}")

    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO lora_combined_profiles (
                profile_name,
                recipe_json,
                combined_payload_json,
                validated_base_model,
                validated_layout,
                included_loras_json,
                excluded_loras_json,
                warnings_json,
                reasons_json,
                response_schema_version,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile_name,
                json.dumps(body.recipe),
                json.dumps(combined_payload),
                combine_response["validated_base_model"],
                combine_response["validated_layout"],
                json.dumps(combine_response["included_loras"]),
                json.dumps(combine_response["excluded_loras"]),
                json.dumps(combine_response["warnings"]),
                json.dumps(combine_response["reasons"]),
                combine_response["response_schema_version"],
                now,
                now,
            ),
        )
        conn.commit()
        combined_profile_id = cur.lastrowid

        return {
            "id": combined_profile_id,
            "profile_name": profile_name,
            "created_at": now,
            "updated_at": now,
            "response_schema_version": combine_response["response_schema_version"],
            "validated_base_model": combine_response["validated_base_model"],
            "validated_layout": combine_response["validated_layout"],
        }
    finally:
        conn.close()


@app.get("/api/lora/combined-profiles")
def api_lora_combined_profiles_list():
    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB open failed: {e}")

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                id,
                profile_name,
                validated_base_model,
                validated_layout,
                response_schema_version,
                created_at,
                updated_at
            FROM lora_combined_profiles
            ORDER BY updated_at DESC, created_at DESC;
            """
        )
        rows = cur.fetchall()

        return {
            "profiles": [
                {
                    "id": row["id"],
                    "profile_name": row["profile_name"],
                    "validated_base_model": row["validated_base_model"],
                    "validated_layout": row["validated_layout"],
                    "response_schema_version": row["response_schema_version"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in rows
            ]
        }
    finally:
        conn.close()


@app.get("/api/lora/combined-profile/{combined_profile_id}")
def api_lora_combined_profile_get_by_id(combined_profile_id: int):
    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB open failed: {e}")

    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM lora_combined_profiles WHERE id = ?;", (combined_profile_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Combined profile {combined_profile_id} not found.")

        return _combined_profile_row_to_response(row)
    finally:
        conn.close()


@app.get("/api/lora/combined-profile/by-name/{profile_name}")
def api_lora_combined_profile_get_by_name(profile_name: str):
    normalized_name = profile_name.strip()
    if not normalized_name:
        raise HTTPException(status_code=404, detail="Combined profile name must be non-empty.")

    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB open failed: {e}")

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM lora_combined_profiles
            WHERE profile_name = ?
            ORDER BY updated_at DESC, created_at DESC
            LIMIT 1;
            """,
            (normalized_name,),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Combined profile '{normalized_name}' not found.")

        return _combined_profile_row_to_response(row)
    finally:
        conn.close()

# ----------------------------------------------------------------------
# /api/lora/catalog â€“ catalog alias endpoint
# ----------------------------------------------------------------------

@app.get("/api/lora/catalog")
def api_lora_catalog(
    base: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
    has_blocks: Optional[int] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
):
    return api_lora_search(
        base=base,
        category=category,
        search=search,
        has_blocks=has_blocks,
        limit=limit,
        offset=offset,
    )


# ----------------------------------------------------------------------
# /api/lora/search â€“ main list endpoint used by the React UI
# ----------------------------------------------------------------------

@app.get("/api/lora/search")
def api_lora_search(
    base: Optional[str] = Query(
        default=None,
        description="Base model code (FLX, FLK, W22, SDX, etc.). Use 'ALL' or omit for any.",
    ),
    category: Optional[str] = Query(
        default=None,
        description="Category code (PPL, STL, UTL, etc.). Use 'ALL' or omit for any.",
    ),
    search: Optional[str] = Query(
        default=None,
        description="Substring match on filename (case-insensitive).",
    ),
    has_blocks: Optional[int] = Query(
        default=None,
        description="If 1, only return LoRAs with has_block_weights = 1.",
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=5000,
        description="Max number of results per page.",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of rows to skip (for pagination).",
    ),
):
    """
    Search LoRAs in lora_master.db with pagination support.
    """
    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB open failed: {e}")

    try:
        base_sql = " FROM lora"

        where_clauses: List[str] = []
        params: List[Any] = []

        if base and base.upper() != "ALL":
            where_clauses.append("base_model_code = ?")
            params.append(base.upper())

        if category and category.upper() != "ALL":
            where_clauses.append("category_code = ?")
            params.append(category.upper())

        if search and search.strip():
            where_clauses.append("LOWER(filename) LIKE ?")
            params.append(f"%{search.strip().lower()}%")

        if has_blocks == 1:
            where_clauses.append("has_block_weights = 1")

        where_sql = ""
        if where_clauses:
            where_sql = " WHERE " + " AND ".join(where_clauses)

        # Total count (for pagination)
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) AS cnt{base_sql}{where_sql}", params)
        total = cur.fetchone()["cnt"]

        # Paginated results
        select_sql = """
            SELECT
                id, stable_id, filename, file_path,
                base_model_name, base_model_code,
                category_name, category_code,
                model_family, lora_type, rank,
                has_block_weights, block_layout, clip_contributor,
                created_at, updated_at
        """
        order_sql = " ORDER BY filename ASC LIMIT ? OFFSET ?"
        page_params = params + [limit, offset]

        cur.execute(f"{select_sql}{base_sql}{where_sql}{order_sql}", page_params)
        rows = cur.fetchall()

        results = []
        for row in rows:
            result = row_to_dict(row)
            result["clip_contributor"] = bool(result.get("clip_contributor"))
            result["role"] = derive_role_from_path(result.get("file_path") or "")
            layout, warnings = validate_block_layout_for_search_row(result)
            result["block_layout"] = layout
            if warnings:
                result["validation_warnings"] = warnings
            results.append(result)

        return {
            "results": results,
            "count": len(results),
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    finally:
        conn.close()


# ----------------------------------------------------------------------
# /api/lora/index_status â€“ rescan progress indicator (Phase 5.1)
# NOTE: This must be defined BEFORE /api/lora/{stable_id}, otherwise Starlette
# will match 'index_status' as a stable_id and return 404.
# ----------------------------------------------------------------------

@app.get("/api/lora/index_status")
def api_index_status():
    """Return current indexing status for the frontend progress indicator."""
    with _index_status_lock:
        return dict(_index_status)

# Alias for older/debug callers
@app.get("/api/index_status")
def api_index_status_alias():
    return api_index_status()


# ----------------------------------------------------------------------
# /api/lora/{stable_id} â€“ single LoRA details
# ----------------------------------------------------------------------

@app.get("/api/lora/{stable_id}")
def api_lora_details(stable_id: str):
    """
    Return full details for a LoRA identified by its stable_id.
    Used by the details panel in the UI.
    """
    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB open failed: {e}")

    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM lora WHERE stable_id = ?;", (stable_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"No LoRA found with stable_id '{stable_id}'",
            )
        result = row_to_dict(row)
        result["clip_contributor"] = bool(result.get("clip_contributor"))
        return result
    finally:
        conn.close()


# ----------------------------------------------------------------------
# /api/lora/{stable_id}/blocks â€“ block weight profile
# ----------------------------------------------------------------------

@app.get("/api/lora/{stable_id}/blocks")
def api_lora_blocks(stable_id: str):
    """
    Return per-block weights for a LoRA (if present).
    """
    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB open failed: {e}")

    try:
        cur = conn.cursor()

        # Look up LoRA by stable_id first
        cur.execute(
            "SELECT id, has_block_weights, lora_type, block_layout, base_model_code FROM lora WHERE stable_id = ?;",
            (stable_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"No LoRA found with stable_id '{stable_id}'",
            )

        lora_id = row["id"]
        has_blocks = bool(row["has_block_weights"])
        lora_type = row["lora_type"]
        base_model_code = row["base_model_code"]
        block_layout = row["block_layout"]

        if not has_blocks:
            normalized_layout = normalize_block_layout(block_layout)
            if _should_force_flux_fallback_layout(base_model_code, has_blocks):
                normalized_layout = FLUX_FALLBACK_16

            fallback_count = fallback_block_count_for_layout(normalized_layout)
            fallback = fallback_count is not None
            fallback_reason = (
                "No stored block weights; using neutral fallback profile for "
                f"layout {normalized_layout}"
                if fallback and normalized_layout
                else None
            )

            fallback_blocks = (
                [
                    {"block_index": i, "weight": 1.0, "raw_strength": None}
                    for i in range(fallback_count)
                ]
                if fallback_count is not None
                else []
            )

            final_layout, final_blocks, warnings = validate_blocks_response(
                stable_id=stable_id,
                base_model_code=base_model_code,
                has_blocks=False,
                lora_type=lora_type,
                block_layout=normalized_layout,
                blocks=fallback_blocks,
                fallback=fallback,
            )

            return {
                "stable_id": stable_id,
                "has_block_weights": False,
                "block_layout": final_layout,
                "fallback": fallback,
                "fallback_reason": fallback_reason,
                "blocks": final_blocks,
                "validation_warnings": warnings,
            }

        # Has blocks: fetch them
        cur.execute(
            """
            SELECT block_index, weight, raw_strength
            FROM lora_block_weights
            WHERE lora_id = ?
            ORDER BY block_index ASC;
            """,
            (lora_id,),
        )
        blocks_rows = cur.fetchall()

        blocks = [
            {
                "block_index": r["block_index"],
                "weight": float(r["weight"]),
                "raw_strength": float(r["raw_strength"])
                if r["raw_strength"] is not None
                else None,
            }
            for r in blocks_rows
        ]

        final_layout, final_blocks, warnings = validate_blocks_response(
            stable_id=stable_id,
            base_model_code=base_model_code,
            has_blocks=True,
            lora_type=lora_type,
            block_layout=block_layout,
            blocks=blocks,
            fallback=False,
        )

        return {
            "stable_id": stable_id,
            "has_block_weights": bool(final_blocks),
            "block_layout": final_layout,
            "fallback": False,
            "fallback_reason": None,
            "blocks": final_blocks,
            "validation_warnings": warnings,
        }
    finally:
        conn.close()


# ----------------------------------------------------------------------
# /api/lora/{stable_id}/profiles â€“ user override profiles (Phase 5.1)
# ----------------------------------------------------------------------

def _lookup_lora_by_stable_id(conn: sqlite3.Connection, stable_id: str) -> sqlite3.Row:
    cur = conn.cursor()
    cur.execute("SELECT id, stable_id, block_layout FROM lora WHERE stable_id = ?;", (stable_id,))
    row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No LoRA found with stable_id '{stable_id}'")
    return row


@app.get("/api/lora/{stable_id}/profiles")
def api_lora_profiles_list(stable_id: str):
    """List all saved user profiles for a LoRA."""
    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB open failed: {e}")

    try:
        _lookup_lora_by_stable_id(conn, stable_id)

        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, profile_name, block_weights, created_at, updated_at
            FROM lora_user_profiles
            WHERE stable_id = ?
            ORDER BY created_at ASC;
            """,
            (stable_id,),
        )
        rows = cur.fetchall()

        profiles = []
        for r in rows:
            try:
                weights = json.loads(r["block_weights"])
            except (json.JSONDecodeError, TypeError):
                weights = []
            profiles.append({
                "id": r["id"],
                "profile_name": r["profile_name"],
                "block_weights": weights,
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            })

        return {"stable_id": stable_id, "profiles": profiles}
    finally:
        conn.close()


@app.post("/api/lora/{stable_id}/profiles")
def api_lora_profiles_create(stable_id: str, body: Dict[str, Any] = Body(...)):
    """Create a new user override profile for a LoRA."""
    profile_name = (body.get("profile_name") or "").strip()
    if not profile_name:
        raise HTTPException(status_code=400, detail="profile_name is required and must be non-empty.")

    block_weights = body.get("block_weights")
    if not isinstance(block_weights, list):
        raise HTTPException(status_code=400, detail="block_weights must be an array of floats.")

    try:
        block_weights = [float(w) for w in block_weights]
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="All block_weights values must be numeric.")

    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB open failed: {e}")

    try:
        lora_row = _lookup_lora_by_stable_id(conn, stable_id)
        lora_id = lora_row["id"]
        layout = lora_row["block_layout"]

        if layout:
            expected = expected_block_count_for_layout(layout)
            if expected is not None and len(block_weights) != expected:
                raise HTTPException(
                    status_code=400,
                    detail=f"block_weights length {len(block_weights)} does not match expected {expected} for layout '{layout}'.",
                )

        now = _now_iso()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO lora_user_profiles (lora_id, stable_id, profile_name, block_weights, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (lora_id, stable_id, profile_name, json.dumps(block_weights), now, now),
        )
        conn.commit()
        new_id = cur.lastrowid

        return {
            "id": new_id,
            "profile_name": profile_name,
            "block_weights": block_weights,
            "created_at": now,
            "updated_at": now,
        }
    finally:
        conn.close()


@app.put("/api/lora/{stable_id}/profiles/{profile_id}")
def api_lora_profiles_update(stable_id: str, profile_id: int, body: Dict[str, Any] = Body(...)):
    """Update an existing user override profile."""
    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB open failed: {e}")

    try:
        lora_row = _lookup_lora_by_stable_id(conn, stable_id)
        layout = lora_row["block_layout"]

        cur = conn.cursor()
        cur.execute(
            "SELECT id, profile_name, block_weights FROM lora_user_profiles WHERE id = ? AND stable_id = ?;",
            (profile_id, stable_id),
        )
        existing = cur.fetchone()
        if existing is None:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found for LoRA '{stable_id}'.")

        profile_name = body.get("profile_name")
        if profile_name is not None:
            profile_name = profile_name.strip()
            if not profile_name:
                raise HTTPException(status_code=400, detail="profile_name must be non-empty if provided.")
        else:
            profile_name = existing["profile_name"]

        block_weights = body.get("block_weights")
        if block_weights is not None:
            if not isinstance(block_weights, list):
                raise HTTPException(status_code=400, detail="block_weights must be an array of floats.")
            try:
                block_weights = [float(w) for w in block_weights]
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="All block_weights values must be numeric.")

            if layout:
                expected = expected_block_count_for_layout(layout)
                if expected is not None and len(block_weights) != expected:
                    raise HTTPException(
                        status_code=400,
                        detail=f"block_weights length {len(block_weights)} does not match expected {expected} for layout '{layout}'.",
                    )
        else:
            try:
                block_weights = json.loads(existing["block_weights"])
            except (json.JSONDecodeError, TypeError):
                block_weights = []

        now = _now_iso()
        cur.execute(
            """
            UPDATE lora_user_profiles SET profile_name = ?, block_weights = ?, updated_at = ?
            WHERE id = ? AND stable_id = ?;
            """,
            (profile_name, json.dumps(block_weights), now, profile_id, stable_id),
        )
        conn.commit()

        return {
            "id": profile_id,
            "profile_name": profile_name,
            "block_weights": block_weights,
            "created_at": existing["created_at"] if "created_at" in existing.keys() else now,
            "updated_at": now,
        }
    finally:
        conn.close()


@app.delete("/api/lora/{stable_id}/profiles/{profile_id}")
def api_lora_profiles_delete(stable_id: str, profile_id: int):
    """Delete a user override profile."""
    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB open failed: {e}")

    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM lora_user_profiles WHERE id = ? AND stable_id = ?;",
            (profile_id, stable_id),
        )
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found for LoRA '{stable_id}'.")

        cur.execute("DELETE FROM lora_user_profiles WHERE id = ? AND stable_id = ?;", (profile_id, stable_id))
        conn.commit()

        return {"status": "ok"}
    finally:
        conn.close()


# ----------------------------------------------------------------------
# /api/lora/{stable_id}/export â€“ CSV export (Phase 5.1)
# ----------------------------------------------------------------------

@app.get("/api/lora/{stable_id}/export")
def api_lora_export_csv(stable_id: str):
    """Export block weights as a CSV file."""
    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB open failed: {e}")

    try:
        cur = conn.cursor()
        cur.execute("SELECT id, has_block_weights, block_layout FROM lora WHERE stable_id = ?;", (stable_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"No LoRA found with stable_id '{stable_id}'")

        lora_id = row["id"]
        has_blocks = bool(row["has_block_weights"])

        if not has_blocks:
            raise HTTPException(status_code=404, detail=f"LoRA '{stable_id}' has no extracted block weights to export.")

        cur.execute(
            """
            SELECT block_index, weight, raw_strength
            FROM lora_block_weights
            WHERE lora_id = ?
            ORDER BY block_index ASC;
            """,
            (lora_id,),
        )
        blocks = cur.fetchall()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["block_index", "weight", "raw_strength"])
        for b in blocks:
            writer.writerow([
                b["block_index"],
                f"{float(b['weight']):.6f}",
                f"{float(b['raw_strength']):.6f}" if b["raw_strength"] is not None else "",
            ])

        output.seek(0)
        filename = f"{stable_id}_blocks.csv"
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    finally:
        conn.close()


# ----------------------------------------------------------------------
# Optional: ad-hoc inspection endpoint (not used by the UI, handy for testing)
# ----------------------------------------------------------------------

@app.post("/inspect")
def api_inspect_lora(path: str, base_model_code: Optional[str] = None):
    """
    Quick helper to run the delta_inspector_engine on an arbitrary file.
    """
    try:
        result = inspect_lora(path, base_model_code=base_model_code)
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No such file: {path}")
    except NotImplementedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------

@app.post("/api/lora/reindex_one/{stable_id}")
async def api_reindex_one(stable_id: str):
    """
    Reindex a SINGLE LoRA by stable_id.
    """

    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB open failed: {e}")

    try:
        cur = conn.cursor()

        # Fetch LoRA row
        cur.execute(
            """
            SELECT id, stable_id, file_path, base_model_code, lora_type, block_layout, last_modified
            FROM lora
            WHERE stable_id = ?;
            """,
            (stable_id,),
        )
        row = cur.fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"No LoRA found with stable_id '{stable_id}'",
            )

        result = _persist_analysis_for_lora(conn, row)
        return {"status": "ok", **result}

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    finally:
        conn.close()


@app.post("/api/lora/reindex_unet57")
async def api_reindex_unet57(limit: int = Query(default=0, ge=0, le=50000)):
    """Bulk reindex rows that qualify for UNet 57 extraction."""
    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB open failed: {e}")

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, stable_id, file_path, base_model_code, lora_type, block_layout
            FROM lora
            WHERE stable_id IS NOT NULL
            ORDER BY id ASC
            """
        )
        rows = cur.fetchall()
        candidates = [row for row in rows if _is_unet57_candidate_row(row)]
        if limit > 0:
            candidates = candidates[:limit]

        processed = 0
        failures: List[Dict[str, str]] = []
        for row in candidates:
            try:
                _persist_analysis_for_lora(conn, row)
                processed += 1
            except Exception as exc:
                failures.append({"stable_id": row["stable_id"], "error": str(exc)})

        return {
            "status": "ok",
            "candidates": len(candidates),
            "processed": processed,
            "failed": len(failures),
            "failures": failures[:25],
        }
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "lora_api_server:app",
        host="127.0.0.1",
        port=5001,
        reload=False,
    )

