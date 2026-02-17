from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
import time

import os
from datetime import datetime, timezone

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from delta_inspector_engine import inspect_lora  # optional helper
from lora_indexer import main as index_all_loras
from lora_id_assigner import main as assign_stable_ids
from block_layouts import (
    FLUX_FALLBACK_16,
    expected_block_count_for_layout,
    fallback_block_count_for_layout,
    infer_layout_from_block_count,
    make_flux_layout,
    normalize_block_layout,
)
from lora_composer import LoRAComposeInput, combine_weights_weighted_average, validate_compatibility

# ----------------------------------------------------------------------
# Paths & basic config
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Main SQLite DB (same path as your indexer/inspector scripts)
DB_PATH = BASE_DIR.parent / "lora_master.db"

# Add future self-healing columns here, e.g. {"new_column": "INTEGER DEFAULT 0"}.
REQUIRED_LORA_COLUMNS = {
    "block_layout": "TEXT",
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
        conn.commit()

        _schema_migrations_done = True


def get_db_connection() -> sqlite3.Connection:
    """
    Open a SQLite connection with Row factory enabled.

    We open a fresh connection per request – totally fine for your usage.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_safe_schema_migrations(conn)
    return conn


def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


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
            blocks_sorted = sorted(blocks, key=lambda b: int(b.get("block_index", 0)))
        except Exception:
            blocks_sorted = blocks
            warnings.append("Could not sort blocks by block_index (unexpected block_index values).")

        # Check contiguous indices (non-fatal)
        indices: List[int] = []
        try:
            indices = [int(b.get("block_index")) for b in blocks_sorted]
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
            conn.close()
        except Exception:
            pass

app = FastAPI(
    title="LoRA Master API",
    version="0.2",
    description="Backend API for LoRA Master (DB-backed).",
)

app.add_event_handler("startup", on_startup_backfills)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev – you can tighten this later
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


@app.post("/api/lora/reindex_all")
async def api_reindex_all():
    """
    Full rescan + reindex of ALL LoRA files.

    - Runs the filesystem indexer (lora_indexer.main via index_all_loras)
    - Then assigns/refreshes stable IDs (lora_id_assigner.main)
    - Returns a small summary for the UI to display.
    """
    with _index_status_lock:
        if _index_status["indexing"]:
            return {"status": "already_running", "message": "Indexing is already in progress."}
        _index_status["indexing"] = True

    start = time.time()

    try:
        # 1) Re-scan the whole E:\models\loras tree and update lora_master.db
        index_all_loras()

        # 2) Ensure stable_id column exists and is filled/updated
        assign_stable_ids()

        duration = round(time.time() - start, 1)

        # 3) Build a quick DB summary for the UI
        summary = get_index_summary()

        with _index_status_lock:
            _index_status["indexing"] = False
            _index_status["last_scan"] = _now_iso()
            _index_status["total_loras"] = summary.get("total", 0)
            _index_status["with_blocks"] = summary.get("with_blocks", 0)
            _index_status["duration_last_scan_sec"] = duration

        return {
            "status": "ok",
            "duration_sec": duration,
            "summary": summary,
        }
    except Exception:
        with _index_status_lock:
            _index_status["indexing"] = False
        raise


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
        cur.execute(
            f"""
            SELECT id, stable_id, base_model_code, block_layout, has_block_weights
            FROM lora
            WHERE stable_id IN ({placeholders});
            """,
            deduped_stable_ids,
        )
        lora_rows = cur.fetchall()

        rows_by_sid = {row["stable_id"]: row for row in lora_rows}
        missing_ids = [sid for sid in deduped_stable_ids if sid not in rows_by_sid]

        excluded_loras: List[str] = []
        warnings: List[str] = []
        included_loras: List[LoRAComposeInput] = []

        for stable_id in deduped_stable_ids:
            if stable_id in missing_ids:
                excluded_loras.append(stable_id)
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
                excluded_loras.append(stable_id)
                if has_flag:
                    warnings.append(
                        f"LoRA {stable_id} indicates block weights in metadata but has no scanned rows and was excluded from combination."
                    )
                else:
                    warnings.append(
                        f"LoRA {stable_id} has no scanned block weights and was excluded from combination."
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

        if not included_loras:
            raise HTTPException(
                status_code=400,
                detail={
                    "compatible": False,
                    "validated_base_model": None,
                    "validated_layout": None,
                    "included_loras": [],
                    "excluded_loras": excluded_loras,
                    "reasons": ["No LoRAs with scanned block weights were available to combine."],
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

        per_lora_cfg = {
            stable_id: cfg.model_dump(exclude_none=True)
            for stable_id, cfg in body.per_lora.items()
        }

        combine_result = combine_weights_weighted_average(
            included_loras=included_loras,
            per_lora=per_lora_cfg,
            validated_layout=validation["validated_layout"],
        )

        return {
            "compatible": True,
            "validated_base_model": validation["validated_base_model"],
            "validated_layout": validation["validated_layout"],
            "included_loras": [l.stable_id for l in included_loras],
            "excluded_loras": excluded_loras,
            "reasons": [],
            "warnings": warnings + combine_result["warnings"],
            "combined": combine_result["combined"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        conn.close()

# ----------------------------------------------------------------------
# /api/lora/search – main list endpoint used by the React UI
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
                has_block_weights, block_layout,
                created_at, updated_at
        """
        order_sql = " ORDER BY filename ASC LIMIT ? OFFSET ?"
        page_params = params + [limit, offset]

        cur.execute(f"{select_sql}{base_sql}{where_sql}{order_sql}", page_params)
        rows = cur.fetchall()

        results = []
        for row in rows:
            result = row_to_dict(row)
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
# /api/lora/{stable_id} – single LoRA details
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

        return row_to_dict(row)
    finally:
        conn.close()


# ----------------------------------------------------------------------
# /api/lora/{stable_id}/blocks – block weight profile
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
# /api/lora/index_status – rescan progress indicator (Phase 5.1)
# ----------------------------------------------------------------------

@app.get("/api/lora/index_status")
def api_index_status():
    """Return current indexing status for the frontend progress indicator."""
    with _index_status_lock:
        return dict(_index_status)


# ----------------------------------------------------------------------
# /api/lora/{stable_id}/profiles – user override profiles (Phase 5.1)
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
# /api/lora/{stable_id}/export – CSV export (Phase 5.1)
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
