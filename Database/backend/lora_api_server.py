from __future__ import annotations

import sqlite3
import threading
import time

import os
from datetime import datetime

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from delta_inspector_engine import inspect_lora  # optional helper
from lora_indexer import main as index_all_loras
from lora_id_assigner import main as assign_stable_ids
from block_layouts import (
    FLUX_FALLBACK_16,
    UNET_57,
    expected_block_count_for_layout,
    infer_layout_from_block_count,
    make_flux_layout,
    normalize_block_layout,
)

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


_schema_migrations_lock = threading.Lock()
_schema_migrations_done = False


_layout_backfill_lock = threading.Lock()
_layout_backfill_done = False


def _infer_layout_for_existing_row(base_model_code: Optional[str], lora_type: Optional[str], block_count: int) -> Optional[str]:
    if block_count <= 0:
        return None

    code = (base_model_code or "").upper()
    if code in ("FLX", "FLK"):
        from_type = make_flux_layout(lora_type, block_count)
        if from_type:
            return from_type
        return f"flux_transformer_{block_count}"

    if block_count == 57:
        return UNET_57

    return None


def backfill_missing_block_layouts(conn: sqlite3.Connection) -> None:
    """
    Startup-safe backfill: if a row has blocks but empty block_layout, infer one.
    """
    global _layout_backfill_done

    if _layout_backfill_done:
        return

    with _layout_backfill_lock:
        if _layout_backfill_done:
            return

        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                l.id,
                l.base_model_code,
                l.lora_type,
                COUNT(w.id) AS block_count
            FROM lora AS l
            JOIN lora_block_weights AS w ON w.lora_id = l.id
            WHERE l.has_block_weights = 1
              AND (l.block_layout IS NULL OR TRIM(l.block_layout) = '')
            GROUP BY l.id, l.base_model_code, l.lora_type
            """
        )
        rows = cur.fetchall()

        for row in rows:
            layout = _infer_layout_for_existing_row(
                row["base_model_code"],
                row["lora_type"],
                int(row["block_count"] or 0),
            )
            if not layout:
                continue
            cur.execute(
                "UPDATE lora SET block_layout = ? WHERE id = ?",
                (layout, row["id"]),
            )

        conn.commit()
        _layout_backfill_done = True


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

        _schema_migrations_done = True


def get_db_connection() -> sqlite3.Connection:
    """
    Open a SQLite connection with Row factory enabled.

    We open a fresh connection per request – totally fine for your usage.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_safe_schema_migrations(conn)
    backfill_missing_block_layouts(conn)
    return conn


def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


# ----------------------------------------------------------------------
# Block layout validation helpers
# ----------------------------------------------------------------------

def _infer_layout_from_block_count(block_count: int) -> Optional[str]:
    """
    Infer a layout identifier when only count is known.
    """
    return infer_layout_from_block_count(block_count)


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
    - Always normalize block_layout against the shared taxonomy helper
    - For Flux/FLK with no blocks, enforce fallback layout
    - If layout is missing but blocks exist, infer from block count
    - If layout is present and block count mismatches, warn (don't crash)
    """
    warnings: List[str] = []

    base_code = (base_model_code or "").upper() or None
    layout = normalize_block_layout(block_layout)

    # Enforce Flux fallback layout for the "no blocks" case
    if _should_force_flux_fallback_layout(base_code, has_blocks):
        if layout != FLUX_FALLBACK_16:
            layout = FLUX_FALLBACK_16

    # If we have blocks but no layout, try infer
    if not fallback and blocks and layout is None:
        inferred = _infer_layout_from_block_count(len(blocks))
        if inferred:
            layout = inferred
        else:
            warnings.append(
                f"block_layout is null and could not be inferred from block count {len(blocks)}."
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


app = FastAPI(
    title="LoRA Master API",
    version="0.2",
    description="Backend API for LoRA Master (DB-backed).",
)



@app.on_event("startup")
def startup_backfill_layouts() -> None:
    conn = get_db_connection()
    conn.close()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev – you can tighten this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/lora/reindex_all")
async def api_reindex_all():
    """
    Full rescan + reindex of ALL LoRA files.

    - Runs the filesystem indexer (lora_indexer.main via index_all_loras)
    - Then assigns/refreshes stable IDs (lora_id_assigner.main)
    - Returns a small summary for the UI to display.
    """
    start = time.time()

    # 1) Re-scan the whole E:\models\loras tree and update lora_master.db
    index_all_loras()

    # 2) Ensure stable_id column exists and is filled/updated
    assign_stable_ids()

    duration = round(time.time() - start, 1)

    # 3) Build a quick DB summary for the UI
    summary = get_index_summary()

    return {
        "status": "ok",
        "duration_sec": duration,
        "summary": summary,
    }


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
        default=200,
        ge=1,
        le=5000,
        description="Max number of results to return.",
    ),
):
    """
    Search LoRAs in lora_master.db.
    """
    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB open failed: {e}")

    try:
        sql = """
            SELECT
                id,
                stable_id,
                filename,
                file_path,
                base_model_name,
                base_model_code,
                category_name,
                category_code,
                model_family,
                lora_type,
                rank,
                has_block_weights,
                block_layout,
                created_at,
                updated_at
            FROM lora
        """

        where_clauses: List[str] = []
        params: List[Any] = []

        # Base model filter (unless ALL / blank)
        if base and base.upper() != "ALL":
            where_clauses.append("base_model_code = ?")
            params.append(base.upper())

        # Category filter (unless ALL / blank)
        if category and category.upper() != "ALL":
            where_clauses.append("category_code = ?")
            params.append(category.upper())

        # Filename search (case-insensitive)
        if search and search.strip():
            where_clauses.append("LOWER(filename) LIKE ?")
            params.append(f"%{search.strip().lower()}%")

        # Only LoRAs with stored block weights
        if has_blocks == 1:
            where_clauses.append("has_block_weights = 1")

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        sql += " ORDER BY filename ASC LIMIT ?"
        params.append(limit)

        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()

        results = []
        warnings_total: List[str] = []

        for row in rows:
            result = row_to_dict(row)

            layout, warnings = validate_block_layout_for_search_row(result)
            result["block_layout"] = layout

            # Optional: surface warnings for debugging (UI can ignore)
            if warnings:
                result["validation_warnings"] = warnings
                warnings_total.extend([f"{result.get('stable_id') or result.get('id')}: {w}" for w in warnings])

            results.append(result)

        return {
            "results": results,
            "count": len(results),
            # Helpful for debugging during Phase 1; UI can ignore.
            "validation_warnings_total": warnings_total,
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
            # Always return neutral fallback blocks.
            fallback_blocks = [
                {"block_index": i, "weight": 1.0, "raw_strength": None}
                for i in range(16)
            ]

            final_layout, final_blocks, warnings = validate_blocks_response(
                stable_id=stable_id,
                base_model_code=base_model_code,
                has_blocks=False,
                lora_type=lora_type,
                block_layout=block_layout,
                blocks=fallback_blocks,
                fallback=True,
            )

            return {
                "stable_id": stable_id,
                "has_block_weights": False,
                "fallback": True,
                "fallback_reason": "LoRA has_block_weights is false; returning neutral fallback blocks.",
                "lora_type": lora_type,
                "block_layout": final_layout,
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
            "blocks": final_blocks,
            "validation_warnings": warnings,
        }
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
    Supports FLX / FLK (Flux) only for now.
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
            SELECT id, file_path, base_model_code, last_modified
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

        lora_id = row["id"]
        file_path = row["file_path"]
        base_model_code = (row["base_model_code"] or "").upper()

        if base_model_code not in ("FLX", "FLK"):
            raise HTTPException(
                status_code=400,
                detail=f"Single reindex only supported for FLX / FLK",
            )

        if not file_path or not os.path.isfile(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"LoRA file not found on disk: {file_path}",
            )

        analysis = inspect_lora(file_path, base_model_code=base_model_code)

        block_weights = analysis.get("block_weights") or []
        raw_strengths = analysis.get("raw_block_strengths") or []
        has_blocks = bool(block_weights)

        # Determine layout
        if not has_blocks:
            block_layout = FLUX_FALLBACK_16
        else:
            block_layout = make_flux_layout(analysis.get("lora_type"), len(block_weights))
            if block_layout is None:
                block_layout = _infer_layout_from_block_count(len(block_weights))

        # Update DB row
        now_iso = datetime.utcnow().isoformat(timespec="seconds")
        mtime = os.path.getmtime(file_path)

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

        # Replace block weights
        cur.execute("DELETE FROM lora_block_weights WHERE lora_id = ?", (lora_id,))
        if has_blocks:
            for idx, (w, r) in enumerate(zip(block_weights, raw_strengths)):
                cur.execute(
                    """
                    INSERT INTO lora_block_weights
                    (lora_id, block_index, weight, raw_strength)
                    VALUES (?, ?, ?, ?);
                    """,
                    (lora_id, idx, float(w), float(r) if r is not None else None),
                )

        conn.commit()

        return {
            "status": "ok",
            "stable_id": stable_id,
            "has_block_weights": has_blocks,
            "block_count": len(block_weights),
            "block_layout": block_layout,
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
