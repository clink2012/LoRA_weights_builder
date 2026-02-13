from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from delta_inspector_engine import inspect_lora  # optional helper
from lora_indexer import main as index_all_loras
from lora_id_assigner import main as assign_stable_ids

# ----------------------------------------------------------------------
# Paths & basic config
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Main SQLite DB (same path as your indexer/inspector scripts)
DB_PATH = BASE_DIR.parent / "lora_master.db"


def get_db_connection() -> sqlite3.Connection:
    """
    Open a SQLite connection with Row factory enabled.

    We open a fresh connection per request – totally fine for your usage.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


# ----------------------------------------------------------------------
# FastAPI application
# ----------------------------------------------------------------------

def get_index_summary() -> dict:
    """
    Quick summary of what's in the DB, for UI display after a rescan.
    We keep it generic (not Flux-only):

    - total: all LoRAs in the DB
    - with_blocks: LoRAs that have block weights (block_count > 0)
    - no_blocks: LoRAs with no block weights (block_count = 0)
    """
    summary = {
        "total": 0,
        "with_blocks": 0,
        "no_blocks": 0,
    }

    try:
        conn = sqlite3.connect(LORA_DB_PATH)
        cur = conn.cursor()

        # Total number of rows
        cur.execute("SELECT COUNT(*) FROM loras")
        row = cur.fetchone()
        summary["total"] = int(row[0] or 0)

        # With block weights
        cur.execute("SELECT COUNT(*) FROM loras WHERE block_count > 0")
        row = cur.fetchone()
        summary["with_blocks"] = int(row[0] or 0)

        # No block weights
        cur.execute("SELECT COUNT(*) FROM loras WHERE block_count = 0")
        row = cur.fetchone()
        summary["no_blocks"] = int(row[0] or 0)

    except Exception as e:
        # Don't crash the API if stats fail – just log and return zeros
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

    This is designed to match what App.jsx expects:
      - GET /api/lora/search?base=FLX&category=PPL&search=emma&has_blocks=1
      - Response: { "results": [ { ...lora row... }, ... ] }
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

        results = [row_to_dict(r) for r in rows]

        return {
            "results": results,
            "count": len(results),
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

    Shape is designed for App.jsx:
      {
        "stable_id": "...",
        "has_block_weights": true/false,
        "blocks": [
          { "block_index": 0, "weight": 0.95, "raw_strength": 12.34 },
          ...
        ]
      }

    # Quick manual check:
    # curl -s "http://127.0.0.1:8000/api/lora/<stable_id>/blocks" | jq
    """
    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB open failed: {e}")

    try:
        cur = conn.cursor()

        # Look up LoRA by stable_id first
        cur.execute(
            "SELECT id, has_block_weights, lora_type FROM lora WHERE stable_id = ?;",
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

        if not has_blocks:
            lora_type = row["lora_type"]
            if lora_type is None:
                cur.execute(
                    "SELECT lora_type FROM lora WHERE id = ?;",
                    (lora_id,),
                )
                lora_type_row = cur.fetchone()
                lora_type = lora_type_row["lora_type"] if lora_type_row is not None else None

            fallback_blocks = [
                {
                    "block_index": i,
                    "weight": 1.0,
                    "raw_strength": None,
                }
                for i in range(16)
            ]

            return {
                "stable_id": stable_id,
                "has_block_weights": False,
                "fallback": True,
                "fallback_reason": "LoRA has_block_weights is false; returning neutral fallback blocks.",
                "lora_type": lora_type,
                "blocks": fallback_blocks,
            }

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

        return {
            "stable_id": stable_id,
            "has_block_weights": bool(blocks),
            "blocks": blocks,
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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "lora_api_server:app",
        host="127.0.0.1",
        port=5001,
        reload=False,
    )
