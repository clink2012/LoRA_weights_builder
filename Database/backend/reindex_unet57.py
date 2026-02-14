from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import List, Dict

from lora_api_server import get_db_connection, _is_unet57_candidate_row, _persist_analysis_for_lora


def reindex_bulk(limit: int = 0) -> Dict[str, int]:
    conn = get_db_connection()
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
        failed = 0
        for row in candidates:
            try:
                _persist_analysis_for_lora(conn, row)
                processed += 1
            except Exception as exc:
                failed += 1
                print(f"[FAIL] {row['stable_id']}: {exc}")

        return {"candidates": len(candidates), "processed": processed, "failed": failed}
    finally:
        conn.close()


def reindex_single(stable_id: str) -> None:
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, stable_id, file_path, base_model_code, lora_type, block_layout
            FROM lora WHERE stable_id = ?
            """,
            (stable_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise SystemExit(f"No LoRA found with stable_id={stable_id}")
        _persist_analysis_for_lora(conn, row)
        print(f"[OK] Reindexed {stable_id}")
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Reindex UNet 57 candidates")
    parser.add_argument("--stable-id", default=None, help="Reindex one row by stable_id")
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows for bulk mode")
    args = parser.parse_args()

    if args.stable_id:
        reindex_single(args.stable_id)
        return 0

    stats = reindex_bulk(limit=args.limit)
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
