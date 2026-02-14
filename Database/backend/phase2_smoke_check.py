import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR.parent / "lora_master.db"


def main() -> int:
    print("=== Phase 2 smoke check ===")
    print(f"DB: {DB_PATH}")

    if not DB_PATH.exists():
        print("[WARN] DB file not found. Nothing to check.")
        return 0

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT COUNT(1) AS cnt
        FROM lora
        WHERE UPPER(COALESCE(base_model_code, '')) IN ('FLX','FLK')
          AND block_layout IS NULL;
        """
    )
    flux_null_layout = int(cur.fetchone()["cnt"] or 0)

    cur.execute(
        """
        SELECT COUNT(1) AS dup_groups
        FROM (
            SELECT stable_id
            FROM lora
            WHERE stable_id IS NOT NULL
            GROUP BY stable_id
            HAVING COUNT(1) > 1
        );
        """
    )
    duplicate_groups = int(cur.fetchone()["dup_groups"] or 0)

    print()
    print("[CHECK] Flux rows with null block_layout")
    print(f"  value={flux_null_layout}")
    print("  PASS" if flux_null_layout == 0 else "  FAIL")

    print()
    print("[CHECK] Duplicate stable_id groups")
    print(f"  value={duplicate_groups}")
    print("  PASS" if duplicate_groups == 0 else "  FAIL")

    print()
    print("[SAMPLE] 10 Flux rows")
    cur.execute(
        """
        SELECT stable_id, filename, has_block_weights, block_layout
        FROM lora
        WHERE UPPER(COALESCE(base_model_code, '')) IN ('FLX','FLK')
        ORDER BY id DESC
        LIMIT 10;
        """
    )
    rows = cur.fetchall()
    for row in rows:
        print(
            f"  {row['stable_id'] or 'NULL':<16} | "
            f"has_blocks={int(row['has_block_weights'])} | "
            f"layout={row['block_layout'] or 'NULL'} | "
            f"{row['filename']}"
        )

    conn.close()
    print("\n=== Smoke check complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
