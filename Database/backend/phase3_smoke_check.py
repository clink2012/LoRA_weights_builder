import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR.parent / "lora_master.db"


def main() -> int:
    print("=== Phase 3 smoke check ===")
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
        WHERE block_layout = 'unet_57' AND has_block_weights = 1;
        """
    )
    with_weights = int(cur.fetchone()["cnt"] or 0)

    cur.execute(
        """
        SELECT COUNT(1) AS cnt
        FROM lora l
        LEFT JOIN (
            SELECT lora_id, COUNT(1) AS bw_count
            FROM lora_block_weights
            GROUP BY lora_id
        ) bw ON bw.lora_id = l.id
        WHERE l.block_layout = 'unet_57'
          AND l.has_block_weights = 1
          AND COALESCE(bw.bw_count, 0) != 57;
        """
    )
    mismatched = int(cur.fetchone()["cnt"] or 0)

    print(f"UNet-57 rows with has_block_weights=1: {with_weights}")
    print(f"UNet-57 rows with block weight count !=57: {mismatched}")

    print("\n[SAMPLE] UNet-57 rows")
    cur.execute(
        """
        SELECT stable_id, filename, has_block_weights
        FROM lora
        WHERE block_layout = 'unet_57'
        ORDER BY id DESC
        LIMIT 10;
        """
    )
    for row in cur.fetchall():
        print(f"  {row['stable_id'] or 'NULL':<16} | has_blocks={int(row['has_block_weights'])} | {row['filename']}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
