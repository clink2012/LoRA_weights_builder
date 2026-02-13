import os
import sqlite3
from datetime import datetime

DB_PATH = r"E:\models\loras\Database\lora_master.db"


def ensure_stable_id_column(conn: sqlite3.Connection):
    cur = conn.cursor()

    # Check if the column exists
    cur.execute("PRAGMA table_info(lora);")
    cols = [row[1] for row in cur.fetchall()]

    if "stable_id" not in cols:
        print("Adding 'stable_id' column to lora table...")
        cur.execute(
            "ALTER TABLE lora ADD COLUMN stable_id TEXT;"
        )
        conn.commit()
        print("Column added.\n")
    else:
        print("'stable_id' column already exists.\n")


def generate_stable_id(base_code: str, cat_code: str, index: int) -> str:
    return f"{base_code}-{cat_code}-{index:03d}"


def assign_ids(conn: sqlite3.Connection):
    cur = conn.cursor()

    print("=== Assigning Stable LoRA IDs ===\n")

    # Fetch all LoRAs with base code + category
    cur.execute(
        """
        SELECT id, filename, base_model_code, category_code
        FROM lora
        WHERE base_model_code IS NOT NULL
          AND category_code IS NOT NULL
        ORDER BY base_model_code ASC,
                 category_code ASC,
                 filename ASC;
        """
    )

    rows = cur.fetchall()

    # Group files by (base_model_code, category_code)
    groups = {}
    for row in rows:
        key = (row["base_model_code"], row["category_code"])
        groups.setdefault(key, []).append(row)

    total_assigned = 0
    skipped = 0

    # Assign IDs to each group
    for (base_code, cat_code), items in groups.items():
        print(f"Processing group: {base_code}-{cat_code} ({len(items)} LoRAs)")

        for idx, item in enumerate(items, start=1):
            lora_id = item["id"]

            # Check if ID already exists
            cur.execute(
                "SELECT stable_id FROM lora WHERE id = ?",
                (lora_id,),
            )
            existing = cur.fetchone()[0]

            if existing:
                skipped += 1
                continue

            new_id = generate_stable_id(base_code, cat_code, idx)

            cur.execute(
                """
                UPDATE lora
                SET stable_id = ?
                WHERE id = ?;
                """,
                (new_id, lora_id),
            )
            total_assigned += 1

        print(f"  Assigned IDs in group: {total_assigned}\n")

    conn.commit()

    print("=== ID Assignment Complete ===")
    print(f"Total IDs assigned : {total_assigned}")
    print(f"Skipped (already had ID): {skipped}")


def main():
    print(f"Database: {DB_PATH}\n")

    if not os.path.isfile(DB_PATH):
        print("ERROR: Database not found. Run the indexer first.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    ensure_stable_id_column(conn)
    assign_ids(conn)

    conn.close()


if __name__ == "__main__":
    main()
