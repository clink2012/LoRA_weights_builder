import os
import sqlite3
from typing import Optional

DB_PATH = r"E:\models\loras\Database\lora_master.db"


def connect_db() -> sqlite3.Connection:
    if not os.path.isfile(DB_PATH):
        raise FileNotFoundError(f"Database not found at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def show_summary(conn: sqlite3.Connection):
    cur = conn.cursor()

    print("=== LoRA DB Summary ===\n")

    # Total LoRAs
    cur.execute("SELECT COUNT(*) AS cnt FROM lora;")
    total = cur.fetchone()["cnt"]
    print(f"Total LoRAs in DB: {total}")

    # How many have stable IDs vs not
    cur.execute("SELECT COUNT(*) AS cnt FROM lora WHERE stable_id IS NOT NULL;")
    with_id = cur.fetchone()["cnt"]
    without_id = total - with_id
    print(f"With stable_id      : {with_id}")
    print(f"Without stable_id   : {without_id}")

    # By base model
    print("\nBy base_model_code:")
    cur.execute(
        """
        SELECT
            COALESCE(base_model_code, 'NULL') AS code,
            COUNT(*) AS cnt,
            SUM(CASE WHEN stable_id IS NOT NULL THEN 1 ELSE 0 END) AS with_id
        FROM lora
        GROUP BY COALESCE(base_model_code, 'NULL')
        ORDER BY cnt DESC;
        """
    )
    for row in cur.fetchall():
        code = row["code"]
        cnt = row["cnt"]
        wid = row["with_id"] or 0
        print(f"  {code:4} : {cnt:4}  (with ID: {wid})")

    # Flux breakdown
    print("\nFlux breakdown (FLX / FLK):")
    cur.execute(
        """
        SELECT
            COUNT(*) AS total_flux,
            SUM(CASE WHEN has_block_weights = 1 THEN 1 ELSE 0 END) AS with_weights
        FROM lora
        WHERE base_model_code IN ('FLX','FLK');
        """
    )
    flux = cur.fetchone()
    total_flux = flux["total_flux"] or 0
    with_weights = flux["with_weights"] or 0
    print(f"  Total Flux LoRAs       : {total_flux}")
    print(f"  With block weights     : {with_weights}")
    print(f"  Without block weights  : {total_flux - with_weights}")

    print("\nDone.\n")


def list_flux_with_weights(conn: sqlite3.Connection, limit: int = 20):
    cur = conn.cursor()
    print(f"=== Flux LoRAs with block weights (showing up to {limit}) ===\n")

    cur.execute(
        """
        SELECT id, filename, file_path, lora_type, stable_id
        FROM lora
        WHERE base_model_code IN ('FLX','FLK')
          AND has_block_weights = 1
        ORDER BY filename ASC
        LIMIT ?;
        """,
        (limit,),
    )

    rows = cur.fetchall()
    if not rows:
        print("No Flux LoRAs with block weights found.")
        return

    for row in rows:
        sid = row["stable_id"] or "(no ID)"
        print(f"[{row['id']}] {row['filename']}")
        print(f"     Stable ID: {sid}")
        print(f"     Type     : {row['lora_type']}")
        print(f"     Path     : {row['file_path']}")
        print()


def inspect_single_lora(conn: sqlite3.Connection, lora_id: int):
    cur = conn.cursor()
    cur.execute("SELECT * FROM lora WHERE id = ?;", (lora_id,))
    row = cur.fetchone()
    if not row:
        print(f"No LoRA found with id={lora_id}")
        return

    print("=== LoRA Details ===")
    print(f"ID              : {row['id']}")
    print(f"Stable ID       : {row['stable_id']}")
    print(f"Filename        : {row['filename']}")
    print(f"File path       : {row['file_path']}")
    print(f"Base model name : {row['base_model_name']}")
    print(f"Base model code : {row['base_model_code']}")
    print(f"Category name   : {row['category_name']}")
    print(f"Category code   : {row['category_code']}")
    print(f"Model family    : {row['model_family']}")
    print(f"LoRA type       : {row['lora_type']}")
    print(f"Rank            : {row['rank']}")
    print(f"Has block weights: {row['has_block_weights']}")
    print(f"Last modified   : {row['last_modified']}")
    print(f"Created at      : {row['created_at']}")
    print(f"Updated at      : {row['updated_at']}")
    print()

    if not row["has_block_weights"]:
        print("This LoRA has no stored block weights.")
        return

    # Fetch block weights
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

    if not blocks:
        print("No block weights found in lora_block_weights (unexpected).")
        return

    print("=== Block Weights ===")
    weights = [b["weight"] for b in blocks]
    raw = [b["raw_strength"] for b in blocks]

    print(f"Total blocks   : {len(blocks)}")
    print(f"Weights (0–1)  : {weights}")
    print(f"Raw strengths  : {raw}")


def main():
    print("=== LoRA DB Inspector v0.2 ===")
    print(f"Database path: {DB_PATH}")
    print()

    try:
        conn = connect_db()
    except Exception as e:
        print("ERROR connecting to DB:")
        print(e)
        return

    show_summary(conn)
    list_flux_with_weights(conn, limit=20)

    print("You can inspect a specific LoRA by ID from the list above.")
    user = input("\nEnter a LoRA ID to inspect in detail (or press Enter to exit): ").strip()
    if not user:
        print("Exiting.")
        conn.close()
        return

    try:
        lora_id = int(user)
    except ValueError:
        print("Not a valid integer ID. Exiting.")
        conn.close()
        return

    print()
    inspect_single_lora(conn, lora_id)
    conn.close()


if __name__ == "__main__":
    main()
