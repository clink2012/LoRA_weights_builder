import os
import re
import sqlite3

DB_PATH = r"E:/LoRA Project/Database/lora_master.db"


def ensure_stable_id_column(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    # Check if the column exists
    cur.execute("PRAGMA table_info(lora);")
    cols = [row[1] for row in cur.fetchall()]

    if "stable_id" not in cols:
        print("Adding 'stable_id' column to lora table...")
        cur.execute("ALTER TABLE lora ADD COLUMN stable_id TEXT;")
        conn.commit()
        print("Column added.")
        print("")
    else:
        print("'stable_id' column already exists.")
        print("")


def generate_stable_id(base_code: str, cat_code: str, index: int) -> str:
    return f"{base_code}-{cat_code}-{index:03d}"


_STABLE_ID_SUFFIX_RE = re.compile(
    r"^(?P<prefix>[A-Z0-9]{3}-[A-Z0-9]{3})-(?P<num>[0-9]{3})$"
)


def _extract_numeric_suffix(stable_id: str, expected_prefix: str) -> int | None:
    """Return the numeric ### suffix if stable_id matches PREFIX-###, else None."""
    if not stable_id:
        return None

    m = _STABLE_ID_SUFFIX_RE.match(stable_id.strip().upper())
    if not m:
        return None

    if m.group("prefix") != expected_prefix:
        return None

    try:
        return int(m.group("num"))
    except ValueError:
        return None


def assign_ids(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    print("=== Assigning Stable LoRA IDs ===")
    print("")

    # Fetch all LoRAs with base code + category (include stable_id so we can avoid per-row queries)
    cur.execute(
        """
        SELECT id, filename, base_model_code, category_code, stable_id
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
    groups: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for row in rows:
        key = (row["base_model_code"], row["category_code"])
        groups.setdefault(key, []).append(row)

    total_assigned = 0
    total_skipped = 0

    # Assign IDs to each group
    for (base_code, cat_code), items in groups.items():
        prefix = f"{base_code}-{cat_code}".upper()
        print(f"Processing group: {prefix} ({len(items)} LoRAs)")

        # Collect numeric suffixes already in use for this base/category.
        used_numbers: set[int] = set()
        for item in items:
            existing = item["stable_id"]
            n = _extract_numeric_suffix(existing, prefix) if existing else None
            if n is not None:
                used_numbers.add(n)

        group_assigned = 0
        group_skipped = 0

        next_candidate = 1
        for item in items:
            if item["stable_id"]:
                group_skipped += 1
                continue

            # Find the next available numeric suffix not already used.
            while next_candidate in used_numbers:
                next_candidate += 1

            new_id = generate_stable_id(base_code, cat_code, next_candidate)

            cur.execute(
                """
                UPDATE lora
                SET stable_id = ?
                WHERE id = ?;
                """,
                (new_id, item["id"]),
            )

            used_numbers.add(next_candidate)
            next_candidate += 1

            group_assigned += 1
            total_assigned += 1

        total_skipped += group_skipped
        print(f"  Assigned IDs in group: {group_assigned}")
        print(f"  Skipped (already had ID): {group_skipped}")
        print("")

    conn.commit()

    print("=== ID Assignment Complete ===")
    print(f"Total IDs assigned : {total_assigned}")
    print(f"Skipped (already had ID): {total_skipped}")


def main() -> None:
    print(f"Database: {DB_PATH}")
    print("")

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
