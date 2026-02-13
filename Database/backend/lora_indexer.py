import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Tuple

from delta_inspector_engine import inspect_lora
from block_layouts import FLUX_FALLBACK_16, make_flux_layout


# --- CONFIG --- #

LORA_ROOT = r"E:\models\loras"
DB_PATH = r"E:\LoRA Project\Database\lora_master.db"

# --- MAPPINGS (same as catalog skeleton) --- #

BASE_MODEL_MAP: Dict[str, Tuple[str, str]] = {
    "FLUX": ("FLX", "Flux"),
    "Flux Krea": ("FLK", "Flux Krea"),
    "Illustrious": ("ILL", "Illustrious"),
    "PONY": ("PNY", "Pony"),
    "SD": ("SD1", "SD"),
    "SDXL": ("SDX", "SDXL"),
    "WAN2.1": ("W21", "WAN2.1"),
    "WAN2.2": ("W22", "WAN2.2"),
}

CATEGORY_INDEX_MAP: Dict[str, Tuple[str, str]] = {
    "01": ("PPL", "People"),
    "02": ("STL", "Styles"),
    "03": ("UTL", "Utils"),
    "04": ("ACT", "Action"),
    "05": ("BDY", "Body"),
    "06": ("CHT", "Characters"),
    "07": ("MCV", "Machines_Vehicles"),
    "08": ("CLT", "Clothing"),
    "09": ("ANM", "Animals"),
    "10": ("BLD", "Buildings"),
    "11": ("NAT", "Nature"),
}


# --- DATA STRUCTURES --- #

@dataclass
class LoraRecord:
    file_path: str
    filename: str

    base_model_name: Optional[str]
    base_model_code: Optional[str]

    category_name: Optional[str]
    category_code: Optional[str]

    # From analysis engine (Flux etc.)
    model_family: Optional[str] = None
    lora_type: Optional[str] = None
    rank: Optional[int] = None

    has_block_weights: bool = False
    block_layout: Optional[str] = None  # <-- NEW
    last_modified: float = 0.0  # filesystem mtime


def normalise_path(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def parse_base_and_category(file_path: str, root_dir: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    root_dir = normalise_path(root_dir)
    file_path_norm = normalise_path(file_path)

    try:
        rel_path = os.path.relpath(file_path_norm, root_dir)
    except ValueError:
        return None, None, None, None

    parts = rel_path.split(os.sep)
    # Typical:
    #   BASE\NN - Category\file.safetensors
    # WAN style:
    #   WAN2.2\T2V\NN - Category\file.safetensors
    if len(parts) < 3:
        return None, None, None, None

    base_model_folder = parts[0]

    # Detect WAN "mode" folder (T2V/I2V/etc) and shift the category folder index
    category_index = 1
    if base_model_folder in ("WAN2.1", "WAN2.2") and len(parts) >= 4:
        mode_folder = (parts[1] or "").strip().upper()
        if mode_folder in ("T2V", "I2V", "V2V", "T2I", "I2I", "IMG2VID", "IMAGE2VIDEO"):
            category_index = 2

    category_folder = parts[category_index]

    # --- Base model --- #
    base_model_name = base_model_folder
    base_model_code = None

    if base_model_folder in BASE_MODEL_MAP:
        code, human = BASE_MODEL_MAP[base_model_folder]
        base_model_code = code
        base_model_name = human

    # --- Category --- #
    category_code = None
    category_name = None

    first_token = category_folder.split(" ")[0].strip()
    if first_token in CATEGORY_INDEX_MAP:
        cat_short, cat_name = CATEGORY_INDEX_MAP[first_token]
        category_code = cat_short
        category_name = cat_name
    else:
        category_name = category_folder

    return base_model_name, base_model_code, category_name, category_code

def find_lora_files(root_dir: str) -> List[str]:
    root_dir = normalise_path(root_dir)
    results: List[str] = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(".safetensors"):
                results.append(os.path.join(dirpath, name))

    return results


# --- DATABASE SETUP --- #

def _ensure_column_exists(conn: sqlite3.Connection, table: str, column: str, col_def: str) -> None:
    """
    Idempotent: adds column if missing. Safe to run every time.
    """
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    cols = {row[1] for row in cur.fetchall()}
    if column not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def};")
        conn.commit()


def ensure_db():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.isdir(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Main LoRA table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS lora (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL UNIQUE,
            filename TEXT NOT NULL,

            base_model_name TEXT,
            base_model_code TEXT,
            category_name TEXT,
            category_code TEXT,

            model_family TEXT,
            lora_type TEXT,
            rank INTEGER,

            has_block_weights INTEGER NOT NULL DEFAULT 0,
            block_layout TEXT,

            last_modified REAL NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )

    # Ensure block_layout exists even if DB was created before we added it
    _ensure_column_exists(conn, "lora", "block_layout", "TEXT")

    # Per-block weights (base analysis)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS lora_block_weights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lora_id INTEGER NOT NULL,
            block_index INTEGER NOT NULL,
            weight REAL NOT NULL,
            raw_strength REAL,

            FOREIGN KEY (lora_id) REFERENCES lora(id) ON DELETE CASCADE
        );
        """
    )

    # Placeholder for future: Clink override patterns / notes
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS lora_clink_overrides (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lora_id INTEGER NOT NULL,
            profile_name TEXT NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,

            FOREIGN KEY (lora_id) REFERENCES lora(id) ON DELETE CASCADE
        );
        """
    )

    conn.commit()
    return conn


# --- DB HELPERS --- #

def get_existing_lora_row(cur: sqlite3.Cursor, file_path: str) -> Optional[sqlite3.Row]:
    cur.execute("SELECT * FROM lora WHERE file_path = ?", (file_path,))
    row = cur.fetchone()
    return row


def upsert_lora(cur: sqlite3.Cursor, rec: LoraRecord) -> int:
    now_iso = datetime.utcnow().isoformat(timespec="seconds")

    row = get_existing_lora_row(cur, rec.file_path)
    if row is None:
        # Insert new row
        cur.execute(
            """
            INSERT INTO lora (
                file_path, filename,
                base_model_name, base_model_code,
                category_name, category_code,
                model_family, lora_type, rank,
                has_block_weights, block_layout,
                last_modified, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                rec.file_path,
                rec.filename,
                rec.base_model_name,
                rec.base_model_code,
                rec.category_name,
                rec.category_code,
                rec.model_family,
                rec.lora_type,
                rec.rank,
                1 if rec.has_block_weights else 0,
                rec.block_layout,
                rec.last_modified,
                now_iso,
                now_iso,
            ),
        )
        return cur.lastrowid
    else:
        # Update existing row
        cur.execute(
            """
            UPDATE lora
            SET
                filename = ?,
                base_model_name = ?,
                base_model_code = ?,
                category_name = ?,
                category_code = ?,
                model_family = ?,
                lora_type = ?,
                rank = ?,
                has_block_weights = ?,
                block_layout = ?,
                last_modified = ?,
                updated_at = ?
            WHERE file_path = ?;
            """,
            (
                rec.filename,
                rec.base_model_name,
                rec.base_model_code,
                rec.category_name,
                rec.category_code,
                rec.model_family,
                rec.lora_type,
                rec.rank,
                1 if rec.has_block_weights else 0,
                rec.block_layout,
                rec.last_modified,
                now_iso,
                rec.file_path,
            ),
        )
        cur.execute("SELECT id FROM lora WHERE file_path = ?", (rec.file_path,))
        row = cur.fetchone()
        return row[0]


def replace_block_weights(
    cur: sqlite3.Cursor,
    lora_db_id: int,
    block_weights: List[float],
    raw_strengths: List[float],
):
    # Clear existing
    cur.execute("DELETE FROM lora_block_weights WHERE lora_id = ?", (lora_db_id,))

    # Insert new
    for idx, (w, r) in enumerate(zip(block_weights, raw_strengths)):
        cur.execute(
            """
            INSERT INTO lora_block_weights (
                lora_id, block_index, weight, raw_strength
            )
            VALUES (?, ?, ?, ?);
            """,
            (lora_db_id, idx, float(w), float(r)),
        )


# --- MAIN INDEXING LOGIC --- #

def main():
    print("=== LoRA Indexer v0.1 ===")
    print(f"Root directory : {LORA_ROOT}")
    print(f"Database path  : {DB_PATH}")
    print()

    root_dir = normalise_path(LORA_ROOT)
    if not os.path.isdir(root_dir):
        print(f"ERROR: Root directory does not exist: {root_dir}")
        return

    conn = ensure_db()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Discover all LoRA files
    print("Scanning filesystem for .safetensors...")
    all_files = find_lora_files(root_dir)
    print(f"Found {len(all_files)} file(s).")
    print()

    processed = 0
    skipped_unchanged = 0
    errors = 0
    flux_with_weights = 0
    flux_sdxl_style = 0

    for idx, path in enumerate(sorted(all_files)):
        file_path = normalise_path(path)
        filename = os.path.basename(file_path)
        mtime = os.path.getmtime(file_path)

        base_model_name, base_model_code, category_name, category_code = parse_base_and_category(
            file_path, root_dir
        )

        # Check if we already have this file and if it's unchanged
        existing = get_existing_lora_row(cur, file_path)
        if existing is not None:
            last_mod = existing["last_modified"]
            if abs(last_mod - mtime) < 1e-6:
                # No change – skip
                skipped_unchanged += 1
                continue

        rec = LoraRecord(
            file_path=file_path,
            filename=filename,
            base_model_name=base_model_name,
            base_model_code=base_model_code,
            category_name=category_name,
            category_code=category_code,
            model_family=None,
            lora_type=None,
            rank=None,
            has_block_weights=False,
            block_layout=None,   # <-- NEW
            last_modified=mtime,
        )

        # Run analysis only for Flux / Flux Krea (for now)
        try:
            if rec.base_model_code in ("FLX", "FLK"):
                analysis = inspect_lora(rec.file_path, base_model_code=rec.base_model_code)
                rec.model_family = analysis.get("model_family")
                rec.lora_type = analysis.get("lora_type")
                rec.rank = analysis.get("rank")
                block_weights = analysis.get("block_weights") or []
                raw_strengths = analysis.get("raw_block_strengths") or []

                if block_weights:
                    rec.has_block_weights = True
                    rec.block_layout = make_flux_layout(rec.lora_type, len(block_weights))
                    flux_with_weights += 1
                else:
                    rec.has_block_weights = False
                    rec.block_layout = FLUX_FALLBACK_16
                    flux_sdxl_style += 1
            else:
                # For non-Flux base models, just store metadata for now
                rec.model_family = None
                rec.lora_type = None
                rec.rank = None
                rec.has_block_weights = False
                rec.block_layout = None
                block_weights = []
                raw_strengths = []

        except Exception as e:
            errors += 1
            print(f"[ERROR] {file_path}")
            print(f"        {e}")
            block_weights = []
            raw_strengths = []
            rec.has_block_weights = False
            rec.block_layout = None

        # Insert/update row
        lora_id = upsert_lora(cur, rec)

        # Store block weights if any
        if rec.has_block_weights and block_weights:
            replace_block_weights(cur, lora_id, block_weights, raw_strengths)

        processed += 1

        # Light progress feedback every 50 files
        if processed % 50 == 0:
            print(
                f"Processed {processed}/{len(all_files)} "
                f"(skipped unchanged: {skipped_unchanged}, errors: {errors})"
            )

    conn.commit()
    conn.close()

    print()
    print("=== Indexing complete ===")
    print(f"Total files discovered       : {len(all_files)}")
    print(f"Processed (inserted/updated) : {processed}")
    print(f"Skipped unchanged            : {skipped_unchanged}")
    print(f"Errors                       : {errors}")
    print(f"Flux with block weights      : {flux_with_weights}")
    print(f"Flux SDXL-style (no weights) : {flux_sdxl_style}")
    print()
    print("You can re-run this script any time; it will only re-process")
    print("files whose modification time has changed since the last run.")


if __name__ == "__main__":
    main()
