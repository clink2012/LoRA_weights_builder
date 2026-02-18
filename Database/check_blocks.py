import sqlite3

db = r"E:\LoRA Project\Database\lora_master.db"
con = sqlite3.connect(db)
cur = con.cursor()

sids = [
    "FLX-PPL-020",
    "FLX-STL-059",
    "FLX-CHT-005",
    "FLX-CHT-011",
]

print("\n=== tables ===")
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
for (name,) in cur.fetchall():
    print("-", name)

print("\n=== lora schema (columns) ===")
cur.execute("PRAGMA table_info(lora);")
cols = cur.fetchall()
for c in cols:
    # (cid, name, type, notnull, dflt_value, pk)
    print(c)

print("\n=== lora_block_weights rows ===")
for sid in sids:
    cur.execute("SELECT COUNT(*) FROM lora_block_weights WHERE stable_id=?", (sid,))
    print(sid, "block_rows =", cur.fetchone()[0])

print("\n=== lora table rows (selected columns) ===")
# Build a safe SELECT using columns that actually exist
col_names = [c[1] for c in cols]
wanted = ["stable_id", "filename", "file_path", "has_block_weights", "block_layout", "lora_type", "base_model_code", "category_code"]
selected = [c for c in wanted if c in col_names]
print("Using columns:", selected)

for sid in sids:
    cur.execute(f"SELECT {', '.join(selected)} FROM lora WHERE stable_id=?", (sid,))
    print(cur.fetchone())

con.close()
