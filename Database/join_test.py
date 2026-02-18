import sqlite3

db = r"E:\LoRA Project\Database\lora_master.db"
con = sqlite3.connect(db)
cur = con.cursor()

print("\n=== sample join test ===")

cur.execute("""
SELECT l.stable_id, COUNT(w.id)
FROM lora l
LEFT JOIN lora_block_weights w
  ON l.id = w.lora_id
WHERE l.stable_id IN ('FLX-PPL-020','FLX-STL-059','FLX-CHT-005','FLX-CHT-011')
GROUP BY l.stable_id
""")

rows = cur.fetchall()

if not rows:
    print("No rows returned.")
else:
    for row in rows:
        print(row)

con.close()
