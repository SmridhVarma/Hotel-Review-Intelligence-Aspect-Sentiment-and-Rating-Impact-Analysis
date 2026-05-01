# check_chroma.py — Stage 5 utility | Module C (Agent): Inspects the current ChromaDB state by querying the SQLite backing store directly.
#
# Input:  chromadb/chroma.sqlite3
# Output: None (prints collection stats to stdout)
import sqlite3
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(_ROOT, "chromadb", "chroma.sqlite3")
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# List all tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print("Tables:", tables)

# List collections
try:
    cur.execute("SELECT id, name FROM collections")
    rows = cur.fetchall()
    print(f"\nCollections ({len(rows)}):")
    for r in rows:
        print(f"  id={r[0]}  name={r[1]}")
except Exception as e:
    print(f"No collections table: {e}")

# Count embeddings if table exists
for t in tables:
    try:
        cur.execute(f"SELECT count(*) FROM [{t}]")
        cnt = cur.fetchone()[0]
        print(f"  {t}: {cnt:,} rows")
    except Exception as e:
        print(f"  {t}: error - {e}")

conn.close()
