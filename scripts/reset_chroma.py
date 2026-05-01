# reset_chroma.py — Stage 5 utility | Module C (Agent): Deletes any empty ChromaDB collections so re-ingest can start cleanly.
#
# Input:  chromadb/
# Output: chromadb/ (empty collections removed)
import chromadb
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_dir = os.path.join(_ROOT, "chromadb")
client = chromadb.PersistentClient(path=db_dir)

for col in client.list_collections():
    name = col.name
    count = col.count()
    print(f"  Collection '{name}': {count:,} documents")
    if count == 0:
        client.delete_collection(name)
        print(f"    -> Deleted (was empty)")
    else:
        print(f"    -> Kept")

print("\nDone. You can now run: python -m src.agent.ingest")
