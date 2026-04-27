"""Reset ChromaDB - delete the empty evidence_store so ingest starts cleanly."""
import chromadb
import os

db_dir = os.path.join(os.path.dirname(__file__), "chromadb")
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
