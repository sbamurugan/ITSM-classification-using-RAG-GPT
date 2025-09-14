import os
import pandas as pd
import requests
import time
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain_chroma import Chroma
import uuid

# --------------------------
# CONFIG
# --------------------------
CORE42_API_KEY = os.getenv("CORE42_API_KEY", "3367724bb8cc48829031ad811d047cad")
CORE42_BASE = "https://api.core42.ai/v1"
os.environ["OPENAI_API_KEY"] = CORE42_API_KEY  # keep compatibility

file_path = os.path.join(os.getcwd(), "Helpdesk", "Tickets 2025 Jan-Jun.xlsx")
PERSIST_DIR = "./helpdesk4_chromadb_store"
COLLECTION_NAME = "helpdesk_requests4"
BATCH_SIZE = 500  # adjust as needed

headers = {"Content-Type": "application/json", "Authorization": f"Bearer {CORE42_API_KEY}"}

# --------------------------
# LOAD EXCEL
# --------------------------
df = pd.read_excel(file_path, header=1)
total_rows = len(df)
print(f"Total rows in Excel: {total_rows}")

# --------------------------
# PREVIEW METADATA BEFORE EMBEDDING
# --------------------------
print("\n👀 Previewing first 5 rows metadata before embedding:")
for i, row in df.head(5).iterrows():
    meta = {
        "text": str(row.get("Subject", "")),
        "Category": str(row.get("Category", "None")),
        "Subcategory": str(row.get("Subcategory", "None")),
        "Group": str(row.get("Group", "None")),
        "Priority": str(row.get("Priority", "None")),
        "Service Category": str(row.get("Service Category", "Others")),
        "Item": str(row.get("Item", "-"))
    }
    print(f"\n--- Row {i} Metadata ---")
    for k, v in meta.items():
        print(f"{k}: {v}")

confirm = input("\nDo you want to continue with embeddings and storing in ChromaDB? (yes/no): ").strip().lower()
if confirm != "yes":
    print("Operation cancelled.")
    exit(0)

# --------------------------
# INITIALIZE CHROMA DB
# --------------------------
db = Chroma(persist_directory=PERSIST_DIR, collection_name=COLLECTION_NAME, embedding_function=None)

# Reset collection if exists
if db._collection.count() > 0:
    print(f"Existing collection has {db._collection.count()} documents.")
    confirm_reset = input("Do you want to delete and reset the collection? (yes/no): ").strip().lower()
    if confirm_reset == "yes":
        db.delete_collection()
        db.reset_collection()
        print("Collection reset successfully!")

# --------------------------
# HELPER FUNCTION: GET EMBEDDINGS
# --------------------------
def get_embedding(text: str, retries=3, backoff=5):
    payload = {"model": "text-embedding-3-large", "input": text}
    for attempt in range(retries):
        try:
            r = requests.post(f"{CORE42_BASE}/embeddings", headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            return r.json()["data"][0]["embedding"]
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")
            time.sleep(backoff * (attempt+1))
    raise RuntimeError(f"❌ Failed to get embedding for text: {text[:50]}...")

# --------------------------
# PRELOAD EXISTING DOCUMENTS TO SKIP DUPLICATES
# --------------------------
stored_texts = set()
if db._collection.count() > 0:
    existing_docs = db._collection.get(include=["documents"])["documents"]
    for doc_list in existing_docs:
        stored_texts.update(doc_list)

# --------------------------
# PROCESS IN BATCHES
# --------------------------
print("\n🔄 Generating embeddings and storing in ChromaDB in batches...")

for start_idx in range(0, total_rows, BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, total_rows)
    batch_df = df.iloc[start_idx:end_idx]

    docs, embeddings, ids = [], [], []

    for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Processing rows {start_idx}-{end_idx-1}"):
        text = str(row.get("Subject", "")).strip()
        group_val = str(row.get("Group", "None")).strip().lower()

        # --- Skip empty, duplicate, or Servicedesk group ---
        if not text or text in stored_texts or group_val == "SERVICE DESK":
            continue

        # Build metadata
        meta = {
            "text": text,
            "Category": str(row.get("Category", "None")),
            "Subcategory": str(row.get("Subcategory", "None")),
            "Group": str(row.get("Group", "None")),
            "Priority": str(row.get("Priority", "None")),
            "Service Category": str(row.get("Service Category", "Others")),
            "Item": str(row.get("Item", "-"))
        }

        docs.append(Document(page_content=text, metadata=meta))
        embeddings.append(get_embedding(text))
        ids.append(str(uuid.uuid4()))
        stored_texts.add(text)  # prevent duplicates

    if docs:
        db._collection.add(
            ids=ids,
            documents=[d.page_content for d in docs],
            metadatas=[d.metadata for d in docs],
            embeddings=embeddings
        )
        print(f"✅ Stored batch {start_idx}-{end_idx-1} in ChromaDB")
    else:
        print(f"⚠️ No new documents to store in batch {start_idx}-{end_idx-1}")

print("\n🎉 All batches processed successfully!")

# --------------------------
# VERIFY FIRST 3 DOCUMENTS
# --------------------------
stored_docs = db._collection.get(include=["metadatas", "documents"], n_results=3)["metadatas"]
for i, doc in enumerate(stored_docs, 1):
    print(f"\n--- Stored Document {i} ---")
    for k, v in doc.items():
        print(f"{k}: {v}")
