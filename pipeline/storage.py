import os
import json
import hashlib
import openai
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# === Azure OpenAI Setup ===
openai.api_key = os.getenv("AZURE_API_KEY")
openai.azure_endpoint = os.getenv("AZURE_API_BASE")
openai.api_version = "2023-06-01-preview"
embedding_model = "azure.text-embedding-3-small"

# === File Paths ===
VECTOR_DIR = "data/output/vector_store"
os.makedirs(VECTOR_DIR, exist_ok=True)

FINE_TUNE_FILE = "data/output/fine_tune.jsonl"
DISCARD_LOG = "data/output/discard.log"
VECTOR_JSON = os.path.join(VECTOR_DIR, "embedded_data.json")
VECTOR_INDEX = os.path.join(VECTOR_DIR, "faiss_index.index")

# === Save Final Chunk ===
def save_chunk(chunk: str, decision: str, output_dir: str):
    if decision == "fine_tune":
        save_to_jsonl(chunk, FINE_TUNE_FILE)
    elif decision == "vector_only":
        save_to_vector_db(chunk)
    elif decision == "discard":
        log_discard(chunk, DISCARD_LOG)

# === Fine-tune JSONL ===
def save_to_jsonl(text: str, file_path: str):
    with open(file_path, "a", encoding="utf-8") as f:
        json.dump({"text": text}, f)
        f.write("\n")
    print("✅ Saved to fine_tune.jsonl")

# === Discard Log ===
def log_discard(text: str, file_path: str):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"---\n{text.strip()}\n---\n")
    print("❌ Discarded and logged")

# === Vector Store (Manual Azure API + FAISS) ===
def save_to_vector_db(text: str):
    try:
        response = openai.embeddings.create(
            model=embedding_model,
            input=text
        )
        embedding = response.data[0].embedding
    except Exception as e:
        print(f"⚠️ Embedding failed: {e}")
        return

    doc_id = hashlib.sha256(text.encode()).hexdigest()

    # Load existing data
    if os.path.exists(VECTOR_JSON):
        with open(VECTOR_JSON, "r", encoding="utf-8") as f:
            embedded_data = json.load(f)
    else:
        embedded_data = []

    embedded_data.append({
        "id": doc_id,
        "text": text,
        "embedding": embedding
    })

    with open(VECTOR_JSON, "w", encoding="utf-8") as f:
        json.dump(embedded_data, f, ensure_ascii=False, indent=2)
    print("🔍 Saved chunk to embedded_data.json")

    # Save to FAISS
    save_to_faiss_index(embedded_data)

def save_to_faiss_index(embedded_data):
    if not embedded_data:
        return

    dim = len(embedded_data[0]["embedding"])
    index = faiss.IndexFlatL2(dim)

    embeddings_array = np.array([e["embedding"] for e in embedded_data]).astype("float32")
    index.add(embeddings_array)

    faiss.write_index(index, VECTOR_INDEX)
    print(f"📦 Saved FAISS index with {len(embedded_data)} vectors")


