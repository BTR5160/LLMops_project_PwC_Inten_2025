import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.corpus import stopwords
import hashlib

# === Step 1: Clean ===
def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)  # Remove HTML
    text = re.sub(r"\s+", " ", text)     # Collapse whitespace
    text = text.strip()
    return text

# === Step 2: Chunk ===
def chunk_text(text: str, chunk_size=800, chunk_overlap=100) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

# === Step 3: Deduplicate ===
def deduplicate_chunks(chunks: list) -> list:
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        h = hashlib.sha256(chunk.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_chunks.append(chunk)
    return unique_chunks

# === Main Entry ===
def preprocess_text(raw_text: str) -> list:
    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned)
    deduped = deduplicate_chunks(chunks)
    return deduped
