from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pathlib import Path
import json
# from .config import (
#     CHUNKS_FILE,
#     CHROMA_PATH,
#     EMBEDDING_MODEL_NAME
# )

CHUNKS_FILE = "/home/ubuntu/Conversational-Chatbot/Chunks/chunks_with_meta.jsonl"
persist_directory = "/home/ubuntu/Conversational-Chatbot/chroma_db_ml"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"


# Load chunks
docs = []
with open(CHUNKS_FILE, "r", encoding="utf8") as f:
    for line in f:
        rec = json.loads(line)
        # meta = rec.get("metadata",{})
        # text = rec.get("text")
        meta = rec["metadata"]
        text = rec["text"]

        # slim_meta = {
        #     "doc_id": Path(meta["source"]).stem.replace(" ", "_").lower(),
        #     "source": meta.get("source"),
        #     "heading_path": meta.get("heading_path"),
        #     "chunk_index": meta.get("chunk_index"),
        #     "source_para_indices": meta.get("source_para_indices") or meta.get("para_indices"),
        #     "char_start": meta.get("char_start"),
        #     "char_end": meta.get("char_end"),
        # }
        slim_meta = {
            "source": meta.get("source"),
            "heading_path": meta.get("heading_path"),
            "chunk_index": meta.get("chunk_index"),
            "char_start": meta.get("char_start"),
            "char_end": meta.get("char_end"),            
        }

        docs.append(Document(page_content=text, metadata=slim_meta))

print(f"✅ Loaded {len(docs)} chunks")

print(f"Creating embeddings with model: {embedding_model_name}")
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

print(f"Persisting Chroma DB to: {persist_directory}")
vectorstore = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory=persist_directory,
)

print("✅ Chroma store created and persisted.")