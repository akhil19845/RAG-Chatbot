# Embed_Chunks.py
#
# Embeds the structured chunks.json (produced by rag_chunker.py) into ChromaDB.
#
# Metadata stored per chunk (all retrievable at query time):
# ┌─────────────────┬──────────────────────────────────────────────────────┐
# │ Field           │ Description                                          │
# ├─────────────────┼──────────────────────────────────────────────────────┤
# │ chunk_id        │ Unique integer ID                                    │
# │ section_title   │ Nearest heading text — shown as source in answers    │
# │ heading_level   │ 1=top section, higher=deeper sub-section             │
# │ page_start      │ First page this chunk appears on                     │
# │ page_end        │ Last page this chunk spans                           │
# │ element_types   │ What the chunk contains (paragraph/list/table/etc.)  │
# │ doc_title       │ Document title                                       │
# │ doc_author      │ Document author                                      │
# │ doc_pages       │ Total pages in the source document                   │
# └─────────────────┴──────────────────────────────────────────────────────┘
#
# NOTE: ChromaDB only accepts str/int/float/bool metadata values.
#       Lists (element_types, source_ids) are stored as comma-separated strings
#       and can be split back on retrieval.

import json
import logging
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
# Swap these out for your .config imports when integrating into the app:
#
# from .config import CHUNKS_FILE, CHROMA_PATH, EMBEDDING_MODEL_NAME
#
# CHUNKS_FILE        = "chunks.json"
# CHROMA_PATH        = "./chroma_db"
# EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

CHUNKS_FILE = "/home/ubuntu/RAG-Chatbot/Chunks/chunks.json"
CHROMA_PATH = "/home/ubuntu/RAG-Chatbot/chroma_db_ml"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Other solid options:
#   "sentence-transformers/all-MiniLM-L6-v2"  — fast, lightweight
#   "intfloat/e5-large-v2"                    — strong for technical docs
# ─────────────────────────────────────────────────────────────────────────────


def load_chunks(chunks_file: str) -> list[dict]:
    """Load chunks from the JSON file produced by rag_chunker.py."""
    path = Path(chunks_file)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info("Loaded %d chunks from %s", len(chunks), chunks_file)
    return chunks


def chunk_to_document(chunk: dict) -> Document:
    """
    Convert a single chunk dict to a LangChain Document.

    page_content  → the text that gets embedded and retrieved
    metadata      → everything stored alongside for reference at query time

    ChromaDB restriction: metadata values must be scalar (str/int/float/bool).
    Lists are serialised as comma-separated strings.
    """
    inner_meta: dict = chunk.get("metadata", {})

    # element_types is a list  e.g. ["paragraph", "list", "table"]
    element_types: list = chunk.get("element_types", [])
    # source_ids is a list of original element IDs from the parsed PDF
    source_ids: list = chunk.get("source_ids", [])

    metadata = {
        # ── Chunk identity ────────────────────────────────────────────────
        "chunk_id":      chunk["chunk_id"],

        # ── Section context (most useful for citing sources in answers) ───
        "section_title": chunk.get("section_title", ""),
        "heading_level": chunk.get("heading_level", 0),

        # ── Page location (for "see page X" style citations) ─────────────
        "page_start":    chunk.get("page_start", 0),
        "page_end":      chunk.get("page_end", 0),

        # ── Content composition (useful for filtering, e.g. tables only) ─
        # Stored as comma-separated string because ChromaDB requires scalars
        "element_types": ", ".join(element_types),
        "has_table":     "table" in element_types,      # bool — easy to filter
        "has_list":      "list" in element_types,
        "has_code":      "code" in element_types,

        # ── Source document info ──────────────────────────────────────────
        "doc_title":     inner_meta.get("doc_title", ""),
        "doc_author":    inner_meta.get("doc_author", ""),
        "doc_pages":     inner_meta.get("doc_pages", 0),

        # ── Traceability back to the original parsed JSON ─────────────────
        "source_ids":    ", ".join(str(i) for i in source_ids),
    }

    return Document(page_content=chunk["content"], metadata=metadata)


def build_vectorstore(
    chunks_file: str = CHUNKS_FILE,
    persist_directory: str = CHROMA_PATH,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
) -> Chroma:
    """
    Load chunks → build LangChain Documents → embed → persist to ChromaDB.
    Returns the Chroma vectorstore instance.
    """
    # 1. Load chunks
    chunks = load_chunks(chunks_file)

    # 2. Filter out empty content chunks (e.g. title-only sections)
    before = len(chunks)
    chunks = [c for c in chunks if c.get("content", "").strip()]
    skipped = before - len(chunks)
    if skipped:
        logger.warning("Skipped %d empty chunks.", skipped)

    # 3. Convert to LangChain Documents
    docs = [chunk_to_document(c) for c in chunks]
    logger.info("Prepared %d documents for embedding.", len(docs))

    # Log a sample so you can verify metadata looks right
    sample = docs[5] if len(docs) > 5 else docs[0]
    logger.info("Sample document metadata: %s", sample.metadata)
    logger.info("Sample content preview : %s", sample.page_content[:120])

    # 4. Load embedding model
    logger.info("Loading embedding model: %s", embedding_model_name)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"},       # change to "cuda" if GPU available
        encode_kwargs={"normalize_embeddings": True},  # needed for cosine similarity
    )

    # 5. Embed and persist
    logger.info("Embedding and persisting to ChromaDB at: %s", persist_directory)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    logger.info("✅ ChromaDB created with %d documents at %s", len(docs), persist_directory)
    return vectorstore


def verify_vectorstore(persist_directory: str = CHROMA_PATH, embedding_model_name: str = EMBEDDING_MODEL_NAME):
    """
    Quick sanity check — reload the store and run a test query.
    Shows exactly what metadata your retriever will see at query time.
    """
    logger.info("Verifying ChromaDB at %s …", persist_directory)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    test_query = "How does P4 handle packet parsing?"
    results = vs.similarity_search(test_query, k=3)

    print(f"\n{'='*60}")
    print(f"Test query: '{test_query}'")
    print(f"{'='*60}")
    for i, doc in enumerate(results, 1):
        m = doc.metadata
        print(f"\n── Result {i} ──────────────────────────────────────────")
        print(f"  Section : {m.get('section_title')}  (H{m.get('heading_level')})")
        print(f"  Pages   : {m.get('page_start')} – {m.get('page_end')}")
        print(f"  Types   : {m.get('element_types')}")
        print(f"  Has table: {m.get('has_table')}  |  Has list: {m.get('has_list')}")
        print(f"  Doc     : {m.get('doc_title')} by {m.get('doc_author')}")
        print(f"  Content : {doc.page_content[:200]} …")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    build_vectorstore()
    verify_vectorstore()
