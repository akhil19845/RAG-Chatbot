# app/indexer.py
import os
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredPDFLoader
import json
from .config import (
    DATA_FILE,
    CHROMA_PATH,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def process_unstructured_elements(docs: List[Document]) -> List[Document]:

    print("\nProcessing Unstructured elements for header injection...")

    final_content_docs: List[Document] = []
    current_header: Optional[str] = None

    for doc in docs:

        raw_category = (doc.metadata or {}).get("category") or (doc.metadata or {}).get("Category") or ""
        
        category = raw_category.strip().title() if raw_category else ""

        text = (doc.page_content or "").strip()
        if not text:
            continue

        if category == "Title":
            current_header = text
            continue

        if category not in ("Narrativetext", "Listitem", "Table"):

            continue

        # Noise filter
        if len(text) < 50:
            continue

        # enriched_text = f"{current_header}\n\n{text}" if current_header else text
        enriched_text  = text
        doc.page_content = enriched_text

        allowed_keys = {"parent_id", "element_id", "page_number", "source"}
        cleaned_meta = {k: v for k, v in (doc.metadata or {}).items() if k in allowed_keys}
        cleaned_meta["category"] = category
        if current_header:
            cleaned_meta["section_header"] = current_header

        doc.metadata = cleaned_meta
        final_content_docs.append(doc)

    print(f"Metadata Filtered to {len(final_content_docs)} content blocks ready for splitting.")
    return final_content_docs


def chunks_to_serializable(final_chunks: List, include_text_key: str = "page_content"):

    out = []
    for i, doc in enumerate(final_chunks):
        text = getattr(doc, include_text_key, None) or getattr(doc, "text", None) or str(doc)
        meta = getattr(doc, "metadata", None) or {}
        if "chunk_index" not in meta:
            meta = dict(meta)
            meta.setdefault("chunk_index", i)
        out.append({"document": text, "metadata": meta})    
    return out


def build_index(data_file: str = None,persist_directory: str = None,embedding_model_name: str = None,chunk_size: int = None,chunk_overlap: int = None,) -> Chroma:

    data_file = str(data_file or DATA_FILE)
    persist_directory = str(persist_directory or CHROMA_PATH)
    embedding_model_name = embedding_model_name or EMBEDDING_MODEL_NAME
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP

    print("\nBuilding index")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Input PDF file not found at: {data_file}")

    print("Loading PDF elements via UnstructuredPDFLoader")
    loader = UnstructuredPDFLoader(data_file, mode="elements")
    raw_docs = loader.load()

    processed_docs = process_unstructured_elements(raw_docs)

    print(f"Splitting into chunks (size={chunk_size}, overlap={chunk_overlap})")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", " "],
    )
    final_chunks = text_splitter.split_documents(processed_docs)

    # out = chunks_to_serializable(final_chunks)

    # out_path = "chroma_chunks_dump.json"
    # with open(out_path, "w", encoding="utf-8") as f:
    #     json.dump(out, f, ensure_ascii=False, indent=2)

    # print(f"Wrote {len(out)} chunks to {out_path} \n")



    print(f"Creating embeddings with model: {embedding_model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    print(f"Persisting Chroma DB to: {persist_directory}")
    vectorstore = Chroma.from_documents(
        final_chunks,
        embeddings,
        persist_directory=persist_directory,
    )

    print(f"Index build complete. Indexed {len(final_chunks)} chunks.")
    return vectorstore


def setup_standard_rag_retriever(data_file: str = None,persist_directory: str = None,embedding_model_name: str = None,k: int = 4,build_if_missing: bool = True,):

    data_file = str(data_file or DATA_FILE)
    persist_directory = str(persist_directory or CHROMA_PATH)
    embedding_model_name = embedding_model_name or EMBEDDING_MODEL_NAME

    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        if build_if_missing:
            print("Persisted Chroma DB missing or empty — building index now.")
            build_index(data_file=data_file, persist_directory=persist_directory, embedding_model_name=embedding_model_name)
        else:
            print(f"Persisted Chroma DB not found at '{persist_directory}'. Set build_if_missing=True to auto-build.")
            return None

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    print(f"Created retriever (k={k}) from persisted Chroma at: {persist_directory}")
    return retriever


if __name__ == "__main__":
    try:
        print("Running app/indexer.py as script — building index now.")
        build_index()
    except Exception as e:
        print("Index build failed:", str(e))