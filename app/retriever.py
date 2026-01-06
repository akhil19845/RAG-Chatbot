# app/retriever.py
from typing import Optional
from threading import Lock
import os
import logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from .config import CHROMA_PATH, EMBEDDING_MODEL_NAME
from .indexer import build_index

logger = logging.getLogger(__name__)

_vectorstore: Optional[Chroma] = None
_lock = Lock()

def _persist_dir_has_data(persist_directory: str) -> bool:
    try:
        return os.path.isdir(persist_directory) and any(os.scandir(persist_directory))
    except Exception:
        return False

def get_vectorstore(persist_directory: Optional[str] = None, embedding_model: Optional[str] = None, build_if_missing: bool = True,) -> Chroma:

    global _vectorstore
    persist_directory = str(persist_directory or CHROMA_PATH)
    embedding_model = embedding_model or EMBEDDING_MODEL_NAME

    if _vectorstore is None:
        with _lock:
            if _vectorstore is None:
                if not _persist_dir_has_data(persist_directory):
                    if build_if_missing:
                        logger.info("Persist directory %s missing/empty; building index now.", persist_directory)
                        try:
                            vs = build_index(persist_directory=persist_directory, embedding_model_name=embedding_model)
                            _vectorstore = vs
                            logger.info("Index built and loaded from %s", persist_directory)
                            return _vectorstore
                        except Exception as e:
                            logger.exception("Failed to build index: %s", e)
                            raise
                    else:
                        logger.warning("Persist directory %s missing or empty and build_if_missing=False", persist_directory)

                embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
                _vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return _vectorstore

def get_retriever(k: int = 4, persist_directory: Optional[str] = None, embedding_model: Optional[str] = None, build_if_missing: bool = True):

    vs = get_vectorstore(persist_directory=persist_directory, embedding_model=embedding_model, build_if_missing=build_if_missing)
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever