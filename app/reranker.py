# app/retriever.py
from typing import Optional, List, Dict, Any
from threading import Lock
import os
import logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_classic.schema.retriever import BaseRetriever
from pydantic import Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from .config import CHROMA_PATH, EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, INITIAL_RETRIEVAL_K
from .indexer import build_index 
logger = logging.getLogger(__name__)


_vectorstore: Optional[Chroma] = None
_reranker: Optional[Dict[str, Any]] = None
_lock = Lock()


def _persist_dir_has_data(persist_directory: str) -> bool:
    try:
        return os.path.isdir(persist_directory)
    except Exception:
        return False


def get_vectorstore(persist_directory: Optional[str] = None, embedding_model: Optional[str] = None, build_if_missing: bool = True) -> Chroma:
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


def get_reranker_model() -> Dict[str, Any]:

    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                logger.info("Loading Reranker Model: %s", RERANKER_MODEL_NAME)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
                    model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
                    _reranker = {"tokenizer": tokenizer, "model": model}
                    logger.info("Reranker Model loaded successfully.")
                except Exception as e:
                    logger.exception("Failed to load Reranker Model: %s", e)
                    raise
    return _reranker

class TwoStageRerankingRetriever(BaseRetriever):

    vs: Chroma = Field(description="The Chroma vector store instance.")
    initial_k: int = Field(description="The initial number of documents retrieved from the vector store.")
    final_k: int = Field(description="The final number of documents returned after re-ranking.")
    reranker_components: Dict[str, Any] = Field(description="The loaded HuggingFace tokenizer and model.")
 
    def __init__(self, vs: Chroma, initial_k: int, final_k: int, reranker_components: Dict[str, Any], **kwargs):
        super().__init__(
            vs=vs,
            initial_k=initial_k,
            final_k=final_k,
            reranker_components=reranker_components,
            **kwargs
        ) 

        self._tokenizer = self.reranker_components["tokenizer"]
        self._model = self.reranker_components["model"]

    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:

        retriever = self.vs.as_retriever(search_type="similarity", search_kwargs={"k": self.initial_k})
        candidate_docs: List[Document] = retriever.invoke(query, k=self.initial_k)
        
        if not candidate_docs:
            return []

        pairs = [(query, doc.page_content) for doc in candidate_docs]
        inputs = self._tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            scores = self._model(**inputs).logits.squeeze()

        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
            
        top_k_indices = torch.topk(scores, min(self.final_k, len(candidate_docs))).indices.tolist()
        reranked_docs = [candidate_docs[i] for i in top_k_indices]
        
        return reranked_docs


def get_retriever(k: int = 4, initial_k: int = INITIAL_RETRIEVAL_K, persist_directory: Optional[str] = None, embedding_model: Optional[str] = None, build_if_missing: bool = True) -> BaseRetriever:

    vs = get_vectorstore(persist_directory=persist_directory, embedding_model=embedding_model, build_if_missing=build_if_missing)
    reranker_components = get_reranker_model()

    return TwoStageRerankingRetriever(
        vs=vs,
        initial_k=initial_k,
        final_k=k,
        reranker_components=reranker_components
    )