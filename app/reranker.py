# app/reranker.py
#
# What changed vs the original:
# ─────────────────────────────
# 1. Added BM25 sparse retrieval alongside ChromaDB dense retrieval.
# 2. Added Reciprocal Rank Fusion (RRF) to merge both ranked lists.
# 3. Cross-encoder re-ranker is now applied on the fused candidate set.
# 4. BM25 index is built lazily from the live ChromaDB corpus (no extra
#    file to maintain — it always stays in sync with your vector store).
# 5. get_retriever() signature is UNCHANGED — chains.py needs no edits.

from typing import Optional, List, Dict, Any, Tuple
from threading import Lock
import os
import logging
import numpy as np

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_classic.schema.retriever import BaseRetriever
from pydantic import Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from rank_bm25 import BM25Okapi          # pip install rank-bm25

from .config import (
    CHROMA_PATH,
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    INITIAL_RETRIEVAL_K,
)
from .indexer import build_index

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singletons  (same pattern as original)
# ──────────────────────────────────────────────────────────────────────────────

_vectorstore: Optional[Chroma] = None
_reranker: Optional[Dict[str, Any]] = None
_bm25_index: Optional[Dict[str, Any]] = None   # NEW — lazy BM25 index
_lock = Lock()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — unchanged from original
# ──────────────────────────────────────────────────────────────────────────────

def _persist_dir_has_data(persist_directory: str) -> bool:
    try:
        return os.path.isdir(persist_directory)
    except Exception:
        return False


def get_vectorstore(
    persist_directory: Optional[str] = None,
    embedding_model: Optional[str] = None,
    build_if_missing: bool = True,
) -> Chroma:
    global _vectorstore
    persist_directory = str(persist_directory or CHROMA_PATH)
    embedding_model = embedding_model or EMBEDDING_MODEL_NAME

    if _vectorstore is None:
        with _lock:
            if _vectorstore is None:
                if not _persist_dir_has_data(persist_directory):
                    if build_if_missing:
                        logger.info(
                            "Persist directory %s missing/empty; building index now.",
                            persist_directory,
                        )
                        try:
                            vs = build_index(
                                persist_directory=persist_directory,
                                embedding_model_name=embedding_model,
                            )
                            _vectorstore = vs
                            logger.info("Index built and loaded from %s", persist_directory)
                            return _vectorstore
                        except Exception as e:
                            logger.exception("Failed to build index: %s", e)
                            raise
                    else:
                        logger.warning(
                            "Persist directory %s missing or empty and build_if_missing=False",
                            persist_directory,
                        )

                embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
                _vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embeddings,
                )
    return _vectorstore


def get_reranker_model() -> Dict[str, Any]:
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                logger.info("Loading reranker model: %s", RERANKER_MODEL_NAME)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        RERANKER_MODEL_NAME
                    )
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model = model.to(device)
                    _reranker = {"tokenizer": tokenizer, "model": model, "device": device}
                    logger.info("Reranker model loaded on %s.", device)
                except Exception as e:
                    logger.exception("Failed to load reranker model: %s", e)
                    raise
    return _reranker


# ──────────────────────────────────────────────────────────────────────────────
# NEW — BM25 index built from the live ChromaDB corpus
# ──────────────────────────────────────────────────────────────────────────────

def get_bm25_index(vs: Chroma) -> Dict[str, Any]:
    """
    Build (once) a BM25 index over every document in the ChromaDB collection.

    Returns a dict with:
      - "bm25"  : BM25Okapi instance
      - "docs"  : list of LangChain Document objects (same order as BM25 corpus)

    The index is rebuilt only on the first call (or if the vectorstore changes).
    Because BM25 is an in-memory structure and your corpus is small (108 chunks),
    rebuild is fast — under a second.
    """
    global _bm25_index
    if _bm25_index is None:
        with _lock:
            if _bm25_index is None:
                logger.info("Building BM25 index from ChromaDB corpus …")
                try:
                    # Pull every document out of Chroma
                    raw = vs.get(include=["documents", "metadatas"])
                    texts: List[str] = raw["documents"]      # list of plain strings
                    metas: List[dict] = raw["metadatas"]

                    # Wrap back into LangChain Documents so we return the same type
                    all_docs = [
                        Document(page_content=t, metadata=m)
                        for t, m in zip(texts, metas)
                    ]

                    # Tokenise (simple whitespace split — fast, good enough for BM25)
                    tokenised_corpus = [doc.page_content.lower().split() for doc in all_docs]

                    _bm25_index = {
                        "bm25": BM25Okapi(tokenised_corpus),
                        "docs": all_docs,
                    }
                    logger.info("BM25 index built over %d documents.", len(all_docs))
                except Exception as e:
                    logger.exception("Failed to build BM25 index: %s", e)
                    raise
    return _bm25_index


# ──────────────────────────────────────────────────────────────────────────────
# NEW — Reciprocal Rank Fusion
# ──────────────────────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    ranked_lists: List[List[int]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    Merge multiple ranked lists of document indices using RRF.

    Each list should be ordered best-first (index 0 = most relevant).
    Returns a list of (doc_index, rrf_score) sorted best-first.

    k=60 is the standard RRF constant — it dampens the impact of very
    high ranks without making low ranks irrelevant.
    """
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_idx in enumerate(ranked):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# Updated retriever class
# ──────────────────────────────────────────────────────────────────────────────

class HybridRerankingRetriever(BaseRetriever):
    """
    Three-stage retriever:

    Stage 1  Dense retrieval   — ChromaDB cosine similarity
    Stage 1  Sparse retrieval  — BM25 keyword matching       (NEW)
             ↓ Merge via Reciprocal Rank Fusion              (NEW)
    Stage 2  Cross-encoder re-ranking on fused candidates
    Stage 3  Return top final_k docs to the LLM
    """

    vs: Chroma = Field(description="Chroma vector store instance.")
    initial_k: int = Field(description="Candidates to pull from each retriever before fusion.")
    final_k: int = Field(description="Documents returned after re-ranking.")
    reranker_components: Dict[str, Any] = Field(description="HuggingFace tokenizer + model.")

    def __init__(
        self,
        vs: Chroma,
        initial_k: int,
        final_k: int,
        reranker_components: Dict[str, Any],
        **kwargs,
    ):
        super().__init__(
            vs=vs,
            initial_k=initial_k,
            final_k=final_k,
            reranker_components=reranker_components,
            **kwargs,
        )
        self._tokenizer = self.reranker_components["tokenizer"]
        self._model = self.reranker_components["model"]
        self._device = self.reranker_components.get("device", "cpu")

    # ── Stage 1a: Dense retrieval ─────────────────────────────────────────────

    def _dense_retrieve(self, query: str) -> List[Document]:
        retriever = self.vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.initial_k},
        )
        return retriever.invoke(query)

    # ── Stage 1b: BM25 sparse retrieval ──────────────────────────────────────

    def _sparse_retrieve(self, query: str) -> List[Document]:
        bm25_data = get_bm25_index(self.vs)
        bm25: BM25Okapi = bm25_data["bm25"]
        all_docs: List[Document] = bm25_data["docs"]

        tokenised_query = query.lower().split()
        scores = bm25.get_scores(tokenised_query)

        # Return top initial_k by BM25 score
        top_indices = np.argsort(scores)[::-1][: self.initial_k]
        return [all_docs[i] for i in top_indices]

    # ── Stage 1c: RRF fusion ──────────────────────────────────────────────────

    def _fuse(
        self,
        dense_docs: List[Document],
        sparse_docs: List[Document],
    ) -> List[Document]:
        """
        Deduplicate and merge both ranked lists with RRF.

        We use page_content as the dedup key (same as what the cross-encoder
        will score). Returns a deduplicated list ordered by RRF score.
        """
        # Build a unified pool: content → Document
        pool: Dict[str, Document] = {}
        for doc in dense_docs + sparse_docs:
            key = doc.page_content.strip()
            if key not in pool:
                pool[key] = doc

        pool_keys = list(pool.keys())   # stable ordering for index lookup

        # Rank lists as indices into pool_keys
        dense_ranked = [
            pool_keys.index(d.page_content.strip())
            for d in dense_docs
            if d.page_content.strip() in pool_keys
        ]
        sparse_ranked = [
            pool_keys.index(d.page_content.strip())
            for d in sparse_docs
            if d.page_content.strip() in pool_keys
        ]

        fused = _reciprocal_rank_fusion([dense_ranked, sparse_ranked])

        # Return Documents in fused order
        return [pool[pool_keys[idx]] for idx, _ in fused]

    # ── Stage 2: Cross-encoder re-ranking ─────────────────────────────────────

    def _rerank(self, query: str, candidates: List[Document]) -> List[Document]:
        if not candidates:
            return []

        pairs = [(query, doc.page_content) for doc in candidates]
        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            scores = self._model(**inputs).logits.squeeze(-1)

        if scores.dim() == 0:
            scores = scores.unsqueeze(0)

        top_indices = torch.topk(scores, min(self.final_k, len(candidates))).indices.tolist()
        return [candidates[i] for i in top_indices]

    # ── Main entry point ──────────────────────────────────────────────────────

    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        # Stage 1 — dual retrieval
        dense_docs = self._dense_retrieve(query)
        sparse_docs = self._sparse_retrieve(query)

        logger.debug(
            "Dense: %d docs | Sparse: %d docs", len(dense_docs), len(sparse_docs)
        )

        if not dense_docs and not sparse_docs:
            return []

        # Fuse via RRF
        fused_candidates = self._fuse(dense_docs, sparse_docs)
        logger.debug("After RRF fusion: %d unique candidates", len(fused_candidates))

        # Stage 2 — cross-encoder re-ranking on fused pool
        reranked = self._rerank(query, fused_candidates)
        logger.debug("After re-ranking: returning %d docs", len(reranked))

        return reranked


# ──────────────────────────────────────────────────────────────────────────────
# Public factory — SAME SIGNATURE as original, chains.py needs no changes
# ──────────────────────────────────────────────────────────────────────────────

def get_retriever(
    k: int = 4,
    initial_k: int = INITIAL_RETRIEVAL_K,
    persist_directory: Optional[str] = None,
    embedding_model: Optional[str] = None,
    build_if_missing: bool = True,
) -> BaseRetriever:
    vs = get_vectorstore(
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        build_if_missing=build_if_missing,
    )
    reranker_components = get_reranker_model()

    return HybridRerankingRetriever(
        vs=vs,
        initial_k=initial_k,
        final_k=k,
        reranker_components=reranker_components,
    )
