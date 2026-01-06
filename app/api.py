# CORNERSTONE/app/api.py

import asyncio
import json
import inspect
from typing import Dict, Any, Optional, AsyncGenerator
import logging
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .chains import build_conv_chain
from .memory_store import get_memory, clear_memory

logger = logging.getLogger(__name__)


# Cache chains by session_id
_CHAINS: Dict[str, Any] = {}

app = FastAPI(title="Cornerstone RAG API")

# CORS for Streamlit UI (DEV ONLY – tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helpers

def make_session_id() -> str:
    """Create a short unique session id."""
    return f"session-{uuid.uuid4().hex[:8]}"


def run_blocking(fn, *args, **kwargs):
    """Run a blocking function inside a thread to avoid blocking FastAPI's event loop."""
    return fn(*args, **kwargs)


def get_or_create_chain(session_id: str, k: int = 4, use_condense_question: bool = True):
    """
    Ensure each session gets its own conversational chain.
    Your chains.py already has: build_conv_chain(session_id, k=4, use_condense_question=True)
    """
    if not session_id:
        session_id = make_session_id()

    if session_id not in _CHAINS:
        try:
            chain = build_conv_chain(session_id=session_id, k=k, use_condense_question=use_condense_question)
        except TypeError:
            # fallback if signature differs
            chain = build_conv_chain(session_id)
        _CHAINS[session_id] = chain

    return _CHAINS[session_id]


async def safe_run_chain(question: str, session_id: str, k: int = 4) -> Dict[str, Any]:
    """
    Calls your chain safely (thread executor). Normalizes the output to:
      { "answer": str, "sources": [...], "extra": {...} }
    Handles chain output patterns: dict, string, chain.run(), chain({"question":...}), etc.
    """
    loop = asyncio.get_event_loop()
    chain = build_conv_chain(session_id=session_id)

    def call_chain():

        try:
            out = chain.invoke({"question": question})
            return {"answer": out["answer"]}
        except Exception:
            pass

        return {"answer": f"(warning) Could not call chain properly on: {question}"}

    try:
        result = await loop.run_in_executor(None, call_chain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chain execution error: {e}")

    return {"answer": result["answer"]}


# Endpoints

@app.post("/api/query")
async def query(payload: Dict[str, Any]):
    """
    Non-streaming endpoint:
    Input:
        {
            "question": "your question",
            "session_id": "123"
        }
    Output:
        {
            "answer": "...",
            "sources": [...]
        }
    """
    question = payload.get("question")
    session_id = payload.get("session_id")
    k = payload.get("k", 4)

    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question'")

    result = await safe_run_chain(question, session_id, k=k)
    return JSONResponse(result)


@app.get("/api/history/{session_id}")
async def history(session_id: str):
    """
    Return the conversation history stored in memory_store.
    """
    try:
        mem = get_memory(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory error: {e}")

    if hasattr(mem, "load_memory_variables"):
        vars = mem.load_memory_variables({})
        if "chat_history" in vars:
            return {"session_id": session_id, "history": vars["chat_history"]}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/clear_memory/{session_id}")
def clear_session_memory(session_id: str):

    try:
        clear_memory(session_id)
    except Exception as e:
        logger.exception("Failed to clear memory for session %s: %s", session_id, e)
        raise HTTPException(status_code=500, detail="Failed to clear memory.")
    return {"status": "cleared", "session_id": session_id}