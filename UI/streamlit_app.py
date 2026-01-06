# ui/streamlit_app.py
import streamlit as st
import requests
import time
from typing import List, Dict, Any
import uuid


# Change to your FastAPI base if needed or set via Streamlit secrets:

API_BASE = "http://localhost:8000/api"

# -------------------------
# Session-state initialization (MUST run before any widgets)
# -------------------------

def make_session_id() -> str:
    return f"session-{uuid.uuid4().hex[:8]}"

# canonical session id
if "session_id" not in st.session_state:
    st.session_state["session_id"] = make_session_id()

# flag used to copy canonical session_id -> session_id_input BEFORE widgets are created
if "sync_session_input" not in st.session_state:
    st.session_state["sync_session_input"] = False

# flag used to clear the input textbox before widget creation
if "clear_input" not in st.session_state:
    st.session_state["clear_input"] = False

# chat history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Ensure the widget-backed keys exist (optional initialization)
if "session_id_input" not in st.session_state:
    st.session_state["session_id_input"] = st.session_state["session_id"]

if "input_question" not in st.session_state:
    st.session_state["input_question"] = ""

# Perform the sync actions BEFORE any widgets are created
if st.session_state.get("sync_session_input", False):
    st.session_state["session_id_input"] = st.session_state["session_id"]
    st.session_state["sync_session_input"] = False

if st.session_state.get("clear_input", False):
    st.session_state["input_question"] = ""
    st.session_state["clear_input"] = False



st.set_page_config(page_title="Conversational RAG Application", layout="wide")



# -----------------------
# Styles (mimic screenshot)
# -----------------------
st.markdown(
    """
    <style>
    /* page background */
    .reportview-container {
        background-color: #f6f8fa;
    }
    /* heading label */
    .label {
        color: #14a248;
        font-family: 'Courier New', monospace;
        font-weight: 700;
        font-size: 40px;
        background: none;
        padding: 6px 10px;
        display:inline-block;
        border-radius: 6px;
    }
    /* blue info boxes */
    .panel {
        background: #000000; 
        color: #ffffff;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: none;
    }
    .panel--muted {
        background: #f2f6f9;
        padding: 10px;
        border-radius: 6px;
    }
    .question-box {
        background: white;
        border-radius: 8px;
        padding: 10px 12px;
        border: 1px solid #e1e6ea;
    }
    .source-list {
        background: #ffffff;
        border-radius: 6px;
        padding: 12px;
        border: 1px solid #e1e6ea;
    }
    .small-caption {
        color: #6b7280;
        font-size: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



st.markdown('<div class="label" style="display:flex; justify-content:center; align-items:center; text-align:center;">Conversational RAG Application</div>', unsafe_allow_html=True)


# Sidebar for session controls (kept minimal)
with st.sidebar:
    st.markdown(
        """
        <h1 style="
            text-align: center;
            font-size: 32px;
            margin-top: 0px;
            margin-bottom: 0px;
        ">
            Session
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Display the session ID widget (value is set via session_state sync)
    session_widget_val = st.text_input(
        "Session ID",
        key="session_id_input"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Session"):
            st.session_state.session_id = make_session_id()
            st.session_state.sync_session_input = True
            st.session_state.history = []
            st.rerun()
    with col2:
        if st.button("Load History"):
            st.session_state.session_id = st.session_state.session_id_input

            try:
                r = requests.get(f"{API_BASE}/history/{st.session_state.session_id}", timeout=8)
                r.raise_for_status()
                payload = r.json()
                hist = payload.get("history", [])


                st.session_state.history = []

                normalized = []
                for i, turn in enumerate(hist):
                    if isinstance(turn, dict):
                        raw_role = str(turn.get("role", "")).strip().lower()

                        if any(k in raw_role for k in ("user", "human", "question")):
                            role = "user"
                        elif any(k in raw_role for k in ("assistant", "bot", "system", "agent", "answer")):
                            role = "assistant"
                        else:

                            role = "user" if (i % 2 == 0) else "assistant"

                        text = turn.get("text") or turn.get("content") or turn.get("message") or ""

                        sources = turn.get("sources") or turn.get("source_documents") or turn.get("source_docs") or []
                    else:

                        role = "assistant"
                        text = str(turn)
                        sources = []
                    

                    normalized.append({"role": role, "text": text, "sources": sources})

                st.session_state.history = normalized

                st.success("History loaded")
                st.rerun()

            except Exception as e:
                st.error(f"Failed to load history: {e}")



    st.markdown("---")
    if st.button("Clear server memory"):
        try:
            r = requests.post(f"{API_BASE}/clear_memory/{st.session_state.session_id}", timeout=8)
            r.raise_for_status()
            st.session_state.history = []
            st.success("Server memory cleared")
        except Exception as e:
            st.error(f"Clear failed: {e}")


# -----------------------
# Main chat UI
# -----------------------
st.markdown("""
<div class="panel" style="display:flex; justify-content:center; align-items:center; text-align:center;">
    <label class="small-caption">I can answer questions on Machine Learning Wikipedia data</label>
</div>
""", unsafe_allow_html=True)


question = st.text_input(
    "Ask something",
    key="input_question",
    placeholder="Type your question here...",
    label_visibility="collapsed"
)


if st.button("Ask") and question.strip():
    st.session_state.history.append({"role": "user", "text": question})
    # Append assistant placeholder marked as pending (only this one will trigger a backend fetch)
    st.session_state.history.append({"role": "assistant", "text": "_thinking..._", "sources": [], "pending": True})
    st.session_state["clear_input"] = True
    st.rerun()



for idx, turn in enumerate(st.session_state.history):
    if turn["role"] == "user":
        st.markdown(f"**Ask:** {turn['text']}")
    else:
        text = turn.get("text", "")
        is_last = (idx == len(st.session_state.history) - 1)
        pending = bool(turn.get("pending", False))

        # Only auto-fetch when this assistant turn is pending AND is the last turn (live send)
        if text == "_thinking..._" and pending and is_last:
            try:
                if idx > 0 and st.session_state.history[idx - 1]["role"] == "user":
                    user_q = st.session_state.history[idx - 1]["text"]
                else:
                    user_q = None

                if user_q:
                    payload = {"question": user_q, "session_id": st.session_state.session_id}
                    r = requests.post(f"{API_BASE}/query", json=payload, timeout=60)
                    r.raise_for_status()
                    resp = r.json()
                    answer = resp.get("answer") or resp.get("text") or resp.get("response") or ""
                    sources = resp.get("sources") or resp.get("source_documents") or resp.get("source_docs") or []
                else:
                    answer = "Error: no user question found for pending assistant turn."
                    sources = []
            except Exception as e:
                answer = f"Error: {e}"
                sources = []

            # replace the placeholder with real answer and clear 'pending'
            st.session_state.history[idx] = {"role": "assistant", "text": answer, "sources": sources}
            text = answer

        st.markdown(f'<div class="panel"><strong>Answer:</strong><div style="height:8px"></div>{text}</div>', unsafe_allow_html=True)

        if st.session_state.history[idx].get("sources"):
            st.markdown('<div class="panel--muted"><strong>Source:</strong></div>', unsafe_allow_html=True)
            src_html = '<div class="source-list"><ul>'
            for s in st.session_state.history[idx]["sources"]:
                title = s.get("title") or s.get("id") or s.get("source") or "source"
                excerpt = s.get("excerpt") or s.get("cursor") or ""
                url = s.get("url") or s.get("source")
                if url and isinstance(url, str) and url.startswith("http"):
                    src_html += f'<li><a href="{url}" target="_blank">{title}</a> — {excerpt}</li>'
                else:
                    src_html += f'<li><strong>{title}</strong> — {excerpt}</li>'
            src_html += "</ul></div>"
            st.markdown(src_html, unsafe_allow_html=True)

# Footer small note
st.markdown('<div class="small-caption">No images — minimal chat UI. Add streaming later if desired.</div>', unsafe_allow_html=True)