#!/usr/bin/env python3
# tests/debug_followup_empty.py
import sys, os, pprint
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.chains import build_conv_chain
from app.memory_store import get_memory
from pprint import pprint
import time

def safe_invoke(chain, inputs):
    try:
        return chain.invoke(inputs)
    except AttributeError:
        try:
            return chain(inputs)
        except Exception:
            return chain.__call__(**inputs)

def show_condensed(chain, session_id, follow_up):
    mem = get_memory(session_id)
    mem_vars = mem.load_memory_variables({})
    chat_history = mem_vars.get("chat_history", [])
    qgen = getattr(chain, "question_generator", None)
    if qgen is None:
        print("No question_generator installed")
        return None
    # run question generator
    try:
        out = qgen.invoke({"question": follow_up, "chat_history": chat_history})
    except AttributeError:
        out = qgen({"question": follow_up, "chat_history": chat_history})
    pprint({"qgen_raw": out})
    # heuristics
    if isinstance(out, dict):
        for key in ("question","standalone_question","output_text","answer"):
            if key in out and isinstance(out[key], str):
                return out[key]
        for v in out.values():
            if isinstance(v, str) and len(v.strip())>0:
                return v
    elif isinstance(out, str) and out.strip():
        return out
    return None

def print_memory(session_id):
    mem = get_memory(session_id)
    vars = mem.load_memory_variables({})
    print("--- memory ---")
    pprint(vars)
    return vars

def inspect_filled_combine(chain, question, retrieved_docs_text):
    combine_chain = chain.combine_docs_chain
    prompt_obj = combine_chain.llm_chain.prompt
    # format_prompt -> PromptValue with to_string / to_messages
    try:
        pv = prompt_obj.format_prompt(context=retrieved_docs_text, question=question)
        if hasattr(pv, "to_string"):
            s = pv.to_string()
            print("\n--- FILLED PROMPT (string) ---\n")
            print(s[:4000])
        elif hasattr(pv, "to_messages"):
            print("\n--- FILLED PROMPT (chat messages) ---\n")
            for m in pv.to_messages():
                print(f"{m.type}: {m.content[:2000]}")
        else:
            print("\n--- Filled prompt object (repr) ---")
            print(repr(pv)[:2000])
    except Exception as e:
        print("Could not format combine prompt:", e)

def main():
    session_id = "test_history_1"
    chain = build_conv_chain(session_id=session_id, k=4)
    # TURN 1
    q1 = "What is Deep Learning ?"
    print("\n=== TURN 1 ===\nQuestion:", q1)
    r1 = safe_invoke(chain, {"question": q1})
    print("\nTURN1 keys:", list(r1.keys()))
    print("\nTURN1 answer repr:", repr(r1.get("answer")))
    print_memory(session_id)

    # prepare follow-up
    follow_up = "What are its applications and importance ?"
    print("\n=== CONDENSED (what qgen produces) ===")
    condensed = show_condensed(chain, session_id, follow_up)
    print("Condensed question:", repr(condensed))

    # Build simple retrieved text (join top docs) to feed into prompt inspector:
    # get retriever's docs (fallback)
    retriever = chain.retriever
    try:
        docs = retriever.invoke(condensed or follow_up)
    except Exception:
        try:
            docs = retriever._get_relevant_documents(condensed or follow_up)
        except Exception:
            docs = []
    retrieved_text = "\n\n".join(getattr(d, "page_content", "") for d in docs)
    inspect_filled_combine(chain, condensed or follow_up, retrieved_text[:4000])

    # TURN 2: actually call
    print("\n=== TURN 2 ===\nQuestion:", follow_up)
    r2 = safe_invoke(chain, {"question": follow_up})
    print("\nTURN2 keys:", list(r2.keys()))
    # show raw repr to reveal whitespace
    print("\nTURN2 answer repr:", repr(r2.get("answer")))
    print_memory(session_id)

if __name__ == "__main__":
    main()