"""
Menu-driven conversational CLI for testing the RAG Chatbot.
Tests the full pipeline:
  chunks.json → ChromaDB → Hybrid Retrieval (Dense + BM25 + RRF) → Cross-Encoder → LLM

Usage:
    python Demo.py
"""

import uuid
from dotenv import load_dotenv
load_dotenv()

from app.memory_store import get_memory, clear_memory
from app.chains import build_conv_chain


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_session_id() -> str:
    return f"session-{uuid.uuid4().hex[:8]}"


def print_sources(source_docs: list):
    """
    Print the retrieved source chunks the LLM used to answer.
    This is the key addition — lets you verify retrieval quality during testing.
    """
    if not source_docs:
        print("  (no source documents returned)")
        return

    print(f"\n  Sources used ({len(source_docs)} chunks):")
    print("  " + "-" * 56)
    for i, doc in enumerate(source_docs, 1):
        m = doc.metadata
        section  = m.get("section_title", "Unknown section")
        h_level  = m.get("heading_level", "?")
        p_start  = m.get("page_start", "?")
        p_end    = m.get("page_end", "?")
        el_types = m.get("element_types", "")
        has_tbl  = "[table]" if m.get("has_table") else ""
        has_lst  = "[list]"  if m.get("has_list")  else ""

        print(f"  [{i}] {section}  (H{h_level} | p.{p_start}-{p_end})")
        print(f"       Types: {el_types}  {has_tbl} {has_lst}")
        print(f"       Preview: {doc.page_content[:120].strip()} ...")
        print()


def ask_question(chain, session_id: str, question: str):
    """Run a question through the chain and display answer + sources + memory."""
    print("\nThinking ...\n")
    try:
        resp = chain.invoke({"question": question})
    except Exception as e:
        print(f"Error invoking chain: {e}")
        return None

    # ── Answer ────────────────────────────────────────────────────────────────
    answer = resp.get("answer", "") if isinstance(resp, dict) else str(resp)
    print("=" * 60)
    print("Answer:")
    print(answer)
    print("=" * 60)

    # ── Sources (retrieval transparency) ──────────────────────────────────────
    # This shows WHICH chunks were retrieved and re-ranked to produce the answer.
    # Use this during testing to verify your hybrid retriever is working correctly.
    source_docs = resp.get("source_documents", []) if isinstance(resp, dict) else []
    print_sources(source_docs)

    # ── Memory snapshot ───────────────────────────────────────────────────────
    memory = get_memory(session_id)
    try:
        chat_hist = memory.load_memory_variables({}).get("chat_history", [])
        turn_count = len(chat_hist) // 2   # each turn = 1 human + 1 AI message
        print(f"  Conversation turns so far: {turn_count}")
    except Exception:
        pass

    print()
    return resp


def print_menu():
    print("\n" + "=" * 60)
    print("       RAG Chatbot -- Test CLI")
    print("=" * 60)
    print("  1  ->  New session  (clears memory, starts fresh)")
    print("  2  ->  Follow-up    (continues current session)")
    print("  3  ->  Show current session info")
    print("  4  ->  Clear current session memory")
    print("  5  ->  Exit")
    print("=" * 60)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_menu():
    active_session_id = None
    active_chain      = None

    print_menu()

    while True:
        try:
            choice = input("\nChoose option (1-5): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting. Bye!")
            break

        # ── Option 1: New session ─────────────────────────────────────────────
        if choice == "1":
            session_id = make_session_id()
            print(f"\nNew session: {session_id}")
            print("Loading chain (embedding model + reranker — first load may take ~30s) ...")
            try:
                active_chain = build_conv_chain(session_id=session_id)
                active_session_id = session_id
                print("Chain ready.\n")
            except Exception as e:
                print(f"Failed to build chain: {e}")
                continue

            q = input("Your question: ").strip()
            if not q:
                print("No question entered.")
                continue
            ask_question(active_chain, active_session_id, q)

        # ── Option 2: Follow-up ───────────────────────────────────────────────
        elif choice == "2":
            if not active_chain:
                print("\nNo active session. Choose option 1 first.")
                continue
            q = input(f"[{active_session_id}] Follow-up question: ").strip()
            if not q:
                print("No question entered.")
                continue
            ask_question(active_chain, active_session_id, q)

        # ── Option 3: Session info ────────────────────────────────────────────
        elif choice == "3":
            if not active_session_id:
                print("\nNo active session.")
            else:
                memory = get_memory(active_session_id)
                try:
                    hist = memory.load_memory_variables({}).get("chat_history", [])
                    turns = len(hist) // 2
                except Exception:
                    turns = "?"
                print(f"\n  Session ID : {active_session_id}")
                print(f"  Turns      : {turns}")

        # ── Option 4: Clear memory ────────────────────────────────────────────
        elif choice == "4":
            if not active_session_id:
                print("\nNo active session.")
            else:
                clear_memory(active_session_id)
                # Rebuild chain with fresh memory for the same session
                active_chain = build_conv_chain(session_id=active_session_id)
                print(f"\nMemory cleared for session {active_session_id}.")

        # ── Option 5: Exit ────────────────────────────────────────────────────
        elif choice == "5":
            print("\nGoodbye!")
            break

        else:
            print("Invalid option. Pick 1-5.")


if __name__ == "__main__":
    run_menu()
