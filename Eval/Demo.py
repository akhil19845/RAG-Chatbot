"""
Menu-driven conversational CLI for your ConversationalRetrievalChain.
Requires your existing functions:
 - build_conv_chain(session_id: str) -> ConversationalRetrievalChain
 - get_memory(session_id: str) -> Memory object
 - (and any environment setup you already use, e.g., load_dotenv())
"""

import uuid
import time
from dotenv import load_dotenv

# import your project functions
from app.memory_store import get_memory
from app.chains import build_conv_chain

# Optionally, if you want deterministic LLM behavior while debugging,
# you can set temperature on the LLM inside build_conv_chain or in _make_llm().

def make_session_id() -> str:
    """Create a short unique session id."""
    return f"session-{uuid.uuid4().hex[:8]}"

def ask_question(chain, session_id: str, question: str):
    """
    Ask the chain a question and print the answer and memory snapshot.
    Returns the chain response dict.
    """
    try:
        resp = chain.invoke({"question": question})
        print("Response from Chain invoking is: ", resp)
        print("\n")
    except Exception as e:
        print("Error invoking chain:", e)
        return None

    answer = resp.get("answer") if isinstance(resp, dict) else resp
    print("\n--- Answer ---")
    print(answer)
    print("--------------\n")

    # show memory snapshot (friendly)
    memory = get_memory(session_id)
    try:
        mem_vars = memory.load_memory_variables({})
        chat_hist = mem_vars.get("chat_history")
    except Exception:
        # fallback: the memory object might be of different shape
        chat_hist = memory

    print("Memory snapshot (latest messages):")
    # Print a short view rather than raw object dump for readability
    if isinstance(chat_hist, list):
        for i, m in enumerate(chat_hist[-6:]):  # show last 6 messages max
            role = getattr(m, "type", None) or getattr(m, "__class__", m).__name__
            content = getattr(m, "content", str(m))
            print(f" {i+1}. {content}")
    else:
        print(chat_hist)
    print()

    return resp

def run_menu():
    load_dotenv()

    active_session_id = None
    active_chain = None

    print("=== Conversational Retrieval CLI ===")
    print("Options:")
    print("  1) Ask a new question (starts a new session)")
    print("  2) Ask a follow-up question (uses current session)")
    print("  3) Exit")
    print("====================================\n")

    while True:
        try:
            choice = input("Choose option (1/2/3): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Bye.")
            break

        if choice == "1":
            # New session flow
            session_id = make_session_id()
            print(f"\nStarting new session: {session_id}")
            print("Building chain for this session (may take a moment)...")
            # build the chain (this attaches memory for session_id)
            active_chain = build_conv_chain(session_id=session_id)
            active_session_id = session_id

            q = input("Enter your question: ").strip()
            if not q:
                print("No question entered — returning to menu.\n")
                continue

            ask_question(active_chain, active_session_id, q)

        elif choice == "2":
            # Follow-up flow
            if active_session_id is None or active_chain is None:
                print("\nNo active session. Please choose option 1 first to start a session.\n")
                continue

            q = input(f"[{active_session_id}] Enter follow-up question: ").strip()
            if not q:
                print("No question entered — returning to menu.\n")
                continue

            ask_question(active_chain, active_session_id, q)

        elif choice == "3":
            print("\nExit selected. Goodbye.")
            break

        else:
            print("Invalid option. Please pick 1, 2, or 3.\n")

if __name__ == "__main__":
    run_menu()