#!/usr/bin/env python3
"""
scripts/check_chains.py

Usage examples:
# build only (no external calls)
python scripts/check_chains.py --session test-session

# build and run a single question (will call the real LLM & retriever -> needs GOOGLE_API_KEY and working retriever)
python scripts/check_chains.py --session test-session --run --question "What's the summary of the repo?"

# build but don't call remote LLM (explicit mock/dry)
python scripts/check_chains.py --session test-session --dry
"""

import argparse
import os
import sys
from dotenv import load_dotenv
import traceback
from typing import Any

# ensure project root is on path if needed
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# import your factory
try:
    from app.chains import build_conv_chain, _make_llm  # noqa: E402
except Exception as e:
    print("ERROR importing app.chains — fix import paths first.")
    traceback.print_exc()
    raise SystemExit(1)

def pretty_print_component(name: str, comp: Any):
    t = type(comp)
    print(f"\n{name}:")
    print(f"  type: {t}")
    # Try to print lightweight identifying info without calling network
    info = {}
    # common attributes to surface
    for attr in ("model", "temperature", "max_output_tokens", "google_api_key"):
        if hasattr(comp, attr):
            try:
                info[attr] = getattr(comp, attr)
            except Exception:
                info[attr] = "<unreadable>"
    if info:
        for k, v in info.items():
            print(f"  {k}: {v}")
    else:
        # fallback to repr but truncated
        r = repr(comp)
        print("  repr:", (r[:400] + "..." if len(r) > 400 else r))

def main():
    load_dotenv()  # loads .env from project root (if present)
    parser = argparse.ArgumentParser(description="Check app.chains wiring (build conv chain and optionally run).")
    parser.add_argument("--session", "-s", required=True, help="session id to pass to build_conv_chain")
    parser.add_argument("--k", type=int, default=4, help="number of docs to retrieve (k)")
    parser.add_argument("--run", action="store_true", help="actually run the chain against a question (calls LLM & retriever)")
    parser.add_argument("--question", "-q", default="Hello, can you summarize the repository?", help="question to ask when --run is used")
    parser.add_argument("--dry", action="store_true", help="dry run / mock mode (do not call external APIs). Recommended for quick checks.")
    args = parser.parse_args()

    print("Loaded environment variables (GOOGLE_API_KEY present?):", bool(os.getenv("GOOGLE_API_KEY")))

    # 1) Build the chain
    try:
        print("\nBuilding ConversationalRetrievalChain (this may construct LLM/retriever/memory)...")
        conv_chain = build_conv_chain(session_id=args.session, k=args.k)
    except Exception as e:
        print("Failed to build ConversationalRetrievalChain.")
        traceback.print_exc()
        raise SystemExit(2)

    # 2) Inspect components: llm (if available), retriever, memory
    try:
        # ConversationalRetrievalChain stores components in attrs (implementation may vary by langchain version)
        llm = getattr(conv_chain, "llm", None)
        retriever = getattr(conv_chain, "retriever", None)
        memory = getattr(conv_chain, "memory", None)
        combine_chain = getattr(conv_chain, "combine_docs_chain", None)
    except Exception:
        llm = retriever = memory = combine_chain = None

    if llm is None:
        # try to access via nested attributes (older/newer LC versions differ)
        try:
            llm = conv_chain.llm_chain.llm
        except Exception:
            pass

    pretty_print_component("LLM", llm if llm is not None else "<not found>")
    pretty_print_component("Retriever", retriever if retriever is not None else "<not found>")
    pretty_print_component("Memory", memory if memory is not None else "<not found>")
    pretty_print_component("Combine docs chain", combine_chain if combine_chain is not None else "<not found>")

    # 3) Optionally run the chain
    if args.run:
        if args.dry:
            print("\n--dry specified together with --run: skipping external call (dry run).")
        else:
            if not os.getenv("GOOGLE_API_KEY"):
                print("\nWARNING: No GOOGLE_API_KEY found in environment. Running will likely fail.")
                print("If you really want to proceed, re-run with GOOGLE_API_KEY in environment or use --dry.")
                raise SystemExit(3)

            # Try to call the chain. Depending on langchain version the call signature may differ.
            print(f"\nRunning chain with question: {args.question!r}")
            try:
                # Preferred input for ConversationalRetrievalChain: {"question": q, "chat_history": []}
                response = conv_chain.invoke({"question": args.question})
                print("\nChain response (raw):")
                # If returned is an object/dict print sensibly
                if isinstance(response, dict):
                    for k, v in response.items():
                        print(f"  {k}: {v}")
                else:
                    print("  ", response)
            except TypeError:
                # fallback: older versions may support .run()
                try:
                    out = conv_chain.run(args.question)
                    print("\nChain.run output:\n", out)
                except Exception:
                    print("Failed to execute chain via either callable or .run().")
                    traceback.print_exc()
            except Exception:
                print("Error while executing chain (likely network/credential/ retriever issue).")
                traceback.print_exc()
    else:
        print("\nNot running the chain. To execute an example question against the LLM pass --run.")
    print("\nDone.")

if __name__ == "__main__":
    main()