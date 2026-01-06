#!/usr/bin/env python3

import os
import sys
import json
import time
from dotenv import load_dotenv
import uuid

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.chains import build_conv_chain
from app.reranker import get_retriever
from app.config import (
    INPUT_FILE,
    OUTPUT_FILE
)

def make_session_id() -> str:
    """Create a short unique session id."""
    return f"session-{uuid.uuid4()}"


def main():
    load_dotenv()
    # session_id = os.getenv("SESSION_ID", "eval-session")
    # session_id = "reranker-session"

    top_k = 3
    input_file = INPUT_FILE
    output_file = OUTPUT_FILE

    if not os.path.exists(input_file):
        raise SystemExit(f"Input file not found: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # chain = build_conv_chain(session_id=session_id, k=top_k)

    retriever = get_retriever(k=top_k)

    # data = data[0]
    results = []
    for idx, item in enumerate(data, start=0):
        session_id = make_session_id()
        chain = build_conv_chain(session_id=session_id, k=top_k)
        question = item["question"]
        docs = retriever.invoke(question)
        context_texts = [doc.page_content for doc in docs]

        resp = chain.invoke({"question": question, "chat_history": []})

        answer = resp["answer"]

        results.append({
            "question" : question,
            "ground_truth" : item["ground_truth"],
            "chatbot_answer" : answer,
            "contexts" : context_texts
        })

        print(f"q{idx} processed")

        time.sleep(60)
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()