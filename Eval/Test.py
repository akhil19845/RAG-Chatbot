from app.memory_store import get_memory
import os
import sys
import json
import time
from dotenv import load_dotenv

# project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.chains import build_conv_chain
from app.reranker import get_retriever

from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from app.config import LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_MODEL
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional


def _make_llm(model_name: Optional[str] = None):
    
    model = model_name or LLM_MODEL
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_TOKENS,
        google_api_key=api_key,
    )
    return llm

def main():
    load_dotenv()

    q1 = "what is deep learning ?"
    q2 = "what is its importance ?"
    session_id = "demo"
    
    # retriever = get_retriever()
    chain = build_conv_chain(session_id=session_id)

    response1 = chain.invoke({"question":q1})

    # Put this after you build chain and after the first question has been asked

    # 1) Inspect memory (you already did — shows messages)
    memory = get_memory(session_id)
    print("Memory (raw):", memory.load_memory_variables({})["chat_history"])

    # 2) If you provided condense_question_llm, explicitly produce the condensed question (debug)
    # Create a simple condense prompt for debugging (this won't change your chain's behavior)

    CONDENSE_DEBUG_PROMPT = PromptTemplate.from_template(
        "Given the conversation history:\n\n{chat_history}\n\n"
        "And the last user message:\n\n{question}\n\n"
        "Rewrite the last user message as a concise, standalone question suitable for document retrieval.\n"
        "Output only the rewritten question."
    )

    llm = _make_llm()

    # use same llm you used in the chain: llm = _make_llm()
    condense_chain = LLMChain(llm=llm, prompt=CONDENSE_DEBUG_PROMPT)
    mem_vars = memory.load_memory_variables({})
    chat_history = mem_vars["chat_history"]
    latest_user = chat_history[-1].content if len(chat_history) else ""
    condensed = condense_chain.run({"chat_history": chat_history, "question": latest_user})
    print("Condensed question (debug):", condensed)

    # 3) Ask the retriever directly with the condensed question and inspect returned docs
    retriever = get_retriever(k=4)   # make sure same params as build_conv_chain
    docs = retriever.invoke(condensed)
    print("Number of docs returned by retriever:", len(docs))
    for i, d in enumerate(docs):
        print(f"--- Doc {i} ---")
        print("page_content:", d.page_content[:500])   # first 500 chars
        print("metdata:", getattr(d, "metadata", {}))
        print("\n")

if __name__ == "__main__":
    main()
