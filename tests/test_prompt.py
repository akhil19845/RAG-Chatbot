#!/usr/bin/env python3
"""
Standalone script to inspect LangChain's default prompt templates
for ConversationalRetrievalChain — no project imports required.
"""
from typing import Optional
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI  # or another LLM; we only need a dummy instance
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env


def _make_llm(model_name: Optional[str] = None):
    """
    Creates a Chat model backed by Google Gemini (via the Generative AI API).
    - Requires GOOGLE_API_KEY in the environment or pass google_api_key here.
    - Uses the LangChain adapter ChatGoogleGenerativeAI so it plugs into ConversationalRetrievalChain.
    """
    model = "gemini-2.5-flash"  # e.g., "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", etc.
    # The ChatGoogleGenerativeAI constructor accepts model, temperature, max_output_tokens, google_api_key (optional)
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0.0,
        max_output_tokens=200,
        google_api_key=api_key,  # optional if you prefer to pass explicitly
    )
    return llm

def main():
    # Build a dummy retriever so the chain can initialize
    retriever = Chroma().as_retriever()

    llm = _make_llm()

    # Build the chain (this uses all LangChain built-in defaults)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    # --- Combine (QA) prompt ---
    combine_prompt = chain.combine_docs_chain.llm_chain.prompt
    print("\n=== DEFAULT COMBINE (QA) PROMPT ===\n")
    print(combine_prompt.template)
    print("\nInput variables:", combine_prompt.input_variables)

    # # --- Condense (follow-up) prompt ---
    # condense_prompt = chain.question_generator.llm_chain.prompt
    # print("\n=== DEFAULT CONDENSE (FOLLOW-UP) PROMPT ===\n")
    # print(condense_prompt.template)
    # print("\nInput variables:", condense_prompt.input_variables)

if __name__ == "__main__":
    main()