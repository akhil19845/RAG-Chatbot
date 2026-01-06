# app/chains.py
from typing import Optional
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_ollama import ChatOllama
from langchain_classic.prompts import PromptTemplate
from .reranker import get_retriever
from .memory_store import get_memory
from .config import LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_MODEL
from dotenv import load_dotenv
import os

load_dotenv()

CONDENSE_PROMPT = PromptTemplate.from_template(
    "Given the conversation history and the user's last message, rewrite the last message as a concise, "
    "standalone question suitable for document retrieval. Preserve entity names and technical terms. Output only the rewritten question.\n\n"
    "Conversation history:\n{chat_history}\n\nLast user message:\n{question}\n\nRewritten question:"
)

COMBINE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=("""
    You are a factual QA chatbot assistant.

    You must follow these rules strictly:
    1. Use ONLY the information in the provided context to answer the question.
    2. DO NOT add any explanations, examples, or additional information beyond what appears in the context.
    3. If the context DOES NOT directly contain the answer, reply exactly: "I do not have enough knowledge to answer this question."
    4. If the answer is a command, code, or specific syntax, return ONLY that command or syntax — nothing else.
    5. DO NOT return any confidential information, mask it if required. "

    Context:
    {context}

    Question:
    {question}

    Final Answer:
    """)
)

def _make_llm(model_name: Optional[str] = None):
    model = model_name or LLM_MODEL

    llm = ChatOllama(
        model=model,                     # e.g. "llama3.1"
        temperature=LLM_TEMPERATURE,
        num_predict=LLM_MAX_TOKENS,       # Ollama's max tokens parameter
        base_url="http://localhost:11434" # optional if using default
    )
    return llm


def build_conv_chain(session_id: str, k: int = 4, use_condense_question: bool = True) -> ConversationalRetrievalChain:

    retriever = get_retriever(k=k)
    memory = get_memory(session_id)
    llm = _make_llm()

    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_llm=llm,
        condense_question_prompt=CONDENSE_PROMPT,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": COMBINE_PROMPT}
    )
    return conv_chain