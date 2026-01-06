# app/memory_store.py

from typing import Dict
from langchain_classic.memory import ConversationBufferMemory


_memories: Dict[str, ConversationBufferMemory] = {}


def get_memory(session_id: str) -> ConversationBufferMemory:

    if session_id not in _memories:
        _memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history",   
            return_messages=True,        
            output_key = "answer"
        )
    return _memories[session_id]


def clear_memory(session_id: str) -> None:

    if session_id in _memories:
        del _memories[session_id]


def clear_all() -> None:
    _memories.clear()