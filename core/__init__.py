"""Core module - RAG pipeline, document processing, embeddings, vector store, memory."""
from .document_processor import DocumentProcessor, sanitize_input
from .embeddings import get_embeddings, EmbeddingFactory
from .vectorstore import VectorStoreManager, get_vectorstore, add_documents, search_similar
from .memory import (
    ConversationMemory,
    create_session,
    get_or_create_session,
    add_to_history,
    get_history,
    format_history_for_rag
)
from .rag_chain import answer_question, check_context_relevance, NOT_FOUND_MESSAGE

__all__ = [
    "DocumentProcessor",
    "sanitize_input",
    "get_embeddings",
    "EmbeddingFactory",
    "VectorStoreManager",
    "get_vectorstore",
    "add_documents",
    "search_similar",
    "ConversationMemory",
    "create_session",
    "get_or_create_session",
    "add_to_history",
    "get_history",
    "format_history_for_rag",
    "answer_question",
    "check_context_relevance",
    "NOT_FOUND_MESSAGE"
]