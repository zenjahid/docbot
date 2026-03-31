"""RAG chain with hallucination control.

This module implements the core RAG pipeline with strict grounding
to prevent hallucinations and handle "not found" scenarios.
"""
from typing import List, Tuple, Optional, Dict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger

from config.settings import get_settings
from core.vectorstore import search_similar
from core.memory import format_history_for_rag, add_to_history
from core.document_processor import sanitize_input
from models.schemas import SourceDocument


# Constants
NOT_FOUND_MESSAGE = "This information is not present in the provided document."
MIN_RELEVANCE_SCORE = 0.3  # Minimum similarity score to consider relevant
CONTEXT_THRESHOLD = 0.2    # If no context scores above this, answer not found


def get_llm(provider: str = "gemini") -> any:
    """Get LLM instance based on provider."""
    settings = get_settings()
    
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI")
        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            temperature=0.3  # Low temperature for factual responses
        )
    elif provider == "gemini":
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required for Gemini")
        return ChatGoogleGenerativeAI(
            google_api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
            temperature=0.3
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# System prompt for strict grounding
RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based ONLY on the provided context.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided context
2. If the context does not contain enough information to answer the question, you MUST respond with exactly: "This information is not present in the provided document."
3. Do NOT make up or guess any information not present in the context
4. Do NOT say you "think" or "believe" something unless it's explicitly in the context
5. If you can partially answer, provide what you can and state what is missing
6. Always cite which part of the context your answer is based on

Your goal is to be accurate and helpful while strictly adhering to the provided context."""

# Prompt for checking relevance
RELEVANCE_CHECK_PROMPT = """Analyze the following question and context to determine if you can provide a helpful answer.

Question: {question}

Context:
{context}

Respond with ONLY one of these exact options:
- "CAN_ANSWER" if the context contains sufficient information to fully answer the question
- "PARTIAL" if the context contains some related information but not enough for a complete answer
- "NOT_FOUND" if the context does not contain information relevant to the question

Do not include any other text."""


def check_context_relevance(question: str, documents: List[Tuple[Document, float]]) -> str:
    """
    Check if retrieved context is relevant to the question.
    
    Returns: "CAN_ANSWER", "PARTIAL", or "NOT_FOUND"
    """
    if not documents:
        return "NOT_FOUND"
    
    # Check if any documents have sufficient similarity
    max_score = max(score for _, score in documents)
    
    if max_score < CONTEXT_THRESHOLD:
        logger.info(f"Max similarity score {max_score} below threshold {CONTEXT_THRESHOLD}")
        return "NOT_FOUND"
    
    # Combine context for LLM check
    context_text = "\n\n".join([doc.page_content for doc, _ in documents])
    
    try:
        llm = get_llm("gemini")  # Use Gemini for relevance check (free tier)
        
        prompt = RELEVANCE_CHECK_PROMPT.format(
            question=sanitize_input(question),
            context=context_text[:4000]  # Limit context length
        )
        
        response = llm.invoke(prompt)
        result = response.content.strip().upper()
        
        if "CAN_ANSWER" in result:
            return "CAN_ANSWER"
        elif "PARTIAL" in result:
            return "PARTIAL"
        else:
            return "NOT_FOUND"
            
    except Exception as e:
        logger.warning(f"LLM relevance check failed: {e}, falling back to score threshold")
        # Fallback: use score threshold
        return "CAN_ANSWER" if max_score > MIN_RELEVANCE_SCORE else "NOT_FOUND"


def create_rag_prompt() -> ChatPromptTemplate:
    """Create the RAG prompt template with chat history support."""
    return ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "Context from documents:\n{context}"),
        ("human", "\n\nQuestion: {question}"),
    ])


def format_sources(documents: List[Tuple[Document, float]]) -> List[SourceDocument]:
    """Format retrieved documents as source citations."""
    sources = []
    for doc, score in documents:
        source = SourceDocument(
            content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            source=doc.metadata.get("source_file", "Unknown"),
            page=doc.metadata.get("page"),
            score=round(score, 4)
        )
        sources.append(source)
    return sources


def answer_question(
    question: str,
    session_id: str,
    llm_provider: str = "gemini",
    embedding_provider: str = "free",
    k: int = 5
) -> Tuple[str, List[SourceDocument]]:
    """
    Main RAG function to answer questions with hallucination control.
    
    Args:
        question: User's question
        session_id: Session for conversation continuity
        llm_provider: LLM to use ("openai" or "gemini")
        embedding_provider: Embedding provider for search
        k: Number of documents to retrieve
        
    Returns:
        Tuple of (answer, sources)
    """
    # Sanitize input
    sanitized_question = sanitize_input(question)
    logger.info(f"Processing question: {sanitized_question[:100]}...")
    
    # Retrieve relevant documents
    documents = search_similar(
        query=sanitized_question,
        k=k,
        provider=embedding_provider
    )
    
    logger.info(f"Retrieved {len(documents)} documents")
    
    # Check context relevance
    relevance = check_context_relevance(sanitized_question, documents)
    logger.info(f"Context relevance check: {relevance}")
    
    if relevance == "NOT_FOUND":
        # Return the exact required message
        answer = NOT_FOUND_MESSAGE
        sources = []
    else:
        # Build RAG chain
        llm = get_llm(llm_provider)
        
        # Get chat history for context
        chat_history = format_history_for_rag(session_id, max_pairs=3)
        
        # Combine context
        context_docs = [doc for doc, _ in documents]
        context_text = "\n\n---\n\n".join([
            f"[Source: {doc.metadata.get('source_file', 'Unknown')}] {doc.page_content}"
            for doc in context_docs
        ])
        
        # Create and run chain
        prompt = create_rag_prompt()
        
        chain = prompt | llm
        
        try:
            response = chain.invoke({
                "question": sanitized_question,
                "context": context_text,
                "chat_history": chat_history
            })
            
            answer = response.content
            
            # Verify answer doesn't contradict the "not found" requirement
            if not any(keyword in answer.lower() for keyword in ["not found", "not present", "not available", "cannot find"]):
                if relevance == "PARTIAL":
                    # Append note that partial information was used
                    answer = f"{answer}\n\n*Note: This is a partial answer based on available context.*"
                    
        except Exception as e:
            logger.error(f"RAG chain error: {e}")
            answer = NOT_FOUND_MESSAGE
    
    # Add to conversation history
    add_to_history(session_id, sanitized_question, answer)
    
    # Format sources
    sources = format_sources(documents[:3])  # Top 3 sources
    
    return answer, sources


def check_hallucination(answer: str, context: str) -> bool:
    """
    Verify that the answer is grounded in the context.
    
    Returns True if answer appears grounded, False if potential hallucination detected.
    """
    # Simple heuristic: check if answer references terms not in context
    # This is a basic check - production would use more sophisticated methods
    
    answer_lower = answer.lower()
    context_lower = context.lower()
    
    # Check for confidence indicators that might indicate hallucination
    hallucination_indicators = [
        "i believe", "i think", "probably", "maybe", 
        "it's possible", "as far as i know", "if i recall"
    ]
    
    for indicator in hallucination_indicators:
        if indicator in answer_lower:
            logger.warning(f"Potential hallucination indicator found: {indicator}")
            return False
    
    return True