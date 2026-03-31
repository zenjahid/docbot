"""Conversational memory management for multi-turn chat."""
import uuid
from typing import List, Optional, Dict
from datetime import datetime
from loguru import logger

# In-memory storage for simplicity (can be replaced with SQLite/Redis for production)
_session_store: Dict[str, List[Dict]] = {}


class ConversationMemory:
    """Manages conversation history for multi-turn chat support."""
    
    @staticmethod
    def create_session() -> str:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())
        _session_store[session_id] = []
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    @staticmethod
    def get_session(session_id: str) -> Optional[str]:
        """Check if session exists."""
        return session_id if session_id in _session_store else None
    
    @staticmethod
    def add_message(session_id: str, role: str, content: str) -> None:
        """
        Add a message to session history.
        
        Args:
            session_id: Session identifier
            role: "human" or "ai"
            content: Message content
        """
        if session_id not in _session_store:
            _session_store[session_id] = []
        
        _session_store[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.debug(f"Added {role} message to session {session_id}")
    
    @staticmethod
    def get_history(session_id: str, max_messages: int = 20) -> List[Dict]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            max_messages: Maximum number of messages to return (most recent)
            
        Returns:
            List of message dictionaries
        """
        if session_id not in _session_store:
            return []
        
        history = _session_store[session_id]
        # Return most recent messages (keeping context for multi-turn)
        return history[-max_messages:] if len(history) > max_messages else history
    
    @staticmethod
    def get_chat_history_formatted(session_id: str, max_messages: int = 10) -> List[tuple]:
        """
        Get chat history as a list of (role, content) tuples for LangChain.
        
        Args:
            session_id: Session identifier
            max_messages: Maximum number of message pairs
            
        Returns:
            List of (role, content) tuples
        """
        history = ConversationMemory.get_history(session_id, max_messages * 2)
        
        # Convert to tuples for LangChain
        formatted = [(msg["role"], msg["content"]) for msg in history]
        return formatted
    
    @staticmethod
    def clear_session(session_id: str) -> bool:
        """Clear all messages in a session."""
        if session_id in _session_store:
            _session_store[session_id] = []
            logger.info(f"Cleared session: {session_id}")
            return True
        return False
    
    @staticmethod
    def delete_session(session_id: str) -> bool:
        """Delete a session completely."""
        if session_id in _session_store:
            del _session_store[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False
    
    @staticmethod
    def list_sessions() -> List[str]:
        """List all active session IDs."""
        return list(_session_store.keys())
    
    @staticmethod
    def get_session_count() -> int:
        """Get total number of active sessions."""
        return len(_session_store)


# Convenience functions
def create_session() -> str:
    """Create a new conversation session."""
    return ConversationMemory.create_session()


def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get existing session or create new one."""
    if session_id and ConversationMemory.get_session(session_id):
        return session_id
    return ConversationMemory.create_session()


def add_to_history(session_id: str, question: str, answer: str) -> None:
    """Add a question-answer pair to session history."""
    ConversationMemory.add_message(session_id, "human", question)
    ConversationMemory.add_message(session_id, "ai", answer)


def get_history(session_id: str, max_messages: int = 20) -> List[Dict]:
    """Get conversation history."""
    return ConversationMemory.get_history(session_id, max_messages)


def format_history_for_rag(session_id: str, max_pairs: int = 5) -> List[tuple]:
    """Format history for RAG chain input."""
    return ConversationMemory.get_chat_history_formatted(session_id, max_pairs)