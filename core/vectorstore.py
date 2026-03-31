"""ChromaDB vector store management."""
import uuid
from typing import List, Optional, Tuple
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from loguru import logger

from config.settings import get_settings
from core.embeddings import get_embeddings


class VectorStoreManager:
    """Manages ChromaDB vector store operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self._vectorstore: Optional[Chroma] = None
        self._current_provider: Optional[str] = None
    
    def _get_vectorstore(self, provider: Optional[str] = None) -> Chroma:
        """
        Get or create ChromaDB vector store.
        
        Args:
            provider: Embedding provider to use
            
        Returns:
            Chroma vector store instance
        """
        provider = provider or self.settings.EMBEDDING_PROVIDER
        
        # Recreate if provider changed
        if self._vectorstore is None or self._current_provider != provider:
            logger.info(f"Creating ChromaDB vector store with provider: {provider}")
            
            # Ensure directory exists
            persist_dir = Path(self.settings.CHROMA_PERSIST_DIRECTORY)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            embeddings = get_embeddings(provider)
            self._vectorstore = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=embeddings
            )
            self._current_provider = provider
            
        return self._vectorstore
    
    def add_documents(
        self, 
        documents: List[Document], 
        provider: Optional[str] = None,
        file_id: Optional[str] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document chunks to add
            provider: Embedding provider override
            file_id: Optional file identifier for grouping
            
        Returns:
            List of chunk IDs
        """
        vectorstore = self._get_vectorstore(provider)
        
        # Add metadata
        for doc in documents:
            if file_id:
                doc.metadata['file_id'] = file_id
            doc.metadata['chunk_id'] = str(uuid.uuid4())
        
        ids = vectorstore.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to vector store")
        return ids
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[dict] = None,
        provider: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with relevance scores.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            provider: Embedding provider override
            
        Returns:
            List of (Document, score) tuples
        """
        vectorstore = self._get_vectorstore(provider)
        
        results = vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
        
        logger.info(f"Found {len(results)} similar documents")
        return results
    
    def delete_by_file_id(self, file_id: str) -> bool:
        """
        Delete all documents associated with a file ID.
        
        Args:
            file_id: File identifier to delete
            
        Returns:
            True if successful
        """
        try:
            vectorstore = self._get_vectorstore()
            
            # Get all IDs for this file
            results = vectorstore.get(where={"file_id": file_id})
            
            if results and results['ids']:
                vectorstore.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} documents for file_id: {file_id}")
                return True
            
            logger.warning(f"No documents found for file_id: {file_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_document_count(self, filter: Optional[dict] = None) -> int:
        """Get total number of documents in the store."""
        vectorstore = self._get_vectorstore()
        
        if filter:
            results = vectorstore.get(where=filter)
        else:
            results = vectorstore.get()
        
        return len(results['ids']) if results else 0
    
    def list_files(self) -> List[dict]:
        """List all unique files in the vector store."""
        vectorstore = self._get_vectorstore()
        
        try:
            results = vectorstore.get()
            
            if not results or not results['metadatas']:
                return []
            
            # Extract unique files
            files = {}
            for metadata in results['metadatas']:
                file_id = metadata.get('file_id')
                if file_id and file_id not in files:
                    files[file_id] = {
                        'file_id': file_id,
                        'source_file': metadata.get('source_file', 'Unknown'),
                        'chunks_count': 1
                    }
                elif file_id:
                    files[file_id]['chunks_count'] += 1
            
            return list(files.values())
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    def reset(self):
        """Reset the vector store (for testing or reinitialization)."""
        self._vectorstore = None
        self._current_provider = None


# Global instance
_vector_store_manager = VectorStoreManager()


def get_vectorstore(provider: Optional[str] = None) -> Chroma:
    """Convenience function to get vector store."""
    return _vector_store_manager._get_vectorstore(provider)


def add_documents(documents: List[Document], provider: Optional[str] = None, file_id: Optional[str] = None) -> List[str]:
    """Convenience function to add documents."""
    return _vector_store_manager.add_documents(documents, provider, file_id)


def search_similar(
    query: str, 
    k: int = 5,
    filter: Optional[dict] = None,
    provider: Optional[str] = None
) -> List[Tuple[Document, float]]:
    """Convenience function to search documents."""
    return _vector_store_manager.similarity_search_with_score(query, k, filter, provider)