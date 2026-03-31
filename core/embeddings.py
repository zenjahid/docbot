"""Embedding factory - supports free and paid embedding providers.

Free options:
- HuggingFace (sentence-transformers): Local, no API key needed
- IBM WatsonX: Free Lite tier, requires IBM Cloud account

Paid options:
- OpenAI: Requires API key
- Gemini: Requires API key
"""
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger

from config.settings import get_settings


class EmbeddingFactory:
    """Factory for creating embedding models based on configuration."""
    
    def __init__(self):
        self.settings = get_settings()
        self._embedding_cache: Optional[Embeddings] = None
        self._last_provider: Optional[str] = None
    
    def get_embeddings(self, provider: Optional[str] = None) -> Embeddings:
        """
        Get embedding model based on provider.
        
        Args:
            provider: Override the default provider
                - "free_huggingface": Local sentence-transformers (FREE)
                - "free_watsonx": IBM WatsonX (FREE tier)
                - "paid_openai": OpenAI embeddings (PAID)
                - "paid_gemini": Google Gemini (PAID with free tier)
            
        Returns:
            Configured embeddings instance
        """
        provider = provider or self.settings.EMBEDDING_PROVIDER
        
        # Return cached if same provider
        if self._embedding_cache is not None and self._last_provider == provider:
            return self._embedding_cache
        
        # Create new embeddings based on provider
        if provider == "free_huggingface":
            self._embedding_cache = self._get_huggingface_embeddings()
        elif provider == "free_watsonx":
            self._embedding_cache = self._get_watsonx_embeddings()
        elif provider == "paid_openai":
            self._embedding_cache = self._get_openai_embeddings()
        elif provider == "paid_gemini":
            self._embedding_cache = self._get_gemini_embeddings()
        else:
            logger.warning(f"Unknown provider '{provider}', defaulting to HuggingFace")
            self._embedding_cache = self._get_huggingface_embeddings()
        
        self._last_provider = provider
        return self._embedding_cache
    
    def _get_huggingface_embeddings(self) -> Embeddings:
        """
        Get free HuggingFace embeddings using sentence-transformers.
        
        No API key required - runs entirely locally.
        """
        logger.info(f"Initializing HuggingFace embeddings: {self.settings.FREE_EMBEDDING_MODEL}")
        
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.settings.FREE_EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            # Test the embeddings
            _ = embeddings.embed_query("test")
            logger.info("HuggingFace embeddings initialized successfully")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace embeddings: {e}")
            raise
    
    def _get_watsonx_embeddings(self) -> Embeddings:
        """
        Get free IBM WatsonX embeddings (Lite tier).
        
        Requires IBM Cloud account with free Lite plan.
        """
        if not self.settings.WATSONX_PROJECT_ID:
            raise ValueError(
                "WATSONX_PROJECT_ID is required for WatsonX embeddings. "
                "Sign up for free at https://cloud.ibm.com/"
            )
        
        logger.info("Initializing IBM WatsonX embeddings...")
        
        try:
            from langchain_ibm import WatsonxEmbeddings
            
            embeddings = WatsonxEmbeddings(
                model_id=self.settings.WATSONX_EMBEDDING_MODEL,
                url=self.settings.WATSONX_URL,
                project_id=self.settings.WATSONX_PROJECT_ID
            )
            
            # Test the embeddings
            _ = embeddings.embed_query("test")
            logger.info("WatsonX embeddings initialized successfully")
            return embeddings
            
        except ImportError:
            logger.error("langchain-ibm not installed. Run: pip install langchain-ibm")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize WatsonX embeddings: {e}")
            raise
    
    def _get_openai_embeddings(self) -> Embeddings:
        """Get OpenAI embeddings (paid)."""
        if not self.settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        
        logger.info("Initializing OpenAI embeddings")
        return OpenAIEmbeddings(
            api_key=self.settings.OPENAI_API_KEY,
            model=self.settings.OPENAI_EMBEDDING_MODEL
        )
    
    def _get_gemini_embeddings(self) -> Embeddings:
        """Get Gemini embeddings (paid with free tier)."""
        if not self.settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required for Gemini embeddings")
        
        logger.info("Initializing Gemini embeddings")
        return GoogleGenerativeAIEmbeddings(
            google_api_key=self.settings.GEMINI_API_KEY,
            model="models/embedding-001"
        )
    
    def clear_cache(self):
        """Clear the embedding cache to force re-initialization."""
        self._embedding_cache = None
        self._last_provider = None


# Global instance for reuse
_embedding_factory = EmbeddingFactory()


def get_embeddings(provider: Optional[str] = None) -> Embeddings:
    """Convenience function to get embeddings."""
    return _embedding_factory.get_embeddings(provider)


def get_embedding_dimension(embeddings: Embeddings) -> int:
    """Get the embedding dimension by testing with a sample text."""
    sample_embedding = embeddings.embed_query("test")
    return len(sample_embedding)