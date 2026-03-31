"""DocBot Settings - Configuration management using Pydantic."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Literal
from functools import lru_cache


# Get the project root directory (parent of config/)
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Embedding Configuration
    # Options: "free_huggingface", "free_watsonx", "paid_openai", "paid_gemini"
    EMBEDDING_PROVIDER: str = "free_huggingface"
    
    # Free HuggingFace (sentence-transformers)
    FREE_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Free IBM WatsonX (free Lite tier)
    WATSONX_URL: str = "https://us-south.ml.cloud.ibm.com"
    WATSONX_PROJECT_ID: str = ""
    WATSONX_EMBEDDING_MODEL: str = "ibm/slate-125m-english-rtrvr"
    
    # Paid OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Paid Gemini
    GEMINI_API_KEY: str = ""
    
    # LLM Provider Configuration
    LLM_PROVIDER: Literal["openai", "gemini"] = "gemini"
    GEMINI_MODEL: str = "gemini-2.5-flash"
    OPENAI_MODEL: str = "gpt-4o-mini"
    
    # Vector Database
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    
    # Document Processing
    MAX_FILE_SIZE_MB: int = 50
    MAX_TOTAL_SIZE_MB: int = 200
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


def get_settings() -> Settings:
    """Get settings instance - loads from .env file."""
    if not hasattr(get_settings, '_instance'):
        get_settings._instance = Settings()
    return get_settings._instance


# Initialize on module load
get_settings._instance = Settings()