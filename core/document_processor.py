"""Document processing utilities for PDF and DOCX files."""
import os
from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config.settings import get_settings


class DocumentProcessor:
    """Handles document loading, parsing, and chunking."""
    
    def __init__(self):
        self.settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self._supported_extensions = {'.pdf', '.docx'}
    
    @property
    def supported_extensions(self) -> set:
        """Get supported file extensions."""
        return self._supported_extensions.copy()
    
    def is_supported(self, file_path: str) -> bool:
        """Check if file extension is supported."""
        return Path(file_path).suffix.lower() in self._supported_extensions
    
    def validate_file(self, file_path: str) -> bool:
        """Validate file exists and is accessible."""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        if path.stat().st_size > self.settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            logger.error(f"File exceeds max size: {file_path}")
            return False
        return True
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document from file path.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of LangChain Document objects
        """
        if not self.validate_file(file_path):
            raise ValueError(f"Invalid file: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.pdf':
                return self._load_pdf(file_path)
            elif file_ext == '.docx':
                return self._load_docx(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load and extract text from PDF file."""
        logger.info(f"Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from PDF")
        return documents
    
    def _load_docx(self, file_path: str) -> List[Document]:
        """Load and extract text from DOCX file."""
        logger.info(f"Loading DOCX: {file_path}")
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} sections from DOCX")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of chunked Document objects
        """
        logger.info(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def process_file(self, file_path: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        Complete pipeline: load -> split -> add metadata.
        
        Args:
            file_path: Path to document file
            metadata: Optional metadata to add to chunks
            
        Returns:
            List of processed Document chunks
        """
        logger.info(f"Processing file: {file_path}")
        
        # Load document
        documents = self.load_document(file_path)
        
        # Split into chunks
        chunks = self.split_documents(documents)
        
        # Add file metadata to each chunk
        file_name = Path(file_path).name
        for chunk in chunks:
            if metadata:
                chunk.metadata.update(metadata)
            chunk.metadata['source_file'] = file_name
            chunk.metadata['file_path'] = str(file_path)
        
        logger.info(f"Successfully processed {len(chunks)} chunks from {file_name}")
        return chunks


def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent prompt injection.
    
    Args:
        text: Raw user input
        
    Returns:
        Sanitized text safe for use in prompts
    """
    # Remove potential prompt injection patterns
    dangerous_patterns = [
        "ignore previous instructions",
        "ignore all previous",
        "disregard your instructions",
        "system prompt",
        "you are now",
        "pretend you are",
        "ignore system",
    ]
    
    sanitized = text
    for pattern in dangerous_patterns:
        sanitized = sanitized.replace(pattern, "")
    
    # Strip extra whitespace
    return " ".join(sanitized.split())