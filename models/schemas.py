"""Pydantic models for request/response validation."""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    question: str = Field(..., description="User's question")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    embedding_provider: str = Field(default="free_huggingface")
    llm_provider: str = Field(default="gemini")


class SourceDocument(BaseModel):
    """Source document with citation info."""
    content: str
    source: str
    page: Optional[int] = None
    score: float


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str
    session_id: str
    sources: List[SourceDocument] = []
    model_used: str


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    filename: str
    file_id: str
    chunks_created: int
    message: str


class DocumentInfo(BaseModel):
    """Document metadata model."""
    file_id: str
    filename: str
    upload_timestamp: datetime
    chunks_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    vectorstore_ready: bool
    documents_count: int
    timestamp: datetime