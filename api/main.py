"""FastAPI application for DocBot RAG Chatbot."""
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import uuid
import tempfile
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config.settings import get_settings
from models.schemas import (
    ChatRequest, ChatResponse, DocumentUploadResponse,
    DocumentInfo, HealthResponse, SourceDocument
)
from core import (
    DocumentProcessor, get_or_create_session, answer_question,
    add_documents, search_similar
)
from core.vectorstore import VectorStoreManager
from core.memory import ConversationMemory


# Settings
settings = get_settings()
vector_manager = VectorStoreManager()
doc_processor = DocumentProcessor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    logger.info("DocBot API starting up...")
    
    # Ensure chroma directory exists
    Path(settings.CHROMA_PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)
    
    logger.info("DocBot API ready")
    yield
    
    logger.info("DocBot API shutting down...")


# Create FastAPI app
app = FastAPI(
    title="DocBot API",
    description="RAG Chatbot with hallucination control and conversational memory",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    try:
        doc_count = vector_manager.get_document_count()
        return HealthResponse(
            status="healthy",
            vectorstore_ready=True,
            documents_count=doc_count,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            vectorstore_ready=False,
            documents_count=0,
            timestamp=datetime.now()
        )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the document chatbot.
    
    Handles multi-turn conversation with context memory.
    Returns "This information is not present in the provided document." when answer not found.
    """
    # Validate question
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Get or create session
    session_id = get_or_create_session(request.session_id)
    
    logger.info(f"Chat request - Session: {session_id}, Question: {request.question[:50]}...")
    
    try:
        # Get answer with RAG
        answer, sources = answer_question(
            question=request.question,
            session_id=session_id,
            llm_provider=request.llm_provider,
            embedding_provider=request.embedding_provider,
            k=5
        )
        
        # Determine model used
        model_used = f"{request.llm_provider.upper()}"
        
        return ChatResponse(
            answer=answer,
            session_id=session_id,
            sources=sources,
            model_used=model_used
        )
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Error processing your question")


@app.post("/upload-doc", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a document (PDF or DOCX).
    
    Documents are split into chunks and stored in ChromaDB.
    """
    # Validate file type
    allowed_extensions = {'.pdf', '.docx'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Validate file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
        )
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Generate file ID
        file_id = str(uuid.uuid4())
        
        # Process document
        chunks = doc_processor.process_file(
            temp_path,
            metadata={"uploaded_at": datetime.now().isoformat()}
        )
        
        # Add to vector store
        add_documents(chunks, provider=settings.EMBEDDING_PROVIDER, file_id=file_id)
        
        logger.info(f"Uploaded {file.filename} with {len(chunks)} chunks")
        
        return DocumentUploadResponse(
            filename=file.filename,
            file_id=file_id,
            chunks_created=len(chunks),
            message=f"Document '{file.filename}' uploaded and indexed successfully"
        )
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/list-docs", response_model=list[DocumentInfo])
async def list_documents():
    """List all uploaded documents."""
    try:
        files = vector_manager.list_files()
        
        return [
            DocumentInfo(
                file_id=f['file_id'],
                filename=f['source_file'],
                upload_timestamp=datetime.now(),  # Would need metadata for actual time
                chunks_count=f['chunks_count']
            )
            for f in files
        ]
    except Exception as e:
        logger.error(f"List docs error: {e}")
        raise HTTPException(status_code=500, detail="Error listing documents")


@app.post("/delete-doc")
async def delete_document(file_id: str = Query(...)):
    """Delete a document by file ID."""
    try:
        success = vector_manager.delete_by_file_id(file_id)
        
        if success:
            return {"message": f"Document {file_id} deleted successfully", "success": True}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail="Error deleting document")


@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str, max_messages: int = Query(20, le=100)):
    """Get conversation history for a session."""
    history = ConversationMemory.get_history(session_id, max_messages)
    return {"session_id": session_id, "history": history}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a conversation session."""
    success = ConversationMemory.delete_session(session_id)
    return {"message": "Session deleted" if success else "Session not found", "success": success}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)