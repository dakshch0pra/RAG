from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import tempfile
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Global FastAPI app
app = FastAPI(
    title="Mini RAG System API",
    description="Professional Retrieval-Augmented Generation System",
    version="1.0.0"
)

# Global RAG system instance (will be injected from app.py)
rag_system = None

# -------------------------
# Pydantic Models
# -------------------------
class QueryRequest(BaseModel):
    question: str
    retrieve_k: int = 10
    rerank_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    citations: List[str]
    metadata: Dict[str, Any]

class TextContent(BaseModel):
    content: str
    title: str = "Manual Entry"
    source: str = "Text Input"

class DocumentStats(BaseModel):
    total_documents: int
    index_size: int
    embedding_dimension: int

class UploadResponse(BaseModel):
    success: bool
    message: str
    chunks_added: int

# -------------------------
# Dependency Injection
# -------------------------
def init_rag_system(rag_instance):
    """Attach RAG system instance to backend"""
    global rag_system
    rag_system = rag_instance

# -------------------------
# Routes
# -------------------------
@app.get("/", response_model=Dict[str, str])
async def root():
    return {"message": "Mini RAG System API", "status": "active", "version": "1.0.0"}

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = rag_system.query(
        request.question,
        retrieve_k=request.retrieve_k,
        rerank_k=request.rerank_k
    )
    return QueryResponse(**result)

@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Validate file type
        allowed_types = ['.pdf', '.txt', '.md']
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {allowed_types}"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process the file
            success, message, chunks = rag_system.process_uploaded_file(temp_file_path)
            
            return UploadResponse(
                success=success,
                message=message,
                chunks_added=chunks
            )
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/add-text", response_model=UploadResponse)
async def add_text_content(content: TextContent):
    """Add text content directly to the knowledge base"""
    try:
        if not content.content.strip():
            raise HTTPException(status_code=400, detail="Content cannot be empty")
        
        # Create temporary file with text content
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as temp_file:
            temp_file.write(content.content)
            temp_file_path = temp_file.name
        
        try:
            # Process as if it were an uploaded file
            success, message, chunks = rag_system.process_uploaded_file(temp_file_path)
            
            # Update the metadata to reflect manual entry
            if success and chunks > 0:
                # Update the last added documents with proper metadata
                recent_docs = rag_system.vector_db.metadata[-chunks:]
                for doc_meta in recent_docs:
                    doc_meta['source'] = content.source
                    doc_meta['title'] = f"{content.title} - {doc_meta.get('title', 'Chunk')}"
                    doc_meta['type'] = 'manual'
            
            return UploadResponse(
                success=success,
                message=f"Text content added: {content.title}",
                chunks_added=chunks
            )
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add text error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/stats", response_model=DocumentStats)
async def get_document_stats():
    """Get database statistics"""
    try:
        stats = rag_system.vector_db.get_stats()
        return DocumentStats(**stats)
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/list")
async def list_documents():
    """List all documents in the knowledge base"""
    try:
        documents = []
        sources = {}
        
        # Group documents by source
        for doc, meta in zip(rag_system.vector_db.documents, rag_system.vector_db.metadata):
            source = meta.get('source', 'Unknown')
            if source not in sources:
                sources[source] = {
                    'source': source,
                    'type': meta.get('type', 'unknown'),
                    'chunks': [],
                    'total_chunks': 0,
                    'timestamp': meta.get('timestamp', 'Unknown')
                }
            
            sources[source]['chunks'].append({
                'chunk_id': meta.get('chunk_id', 0),
                'title': meta.get('title', 'Untitled'),
                'section': meta.get('section', 'N/A'),
                'content_preview': doc[:200] + "..." if len(doc) > 200 else doc
            })
            sources[source]['total_chunks'] += 1
        
        return {"documents": list(sources.values())}
        
    except Exception as e:
        logger.error(f"List documents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/clear")
async def clear_database():
    """Clear the entire knowledge base"""
    try:
        rag_system.vector_db.clear_database()
        rag_system.conversation_history.clear_history()
        
        # Reload sample documents
        rag_system._load_sample_documents()
        
        return {"message": "Database cleared and sample documents reloaded"}
        
    except Exception as e:
        logger.error(f"Clear database error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Comprehensive health check"""
    try:
        stats = rag_system.vector_db.get_stats()
        
        return {
            "status": "healthy",
            "components": {
                "vector_database": "operational",
                "gemini_client": "available" if rag_system.gemini_client else "unavailable",
                "cohere_client": "available" if rag_system.cohere_client else "unavailable"
            },
            "metrics": {
                "total_documents": stats['total_documents'],
                "conversations_in_history": len(rag_system.conversation_history.conversations)
            }
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )