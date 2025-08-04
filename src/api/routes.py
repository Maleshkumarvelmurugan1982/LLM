"""
API routes for the LLM Document Query System
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks
from typing import List, Optional
import os
import uuid
import aiofiles
from pathlib import Path

from ..models.schemas import (
    QueryRequest, QueryResponse, DocumentUploadRequest, DocumentUploadResponse,
    DocumentInfo, HealthResponse, DocumentType
)
from ..document_processor import DocumentManager
from ..query_engine import QueryProcessor
from ..semantic_search import SemanticSearchEngine
from ..decision_engine import DecisionEngine
from ..utils.logger import get_logger
from ..config import get_settings

logger = get_logger(__name__)

# Initialize components
settings = get_settings()
document_manager = DocumentManager()
query_processor = QueryProcessor()
search_engine = SemanticSearchEngine()
decision_engine = DecisionEngine()

# Dependency to ensure components are initialized
async def get_initialized_components():
    """Ensure all components are initialized"""
    if not query_processor.entity_extractor.nlp:
        await query_processor.initialize()
    if not search_engine.embedding_engine.model:
        await search_engine.initialize()
    if not decision_engine.ml_engine.model:
        await decision_engine.initialize()
    
    return {
        "document_manager": document_manager,
        "query_processor": query_processor,
        "search_engine": search_engine,
        "decision_engine": decision_engine
    }

# Health Router
health_router = APIRouter()

@health_router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db_status = "healthy"
        try:
            from ..models.database import get_database
            db = await get_database()
            await db.get_collection_stats()
        except Exception as e:
            db_status = f"unhealthy: {str(e)}"
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            dependencies={
                "database": db_status,
                "embedding_model": "loaded" if search_engine.embedding_engine.model else "not loaded",
                "nlp_model": "loaded" if query_processor.entity_extractor.nlp else "not loaded"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unhealthy")

# Query Router
query_router = APIRouter()

@query_router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    components: dict = Depends(get_initialized_components)
):
    """Process a natural language query against documents"""
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Step 1: Process the query
        processed_query = await components["query_processor"].process_query(request.query)
        logger.info(f"Query processed with intent: {processed_query.intent}")
        
        # Step 2: Search for relevant documents
        search_results = await components["search_engine"].search(
            processed_query=processed_query,
            max_results=request.max_results,
            document_ids=request.document_ids
        )
        logger.info(f"Found {len(search_results)} relevant document chunks")
        
        # Step 3: Make decision based on retrieved information
        response = await components["decision_engine"].make_decision(
            processed_query=processed_query,
            search_results=search_results
        )
        
        logger.info(f"Query processed successfully: {response.decision}")
        return response
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@query_router.get("/query/{query_id}/explain")
async def explain_decision(
    query_id: str,
    components: dict = Depends(get_initialized_components)
):
    """Get detailed explanation of a decision"""
    # In a real implementation, you would store query results and retrieve by ID
    # For now, return a placeholder
    return {
        "query_id": query_id,
        "message": "Decision explanation not implemented. Store query results to enable this feature."
    }

@query_router.get("/search/stats")
async def get_search_stats(
    components: dict = Depends(get_initialized_components)
):
    """Get search engine statistics"""
    try:
        stats = await components["search_engine"].get_search_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get search stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get search statistics")

# Document Router
document_router = APIRouter()

@document_router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: Optional[str] = None,
    document_type: Optional[DocumentType] = None,
    components: dict = Depends(get_initialized_components)
):
    """Upload and process a document"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Determine document type from file extension if not provided
        if not document_type:
            file_ext = Path(file.filename).suffix.lower()
            type_mapping = {
                '.pdf': DocumentType.PDF,
                '.docx': DocumentType.DOCX,
                '.txt': DocumentType.TXT,
                '.eml': DocumentType.EMAIL
            }
            document_type = type_mapping.get(file_ext)
            if not document_type:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
        
        # Use filename as name if not provided
        if not name:
            name = Path(file.filename).stem
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Save uploaded file
        upload_dir = Path(settings.documents_path)
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{document_id}_{file.filename}"
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        logger.info(f"File saved: {file_path}")
        
        # Process document in background
        background_tasks.add_task(
            process_document_background,
            str(file_path),
            name,
            document_type,
            document_id,
            components["search_engine"]
        )
        
        return DocumentUploadResponse(
            document_id=document_id,
            status="uploaded",
            message="Document uploaded successfully. Processing in background.",
            chunks_created=0  # Will be updated after processing
        )
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")

async def process_document_background(
    file_path: str,
    name: str,
    document_type: DocumentType,
    document_id: str,
    search_engine: SemanticSearchEngine
):
    """Background task to process uploaded document"""
    try:
        logger.info(f"Starting background processing for document: {name}")
        
        # Process document
        doc_id, chunks = await document_manager.process_document(
            file_path=file_path,
            name=name,
            document_type=document_type
        )
        
        # Index chunks for search
        success = await search_engine.index_documents(chunks)
        
        if success:
            logger.info(f"Document {name} processed and indexed successfully: {len(chunks)} chunks")
        else:
            logger.error(f"Failed to index document {name}")
            
    except Exception as e:
        logger.error(f"Background document processing failed: {e}")

@document_router.get("/documents", response_model=List[str])
async def list_documents():
    """List all uploaded documents"""
    try:
        upload_dir = Path(settings.documents_path)
        if not upload_dir.exists():
            return []
        
        documents = []
        for file_path in upload_dir.iterdir():
            if file_path.is_file():
                documents.append(file_path.name)
        
        return documents
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to list documents")

@document_router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    components: dict = Depends(get_initialized_components)
):
    """Delete a document and its chunks"""
    try:
        # Delete from search index
        success = await components["search_engine"].delete_document_chunks(document_id)
        
        # Delete file (in a real implementation, you'd need to track file paths)
        # For now, just return success based on index deletion
        
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@document_router.get("/documents/{document_id}/chunks")
async def get_document_chunks(
    document_id: str,
    components: dict = Depends(get_initialized_components)
):
    """Get chunks for a specific document"""
    try:
        # This would require storing document metadata
        # For now, return a placeholder
        return {
            "document_id": document_id,
            "message": "Document chunk retrieval not fully implemented. Requires document metadata storage."
        }
        
    except Exception as e:
        logger.error(f"Failed to get document chunks: {e}")
        raise HTTPException(status_code=500, detail="Failed to get document chunks")
