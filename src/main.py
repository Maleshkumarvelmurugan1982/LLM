"""
Main FastAPI application entry point
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from .api.routes import query_router, document_router, health_router
from .config import get_settings
from .models.database import init_database
from .utils.logger import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting LLM Document Query System...")
    try:
        await init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Document Query System...")

# Create FastAPI app
app = FastAPI(
    title="LLM Document Query and Retrieval System",
    description="A system for processing natural language queries and retrieving information from unstructured documents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix="/api/v1", tags=["Health"])
app.include_router(query_router, prefix="/api/v1", tags=["Query"])
app.include_router(document_router, prefix="/api/v1", tags=["Documents"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LLM Document Query and Retrieval System",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
