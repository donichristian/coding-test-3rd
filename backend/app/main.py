"""
FastAPI main application entry point
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.endpoints import documents, funds, chat, metrics

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Fund Performance Analysis System API",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(funds.router, prefix="/api/funds", tags=["funds"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["metrics"])


@app.on_event("startup")
async def startup_event():
    """Preload ML models on startup to avoid runtime downloads."""
    try:
        logger.info("Preloading ML models on FastAPI startup...")
        
        # Import preload functions - single source of truth from preload_models.py
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from preload_models import preload_sentence_transformers, preload_docling_models, preload_rapidocr_models
        
        # Run preloading
        results = {
            "sentence_transformers": preload_sentence_transformers(),
            "docling": preload_docling_models(),
            "rapidocr": preload_rapidocr_models(),
        }
        
        # Log results
        for model_name, success in results.items():
            status = "✓" if success else "✗"
            logger.info(f"{status} {model_name} model preloading")
            
    except Exception as e:
        logger.warning(f"Failed to preload models on startup: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Fund Performance Analysis System API",
        "version": settings.VERSION,
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}