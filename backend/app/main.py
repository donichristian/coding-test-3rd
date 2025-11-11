"""
FastAPI main application entry point
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.endpoints import documents, funds, chat, metrics
from app.services.vector_store import VectorStore
from app.services.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Fund Performance Analysis System API",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Pre-load ML models during application startup
@app.on_event("startup")
async def startup_event():
    """Initialize and cache ML models on application startup."""
    try:
        logger.info("Pre-loading ML models during application startup...")

        # Force initialization of shared VectorStore (loads sentence-transformers model)
        vector_store = VectorStore()
        logger.info("✓ VectorStore initialized with cached models")

        # Force initialization of shared RAG Engine components
        rag_engine = RAGEngine()
        logger.info("✓ RAG Engine initialized with shared components")

        logger.info("✓ All ML models pre-loaded and cached successfully")

    except Exception as e:
        logger.error(f"Failed to pre-load ML models: {e}")
        # Don't fail startup, but log the error
        logger.warning("Application will continue, but model loading may be slower on first requests")

# CORS middleware - Allow all origins in development
if settings.ENVIRONMENT == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins in development
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
else:
    # Production: Use configured origins only
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

# Include routers
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(funds.router, prefix="/api/funds", tags=["funds"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["metrics"])


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
