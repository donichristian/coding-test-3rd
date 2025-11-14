"""
Application configuration
"""
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings
    """

    # Project
    PROJECT_NAME: str = "Fund Performance Analysis System"
    VERSION: str = "1.0.0"

    # API
    API_V1_STR: str = "/api"

    ## CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ]

    # Database
    DATABASE_URL: str = "postgresql://funduser:fundpass@localhost:5432/funddb"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # AI Provider Configurations
    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Google Gemini
    GEMINI_API_KEY: str = os.environ.get('GEMINI_API_KEY')
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Default Embedding Model (used by Docling and other components)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Anthropic (optional)
    ANTHROPIC_API_KEY: str = ""

    # Vector Store
    VECTOR_STORE_PATH: str = "./vector_store"
    FAISS_INDEX_PATH: str = "./faiss_index"

    # File Upload
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB

    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Vector Store
    BATCH_SIZE: int = 10
    EMBEDDING_TIMEOUT: float = 5.0
    SIMILARITY_SEARCH_MULTIPLIER: int = 2

    # RAG
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.3

    # Embedding Configurations
    EMBEDDING_CONFIGS: Dict[str, Dict[str, Any]] = {
        "sentence-transformers": {
            "model": "all-MiniLM-L6-v2",
            "dimensions": 384,
            "description": "Local sentence-transformers model"
        },
        "gemini": {
            "model": "gemini-embedding-001",
            "dimensions": 768,
            "description": "Google Gemini embeddings"
        },
        "openai": {
            "model": "text-embedding-3-small",
            "dimensions": 1536,
            "description": "OpenAI embeddings"
        }
    }

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()