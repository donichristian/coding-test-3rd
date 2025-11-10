"""
Application configuration with best practices.

This module provides centralized configuration management using Pydantic Settings.
All sensitive data is loaded from environment variables with proper validation.
"""
import os
from functools import lru_cache
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings with comprehensive validation and type safety.

    All settings are loaded from environment variables with sensible defaults.
    Sensitive information is properly handled and validated.
    """

    # Application Metadata
    PROJECT_NAME: str = Field(
        default="Fund Performance Analysis System",
        description="Name of the application"
    )
    VERSION: str = Field(
        default="1.0.0",
        description="Application version"
    )
    ENVIRONMENT: str = Field(
        default="development",
        description="Deployment environment (development/staging/production)"
    )

    # API Configuration
    API_V1_STR: str = Field(
        default="/api",
        description="API version prefix"
    )

    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
            "http://localhost:3002",  # Additional development ports
            "http://127.0.0.1:3001",
            "http://127.0.0.1:3002",
            "http://frontend:3000",  # Docker container communication
            "*",  # Allow all origins for development (will be validated by validator)
        ],
        description="Allowed CORS origins"
    )

    # Database Configuration
    DATABASE_URL: str = Field(
        description="PostgreSQL database connection URL",
        default="postgresql://funduser:fundpass@localhost:5432/funddb"
    )

    # Redis Configuration
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )

    # AI Provider Configurations
    # OpenAI Settings
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API key (optional)"
    )
    OPENAI_MODEL: str = Field(
        default="gpt-4-turbo-preview",
        description="OpenAI model for chat completion"
    )
    OPENAI_EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small",
        description="OpenAI model for embeddings"
    )

    # Google Gemini Settings
    GEMINI_API_KEY: Optional[str] = Field(
        default_factory=lambda: os.environ.get('GEMINI_API_KEY') or
                                os.environ.get('GOOGLE_API_KEY'),
        description="Google Gemini API key"
    )
    GEMINI_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="Google Gemini model"
    )

    # Anthropic Settings (Optional)
    ANTHROPIC_API_KEY: Optional[str] = Field(
        default=None,
        description="Anthropic API key (optional)"
    )

    # Vector Store Configuration
    VECTOR_STORE_PATH: str = Field(
        default="./vector_store",
        description="Path for vector store data"
    )
    FAISS_INDEX_PATH: str = Field(
        default="./faiss_index",
        description="Path for FAISS index files"
    )

    # File Upload Configuration
    UPLOAD_DIR: str = Field(
        default="./uploads",
        description="Directory for uploaded files"
    )
    MAX_UPLOAD_SIZE: int = Field(
        default=50 * 1024 * 1024,  # 50MB
        description="Maximum upload file size in bytes",
        gt=0
    )

    # Document Processing Configuration
    CHUNK_SIZE: int = Field(
        default=1000,
        description="Text chunk size for document processing",
        gt=0
    )
    CHUNK_OVERLAP: int = Field(
        default=200,
        description="Overlap between text chunks",
        ge=0
    )

    # RAG Configuration
    TOP_K_RESULTS: int = Field(
        default=5,
        description="Number of top results to retrieve",
        gt=0
    )
    SIMILARITY_THRESHOLD: float = Field(
        default=0.7,
        description="Similarity threshold for retrieval",
        ge=0.0,
        le=1.0
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # Environment variables are typically case-insensitive
        extra="ignore"  # Ignore extra environment variables
    )

    @field_validator("ALLOWED_ORIGINS", mode="after")
    @classmethod
    def validate_cors_origins(cls, v: List[str], info: ValidationInfo) -> List[str]:
        """Validate CORS origins based on environment."""
        env = info.data.get("ENVIRONMENT", "development")
        if env == "production" and "*" in v:
            raise ValueError("Wildcard origin '*' not allowed in production")
        return v

    @field_validator("DATABASE_URL", mode="after")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL format."""
        if not v.startswith(("postgresql://", "postgresql+asyncpg://")):
            raise ValueError("Database URL must be a valid PostgreSQL connection string")
        return v

    @field_validator("REDIS_URL", mode="after")
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        """Validate Redis URL format."""
        if not v.startswith("redis://"):
            raise ValueError("Redis URL must start with 'redis://'")
        return v

    @field_validator("MAX_UPLOAD_SIZE", mode="after")
    @classmethod
    def validate_max_upload_size(cls, v: int) -> int:
        """Validate maximum upload size."""
        max_allowed = 100 * 1024 * 1024  # 100MB absolute maximum
        if v > max_allowed:
            raise ValueError(f"Maximum upload size cannot exceed {max_allowed} bytes")
        return v

    @field_validator("CHUNK_OVERLAP", mode="after")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info: ValidationInfo) -> int:
        """Validate chunk overlap doesn't exceed chunk size."""
        chunk_size = info.data.get("CHUNK_SIZE", 1000)
        if v >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v

    def get_ai_provider_config(self, provider: str) -> dict:
        """
        Get configuration for a specific AI provider.

        Args:
            provider: Name of the provider ('openai', 'gemini', 'anthropic')

        Returns:
            Dictionary with provider-specific configuration

        Raises:
            ValueError: If provider is not supported or not configured
        """
        configs = {
            "openai": {
                "api_key": self.OPENAI_API_KEY,
                "model": self.OPENAI_MODEL,
                "embedding_model": self.OPENAI_EMBEDDING_MODEL,
            },
            "gemini": {
                "api_key": self.GEMINI_API_KEY,
                "model": self.GEMINI_MODEL,
            },
            "anthropic": {
                "api_key": self.ANTHROPIC_API_KEY,
            }
        }

        if provider not in configs:
            raise ValueError(f"Unsupported AI provider: {provider}")

        config = configs[provider]
        if not config.get("api_key"):
            raise ValueError(f"API key not configured for provider: {provider}")

        return config


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Global settings instance
settings = get_settings()
