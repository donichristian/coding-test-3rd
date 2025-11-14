"""
Lazy model loader for ML models.

This module provides on-demand loading of ML models to avoid
downloading large models during Docker build time, reducing image size.
"""
import logging
import threading
from typing import Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Global model cache with thread safety
_model_cache = {}
_load_lock = threading.Lock()


@contextmanager
def _loading_context(model_name: str):
    """Context manager for safe model loading."""
    if model_name in _model_cache:
        yield _model_cache[model_name]
        return

    with _load_lock:
        if model_name in _model_cache:
            yield _model_cache[model_name]
            return

        try:
            logger.info(f"Loading {model_name} model on-demand...")
            yield None  # Signal to load
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise


def get_docling_converter():
    """Lazy load Docling DocumentConverter."""
    model_name = 'docling_converter'

    with _loading_context(model_name) as cached:
        if cached is not None:
            return cached

    # Load the model
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption
        from docling.datamodel.base_models import InputFormat

        logger.info("Initializing Docling converter with OCR disabled for performance...")
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False  # Disable OCR for performance
        pipeline_options.do_table_structure = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        _model_cache[model_name] = converter
        logger.info("✓ Docling converter loaded successfully")
        return converter

    except Exception as e:
        logger.error(f"Failed to load Docling converter: {e}")
        raise


def get_sentence_transformer(model_name: str = 'all-MiniLM-L6-v2'):
    """Lazy load SentenceTransformer model."""
    cache_key = f'sentence_transformer_{model_name}'

    with _loading_context(cache_key) as cached:
        if cached is not None:
            return cached

    # Load the model
    try:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading sentence-transformers model: {model_name}...")
        model = SentenceTransformer(model_name)

        _model_cache[cache_key] = model
        logger.info(f"✓ Sentence-transformers model {model_name} loaded successfully")
        return model

    except Exception as e:
        logger.error(f"Failed to load sentence-transformers model {model_name}: {e}")
        raise


def get_openai_client():
    """Lazy load OpenAI client."""
    model_name = 'openai_client'

    with _loading_context(model_name) as cached:
        if cached is not None:
            return cached

    # Load the client
    try:
        from app.core.config import settings
        from openai import OpenAI

        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not configured")

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        _model_cache[model_name] = client
        logger.info("✓ OpenAI client initialized")
        return client

    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise


def get_gemini_client():
    """Lazy load Gemini client."""
    model_name = 'gemini_client'

    with _loading_context(model_name) as cached:
        if cached is not None:
            return cached

    # Load the client
    try:
        from app.core.config import settings
        import google.genai as genai

        if not settings.GEMINI_API_KEY:
            raise ValueError("Gemini API key not configured")

        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        _model_cache[model_name] = client
        logger.info("✓ Gemini client initialized")
        return client

    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        raise


def preload_critical_models():
    """
    Preload only the most critical models that benefit from caching.

    This is called during Celery worker startup, not Docker build.
    """
    try:
        logger.info("Preloading critical models for Celery worker...")

        # Preload Docling converter and actually download models by processing dummy document
        preload_docling_models()

        # Preload sentence transformer for embeddings
        try:
            get_sentence_transformer()
        except Exception as e:
            logger.warning(f"Could not preload sentence transformer: {e}")

        # Preload LLM clients if API keys are available
        try:
            from app.core.config import settings

            # Try to preload OpenAI client
            if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
                try:
                    get_openai_client()
                    logger.info("✓ OpenAI client preloaded")
                except Exception as e:
                    logger.warning(f"Could not preload OpenAI client: {e}")

            # Try to preload Gemini client
            if hasattr(settings, 'GEMINI_API_KEY') and settings.GEMINI_API_KEY:
                try:
                    get_gemini_client()
                    logger.info("✓ Gemini client preloaded")
                except Exception as e:
                    logger.warning(f"Could not preload Gemini client: {e}")

        except Exception as e:
            logger.warning(f"Could not check for LLM API keys: {e}")

        logger.info("✓ Critical model preloading completed")

    except Exception as e:
        logger.error(f"Failed to preload critical models: {e}")


def preload_docling_models():
    """
    Preload Docling models by processing a dummy document to trigger downloads.

    This ensures models are cached before actual document processing.
    """
    try:
        logger.info("Preloading Docling models by processing dummy document...")

        # Get the converter (this creates it but doesn't download models yet)
        converter = get_docling_converter()

        # Create a minimal dummy PDF to trigger model downloads
        import tempfile
        import os
        from pypdf import PdfWriter

        # Create minimal valid PDF
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_pdf_path = temp_file.name
            writer.write(temp_file)

        try:
            # Process the dummy PDF to trigger model downloads
            logger.info("Processing dummy PDF to download Docling models...")
            result = converter.convert(temp_pdf_path)
            logger.info("✓ Docling models downloaded and cached successfully")
        finally:
            # Clean up
            try:
                os.unlink(temp_pdf_path)
            except:
                pass

    except ImportError:
        logger.warning("pypdf not available for dummy PDF creation - models will download on first use")
    except Exception as e:
        logger.warning(f"Could not preload Docling models: {e} - models will download on first use")


def clear_model_cache():
    """Clear the model cache to free memory."""
    global _model_cache
    with _load_lock:
        _model_cache.clear()
        logger.info("Model cache cleared")


def get_cache_stats():
    """Get statistics about cached models."""
    return {
        "cached_models": list(_model_cache.keys()),
        "cache_size": len(_model_cache)
    }