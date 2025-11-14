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
            model = get_sentence_transformer()
            if model is not None:
                logger.info("✓ Sentence transformer preloaded successfully")
            else:
                logger.warning("Sentence transformer preloading returned None")
        except Exception as e:
            logger.warning(f"Could not preload sentence transformer: {e}")

        # Preload LLM clients if API keys are available
        try:
            from app.core.config import settings

            # Try to preload OpenAI client
            if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
                try:
                    client = get_openai_client()
                    if client is not None:
                        logger.info("✓ OpenAI client preloaded")
                    else:
                        logger.warning("OpenAI client preloading returned None")
                except Exception as e:
                    logger.warning(f"Could not preload OpenAI client: {e}")

            # Try to preload Gemini client
            if hasattr(settings, 'GEMINI_API_KEY') and settings.GEMINI_API_KEY:
                try:
                    client = get_gemini_client()
                    if client is not None:
                        logger.info("✓ Gemini client preloaded")
                    else:
                        logger.warning("Gemini client preloading returned None")
                except Exception as e:
                    logger.warning(f"Could not preload Gemini client: {e}")

        except Exception as e:
            logger.warning(f"Could not check for LLM API keys: {e}")

        logger.info("✓ Critical model preloading completed")

    except Exception as e:
        logger.error(f"Failed to preload critical models: {e}")
        import traceback
        traceback.print_exc()


def preload_docling_models():
    """
    Preload Docling models by processing a dummy document to trigger downloads.

    This ensures models are cached before actual document processing.
    """
    import os

    try:
        logger.info("Preloading Docling models by processing dummy document...")

        # Get the converter (this creates it but doesn't download models yet)
        converter = get_docling_converter()

        # Try to use an existing sample PDF if available
        sample_pdf_path = None
        possible_paths = [
            "/app/files/Sample_Fund_Performance_Report.pdf",
            "/app/files/ILPA based Capital Accounting and Performance Metrics_ PIC, Net PIC, DPI, IRR  .pdf",
            "./files/Sample_Fund_Performance_Report.pdf",
            "./files/ILPA based Capital Accounting and Performance Metrics_ PIC, Net PIC, DPI, IRR  .pdf"
        ]

        for path in possible_paths:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                try:
                    # Check if file is readable
                    with open(path, 'rb') as f:
                        f.read(100)  # Read first 100 bytes
                    sample_pdf_path = path
                    logger.info(f"Using existing sample PDF: {path} (size: {os.path.getsize(path)} bytes)")
                    break
                except Exception as e:
                    logger.warning(f"Sample PDF {path} exists but is not readable: {e}")
                    continue

        # If no sample PDF found, create a more complex dummy PDF
        if not sample_pdf_path:
            import tempfile
            import os
            try:
                from pypdf import PdfWriter
                from reportlab.pdfgen import canvas
                from reportlab.lib.pagesizes import letter

                # Create a PDF with some text content to better trigger model downloads
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_pdf_path = temp_file.name

                # Use reportlab to create a PDF with text
                c = canvas.Canvas(temp_pdf_path, pagesize=letter)
                c.drawString(100, 750, "Sample Fund Performance Report")
                c.drawString(100, 730, "Capital Calls: $1,000,000")
                c.drawString(100, 710, "Distributions: $500,000")
                c.drawString(100, 690, "Date: 2024-01-01")
                c.save()

                sample_pdf_path = temp_pdf_path
                logger.info("Created dummy PDF with text content for model preloading")

            except ImportError:
                # Fallback to pypdf only
                from pypdf import PdfWriter

                # Create minimal valid PDF
                writer = PdfWriter()
                writer.add_blank_page(width=612, height=792)

                # Write to temporary file
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_pdf_path = temp_file.name
                    writer.write(temp_file)

                sample_pdf_path = temp_pdf_path
                logger.info("Created minimal dummy PDF for model preloading")

        try:
            # Process the PDF to trigger model downloads
            import os
            logger.info(f"Processing PDF {os.path.basename(sample_pdf_path)} to download Docling models...")
            import time
            start_time = time.time()
            result = converter.convert(sample_pdf_path)
            conversion_time = time.time() - start_time

            if result and hasattr(result, 'document'):
                logger.info(f"Finished converting document {os.path.basename(sample_pdf_path)} in {conversion_time:.2f} sec.")
                logger.info("✓ Docling models downloaded and cached successfully")
                # Log some info about the result
                doc = result.document
                if hasattr(doc, 'pages'):
                    logger.info(f"Preloading conversion: {len(doc.pages)} pages processed")
                if hasattr(doc, 'tables'):
                    logger.info(f"Preloading conversion: {len(doc.tables)} tables found")
            else:
                logger.warning("Docling conversion during preloading returned no result")
        except Exception as convert_error:
            logger.warning(f"Docling conversion during preloading failed: {convert_error}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up if we created a temp file
            if sample_pdf_path and sample_pdf_path.startswith('/tmp/'):
                try:
                    os.unlink(sample_pdf_path)
                except:
                    pass

    except Exception as e:
        logger.warning(f"Could not preload Docling models: {e} - models will download on first use")
        import traceback
        traceback.print_exc()


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