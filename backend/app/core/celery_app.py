"""
Celery configuration for background task processing.

This module configures Celery for asynchronous document processing tasks.
Pre-loads ML models during worker startup to avoid runtime downloads.
"""

import logging
from celery import Celery
from celery.signals import worker_process_init
from app.core.config import settings

logger = logging.getLogger(__name__)

celery_app = Celery(
    "fund_analysis",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks.document_tasks"]
)

# Configuration for production use
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes max execution time
    task_soft_time_limit=25 * 60,  # 25 minutes soft limit
    worker_prefetch_multiplier=1,  # Fair task distribution
    task_acks_late=True,  # Acknowledge after task completion
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks
)

# Global model cache for Celery workers
_model_cache = {}

@worker_process_init.connect
def preload_models_on_worker_init(sender, **kwargs):
    """
    Pre-load ML models when Celery worker starts.

    This ensures models are cached in each worker process,
    avoiding runtime downloads and improving performance.
    """
    try:
        logger.info("Pre-loading ML models in Celery worker...")

        # Pre-load sentence-transformers model
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers model...")
            _model_cache['sentence_transformers'] = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✓ Sentence-transformers model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to pre-load sentence-transformers: {e}")

        # Pre-load docling converter (this will cache RapidOCR models)
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption
            from docling.datamodel.base_models import InputFormat

            logger.info("Loading Docling converter with OCR...")
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True

            _model_cache['docling_converter'] = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            logger.info("✓ Docling converter loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to pre-load Docling converter: {e}")

        # Verify RapidOCR models are cached
        try:
            import os
            rapidocr_dir = "/usr/local/lib/python3.11/site-packages/rapidocr/models"
            if os.path.exists(rapidocr_dir):
                model_files = [
                    "ch_PP-OCRv4_det_infer.pth",
                    "ch_ptocr_mobile_v2.0_cls_infer.pth",
                    "ch_PP-OCRv4_rec_infer.pth"
                ]
                found = sum(1 for f in model_files if os.path.exists(os.path.join(rapidocr_dir, f)))
                logger.info(f"✓ Found {found}/{len(model_files)} RapidOCR model files cached")
        except Exception as e:
            logger.debug(f"Could not verify RapidOCR models: {e}")

        logger.info("✓ Model pre-loading complete in Celery worker")

    except Exception as e:
        logger.error(f"Failed to pre-load models in worker: {e}")

def get_cached_model(model_name: str):
    """Get a cached model instance."""
    return _model_cache.get(model_name)

# For development - allows running celery from command line
if __name__ == "__main__":
    celery_app.start()