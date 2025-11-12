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

# Global model cache for Celery workers (legacy - now uses shared global cache)
_model_cache = {}

@worker_process_init.connect
def preload_models_on_worker_init(sender, **kwargs):
    """
    Pre-load ML models when Celery worker starts.

    This ensures models are downloaded during Docker build to avoid runtime downloads.
    Uses preload_models.py for actual loading logic.
    """
    try:
        logger.info("Pre-loading ML models in Celery worker...")
        
        # Import preload functions
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
            logger.info(f"{status} {model_name} model preloading in Celery worker")

    except Exception as e:
        logger.error(f"Failed to pre-load models in worker: {e}")

def get_cached_model(model_name: str):
    """Get a cached model instance."""
    # Try global cache first, then fallback to local cache
    try:
        from app.core.model_cache import get_cached_model as get_global_model
        model = get_global_model(model_name)
        if model is not None:
            return model
    except ImportError:
        pass
    
    # Fallback to local cache
    return _model_cache.get(model_name)

# For development - allows running celery from command line
if __name__ == "__main__":
    celery_app.start()