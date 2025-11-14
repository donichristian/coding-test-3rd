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
    include=["app.tasks.document_tasks", "app.tasks.chat_tasks"]
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
    broker_connection_retry_on_startup = True
)

@worker_process_init.connect
def initialize_lazy_model_loader(sender, **kwargs):
    """
    Initialize lazy model loader when Celery worker starts.

    This sets up the lazy loading system without pre-downloading models,
    reducing startup time and Docker image size.
    """
    try:
        logger.info("Initializing lazy model loader in Celery worker...")

        # Import lazy model loader to ensure it's available
        import sys
        sys.path.insert(0, '/app')
        from lazy_model_loader import preload_critical_models

        # Preload only the most critical models (optional, can be removed for maximum size reduction)
        try:
            preload_critical_models()
            logger.info("✓ Critical models preloaded successfully in worker")
        except Exception as e:
            logger.warning(f"Could not preload critical models: {e}")
            import traceback
            traceback.print_exc()

        logger.info("✓ Lazy model loader initialized in Celery worker")

    except Exception as e:
        logger.error(f"Failed to initialize lazy model loader: {e}")
        import traceback
        traceback.print_exc()

def get_cached_model(model_name: str):
    """Get a cached model instance using lazy loading."""
    import sys
    sys.path.insert(0, '/app')
    from lazy_model_loader import (
        get_docling_converter,
        get_sentence_transformer,
        get_openai_client,
        get_gemini_client
    )

    try:
        if model_name == 'docling_converter':
            return get_docling_converter()
        elif model_name == 'sentence_transformers':
            return get_sentence_transformer()
        elif model_name == 'openai_client':
            return get_openai_client()
        elif model_name == 'gemini_client':
            return get_gemini_client()
        else:
            logger.warning(f"Unknown model name: {model_name}")
            return None
    except Exception as e:
        logger.error(f"Failed to get cached model {model_name}: {e}")
        return None

# For development - allows running celery from command line
if __name__ == "__main__":
    celery_app.start()