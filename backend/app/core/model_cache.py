"""
Global model cache for ML models.

This module provides a simple shared cache for ML models that works across
both FastAPI web server and Celery worker contexts.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global model cache - shared across all processes
_global_model_cache: Dict[str, Any] = {}


def get_cached_model(model_name: str) -> Optional[Any]:
    """Get a cached model instance."""
    return _global_model_cache.get(model_name)


def set_cached_model(model_name: str, model: Any) -> None:
    """Set a model in the global cache."""
    _global_model_cache[model_name] = model
    logger.debug(f"Cached model: {model_name}")


def is_model_cached(model_name: str) -> bool:
    """Check if a model is cached."""
    return model_name in _global_model_cache and _global_model_cache[model_name] is not None