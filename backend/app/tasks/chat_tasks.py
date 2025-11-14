"""
Celery tasks for chat processing using RAG.

This module contains asynchronous tasks for processing chat queries
using the RAG engine with LangChain and LLM providers.
"""

from app.core.celery_app import celery_app
from app.services.query_engine import QueryEngine
from app.db.session import SessionLocal
import logging

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=2)
def process_chat_query_task(self, query: str, fund_id: int = None, conversation_id: str = None, document_id: int = None):
    """
    Asynchronous task to process a chat query using RAG.

    Args:
        query: User question
        fund_id: Optional fund ID for context filtering
        conversation_id: Optional conversation ID for context
        document_id: Optional document ID (deprecated, use fund_id)

    Returns:
        Dictionary with query results
    """
    logger.info(f"Starting chat query processing task for query: {query[:50]}...")

    try:
        # Create database session
        db = SessionLocal()

        try:
            # Create query engine
            query_engine = QueryEngine(db)

            # Get conversation history if conversation_id provided
            conversation_history = []
            if conversation_id:
                # This would need to be implemented to retrieve conversation history
                # For now, we'll pass empty history
                pass

            # Debug: Check vector store stats before processing
            try:
                vector_store_stats = query_engine.rag_engine.vector_store.get_stats()
                logger.info(f"Vector store stats before query: {vector_store_stats}")
            except Exception as stats_error:
                logger.warning(f"Could not get vector store stats: {stats_error}")
                vector_store_stats = {"error": str(stats_error)}

            # Process query synchronously (since we're already in async context)
            result = query_engine.process_query_sync(
                query=query,
                fund_id=fund_id,
                document_id=document_id,
                conversation_history=conversation_history
            )

            logger.info(f"Chat query processing completed successfully")
            return result

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Chat query processing task failed: {e}")

        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=30 * (2 ** self.request.retries))