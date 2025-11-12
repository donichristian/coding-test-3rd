"""
Celery tasks for document processing.

This module contains asynchronous tasks for processing uploaded documents.
"""

from app.core.celery_app import celery_app
from app.services.document_processor import DocumentProcessor
from app.services.document_service import DocumentService
import logging

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3)
def process_document_task(self, document_id: int, file_path: str, fund_id: int):
    """
    Asynchronous task to process a document.

    Args:
        document_id: Database document ID
        file_path: Path to the uploaded file
        fund_id: Associated fund ID

    Returns:
        Dictionary with processing results
    """
    logger.info(f"Starting document processing task for document {document_id}")

    try:
        # Use DocumentService to handle all document-related operations
        document_service = DocumentService()

        # Ensure fund exists and get correct fund_id
        fund_id = document_service.ensure_fund_exists(fund_id)

        # Update document status to processing
        document_service.update_document_status(document_id, "processing")

        # Process the document
        processor = DocumentProcessor()
        result = processor.process_document_sync(file_path, document_id, fund_id)

        # Update document status based on result
        if result and result.status and result.status.value == "success":
            document_service.update_document_status(document_id, "completed")
        else:
            error_msg = result.error if result and hasattr(result, 'error') else "Processing failed"
            document_service.update_document_status(document_id, "failed", error_msg)

        logger.info(f"Document processing completed for document {document_id}")

        return {
            "document_id": document_id,
            "status": result.status.value,
            "tables_extracted": result.tables_extracted,
            "text_chunks_created": result.text_chunks_created
        }

    except Exception as e:
        logger.error(f"Document processing task failed for document {document_id}: {e}")

        # Update document status to failed
        try:
            document_service = DocumentService()
            document_service.update_document_status(document_id, "failed", str(e))
        except Exception as db_error:
            logger.error(f"Failed to update document status: {db_error}")

        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))
