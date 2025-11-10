"""
Celery tasks for document processing.

This module contains asynchronous tasks for processing uploaded documents.
"""

from app.core.celery_app import celery_app
from app.services.document_processor import DocumentProcessor
from app.db.session import SessionLocal
from app.models.document import Document
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3)
def process_document_task(self, document_id: int, file_path: str, fund_id: int = None):
    """
    Asynchronous task to process a document.

    Args:
        document_id: Database document ID
        file_path: Path to the uploaded file
        fund_id: Associated fund ID (optional)

    Returns:
        Dictionary with processing results
    """
    logger.info(f"Starting document processing task for document {document_id}")

    db = SessionLocal()
    try:
        # Update document status to processing
        document = db.query(Document).filter(Document.id == document_id).first()
        if document is None:
            logger.error(f"Document {document_id} not found in database")
            return {
                "document_id": document_id,
                "status": "error",
                "error": "Document not found",
                "tables_extracted": {"capital_calls": 0, "distributions": 0, "adjustments": 0},
                "text_chunks_created": 0
            }
        document.parsing_status = "processing"
        db.commit()

        # Process the document using synchronous version
        # Pass the database session to avoid None errors in VectorStore
        processor = DocumentProcessor(db_session=db)
        result = processor.process_document_sync(file_path, document_id, fund_id)

        # Update document status based on result
        if result and result.status and result.status.value == "success":
            document.parsing_status = "completed"
            document.error_message = None
        else:
            document.parsing_status = "failed"
            document.error_message = result.error if result and hasattr(result, 'error') else "Processing failed"

        db.commit()
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
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.parsing_status = "failed"
                document.error_message = str(e)
                db.commit()
        except Exception as db_error:
            logger.error(f"Failed to update document status: {db_error}")
            # Don't re-raise here as we're already in error handling

        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))

    finally:
        db.close()