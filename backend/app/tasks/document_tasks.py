"""
Celery tasks for document processing using Docling.

This module contains asynchronous tasks for processing uploaded documents
using Docling's native capabilities for document conversion and chunking.
"""

from app.core.celery_app import celery_app
from app.services.document_processor import DoclingDocumentProcessor
from app.services.document_processor import DocumentService
import logging

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3)
def process_document_task(self, document_id: int, file_path: str, fund_id: int):
    """
    Asynchronous task to process a document using Docling.

    Args:
        document_id: Database document ID
        file_path: Path to the uploaded file
        fund_id: Associated fund ID

    Returns:
        Dictionary with processing results
    """
    logger.info(f"Starting Docling document processing task for document {document_id}")

    try:
        # Use the new DoclingDocumentProcessor
        document_service = DocumentService()

        # Ensure fund exists and get correct fund_id
        fund_id = document_service.ensure_fund_exists(fund_id)

        # Update document status to processing
        document_service.update_document_status(document_id, "processing")

        # Process the document using Docling
        processor = DoclingDocumentProcessor()
        result = processor.process_document_sync(file_path, document_id, fund_id)

        # Update document status based on result
        if result and result.status and result.status.value == "success":
            document_service.update_document_status(document_id, "completed")
            logger.info(f"Document processing completed successfully for document {document_id}")
        else:
            error_msg = result.error if result and hasattr(result, 'error') else "Processing failed"
            document_service.update_document_status(document_id, "failed", error_msg)
            logger.warning(f"Document processing failed for document {document_id}: {error_msg}")

        return {
            "document_id": document_id,
            "status": result.status.value if result else "error",
            "tables_extracted": result.tables_extracted if result else {"capital_calls": 0, "distributions": 0, "adjustments": 0},
            "text_chunks_created": result.text_chunks_created if result else 0,
            "processing_method": result.processing_method if result else "docling_error",
            "total_pages": result.total_pages if result else 0,
            "processing_time": result.processing_time if result else 0.0
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


@celery_app.task(bind=True, max_retries=3)
def batch_process_documents_task(self, document_ids: list, file_paths: list, fund_id: int):
    """
    Batch process multiple documents using Docling.

    Args:
        document_ids: List of database document IDs
        file_paths: List of file paths
        fund_id: Associated fund ID

    Returns:
        Dictionary with batch processing results
    """
    logger.info(f"Starting batch Docling document processing task for {len(document_ids)} documents")

    try:
        document_service = DocumentService()
        processor = DoclingDocumentProcessor()
        
        # Ensure fund exists
        fund_id = document_service.ensure_fund_exists(fund_id)

        results = {
            "total_documents": len(document_ids),
            "successful": 0,
            "failed": 0,
            "documents": []
        }

        for document_id, file_path in zip(document_ids, file_paths):
            try:
                # Update status to processing
                document_service.update_document_status(document_id, "processing")

                # Process document
                result = processor.process_document_sync(file_path, document_id, fund_id)

                # Update status based on result
                if result and result.status and result.status.value == "success":
                    document_service.update_document_status(document_id, "completed")
                    results["successful"] += 1
                    status = "completed"
                else:
                    error_msg = result.error if result and hasattr(result, 'error') else "Processing failed"
                    document_service.update_document_status(document_id, "failed", error_msg)
                    results["failed"] += 1
                    status = "failed"

                results["documents"].append({
                    "document_id": document_id,
                    "status": status,
                    "tables_extracted": result.tables_extracted if result else {"capital_calls": 0, "distributions": 0, "adjustments": 0},
                    "text_chunks_created": result.text_chunks_created if result else 0,
                    "processing_method": result.processing_method if result else "docling_error",
                    "error": result.error if result else "Unknown error"
                })

            except Exception as doc_error:
                logger.error(f"Failed to process document {document_id}: {doc_error}")
                document_service.update_document_status(document_id, "failed", str(doc_error))
                results["failed"] += 1
                results["documents"].append({
                    "document_id": document_id,
                    "status": "failed",
                    "error": str(doc_error)
                })

        logger.info(f"Batch processing completed: {results['successful']} successful, {results['failed']} failed")
        return results

    except Exception as e:
        logger.error(f"Batch document processing task failed: {e}")
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))


@celery_app.task
def prefetch_docling_models_task():
    """
    Prefetch Docling models for offline usage.
    
    This task downloads all required Docling models to avoid runtime downloads.
    """
    logger.info("Starting Docling model prefetching task")
    
    try:
        # Import Docling utilities
        from docling.utils.model_downloader import download_models
        
        # Download models
        download_models()
        
        logger.info("✓ Docling models prefetching completed successfully")
        return {"status": "success", "message": "Models downloaded successfully"}
        
    except Exception as e:
        logger.error(f"Docling model prefetching failed: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(bind=True, max_retries=2)
def cleanup_processing_cache_task(self):
    """
    Cleanup task for Docling processing cache and temporary files.
    
    This task cleans up temporary files and cache data from document processing.
    """
    logger.info("Starting cache cleanup task")
    
    try:
        import shutil
        import os
        from pathlib import Path
        
        # Clean up temporary directories
        temp_dirs = [
            "/tmp/docling_*",
            "/tmp/docling_*",
            "./temp",
            "./cache"
        ]
        
        cleaned_count = 0
        for pattern in temp_dirs:
            for temp_path in Path(".").glob(pattern.replace("*", "*")):
                if temp_path.is_dir():
                    try:
                        shutil.rmtree(temp_path)
                        cleaned_count += 1
                        logger.debug(f"Cleaned up directory: {temp_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up {temp_path}: {e}")
        
        logger.info(f"Cache cleanup completed: {cleaned_count} directories removed")
        return {"status": "success", "cleaned_directories": cleaned_count}
        
    except Exception as e:
        logger.error(f"Cache cleanup task failed: {e}")
        # Don't retry on cleanup failures
        return {"status": "error", "message": str(e)}
