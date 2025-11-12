"""
Document service for handling document-related business logic.
"""

from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.models.document import Document
from app.models.fund import Fund
import logging

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for document-related operations."""

    def __init__(self, db: Session = None):
        """Initialize with optional database session."""
        self.db = db or SessionLocal()

    def ensure_fund_exists(self, fund_id: int) -> int:
        """
        Ensure fund exists, create default if needed.

        Args:
            fund_id: Requested fund ID

        Returns:
            Valid fund ID (either existing or newly created)
        """
        fund = self.db.query(Fund).filter(Fund.id == fund_id).first()
        if not fund:
            # Create default fund if it doesn't exist
            fund = Fund(
                name="Default Fund",
                gp_name="Default GP",
                fund_type="Default",
                vintage_year=2024
            )
            self.db.add(fund)
            self.db.commit()
            self.db.refresh(fund)
            logger.info(f"Created default fund with ID {fund.id}")
            return fund.id

        return fund_id

    def update_document_status(
        self,
        document_id: int,
        status: str,
        error_message: str = None
    ) -> bool:
        """
        Update document parsing status.

        Args:
            document_id: Document ID
            status: New status (pending, processing, completed, failed)
            error_message: Optional error message

        Returns:
            True if update successful, False otherwise
        """
        try:
            document = self.db.query(Document).filter(Document.id == document_id).first()
            if not document:
                logger.error(f"Document {document_id} not found")
                return False

            # Update fund_id if it was None
            if document.fund_id is None:
                document.fund_id = self.ensure_fund_exists(1)  # Default fund

            document.parsing_status = status
            if error_message:
                document.error_message = error_message
            elif status == "completed":
                document.error_message = None

            self.db.commit()
            logger.info(f"Updated document {document_id} status to {status}")
            return True

        except Exception as e:
            logger.error(f"Failed to update document {document_id} status: {e}")
            self.db.rollback()
            return False

    def get_document(self, document_id: int) -> Document:
        """
        Get document by ID.

        Args:
            document_id: Document ID

        Returns:
            Document instance or None if not found
        """
        return self.db.query(Document).filter(Document.id == document_id).first()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close database session."""
        if self.db:
            self.db.close()