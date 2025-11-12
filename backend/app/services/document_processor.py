"""
Document processing service using Docling's native capabilities.

This service leverages Docling's comprehensive document processing pipeline
for extracting structured data from PDFs and other document formats.
"""
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional

from docling.document_converter import DocumentConverter

from app.core.config import settings
from app.services.vector_store import VectorStore
from app.db.session import SessionLocal
from app.models.transaction import CapitalCall, Distribution, Adjustment
from app.models.document import Document
from app.models.fund import Fund
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Enumeration of processing statuses."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


@dataclass
class ProcessingResult:
    """Data class for processing results."""
    status: ProcessingStatus
    document_id: int
    fund_id: int
    tables_extracted: Dict[str, int]
    text_chunks_created: int
    total_pages: int
    processing_time: float
    processing_method: str
    error: Optional[str] = None
    note: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value if isinstance(self.status, ProcessingStatus) else self.status,
            "document_id": self.document_id,
            "fund_id": self.fund_id,
            "tables_extracted": self.tables_extracted,
            "text_chunks_created": self.text_chunks_created,
            "total_pages": self.total_pages,
            "processing_time": self.processing_time,
            "processing_method": self.processing_method,
            "error": self.error,
            "note": self.note
        }


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


class DoclingDocumentProcessor:
    """
    Document processor leveraging Docling's native capabilities.
    
    This class uses Docling's built-in document processing, table extraction,
    and chunking features, eliminating the need for custom implementations.
    """

    def __init__(
        self,
        converter: Optional[DocumentConverter] = None,
        db_session: Optional[Session] = None
    ):
        """
        Initialize the Docling-based document processor.
        
        Args:
            converter: Docling DocumentConverter instance (uses cached version if available)
            db_session: SQLAlchemy session for database operations
        """
        self.converter = converter or self._get_cached_converter()
        self.db_session = db_session
        self.data_parser = DataParser()
        self.vector_store = VectorStore(db=db_session)

    def _get_cached_converter(self) -> DocumentConverter:
        """
        Get cached Docling converter from Celery if available.
        
        Returns:
            DocumentConverter instance
        """
        try:
            from app.core.celery_app import get_cached_model
            cached_converter = get_cached_model('docling_converter')
            if cached_converter is not None:
                logger.info("Using cached Docling converter")
                return cached_converter
        except (ImportError, Exception):
            pass  # Not in Celery context or cache miss

        # Fallback to creating a new converter
        logger.info("Creating new Docling converter")
        return DocumentConverter()

    async def process_document(
        self, 
        file_path: str, 
        document_id: int, 
        fund_id: int
    ) -> ProcessingResult:
        """
        Process a document using Docling's native capabilities.

        Args:
            file_path: Path to document file
            document_id: Database document ID
            fund_id: Associated fund ID

        Returns:
            Processing result with statistics
        """
        return await self._process_document_async(file_path, document_id, fund_id)

    async def _process_document_async(
        self, 
        file_path: str, 
        document_id: int, 
        fund_id: int
    ) -> ProcessingResult:
        """
        Process document asynchronously using Docling's pipeline.

        Args:
            file_path: Path to document file
            document_id: Database document ID
            fund_id: Associated fund ID

        Returns:
            Processing result with comprehensive statistics
        """
        start_time = time.time()

        try:
            logger.info(f"Starting Docling processing for document {document_id}")
            
            # Convert document using Docling's native pipeline
            conversion_result = self.converter.convert(file_path)
            doc = conversion_result.document

            # Extract content using Docling's native features
            text_content = self._extract_text_content(doc)
            tables_data = self._extract_tables_with_docling(doc)

            # Use Docling's hybrid chunking for optimal text segmentation
            text_chunks = await self._create_chunks_with_docling(text_content, document_id, fund_id)

            # Store data in databases
            await self._store_processed_data(text_chunks, tables_data, document_id, fund_id)

            # Generate processing statistics
            processing_time = time.time() - start_time
            total_pages = getattr(doc, 'pages', []) and len(doc.pages) or 0
            
            tables_extracted = {
                "capital_calls": len(tables_data.get("capital_calls", [])),
                "distributions": len(tables_data.get("distributions", [])),
                "adjustments": len(tables_data.get("adjustments", []))
            }

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                document_id=document_id,
                fund_id=fund_id,
                tables_extracted=tables_extracted,
                text_chunks_created=len(text_chunks),
                total_pages=total_pages,
                processing_time=processing_time,
                processing_method="docling_native",
                note="Document processed using Docling's native capabilities"
            )

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                document_id=document_id,
                fund_id=fund_id,
                tables_extracted={"capital_calls": 0, "distributions": 0, "adjustments": 0},
                text_chunks_created=0,
                total_pages=0,
                processing_time=time.time() - start_time,
                processing_method="error",
                error=str(e)
            )

    def _extract_text_content(self, doc) -> List[Dict[str, Any]]:
        """
        Extract text content using Docling's native document model.
        
        Args:
            doc: Docling Document object
            
        Returns:
            List of text content items with Docling metadata
        """
        try:
            # Use Docling's export to markdown for structured text
            markdown_content = doc.export_to_markdown()
            
            return [{
                "page": 1,
                "content": markdown_content,
                "type": "markdown_text",
                "metadata": {
                    "document_structure": getattr(doc, 'content_items', []),
                    "total_text_length": len(markdown_content),
                    "extraction_method": "docling_markdown_export"
                }
            }]
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return [{
                "page": 1,
                "content": "Text extraction failed",
                "type": "error",
                "metadata": {"error": str(e)}
            }]

    def _extract_tables_with_docling(self, doc) -> Dict[str, Any]:
        """
        Extract tables using Docling's native table structure recognition.
        
        Args:
            doc: Docling Document object
            
        Returns:
            Dictionary with classified tables and Docling metadata
        """
        try:
            tables_data = {
                "capital_calls": [],
                "distributions": [],
                "adjustments": [],
                "processing_method": "docling_native"
            }

            # Get tables from Docling's document model
            if hasattr(doc, 'tables') and doc.tables:
                logger.info(f"Found {len(doc.tables)} tables via Docling")
                
                for table in doc.tables:
                    # Use Docling's built-in DataFrame export
                    if hasattr(table, 'export_to_dataframe'):
                        df = table.export_to_dataframe(doc=doc)
                        if not df.empty:
                            # Convert to records format
                            table_records = df.to_dict('records')
                            
                            # Classify table using Docling's table type if available
                            table_type = getattr(table, 'table_type', 'unknown')
                            
                            # Apply business logic classification
                            classified_type = self._classify_table_for_business_logic(
                                table_records, table_type
                            )
                            
                            if classified_type:
                                tables_data[classified_type].append({
                                    "headers": list(df.columns),
                                    "rows": table_records,
                                    "metadata": {
                                        "docling_table_type": table_type,
                                        "confidence": getattr(table, 'confidence', 1.0),
                                        "extraction_method": "docling_dataframe",
                                        "page_number": getattr(table, 'page_number', 1)
                                    }
                                })

            logger.info(f"Extracted tables: {sum(len(v) for k, v in tables_data.items() if k != 'processing_method')}")
            return tables_data

        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return {
                "capital_calls": [],
                "distributions": [],
                "adjustments": [],
                "processing_method": "error",
                "error": str(e)
            }

    def _classify_table_for_business_logic(
        self, 
        table_records: List[Dict[str, Any]], 
        docling_table_type: str
    ) -> Optional[str]:
        """
        Classify table type for business logic using Docling insights.
        
        Args:
            table_records: Table data as list of records
            docling_table_type: Docling's table classification
            
        Returns:
            Business logic table type or None
        """
        if not table_records:
            return None

        # Use Docling's table type as primary indicator
        if docling_table_type and docling_table_type != 'unknown':
            if 'capital' in docling_table_type.lower() or 'call' in docling_table_type.lower():
                return "capital_calls"
            elif 'distribution' in docling_table_type.lower() or 'return' in docling_table_type.lower():
                return "distributions"
            elif 'fee' in docling_table_type.lower() or 'expense' in docling_table_type.lower():
                return "adjustments"

        # Fallback to content-based classification
        headers = list(table_records[0].keys()) if table_records else []
        headers_text = " ".join(headers).lower()
        
        # Enhanced classification using business keywords
        if any(keyword in headers_text for keyword in ["call", "capital", "commitment"]):
            return "capital_calls"
        elif any(keyword in headers_text for keyword in ["distribution", "return", "dividend"]):
            return "distributions"
        elif any(keyword in headers_text for keyword in ["fee", "expense", "adjustment"]):
            return "adjustments"
            
        return None

    async def _create_chunks_with_docling(
        self, 
        text_content: List[Dict[str, Any]], 
        document_id: int, 
        fund_id: int
    ) -> List[Dict[str, Any]]:
        """
        Create text chunks using structured approach based on Docling content.
        
        Args:
            text_content: Extracted text content
            document_id: Database document ID
            fund_id: Associated fund ID
            
        Returns:
            List of text chunks with Docling metadata
        """
        try:
            # Process each text content item using structured approach
            all_chunks = []
            for content_item in text_content:
                content = content_item["content"]
                metadata = content_item["metadata"]
                
                # Create chunks based on document sections
                chunks = self._create_structured_chunks(content, metadata, document_id, fund_id)
                all_chunks.extend(chunks)
            
            logger.info(f"Created {len(all_chunks)} chunks using Docling's structured approach")
            return all_chunks

        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            # Fallback to simple chunking
            return self._fallback_chunking(text_content, document_id, fund_id)

    def _create_structured_chunks(
        self, 
        content: str, 
        metadata: Dict[str, Any], 
        document_id: int, 
        fund_id: int
    ) -> List[Dict[str, Any]]:
        """
        Create structured chunks based on document hierarchy.
        
        Args:
            content: Text content
            metadata: Document metadata
            document_id: Database document ID
            fund_id: Associated fund ID
            
        Returns:
            List of structured chunks
        """
        import re
        
        chunks = []
        
        # Split content into sections based on headings
        sections = re.split(r'\n#{1,3}\s+', content)
        
        for i, section in enumerate(sections):
            if section.strip():
                chunk = {
                    "content": section.strip(),
                    "metadata": {
                        **metadata,
                        "document_id": document_id,
                        "fund_id": fund_id,
                        "chunk_index": i,
                        "chunk_type": "section",
                        "chunking_method": "docling_structured",
                        "content_length": len(section),
                        "word_count": len(section.split())
                    }
                }
                chunks.append(chunk)
        
        # If no sections found, create chunks of fixed size
        if not chunks and content:
            chunk_size = settings.CHUNK_SIZE
            overlap = settings.CHUNK_OVERLAP
            
            for i in range(0, len(content), chunk_size - overlap):
                chunk_content = content[i:i + chunk_size].strip()
                if chunk_content:
                    chunk = {
                        "content": chunk_content,
                        "metadata": {
                            **metadata,
                            "document_id": document_id,
                            "fund_id": fund_id,
                            "chunk_index": len(chunks),
                            "chunk_type": "fixed_size",
                            "chunking_method": "docling_fallback",
                            "content_length": len(chunk_content),
                            "word_count": len(chunk_content.split())
                        }
                    }
                    chunks.append(chunk)
        
        return chunks

    def _fallback_chunking(
        self, 
        text_content: List[Dict[str, Any]], 
        document_id: int, 
        fund_id: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback chunking when Docling chunking is unavailable.
        
        Args:
            text_content: Extracted text content
            document_id: Database document ID
            fund_id: Associated fund ID
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        for content_item in text_content:
            content = content_item["content"]
            metadata = content_item["metadata"]
            
            # Simple character-based chunking as fallback
            chunk_size = settings.CHUNK_SIZE
            overlap = settings.CHUNK_OVERLAP
            
            for i in range(0, len(content), chunk_size - overlap):
                chunk_content = content[i:i + chunk_size].strip()
                if chunk_content:
                    chunk = {
                        "content": chunk_content,
                        "metadata": {
                            **metadata,
                            "document_id": document_id,
                            "fund_id": fund_id,
                            "chunk_index": len(chunks),
                            "chunk_type": "fallback",
                            "chunking_method": "simple_character_split",
                            "content_length": len(chunk_content),
                            "word_count": len(chunk_content.split())
                        }
                    }
                    chunks.append(chunk)
        
        return chunks

    async def _store_processed_data(
        self,
        text_chunks: List[Dict[str, Any]],
        tables_data: Dict[str, Any],
        document_id: int,
        fund_id: int
    ) -> None:
        """
        Store processed data in databases.
        
        Args:
            text_chunks: Processed text chunks
            tables_data: Extracted table data
            document_id: Database document ID
            fund_id: Associated fund ID
        """
        try:
            # Store text chunks in vector database
            if text_chunks:
                chunks_stored = self.vector_store.store_chunks_sync(text_chunks)
                logger.info(f"Stored {len(text_chunks)} text chunks" if chunks_stored else "Failed to store text chunks")

            # Store table data in PostgreSQL
            tables_stored = await self._store_table_data(tables_data, fund_id)
            logger.info(f"Stored {tables_stored} table records")

        except Exception as e:
            logger.error(f"Failed to store processed data: {e}")
            raise

    async def _store_table_data(self, tables_data: Dict[str, Any], fund_id: int) -> int:
        """
        Store table data in PostgreSQL database.
        
        Args:
            tables_data: Extracted table data
            fund_id: Associated fund ID
            
        Returns:
            Number of stored table records
        """
        db = SessionLocal()
        stored_count = 0
        
        try:
            # Clear existing data for this fund
            db.query(CapitalCall).filter(CapitalCall.fund_id == fund_id).delete()
            db.query(Distribution).filter(Distribution.fund_id == fund_id).delete()
            db.query(Adjustment).filter(Adjustment.fund_id == fund_id).delete()
            db.commit()

            # Process each table type
            objects_to_insert = []
            
            for table_type, tables in tables_data.items():
                if table_type == "processing_method":
                    continue

                for table in tables:
                    rows = table.get("rows", [])
                    for row in rows:
                        obj = self._create_transaction_object(table_type, row, fund_id)
                        if obj:
                            objects_to_insert.append(obj)

            # Bulk insert
            if objects_to_insert:
                db.add_all(objects_to_insert)
                db.commit()
                stored_count = len(objects_to_insert)

            return stored_count

        except Exception as e:
            db.rollback()
            logger.error(f"Database error storing tables: {e}")
            raise
        finally:
            db.close()

    def _create_transaction_object(self, table_type: str, row: Dict[str, Any], fund_id: int):
        """
        Create transaction object from table row.
        
        Args:
            table_type: Type of table (capital_calls, distributions, adjustments)
            row: Table row data
            fund_id: Associated fund ID
            
        Returns:
            Transaction object or None if creation fails
        """
        try:
            if table_type == "capital_calls":
                call_date = self.data_parser.parse_date(row.get("date") or row.get("Date"))
                amount = self.data_parser.parse_amount(row.get("amount") or row.get("Amount") or row.get("Called"))

                if call_date and amount is not None:
                    return CapitalCall(
                        fund_id=fund_id,
                        call_date=call_date,
                        amount=amount,
                        call_type=row.get("type") or row.get("Type") or "Regular",
                        description=row.get("description") or row.get("Description") or ""
                    )

            elif table_type == "distributions":
                distribution_date = self.data_parser.parse_date(row.get("date") or row.get("Date"))
                amount = self.data_parser.parse_amount(row.get("amount") or row.get("Amount") or row.get("Distributed"))
                is_recallable = self.data_parser.parse_boolean(row.get("recallable") or row.get("Recallable"))

                if distribution_date and amount is not None:
                    return Distribution(
                        fund_id=fund_id,
                        distribution_date=distribution_date,
                        amount=amount,
                        distribution_type=row.get("type") or row.get("Type") or "Return",
                        is_recallable=is_recallable,
                        description=row.get("description") or row.get("Description") or ""
                    )

            elif table_type == "adjustments":
                adjustment_date = self.data_parser.parse_date(row.get("date") or row.get("Date"))
                amount = self.data_parser.parse_amount(row.get("amount") or row.get("Amount") or row.get("Adjustment"))

                if adjustment_date and amount is not None:
                    return Adjustment(
                        fund_id=fund_id,
                        adjustment_date=adjustment_date,
                        amount=amount,
                        adjustment_type=row.get("type") or row.get("Type") or "Fee",
                        category=row.get("category") or row.get("Category") or "General",
                        is_contribution_adjustment=amount < 0,
                        description=row.get("description") or row.get("Description") or ""
                    )

        except Exception as e:
            logger.warning(f"Failed to create object for {table_type}: {e}")

        return None

    def process_document_sync(
        self, 
        file_path: str, 
        document_id: int, 
        fund_id: int
    ) -> ProcessingResult:
        """
        Process document synchronously for Celery tasks.
        
        Args:
            file_path: Path to document file
            document_id: Database document ID
            fund_id: Associated fund ID
            
        Returns:
            Processing result
        """
        import asyncio

        # Create new event loop for synchronous context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._process_document_async(file_path, document_id, fund_id)
            )
        finally:
            loop.close()


class DataParser:
    """Data parsing utilities for financial data."""

    DATE_FORMATS = [
        "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y",
        "%b %d, %Y", "%Y/%m/%d", "%d-%b-%Y", "%d-%B-%Y"
    ]

    DATE_PATTERNS = [
        r"(\d{4})-(\d{2})-(\d{2})",
        r"(\d{2})/(\d{2})/(\d{4})",
        r"(\d{2})-(\d{2})-(\d{4})"
    ]

    def parse_date(self, date_str: str):
        """Parse date string into date object."""
        if not date_str:
            return None

        import re
        from datetime import datetime

        # Try direct parsing
        for fmt in self.DATE_FORMATS:
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue

        # Try regex extraction
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, date_str)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) == 3:
                        y, m, d = groups
                        if len(y) == 4:  # YYYY-MM-DD
                            return datetime(int(y), int(m), int(d)).date()
                        else:  # Assume MM/DD/YYYY
                            return datetime(int(d), int(m), int(y)).date()
                except ValueError:
                    continue

        return None

    def parse_amount(self, amount_str: str):
        """Parse amount string into float."""
        if not amount_str:
            return None

        import re
        from datetime import datetime

        # Clean string
        cleaned = re.sub(r"[$,€£¥₹\s]", "", str(amount_str))

        # Handle parentheses for negatives
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]

        try:
            return float(cleaned)
        except ValueError:
            return None

    def parse_boolean(self, bool_str: str) -> bool:
        """Parse boolean string."""
        if not bool_str:
            return False

        return str(bool_str).lower() in ["yes", "true", "1", "y", "recallable"]
