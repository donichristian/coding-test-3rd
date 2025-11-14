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
from sqlalchemy import text

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
        Update document parsing status efficiently.

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
        from app.services.table_parser import TableParser
        import json
        
        self.converter = converter or self._get_cached_converter()
        self.db_session = db_session
        self.table_parser = TableParser()
        self.data_parser = DataParser()
        self.vector_store = VectorStore(db=db_session)
        self._text_serialization_formats = ["markdown", "text", "html", "json"]

    def _get_cached_converter(self) -> DocumentConverter:
        """
        Get Docling converter using lazy loading.

        Returns:
            DocumentConverter instance
        """
        try:
            from lazy_model_loader import get_docling_converter
            converter = get_docling_converter()
            logger.info("Using lazy-loaded Docling converter (OCR disabled)")
            return converter
        except Exception as e:
            logger.warning(f"Could not get lazy-loaded converter, creating new one: {e}")

        # Fallback to creating a new converter without OCR
        logger.info("Creating new Docling converter (OCR disabled for performance)")
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption
        from docling.datamodel.base_models import InputFormat

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False  # Performance optimization
        pipeline_options.do_table_structure = True

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

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
        Extract text content using Docling's multiple native serialization formats.
        
        Args:
            doc: Docling Document object
            
        Returns:
            List of text content items with comprehensive Docling metadata
        """
        try:
            # Extract using multiple Docling native export formats
            text_formats = {}
            for format_name in self._text_serialization_formats:
                try:
                    if format_name == "markdown":
                        text_formats["markdown"] = doc.export_to_markdown()
                    elif format_name == "text":
                        text_formats["text"] = doc.export_to_text()
                    elif format_name == "html":
                        text_formats["html"] = doc.export_to_html()
                    elif format_name == "json":
                        text_formats["json"] = json.dumps(doc.export_to_dict(), indent=2)
                except Exception as format_error:
                    logger.warning(f"Failed to export {format_name}: {format_error}")
                    text_formats[format_name] = ""

            # Extract comprehensive document metadata
            doc_metadata = self._extract_docling_document_metadata(doc)
            
            # Create content items for each format
            content_items = []
            for format_name, content in text_formats.items():
                if content:  # Only include non-empty formats
                    content_items.append({
                        "page": 1,
                        "content": content,
                        "type": f"{format_name}_text",
                        "format": format_name,
                        "metadata": {
                            **doc_metadata,
                            "extraction_method": f"docling_{format_name}_export",
                            "format_length": len(content),
                            "available_formats": list(text_formats.keys())
                        }
                    })

            return content_items if content_items else [{
                "page": 1,
                "content": "Text extraction failed",
                "type": "error",
                "metadata": {"error": "All text extraction formats failed"}
            }]
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return [{
                "page": 1,
                "content": "Text extraction failed",
                "type": "error",
                "metadata": {"error": str(e)}
            }]

    def _extract_docling_document_metadata(self, doc) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from Docling document structure.
        
        Args:
            doc: Docling Document object
            
        Returns:
            Dictionary with detailed document metadata
        """
        metadata = {
            "page_count": 0,
            "section_count": 0,
            "has_tables": False,
            "has_images": False,
            "document_structure": [],
            "paragraph_count": 0,
            "table_count": 0,
            "picture_count": 0
        }

        try:
            # Basic document properties
            if hasattr(doc, 'pages'):
                metadata["page_count"] = len(doc.pages)
            
            # Check for tables and images
            if hasattr(doc, 'tables') and doc.tables:
                metadata["has_tables"] = True
                metadata["table_count"] = len(doc.tables)
                
            if hasattr(doc, 'pictures') and doc.pictures:
                metadata["has_images"] = True
                metadata["picture_count"] = len(doc.pictures)

            # Extract content items for structure analysis
            if hasattr(doc, 'content_items'):
                content_items = doc.content_items or []
                metadata["document_structure"] = [
                    {
                        "type": getattr(item, 'type', 'unknown'),
                        "level": getattr(item, 'level', 0),
                        "text": str(getattr(item, 'text', ''))[:100],
                        "page": getattr(item, 'page', 1)
                    }
                    for item in content_items
                    if hasattr(item, 'text')
                ]
                
                # Count sections and paragraphs
                sections = [
                    item for item in content_items
                    if hasattr(item, 'type') and item.type in ['title', 'heading']
                ]
                metadata["section_count"] = len(sections)
                
                paragraphs = [
                    item for item in content_items
                    if hasattr(item, 'type') and item.type in ['paragraph', 'text']
                ]
                metadata["paragraph_count"] = len(paragraphs)

        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")

        return metadata

    def _extract_tables_with_docling(self, doc) -> Dict[str, Any]:
        """
        Extract tables using the TableParser service.
        
        Args:
            doc: Docling Document object
            
        Returns:
            Dictionary with classified tables and Docling metadata
        """
        return self.table_parser.extract_tables_with_docling(doc)

    async def _create_chunks_with_docling(
        self,
        text_content: List[Dict[str, Any]],
        document_id: int,
        fund_id: int
    ) -> List[Dict[str, Any]]:
        """
        Create text chunks using Docling's native document structure awareness.
        
        Args:
            text_content: Extracted text content from Docling
            document_id: Database document ID
            fund_id: Associated fund ID
            
        Returns:
            List of text chunks with Docling structure metadata
        """
        try:
            # Use Docling's document structure for intelligent chunking
            all_chunks = []
            
            # Process each format (prefer markdown for structure)
            for content_item in text_content:
                content = content_item["content"]
                metadata = content_item["metadata"]
                format_type = content_item.get("format", "unknown")
                
                # Use adaptive chunking strategy based on content structure
                if metadata.get("section_count", 0) > 2:
                    # Use structure-based chunking for well-structured documents
                    chunks = self._create_docling_structure_chunks(
                        content, metadata, document_id, fund_id, format_type
                    )
                else:
                    # Use semantic chunking for less structured content
                    chunks = self._create_semantic_chunks_with_docling(
                        content, metadata, document_id, fund_id, format_type
                    )
                
                all_chunks.extend(chunks)
            
            # Sort chunks by page and chunk index for consistent ordering
            all_chunks.sort(key=lambda x: (x["metadata"].get("page_number", 1), x["metadata"].get("chunk_index", 0)))
            
            logger.info(f"Created {len(all_chunks)} chunks using Docling's structure-aware approach")
            return all_chunks

        except Exception as e:
            logger.error(f"Docling chunking failed: {e}")
            # Fallback to simple chunking
            return self._fallback_chunking(text_content, document_id, fund_id)

    def _create_docling_structure_chunks(
        self,
        content: str,
        metadata: Dict[str, Any],
        document_id: int,
        fund_id: int,
        format_type: str
    ) -> List[Dict[str, Any]]:
        """
        Create chunks based on Docling's document structure hierarchy.
        
        Args:
            content: Text content in specified format
            metadata: Docling document metadata
            document_id: Database document ID
            fund_id: Associated fund ID
            format_type: Text format (markdown, text, etc.)
            
        Returns:
            List of chunks with Docling structure metadata
        """
        import re
        
        chunks = []
        
        if format_type == "markdown":
            # Use markdown headers for natural structure
            import re
            
            # Split by markdown headings while preserving structure
            lines = content.split('\n')
            current_section = []
            chunk_index = 0
            
            for line in lines:
                # Check if line is a heading
                if line.strip().startswith('#'):
                    # Save previous section
                    if current_section:
                        section_text = '\n'.join(current_section).strip()
                        if section_text:
                            chunk = self._create_enhanced_chunk(
                                section_text, "section", metadata, document_id, fund_id, chunk_index
                            )
                            chunks.append(chunk)
                            chunk_index += 1
                    
                    current_section = [line]  # Start new section with heading
                else:
                    current_section.append(line)
            
            # Add final section
            if current_section:
                section_text = '\n'.join(current_section).strip()
                if section_text:
                    chunk = self._create_enhanced_chunk(
                        section_text, "section", metadata, document_id, fund_id, chunk_index
                    )
                    chunks.append(chunk)
        
        else:
            # For non-markdown formats, use paragraph-based approach
            chunks = self._create_paragraph_chunks_with_docling(
                content, metadata, document_id, fund_id
            )
        
        return chunks

    def _create_enhanced_chunk(
        self,
        content: str,
        chunk_type: str,
        metadata: Dict[str, Any],
        document_id: int,
        fund_id: int,
        chunk_index: int = None
    ) -> Dict[str, Any]:
        """
        Create an enhanced chunk with Docling-specific metadata.
        
        Args:
            content: Chunk text content
            chunk_type: Type of chunk (section, semantic, paragraph)
            metadata: Docling document metadata
            document_id: Database document ID
            fund_id: Associated fund ID
            chunk_index: Optional chunk index
            
        Returns:
            Dictionary representing an enhanced text chunk
        """
        if chunk_index is None:
            chunk_index = 0
            
        # Calculate word count and other metrics
        words = content.split()
        char_count = len(content)
        
        return {
            "content": content,
            "metadata": {
                **metadata,
                "document_id": document_id,
                "fund_id": fund_id,
                "chunk_index": chunk_index,
                "chunk_type": chunk_type,
                "chunking_method": "docling_enhanced",
                "content_length": char_count,
                "word_count": len(words),
                "has_structure": chunk_type == "section",
                "processing_method": "docling_native_enhanced"
            }
        }

    def _create_paragraph_chunks_with_docling(
        self,
        content: str,
        metadata: Dict[str, Any],
        document_id: int,
        fund_id: int
    ) -> List[Dict[str, Any]]:
        """
        Create chunks based on paragraphs using Docling structure analysis.
        
        Args:
            content: Text content
            metadata: Docling document metadata
            document_id: Database document ID
            fund_id: Associated fund ID
            
        Returns:
            List of paragraph-based chunks
        """
        import re
        
        # Split by paragraphs (double newlines or single newlines with proper spacing)
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        chunk_index = 0
        
        for paragraph in paragraphs:
            if paragraph.strip():
                chunk = self._create_enhanced_chunk(
                    paragraph.strip(), "paragraph", metadata, document_id, fund_id, chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks

    def _create_semantic_chunks_with_docling(
        self,
        content: str,
        metadata: Dict[str, Any],
        document_id: int,
        fund_id: int,
        format_type: str
    ) -> List[Dict[str, Any]]:
        """
        Create semantically-aware chunks using Docling content analysis.
        
        Args:
            content: Text content
            metadata: Docling document metadata
            document_id: Database document ID
            fund_id: Associated fund ID
            format_type: Text format
            
        Returns:
            List of semantic chunks with Docling metadata
        """
        import re
        
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        current_chunk = ""
        chunk_index = 0
        
        # Check if content has clear semantic boundaries
        paragraph_count = metadata.get("paragraph_count", 0)
        use_paragraphs = paragraph_count > 5
        
        if use_paragraphs:
            # Split by paragraphs for better semantic boundaries
            paragraphs = re.split(r'\n\s*\n', content)
            for paragraph in paragraphs:
                if paragraph.strip():
                    if len(current_chunk) + len(paragraph) > settings.CHUNK_SIZE and current_chunk:
                        chunk = self._create_enhanced_chunk(
                            current_chunk.strip(), "semantic", metadata, document_id, fund_id, chunk_index
                        )
                        chunks.append(chunk)
                        
                        # Start new chunk
                        current_chunk = paragraph
                        chunk_index += 1
                    else:
                        if current_chunk:
                            current_chunk += "\n\n" + paragraph
                        else:
                            current_chunk = paragraph
        else:
            # Use sentence-based chunking
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
                
                if len(potential_chunk) > settings.CHUNK_SIZE and current_chunk:
                    chunk = self._create_enhanced_chunk(
                        current_chunk.strip(), "semantic", metadata, document_id, fund_id, chunk_index
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    overlap_words = current_chunk.split()[-settings.CHUNK_OVERLAP//4:]
                    current_chunk = " ".join(overlap_words) + " " + sentence if overlap_words else sentence
                    chunk_index += 1
                else:
                    current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_enhanced_chunk(
                current_chunk.strip(), "semantic", metadata, document_id, fund_id, chunk_index
            )
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
            # Store text chunks in vector database first
            if text_chunks:
                logger.info("Starting vector store operations...")
                chunks_stored = self.vector_store.store_chunks_sync(text_chunks)
                if chunks_stored:
                    logger.info(f"✓ Successfully stored {len(text_chunks)} text chunks")
                else:
                    logger.warning("Failed to store text chunks in vector database")

            # Store table data in PostgreSQL
            logger.info("Starting PostgreSQL table storage...")
            tables_stored = await self._store_table_data(tables_data, fund_id)
            logger.info(f"✓ Successfully stored {tables_stored} table records")

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
            # Debug logging for row data
            logger.debug(f"Creating {table_type} object from row: {row}")
            
            if table_type == "capital_calls":
                call_date = self.data_parser.parse_date(
                    row.get("date") or row.get("Date") or row.get("Call Date")
                )
                amount = self.data_parser.parse_amount(
                    row.get("amount") or row.get("Amount") or row.get("Called") or row.get("Call Amount")
                )
                call_number = row.get("call number") or row.get("Call Number") or row.get("call_number")
                description = row.get("description") or row.get("Description") or ""

                if call_date and amount is not None:
                    # Use call number as part of description if available
                    if call_number and not description:
                        description = call_number
                    elif call_number and description:
                        description = f"{call_number}: {description}"
                    
                    logger.debug(f"Created CapitalCall: date={call_date}, amount={amount}, desc={description}")
                    return CapitalCall(
                        fund_id=fund_id,
                        call_date=call_date,
                        amount=amount,
                        call_type=row.get("type") or row.get("Type") or "Regular",
                        description=description
                    )

            elif table_type == "distributions":
                distribution_date = self.data_parser.parse_date(
                    row.get("date") or row.get("Date") or row.get("Distribution Date")
                )
                amount = self.data_parser.parse_amount(
                    row.get("amount") or row.get("Amount") or row.get("Distributed") or row.get("Distribution Amount")
                )
                is_recallable = self.data_parser.parse_boolean(
                    row.get("recallable") or row.get("Recallable") or row.get("is_recallable")
                )
                distribution_type = row.get("type") or row.get("Type") or row.get("Distribution Type") or "Return"
                description = row.get("description") or row.get("Description") or ""

                if distribution_date and amount is not None:
                    logger.debug(f"Created Distribution: date={distribution_date}, amount={amount}, type={distribution_type}, recallable={is_recallable}")
                    return Distribution(
                        fund_id=fund_id,
                        distribution_date=distribution_date,
                        amount=amount,
                        distribution_type=distribution_type,
                        is_recallable=is_recallable,
                        description=description
                    )

            elif table_type == "adjustments":
                adjustment_date = self.data_parser.parse_date(
                    row.get("date") or row.get("Date") or row.get("Adjustment Date")
                )
                amount = self.data_parser.parse_amount(
                    row.get("amount") or row.get("Amount") or row.get("Adjustment") or row.get("Adjustment Amount")
                )
                adjustment_type = row.get("type") or row.get("Type") or row.get("Adjustment Type") or "Fee"
                category = row.get("category") or row.get("Category") or "General"
                description = row.get("description") or row.get("Description") or ""

                if adjustment_date and amount is not None:
                    logger.debug(f"Created Adjustment: date={adjustment_date}, amount={amount}, type={adjustment_type}")
                    return Adjustment(
                        fund_id=fund_id,
                        adjustment_date=adjustment_date,
                        amount=amount,
                        adjustment_type=adjustment_type,
                        category=category,
                        is_contribution_adjustment=amount < 0,
                        description=description
                    )

        except Exception as e:
            logger.warning(f"Failed to create object for {table_type}: {e}", exc_info=True)

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
