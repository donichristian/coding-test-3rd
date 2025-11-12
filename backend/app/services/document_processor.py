"""
Document processing service using docling for extracting structured data from PDFs.

This service processes fund performance documents, extracts tables and text,
and prepares content for vector storage and RAG applications.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Protocol
import asyncio
import logging
import os
import re
import time
from datetime import datetime
from contextlib import asynccontextmanager

from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from docling.datamodel.base_models import InputFormat

from app.core.config import settings
from app.services.table_parser import TableParser
from app.services.vector_store import VectorStore
from app.db.session import SessionLocal
from app.models.transaction import CapitalCall, Distribution, Adjustment
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    """Enumeration of processing statuses."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


@dataclass
class ProcessingResult:
    """Data class for processing results with dictionary-like access."""
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
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access for backward compatibility."""
        return getattr(self, key)
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator."""
        return hasattr(self, key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-like get method."""
        return getattr(self, key, default)
    
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
    
class DocumentExtractorProtocol(Protocol):
    """Protocol for document extractors."""
    
    async def extract_text(self, doc) -> List[Dict[str, Any]]:
        """Extract text content from document."""
        ...
    
    async def extract_tables(self, file_path: str, doc) -> Dict[str, List[Dict[str, Any]]]:
        """Extract and classify tables."""
        ...
        
class TextChunkerProtocol(Protocol):
    """Protocol for text chunkers."""
    
    def chunk_text(self, text_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk text content."""
        ...

class DataStorerProtocol(Protocol):
    """Protocol for data storers."""
    
    async def store_chunks(self, chunks: List[Dict[str, Any]], document_id: int, fund_id: int) -> bool:
        """Store text chunks."""
        ...
    
    async def store_tables(self, tables_data: Dict[str, List[Dict[str, Any]]], fund_id: int) -> int:
        """Store table data. Returns count of stored items."""
        ...

class DataParserProtocol(Protocol):
    """Protocol for data parsers."""
    
    def parse_date(self, date_str: str) -> Optional[datetime.date]:
        """Parse date string."""
        ...
    
    def parse_amount(self, amount_str: str) -> Optional[float]:
        """Parse amount string."""
        ...
    
    def parse_boolean(self, bool_str: str) -> bool:
        """Parse boolean string."""
        ...

class DocumentExtractor:
    """Responsible for extracting content from documents."""
    
    def __init__(self, table_parser: TableParser):
        self.table_parser = table_parser
    
    def extract_text(self, doc) -> List[Dict[str, Any]]:
        """Extract text content from docling document."""
        try:
            text_content = []
            
            # Primary method: direct text extraction
            if hasattr(doc, 'text') and doc.text:
                text_content.append({
                    "page": 1,
                    "content": doc.text,
                    "type": "document_text"
                })
            else:
                # Fallback: markdown export
                markdown_content = doc.export_to_markdown()
                if markdown_content.strip():
                    text_content.append({
                        "page": 1,
                        "content": markdown_content,
                        "type": "markdown_text"
                    })
            
            return text_content
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return [{
                "page": 1,
                "content": "Document processed but text extraction failed",
                "type": "error_text"
            }]
    
    def extract_tables(self, file_path: str, doc) -> Dict[str, List[Dict[str, Any]]]:
        """Extract and classify tables."""
        try:
            return self.table_parser.parse_tables(file_path=file_path, doc=doc)
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return {
                "capital_calls": [],
                "distributions": [],
                "adjustments": [],
                "processing_method": "error"
            }

class TextChunker:
    """Responsible for chunking text content with semantic awareness."""

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.chunk_size = chunk_size or int(os.getenv('CHUNK_SIZE', settings.CHUNK_SIZE))
        self.chunk_overlap = chunk_overlap or int(os.getenv('CHUNK_OVERLAP', settings.CHUNK_OVERLAP))

    def chunk_text(self, text_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk text content into manageable pieces with semantic awareness."""
        chunks = []

        for item in text_content:
            content = item["content"]

            if len(content) <= self.chunk_size:
                # Single chunk - no need to split
                chunks.append({
                    "content": content,
                    "metadata": {
                        "page": item["page"],
                        "type": item["type"],
                        "chunk_index": 0,
                        "is_complete": True
                    }
                })
            else:
                # Multiple chunks with semantic splitting
                semantic_chunks = self._split_semantic_chunks(content, self.chunk_size, self.chunk_overlap)

                for i, chunk_content in enumerate(semantic_chunks):
                    chunks.append({
                        "content": chunk_content,
                        "metadata": {
                            "page": item["page"],
                            "type": item["type"],
                            "chunk_index": len(chunks),
                            "is_complete": self._is_complete_chunk(chunk_content)
                        }
                    })

        return chunks

    def _split_semantic_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into semantic chunks, preferring to break at natural boundaries."""
        if len(text) <= chunk_size:
            return [text]

        # First, identify table sections and treat them as atomic units
        table_sections = self._identify_table_sections(text)

        if not table_sections:
            # No tables detected, use regular semantic chunking
            return self._chunk_regular_text(text, chunk_size, overlap)

        # Handle text with tables
        return self._chunk_text_with_tables(text, table_sections, chunk_size, overlap)

    def _identify_table_sections(self, text: str) -> List[tuple[int, int]]:
        """Identify table sections in the text and return their start/end positions."""
        sections = []
        lines = text.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for table patterns (multiple pipes, or structured data)
            if self._is_table_line(line):
                table_start = text.find(line, text.find('\n', text.find(lines[i-1]) if i > 0 else 0) + 1 if i > 0 else 0)

                # Find the end of the table
                table_end = table_start + len(line)
                j = i + 1

                while j < len(lines):
                    next_line = lines[j].strip()
                    if self._is_table_line(next_line) or (next_line and not next_line[0].isupper() and '|' in next_line):
                        table_end = text.find(next_line, table_end) + len(next_line)
                        j += 1
                    else:
                        break

                if table_end > table_start:
                    sections.append((table_start, table_end))
                    i = j
                else:
                    i += 1
            else:
                i += 1

        return sections

    def _is_table_line(self, line: str) -> bool:
        """Check if a line appears to be part of a table."""
        if not line:
            return False

        # Count pipes
        pipe_count = line.count('|')

        # Check for structured data patterns
        if pipe_count >= 2:
            return True

        # Check for tab-separated values
        if '\t' in line and len(line.split('\t')) >= 3:
            return True

        # Check for aligned columns (multiple spaces separating data)
        parts = line.split()
        if len(parts) >= 4 and any(len(part) > 10 for part in parts):
            return True

        return False

    def _chunk_text_with_tables(self, text: str, table_sections: List[tuple[int, int]], chunk_size: int, overlap: int) -> List[str]:
        """Chunk text that contains tables, treating tables as atomic units."""
        chunks = []
        start = 0

        for table_start, table_end in table_sections:
            # Handle text before the table
            if start < table_start:
                pre_table_text = text[start:table_start].strip()
                if pre_table_text:
                    # Chunk the pre-table text
                    pre_chunks = self._chunk_regular_text(pre_table_text, chunk_size, overlap)
                    chunks.extend(pre_chunks)

            # Add the entire table as one chunk (if it fits)
            table_text = text[table_start:table_end].strip()
            if len(table_text) <= chunk_size:
                chunks.append(table_text)
            else:
                # If table is too large, split it at row boundaries
                table_chunks = self._chunk_table(table_text, chunk_size, overlap)
                chunks.extend(table_chunks)

            start = table_end

        # Handle remaining text after the last table
        if start < len(text):
            remaining_text = text[start:].strip()
            if remaining_text:
                remaining_chunks = self._chunk_regular_text(remaining_text, chunk_size, overlap)
                chunks.extend(remaining_chunks)

        return chunks

    def _chunk_table(self, table_text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split a large table into chunks at row boundaries."""
        lines = table_text.split('\n')
        chunks = []

        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n'.join(current_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text)

                # Start new chunk with overlap (keep last few lines)
                overlap_lines = min(len(current_chunk), 2)  # Keep up to 2 lines for overlap
                current_chunk = current_chunk[-overlap_lines:] if overlap_lines > 0 else []
                current_size = sum(len(line) + 1 for line in current_chunk)

            current_chunk.append(line)
            current_size += line_size

        # Add the last chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text)

        return chunks

    def _chunk_regular_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Chunk regular (non-table) text using semantic boundaries."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            if end >= len(text):
                chunk_text = text[start:].strip()
                if chunk_text:
                    chunks.append(chunk_text)
                break

            # Try to find a good breaking point within the last 30% of the chunk
            break_search_start = max(start + int(chunk_size * 0.7), start)
            break_pos = self._find_semantic_break(text, break_search_start, end)

            if break_pos > start:
                chunk_text = text[start:break_pos].strip()
                if chunk_text:
                    chunks.append(chunk_text)
                start = break_pos
            else:
                # No good break point found, force break at a reasonable position
                fallback_end = self._find_fallback_break(text, start, end)
                chunk_text = text[start:fallback_end].strip()
                if chunk_text:
                    chunks.append(chunk_text)
                start = max(fallback_end - overlap, start + 50)

    def _find_fallback_break(self, text: str, start: int, end: int) -> int:
        """Find a fallback breaking point when no semantic break is found."""
        # Look for whitespace characters in reverse order
        for i in range(end - 1, start - 1, -1):
            if text[i].isspace():
                return i + 1  # Include the whitespace

        # If no whitespace found, break at a word boundary if possible
        for i in range(end - 1, max(start, end - 20), -1):
            if not text[i].isalnum() and text[i] not in ['_', '-']:
                return i

        # Last resort: break exactly at chunk_size
        return end

    def _find_semantic_break(self, text: str, search_start: int, search_end: int) -> int:
        """Find the best semantic breaking point in the text."""
        # Define semantic break patterns in order of preference
        break_patterns = [
            r'\n\s*\n',  # Double newlines (paragraph breaks)
            r'(?<=\.)\s+(?=[A-Z])',  # Period followed by capital letter (sentence end)
            r'(?<=\!|\?)\s+',  # Exclamation/question marks
            r'\n',  # Single newlines
            r'(?<=\;|\:)\s+',  # Semicolon/colon
            r'\s+',  # Whitespace (last resort)
        ]

        # Search backwards from search_end to find the best break
        for pattern in break_patterns:
            import re
            matches = list(re.finditer(pattern, text[search_start:search_end]))
            if matches:
                # Take the last (rightmost) match
                match = matches[-1]
                return search_start + match.end()

        return -1  # No suitable break found

    def _is_complete_chunk(self, chunk_text: str) -> bool:
        """Determine if a chunk appears to be complete (not cut off mid-sentence/table)."""
        if not chunk_text or len(chunk_text.strip()) == 0:
            return False

        text = chunk_text.strip()

        # Check for incomplete sentences (ends with incomplete words or punctuation)
        if text.endswith(('...', '…', 'etc.', 'i.e.', 'e.g.', 'vs.', 'cf.', 'Dr.', 'Mr.', 'Mrs.', 'Ms.')):
            return False

        # Check for table-like content that might be cut off
        lines = text.split('\n')
        if len(lines) > 1:
            # Check if it looks like a table (multiple | separators)
            pipe_count = text.count('|')
            if pipe_count >= 4:  # Likely a table
                # For tables, check if we have complete rows
                table_lines = [line for line in lines if line.strip() and '|' in line]
                if len(table_lines) > 1:  # Has header + at least one data row
                    # Check if the table looks complete (has balanced structure)
                    first_row_pipes = table_lines[0].count('|')
                    last_row_pipes = table_lines[-1].count('|')
                    if abs(first_row_pipes - last_row_pipes) > 1:  # Significant imbalance
                        return False
                else:
                    # Single table line, might be incomplete
                    return False

        # Check for incomplete parentheses/brackets
        if text.count('(') > text.count(')') or text.count('[') > text.count(']'):
            return False

        # Check for trailing punctuation that suggests incompleteness
        if text.endswith(('.', '!', '?', ':', ';', ',')):
            return True
        elif text.endswith(('-', '–', '—')):
            return False

        # Additional check: if text ends with a word that's cut off
        words = text.split()
        if words and len(words[-1]) < 3 and not any(words[-1].endswith(punct) for punct in '.!?,;:'):
            return False

        return True

    def post_process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process chunks to improve readability and completeness."""
        processed_chunks = []

        for chunk in chunks:
            content = chunk["content"]
            metadata = chunk["metadata"]

            # Clean up table formatting
            if '|' in content and '\n' in content:
                content = self._clean_table_formatting(content)

            # Add continuation markers for incomplete chunks
            if not metadata.get("is_complete", True):
                content = self._add_continuation_marker(content)

            # Update the chunk
            processed_chunk = chunk.copy()
            processed_chunk["content"] = content
            processed_chunks.append(processed_chunk)

        return processed_chunks

    def _clean_table_formatting(self, content: str) -> str:
        """Clean up table formatting for better readability."""
        lines = content.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if line:
                # Ensure consistent spacing around pipes
                line = re.sub(r'\s*\|\s*', ' | ', line)
                line = re.sub(r'^\s*\|\s*', '', line)  # Remove leading pipe
                line = re.sub(r'\s*\|\s*$', '', line)  # Remove trailing pipe
                # Clean up multiple spaces
                line = re.sub(r'\s+', ' ', line)
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _add_continuation_marker(self, content: str) -> str:
        """Add a continuation marker to incomplete chunks."""
        if content.endswith(('...', '…')):
            return content  # Already has continuation marker
        elif content.endswith(('.', '!', '?', ':', ';')):
            return content + " [continued...]"
        else:
            return content + "... [continued]"
    
class DataStorer:
    """Responsible for storing data in databases."""
    
    def __init__(self, vector_store: VectorStore, data_parser: DataParserProtocol):
        self.vector_store = vector_store
        self.data_parser = data_parser
    
    async def store_chunks(self, chunks: List[Dict[str, Any]], document_id: int, fund_id: int) -> bool:
        """Store text chunks in vector database."""
        if not chunks:
            return True
        
        try:
            # Add metadata
            for chunk in chunks:
                chunk["metadata"].update({
                    "document_id": document_id,
                    "fund_id": fund_id
                })
            
            logger.info(f"Storing {len(chunks)} text chunks")
            return await self.vector_store.store_chunks(chunks)
            
        except Exception as e:
            logger.error(f"Chunk storage failed: {e}")
            return False
    
    async def store_tables(self, tables_data: Dict[str, List[Dict[str, Any]]], fund_id: int) -> int:
        """Store table data with duplicate prevention."""
        stored_count = 0

        async with self._get_db_session() as db:
            try:
                # Clear existing data for this fund
                await self._clear_existing_data(db, fund_id)

                # Collect objects for bulk insert
                objects_to_insert = []

                for table_type, tables in tables_data.items():
                    if table_type == "processing_method":
                        continue

                    for table in tables:
                        rows = table.get("rows", [])
                        for row in rows:
                            obj = self._create_transaction_object(table_type, row, fund_id)
                            if obj and await self._is_duplicate(db, obj):
                                objects_to_insert.append(obj)

                # Bulk insert
                if objects_to_insert:
                    db.add_all(objects_to_insert)
                    await db.commit()
                    stored_count = len(objects_to_insert)
                    logger.info(f"Stored {stored_count} transactions")

            except Exception as e:
                await db.rollback()
                logger.error(f"Table storage failed: {e}")
                raise

        return stored_count

    def store_tables_sync(self, tables_data: Dict[str, List[Dict[str, Any]]], fund_id: int) -> int:
        """Synchronous version of store_tables for Celery context."""
        import asyncio

        try:
            # Create new event loop for synchronous context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.store_tables(tables_data, fund_id))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Failed to store tables synchronously: {e}")
            return 0
    
    @asynccontextmanager
    async def _get_db_session(self):
        """Get database session with proper cleanup."""
        db = SessionLocal()
        try:
            yield db
        finally:
            await db.close()
    
    async def _clear_existing_data(self, db, fund_id: int):
        """Clear existing transaction data for fund."""
        deleted_calls = await db.query(CapitalCall).filter(CapitalCall.fund_id == fund_id).delete()
        deleted_distributions = await db.query(Distribution).filter(Distribution.fund_id == fund_id).delete()
        deleted_adjustments = await db.query(Adjustment).filter(Adjustment.fund_id == fund_id).delete()
        
        logger.info(f"Cleared {deleted_calls} calls, {deleted_distributions} distributions, {deleted_adjustments} adjustments")
    
    def _create_transaction_object(self, table_type: str, row: Dict[str, Any], fund_id: int):
        """Create transaction object from row data."""
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
    
    async def _is_duplicate(self, db, obj) -> bool:
        """Check if transaction object already exists."""
        try:
            if isinstance(obj, CapitalCall):
                existing = await db.query(CapitalCall).filter(
                    CapitalCall.fund_id == obj.fund_id,
                    CapitalCall.call_date == obj.call_date,
                    CapitalCall.amount == obj.amount
                ).first()
            elif isinstance(obj, Distribution):
                existing = await db.query(Distribution).filter(
                    Distribution.fund_id == obj.fund_id,
                    Distribution.distribution_date == obj.distribution_date,
                    Distribution.amount == obj.amount
                ).first()
            elif isinstance(obj, Adjustment):
                existing = await db.query(Adjustment).filter(
                    Adjustment.fund_id == obj.fund_id,
                    Adjustment.adjustment_date == obj.adjustment_date,
                    Adjustment.amount == obj.amount
                ).first()
            else:
                return True
            
            return existing is None
        
        except Exception as e:
            logger.error(f"Duplicate check failed: {e}")
            return False

class DataParser:
    """Responsible for parsing various data types."""
    
    DATE_FORMATS = [
        "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y",
        "%b %d, %Y", "%Y/%m/%d", "%d-%b-%Y", "%d-%B-%Y"
    ]
    
    DATE_PATTERNS = [
        r"(\d{4})-(\d{2})-(\d{2})",
        r"(\d{2})/(\d{2})/(\d{4})",
        r"(\d{2})-(\d{2})-(\d{4})"
    ]
    
    def parse_date(self, date_str: str) -> Optional[datetime.date]:
        """Parse date string into date object."""
        if not date_str:
            return None
        
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
    
    def parse_amount(self, amount_str: str) -> Optional[float]:
        """Parse amount string into float."""
        if not amount_str:
            return None
        
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

class DocumentProcessor:
    """Process PDF documents and extract structured data using docling"""

    def __init__(
        self,
        document_extractor: Optional[DocumentExtractorProtocol] = None,
        text_chunker: Optional[TextChunkerProtocol] = None,
        data_storer: Optional[DataStorerProtocol] = None,
        converter: Optional[DocumentConverter] = None,
        db_session: Optional[Session] = None
    ):
        # Dependency injection with defaults
        self.document_extractor = document_extractor or DocumentExtractor(TableParser())
        self.text_chunker = text_chunker or TextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        # Pass database session to avoid None errors in Celery context
        self.data_storer = data_storer or DataStorer(VectorStore(db_session), DataParser())
        self.converter = converter or self._initialize_converter()

        # Performance controls
        self.processing_semaphore = asyncio.Semaphore(1)  # Single processing
    
    def _initialize_converter(self) -> DocumentConverter:
        """Initialize Docling converter with proper configuration."""
        # Try to get cached converter from Celery first
        try:
            from app.core.celery_app import get_cached_model
            cached_converter = get_cached_model('docling_converter')
            if cached_converter is not None:
                logger.info("Using cached Docling converter from Celery")
                return cached_converter
        except ImportError:
            pass  # Not in Celery context

        # Fallback: initialize converter normally
        self._verify_models_preloaded()

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    
    def _verify_models_preloaded(self):
        """Verify RapidOCR models are pre-loaded."""
        rapidocr_dir = "/usr/local/lib/python3.11/site-packages/rapidocr/models"
        required_models = [
            "ch_PP-OCRv4_det_infer.pth",
            "ch_ptocr_mobile_v2.0_cls_infer.pth", 
            "ch_PP-OCRv4_rec_infer.pth"
        ]
        
        if os.path.exists(rapidocr_dir):
            found = sum(1 for model in required_models if os.path.exists(os.path.join(rapidocr_dir, model)))
            if found == len(required_models):
                logger.info(f"✓ All {found} RapidOCR models pre-loaded")
            elif found > 0:
                logger.warning(f"⚠ Partial RapidOCR models pre-loaded ({found}/{len(required_models)})")
            else:
                logger.warning("⚠ RapidOCR models not found - will download on first use")
        else:
            logger.warning("⚠ RapidOCR model directory not found")
    
    async def process_document(self, file_path: str, document_id: int, fund_id: int) -> ProcessingResult:
        """
        Process a PDF document and extract structured data (async version).

        Args:
            file_path: Path to PDF file
            document_id: Document ID
            fund_id: Fund ID

        Returns:
            Processing result
        """
        # For async contexts (like API endpoints), delegate to sync version
        return await asyncio.to_thread(self.process_document_sync, file_path, document_id, fund_id)

    def process_document_sync(self, file_path: str, document_id: int, fund_id: int) -> ProcessingResult:
        """
        Process a PDF document and extract structured data (sync version for Celery).

        Args:
            file_path: Path to PDF file
            document_id: Document ID
            fund_id: Fund ID

        Returns:
            Processing result
        """
        start_time = time.time()

        try:
            # Convert document (CPU-bound, run synchronously)
            result = self.converter.convert(file_path)
            doc = result.document

            # Extract content synchronously
            text_content = self.document_extractor.extract_text(doc)
            tables_data = self.document_extractor.extract_tables(file_path, doc)

            # Process text synchronously
            text_chunks = self.text_chunker.chunk_text(text_content)
            text_chunks = self.text_chunker.post_process_chunks(text_chunks)

            # Store data synchronously using synchronous database operations
            chunks_stored = self._store_chunks_sync(text_chunks, document_id, fund_id)
            tables_stored = self._store_tables_sync(tables_data, fund_id)

            # Calculate statistics
            processing_time = time.time() - start_time
            total_pages = len(doc.pages) if hasattr(doc, 'pages') else 0
            processing_method = tables_data.get("processing_method", "docling")

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
                processing_method=processing_method,
                note="Document processed successfully"
            )

        except Exception as e:
            logger.error(f"Document processing failed for {file_path}: {e}")
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

    def _store_chunks_sync(self, chunks: List[Dict[str, Any]], document_id: int, fund_id: int) -> bool:
        """Store chunks using synchronous database access."""
        try:
            # Add fund_id to each chunk's metadata before storing
            for chunk in chunks:
                if "metadata" not in chunk:
                    chunk["metadata"] = {}
                chunk["metadata"]["document_id"] = document_id
                chunk["metadata"]["fund_id"] = fund_id

            # Use synchronous version of vector store operations
            # Create a synchronous vector store instance
            from app.services.vector_store import VectorStore
            sync_vector_store = VectorStore(db=None)  # No DB session needed for sync operations

            # Generate embeddings synchronously
            for chunk in chunks:
                try:
                    # Get embedding synchronously
                    text = chunk["content"]
                    embedding = sync_vector_store._get_embedding_sync(text)
                    chunk["embedding"] = embedding
                except Exception as e:
                    logger.error(f"Failed to generate embedding for chunk: {e}")
                    chunk["embedding"] = None

            # Store synchronously using direct SQL
            return self._store_chunks_direct_sql(chunks, document_id, fund_id)

        except Exception as e:
            logger.error(f"Failed to store chunks synchronously: {e}")
            return False

    def _store_chunks_direct_sql(self, chunks: List[Dict[str, Any]], document_id: int, fund_id: int) -> bool:
        """Store chunks directly using SQLAlchemy synchronously."""
        try:
            from app.db.session import SessionLocal
            import json

            db = SessionLocal()
            try:
                for chunk in chunks:
                    if chunk.get("embedding") is None:
                        continue  # Skip chunks without embeddings

                    embedding_list = chunk["embedding"].tolist()
                    metadata_json = json.dumps(chunk.get("metadata", {}))

                    # Insert directly using SQLAlchemy
                    from sqlalchemy import text
                    insert_sql = text("""
                        INSERT INTO document_embeddings (document_id, fund_id, content, embedding, metadata)
                        VALUES (:document_id, :fund_id, :content, :embedding, :metadata)
                    """)

                    db.execute(insert_sql, {
                        "document_id": document_id,
                        "fund_id": fund_id,
                        "content": chunk["content"],
                        "embedding": f"[{','.join(map(str, embedding_list))}]",
                        "metadata": metadata_json
                    })

                db.commit()
                logger.info(f"Successfully stored {len([c for c in chunks if c.get('embedding') is not None])} chunks")
                return True

            except Exception as e:
                db.rollback()
                logger.error(f"Database error storing chunks: {e}")
                raise
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Failed to store chunks with direct SQL: {e}")
            return False

    def _store_tables_sync(self, tables_data: Dict[str, List[Dict[str, Any]]], fund_id: int) -> int:
        """Store tables using synchronous database access."""
        try:
            # Create synchronous version of table storage
            return self._store_tables_direct_sql(tables_data, fund_id)
        except Exception as e:
            logger.error(f"Failed to store tables synchronously: {e}")
            return 0

    def _store_tables_direct_sql(self, tables_data: Dict[str, List[Dict[str, Any]]], fund_id: int) -> int:
        """Store table data directly using SQLAlchemy synchronously."""
        from app.db.session import SessionLocal
        from app.models.transaction import CapitalCall, Distribution, Adjustment
        import json

        stored_count = 0
        data_parser = DataParser()

        db = SessionLocal()
        try:
            # Clear existing data for this fund
            db.query(CapitalCall).filter(CapitalCall.fund_id == fund_id).delete()
            db.query(Distribution).filter(Distribution.fund_id == fund_id).delete()
            db.query(Adjustment).filter(Adjustment.fund_id == fund_id).delete()
            db.commit()

            # Collect objects for bulk insert
            objects_to_insert = []

            for table_type, tables in tables_data.items():
                if table_type == "processing_method":
                    continue

                for table in tables:
                    rows = table.get("rows", [])
                    for row in rows:
                        obj = self._create_transaction_object_sync(table_type, row, fund_id, data_parser)
                        if obj:
                            objects_to_insert.append(obj)

            # Bulk insert
            if objects_to_insert:
                db.add_all(objects_to_insert)
                db.commit()
                stored_count = len(objects_to_insert)
                logger.info(f"Stored {stored_count} transactions synchronously")

            return stored_count

        except Exception as e:
            db.rollback()
            logger.error(f"Database error storing tables: {e}")
            raise
        finally:
            db.close()

    def _create_transaction_object_sync(self, table_type: str, row: Dict[str, Any], fund_id: int, data_parser: 'DataParser'):
        """Create transaction object synchronously."""
        try:
            if table_type == "capital_calls":
                call_date = data_parser.parse_date(row.get("date") or row.get("Date"))
                amount = data_parser.parse_amount(row.get("amount") or row.get("Amount") or row.get("Called"))

                if call_date and amount is not None:
                    return CapitalCall(
                        fund_id=fund_id,
                        call_date=call_date,
                        amount=amount,
                        call_type=row.get("type") or row.get("Type") or "Regular",
                        description=row.get("description") or row.get("Description") or ""
                    )

            elif table_type == "distributions":
                distribution_date = data_parser.parse_date(row.get("date") or row.get("Date"))
                amount = data_parser.parse_amount(row.get("amount") or row.get("Amount") or row.get("Distributed"))
                is_recallable = data_parser.parse_boolean(row.get("recallable") or row.get("Recallable"))

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
                adjustment_date = data_parser.parse_date(row.get("date") or row.get("Date"))
                amount = data_parser.parse_amount(row.get("amount") or row.get("Amount") or row.get("Adjustment"))

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
