"""
Document processing service using docling for extracting structured data from PDFs.

This service processes fund performance documents, extracts tables and text,
and prepares content for vector storage and RAG applications.
"""
from typing import Dict, List, Any
import asyncio
import time
from datetime import datetime
import re
from docling.document_converter import DocumentConverter
from app.services.table_parser import TableParser

class DocumentProcessor:
    """Process PDF documents and extract structured data using docling"""

    def __init__(self):
        # Use Docling for full document processing (text + tables)
        # Pre-install OCR models to avoid runtime downloads
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption
        from docling.datamodel.base_models import InputFormat

        # Verify RapidOCR models are pre-loaded (should be done during Docker build)
        self._verify_models_preloaded()

        # Configure Docling with OCR enabled (models are pre-downloaded during Docker build)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # OCR enabled - models pre-loaded in Docker image
        pipeline_options.do_table_structure = True  # Table structure enabled

        print("Initializing DocumentConverter (models should be pre-loaded)...")
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        print("DocumentConverter initialized - using pre-loaded models")
        self.table_parser = TableParser()

        # Performance optimizations
        self.processing_semaphore = asyncio.Semaphore(1)  # Single processing to avoid resource conflicts
        self.chunk_size = 500  # Smaller chunks for speed
        self.chunk_overlap = 20  # Minimal overlap

    def _verify_models_preloaded(self):
        """Verify that RapidOCR models are pre-loaded to avoid runtime downloads"""
        import os
        rapidocr_model_dir = "/usr/local/lib/python3.11/site-packages/rapidocr/models"
        required_models = [
            "ch_PP-OCRv4_det_infer.pth",
            "ch_ptocr_mobile_v2.0_cls_infer.pth",
            "ch_PP-OCRv4_rec_infer.pth"
        ]
        
        if os.path.exists(rapidocr_model_dir):
            found_models = []
            for model_file in required_models:
                model_path = os.path.join(rapidocr_model_dir, model_file)
                if os.path.exists(model_path):
                    found_models.append(model_file)
            
            if len(found_models) == len(required_models):
                print(f"✓ All RapidOCR models pre-loaded ({len(found_models)}/3)")
            elif found_models:
                print(f"⚠ Partial RapidOCR models pre-loaded ({len(found_models)}/3) - some may download on first use")
            else:
                print("⚠ RapidOCR models not found - will download on first document processing (slow!)")
                print("  To fix: Rebuild Docker image with 'make dev-build' or 'make prod-build'")
        else:
            print("⚠ RapidOCR model directory not found - models will download on first use (slow!)")
            print("  To fix: Rebuild Docker image with 'make dev-build' or 'make prod-build'")

    async def process_document(self, file_path: str, document_id: int, fund_id: int) -> Dict[str, Any]:
        """
        Process a PDF document and extract structured data

        Args:
            file_path: Path to the PDF file
            document_id: Database document ID
            fund_id: Fund ID

        Returns:
            Processing result with statistics
        """
        async with self.processing_semaphore:
            start_time = time.time()

            try:
                # Use Docling for full document processing (text + tables)
                # Single conversion - document will be reused for table extraction
                result = self.converter.convert(file_path)
                doc = result.document

                # Extract text content from the document
                text_content = self._extract_text_content(doc)

                # Chunk the text content
                text_chunks = self._chunk_text(text_content)

                # Store text chunks in vector database
                await self._store_chunks(text_chunks, document_id, fund_id)

                # Extract and classify tables using TableParser (reuse converted document)
                try:
                    # Pass the already-converted document to avoid duplicate conversion
                    tables_data = self.table_parser.parse_tables(file_path=file_path, doc=doc)
                    processing_method = tables_data.get("processing_method", "unknown")
                    print(f"Tables extracted using {processing_method}")
                    print(f"Total tables found: {sum(len(tables) for key, tables in tables_data.items() if key != 'processing_method')}")
                except Exception as table_error:
                    print(f"Table parsing failed: {table_error}")
                    import traceback
                    traceback.print_exc()
                    tables_data = {"capital_calls": [], "distributions": [], "adjustments": [], "processing_method": "error"}

                # Store tables
                await self._store_tables(tables_data, fund_id)

                total_time = time.time() - start_time

                # Get page count from docling document
                total_pages = len(doc.pages) if hasattr(doc, 'pages') else 0

                # Return processing statistics
                return {
                    "status": "success",
                    "document_id": document_id,
                    "fund_id": fund_id,
                    "tables_extracted": {
                        "capital_calls": len(tables_data["capital_calls"]),
                        "distributions": len(tables_data["distributions"]),
                        "adjustments": len(tables_data["adjustments"])
                    },
                    "text_chunks_created": len(text_chunks),
                    "total_pages": total_pages,
                    "processing_time": total_time,
                    "processing_method": processing_method,
                    "note": "Document processed with full Docling pipeline (text + tables)"
                }

            except Exception as e:
                error_msg = str(e)
                print(f"Error processing document {file_path}: {error_msg}")
                import traceback
                traceback.print_exc()

                # This should not happen anymore since we skip docling conversion
                print(f"Unexpected error in document processing: {error_msg}")

                return {
                    "status": "error",
                    "document_id": document_id,
                    "fund_id": fund_id,
                    "error": error_msg,
                    "tables_extracted": {"capital_calls": 0, "distributions": 0, "adjustments": 0},
                    "text_chunks_created": 0,
                    "total_pages": 0,
                    "processing_time": time.time() - start_time
                }

    def _extract_text_content(self, doc) -> List[Dict[str, Any]]:
        """
        Extract text content from docling document using fastest method

        Args:
            doc: Docling document object

        Returns:
            List of text content with metadata
        """
        text_content = []

        try:
            # Use simple text extraction for maximum speed
            if hasattr(doc, 'text') and doc.text:
                text_content.append({
                    "page": 1,
                    "content": doc.text,
                    "type": "document_text"
                })
            else:
                # Fallback to markdown if direct text not available
                markdown_content = doc.export_to_markdown()
                if markdown_content.strip():
                    text_content.append({
                        "page": 1,
                        "content": markdown_content,
                        "type": "markdown_text"
                    })

        except Exception as e:
            print(f"Error extracting text content: {e}")
            # Minimal fallback
            text_content.append({
                "page": 1,
                "content": "Document processed but text extraction failed",
                "type": "error_text"
            })

        return text_content


    def _chunk_text(self, text_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simple text chunking for speed

        Args:
            text_content: List of text content with metadata

        Returns:
            List of text chunks with metadata
        """
        chunks = []

        for item in text_content:
            content = item["content"]
            # Create single chunk per content item for maximum speed
            if len(content) > self.chunk_size:
                # Split large content into chunks
                for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
                    chunk_content = content[i:i + self.chunk_size]
                    chunks.append({
                        "content": chunk_content,
                        "metadata": {
                            "page": item["page"],
                            "type": item["type"],
                            "chunk_index": len(chunks)
                        }
                    })
            else:
                chunks.append({
                    "content": content,
                    "metadata": {
                        "page": item["page"],
                        "type": item["type"],
                        "chunk_index": 0
                    }
                })

        return chunks

    async def _store_chunks(self, chunks: List[Dict[str, Any]], document_id: int, fund_id: int):
        """
        Store text chunks in vector database

        Args:
            chunks: List of text chunks with metadata
            document_id: Document ID
            fund_id: Fund ID
        """
        if not chunks:
            return

        try:
            # Initialize vector store if needed
            if not hasattr(self, 'vector_store') or self.vector_store is None:
                from app.services.vector_store import VectorStore
                self.vector_store = VectorStore()

            # Add document and fund metadata to chunks
            for chunk in chunks:
                chunk["metadata"].update({
                    "document_id": document_id,
                    "fund_id": fund_id
                })

            # Store in vector database asynchronously
            print(f"Storing {len(chunks)} text chunks in vector database...")
            success = await self.vector_store.store_chunks(chunks)
            if success:
                print("Text chunks stored successfully in vector database")
            else:
                print("Failed to store text chunks in vector database")
        except Exception as e:
            print(f"Error storing chunks in vector database: {e}")
            # Don't fail the entire process if vector storage fails

    async def _store_tables(self, tables_data: Dict[str, List[Dict[str, Any]]], fund_id: int):
        """
        Store extracted tables in database

        Args:
            tables_data: Classified tables data
            fund_id: Fund ID
        """
        from app.db.session import SessionLocal
        from app.models.transaction import CapitalCall, Distribution, Adjustment

        db = SessionLocal()
        try:
            # Collect all objects for bulk insert (much faster)
            capital_calls = []
            distributions = []
            adjustments = []
            
            # Process all table types and collect objects
            for table_type, tables in tables_data.items():
                if table_type == "processing_method":
                    continue  # Skip processing method metadata
                for table in tables:
                    rows = table.get("rows", [])
                    for row in rows:
                        try:
                            if table_type == "capital_calls":
                                call_date = self._parse_date(row.get("date") or row.get("Date"))
                                amount = self._parse_amount(row.get("amount") or row.get("Amount") or row.get("Called"))
                                if call_date and amount:
                                    capital_calls.append(CapitalCall(
                                        fund_id=fund_id,
                                        call_date=call_date,
                                        amount=amount,
                                        call_type=row.get("type") or row.get("Type") or "Regular",
                                        description=row.get("description") or row.get("Description") or ""
                                    ))

                            elif table_type == "distributions":
                                distribution_date = self._parse_date(row.get("date") or row.get("Date"))
                                amount = self._parse_amount(row.get("amount") or row.get("Amount") or row.get("Distributed"))
                                is_recallable = self._parse_boolean(row.get("recallable") or row.get("Recallable"))
                                if distribution_date and amount:
                                    distributions.append(Distribution(
                                        fund_id=fund_id,
                                        distribution_date=distribution_date,
                                        amount=amount,
                                        distribution_type=row.get("type") or row.get("Type") or "Return",
                                        is_recallable=is_recallable,
                                        description=row.get("description") or row.get("Description") or ""
                                    ))

                            elif table_type == "adjustments":
                                adjustment_date = self._parse_date(row.get("date") or row.get("Date"))
                                amount = self._parse_amount(row.get("amount") or row.get("Amount") or row.get("Adjustment"))
                                if adjustment_date and amount:
                                    adjustments.append(Adjustment(
                                        fund_id=fund_id,
                                        adjustment_date=adjustment_date,
                                        amount=amount,
                                        adjustment_type=row.get("type") or row.get("Type") or "Fee",
                                        category=row.get("category") or row.get("Category") or "General",
                                        is_contribution_adjustment=amount < 0,
                                        description=row.get("description") or row.get("Description") or ""
                                    ))

                        except Exception as e:
                            # Skip invalid rows silently
                            continue

            # Bulk insert all objects at once (much faster than individual adds)
            all_objects = capital_calls + distributions + adjustments
            if all_objects:
                db.add_all(all_objects)
            
            stored_count = len(all_objects)
            db.commit()
        except Exception as e:
            print(f"Error storing tables: {e}")
            import traceback
            traceback.print_exc()
            db.rollback()
        finally:
            db.close()

    def _parse_date(self, date_str: str) -> datetime.date:
        """Parse date string into date object"""
        if not date_str:
            return None

        # Common date formats
        formats = [
            "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y",
            "%b %d, %Y", "%Y/%m/%d", "%d-%b-%Y", "%d-%B-%Y"
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue

        # Try to extract date using regex
        date_patterns = [
            r"(\d{4})-(\d{2})-(\d{2})",
            r"(\d{2})/(\d{2})/(\d{4})",
            r"(\d{2})-(\d{2})-(\d{4})"
        ]

        for pattern in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    if len(match.groups()) == 3:
                        y, m, d = match.groups()
                        if len(y) == 4:  # YYYY-MM-DD
                            return datetime(int(y), int(m), int(d)).date()
                        else:  # DD/MM/YYYY or MM/DD/YYYY
                            # Assume MM/DD/YYYY for US format
                            return datetime(int(d), int(m), int(y)).date()
                except ValueError:
                    continue

        return None

    def _parse_amount(self, amount_str: str) -> float:
        """Parse amount string into float"""
        if not amount_str:
            return None

        # Remove currency symbols and commas
        cleaned = re.sub(r"[$,€£¥₹\s]", "", str(amount_str))

        # Handle parentheses for negative amounts
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]

        try:
            return float(cleaned)
        except ValueError:
            return None

    def _parse_boolean(self, bool_str: str) -> bool:
        """Parse boolean string"""
        if not bool_str:
            return False

        return str(bool_str).lower() in ["yes", "true", "1", "y", "recallable"]
