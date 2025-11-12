"""
Table parsing service using docling (with pdfplumber fallback) for extracting structured data from PDF tables.

This service classifies and parses tables from fund performance documents,
specifically handling capital calls, distributions, and adjustments.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Protocol
import logging
import re

import pdfplumber

logger = logging.getLogger(__name__)

class TableType(Enum):
    """Enumeration of supported table types."""
    CAPITAL_CALLS = "capital_calls"
    DISTRIBUTIONS = "distributions"
    ADJUSTMENTS = "adjustments"


class ProcessingMethod(Enum):
    """Enumeration of processing methods."""
    DOCLING = "docling"
    PDFPLUMBER = "pdfplumber"
    ERROR = "error"

@dataclass
class TableData:
    """Data class for extracted table data."""
    headers: List[str]
    rows: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class ClassificationResult:
    """Data class for table classification results."""
    table_type: Optional[TableType]
    confidence: float
    reason: str


class TableExtractorProtocol(Protocol):
    """Protocol for table extractors."""
    
    def extract_tables(self, file_path: str, doc: Optional[Any] = None) -> List[TableData]:
        """Extract tables from document."""
        ...


class TableClassifierProtocol(Protocol):
    """Protocol for table classifiers."""
    
    def classify(self, table_data: TableData) -> ClassificationResult:
        """Classify table type."""
        ...

class DoclingTableExtractor:
    """Table extractor using Docling library."""
    
    def __init__(self):
        self.converter = self._initialize_converter()
    
    def _initialize_converter(self):
        """Initialize Docling converter for table extraction."""
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False  # Not needed for table extraction
            pipeline_options.do_table_structure = True
            
            return DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        except ImportError:
            logger.warning("Docling not available")
            return None
    
    def extract_tables(self, file_path: str, doc: Optional[Any] = None) -> List[TableData]:
        """Extract tables using Docling."""
        if not self.converter:
            return []
        
        try:
            # Use provided document or convert file
            if doc is None:
                result = self.converter.convert(file_path)
                doc = result.document
            
            tables = []
            
            # Extract tables from document
            if hasattr(doc, 'tables') and doc.tables:
                for table in doc.tables:
                    table_data = self._extract_docling_table(table, doc)
                    if table_data:
                        tables.append(table_data)
            
            # Check pages for tables
            elif hasattr(doc, 'pages'):
                for page in doc.pages:
                    if hasattr(page, 'tables') and page.tables:
                        for table in page.tables:
                            table_data = self._extract_docling_table(table, doc)
                            if table_data:
                                tables.append(table_data)
            
            logger.info(f"Docling extracted {len(tables)} tables")
            return tables
            
        except Exception as e:
            logger.error(f"Docling table extraction failed: {e}")
            return []
    
    def _extract_docling_table(self, table, doc: Any) -> Optional[TableData]:
        """Extract data from Docling table object with enhanced native capabilities."""
        try:
            # Method 1: Try DataFrame export first (most reliable)
            if hasattr(table, 'export_to_dataframe'):
                df = table.export_to_dataframe(doc=doc)
                if not df.empty:
                    # Enhanced metadata from Docling's native analysis
                    enhanced_metadata = self._extract_enhanced_table_metadata(table, doc)
                    
                    return TableData(
                        headers=list(df.columns),
                        rows=df.to_dict('records'),
                        metadata={
                            **enhanced_metadata,
                            "source": "docling_dataframe",
                            "extraction_method": "pandas_dataframe"
                        }
                    )
            
            # Method 2: Enhanced direct data access with structure analysis
            if hasattr(table, 'data') and table.data:
                table_data = self._extract_table_with_structure(table, doc)
                if table_data:
                    return table_data
            
            # Method 3: Try docling's table properties
            table_properties = self._extract_table_properties(table, doc)
            if table_properties:
                return table_properties
        
        except Exception as e:
            logger.debug(f"Failed to extract Docling table with enhanced method: {e}")
        
        return None

    def _extract_enhanced_table_metadata(self, table, doc: Any) -> Dict[str, Any]:
        """Extract enhanced metadata using Docling's native table analysis."""
        metadata = {
            "row_count": 0,
            "column_count": 0,
            "table_type": getattr(table, 'table_type', 'unknown'),
            "confidence": getattr(table, 'confidence', 1.0),
            "structure": getattr(table, 'structure', {}),
            "page_number": getattr(table, 'page_number', 1),
            "bbox": getattr(table, 'bbox', {}),
            "has_header": getattr(table, 'has_header', True),
            "header_rows": getattr(table, 'header_rows', 1),
        }
        
        try:
            # Extract bounding box information
            if hasattr(table, 'bbox'):
                bbox = table.bbox
                if hasattr(bbox, 'to_dict'):
                    metadata["bbox"] = bbox.to_dict()
                else:
                    metadata["bbox"] = str(bbox)
            
            # Extract table structure information
            if hasattr(table, 'structure'):
                structure = table.structure
                if isinstance(structure, dict):
                    metadata["structure"] = structure
                else:
                    metadata["structure"] = str(structure)
            
            # Get page information
            if hasattr(table, 'page'):
                page = table.page
                metadata["page_number"] = getattr(page, 'page_number', 1)
                metadata["page_bbox"] = str(getattr(page, 'bbox', {}))
            
            # Extract cell merge information
            if hasattr(table, 'merged_cells'):
                metadata["merged_cells"] = len(table.merged_cells)
            
            # Extract table statistics
            if hasattr(table, 'statistics'):
                stats = table.statistics
                metadata["table_statistics"] = {
                    "avg_row_height": getattr(stats, 'avg_row_height', 0),
                    "avg_col_width": getattr(stats, 'avg_col_width', 0),
                    "complexity_score": getattr(stats, 'complexity_score', 0)
                }
                
        except Exception as e:
            logger.debug(f"Error extracting enhanced table metadata: {e}")
        
        return metadata

    def _extract_table_with_structure(self, table, doc: Any) -> Optional[TableData]:
        """Extract table with detailed structure analysis."""
        try:
            # Access table data with enhanced error handling
            table_data = getattr(table, 'data', [])
            if not table_data:
                return None
            
            # Extract headers with better cell handling
            headers = []
            if len(table_data) > 0:
                first_row = table_data[0]
                headers = []
                for i, cell in enumerate(first_row):
                    if hasattr(cell, 'text'):
                        cell_text = cell.text.strip()
                    elif hasattr(cell, 'value'):
                        cell_text = str(cell.value).strip()
                    else:
                        cell_text = str(cell).strip()
                    
                    if cell_text:
                        headers.append(cell_text)
                    else:
                        headers.append(f"Column_{i+1}")  # Fallback header
            
            # Extract rows with cell-level analysis
            rows = []
            for row_idx, row in enumerate(table_data[1:], 1):
                row_data = {}
                for col_idx, cell in enumerate(row):
                    if col_idx < len(headers):
                        header = headers[col_idx]
                        
                        # Enhanced cell value extraction
                        if hasattr(cell, 'text'):
                            cell_value = cell.text.strip()
                        elif hasattr(cell, 'value'):
                            cell_value = str(cell.value).strip()
                        else:
                            cell_value = str(cell).strip()
                        
                        if cell_value:
                            row_data[header] = cell_value
                        else:
                            row_data[header] = ""
                
                if row_data:
                    rows.append(row_data)
            
            if not headers or not rows:
                return None
            
            # Create enhanced metadata
            enhanced_metadata = self._extract_enhanced_table_metadata(table, doc)
            
            # Add structural information
            enhanced_metadata.update({
                "extraction_method": "docling_structure_analysis",
                "total_cells": len(headers) * len(rows),
                "empty_cells": sum(1 for row in rows for cell in row.values() if not cell.strip()),
                "complexity_metrics": {
                    "header_row_count": 1,
                    "data_row_count": len(rows),
                    "column_count": len(headers),
                    "has_multi_line_cells": any('\n' in str(cell) for row in rows for cell in row.values()),
                    "has_merged_appearance": self._detect_merged_appearance(table_data)
                }
            })
            
            return TableData(
                headers=headers,
                rows=rows,
                metadata=enhanced_metadata
            )
            
        except Exception as e:
            logger.debug(f"Failed to extract table with structure: {e}")
            return None

    def _detect_merged_appearance(self, table_data: List) -> bool:
        """Detect if table appears to have merged cells based on spacing patterns."""
        try:
            if len(table_data) < 2:
                return False
            
            # Look for patterns that suggest merged cells
            first_row = table_data[0]
            
            # Check for large empty cells (potential merged cells)
            empty_count = sum(1 for cell in first_row if str(cell).strip() == "")
            total_cells = len(first_row)
            
            # If more than 30% of first row cells are empty, likely merged
            if total_cells > 0 and empty_count / total_cells > 0.3:
                return True
            
            return False
        except Exception:
            return False

    def _extract_table_properties(self, table, doc: Any) -> Optional[TableData]:
        """Extract table using Docling's native properties."""
        try:
            # Check for table title/caption
            title = getattr(table, 'title', None)
            caption = getattr(table, 'caption', None)
            
            # Check for table notes
            notes = getattr(table, 'notes', [])
            
            # Get table provenance
            provenance = getattr(table, 'provenance', {})
            
            # Check for table relationships
            relationships = getattr(table, 'relationships', [])
            
            # Extract hierarchical information
            section = getattr(table, 'section', None)
            subsection = getattr(table, 'subsection', None)
            
            if title or caption or notes:
                metadata = {
                    "source": "docling_properties",
                    "title": title,
                    "caption": caption,
                    "notes": notes,
                    "provenance": provenance,
                    "relationships": relationships,
                    "section": section,
                    "subsection": subsection
                }
                
                # Try to get basic table structure
                data = getattr(table, 'data', [])
                if data:
                    # Convert to basic format
                    headers = [str(cell).strip() for cell in data[0]] if data else []
                    rows = []
                    for row in data[1:]:
                        row_data = {}
                        for i, cell in enumerate(row):
                            if i < len(headers):
                                row_data[headers[i]] = str(cell).strip()
                        rows.append(row_data)
                    
                    return TableData(
                        headers=headers,
                        rows=rows,
                        metadata=metadata
                    )
        
        except Exception as e:
            logger.debug(f"Failed to extract table properties: {e}")
        
        return None


class PdfPlumberTableExtractor:
    """Table extractor using pdfplumber library."""
    
    def extract_tables(self, file_path: str, doc: Optional[Any] = None) -> List[TableData]:
        """Extract tables using pdfplumber."""
        try:
            tables = []
            
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        table_data = self._extract_pdfplumber_table(table)
                        if table_data:
                            tables.append(table_data)
            
            logger.info(f"pdfplumber extracted {len(tables)} tables")
            return tables
            
        except Exception as e:
            logger.error(f"pdfplumber table extraction failed: {e}")
            return []
    
    def _extract_pdfplumber_table(self, table) -> Optional[TableData]:
        """Extract data from pdfplumber table."""
        try:
            if not table or len(table) < 2:
                return None
            
            # Extract headers
            headers = []
            for cell in table[0]:
                if cell:
                    headers.append(str(cell).strip())
            
            headers = [h for h in headers if h]
            if not headers:
                return None
            
            # Extract rows
            rows = []
            for row in table[1:]:
                row_data = {}
                for i, cell in enumerate(row):
                    if i < len(headers) and cell:
                        cell_value = str(cell).strip()
                        if cell_value:
                            row_data[headers[i]] = cell_value
                if row_data:
                    rows.append(row_data)
            
            if not rows:
                return None
            
            return TableData(
                headers=headers,
                rows=rows,
                metadata={
                    "row_count": len(rows),
                    "column_count": len(headers),
                    "source": "pdfplumber"
                }
            )
            
        except Exception as e:
            logger.debug(f"Failed to extract pdfplumber table: {e}")
            return None


class TableClassifier:
    """Table classifier using rule-based approach."""
    
    # Classification rules and keywords
    RULES = {
        TableType.DISTRIBUTIONS: {
            "unique_columns": ["recallable"],
            "content_keywords": [
                "return of capital", "dividend", "income", "payout", "proceeds"
            ],
            "header_keywords": ["distribution", "return", "dividend", "payout"],
            "row_type_keywords": ["return of capital", "income", "dividend"]
        },
        TableType.CAPITAL_CALLS: {
            "unique_columns": ["call number", "called"],
            "content_keywords": [
                "initial capital", "follow-on", "bridge round", "commitment"
            ],
            "header_keywords": ["capital call", "contribution", "commitment"],
            "patterns": ["date", "amount"]
        },
        TableType.ADJUSTMENTS: {
            "content_keywords": [
                "management fee", "performance fee", "carried interest", "reimbursement"
            ],
            "header_keywords": ["fee", "expense", "adjustment", "rebalance"],
            "patterns": ["fee", "expense"]
        }
    }
    
    def classify(self, table_data: TableData) -> ClassificationResult:
        """
        Classify table using multi-level priority system.
        
        Priority order:
        1. Unique column identifiers
        2. Content-based keywords
        3. Header-based keywords
        4. Row type patterns
        5. Pattern-based heuristics
        """
        headers = [h.lower().strip() for h in table_data.headers]
        headers_text = " ".join(headers)
        
        # Sample content for analysis
        sample_rows = table_data.rows[:3]
        content_text = " ".join(
            str(value).lower() for row in sample_rows for value in row.values()
        )
        
        # Priority 1: Unique column identifiers
        for table_type, rules in self.RULES.items():
            if any(col in headers for col in rules.get("unique_columns", [])):
                return ClassificationResult(
                    table_type=table_type,
                    confidence=1.0,
                    reason=f"Unique column identifier: {rules['unique_columns']}"
                )
        
        # Priority 2: Content-based keywords
        for table_type, rules in self.RULES.items():
            if any(keyword in content_text for keyword in rules.get("content_keywords", [])):
                return ClassificationResult(
                    table_type=table_type,
                    confidence=0.9,
                    reason=f"Content keywords: {rules['content_keywords']}"
                )
        
        # Priority 3: Header-based keywords
        for table_type, rules in self.RULES.items():
            if any(keyword in headers_text for keyword in rules.get("header_keywords", [])):
                return ClassificationResult(
                    table_type=table_type,
                    confidence=0.8,
                    reason=f"Header keywords: {rules['header_keywords']}"
                )
        
        # Priority 4: Row type patterns
        for table_type, rules in self.RULES.items():
            row_type_keywords = rules.get("row_type_keywords", [])
            if row_type_keywords:
                for row in sample_rows:
                    row_type = str(row.get("Type", "")).lower()
                    if any(keyword in row_type for keyword in row_type_keywords):
                        return ClassificationResult(
                            table_type=table_type,
                            confidence=0.7,
                            reason=f"Row type pattern: {row_type_keywords}"
                        )
        
        # Priority 5: Pattern-based heuristics
        if self._matches_distribution_pattern(table_data):
            return ClassificationResult(
                table_type=TableType.DISTRIBUTIONS,
                confidence=0.6,
                reason="Distribution pattern (positive amounts, date+amount)"
            )
        
        logger.debug(f"Could not classify table with headers: {headers}")
        return ClassificationResult(
            table_type=None,
            confidence=0.0,
            reason="No matching patterns found"
        )
    
    def _matches_distribution_pattern(self, table_data: TableData) -> bool:
        """Check if table matches distribution pattern."""
        headers = [h.lower() for h in table_data.headers]
        
        # Must have date and amount columns
        has_date = any("date" in h for h in headers)
        has_amount = any("amount" in h for h in headers)
        
        if not (has_date and has_amount):
            return False
        
        # Check if amounts are mostly positive
        amounts = []
        for row in table_data.rows:
            for key, value in row.items():
                if "amount" in key.lower():
                    try:
                        amount = self._extract_amount(str(value))
                        if amount is not None:
                            amounts.append(amount)
                    except:
                        pass
        
        if not amounts:
            return False
        
        # More positive than negative amounts
        positive_count = len([a for a in amounts if a > 0])
        negative_count = len([a for a in amounts if a < 0])
        
        return positive_count > negative_count
    
    @staticmethod
    def _extract_amount(amount_str: str) -> Optional[float]:
        """Extract numeric amount from string."""
        data_parser = DataParser()
        return data_parser.parse_amount(amount_str)


class TableParser:
    """
    Main table parser coordinating extraction and classification.

    Uses composition and dependency injection for better testability.
    """

    def __init__(
        self,
        primary_extractor: Optional[TableExtractorProtocol] = None,
        fallback_extractor: Optional[TableExtractorProtocol] = None,
        classifier: Optional[TableClassifierProtocol] = None
    ):
        # Dependency injection with defaults
        self.primary_extractor = primary_extractor or DoclingTableExtractor()
        self.fallback_extractor = fallback_extractor or PdfPlumberTableExtractor()
        self.classifier = classifier or TableClassifier()

    def parse_tables(
        self,
        file_path: str = None,
        doc: Any = None
    ) -> Dict[str, Any]:
        """
        Parse and classify tables from PDF using enhanced Docling capabilities.

        Args:
            file_path: Path to PDF file
            doc: Pre-converted Docling document (optional)

        Returns:
            Dictionary with classified tables and enhanced metadata
        """
        if not file_path and not doc:
            raise ValueError("Either file_path or doc must be provided")

        classified_tables = {
            TableType.CAPITAL_CALLS.value: [],
            TableType.DISTRIBUTIONS.value: [],
            TableType.ADJUSTMENTS.value: [],
            "processing_method": ProcessingMethod.ERROR.value,
            "enhanced_metadata": {}
        }

        try:
            # Enhanced Docling table extraction with multiple fallback strategies
            tables = self._extract_tables_with_enhancements(file_path, doc)
            processing_method = tables.get("processing_method", ProcessingMethod.DOCLING)

            # Extract and classify tables
            classified_count = 0
            for table_data in tables.get("tables", []):
                result = self.classifier.classify(table_data)
                if result.table_type:
                    # Enhanced metadata with Docling insights
                    enhanced_metadata = self._enhance_table_metadata(
                        table_data.metadata, result, table_data
                    )
                    
                    classified_table = {
                        "headers": table_data.headers,
                        "rows": table_data.rows,
                        "metadata": enhanced_metadata
                    }
                    
                    classified_tables[result.table_type.value].append(classified_table)
                    classified_count += 1

            # Add enhanced processing metadata
            classified_tables["enhanced_metadata"] = {
                "total_tables_extracted": len(tables.get("tables", [])),
                "tables_classified": classified_count,
                "extraction_methods": tables.get("extraction_methods", []),
                "docling_confidence": tables.get("confidence_scores", []),
                "table_structures": tables.get("structure_types", []),
                "processing_time": tables.get("processing_time", 0)
            }

            classified_tables["processing_method"] = processing_method.value
            logger.info(f"Enhanced processing: {classified_count}/{len(tables.get('tables', []))} tables classified using {processing_method.value}")

        except Exception as e:
            logger.error(f"Enhanced table parsing failed: {e}")
            classified_tables["processing_method"] = ProcessingMethod.ERROR.value
            classified_tables["error"] = str(e)

        return classified_tables

    def _extract_tables_with_enhancements(self, file_path: str, doc: Any) -> Dict[str, Any]:
        """
        Extract tables with enhanced Docling capabilities and multiple fallback strategies.
        
        Args:
            file_path: Path to PDF file
            doc: Pre-converted Docling document (optional)
            
        Returns:
            Dictionary with tables and processing metadata
        """
        import time
        start_time = time.time()
        extraction_results = {
            "tables": [],
            "processing_method": ProcessingMethod.DOCLING,
            "extraction_methods": [],
            "confidence_scores": [],
            "structure_types": [],
            "processing_time": 0
        }

        try:
            # Primary: Enhanced Docling extraction
            if self.primary_extractor and hasattr(self.primary_extractor, 'converter'):
                extraction_results["extraction_methods"].append("docling_enhanced")
                tables = self.primary_extractor.extract_tables(file_path, doc)
                
                if tables:
                    extraction_results["tables"].extend(tables)
                    extraction_results["confidence_scores"].extend([getattr(t, 'confidence', 1.0) for t in tables])
                    extraction_results["structure_types"].extend([getattr(t, 'table_type', 'unknown') for t in tables])
                    logger.info(f"Enhanced Docling extracted {len(tables)} tables")
                else:
                    logger.info("Enhanced Docling found no tables, trying fallback")

            # Secondary: Standard Docling extraction
            if not extraction_results["tables"] and file_path:
                extraction_results["extraction_methods"].append("docling_standard")
                try:
                    from docling.document_converter import DocumentConverter
                    from docling.datamodel.pipeline_options import PdfPipelineOptions
                    from docling.document_converter import PdfFormatOption
                    from docling.datamodel.base_models import InputFormat
                    
                    pipeline_options = PdfPipelineOptions()
                    pipeline_options.do_ocr = False
                    pipeline_options.do_table_structure = True
                    
                    converter = DocumentConverter(
                        format_options={
                            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                        }
                    )
                    
                    if doc is None:
                        result = converter.convert(file_path)
                        doc = result.document
                    
                    # Extract tables using docling's native table structure
                    tables = []
                    if hasattr(doc, 'tables') and doc.tables:
                        for table in doc.tables:
                            if hasattr(table, 'export_to_dataframe'):
                                df = table.export_to_dataframe(doc=doc)
                                if not df.empty:
                                    table_data = TableData(
                                        headers=list(df.columns),
                                        rows=df.to_dict('records'),
                                        metadata={
                                            "source": "docling_native_dataframe",
                                            "confidence": getattr(table, 'confidence', 0.9)
                                        }
                                    )
                                    tables.append(table_data)
                    
                    extraction_results["tables"].extend(tables)
                    extraction_results["confidence_scores"].extend([0.9] * len(tables))
                    extraction_results["structure_types"].extend(["native"] * len(tables))
                    extraction_results["processing_method"] = ProcessingMethod.DOCLING
                    
                except Exception as e:
                    logger.warning(f"Standard Docling extraction failed: {e}")

            # Tertiary: pdfplumber fallback
            if not extraction_results["tables"] and file_path and self.fallback_extractor:
                extraction_results["extraction_methods"].append("pdfplumber_fallback")
                tables = self.fallback_extractor.extract_tables(file_path)
                extraction_results["tables"].extend(tables)
                extraction_results["confidence_scores"].extend([0.7] * len(tables))
                extraction_results["structure_types"].extend(["fallback"] * len(tables))
                extraction_results["processing_method"] = ProcessingMethod.PDFPLUMBER
                logger.info(f"pdfplumber fallback extracted {len(tables)} tables")

            # Quaternary: Manual table detection as last resort
            if not extraction_results["tables"] and file_path:
                extraction_results["extraction_methods"].append("manual_detection")
                manual_tables = self._manual_table_detection(file_path)
                extraction_results["tables"].extend(manual_tables)
                extraction_results["confidence_scores"].extend([0.5] * len(manual_tables))
                extraction_results["structure_types"].extend(["manual"] * len(manual_tables))
                extraction_results["processing_method"] = ProcessingMethod.DOCLING  # Still using docling for processing
            
        except Exception as e:
            logger.error(f"Enhanced table extraction failed: {e}")
            extraction_results["error"] = str(e)

        extraction_results["processing_time"] = time.time() - start_time
        return extraction_results

    def _manual_table_detection(self, file_path: str) -> List[TableData]:
        """Manual table detection as last resort using basic text analysis."""
        try:
            import pdfplumber
            tables = []
            
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Try multiple table extraction strategies
                    page_tables = page.extract_tables() or []
                    
                    for table_idx, table in enumerate(page_tables):
                        if table and len(table) > 1:  # At least header + 1 row
                            # Clean up table data
                            cleaned_table = []
                            for row in table:
                                if row and any(cell and str(cell).strip() for cell in row):
                                    cleaned_row = []
                                    for cell in row:
                                        if cell:
                                            cleaned_row.append(str(cell).strip())
                                        else:
                                            cleaned_row.append("")
                                    cleaned_table.append(cleaned_row)
                            
                            if len(cleaned_table) >= 2:
                                headers = cleaned_table[0]
                                rows = []
                                
                                for row in cleaned_table[1:]:
                                    row_dict = {}
                                    for i, cell in enumerate(row):
                                        if i < len(headers) and cell:
                                            header = headers[i] if headers[i] else f"Column_{i+1}"
                                            row_dict[header] = cell
                                    if row_dict:
                                        rows.append(row_dict)
                                
                                if headers and rows:
                                    table_data = TableData(
                                        headers=headers,
                                        rows=rows,
                                        metadata={
                                            "source": "manual_detection",
                                            "page": page_num,
                                            "table_index": table_idx,
                                            "confidence": 0.5,
                                            "extraction_method": "basic_text_analysis"
                                        }
                                    )
                                    tables.append(table_data)
            
            return tables
            
        except Exception as e:
            logger.warning(f"Manual table detection failed: {e}")
            return []

    def _enhance_table_metadata(self, base_metadata: Dict[str, Any],
                               classification_result: "ClassificationResult",
                               table_data: TableData) -> Dict[str, Any]:
        """
        Enhance table metadata with Docling insights and classification results.
        
        Args:
            base_metadata: Base metadata from extraction
            classification_result: Result from table classification
            table_data: TableData object with extracted content
            
        Returns:
            Enhanced metadata dictionary
        """
        enhanced_metadata = base_metadata.copy()
        
        # Add classification insights
        enhanced_metadata.update({
            "classification_confidence": classification_result.confidence,
            "classification_reason": classification_result.reason,
            "table_type_predicted": classification_result.table_type.value if classification_result.table_type else "unknown"
        })
        
        # Add structural analysis
        if table_data.headers and table_data.rows:
            # Analyze column patterns
            column_patterns = self._analyze_column_patterns(table_data.headers, table_data.rows)
            enhanced_metadata["column_analysis"] = column_patterns
            
            # Calculate table metrics
            table_metrics = self._calculate_table_metrics(table_data)
            enhanced_metadata["table_metrics"] = table_metrics
            
            # Add data quality assessment
            quality_score = self._assess_table_quality(table_data)
            enhanced_metadata["quality_score"] = quality_score
        
        # Add processing insights
        enhanced_metadata.update({
            "enhancement_level": "docling_enhanced",
            "has_structural_data": bool(base_metadata.get("structure")),
            "extraction_confidence": base_metadata.get("confidence", 0.8)
        })
        
        return enhanced_metadata

    def _analyze_column_patterns(self, headers: List[str], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in table columns to enhance understanding."""
        if not headers or not rows:
            return {}
        
        patterns = {
            "date_columns": [],
            "numeric_columns": [],
            "text_columns": [],
            "currency_columns": []
        }
        
        for header in headers:
            header_lower = header.lower()
            sample_values = [row.get(header, "") for row in rows[:5] if header in row]
            
            # Check for date patterns
            date_indicators = ["date", "time", "period", "year", "month"]
            if any(indicator in header_lower for indicator in date_indicators):
                patterns["date_columns"].append(header)
            
            # Check for currency patterns
            currency_indicators = ["amount", "value", "price", "cost", "fee"]
            if any(indicator in header_lower for indicator in currency_indicators):
                patterns["currency_columns"].append(header)
            
            # Check for numeric values
            numeric_count = sum(1 for value in sample_values if self._is_numeric(value))
            if numeric_count > len(sample_values) * 0.7:
                patterns["numeric_columns"].append(header)
            else:
                patterns["text_columns"].append(header)
        
        return patterns

    def _is_numeric(self, value: str) -> bool:
        """Check if a value can be parsed as numeric."""
        if not value or not str(value).strip():
            return False
        
        import re
        # Remove common currency symbols and commas
        cleaned = re.sub(r'[$€£¥,\s]', '', str(value))
        try:
            float(cleaned)
            return True
        except ValueError:
            return False

    def _calculate_table_metrics(self, table_data: TableData) -> Dict[str, Any]:
        """Calculate various metrics about the table structure."""
        if not table_data.rows:
            return {}
        
        metrics = {
            "total_rows": len(table_data.rows),
            "total_columns": len(table_data.headers),
            "completeness_ratio": 0,
            "average_row_width": 0,
            "max_column_width": 0
        }
        
        # Calculate completeness
        total_cells = metrics["total_rows"] * metrics["total_columns"]
        non_empty_cells = sum(
            1 for row in table_data.rows
            for value in row.values()
            if value and str(value).strip()
        )
        
        if total_cells > 0:
            metrics["completeness_ratio"] = non_empty_cells / total_cells
        
        # Calculate width metrics
        column_widths = [len(header) for header in table_data.headers]
        if column_widths:
            metrics["average_row_width"] = sum(column_widths) / len(column_widths)
            metrics["max_column_width"] = max(column_widths)
        
        return metrics

    def _assess_table_quality(self, table_data: TableData) -> float:
        """Assess the quality of extracted table data."""
        quality_score = 0.0
        
        # Check if we have reasonable number of rows and columns
        if 1 <= len(table_data.rows) <= 1000 and 1 <= len(table_data.headers) <= 20:
            quality_score += 0.3
        
        # Check completeness
        metrics = self._calculate_table_metrics(table_data)
        completeness = metrics.get("completeness_ratio", 0)
        if completeness > 0.7:
            quality_score += 0.4
        elif completeness > 0.4:
            quality_score += 0.2
        
        # Check if headers are meaningful
        meaningful_headers = sum(1 for header in table_data.headers if len(header.strip()) > 2)
        if meaningful_headers / len(table_data.headers) > 0.8:
            quality_score += 0.2
        
        # Check for common data patterns
        data_patterns = self._analyze_column_patterns(table_data.headers, table_data.rows)
        if data_patterns["date_columns"] or data_patterns["numeric_columns"]:
            quality_score += 0.1
        
        return min(quality_score, 1.0)