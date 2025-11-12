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
            pipeline_options.do_ocr = False
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
        """Extract data from Docling table object."""
        try:
            # Try DataFrame export first
            if hasattr(table, 'export_to_dataframe'):
                df = table.export_to_dataframe(doc=doc)
                if not df.empty:
                    return TableData(
                        headers=list(df.columns),
                        rows=df.to_dict('records'),
                        metadata={
                            "row_count": len(df),
                            "column_count": len(df.columns),
                            "source": "docling_dataframe"
                        }
                    )
            
            # Fallback to direct data access
            if hasattr(table, 'data') and table.data:
                headers = []
                if table.data:
                    first_row = table.data[0]
                    headers = [
                        cell.text.strip() if hasattr(cell, 'text') else str(cell).strip()
                        for cell in first_row
                    ]
                    headers = [h for h in headers if h]
                
                rows = []
                for row in table.data[1:]:
                    row_data = {}
                    for i, cell in enumerate(row):
                        if i < len(headers):
                            cell_value = cell.text.strip() if hasattr(cell, 'text') else str(cell).strip()
                            if cell_value:
                                row_data[headers[i]] = cell_value
                    if row_data:
                        rows.append(row_data)
                
                if headers and rows:
                    return TableData(
                        headers=headers,
                        rows=rows,
                        metadata={
                            "row_count": len(rows),
                            "column_count": len(headers),
                            "source": "docling_direct"
                        }
                    )
        
        except Exception as e:
            logger.debug(f"Failed to extract Docling table: {e}")
        
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
        if not amount_str:
            return None
        
        # Clean string
        cleaned = re.sub(r"[$,€£¥₹\s]", "", amount_str)
        
        # Handle parentheses
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]
        
        try:
            return float(cleaned)
        except ValueError:
            return None


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
        Parse and classify tables from PDF.
        
        Args:
            file_path: Path to PDF file
            doc: Pre-converted Docling document (optional)
            
        Returns:
            Dictionary with classified tables and metadata
        """
        if not file_path and not doc:
            raise ValueError("Either file_path or doc must be provided")
        
        classified_tables = {
            TableType.CAPITAL_CALLS.value: [],
            TableType.DISTRIBUTIONS.value: [],
            TableType.ADJUSTMENTS.value: [],
            "processing_method": ProcessingMethod.ERROR.value
        }
        
        try:
            # Try primary extractor (Docling)
            tables = self.primary_extractor.extract_tables(file_path, doc)
            processing_method = ProcessingMethod.DOCLING
            
            # Fallback to pdfplumber if no tables found
            if not tables and file_path:
                logger.info("Primary extractor found no tables, trying fallback")
                tables = self.fallback_extractor.extract_tables(file_path)
                processing_method = ProcessingMethod.PDFPLUMBER
            
            # Classify tables
            for table in tables:
                result = self.classifier.classify(table)
                if result.table_type:
                    table.metadata.update({
                        "classification_confidence": result.confidence,
                        "classification_reason": result.reason
                    })
                    classified_tables[result.table_type.value].append({
                        "headers": table.headers,
                        "rows": table.rows,
                        "metadata": table.metadata
                    })
            
            classified_tables["processing_method"] = processing_method.value
            logger.info(f"Processed {len(tables)} tables using {processing_method.value}")
            
        except Exception as e:
            logger.error(f"Table parsing failed: {e}")
            classified_tables["processing_method"] = ProcessingMethod.ERROR.value
        
        return classified_tables