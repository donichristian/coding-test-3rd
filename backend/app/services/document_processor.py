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
from app.services.data_parser import DataParser
from app.services.text_chunker import TextChunker
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
        """Extract text content from docling document with enhanced metadata."""
        try:
            text_content = []

            # Method 1: Extract structured text content with hierarchy
            if hasattr(doc, 'content_items') and doc.content_items:
                text_items = self._extract_structured_text(doc)
                text_content.extend(text_items)
            
            # Method 2: Extract entire document text
            elif hasattr(doc, 'text') and doc.text:
                text_content.append({
                    "page": 1,
                    "content": doc.text,
                    "type": "document_text",
                    "metadata": self._extract_text_metadata(doc)
                })
            
            # Method 3: Fallback to markdown export
            else:
                markdown_content = doc.export_to_markdown()
                if markdown_content.strip():
                    text_content.append({
                        "page": 1,
                        "content": markdown_content,
                        "type": "markdown_text",
                        "metadata": self._extract_text_metadata(doc)
                    })

            return text_content

        except Exception as e:
            logger.error(f"Enhanced text extraction failed: {e}")
            return [{
                "page": 1,
                "content": "Document processed but text extraction failed",
                "type": "error_text",
                "metadata": {"error": str(e)}
            }]

    def _extract_structured_text(self, doc) -> List[Dict[str, Any]]:
        """Extract structured text content using Docling's content items."""
        text_content = []
        
        try:
            # Group content items by page
            page_content = {}
            
            for item in doc.content_items:
                # Get page number
                page_num = getattr(item, 'page_number', 1)
                if page_num not in page_content:
                    page_content[page_num] = {
                        "text_blocks": [],
                        "headings": [],
                        "lists": [],
                        "paragraphs": []
                    }
                
                # Extract different content types
                if hasattr(item, 'text') and item.text:
                    content_type = getattr(item, 'content_type', 'text')
                    
                    content_item = {
                        "page": page_num,
                        "content": item.text,
                        "type": content_type,
                        "metadata": self._extract_content_item_metadata(item, doc)
                    }
                    
                    # Categorize by content type
                    if content_type == 'heading':
                        page_content[page_num]["headings"].append(content_item)
                    elif content_type == 'list_item':
                        page_content[page_num]["lists"].append(content_item)
                    elif content_type == 'paragraph':
                        page_content[page_num]["paragraphs"].append(content_item)
                    else:
                        page_content[page_num]["text_blocks"].append(content_item)
            
            # Combine all content for each page
            for page_num, content_items in page_content.items():
                all_content = []
                
                # Add headings first
                all_content.extend(content_items["headings"])
                
                # Add paragraphs and text blocks
                all_content.extend(content_items["paragraphs"])
                all_content.extend(content_items["text_blocks"])
                
                # Add list items
                all_content.extend(content_items["lists"])
                
                # Combine into full page text
                page_text = "\n\n".join(item["content"] for item in all_content if item["content"])
                
                if page_text.strip():
                    text_content.append({
                        "page": page_num,
                        "content": page_text,
                        "type": "structured_text",
                        "content_breakdown": {
                            "headings": len(content_items["headings"]),
                            "paragraphs": len(content_items["paragraphs"]),
                            "text_blocks": len(content_items["text_blocks"]),
                            "list_items": len(content_items["lists"])
                        },
                        "metadata": self._extract_text_metadata(doc)
                    })
        
        except Exception as e:
            logger.debug(f"Structured text extraction failed: {e}")
            # Fallback to simple text extraction
            if hasattr(doc, 'text') and doc.text:
                text_content.append({
                    "page": 1,
                    "content": doc.text,
                    "type": "document_text",
                    "metadata": self._extract_text_metadata(doc)
                })
        
        return text_content

    def _extract_content_item_metadata(self, item, doc) -> Dict[str, Any]:
        """Extract metadata for individual content items."""
        metadata = {
            "content_type": getattr(item, 'content_type', 'text'),
            "page_number": getattr(item, 'page_number', 1),
            "level": getattr(item, 'level', 0),
            "reading_order": getattr(item, 'reading_order', 0)
        }
        
        # Extract positional information
        if hasattr(item, 'bbox'):
            bbox = item.bbox
            if hasattr(bbox, 'to_dict'):
                metadata["bounding_box"] = bbox.to_dict()
            else:
                metadata["bounding_box"] = str(bbox)
        
        # Extract text properties
        if hasattr(item, 'text_properties'):
            text_props = item.text_properties
            metadata["text_properties"] = {
                "font_size": getattr(text_props, 'font_size', None),
                "font_family": getattr(text_props, 'font_family', None),
                "is_bold": getattr(text_props, 'is_bold', False),
                "is_italic": getattr(text_props, 'is_italic', False)
            }
        
        return metadata

    def _extract_text_metadata(self, doc) -> Dict[str, Any]:
        """Extract comprehensive text metadata using Docling's native capabilities."""
        metadata = {
            "word_count": 0,
            "character_count": 0,
            "sentence_count": 0,
            "paragraph_count": 0,
            "language_detected": "unknown",
            "has_formulas": False,
            "has_citations": False,
            "has_references": False,
            "complexity_score": 0.0
        }
        
        try:
            # Extract text statistics
            if hasattr(doc, 'text') and doc.text:
                text = doc.text
                
                # Basic counts
                metadata["character_count"] = len(text)
                metadata["word_count"] = len(text.split())
                
                # Sentence detection
                import re
                sentences = re.split(r'[.!?]+', text)
                metadata["sentence_count"] = len([s for s in sentences if s.strip()])
                
                # Paragraph detection
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                metadata["paragraph_count"] = len(paragraphs)
                
                # Language detection (basic)
                if self._detect_language(text):
                    metadata["language_detected"] = self._detect_language(text)
                
                # Content analysis
                metadata["has_formulas"] = bool(re.search(r'[\$\\[\\]|\w+_\{.*\}|\^.*\{', text))
                metadata["has_citations"] = bool(re.search(r'\[\d+\]|\(\d{4}\)|\b(?:et al\.|ibid\.|op\. cit\.)\b', text))
                metadata["has_references"] = bool(re.search(r'(?i)references?|bibliography|works cited', text))
                
                # Complexity score (0-1)
                metadata["complexity_score"] = self._calculate_text_complexity(text)
        
        except Exception as e:
            logger.debug(f"Text metadata extraction failed: {e}")
        
        return metadata

    def _detect_language(self, text: str) -> str:
        """Basic language detection using common word patterns."""
        if not text or len(text.strip()) < 50:
            return "unknown"
        
        # Simple language detection based on common words
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were']
        
        words = text.lower().split()
        english_count = sum(1 for word in words if word in english_words)
        
        if english_count / len(words) > 0.05:
            return "en"
        elif any(ord(char) > 127 for char in text[:100]):  # Non-ASCII characters
            return "other"
        
        return "unknown"

    def _calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity score (0-1)."""
        if not text:
            return 0.0
        
        complexity = 0.0
        
        # Average word length
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            complexity += min(avg_word_length / 10, 0.3)
        
        # Average sentence length
        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            complexity += min(avg_sentence_length / 20, 0.4)
        
        # Technical indicators
        technical_indicators = ['however', 'therefore', 'furthermore', 'moreover', 'consequently', 'nevertheless']
        words_lower = text.lower().split()
        technical_count = sum(1 for word in words_lower if word in technical_indicators)
        complexity += min(technical_count / 10, 0.3)
        
        return min(complexity, 1.0)

    def extract_tables(self, file_path: str, doc) -> Dict[str, List[Dict[str, Any]]]:
        """Extract and classify tables with enhanced metadata."""
        try:
            return self.table_parser.parse_tables(file_path=file_path, doc=doc)
        except Exception as e:
            logger.error(f"Enhanced table extraction failed: {e}")
            return {
                "capital_calls": [],
                "distributions": [],
                "adjustments": [],
                "processing_method": "error",
                "error": str(e)
            }

    def extract_document_metadata(self, doc) -> Dict[str, Any]:
        """Extract comprehensive document metadata using Docling's native capabilities."""
        try:
            metadata = {
                # Basic document information
                "title": getattr(doc, 'title', None),
                "author": getattr(doc, 'author', None),
                "subject": getattr(doc, 'subject', None),
                "keywords": getattr(doc, 'keywords', None),
                "creator": getattr(doc, 'creator', None),
                "producer": getattr(doc, 'producer', None),
                "creation_date": getattr(doc, 'creation_date', None),
                "modification_date": getattr(doc, 'modification_date', None),
                
                # Document structure and properties
                "page_count": len(doc.pages) if hasattr(doc, 'pages') else 0,
                "language": getattr(doc, 'language', 'unknown'),
                "document_type": getattr(doc, 'document_type', 'unknown'),
                
                # Content analysis
                "has_text": getattr(doc, 'has_text', False),
                "has_images": getattr(doc, 'has_images', False),
                "has_tables": getattr(doc, 'has_tables', False),
                "has_formulas": getattr(doc, 'has_formulas', False),
                "has_annotations": getattr(doc, 'has_annotations', False),
                
                # Processing information
                "processing_confidence": getattr(doc, 'processing_confidence', 1.0),
                "source_file_info": getattr(doc, 'source_file_info', {}),
                "metadata_version": getattr(doc, 'metadata_version', '1.0')
            }
            
            # Extract document-level statistics
            if hasattr(doc, 'text') and doc.text:
                text_stats = self._analyze_document_text(doc.text)
                metadata.update(text_stats)
            
            # Extract layout information
            if hasattr(doc, 'pages') and doc.pages:
                layout_info = self._analyze_document_layout(doc)
                metadata.update(layout_info)
            
            # Extract reading order and hierarchy
            if hasattr(doc, 'reading_order') and doc.reading_order:
                hierarchy_info = self._analyze_document_hierarchy(doc)
                metadata.update(hierarchy_info)
            
            # Add custom metadata extraction
            custom_metadata = self._extract_custom_metadata(doc)
            metadata.update(custom_metadata)
            
            logger.info(f"✓ Extracted comprehensive metadata: {len(metadata)} fields")
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return {
                "error": str(e),
                "extraction_successful": False,
                "fallback_metadata": True
            }

    def _analyze_document_text(self, text: str) -> Dict[str, Any]:
        """Analyze document text for comprehensive statistics."""
        if not text:
            return {}
        
        import re
        
        # Text statistics
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Content complexity
        unique_words = set(word.lower().strip('.,!?;:"') for word in words)
        vocabulary_richness = len(unique_words) / len(words) if words else 0
        
        # Technical content indicators
        technical_indicators = {
            "formulas": len(re.findall(r'[\$\\[\\]|\w+_\{.*\}|\^.*\{', text)),
            "tables_references": len(re.findall(r'(?i)table\s+\d+|figure\s+\d+|chart\s+\d+', text)),
            "citations": len(re.findall(r'\[\d+\]|\(\d{4}\)|et al\.', text)),
            "acronyms": len(re.findall(r'\b[A-Z]{2,}\b', text)),
            "numbers": len(re.findall(r'\b\d+\b', text))
        }
        
        return {
            "text_statistics": {
                "word_count": len(words),
                "unique_word_count": len(unique_words),
                "sentence_count": len([s for s in sentences if s.strip()]),
                "paragraph_count": len(paragraphs),
                "character_count": len(text),
                "vocabulary_richness": round(vocabulary_richness, 3),
                "average_words_per_sentence": round(len(words) / max(len(sentences), 1), 2)
            },
            "content_analysis": technical_indicators
        }

    def _analyze_document_layout(self, doc) -> Dict[str, Any]:
        """Analyze document layout structure."""
        if not hasattr(doc, 'pages') or not doc.pages:
            return {}
        
        layout_stats = {
            "layout_analysis": {
                "pages_with_text": 0,
                "pages_with_images": 0,
                "pages_with_tables": 0,
                "pages_with_headers": 0,
                "pages_with_footers": 0,
                "multi_column_pages": 0
            }
        }
        
        for page in doc.pages:
            # Check page content types
            if hasattr(page, 'text_items') and page.text_items:
                layout_stats["layout_analysis"]["pages_with_text"] += 1
            
            if hasattr(page, 'image_items') and page.image_items:
                layout_stats["layout_analysis"]["pages_with_images"] += 1
            
            if hasattr(page, 'table_items') and page.table_items:
                layout_stats["layout_analysis"]["pages_with_tables"] += 1
            
            # Check for headers/footers
            if hasattr(page, 'header_items') and page.header_items:
                layout_stats["layout_analysis"]["pages_with_headers"] += 1
            
            if hasattr(page, 'footer_items') and page.footer_items:
                layout_stats["layout_analysis"]["pages_with_footers"] += 1
            
            # Check for multi-column layout
            if hasattr(page, 'column_layout') and page.column_layout:
                layout_stats["layout_analysis"]["multi_column_pages"] += 1
        
        return layout_stats

    def _analyze_document_hierarchy(self, doc) -> Dict[str, Any]:
        """Analyze document reading order and hierarchy."""
        hierarchy_info = {
            "hierarchy_analysis": {
                "has_reading_order": False,
                "section_count": 0,
                "subsection_count": 0,
                "heading_levels": [],
                "document_outline": []
            }
        }
        
        try:
            if hasattr(doc, 'reading_order') and doc.reading_order:
                hierarchy_info["hierarchy_analysis"]["has_reading_order"] = True
                
                # Extract structural elements
                sections = []
                headings = []
                
                if hasattr(doc, 'content_items'):
                    for item in doc.content_items:
                        if hasattr(item, 'content_type'):
                            if item.content_type == 'heading':
                                level = getattr(item, 'level', 1)
                                headings.append({
                                    "level": level,
                                    "text": getattr(item, 'text', ''),
                                    "page": getattr(item, 'page_number', 1)
                                })
                            elif item.content_type == 'section':
                                sections.append({
                                    "text": getattr(item, 'text', ''),
                                    "page": getattr(item, 'page_number', 1)
                                })
                
                hierarchy_info["hierarchy_analysis"]["section_count"] = len(sections)
                hierarchy_info["hierarchy_analysis"]["subsection_count"] = len([h for h in headings if h["level"] > 1])
                hierarchy_info["hierarchy_analysis"]["heading_levels"] = list(set(h["level"] for h in headings))
                hierarchy_info["hierarchy_analysis"]["document_outline"] = headings
        
        except Exception as e:
            logger.debug(f"Hierarchy analysis failed: {e}")
        
        return hierarchy_info

    def _extract_custom_metadata(self, doc) -> Dict[str, Any]:
        """Extract custom metadata based on document type."""
        custom_metadata = {
            "document_classification": {
                "type": "unknown",
                "confidence": 0.0,
                "indicators": []
            }
        }
        
        try:
            # Analyze document characteristics for classification
            indicators = []
            
            # Check for financial documents
            if hasattr(doc, 'text') and doc.text:
                text_lower = doc.text.lower()
                financial_keywords = ['fund', 'investment', 'capital', 'dividend', 'return', 'performance', 'portfolio']
                financial_score = sum(1 for keyword in financial_keywords if keyword in text_lower)
                
                if financial_score > 5:
                    indicators.append("financial_document")
                    custom_metadata["document_classification"]["type"] = "financial"
                    custom_metadata["document_classification"]["confidence"] = min(financial_score / 10, 1.0)
            
            # Check for table-heavy documents
            if hasattr(doc, 'has_tables') and doc.has_tables:
                indicators.append("table_heavy")
                if custom_metadata["document_classification"]["type"] == "unknown":
                    custom_metadata["document_classification"]["type"] = "tabular"
            
            # Check for image-heavy documents
            if hasattr(doc, 'has_images') and doc.has_images:
                indicators.append("image_heavy")
            
            # Check for form-like documents
            if hasattr(doc, 'form_fields') and doc.form_fields:
                indicators.append("form_document")
            
            custom_metadata["document_classification"]["indicators"] = indicators
            
        except Exception as e:
            logger.debug(f"Custom metadata extraction failed: {e}")
        
        return custom_metadata

    def extract_content_types(self, doc) -> Dict[str, List[Dict[str, Any]]]:
        """Extract and classify different content types using Docling's analysis."""
        content_types = {
            "text_blocks": [],
            "tables": [],
            "images": [],
            "formulas": [],
            "captions": [],
            "headings": [],
            "lists": []
        }
        
        try:
            if hasattr(doc, 'content_items'):
                for item in doc.content_items:
                    content_item = {
                        "content": getattr(item, 'text', ''),
                        "page": getattr(item, 'page_number', 1),
                        "bbox": getattr(item, 'bbox', None),
                        "metadata": self._extract_content_item_metadata(item, doc)
                    }
                    
                    content_type = getattr(item, 'content_type', 'text')
                    
                    if content_type == 'heading':
                        content_types["headings"].append(content_item)
                    elif content_type == 'table':
                        content_types["tables"].append(content_item)
                    elif content_type == 'image':
                        content_types["images"].append(content_item)
                    elif content_type == 'formula':
                        content_types["formulas"].append(content_item)
                    elif content_type == 'caption':
                        content_types["captions"].append(content_item)
                    elif content_type == 'list_item':
                        content_types["lists"].append(content_item)
                    else:
                        content_types["text_blocks"].append(content_item)
            
            logger.info(f"✓ Extracted content types: {sum(len(v) for v in content_types.values())} items")
            
        except Exception as e:
            logger.error(f"Content type extraction failed: {e}")
        
        return content_types

    def extract_formulas_and_references(self, doc) -> Dict[str, List[Dict[str, Any]]]:
        """Extract formulas, citations, and references using Docling's native capabilities."""
        formulas_references = {
            "mathematical_formulas": [],
            "chemical_formulas": [],
            "code_snippets": [],
            "citations": [],
            "bibliographic_references": [],
            "figure_references": [],
            "table_references": [],
            "equation_numbers": []
        }
        
        try:
            # Extract formulas using Docling's native formula detection
            if hasattr(doc, 'formulas') and doc.formulas:
                for formula in doc.formulas:
                    formula_item = {
                        "content": getattr(formula, 'text', ''),
                        "latex": getattr(formula, 'latex', None),
                        "page": getattr(formula, 'page_number', 1),
                        "equation_number": getattr(formula, 'equation_number', None),
                        "bbox": getattr(formula, 'bbox', None),
                        "formula_type": getattr(formula, 'formula_type', 'unknown'),
                        "metadata": self._extract_item_metadata(formula)
                    }
                    formulas_references["mathematical_formulas"].append(formula_item)
            
            # Extract from content items for formulas and references
            if hasattr(doc, 'content_items'):
                for item in doc.content_items:
                    content = getattr(item, 'text', '')
                    if not content:
                        continue
                    
                    # Extract mathematical formulas
                    formulas = self._extract_mathematical_formulas(content, item)
                    for formula in formulas:
                        formulas_references["mathematical_formulas"].append(formula)
                    
                    # Extract chemical formulas
                    chemical_formulas = self._extract_chemical_formulas(content, item)
                    formulas_references["chemical_formulas"].extend(chemical_formulas)
                    
                    # Extract code snippets
                    code_snippets = self._extract_code_snippets(content, item)
                    formulas_references["code_snippets"].extend(code_snippets)
                    
                    # Extract citations and references
                    citations = self._extract_citations(content, item)
                    formulas_references["citations"].extend(citations)
                    
                    # Extract references
                    references = self._extract_references(content, item)
                    formulas_references["bibliographic_references"].extend(references)
                    
                    # Extract figure and table references
                    figure_refs = self._extract_figure_references(content, item)
                    table_refs = self._extract_table_references(content, item)
                    formulas_references["figure_references"].extend(figure_refs)
                    formulas_references["table_references"].extend(table_refs)
            
            logger.info(f"✓ Extracted formulas and references: {sum(len(v) for v in formulas_references.values())} items")
            
        except Exception as e:
            logger.error(f"Formula and reference extraction failed: {e}")
        
        return formulas_references

    def _extract_mathematical_formulas(self, content: str, item) -> List[Dict[str, Any]]:
        """Extract mathematical formulas from content."""
        import re
        
        formulas = []
        
        # LaTeX-style formulas
        latex_patterns = [
            r'\$(.+?)\$',  # Inline math
            r'\$\$(.+?)\$\$',  # Display math
            r'\\begin\{equation\}(.+?)\\end\{equation\}',
            r'\\begin\{align\}(.+?)\\end\{align\}'
        ]
        
        for pattern in latex_patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                formulas.append({
                    "content": match.group(1) if match.groups() else match.group(0),
                    "type": "latex",
                    "page": getattr(item, 'page_number', 1),
                    "equation_number": self._extract_equation_number(content, match.start()),
                    "bbox": getattr(item, 'bbox', None),
                    "metadata": self._extract_item_metadata(item)
                })
        
        # Unicode mathematical symbols
        math_symbols = ['∑', '∫', '√', '∞', 'π', 'α', 'β', 'γ', 'σ', 'μ']
        if any(symbol in content for symbol in math_symbols):
            formulas.append({
                "content": content,
                "type": "unicode_math",
                "page": getattr(item, 'page_number', 1),
                "equation_number": self._extract_equation_number(content, 0),
                "bbox": getattr(item, 'bbox', None),
                "metadata": self._extract_item_metadata(item)
            })
        
        return formulas

    def _extract_chemical_formulas(self, content: str, item) -> List[Dict[str, Any]]:
        """Extract chemical formulas from content."""
        import re
        
        # Chemical formula patterns (e.g., H2O, CH3COOH, Fe2O3)
        chemical_pattern = r'\b[A-Z][a-z]?(?:\d*[a-z]?)*(?:\d+)?\b'
        
        matches = re.finditer(chemical_pattern, content)
        formulas = []
        
        for match in matches:
            formula = match.group(0)
            # Filter out common words that might match
            if len(formula) > 1 and not self._is_common_word(formula):
                formulas.append({
                    "content": formula,
                    "type": "chemical",
                    "page": getattr(item, 'page_number', 1),
                    "bbox": getattr(item, 'bbox', None),
                    "metadata": self._extract_item_metadata(item)
                })
        
        return formulas

    def _extract_code_snippets(self, content: str, item) -> List[Dict[str, Any]]:
        """Extract code snippets from content."""
        import re
        
        code_blocks = []
        
        # Markdown code blocks
        code_pattern = r'```[\w]*\n(.*?)\n```'
        matches = re.finditer(code_pattern, content, re.DOTALL)
        
        for match in matches:
            code_content = match.group(1)
            language = self._detect_code_language(code_content)
            
            code_blocks.append({
                "content": code_content,
                "language": language,
                "type": "code_block",
                "page": getattr(item, 'page_number', 1),
                "bbox": getattr(item, 'bbox', None),
                "metadata": self._extract_item_metadata(item)
            })
        
        # Inline code
        inline_code_pattern = r'`([^`]+)`'
        matches = re.finditer(inline_code_pattern, content)
        
        for match in matches:
            code_content = match.group(1)
            if len(code_content.strip()) > 2:  # Avoid very short matches
                code_blocks.append({
                    "content": code_content,
                    "type": "inline_code",
                    "page": getattr(item, 'page_number', 1),
                    "bbox": getattr(item, 'bbox', None),
                    "metadata": self._extract_item_metadata(item)
                })
        
        return code_blocks

    def _extract_citations(self, content: str, item) -> List[Dict[str, Any]]:
        """Extract citations from content."""
        import re
        
        citations = []
        
        # IEEE style: [1], [2, 3], [1-3]
        ieee_pattern = r'\[(\d+(?:[-–]\d+)?(?:,\s*\d+)*)\]'
        matches = re.finditer(ieee_pattern, content)
        
        for match in matches:
            citations.append({
                "content": match.group(0),
                "numbers": match.group(1),
                "style": "ieee",
                "page": getattr(item, 'page_number', 1),
                "position": match.start(),
                "bbox": getattr(item, 'bbox', None),
                "metadata": self._extract_item_metadata(item)
            })
        
        # APA style: (Author, Year)
        apa_pattern = r'\(([A-Z][a-zA-Z]+,?\s+\d{4}[a-z]?)\)'
        matches = re.finditer(apa_pattern, content)
        
        for match in matches:
            citations.append({
                "content": match.group(0),
                "author_year": match.group(1),
                "style": "apa",
                "page": getattr(item, 'page_number', 1),
                "position": match.start(),
                "bbox": getattr(item, 'bbox', None),
                "metadata": self._extract_item_metadata(item)
            })
        
        # et al. references
        etal_pattern = r'\([^)]*et al\.\s*\d{4}[^)]*\)'
        matches = re.finditer(etal_pattern, content, re.IGNORECASE)
        
        for match in matches:
            citations.append({
                "content": match.group(0),
                "style": "etal",
                "page": getattr(item, 'page_number', 1),
                "position": match.start(),
                "bbox": getattr(item, 'bbox', None),
                "metadata": self._extract_item_metadata(item)
            })
        
        return citations

    def _extract_references(self, content: str, item) -> List[Dict[str, Any]]:
        """Extract bibliographic references from content."""
        import re
        
        references = []
        
        # Common reference section headers
        reference_headers = [
            r'references?\b',
            r'works\s+cited\b',
            r'bibliography\b',
            r'literature\s+cited\b'
        ]
        
        for header_pattern in reference_headers:
            header_match = re.search(header_pattern, content, re.IGNORECASE)
            if header_match:
                # Extract text after reference header
                ref_section = content[header_match.end():]
                
                # Split into individual references (usually separated by blank lines)
                ref_blocks = re.split(r'\n\s*\n', ref_section.strip())
                
                for ref_text in ref_blocks:
                    if len(ref_text.strip()) > 50:  # Likely a reference
                        references.append({
                            "content": ref_text.strip(),
                            "type": "bibliographic",
                            "page": getattr(item, 'page_number', 1),
                            "bbox": getattr(item, 'bbox', None),
                            "metadata": self._extract_item_metadata(item)
                        })
        
        return references

    def _extract_figure_references(self, content: str, item) -> List[Dict[str, Any]]:
        """Extract figure references from content."""
        import re
        
        figure_refs = []
        
        # Figure references: "Figure 1.1", "Fig. 2", "Figure 3A"
        figure_pattern = r'(?:Figure|Fig\.)\s+(\d+[A-Z]?)'
        matches = re.finditer(figure_pattern, content, re.IGNORECASE)
        
        for match in matches:
            figure_refs.append({
                "content": match.group(0),
                "figure_number": match.group(1),
                "type": "figure_reference",
                "page": getattr(item, 'page_number', 1),
                "position": match.start(),
                "bbox": getattr(item, 'bbox', None),
                "metadata": self._extract_item_metadata(item)
            })
        
        return figure_refs

    def _extract_table_references(self, content: str, item) -> List[Dict[str, Any]]:
        """Extract table references from content."""
        import re
        
        table_refs = []
        
        # Table references: "Table 2.1", "Table 3"
        table_pattern = r'Table\s+(\d+(?:\.\d+)?)'
        matches = re.finditer(table_pattern, content, re.IGNORECASE)
        
        for match in matches:
            table_refs.append({
                "content": match.group(0),
                "table_number": match.group(1),
                "type": "table_reference",
                "page": getattr(item, 'page_number', 1),
                "position": match.start(),
                "bbox": getattr(item, 'bbox', None),
                "metadata": self._extract_item_metadata(item)
            })
        
        return table_refs

    def _extract_equation_number(self, content: str, position: int) -> str:
        """Extract equation number from nearby content."""
        # Look for equation numbers like (1), (2.1), etc.
        import re
        
        # Search backward and forward from position
        search_start = max(0, position - 100)
        search_end = min(len(content), position + 100)
        search_text = content[search_start:search_end]
        
        equation_pattern = r'\((\d+(?:\.\d+)?)\)'
        match = re.search(equation_pattern, search_text)
        
        if match:
            return match.group(1)
        
        return None

    def _detect_code_language(self, code_content: str) -> str:
        """Detect programming language from code content."""
        # Simple language detection based on keywords and syntax
        language_patterns = {
            'python': [r'\bdef\b', r'\bimport\b', r'\bclass\b', r':\s*$'],
            'javascript': [r'\bfunction\b', r'\bvar\b', r'\blet\b', r'=>'],
            'java': [r'\bpublic\b', r'\bclass\b', r'\bimport\b', r';'],
            'c++': [r'#include', r'\bint\b', r'\bclass\b', r'::'],
            'html': [r'<[^>]+>', r'<!DOCTYPE'],
            'sql': [r'\bSELECT\b', r'\bFROM\b', r'\bWHERE\b', r';'],
        }
        
        code_lower = code_content.lower()
        
        for lang, patterns in language_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, code_lower, re.MULTILINE))
            if score >= 2:  # At least 2 matches
                return lang
        
        return 'unknown'

    def _is_common_word(self, word: str) -> bool:
        """Check if word is a common English word (to avoid false positives in chemical formulas)."""
        common_words = {
            'can', 'old', 'car', 'man', 'new', 'get', 'see', 'use', 'way', 'say', 'her', 'his',
            'one', 'two', 'had', 'but', 'not', 'was', 'all', 'any', 'may', 'she', 'use', 'her'
        }
        return word.lower() in common_words

    def _extract_item_metadata(self, item) -> Dict[str, Any]:
        """Extract standard metadata for any item."""
        return {
            "content_type": getattr(item, 'content_type', 'unknown'),
            "page_number": getattr(item, 'page_number', 1),
            "confidence": getattr(item, 'confidence', 1.0),
            "level": getattr(item, 'level', 0),
            "reading_order": getattr(item, 'reading_order', 0)
        }

    
class DataStorer:
    """Responsible for storing data in databases."""

    def __init__(self, vector_store: VectorStore, data_parser: DataParser):
        self.vector_store = vector_store
        self.data_parser = data_parser
    
    async def store_chunks(self, chunks: List[Dict[str, Any]], document_id: int, fund_id: int) -> bool:
        """Store text chunks synchronously."""
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
            # Use synchronous storage method
            success = self._store_chunks_synchronously(chunks)
            logger.info(f"Successfully stored {len(chunks)} chunks" if success else "Failed to store chunks")
            return success

        except Exception as e:
            logger.error(f"Chunk storage failed: {e}")
            return False

    def _store_chunks_synchronously(self, chunks: List[Dict[str, Any]]) -> bool:
        """Store chunks synchronously using VectorStore's sync method."""
        try:
            # Use the VectorStore's synchronous store_chunks_sync method
            return self.vector_store.store_chunks_sync(chunks)
        except Exception as e:
            logger.error(f"Failed to store chunks synchronously: {e}")
            return False
    
    async def store_tables(self, tables_data: Dict[str, List[Dict[str, Any]]], fund_id: int) -> int:
        """Store table data synchronously."""
        from app.db.session import SessionLocal

        stored_count = 0
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
                        obj = self._create_transaction_object(table_type, row, fund_id)
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
        self.text_chunker = text_chunker or TextChunker()
        # Use None for db session to avoid hanging during initialization
        self.data_storer = data_storer or DataStorer(VectorStore(db=None), DataParser())
        self.converter = converter or self._initialize_converter()
    
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
        Process a PDF document and extract structured data.

        Args:
            file_path: Path to PDF file
            document_id: Document ID
            fund_id: Fund ID

        Returns:
            Processing result
        """
        return await self._process_document_async(file_path, document_id, fund_id)

    async def _process_document_async(self, file_path: str, document_id: int, fund_id: int) -> ProcessingResult:
        """
        Process a PDF document and extract structured data.

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
            logger.info(f"✓ Text chunking completed: {len(text_chunks)} chunks created")

            # Store data asynchronously
            logger.info("Starting vector storage phase...")
            chunks_stored = await self.data_storer.store_chunks(text_chunks, document_id, fund_id)
            logger.info(f"Vector storage completed: {chunks_stored}")
            tables_stored = await self.data_storer.store_tables(tables_data, fund_id)

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
        import asyncio

        # Create new event loop for synchronous context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._process_document_async(file_path, document_id, fund_id))
        finally:
            loop.close()

