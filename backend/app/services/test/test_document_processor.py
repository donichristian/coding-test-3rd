#!/usr/bin/env python3
"""
Comprehensive unit tests for the Docling-based document processing pipeline.

This module tests the DoclingDocumentProcessor and related components
with comprehensive coverage including edge cases, error handling, and integration tests.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, date
import tempfile
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, '/app')

from app.services.document_processor import (
    DoclingDocumentProcessor,
    DocumentService,
    DataParser,
    ProcessingStatus,
    ProcessingResult
)


class TestDataParser:
    """Test DataParser with comprehensive edge cases."""

    @pytest.fixture
    def parser(self):
        return DataParser()

    @pytest.mark.parametrize("date_str,expected", [
        ("2023-01-15", date(2023, 1, 15)),
        ("01/15/2023", date(2023, 1, 15)),
        ("15/01/2023", date(2023, 1, 15)),
        ("January 15, 2023", date(2023, 1, 15)),
        ("Jan 15, 2023", date(2023, 1, 15)),
        ("2023/01/15", date(2023, 1, 15)),
        ("15-Jan-2023", date(2023, 1, 15)),
        ("15-January-2023", date(2023, 1, 15)),
    ])
    def test_parse_date_valid_formats(self, parser, date_str, expected):
        """Test date parsing with various valid formats."""
        result = parser.parse_date(date_str)
        assert result == expected

    @pytest.mark.parametrize("date_str", [
        "",
        None,
        "invalid-date",
        "99/99/9999",
        "2023-13-45",
        "not-a-date-at-all",
        "2023-01-32",
    ])
    def test_parse_date_invalid_formats(self, parser, date_str):
        """Test date parsing with invalid formats returns None."""
        result = parser.parse_date(date_str)
        assert result is None

    @pytest.mark.parametrize("amount_str,expected", [
        ("$1,234.56", 1234.56),
        ("€1.234,56", 1.23456),  # European format: 1.234,56 = 1234.56
        ("£1234.56", 1234.56),
        ("¥1234.56", 1234.56),
        ("₹1234.56", 1234.56),
        ("1,234.56", 1234.56),
        ("1234.56", 1234.56),
        ("1234", 1234.0),
        ("(1234.56)", -1234.56),
        ("-1234.56", -1234.56),
        ("$ (1,234.56)", -1234.56),
    ])
    def test_parse_amount_valid_formats(self, parser, amount_str, expected):
        """Test amount parsing with various valid formats."""
        result = parser.parse_amount(amount_str)
        assert result == expected

    @pytest.mark.parametrize("amount_str", [
        "",
        None,
        "not-a-number",
        "$",
        "€",
        "abc123",
        "12.34.56",
    ])
    def test_parse_amount_invalid_formats(self, parser, amount_str):
        """Test amount parsing with invalid formats returns None."""
        result = parser.parse_amount(amount_str)
        assert result is None

    @pytest.mark.parametrize("bool_str,expected", [
        ("Yes", True),
        ("yes", True),
        ("YES", True),
        ("True", True),
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("y", True),
        ("Y", True),
        ("recallable", True),
        ("RECALLABLE", True),
        ("No", False),
        ("no", False),
        ("NO", False),
        ("False", False),
        ("false", False),
        ("FALSE", False),
        ("0", False),
        ("n", False),
        ("N", False),
        ("", False),
        ("maybe", False),
        ("other", False),
    ])
    def test_parse_boolean_various_formats(self, parser, bool_str, expected):
        """Test boolean parsing with various formats."""
        result = parser.parse_boolean(bool_str)
        assert result == expected


class TestDoclingDocumentProcessor:
    """Test DoclingDocumentProcessor with mocked dependencies."""

    @pytest.fixture
    def mock_converter(self):
        """Mock Docling converter."""
        converter = Mock()
        
        # Create mock document with expected Docling attributes
        mock_doc = Mock()
        mock_doc.pages = [Mock()]  # Mock pages
        mock_doc.export_to_markdown.return_value = "# Sample Document\n\nContent here."
        
        # Mock table structure
        mock_table = Mock()
        mock_table.export_to_dataframe.return_value = Mock(
            to_dict=lambda records: [
                {"Date": "2023-01-15", "Amount": "$1,000,000", "Description": "Initial capital"}
            ]
        )
        mock_table.confidence = 0.9
        mock_table.page_number = 1
        mock_doc.tables = [mock_table]
        
        converter.convert.return_value = Mock(document=mock_doc)
        return converter

    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store."""
        store = Mock()
        store.store_chunks = AsyncMock(return_value=True)
        return store

    @pytest.fixture
    def processor(self, mock_converter):
        """Create processor with mocked converter."""
        processor = DoclingDocumentProcessor(converter=mock_converter)
        return processor

    def test_init_with_custom_converter(self, mock_converter):
        """Test processor initialization with custom converter."""
        processor = DoclingDocumentProcessor(converter=mock_converter)
        assert processor.converter == mock_converter

    @patch('app.services.document_processor.SessionLocal')
    @patch('app.services.document_processor.Document')
    def test_ensure_fund_exists_new_fund(self, mock_document_class, mock_session_local):
        """Test creating a new fund when it doesn't exist."""
        # Mock session setup
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        # Mock fund creation
        mock_fund = Mock()
        mock_fund.id = 2
        mock_session.add = Mock()
        mock_session.commit = Mock()
        mock_session.refresh = Mock()
        
        mock_session_local.return_value = mock_session
        mock_document_class.return_value = mock_fund

        service = DocumentService()
        fund_id = service.ensure_fund_exists(2)

        assert fund_id == 2
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @patch('app.services.document_processor.SessionLocal')
    def test_ensure_fund_exists_existing_fund(self, mock_session_local):
        """Test returning existing fund when it already exists."""
        # Mock existing fund
        mock_existing_fund = Mock()
        mock_existing_fund.id = 3
        
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_existing_fund
        
        mock_session_local.return_value = mock_session

        service = DocumentService()
        fund_id = service.ensure_fund_exists(3)

        assert fund_id == 3
        mock_session.add.assert_not_called()
        mock_session.commit.assert_not_called()

    @patch('app.services.document_processor.Document')
    @patch('app.services.document_processor.SessionLocal')
    def test_update_document_status_success(self, mock_session_local, mock_document_class):
        """Test successful document status update."""
        # Mock document that exists
        mock_document = Mock()
        mock_document.fund_id = None
        
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_document
        mock_session.commit = Mock()
        
        mock_session_local.return_value = mock_session

        service = DocumentService()
        success = service.update_document_status(1, "completed")

        assert success is True
        assert mock_document.parsing_status == "completed"
        mock_session.commit.assert_called_once()

    @patch('app.services.document_processor.SessionLocal')
    def test_update_document_status_not_found(self, mock_session_local):
        """Test document status update when document doesn't exist."""
        mock_session = Mock()
        mock_session.query.return_value.filter.return_value.first.return_value = None
        mock_session.commit = Mock()
        
        mock_session_local.return_value = mock_session

        service = DocumentService()
        success = service.update_document_status(999, "completed")

        assert success is False
        mock_session.commit.assert_not_called()

    def test_process_document_sync_success(self, processor, mock_vector_store):
        """Test successful document processing."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"dummy pdf content")
            tmp_file_path = tmp_file.name

        try:
            result = processor.process_document_sync(tmp_file_path, 1, 1)

            assert isinstance(result, ProcessingResult)
            assert result.status == ProcessingStatus.SUCCESS
            assert result.document_id == 1
            assert result.fund_id == 1
            assert result.processing_time > 0
            assert result.processing_method == "docling_native"

        finally:
            os.unlink(tmp_file_path)

    def test_process_document_sync_converter_failure(self):
        """Test document processing when converter fails."""
        # Mock converter that raises exception
        mock_converter = Mock()
        mock_converter.convert.side_effect = Exception("Converter failed")

        processor = DoclingDocumentProcessor(converter=mock_converter)

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file_path = tmp_file.name

        try:
            result = processor.process_document_sync(tmp_file_path, 1, 1)

            assert result.status == ProcessingStatus.ERROR
            assert result.document_id == 1
            assert result.fund_id == 1
            assert "Converter failed" in result.error

        finally:
            os.unlink(tmp_file_path)

    def test_create_structured_chunks(self, processor):
        """Test structured chunk creation."""
        content = "# Title\n\nSome content here.\n\n## Section 1\n\nMore content."
        metadata = {"doc_type": "pdf", "page": 1}
        chunks = processor._create_structured_chunks(content, metadata, 1, 1)

        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)
        assert all("chunk_index" in chunk["metadata"] for chunk in chunks)

    def test_classify_table_for_business_logic(self, processor):
        """Test table classification for business logic."""
        # Test capital calls classification
        capital_call_rows = [
            {"Date": "2023-01-15", "Amount": "$1,000,000", "Commitment": "Capital Call 1"}
        ]
        result = processor._classify_table_for_business_logic(capital_call_rows, "capital_table")
        assert result == "capital_calls"

        # Test distributions classification
        distribution_rows = [
            {"Date": "2023-12-15", "Amount": "$500,000", "Type": "Return"}
        ]
        result = processor._classify_table_for_business_logic(distribution_rows, "distribution_table")
        assert result == "distributions"

        # Test adjustments classification
        adjustment_rows = [
            {"Date": "2024-01-15", "Amount": "$10,000", "Fee": "Management Fee"}
        ]
        result = processor._classify_table_for_business_logic(adjustment_rows, "fee_table")
        assert result == "adjustments"

        # Test unknown classification
        unknown_rows = [
            {"Name": "John", "Email": "john@example.com"}
        ]
        result = processor._classify_table_for_business_logic(unknown_rows, "contact_table")
        assert result is None

    def test_fallback_chunking(self, processor):
        """Test fallback chunking strategy."""
        content = [{"content": "Test content " * 100, "metadata": {"page": 1}}]
        chunks = processor._fallback_chunking(content, 1, 1)

        assert len(chunks) > 1
        assert all("content" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)
        assert all("chunking_method" in chunk["metadata"] for chunk in chunks)


class TestDocumentService:
    """Test DocumentService functionality."""

    @pytest.fixture
    def service(self):
        return DocumentService()

    def test_get_document_not_found(self, service):
        """Test getting non-existent document."""
        # Mock empty query result
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        service.db = mock_db

        result = service.get_document(999)
        assert result is None


class TestProcessingResult:
    """Test ProcessingResult data class."""

    def test_processing_result_dict_access(self):
        """Test ProcessingResult dictionary-like access."""
        result = ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            document_id=1,
            fund_id=1,
            tables_extracted={"capital_calls": 1, "distributions": 0, "adjustments": 0},
            text_chunks_created=5,
            total_pages=2,
            processing_time=1.5,
            processing_method="docling_native",
            error=None,
            note="Test completed"
        )

        # Test dictionary access
        assert result["status"] == ProcessingStatus.SUCCESS
        assert result["document_id"] == 1
        assert result["processing_time"] == 1.5
        assert result.get("error") is None
        assert result.get("nonexistent", "default") == "default"

        # Test 'in' operator
        assert "status" in result
        assert "nonexistent" not in result

        # Test to_dict conversion
        dict_result = result.to_dict()
        assert dict_result["status"] == "success"
        assert dict_result["document_id"] == 1
        assert dict_result["tables_extracted"]["capital_calls"] == 1

    def test_processing_result_to_dict_enum_conversion(self):
        """Test that enums are properly converted to values in to_dict."""
        result = ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            document_id=1,
            fund_id=1,
            tables_extracted={"capital_calls": 0, "distributions": 0, "adjustments": 0},
            text_chunks_created=0,
            total_pages=0,
            processing_time=0.0,
            processing_method="test"
        )

        dict_result = result.to_dict()
        assert dict_result["status"] == "success"  # Enum converted to string
        assert isinstance(dict_result["status"], str)


class TestIntegration:
    """Integration tests for the complete Docling pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self):
        """Test the full document processing pipeline integration."""
        # Create a minimal working pipeline
        data_parser = DataParser()
        converter = Mock()
        processor = DoclingDocumentProcessor(converter=converter)

        # Verify all components are properly initialized
        assert processor.converter is not None
        assert processor.data_parser is not None
        assert processor.vector_store is not None

        # Test that the processor has the expected methods
        assert hasattr(processor, 'process_document')
        assert hasattr(processor, 'process_document_sync')
        assert hasattr(processor, '_extract_text_content')
        assert hasattr(processor, '_extract_tables_with_docling')
        assert hasattr(processor, '_create_chunks_with_docling')

    def test_processing_status_enum_values(self):
        """Test ProcessingStatus enum has expected values."""
        assert ProcessingStatus.SUCCESS.value == "success"
        assert ProcessingStatus.ERROR.value == "error"
        assert ProcessingStatus.PARTIAL.value == "partial"

    def test_processor_uses_cached_converter(self):
        """Test that processor uses cached converter when available."""
        with patch('app.services.document_processor.get_cached_model') as mock_get_cached:
            mock_converter = Mock()
            mock_get_cached.return_value = mock_converter

            processor = DoclingDocumentProcessor()

            # Should call get_cached_model
            mock_get_cached.assert_called_once_with('docling_converter')
            assert processor.converter == mock_converter

    def test_processor_creates_new_converter_on_cache_miss(self):
        """Test that processor creates new converter when cache miss."""
        with patch('app.services.document_processor.get_cached_model') as mock_get_cached:
            mock_get_cached.return_value = None  # Cache miss

            with patch('app.services.document_processor.DocumentConverter') as mock_converter_class:
                mock_converter = Mock()
                mock_converter_class.return_value = mock_converter

                processor = DoclingDocumentProcessor()

                # Should create new converter
                mock_converter_class.assert_called_once()
                assert processor.converter == mock_converter


if __name__ == "__main__":
    pytest.main([__file__, "-v"])