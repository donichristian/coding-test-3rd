"""
Table parsing service using Docling's native capabilities.

This service provides specialized table extraction and classification
for financial documents using Docling's built-in table structure recognition.
"""
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TableData:
    """Data class for extracted table information."""
    table_type: str
    headers: List[str]
    rows: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    confidence: float


class TableParser:
    """
    Table parser using Docling's native table structure recognition.
    
    This class extracts tables from Docling documents and intelligently
    classifies them for business logic processing.
    """

    def __init__(self):
        """Initialize the table parser."""
        pass

    def extract_tables_with_docling(self, doc) -> Dict[str, Any]:
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
                
                for idx, table in enumerate(doc.tables):
                    logger.debug(f"Processing table {idx + 1}/{len(doc.tables)}")
                    
                    # Use Docling's built-in DataFrame export
                    if hasattr(table, 'export_to_dataframe'):
                        df = table.export_to_dataframe(doc=doc)
                        if not df.empty:
                            # Convert to records format
                            table_records = df.to_dict('records')
                            
                            # Log table details for debugging
                            logger.debug(f"Table {idx + 1} - Columns: {list(df.columns)}")
                            logger.debug(f"Table {idx + 1} - Rows: {len(table_records)}")
                            if table_records:
                                logger.debug(f"Table {idx + 1} - Sample row: {table_records[0]}")
                            
                            # Classify table using Docling's table type if available
                            table_type = getattr(table, 'table_type', 'unknown')
                            logger.debug(f"Table {idx + 1} - Docling type: {table_type}")
                            
                            # Apply business logic classification
                            classified_type = self.classify_table_for_business_logic(
                                table_records, table_type
                            )
                            
                            logger.info(f"Table {idx + 1} - Classified as: {classified_type}")
                            
                            if classified_type:
                                tables_data[classified_type].append({
                                    "headers": list(df.columns),
                                    "rows": table_records,
                                    "metadata": {
                                        "docling_table_type": table_type,
                                        "confidence": getattr(table, 'confidence', 1.0),
                                        "extraction_method": "docling_dataframe",
                                        "page_number": getattr(table, 'page_number', 1),
                                        "table_index": idx
                                    }
                                })
                            else:
                                logger.warning(f"Table {idx + 1} - Could not classify table type")
                        else:
                            logger.warning(f"Table {idx + 1} - Empty DataFrame")
                    else:
                        logger.warning(f"Table {idx + 1} - No export_to_dataframe method")
            else:
                logger.warning("No tables found in document")

            # Log final classification results
            for table_type, tables in tables_data.items():
                if table_type != "processing_method":
                    logger.info(f"Final classification - {table_type}: {len(tables)} tables")

            logger.info(f"Extracted tables: {sum(len(v) for k, v in tables_data.items() if k != 'processing_method')}")
            return tables_data

        except Exception as e:
            logger.error(f"Table extraction failed: {e}", exc_info=True)
            return {
                "capital_calls": [],
                "distributions": [],
                "adjustments": [],
                "processing_method": "error",
                "error": str(e)
            }

    def classify_table_for_business_logic(
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
            docling_type_lower = docling_table_type.lower()
            if 'capital' in docling_type_lower or 'call' in docling_type_lower:
                return "capital_calls"
            elif 'distribution' in docling_type_lower or 'return' in docling_type_lower:
                return "distributions"
            elif 'fee' in docling_type_lower or 'expense' in docling_type_lower or 'adjustment' in docling_type_lower:
                return "adjustments"

        # Enhanced content-based classification with comprehensive keywords
        headers = list(table_records[0].keys()) if table_records else []
        headers_text = " ".join(headers).lower()
        
        # Also check first few rows for better classification accuracy
        sample_rows = table_records[:3] if len(table_records) >= 3 else table_records
        rows_text = " ".join([str(value) for row in sample_rows for value in row.values()]).lower()
        combined_text = headers_text + " " + rows_text
        
        # Enhanced classification using comprehensive business keywords
        capital_keywords = ["call", "capital", "commitment", "called", "funding", "investment"]
        distribution_keywords = ["distribution", "return", "dividend", "income", "distributed", "payout", "recallable"]
        adjustment_keywords = ["fee", "expense", "adjustment", "recallable distribution", "capital call adjustment", "contribution adjustment", "reimbursement"]
        
        # Score each category based on keyword matches
        capital_score = sum(1 for keyword in capital_keywords if keyword in combined_text)
        distribution_score = sum(1 for keyword in distribution_keywords if keyword in combined_text)
        adjustment_score = sum(1 for keyword in adjustment_keywords if keyword in combined_text)
        
        # Log classification details for debugging
        logger.debug(f"Table classification - Headers: {headers_text}")
        logger.debug(f"Table classification - Sample data: {rows_text[:200]}...")
        logger.debug(f"Scores - Capital: {capital_score}, Distribution: {distribution_score}, Adjustment: {adjustment_score}")
        
        # Return the category with highest score (minimum threshold of 1)
        if capital_score >= distribution_score and capital_score >= adjustment_score and capital_score > 0:
            return "capital_calls"
        elif distribution_score >= adjustment_score and distribution_score > 0:
            return "distributions"
        elif adjustment_score > 0:
            return "adjustments"
            
        return None

    def parse_table_metadata(self, table) -> Dict[str, Any]:
        """
        Parse metadata from Docling table object.
        
        Args:
            table: Docling table object
            
        Returns:
            Dictionary with table metadata
        """
        return {
            "table_type": getattr(table, 'table_type', 'unknown'),
            "confidence": getattr(table, 'confidence', 1.0),
            "page_number": getattr(table, 'page_number', 1),
            "bbox": getattr(table, 'bbox', None),
            "extraction_method": "docling_native"
        }

    def extract_tables_from_document(
        self, 
        docling_doc, 
        include_metadata: bool = True
    ) -> List[TableData]:
        """
        Extract tables from Docling document with detailed metadata.
        
        Args:
            docling_doc: Docling document object
            include_metadata: Whether to include detailed metadata
            
        Returns:
            List of TableData objects
        """
        tables = []
        
        try:
            if hasattr(docling_doc, 'tables') and docling_doc.tables:
                for table in docling_doc.tables:
                    if hasattr(table, 'export_to_dataframe'):
                        df = table.export_to_dataframe(doc=docling_doc)
                        if not df.empty:
                            table_records = df.to_dict('records')
                            table_type = self.classify_table_for_business_logic(
                                table_records, getattr(table, 'table_type', 'unknown')
                            )
                            
                            if table_type:
                                metadata = self.parse_table_metadata(table) if include_metadata else {}
                                
                                tables.append(TableData(
                                    table_type=table_type,
                                    headers=list(df.columns),
                                    rows=table_records,
                                    metadata=metadata,
                                    confidence=metadata.get('confidence', 1.0)
                                ))
                                
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            
        return tables

    def get_table_statistics(self, tables_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Get statistics about extracted tables.
        
        Args:
            tables_data: Dictionary with table data from extract_tables_with_docling
            
        Returns:
            Dictionary with table statistics
        """
        stats = {
            "total_tables": 0,
            "capital_calls_tables": 0,
            "distributions_tables": 0,
            "adjustments_tables": 0
        }
        
        for table_type, tables in tables_data.items():
            if table_type != "processing_method":
                count = len(tables)
                stats["total_tables"] += count
                
                if table_type == "capital_calls":
                    stats["capital_calls_tables"] = count
                elif table_type == "distributions":
                    stats["distributions_tables"] = count
                elif table_type == "adjustments":
                    stats["adjustments_tables"] = count
                    
        return stats