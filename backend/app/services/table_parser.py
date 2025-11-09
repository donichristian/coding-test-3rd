"""
Table parsing service using docling (with pdfplumber fallback) for extracting structured data from PDF tables.

This service classifies and parses tables from fund performance documents,
specifically handling capital calls, distributions, and adjustments.
"""
from typing import Dict, List, Any, Optional
import pdfplumber
import re


class TableParser:
    """Parse and classify tables from PDF documents using docling with pdfplumber fallback"""

    def __init__(self):
        # Try to use docling first, fallback to pdfplumber
        self.use_docling = True
        self.docling_converter = None

        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions

            # Configure docling with proper table extraction options
            # Models are pre-loaded during Docker build, so we can enable table extraction
            pipeline_options = PdfPipelineOptions()
            # OCR models are pre-loaded, but we can disable OCR for table extraction (faster)
            pipeline_options.do_ocr = False  # Not needed for table extraction
            # Enable table structure extraction (models are pre-loaded)
            pipeline_options.do_table_structure = True  # Enable table structure extraction

            self.docling_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        except Exception as e:
            print(f"Docling initialization failed: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            self.use_docling = False

    def _is_gui_available(self):
        """Check if GUI libraries are available for docling"""
        try:
            # Try importing cv2 first to see if it's available
            import cv2
            # Try to create a simple image to test if GUI libraries work
            import numpy as np
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            # This will fail if GUI libraries are not available
            cv2.imencode('.png', img)
            print("GUI libraries available for docling")
            return True
        except ImportError as e:
            print(f"OpenCV not available: {e}")
            return False
        except Exception as e:
            print(f"GUI libraries not available for docling: {e}")
            return False

    def parse_tables(self, file_path: str = None, doc: Any = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parse tables from PDF and classify them by type using docling (with pdfplumber fallback)

        Args:
            file_path: Path to the PDF file (optional if doc is provided)
            doc: Pre-converted Docling document object (optional, avoids duplicate conversion)

        Returns:
            Dictionary with classified tables and processing method:
            {
                "capital_calls": [...],
                "distributions": [...],
                "adjustments": [...],
                "processing_method": "docling" or "pdfplumber"
            }
        """
        try:
            classified_tables = {
                "capital_calls": [],
                "distributions": [],
                "adjustments": []
            }

            # Try Docling first (models are pre-loaded during Docker build)
            if self.use_docling and self.docling_converter is not None:
                try:
                    # Use provided document if available, otherwise convert
                    if doc is None:
                        if file_path is None:
                            raise ValueError("Either file_path or doc must be provided")
                        result = self.docling_converter.convert(file_path)
                        doc = result.document
                    
                    # Extract tables from docling document
                    # Docling v2 stores tables in different ways - try multiple approaches
                    docling_tables = []
                    tables_found = []
                    
                    # Method 1: Check if document has direct tables attribute (fastest)
                    if hasattr(doc, 'tables') and doc.tables:
                        tables_found = doc.tables
                    
                    # Method 2: Check tables in pages
                    elif hasattr(doc, 'pages') and doc.pages:
                        for page in doc.pages:
                            if hasattr(page, 'tables') and page.tables:
                                tables_found.extend(page.tables)
                            # Also check for table elements in page content
                            if hasattr(page, 'content'):
                                for item in page.content:
                                    if hasattr(item, 'type') and item.type == 'table':
                                        tables_found.append(item)
                    
                    # Method 3: Check document content for table elements
                    if not tables_found and hasattr(doc, 'content'):
                        for item in doc.content:
                            if hasattr(item, 'type') and item.type == 'table':
                                tables_found.append(item)
                    
                    # Process found tables
                    if tables_found:
                        for table in tables_found:
                            table_data = self._extract_table_data_docling(table, doc)
                            if table_data and table_data.get("rows"):
                                table_type = self._classify_table(table_data)
                                if table_type and table_type in classified_tables:
                                    classified_tables[table_type].append(table_data)
                                    docling_tables.append(table_data)
                    
                    # If Docling successfully extracted tables, use them
                    if docling_tables:
                        print(f"✓ Docling extracted {len(docling_tables)} tables")
                        classified_tables["processing_method"] = "docling"
                        return classified_tables
                    else:
                        print(f"Docling found {len(tables_found)} tables but couldn't extract data, falling back to pdfplumber...")
                except Exception as e:
                    print(f"Docling table extraction failed: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Falling back to pdfplumber...")
            else:
                print("Docling not available, using pdfplumber...")

            # Fallback to pdfplumber if Docling failed or not available
            if file_path is None:
                print("Cannot use pdfplumber fallback: file_path not provided")
                return {"capital_calls": [], "distributions": [], "adjustments": [], "processing_method": "error"}
            
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        table_data = self._extract_table_data_pdfplumber(table)
                        if table_data and table_data.get("rows"):
                            table_type = self._classify_table(table_data)
                            if table_type and table_type in classified_tables:
                                classified_tables[table_type].append(table_data)

            classified_tables["processing_method"] = "pdfplumber"
            return classified_tables

        except Exception as e:
            print(f"Error parsing tables: {e}")
            import traceback
            traceback.print_exc()
            return {"capital_calls": [], "distributions": [], "adjustments": [], "processing_method": "error"}

    def _extract_table_data_docling(self, table, doc: Any = None) -> Dict[str, Any]:
        """
        Extract structured data from a Docling v2 table object

        Args:
            table: Docling v2 table object
            doc: Docling document object (required for export_to_dataframe to avoid deprecation warning)

        Returns:
            Dictionary with table data and metadata
        """
        try:
            # For Docling v2, use export_to_dataframe method (with doc parameter to avoid deprecation)
            if hasattr(table, 'export_to_dataframe'):
                try:
                    # Pass doc parameter to avoid deprecation warning
                    if doc is not None:
                        df = table.export_to_dataframe(doc=doc)
                    else:
                        df = table.export_to_dataframe()

                    if not df.empty:
                        headers = list(df.columns)
                        rows = df.to_dict('records')

                        return {
                            "headers": headers,
                            "rows": rows,
                            "metadata": {
                                "row_count": len(rows),
                                "column_count": len(headers),
                                "table_type": "unknown"
                            }
                        }
                except Exception as e:
                    # Silently fall through to direct data access
                    pass

            # Fallback: try to access table data directly
            if hasattr(table, 'data') and table.data:
                # Extract headers (first row typically)
                headers = []
                if table.data and len(table.data) > 0:
                    first_row = table.data[0]
                    for cell in first_row:
                        if hasattr(cell, 'text'):
                            headers.append(cell.text.strip())
                        else:
                            headers.append(str(cell).strip())

                # Clean headers (remove empty ones)
                headers = [h for h in headers if h]

                # Extract data rows
                rows = []
                for row in table.data[1:]:
                    row_data = {}
                    for i, cell in enumerate(row):
                        if i < len(headers):
                            col_name = headers[i]
                            cell_value = cell.text.strip() if hasattr(cell, 'text') else str(cell).strip()
                            if cell_value:  # Only add non-empty values
                                row_data[col_name] = cell_value
                    if row_data:  # Only add rows with data
                        rows.append(row_data)

                if headers and rows:
                    return {
                        "headers": headers,
                        "rows": rows,
                        "metadata": {
                            "row_count": len(rows),
                            "column_count": len(headers),
                            "table_type": "unknown"
                        }
                    }

            return None

        except Exception as e:
            return None

    def _extract_table_data_pdfplumber(self, table) -> Dict[str, Any]:
        """
        Extract structured data from a pdfplumber table

        Args:
            table: pdfplumber table (list of lists)

        Returns:
            Dictionary with table data and metadata
        """
        try:
            if not table or len(table) < 2:  # Need at least header + 1 data row
                return None

            # Extract headers (first row)
            headers = []
            for cell in table[0]:
                if cell:  # pdfplumber cells are strings or None
                    headers.append(str(cell).strip())
                else:
                    headers.append("")

            # Clean headers (remove empty ones and duplicates)
            headers = [h for h in headers if h]
            if not headers:
                return None

            # Extract data rows
            rows = []
            for row in table[1:]:
                row_data = {}
                for i, cell in enumerate(row):
                    if i < len(headers) and cell:  # Only add non-empty cells within header range
                        col_name = headers[i]
                        cell_value = str(cell).strip()
                        if cell_value:  # Only add non-empty values
                            row_data[col_name] = cell_value
                if row_data:  # Only add rows with data
                    rows.append(row_data)

            # Validate extracted data
            if not rows:
                return None

            return {
                "headers": headers,
                "rows": rows,
                "metadata": {
                    "row_count": len(rows),
                    "column_count": len(headers),
                    "table_type": "unknown"
                }
            }

        except Exception as e:
            print(f"Error extracting pdfplumber table data: {e}")
            return None

    def _classify_table(self, table_data: Dict[str, Any]) -> Optional[str]:
        """
        Classify table type based on headers and content

        Args:
            table_data: Table data with headers and rows

        Returns:
            Table classification: "capital_calls", "distributions", "adjustments", or None
        """
        if not table_data or not table_data.get("headers"):
            return None

        headers = [h.lower().strip() for h in table_data.get("headers", [])]
        headers_text = " ".join(headers)

        # Sample some row data for content-based classification
        sample_rows = table_data.get("rows", [])[:3]  # Check first 3 rows
        content_text = ""
        for row in sample_rows:
            content_text += " ".join(str(v).lower() for v in row.values()) + " "

        # Capital Calls indicators
        capital_call_keywords = [
            "capital call", "capital contribution", "contribution", "commitment",
            "called capital", "capital called", "investment amount", "called amount",
            "capital commitment", "commitment amount"
        ]

        # Distributions indicators
        distribution_keywords = [
            "distribution", "dividend", "return of capital", "return",
            "withdrawal", "payout", "proceeds", "distributed", "distribution amount"
        ]

        # Adjustments indicators
        adjustment_keywords = [
            "adjustment", "fee", "expense", "management fee", "performance fee",
            "carried interest", "carry", "incentive fee", "reimbursement",
            "recallable", "rebalance"
        ]

        # Check headers first
        for keyword in capital_call_keywords:
            if keyword in headers_text:
                return "capital_calls"

        for keyword in distribution_keywords:
            if keyword in headers_text:
                return "distributions"

        for keyword in adjustment_keywords:
            if keyword in headers_text:
                return "adjustments"

        # Check content if headers don't match
        for keyword in capital_call_keywords:
            if keyword in content_text:
                return "capital_calls"

        for keyword in distribution_keywords:
            if keyword in content_text:
                return "distributions"

        for keyword in adjustment_keywords:
            if keyword in content_text:
                return "adjustments"

        # Additional classification based on column patterns
        if self._is_capital_call_table(headers):
            return "capital_calls"
        elif self._is_distribution_table(headers):
            return "distributions"
        elif self._is_adjustment_table(headers):
            return "adjustments"

        # Check content for distribution keywords in the table data
        content_text_lower = content_text.lower()
        if any(keyword in content_text_lower for keyword in distribution_keywords):
            return "distributions"
        elif any(keyword in content_text_lower for keyword in adjustment_keywords):
            return "adjustments"

        # Special case: if table has "Recallable" column, it's likely distributions
        if any("recallable" in h.lower() for h in headers):
            return "distributions"

        # Check for specific patterns in the table that indicate distributions
        # Look for "Return of Capital", "Income", etc. in the Type column
        for row in table_data.get("rows", []):
            row_type = str(row.get("Type", "")).lower()
            if "return of capital" in row_type or "income" in row_type or "dividend" in row_type:
                return "distributions"

        # If we still can't classify, check if it looks like a distribution table by content
        # Tables with amounts and dates that don't match capital calls might be distributions
        if len(headers) >= 3 and any("date" in h.lower() for h in headers) and any("amount" in h.lower() for h in headers):
            # Check if amounts are mostly positive (typical for distributions)
            amounts = []
            for row in table_data.get("rows", []):
                for key, value in row.items():
                    if "amount" in key.lower():
                        try:
                            amount = self._extract_amount_from_string(str(value))
                            if amount is not None:
                                amounts.append(amount)
                        except:
                            pass

            if amounts and len([a for a in amounts if a > 0]) > len([a for a in amounts if a < 0]):
                return "distributions"

    def _extract_amount_from_string(self, amount_str: str) -> float:
        """Extract numeric amount from string like '$1,500,000' or '-$500,000'"""
        if not amount_str:
            return None

        # Remove currency symbols and commas
        cleaned = amount_str.replace('$', '').replace(',', '').strip()

        # Handle parentheses for negative amounts
        if cleaned.startswith('(') and cleaned.endswith(')'):
            cleaned = '-' + cleaned[1:-1]

        try:
            return float(cleaned)
        except ValueError:
            return None

        return None

    def _is_capital_call_table(self, headers: List[str]) -> bool:
        """Check if headers match capital call table pattern"""
        # Look for date + amount pattern
        has_date = any("date" in h or "call" in h for h in headers)
        has_amount = any("amount" in h or "capital" in h or "called" in h for h in headers)

        return has_date and has_amount

    def _is_distribution_table(self, headers: List[str]) -> bool:
        """Check if headers match distribution table pattern"""
        distribution_indicators = ["distribution", "return", "dividend", "payout", "distributed"]
        return any(indicator in " ".join(headers).lower() for indicator in distribution_indicators)

    def _is_adjustment_table(self, headers: List[str]) -> bool:
        """Check if headers match adjustment/fee table pattern"""
        fee_indicators = ["fee", "expense", "adjustment", "management", "performance", "carry", "rebalance"]
        return any(indicator in " ".join(headers).lower() for indicator in fee_indicators)