"""
Shared data parsing utilities for consistent data handling across services.

This module provides centralized parsing logic for dates, amounts, and other
common data types to eliminate duplication and ensure consistency.
"""
from datetime import datetime
from typing import Optional
import re


class DataParser:
    """Centralized data parsing utilities."""

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