"""
Query engine service for RAG-based question answering
"""
from typing import Dict, Any, List, Optional
import time
from app.core.config import settings
from app.services.vector_store import VectorStore
from app.services.metrics_calculator import MetricsCalculator
from app.services.rag_engine import RAGEngine
from sqlalchemy.orm import Session


class QueryEngine:
    """RAG-based query engine for fund analysis"""

    def __init__(self, db: Session):
        self.db = db
        self.rag_engine = RAGEngine()
        self.metrics_calculator = MetricsCalculator(db)
    
    
    async def process_query(
        self,
        query: str,
        fund_id: Optional[int] = None,
        document_id: Optional[int] = None,
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query using RAG

        Args:
            query: User question
            fund_id: Optional fund ID for context
            conversation_history: Previous conversation messages

        Returns:
            Response with answer, sources, and metrics
        """
        start_time = time.time()

        # Use the new RAG engine for processing
        result = self.rag_engine.query(query, fund_id)

        # Add additional metadata
        result["processing_time"] = round(time.time() - start_time, 2)

        # Format sources for backward compatibility
        if "sources" not in result:
            result["sources"] = []

        # Get metrics if fund_id provided and not already included
        if fund_id and "metrics" not in result:
            try:
                metrics = self.metrics_calculator.calculate_all_metrics(fund_id)
                result["metrics"] = metrics
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                result["metrics"] = None

        return result
    
