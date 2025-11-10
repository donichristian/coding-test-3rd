"""
RAG (Retrieval Augmented Generation) Engine using LangChain

This module implements the RAG pipeline for fund performance analysis:
1. Query processing and intent classification
2. Vector similarity search for relevant context
3. LLM generation with retrieved context
4. Response formatting with citations
"""

from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings
from app.services.vector_store import VectorStore
from app.services.metrics_calculator import MetricsCalculator
from app.db.session import SessionLocal
import logging

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG engine for fund performance Q&A using LangChain"""

    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = self._initialize_llm()
        self.metrics_calculator = None  # Will be initialized when needed

    def _initialize_llm(self):
        """Initialize LLM based on configuration"""
        if settings.GEMINI_API_KEY:
            # Use Google Gemini as primary LLM
            return ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.1
            )
        elif settings.OPENAI_API_KEY:
            # Use OpenAI as fallback
            return ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model=settings.OPENAI_MODEL or "gpt-4-turbo-preview",
                temperature=0.1
            )
        else:
            # Fallback to a simple response system if no LLM is configured
            logger.warning("No LLM configuration found. Using fallback response system.")
            return None

    def classify_intent(self, query: str) -> str:
        """
        Classify query intent to determine processing strategy

        Returns:
            "calculation" - for DPI, IRR, PIC calculations
            "definition" - for explaining fund terms
            "retrieval" - for document-specific questions
            "general" - for general fund questions
        """
        query_lower = query.lower()

        # Calculation keywords
        calc_keywords = ["dpi", "irr", "pic", "calculate", "compute", "what is the"]
        if any(keyword in query_lower for keyword in calc_keywords):
            return "calculation"

        # Definition keywords
        def_keywords = ["what does", "explain", "meaning", "definition", "mean"]
        if any(keyword in query_lower for keyword in def_keywords):
            return "definition"

        # Retrieval keywords (specific to documents)
        retrieval_keywords = ["in this", "the document", "the report", "according to"]
        if any(keyword in query_lower for keyword in retrieval_keywords):
            return "retrieval"

        return "general"

    async def retrieve_context(self, query: str, fund_id: Optional[int] = None, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from vector store

        Args:
            query: User query
            fund_id: Optional fund ID filter
            k: Number of results to retrieve

        Returns:
            List of relevant document chunks with metadata
        """
        try:
            filter_metadata = {"fund_id": fund_id} if fund_id else None
            results = await self.vector_store.similarity_search(
                query=query,
                k=k,
                filter_metadata=filter_metadata
            )
            return results
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []

    def get_metrics_data(self, fund_id: int) -> Dict[str, Any]:
        """
        Get current fund metrics for context

        Args:
            fund_id: Fund ID

        Returns:
            Dictionary with current metrics
        """
        try:
            db = SessionLocal()
            if self.metrics_calculator is None:
                from app.services.metrics_calculator import MetricsCalculator
                self.metrics_calculator = MetricsCalculator(db)
            metrics = self.metrics_calculator.calculate_all_metrics(fund_id)
            db.close()
            return metrics
        except Exception as e:
            logger.error(f"Error getting metrics data: {e}")
            return {}

    def format_context(self, retrieved_docs: List[Dict[str, Any]], metrics_data: Dict[str, Any]) -> str:
        """
        Format retrieved context and metrics for LLM consumption

        Args:
            retrieved_docs: Retrieved document chunks
            metrics_data: Current fund metrics

        Returns:
            Formatted context string
        """
        context_parts = []

        # Add metrics data
        if metrics_data:
            context_parts.append("Current Fund Metrics:")
            for key, value in metrics_data.items():
                if isinstance(value, (int, float)):
                    context_parts.append(f"- {key}: {value:.4f}")
                else:
                    context_parts.append(f"- {key}: {value}")
            context_parts.append("")

        # Add retrieved document chunks
        if retrieved_docs:
            context_parts.append("Relevant Document Information:")
            for i, doc in enumerate(retrieved_docs, 1):
                content = doc.get("content", "").strip()
                if content:
                    context_parts.append(f"Document Chunk {i}:")
                    context_parts.append(content)
                    context_parts.append("")

        return "\n".join(context_parts)

    def generate_response(self, query: str, context: str, intent: str) -> Dict[str, Any]:
        """
        Generate response using LLM with retrieved context

        Args:
            query: User query
            context: Formatted context from retrieval
            intent: Classified query intent

        Returns:
            Dictionary with response and metadata
        """
        try:
            # Check if LLM is available
            if self.llm is None:
                # Fallback response system
                return self._generate_fallback_response(query, context, intent)

            # Create prompt based on intent
            if intent == "calculation":
                system_prompt = """You are a fund performance analysis expert. Use the provided metrics and document context to accurately answer calculation questions about fund performance (DPI, IRR, PIC, etc.).

Key guidelines:
- Use the provided metrics data as the primary source of truth
- Explain calculations clearly and show your work
- If metrics data is available, use it directly rather than recalculating
- Be precise with financial terminology
- Cite sources when referencing document information

Context:
{context}

Question: {query}

Answer:"""

            elif intent == "definition":
                system_prompt = """You are a fund performance analysis expert. Explain fund performance concepts and terminology clearly using the provided document context.

Key guidelines:
- Use simple, clear language while maintaining accuracy
- Reference specific document sections when relevant
- Provide examples when helpful
- Distinguish between different types of calculations or concepts

Context:
{context}

Question: {query}

Answer:"""

            else:  # retrieval or general
                system_prompt = """You are a fund performance analysis expert. Answer questions about fund performance using the provided context and metrics data.

Key guidelines:
- Base your answer on the provided context and metrics
- Be comprehensive but concise
- Use specific data points from the context
- Explain any complex concepts
- If the question cannot be answered from the context, say so clearly

Context:
{context}

Question: {query}

Answer:"""

            # Create LangChain prompt template
            prompt = ChatPromptTemplate.from_template(system_prompt)

            # Create chain
            chain = (
                {"context": RunnablePassthrough(), "query": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            # Generate response
            response = chain.invoke({"context": context, "query": query})

            return {
                "response": response,
                "intent": intent,
                "context_used": bool(context.strip()),
                "success": True
            }

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your question. Please try again.",
                "intent": intent,
                "error": str(e),
                "success": False
            }

    def _generate_fallback_response(self, query: str, context: str, intent: str) -> Dict[str, Any]:
        """
        Generate a fallback response when LLM is not available

        Args:
            query: User query
            context: Formatted context
            intent: Classified intent

        Returns:
            Dictionary with fallback response
        """
        query_lower = query.lower()

        # Simple keyword-based responses
        if intent == "calculation" or "dpi" in query_lower:
            if "dpi" in context.lower():
                return {
                    "response": "Based on the document context, I can see DPI (Distributions to Paid-In) information. DPI measures how much capital has been returned to investors relative to their original investment. A DPI of 1.0 means all invested capital has been returned.",
                    "intent": intent,
                    "context_used": True,
                    "success": True
                }
            else:
                return {
                    "response": "DPI (Distributions to Paid-In) measures how much capital has been returned to investors relative to their original investment. A DPI of 1.0 means all invested capital has been returned.",
                    "intent": intent,
                    "context_used": False,
                    "success": True
                }

        elif "irr" in query_lower:
            if "irr" in context.lower():
                return {
                    "response": "Based on the document context, I can see IRR (Internal Rate of Return) information. IRR is the annualized rate of return that makes the net present value of all cash flows equal to zero.",
                    "intent": intent,
                    "context_used": True,
                    "success": True
                }
            else:
                return {
                    "response": "IRR (Internal Rate of Return) is the annualized rate of return that makes the net present value of all cash flows equal to zero.",
                    "intent": intent,
                    "context_used": False,
                    "success": True
                }

        elif "pic" in query_lower or "paid-in capital" in query_lower:
            if "paid-in capital" in context.lower() or "pic" in context.lower():
                return {
                    "response": "Based on the document context, I can see Paid-In Capital (PIC) information. PIC represents the total capital contributed by investors to the fund.",
                    "intent": intent,
                    "context_used": True,
                    "success": True
                }
            else:
                return {
                    "response": "Paid-In Capital (PIC) is the total amount of capital that investors have contributed to the fund.",
                    "intent": intent,
                    "context_used": False,
                    "success": True
                }

        else:
            # Generic response with context
            if context.strip():
                return {
                    "response": f"Based on the document context, I can help you understand fund performance metrics. The document contains relevant information about fund performance. Try asking about specific metrics like DPI, IRR, or Paid-In Capital.",
                    "intent": intent,
                    "context_used": True,
                    "success": True
                }
            else:
                return {
                    "response": "I'm here to help you analyze fund performance data. Please upload a document first, then ask questions about metrics like DPI, IRR, or Paid-In Capital.",
                    "intent": intent,
                    "context_used": False,
                    "success": True
                }

    async def query(self, query: str, fund_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Main query method implementing the complete RAG pipeline

        Args:
            query: User question
            fund_id: Optional fund ID for filtering

        Returns:
            Dictionary with response and metadata
        """
        try:
            # Step 1: Classify intent
            intent = self.classify_intent(query)
            logger.info(f"Classified query intent: {intent}")

            # Step 2: Retrieve context
            retrieved_docs = await self.retrieve_context(query, fund_id)
            logger.info(f"Retrieved {len(retrieved_docs)} document chunks")

            # Step 3: Get metrics data if fund_id provided
            metrics_data = {}
            if fund_id:
                metrics_data = self.get_metrics_data(fund_id)
                logger.info(f"Retrieved metrics data for fund {fund_id}")

            # Step 4: Format context
            context = self.format_context(retrieved_docs, metrics_data)

            # Step 5: Generate response
            result = self.generate_response(query, context, intent)

            # Add metadata
            result.update({
                "query": query,
                "fund_id": fund_id,
                "retrieved_chunks": len(retrieved_docs),
                "has_metrics": bool(metrics_data)
            })

            return result

        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your question. Please try again.",
                "query": query,
                "fund_id": fund_id,
                "error": str(e),
                "success": False
            }