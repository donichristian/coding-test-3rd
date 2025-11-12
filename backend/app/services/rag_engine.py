"""
RAG (Retrieval Augmented Generation) Engine using LangChain

This module implements the RAG pipeline for fund performance analysis:
1. Query processing and intent classification
2. Vector similarity search for relevant context
3. LLM generation with retrieved context
4. Response formatting with citations
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import List, Dict, Any, Optional, Protocol
import asyncio
import time
import logging

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

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Enumeration of possible query intents."""
    CALCULATION = "calculation"
    DEFINITION = "definition"
    RETRIEVAL = "retrieval"
    GENERAL = "general"
    
@dataclass
class QueryContext:
    """Data class for query context."""
    query: str
    fund_id: Optional[int]
    intent: QueryIntent
    retrieved_docs: List[Dict[str, Any]]
    metrics_data: Dict[str, Any]
    
@dataclass
class QueryResult:
   """Data class for query results."""
   response: str
   intent: QueryIntent
   context_used: bool
   success: bool
   error: Optional[str] = None
   metadata: Optional[Dict[str, Any]] = None
   
class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    
    def invoke(self, inputs: Dict[str, Any]) -> str:
        """Invoke the LLM with inputs."""
        ...
        
class IntentClassifier:
    """Class responsible for classifying query intent."""
    
    # Configuration constants
    CALCULATION_KEYWORDS = frozenset([
        "dpi", "irr", "pic", "calculate", "compute", "what is the"
    ])
    
    DEFINITION_KEYWORDS = frozenset([
        "what does", "explain", "meaning", "definition", "mean"
    ])
    
    RETRIEVAL_KEYWORDS = frozenset([
        "in this", "the document", "the report", "according to"
    ])
    
    def classify(self, query: str) -> QueryIntent:
        """
        Classify query intent based on keywords.
        
        Args:
            query: User query string
            
        Returns:
            Classified intent
        """
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in self.CALCULATION_KEYWORDS):
            return QueryIntent.CALCULATION
        elif any(keyword in query_lower for keyword in self.DEFINITION_KEYWORDS):
            return QueryIntent.DEFINITION
        elif any(keyword in query_lower for keyword in self.RETRIEVAL_KEYWORDS):
            return QueryIntent.RETRIEVAL
        
        return QueryIntent.GENERAL
    
class ContextRetriever:
    """Class responsible for retrieving context from vector store."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self._semaphore = asyncio.Semaphore(5) # Limit concurrent requests
        
    def retrieve_sync(
        self,
        query: str,
        fund_id: Optional[int] = None,
        k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Synchronous version of retrieve for API endpoints.

        Args:
            query: Search query
            fund_id: Optional fund filter
            k: Number of results
            similarity_threshold: Minimum similarity score

        Returns:
            List of relevant documents
        """
        try:
            # Use environment variables or config values if not provided
            import os
            k = k or int(os.getenv('TOP_K_RESULTS', settings.TOP_K_RESULTS))
            similarity_threshold = similarity_threshold or float(os.getenv('SIMILARITY_THRESHOLD', settings.SIMILARITY_THRESHOLD))

            filter_metadata = {"fund_id": fund_id} if fund_id else None
            results = self.vector_store.similarity_search_sync(
                query=query,
                k=k,
                filter_metadata=filter_metadata,
                similarity_threshold=similarity_threshold
            )

            logger.debug(f"Retrieved {len(results)} documents for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            # Return empty list instead of failing completely
            return []
            
class MetricsProvider:
    """Class responsible for providing metrics data."""
    
    def __init__(self, metrics_calculator: MetricsCalculator):
        self.metrics_calculator = metrics_calculator
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._cache_ttl = 300  # 5 minutes cache
    
    @lru_cache(maxsize=100)
    def _get_cached_metrics(self, fund_id: int) -> Dict[str, Any]:
        """Cached metrics retrieval."""
        try:
            db = SessionLocal()
            metrics = self.metrics_calculator.calculate_all_metrics(fund_id)
            db.close()
            return metrics
        except Exception as e:
            logger.error(f"Metrics calculation failed for fund {fund_id}: {e}")
            return {}
    
    def get_metrics(self, fund_id: int) -> Dict[str, Any]:
        """
        Get metrics data for a fund.
        
        Args:
            fund_id: Fund ID
            
        Returns:
            Metrics dictionary
        """
        if not fund_id:
            return {}
        
        return self._get_cached_metrics(fund_id)
    
class ContextFormatter:
    """Class responsible for formatting context for LLM consumption."""

    @staticmethod
    def format(
        retrieved_docs: List[Dict[str, Any]],
        metrics_data: Dict[str, Any]
    ) -> str:
        """
        Format retrieved context and metrics.

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
                if isinstance(value, (int, float)) and not (isinstance(value, float) and (value != value)):  # Check for NaN
                    context_parts.append(f"- {key}: {value:.4f}")
                else:
                    context_parts.append(f"- {key}: {value}")
            context_parts.append("")

        # Add retrieved document chunks with citations
        if retrieved_docs:
            context_parts.append("Relevant Document Information:")
            for i, doc in enumerate(retrieved_docs[:3], 1):  # Limit to top 3
                content = doc.get("content", "").strip()
                if content and len(content) < 1000:  # Reasonable chunk size
                    # Create citation without chunk details
                    metadata = doc.get("metadata", {})
                    citation = ContextFormatter._create_clean_citation(metadata)

                    context_parts.append(f"Source: {citation}")
                    context_parts.append(content)
                    context_parts.append("")

        return "\n".join(context_parts)

    @staticmethod
    def _create_clean_citation(metadata: Dict[str, Any]) -> str:
        """Create user-friendly citation without internal chunk details."""
        parts = []

        # Document name (always include if available, fallback to database lookup)
        doc_name = metadata.get("document_name")
        if not doc_name and metadata.get("document_id"):
            # Try to get document name from database
            try:
                from app.models.document import Document
                from app.db.session import SessionLocal
                db = SessionLocal()
                doc_record = db.query(Document).filter(Document.id == metadata["document_id"]).first()
                if doc_record:
                    doc_name = doc_record.file_name
                db.close()
            except Exception as e:
                logger.warning(f"Could not retrieve document name for ID {metadata.get('document_id')}: {e}")

        if doc_name:
            parts.append(doc_name)

        # Page number (most useful for users)
        if page_num := metadata.get("page_number"):
            parts.append(f"page {page_num}")

        # Section title if available and meaningful
        if section := metadata.get("section_title"):
            if len(section) < 50:  # Avoid overly long section names
                parts.append(f"section: {section}")

        return ", ".join(parts) if parts else "Unknown Source"
    
class ResponseGenerator:
    """Class responsible for generating responses using LLM."""
    
    # Prompt templates
    PROMPTS = {
        QueryIntent.CALCULATION: """You are a fund performance analysis expert. Use the provided metrics and document context to accurately answer calculation questions about fund performance (DPI, IRR, PIC, etc.).

Key guidelines:
- Use the provided metrics data as the primary source of truth
- Explain calculations clearly and show your work
- If metrics data is available, use it directly rather than recalculating
- Be precise with financial terminology
- Cite sources when referencing document information

Context:
{context}

Question: {query}

Answer:""",

        QueryIntent.DEFINITION: """You are a fund performance analysis expert. Explain fund performance concepts and terminology clearly using the provided document context.

Key guidelines:
- Use simple, clear language while maintaining accuracy
- Reference specific document sections when relevant
- Provide examples when helpful
- Distinguish between different types of calculations or concepts

Context:
{context}

Question: {query}

Answer:""",

        QueryIntent.RETRIEVAL: """You are a fund performance analysis expert. Answer questions about fund performance using the provided context and metrics data, focusing on document-specific information.

Key guidelines:
- Base your answer on the provided context and metrics
- Reference specific document sections
- Be comprehensive but concise
- Use specific data points from the context
- If the question cannot be answered from the context, say so clearly

Context:
{context}

Question: {query}

Answer:""",

        QueryIntent.GENERAL: """You are a fund performance analysis expert. Answer questions about fund performance using the provided context and metrics data.

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
    }
    
    def __init__(self, llm_provider: Optional[LLMProvider]):
        self.llm_provider = llm_provider
        self._circuit_breaker_failures = 0
        self._circuit_breaker_threshold = 3
    
    def generate(
        self, 
        query: str, 
        context: str, 
        intent: QueryIntent
    ) -> QueryResult:
        """
        Generate response using LLM.
        
        Args:
            query: User query
            context: Formatted context
            intent: Query intent
            
        Returns:
            Query result
        """
        try:
            if self.llm_provider is None or self._circuit_breaker_failures >= self._circuit_breaker_threshold:
                return self._generate_fallback_response(query, context, intent)
            
            # Create prompt
            prompt_template = self.PROMPTS.get(intent, self.PROMPTS[QueryIntent.GENERAL])
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
            # Create chain
            chain = (
                {"context": RunnablePassthrough(), "query": RunnablePassthrough()}
                | prompt
                | self.llm_provider
                | StrOutputParser()
            )
            
            # Generate response
            response = chain.invoke({"context": context, "query": query})
            
            # Reset circuit breaker on success
            self._circuit_breaker_failures = 0
            
            return QueryResult(
                response=response,
                intent=intent,
                context_used=bool(context.strip()),
                success=True,
                metadata={"llm_used": True}
            )
            
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            self._circuit_breaker_failures += 1
            return self._generate_fallback_response(query, context, intent)
    
    def _generate_fallback_response(
        self, 
        query: str, 
        context: str, 
        intent: QueryIntent
    ) -> QueryResult:
        """Generate fallback response when LLM is unavailable."""
        
        # Template-based fallback responses
        templates = {
            QueryIntent.CALCULATION: {
                "template": "For calculation questions, I need access to current metrics data. Please ensure the system is properly configured with an LLM provider.",
                "context_check": lambda c: "dpi" in c.lower() or "irr" in c.lower() or "pic" in c.lower()
            },
            QueryIntent.DEFINITION: {
                "template": "For definitions, I can explain fund performance concepts. {context_info} Try asking about specific terms like DPI, IRR, or PIC.",
                "context_check": lambda c: bool(c.strip())
            },
            QueryIntent.RETRIEVAL: {
                "template": "For document-specific questions, I need to access the uploaded documents. {context_info}",
                "context_check": lambda c: bool(c.strip())
            },
            QueryIntent.GENERAL: {
                "template": "I'm here to help with fund performance analysis. {context_info} Please upload a document and ask questions about metrics like DPI, IRR, or PIC.",
                "context_check": lambda c: bool(c.strip())
            }
        }
        
        config = templates.get(intent, templates[QueryIntent.GENERAL])
        context_info = "Based on the available context, " if config["context_check"](context) else ""
        
        response = config["template"].format(context_info=context_info)
        
        return QueryResult(
            response=response,
            intent=intent,
            context_used=config["context_check"](context),
            success=True,
            metadata={"fallback": True}
        )
        
class LLMFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create_llm() -> Optional[LLMProvider]:
        """Create LLM provider based on configuration."""
        try:
            if settings.GEMINI_API_KEY:
                return ChatGoogleGenerativeAI(
                    model=settings.GEMINI_MODEL or "gemini-pro",
                    google_api_key=settings.GEMINI_API_KEY,
                    temperature=0.1
                )
            elif settings.OPENAI_API_KEY:
                return ChatOpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    model=settings.OPENAI_MODEL or "gpt-4-turbo-preview",
                    temperature=0.1
                )
            elif hasattr(settings, 'OLLAMA_BASE_URL') and settings.OLLAMA_BASE_URL:
                return Ollama(
                    base_url=settings.OLLAMA_BASE_URL,
                    model=settings.OLLAMA_MODEL or "llama3.2"
                )
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}")
        
        logger.warning("No LLM provider configured")
        return None

class RAGEngine:
    """RAG engine for fund performance Q&A using LangChain"""

    # Shared singleton instances to prevent model reloading
    _shared_vector_store: Optional[VectorStore] = None
    _shared_metrics_provider: Optional[MetricsProvider] = None
    _shared_llm_provider: Optional[LLMProvider] = None

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        metrics_calculator: Optional[MetricsCalculator] = None,
        llm_provider: Optional[LLMProvider] = None
    ):
        # Use shared instances to prevent model reloading on each query
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            if RAGEngine._shared_vector_store is None:
                RAGEngine._shared_vector_store = VectorStore()
            self.vector_store = RAGEngine._shared_vector_store

        if metrics_calculator is not None:
            self.metrics_provider = MetricsProvider(metrics_calculator)
        else:
            if RAGEngine._shared_metrics_provider is None:
                RAGEngine._shared_metrics_provider = MetricsProvider(MetricsCalculator(SessionLocal()))
            self.metrics_provider = RAGEngine._shared_metrics_provider

        if llm_provider is not None:
            self.llm_provider = llm_provider
        else:
            if RAGEngine._shared_llm_provider is None:
                RAGEngine._shared_llm_provider = LLMFactory.create_llm()
            self.llm_provider = RAGEngine._shared_llm_provider

        # Component initialization (create fresh instances for thread safety)
        self.intent_classifier = IntentClassifier()
        self.context_retriever = ContextRetriever(self.vector_store)
        self.context_formatter = ContextFormatter()
        self.response_generator = ResponseGenerator(self.llm_provider)
    
    async def query(
        self,
        query: str,
        fund_id: Optional[int] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Main query method implementing the complete RAG pipeline.

        Args:
            query: User question
            fund_id: Optional fund ID for filtering
            conversation_history: Previous conversation messages for context

        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()

        try:
            # Step 1: Classify intent
            intent = self.intent_classifier.classify(query)
            logger.info(f"Classified query intent: {intent.value}")

            # Step 2: Enhance query with conversation context if available
            enhanced_query = self._enhance_query_with_history(query, conversation_history)
            if enhanced_query != query:
                logger.info("Enhanced query with conversation context")

            # Step 3: Retrieve context
            import os
            top_k = int(os.getenv('TOP_K_RESULTS', settings.TOP_K_RESULTS))
            similarity_thresh = float(os.getenv('SIMILARITY_THRESHOLD', settings.SIMILARITY_THRESHOLD))
            retrieved_docs = self.context_retriever.retrieve_sync(
                enhanced_query,
                fund_id,
                k=top_k,
                similarity_threshold=similarity_thresh
            )
            logger.info(f"Retrieved {len(retrieved_docs)} document chunks")

            # Step 4: Get metrics data
            metrics_data = self.metrics_provider.get_metrics(fund_id) if fund_id else {}
            if metrics_data:
                logger.info(f"Retrieved metrics data for fund {fund_id}")

            # Step 5: Format context
            context = self.context_formatter.format(retrieved_docs, metrics_data)

            # Step 6: Generate response
            result = self.response_generator.generate(query, context, intent)

            # Step 7: Format sources with enhanced citation information
            enhanced_sources = self._format_sources_with_citations(retrieved_docs)

            # Add timing and metadata
            processing_time = time.time() - start_time

            return {
                "response": result.response,
                "intent": result.intent.value,
                "context_used": result.context_used,
                "success": result.success,
                "query": query,
                "fund_id": fund_id,
                "retrieved_chunks": len(retrieved_docs),
                "has_metrics": bool(metrics_data),
                "processing_time": round(processing_time, 2),
                "sources": enhanced_sources,
                "metadata": result.metadata or {}
            }

        except Exception as e:
            logger.error(f"RAG query failed: {e}", exc_info=True)
            return {
                "response": "I apologize, but I encountered an error while processing your question. Please try again.",
                "query": query,
                "fund_id": fund_id,
                "error": str(e),
                "success": False,
                "processing_time": round(time.time() - start_time, 2),
                "sources": []
            }

    def _enhance_query_with_history(
        self,
        current_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Enhance the current query with relevant conversation context.

        Args:
            current_query: The current user query
            conversation_history: List of previous messages

        Returns:
            Enhanced query string
        """
        if not conversation_history or len(conversation_history) < 2:
            return current_query

        # Extract recent context (last 4 exchanges to avoid token bloat)
        recent_history = conversation_history[-4:]

        # Build context string from relevant previous interactions
        context_parts = []
        for msg in recent_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                # Include user questions for context
                context_parts.append(f"Previous question: {content}")
            elif role == "assistant" and len(content) < 200:  # Avoid overly long responses
                # Include brief assistant responses for context
                context_parts.append(f"Previous answer: {content[:200]}...")

        if context_parts:
            conversation_context = " ".join(context_parts)
            # Combine current query with conversation context
            enhanced_query = f"Context: {conversation_context}\n\nCurrent question: {current_query}"
            return enhanced_query

        return current_query

    def _format_sources_with_citations(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format retrieved documents with user-friendly citation information.

        Args:
            retrieved_docs: Raw retrieved documents

        Returns:
            List of formatted source documents with clean citations
        """
        formatted_sources = []

        for doc in retrieved_docs[:5]:  # Limit to top 5 sources
            metadata = doc.get("metadata", {})

            # Get document name from metadata, fallback to database lookup if needed
            document_name = metadata.get("document_name")
            if not document_name and metadata.get("document_id"):
                # Try to get document name from database
                try:
                    from app.models.document import Document
                    from app.db.session import SessionLocal
                    db = SessionLocal()
                    doc_record = db.query(Document).filter(Document.id == metadata["document_id"]).first()
                    if doc_record:
                        document_name = doc_record.file_name
                    db.close()
                except Exception as e:
                    logger.warning(f"Could not retrieve document name for ID {metadata.get('document_id')}: {e}")

            # Create clean citation for user display
            citation_parts = []
            if document_name:
                citation_parts.append(document_name)
            if page_num := metadata.get("page_number"):
                citation_parts.append(f"page {page_num}")

            citation_text = ", ".join(citation_parts) if citation_parts else "Unknown Source"

            formatted_source = {
                "content": doc.get("content", "").strip(),
                "metadata": metadata,
                "score": doc.get("score", 0.0),
                "document_name": document_name,
                "page_number": metadata.get("page_number"),
                # Remove chunk_index from user-facing data
                "confidence_score": doc.get("score", 0.0),
                "citation_text": citation_text
            }

            formatted_sources.append(formatted_source)

        return formatted_sources