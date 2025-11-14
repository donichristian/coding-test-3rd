"""
RAG (Retrieval Augmented Generation) Engine using Docling and LangChain

This module implements the RAG pipeline for fund performance analysis using Docling's
native LangChain integration for document processing and chunking.
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
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker

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
        "dpi", "irr", "pic", "calculate", "compute", "what is the",
        "percentage", "percent", "ratio", "how many", "what percentage"
    ])
    
    DEFINITION_KEYWORDS = frozenset([
        "what does", "explain", "meaning", "definition", "mean"
    ])
    
    RETRIEVAL_KEYWORDS = frozenset([
        "in this", "the document", "the report", "according to",
        "show me", "list", "display", "what are", "give me",
        "all", "find", "search", "look for"
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

        # Check calculation keywords first
        if any(keyword in query_lower for keyword in self.CALCULATION_KEYWORDS):
            logger.info(f"Classified as CALCULATION: '{query}' contains calculation keywords")
            return QueryIntent.CALCULATION

        # Check definition keywords
        elif any(keyword in query_lower for keyword in self.DEFINITION_KEYWORDS):
            logger.info(f"Classified as DEFINITION: '{query}' contains definition keywords")
            return QueryIntent.DEFINITION

        # Check retrieval keywords
        elif any(keyword in query_lower for keyword in self.RETRIEVAL_KEYWORDS):
            logger.info(f"Classified as RETRIEVAL: '{query}' contains retrieval keywords")
            return QueryIntent.RETRIEVAL

        # Default to general
        logger.info(f"Classified as GENERAL: '{query}' doesn't match specific keywords")
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
            k: Number of results (uses config.TOP_K_RESULTS if None)

        Returns:
            List of relevant documents
        """
        try:
            # Use config value if k not specified
            if k is None:
                k = settings.TOP_K_RESULTS

            filter_metadata = {"fund_id": fund_id} if fund_id else None
            threshold = similarity_threshold if similarity_threshold is not None else settings.SIMILARITY_THRESHOLD
            results = self.vector_store.similarity_search_sync(
                query=query,
                k=k,
                filter_metadata=filter_metadata,
                similarity_threshold=threshold
            )
            logger.info(f"Retrieved {len(results)} document chunks using k={k}")
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

        # Add retrieved document chunks with clean citations
        if retrieved_docs:
            # Sort by score to ensure most relevant chunks come first
            sorted_docs = sorted(retrieved_docs, key=lambda x: x.get("score", 0), reverse=True)
            context_parts.append("Relevant Document Information:")
            for i, doc in enumerate(sorted_docs[:settings.TOP_K_RESULTS], 1):  # Use config value
                content = doc.get("content", "").strip()
                # Remove length limit to ensure table data is included
                if content:  # Only check that content exists
                    # Create clean citation without chunk details
                    metadata = doc.get("metadata", {})
                    citation = ContextFormatter._create_clean_citation(metadata)

                    context_parts.append(f"Source {i}: {citation}")
                    context_parts.append("Content:")
                    # Truncate very long content but keep table data
                    if len(content) > 3000:
                        content = content[:3000] + "...[content truncated]"
                    context_parts.append(content)
                    context_parts.append("")

                    # Debug: Check if this chunk contains capital calls
                    if "capital calls" in content.lower():
                        logger.info(f"✓ Source {i} contains capital calls data")
                    else:
                        logger.debug(f"Source {i} does not contain capital calls")

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

        QueryIntent.RETRIEVAL: """You are a fund performance analysis expert. Answer questions about fund performance using ONLY the provided context and metrics data. Focus on extracting and presenting specific information from documents.

CRITICAL INSTRUCTIONS:
- You MUST extract and present specific data, tables, and details from the provided context
- If the context contains tables with financial data (capital calls, distributions, adjustments), you MUST include that data in your answer
- Present information in a clear, structured format
- Do NOT say information is not available if it's actually provided in the context
- Reference specific document sections and data points
- Be comprehensive about the requested information

Context:
{context}

Question: {query}

Answer:""",

        QueryIntent.GENERAL: """You are a fund performance analysis expert. Answer questions about fund performance using ONLY the provided context and metrics data.

CRITICAL INSTRUCTIONS:
- You MUST use the information provided in the "Context" section above
- If the context contains tables, data, or specific information relevant to the question, you MUST include that information in your answer
- Do NOT say you don't have information if it's actually provided in the context
- Extract and present specific data points, tables, and details from the context
- Be comprehensive but concise
- If the question cannot be answered from the provided context, say so clearly

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


class DoclingDocumentProcessor:
    """
    Document processor using Docling's LangChain integration.
    
    This class leverages DoclingLoader for seamless document processing
    and chunking, eliminating the need for custom document processing logic.
    """

    def __init__(self):
        """Initialize the Docling document processor."""
        self.chunker = None
        self._initialize_chunker()

    def _initialize_chunker(self):
        """Initialize the Docling chunker with default settings."""
        try:
            # Use Docling's HybridChunker with sentence-transformers model
            # Docling expects the model name, not just the provider type
            self.chunker = HybridChunker(tokenizer="sentence-transformers/all-MiniLM-L6-v2")
            logger.info("✓ Docling HybridChunker initialized with sentence-transformers")
        except Exception as e:
            logger.warning(f"Failed to initialize Docling chunker: {e}")
            self.chunker = None

    def process_documents_for_rag(
        self, 
        file_paths: List[str], 
        export_type: ExportType = ExportType.DOC_CHUNKS
    ) -> List[Dict[str, Any]]:
        """
        Process documents using DoclingLoader for RAG applications.
        
        Args:
            file_paths: List of file paths to process
            export_type: Export type (DOC_CHUNKS or MARKDOWN)
            
        Returns:
            List of processed document chunks ready for LangChain
        """
        try:
            # Create DoclingLoader with hybrid chunking
            loader = DoclingLoader(
                file_path=file_paths,
                export_type=export_type,
                chunker=self.chunker
            )

            # Load documents using Docling's native processing
            docs = loader.load()
            
            # Convert to format expected by RAG pipeline
            processed_docs = []
            for doc in docs:
                processed_docs.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "type": "langchain_document"
                })
            
            logger.info(f"Processed {len(processed_docs)} documents using DoclingLoader")
            return processed_docs

        except Exception as e:
            logger.error(f"DoclingLoader processing failed: {e}")
            return []


class RAGEngine:
    """RAG engine for fund performance Q&A using Docling and LangChain"""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        metrics_calculator: Optional[MetricsCalculator] = None,
        llm_provider: Optional[LLMProvider] = None,
        docling_processor: Optional[DoclingDocumentProcessor] = None
    ):
        # Create fresh instances for each RAG engine to avoid shared state issues in Celery
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            # Create new vector store instance for each query to avoid shared state issues
            self.vector_store = VectorStore()

        if metrics_calculator is not None:
            self.metrics_provider = MetricsProvider(metrics_calculator)
        else:
            # Create fresh metrics provider for each query
            self.metrics_provider = MetricsProvider(MetricsCalculator(SessionLocal()))

        if llm_provider is not None:
            self.llm_provider = llm_provider
        else:
            # Create fresh LLM provider for each query
            self.llm_provider = LLMFactory.create_llm()

        if docling_processor is not None:
            self.docling_processor = docling_processor
        else:
            # Create fresh Docling processor for each query
            self.docling_processor = DoclingDocumentProcessor()

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
            logger.info(f"Searching for: '{enhanced_query}' with fund_id={fund_id}")

            # Debug: Check if fund_id is None and how it affects filtering
            if fund_id is None:
                logger.warning("fund_id is None - searching across all documents without fund filtering")

            # Try with lower similarity threshold for more lenient matching
            retrieved_docs = self.context_retriever.retrieve_sync(enhanced_query, fund_id, similarity_threshold=0.1)
            logger.info(f"Retrieved {len(retrieved_docs)} document chunks with threshold 0.1")

            # Debug: Check similarity scores for failed retrievals
            if len(retrieved_docs) == 0:
                logger.warning(f"No chunks retrieved for query: '{enhanced_query}' with threshold 0.1")
                # Try with very low threshold as last resort
                try:
                    fallback_docs = self.context_retriever.retrieve_sync(enhanced_query, fund_id, similarity_threshold=0.0)
                    logger.info(f"Fallback retrieval with threshold 0.0: {len(fallback_docs)} chunks")
                    if len(fallback_docs) > 0:
                        logger.info("Using fallback results with very low similarity threshold")
                        retrieved_docs = fallback_docs
                except Exception as fallback_error:
                    logger.warning(f"Fallback retrieval failed: {fallback_error}")

            # Check if vector store is empty
            if len(retrieved_docs) == 0:
                logger.warning("No documents found in vector store - user needs to upload documents first")
                # Return helpful message instead of proceeding with empty context
                return {
                    "response": "I don't have any documents to analyze yet. Please upload a fund performance report first, then I can help you analyze metrics like DPI, IRR, and answer questions about capital calls, distributions, and adjustments.",
                    "intent": intent.value,
                    "context_used": False,
                    "success": True,
                    "query": query,
                    "fund_id": fund_id,
                    "retrieved_chunks": 0,
                    "has_metrics": False,
                    "processing_time": round(time.time() - start_time, 2),
                    "sources": [],
                    "metadata": {"no_documents": True}
                }

            # Debug: Log retrieved chunks and their content preview
            for i, doc in enumerate(retrieved_docs[:5]):  # Check more chunks
                content_preview = doc.get("content", "")[:300]  # Longer preview
                score = doc.get("score", 0)
                metadata = doc.get("metadata", {})
                fund_id_meta = metadata.get("fund_id")
                logger.info(f"Chunk {i+1}: score={score:.4f}, fund_id={fund_id_meta}, content='{content_preview}...'")
                if "capital calls" in doc.get("content", "").lower():
                    logger.info(f"✓ Chunk {i+1} CONTAINS CAPITAL CALLS!")
                elif "capital" in doc.get("content", "").lower():
                    logger.info(f"~ Chunk {i+1} contains 'capital' but not 'calls'")

            # Step 4: Get metrics data
            metrics_data = self.metrics_provider.get_metrics(fund_id) if fund_id else {}
            if metrics_data:
                logger.info(f"Retrieved metrics data for fund {fund_id}")

            # Step 5: Format context
            context = self.context_formatter.format(retrieved_docs, metrics_data)
            logger.info(f"Formatted context length: {len(context)}")
            logger.info(f"Context preview: {context[:1000]}...")
            if "capital calls" in context.lower():
                logger.info("✓ Context contains 'capital calls'")
            else:
                logger.warning("✗ Context does NOT contain 'capital calls'")

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

    def process_documents_for_rag(
        self, 
        file_paths: List[str], 
        export_type: ExportType = ExportType.DOC_CHUNKS
    ) -> List[Dict[str, Any]]:
        """
        Process documents using DoclingLoader for RAG ingestion.
        
        Args:
            file_paths: List of file paths to process
            export_type: Export type (DOC_CHUNKS or MARKDOWN)
            
        Returns:
            List of processed document chunks
        """
        return self.docling_processor.process_documents_for_rag(file_paths, export_type)

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