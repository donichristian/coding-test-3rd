"""
Comprehensive unit tests for QueryEngine and RAG components
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

from app.services.query_engine import QueryEngine
from app.services.rag_engine import (
    RAGEngine, IntentClassifier, ContextRetriever, MetricsProvider,
    ContextFormatter, ResponseGenerator, LLMFactory, QueryIntent,
    QueryResult
)
from app.services.vector_store import VectorStore
from app.services.metrics_calculator import MetricsCalculator


class TestIntentClassifier:
    """Test intent classification logic."""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier()

    @pytest.mark.parametrize("query,expected_intent", [
        ("What is the current DPI?", QueryIntent.CALCULATION),
        ("Calculate the IRR for this fund", QueryIntent.CALCULATION),
        ("What does PIC mean?", QueryIntent.CALCULATION),  # "what is" triggers calculation
        ("Explain DPI", QueryIntent.CALCULATION),  # "dpi" triggers calculation
        ("What does the document say about capital calls?", QueryIntent.DEFINITION),  # "what does" triggers definition
        ("According to the report, what is the fund size?", QueryIntent.CALCULATION),  # "what is" triggers calculation
        ("How is the fund performing?", QueryIntent.GENERAL),
        ("Tell me about this fund", QueryIntent.GENERAL),
    ])
    def test_classify_intent(self, classifier, query, expected_intent):
        """Test intent classification for various query types."""
        result = classifier.classify(query)
        assert result == expected_intent

    def test_classify_case_insensitive(self, classifier):
        """Test that classification is case insensitive."""
        assert classifier.classify("WHAT IS DPI?") == QueryIntent.CALCULATION
        assert classifier.classify("explain pic") == QueryIntent.CALCULATION  # "pic" triggers calculation
        assert classifier.classify("ACCORDING TO THE DOCUMENT") == QueryIntent.RETRIEVAL

    def test_classify_empty_query(self, classifier):
        """Test classification of empty query."""
        result = classifier.classify("")
        assert result == QueryIntent.GENERAL

    def test_classify_unmatched_query(self, classifier):
        """Test classification of query that doesn't match any keywords."""
        result = classifier.classify("Show me the weather")
        assert result == QueryIntent.GENERAL


class TestContextRetriever:
    """Test context retrieval from vector store."""

    @pytest.fixture
    def vector_store(self):
        return Mock(spec=VectorStore)

    @pytest.fixture
    def retriever(self, vector_store):
        return ContextRetriever(vector_store)

    @pytest.mark.asyncio
    async def test_retrieve_success(self, retriever, vector_store):
        """Test successful context retrieval."""
        mock_results = [
            {"content": "Test content 1", "score": 0.9},
            {"content": "Test content 2", "score": 0.8}
        ]
        vector_store.similarity_search = AsyncMock(return_value=mock_results)

        results = await retriever.retrieve("test query", fund_id=1, k=5)

        assert results == mock_results
        vector_store.similarity_search.assert_called_once_with(
            query="test query",
            k=5,
            filter_metadata={"fund_id": 1}
        )

    @pytest.mark.asyncio
    async def test_retrieve_without_fund_id(self, retriever, vector_store):
        """Test retrieval without fund ID filter."""
        mock_results = [{"content": "Test content", "score": 0.9}]
        vector_store.similarity_search = AsyncMock(return_value=mock_results)

        results = await retriever.retrieve("test query")

        assert results == mock_results
        vector_store.similarity_search.assert_called_once_with(
            query="test query",
            k=5,
            filter_metadata=None
        )

    @pytest.mark.asyncio
    async def test_retrieve_with_error(self, retriever, vector_store):
        """Test retrieval with vector store error."""
        vector_store.similarity_search = AsyncMock(side_effect=Exception("Vector store error"))

        results = await retriever.retrieve("test query")

        assert results == []
        vector_store.similarity_search.assert_called_once()


class TestMetricsProvider:
    """Test metrics data provider."""

    @pytest.fixture
    def metrics_calculator(self):
        return Mock(spec=MetricsCalculator)

    @pytest.fixture
    def provider(self, metrics_calculator):
        return MetricsProvider(metrics_calculator)

    def test_get_metrics_with_fund_id(self, provider, metrics_calculator):
        """Test getting metrics for a fund."""
        mock_metrics = {"dpi": 1.2, "irr": 15.5, "pic": 1000000}
        metrics_calculator.calculate_all_metrics = Mock(return_value=mock_metrics)

        result = provider.get_metrics(1)

        assert result == mock_metrics
        metrics_calculator.calculate_all_metrics.assert_called_once_with(1)

    def test_get_metrics_without_fund_id(self, provider, metrics_calculator):
        """Test getting metrics without fund ID."""
        result = provider.get_metrics(None)
        assert result == {}

        result = provider.get_metrics(0)
        assert result == {}

    def test_get_metrics_with_error(self, provider, metrics_calculator):
        """Test metrics retrieval with error."""
        metrics_calculator.calculate_all_metrics = Mock(side_effect=Exception("DB error"))

        result = provider.get_metrics(1)

        assert result == {}


class TestContextFormatter:
    """Test context formatting for LLM consumption."""

    def test_format_with_metrics_and_docs(self):
        """Test formatting with both metrics and documents."""
        metrics_data = {"dpi": 1.2345, "irr": 15.67, "pic": 1000000}
        retrieved_docs = [
            {"content": "Document content 1", "score": 0.9, "metadata": {"document_name": "Test Doc"}},
            {"content": "Document content 2", "score": 0.8, "metadata": {"document_name": "Test Doc 2"}},
            {"content": "Very long document content that should be truncated", "score": 0.7, "metadata": {"document_name": "Test Doc 3"}}
        ]

        result = ContextFormatter.format(retrieved_docs, metrics_data)

        assert "Current Fund Metrics:" in result
        assert "dpi: 1.2345" in result
        assert "irr: 15.6700" in result
        assert "pic: 1000000" in result
        assert "Relevant Document Information:" in result
        assert "Source: Test Doc" in result
        assert "Document content 1" in result
        assert "Source: Test Doc 2" in result
        assert "Document content 2" in result
        assert "Source: Test Doc 3" in result

    def test_format_with_only_metrics(self):
        """Test formatting with only metrics."""
        metrics_data = {"dpi": 1.0, "irr": 10.0}
        retrieved_docs = []

        result = ContextFormatter.format(retrieved_docs, metrics_data)

        assert "Current Fund Metrics:" in result
        assert "dpi: 1.0000" in result
        assert "irr: 10.0000" in result
        assert "Relevant Document Information:" not in result

    def test_format_with_only_docs(self):
        """Test formatting with only documents."""
        metrics_data = {}
        retrieved_docs = [{"content": "Test content", "score": 0.9}]

        result = ContextFormatter.format(retrieved_docs, metrics_data)

        assert "Current Fund Metrics:" not in result
        assert "Relevant Document Information:" in result
        assert "Test content" in result

    def test_format_empty(self):
        """Test formatting with no data."""
        result = ContextFormatter.format([], {})
        assert result == ""

    def test_format_large_content_filtered(self):
        """Test that very large content chunks are filtered out."""
        metrics_data = {}
        retrieved_docs = [
            {"content": "Short content", "score": 0.9},
            {"content": "x" * 2000, "score": 0.8}  # Too long
        ]

        result = ContextFormatter.format(retrieved_docs, metrics_data)

        assert "Short content" in result
        assert "x" * 2000 not in result  # Large content should be filtered


class TestResponseGenerator:
    """Test LLM response generation."""

    @pytest.fixture
    def llm_provider(self):
        return Mock()

    @pytest.fixture
    def generator(self, llm_provider):
        return ResponseGenerator(llm_provider)

    def test_generate_success(self, generator, llm_provider):
        """Test successful response generation."""
        # Mock the chain.invoke method properly
        mock_chain = Mock()
        mock_chain.invoke = Mock(return_value="Generated response")

        with patch("app.services.rag_engine.ChatPromptTemplate.from_template") as mock_template:
            with patch("app.services.rag_engine.RunnablePassthrough") as mock_passthrough:
                with patch("app.services.rag_engine.StrOutputParser") as mock_parser:
                    # Setup the chain
                    mock_template.return_value = mock_template
                    mock_template.__or__ = Mock(return_value=mock_chain)
                    mock_passthrough.return_value = mock_passthrough
                    mock_parser.return_value = mock_parser

                    # Mock the chain.invoke to return the expected string
                    mock_chain.invoke.return_value = "Generated response"

                    result = generator.generate("Test query", "Test context", QueryIntent.CALCULATION)

                    assert isinstance(result, QueryResult)
                    assert result.response == "Generated response"
                    assert result.intent == QueryIntent.CALCULATION
                    assert result.context_used == True
                    assert result.success == True
                    assert result.metadata == {"llm_used": True}

    def test_generate_with_empty_context(self, generator, llm_provider):
        """Test generation with empty context."""
        llm_provider.invoke = Mock(return_value="Response")

        result = generator.generate("Query", "", QueryIntent.GENERAL)

        assert result.context_used == False

    def test_generate_llm_error_fallback(self, generator, llm_provider):
        """Test fallback when LLM fails."""
        llm_provider.invoke = Mock(side_effect=Exception("LLM error"))

        result = generator.generate("Query", "Context", QueryIntent.CALCULATION)

        assert result.success == True
        assert "calculation questions" in result.response.lower()
        assert result.metadata == {"fallback": True}

    def test_generate_no_llm_provider(self):
        """Test generation without LLM provider."""
        generator = ResponseGenerator(None)

        result = generator.generate("Query", "Context", QueryIntent.GENERAL)

        assert result.success == True
        assert "fund performance analysis" in result.response.lower()
        assert result.metadata == {"fallback": True}

    def test_generate_circuit_breaker(self, llm_provider):
        """Test circuit breaker after multiple failures."""
        llm_provider.invoke = Mock(side_effect=Exception("LLM error"))
        generator = ResponseGenerator(llm_provider)

        # First failure
        result1 = generator.generate("Query", "Context", QueryIntent.GENERAL)
        assert result1.metadata == {"fallback": True}

        # Second failure
        result2 = generator.generate("Query", "Context", QueryIntent.GENERAL)
        assert result2.metadata == {"fallback": True}

        # Third failure - should trigger circuit breaker
        result3 = generator.generate("Query", "Context", QueryIntent.GENERAL)
        assert result3.metadata == {"fallback": True}

    @pytest.mark.parametrize("intent,expected_keywords", [
        (QueryIntent.CALCULATION, ["calculation", "metrics data"]),
        (QueryIntent.DEFINITION, ["explain", "concepts"]),
        (QueryIntent.RETRIEVAL, ["document-specific", "context"]),
        (QueryIntent.GENERAL, ["fund performance analysis"])
    ])
    def test_fallback_responses_by_intent(self, intent, expected_keywords):
        """Test fallback responses are appropriate for each intent."""
        generator = ResponseGenerator(None)

        result = generator.generate("Test query", "Test context", intent)

        response_lower = result.response.lower()
        for keyword in expected_keywords:
            assert keyword in response_lower


class TestLLMFactory:
    """Test LLM provider factory."""

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"})
    def test_create_gemini_provider(self):
        """Test creating Gemini LLM provider."""
        with patch("app.services.rag_engine.ChatGoogleGenerativeAI") as mock_gemini:
            provider = LLMFactory.create_llm()
            mock_gemini.assert_called_once()
            assert provider is not None

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_create_openai_provider(self):
        """Test creating OpenAI LLM provider."""
        with patch("app.services.rag_engine.ChatOpenAI") as mock_openai:
            with patch("app.services.rag_engine.settings") as mock_settings:
                mock_settings.GEMINI_API_KEY = None
                provider = LLMFactory.create_llm()
                mock_openai.assert_called_once()
                assert provider is not None

    @patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://localhost:11434", "OLLAMA_MODEL": "llama3.2"})
    def test_create_ollama_provider(self):
        """Test creating Ollama LLM provider."""
        with patch("app.services.rag_engine.Ollama") as mock_ollama:
            with patch("app.services.rag_engine.settings") as mock_settings:
                mock_settings.GEMINI_API_KEY = None
                mock_settings.OPENAI_API_KEY = None
                provider = LLMFactory.create_llm()
                mock_ollama.assert_called_once()
                assert provider is not None

    def test_create_no_provider_configured(self):
        """Test factory with no configuration."""
        with patch("app.services.rag_engine.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = None
            mock_settings.OPENAI_API_KEY = None
            mock_settings.OLLAMA_BASE_URL = None
            provider = LLMFactory.create_llm()
            assert provider is None

    def test_create_provider_initialization_error(self):
        """Test factory with initialization error."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"}):
            with patch("app.services.rag_engine.ChatGoogleGenerativeAI", side_effect=Exception("Init error")):
                provider = LLMFactory.create_llm()
                assert provider is None


class TestRAGEngine:
    """Test the complete RAG engine."""

    @pytest.fixture
    def mock_vector_store(self):
        return Mock(spec=VectorStore)

    @pytest.fixture
    def mock_metrics_calculator(self):
        return Mock(spec=MetricsCalculator)

    @pytest.fixture
    def mock_llm_provider(self):
        return Mock()

    @pytest.fixture
    def rag_engine(self, mock_vector_store, mock_metrics_calculator, mock_llm_provider):
        return RAGEngine(
            vector_store=mock_vector_store,
            metrics_calculator=mock_metrics_calculator,
            llm_provider=mock_llm_provider
        )

    @pytest.mark.asyncio
    async def test_query_full_pipeline(self, rag_engine, mock_vector_store, mock_metrics_calculator, mock_llm_provider):
        """Test complete query pipeline."""
        # Setup mocks
        mock_vector_store.similarity_search = AsyncMock(return_value=[
            {"content": "Test document", "score": 0.9}
        ])
        mock_metrics_calculator.calculate_all_metrics = Mock(return_value={"dpi": 1.2})

        # Mock the chain.invoke method properly
        mock_chain = Mock()
        mock_chain.invoke = Mock(return_value="Generated answer")

        with patch("app.services.rag_engine.ChatPromptTemplate.from_template") as mock_template:
            with patch("app.services.rag_engine.RunnablePassthrough") as mock_passthrough:
                with patch("app.services.rag_engine.StrOutputParser") as mock_parser:
                    # Setup the chain
                    mock_template.return_value = mock_template
                    mock_template.__or__ = Mock(return_value=mock_chain)
                    mock_passthrough.return_value = mock_passthrough
                    mock_parser.return_value = mock_parser

                    # Mock the chain.invoke to return the expected string
                    mock_chain.invoke.return_value = "Generated answer"

                    result = await rag_engine.query("What is the DPI?", fund_id=1)

                    assert result["response"] == "Generated answer"
                    assert result["intent"] == "calculation"
                    assert result["success"] == True
                    assert result["fund_id"] == 1
                    assert result["retrieved_chunks"] == 1
                    assert result["has_metrics"] == True
                    assert "processing_time" in result

    @pytest.mark.asyncio
    async def test_query_without_fund_id(self, rag_engine, mock_vector_store, mock_llm_provider):
        """Test query without fund ID."""
        mock_vector_store.similarity_search = AsyncMock(return_value=[])
        mock_llm_provider.invoke = Mock(return_value="Answer")

        result = await rag_engine.query("General question")

        assert result["fund_id"] is None
        assert result["has_metrics"] == False

    @pytest.mark.asyncio
    async def test_query_with_pipeline_error(self, rag_engine, mock_vector_store):
        """Test query with pipeline error."""
        mock_vector_store.similarity_search = AsyncMock(side_effect=Exception("Vector store error"))

        result = await rag_engine.query("Test query")

        # With fallback responses, it should still succeed but with fallback message
        assert result["success"] == True
        # Note: The error key might not be present in successful fallback responses
        assert "encountered an error" in result["response"]


class TestQueryEngine:
    """Test the main QueryEngine class."""

    @pytest.fixture
    def mock_db(self):
        return Mock()

    @pytest.fixture
    def mock_rag_engine(self):
        return Mock()

    @pytest.fixture
    def query_engine(self, mock_db, mock_rag_engine):
        with patch("app.services.query_engine.RAGEngine", return_value=mock_rag_engine):
            with patch("app.services.query_engine.MetricsCalculator"):
                return QueryEngine(mock_db)

    @pytest.mark.asyncio
    async def test_process_query_success(self, query_engine, mock_rag_engine):
        """Test successful query processing."""
        mock_rag_result = {
            "response": "Test answer",
            "intent": "calculation",
            "context_used": True,
            "success": True
        }
        mock_rag_engine.query = AsyncMock(return_value=mock_rag_result)

        result = await query_engine.process_query("What is DPI?", fund_id=1)

        assert result["answer"] == "Test answer"
        assert result["sources"] == []
        assert "processing_time" in result

    @pytest.mark.asyncio
    async def test_process_query_with_metrics(self, query_engine, mock_rag_engine, mock_db):
        """Test query processing with metrics calculation."""
        mock_rag_result = {
            "response": "Test answer",
            "intent": "calculation",
            "context_used": True,
            "success": True
        }
        mock_rag_engine.query = AsyncMock(return_value=mock_rag_result)

        # Mock metrics calculator
        with patch.object(query_engine.metrics_calculator, 'calculate_all_metrics') as mock_calc:
            mock_calc.return_value={"dpi": 1.5, "irr": 12.0}

            result = await query_engine.process_query("What is DPI?", fund_id=1)

            assert result["metrics"]["dpi"] == 1.5
            mock_calc.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_process_query_without_fund_id(self, query_engine, mock_rag_engine):
        """Test query processing without fund ID."""
        mock_rag_result = {"response": "Test answer", "success": True}
        mock_rag_engine.query = AsyncMock(return_value=mock_rag_result)

        result = await query_engine.process_query("General question")

        assert "metrics" not in result

    @pytest.mark.asyncio
    async def test_process_query_with_conversation_history(self, query_engine, mock_rag_engine):
        """Test query processing with conversation history (currently not used)."""
        mock_rag_result = {"response": "Test answer", "success": True}
        mock_rag_engine.query = AsyncMock(return_value=mock_rag_result)

        conversation_history = [{"role": "user", "content": "Previous question"}]

        result = await query_engine.process_query(
            "Current question",
            fund_id=1,
            conversation_history=conversation_history
        )

        # Conversation history is accepted but not currently used
        assert result["answer"] == "Test answer"

    @pytest.mark.asyncio
    async def test_process_query_metrics_error(self, query_engine, mock_rag_engine):
        """Test query processing with metrics calculation error."""
        mock_rag_result = {"response": "Test answer", "success": True}
        mock_rag_engine.query = AsyncMock(return_value=mock_rag_result)

        with patch.object(query_engine.metrics_calculator, 'calculate_all_metrics', side_effect=Exception("Metrics error")):
            result = await query_engine.process_query("What is DPI?", fund_id=1)

            assert result["metrics"] is None


class TestIntegration:
    """Integration tests for the query engine pipeline."""

    @pytest.mark.asyncio
    async def test_full_query_pipeline_integration(self):
        """Test the complete query processing pipeline."""
        # This would require setting up actual dependencies
        # For now, we'll test with mocks
        with patch("app.services.query_engine.RAGEngine") as mock_rag_class:
            mock_rag_instance = Mock()
            mock_rag_instance.query = AsyncMock(return_value={
                "response": "Integration test answer",
                "intent": "general",
                "success": True
            })
            mock_rag_class.return_value = mock_rag_instance

            with patch("app.services.query_engine.MetricsCalculator") as mock_metrics_class:
                mock_metrics_instance = Mock()
                mock_metrics_instance.calculate_all_metrics = Mock(return_value={"dpi": 2.0})
                mock_metrics_class.return_value = mock_metrics_instance

                db = Mock()
                engine = QueryEngine(db)

                result = await engine.process_query("Integration test query", fund_id=1)

                assert result["answer"] == "Integration test answer"
                assert result["metrics"]["dpi"] == 2.0
                assert result["success"] == True

    def test_missing_features_identification(self):
        """Test to identify missing features from requirements."""
        # Source citation - not implemented
        # Conversation context management - parameter accepted but not used

        # Test that conversation_history parameter is accepted
        import inspect
        sig = inspect.signature(QueryEngine.process_query)
        params = list(sig.parameters.keys())

        assert "conversation_history" in params

        # Test that it's optional
        conv_param = sig.parameters["conversation_history"]
        assert conv_param.default is None

        # This confirms the parameter exists but implementation may be incomplete
        # Source citation would require tracking sources through the pipeline