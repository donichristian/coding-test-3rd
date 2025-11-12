"""
Vector store service using pgvector (PostgreSQL extension)

Provides vector storage and similarity search capabilities for document embeddings.
Supports multiple embedding providers with automatic fallback and batch processing.

TODO: Implement vector storage using pgvector
- Create embeddings table in PostgreSQL
- Store document chunks with vector embeddings
- Implement similarity search using pgvector operators
- Handle metadata filtering
"""
import json
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import numpy as np
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


class VectorStore:
    """
    pgvector-based vector store for document embeddings.

    Supports multiple embedding providers with automatic fallback:
    1. sentence-transformers (local, preferred)
    2. Gemini API (cloud, free tier)
    3. OpenAI API (cloud, paid)

    Features:
    - Batch processing for efficiency
    - Proper error handling and logging
    - Database transaction management
    - Metadata filtering for search
    """


    def __init__(self, db: Optional[Session] = None):
        """
        Initialize vector store with database session and embedding model.

        Args:
            db: SQLAlchemy session (creates new one if None)
        """
        self.db = db
        self.embedding_configs = settings.EMBEDDING_CONFIGS
        self.embedding_provider = self._initialize_embeddings()
        self.embedding_dimensions = self.embedding_configs.get(
            self.embedding_provider, {"dimensions": 384}
        )["dimensions"]

        # Initialize embedding model attributes
        self._embedding_model = None
        self._openai_client = None

        # Initialize the embedding model immediately if sentence-transformers is selected
        if self.embedding_provider == "sentence-transformers":
            self._init_sentence_transformers()

        # Only ensure extension if we have a database session
        if self.db is not None:
            self._ensure_extension()

    @asynccontextmanager
    async def _get_db_session(self):
        """Get database session with proper cleanup."""
        session = self.db if self.db else SessionLocal()
        try:
            yield session
        finally:
            if not self.db:  # Only close if we created the session
                await session.close()
    
    def _initialize_embeddings(self) -> Optional[str]:
        """
        Initialize embedding model with fallback hierarchy.

        Priority order:
        1. sentence-transformers (local, no API costs)
        2. Gemini API (free tier available)
        3. OpenAI API (paid, most reliable)

        Returns:
            Provider name or None if no provider available
        """
        providers = [
            ("sentence-transformers", self._init_sentence_transformers),
            ("gemini", self._init_gemini),
            ("openai", self._init_openai)
        ]

        for provider_name, init_func in providers:
            try:
                if init_func():
                    config = self.embedding_configs[provider_name]
                    logger.info(f"✓ Using {provider_name} embeddings: {config['description']}")
                    return provider_name
            except Exception as e:
                logger.warning(f"Failed to initialize {provider_name}: {e}")
                continue

        logger.error("No embedding providers available")
        return None

    def _init_sentence_transformers(self) -> bool:
        """Initialize sentence-transformers model."""
        try:
            # First try to get from global cache
            try:
                from app.core.model_cache import get_cached_model, is_model_cached
                if is_model_cached('sentence_transformers'):
                    cached_model = get_cached_model('sentence_transformers')
                    logger.info("Using cached sentence-transformers model from global cache")
                    self._embedding_model = cached_model
                    return True
            except ImportError:
                pass  # Global cache not available

            # Fallback: initialize model normally
            from sentence_transformers import SentenceTransformer
            logger.info("Initializing sentence-transformers model...")

            # Check if model is already cached from Docker build
            import os
            model_path = os.path.expanduser("~/.cache/torch/sentence_transformers/all-MiniLM-L6-v2")
            if os.path.exists(model_path):
                logger.info("✓ Using pre-cached sentence-transformers model from Docker build")
            else:
                logger.info("Downloading sentence-transformers model...")

            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Cache the model globally for reuse
            try:
                from app.core.model_cache import set_cached_model
                set_cached_model('sentence_transformers', self._embedding_model)
                logger.info("✓ Sentence-transformers model cached globally")
            except ImportError:
                logger.debug("Global cache not available in current context")
            
            logger.info("✓ Sentence-transformers model initialized successfully")
            return True
        except ImportError:
            logger.warning("sentence_transformers package not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize sentence-transformers: {e}")
            return False

    def _init_gemini(self) -> bool:
        """Initialize Gemini API client."""
        if not settings.GEMINI_API_KEY or not settings.GEMINI_API_KEY.strip():
            return False

        try:
            import google.genai
            # Test the API key by creating client
            google.genai.Client(api_key=settings.GEMINI_API_KEY)
            return True
        except (ImportError, Exception):
            return False

    def _init_openai(self) -> bool:
        """Initialize OpenAI API client."""
        if not settings.OPENAI_API_KEY or not settings.OPENAI_API_KEY.strip():
            return False

        try:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            return True
        except ImportError:
            return False
    
    def _ensure_extension(self) -> None:
        """
        Ensure pgvector extension is enabled and table exists.

        Note: Table creation is handled in init_db.py during database initialization.
        This method verifies the setup is correct.
        """
        if self.db is None:
            logger.warning("No database session available for extension verification")
            return

        try:
            # Enable pgvector extension (idempotent operation)
            self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            # Verify table exists
            result = self.db.execute(text("SELECT 1 FROM document_embeddings LIMIT 1;"))
            result.fetchone()  # Consume the result
            logger.info("✓ Table document_embeddings exists and is accessible")

            self.db.commit()

        except Exception as e:
            logger.warning(f"pgvector extension or table verification failed: {e}")
            logger.info("Note: This may be expected during initial setup. Run database initialization if needed.")
            if self.db:
                self.db.rollback()
    
    async def add_document(self, content: str, metadata: Dict[str, Any]) -> bool:
        """
        Add a single document to the vector store.

        TODO: Implement this method
        - Generate embedding for content
        - Insert into document_embeddings table
        - Store metadata as JSONB

        Args:
            content: Document text content
            metadata: Document metadata (document_id, fund_id, etc.)

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If required metadata is missing
            SQLAlchemyError: If database operation fails
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        required_fields = ["document_id", "fund_id"]
        missing_fields = [field for field in required_fields if field not in metadata]
        if missing_fields:
            raise ValueError(f"Missing required metadata fields: {missing_fields}")

        async with self._get_db_session() as db:
            try:
                # Generate embedding
                embedding = await self._get_embedding(content)
                embedding_list = embedding.tolist()

                # Prepare metadata as JSON
                metadata_json = json.dumps(metadata)

                # Insert into database
                insert_sql = text("""
                    INSERT INTO document_embeddings (document_id, fund_id, content, embedding, metadata)
                    VALUES (:document_id, :fund_id, :content, :embedding, :metadata)
                """)

                db.execute(insert_sql, {
                    "document_id": metadata["document_id"],
                    "fund_id": metadata["fund_id"],
                    "content": content,
                    "embedding": f"[{','.join(map(str, embedding_list))}]",
                    "metadata": metadata_json
                })

                await db.commit()
                logger.debug(f"Added document {metadata['document_id']} to vector store")
                return True

            except SQLAlchemyError as e:
                await db.rollback()
                logger.error(f"Database error adding document: {e}")
                raise
            except Exception as e:
                await db.rollback()
                logger.error(f"Error adding document: {e}")
                raise

    async def store_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Store multiple text chunks in the vector database with batch processing.

        Args:
            chunks: List of chunks with content and metadata

        Returns:
            True if all chunks stored successfully, False otherwise
        """
        if not chunks:
            logger.info("No chunks to store")
            return True

        logger.info(f"Storing {len(chunks)} text chunks in vector database")

        # Process chunks in batches to avoid overwhelming the API
        batch_size = 10
        total_stored = 0

        async with self._get_db_session() as db:
            try:
                for batch_idx, i in enumerate(range(0, len(chunks), batch_size)):
                    batch = chunks[i:i + batch_size]
                    batch_num = batch_idx + 1

                    try:
                        # Generate embeddings for batch
                        embeddings = []
                        for chunk in batch:
                            embedding = await self._get_embedding(chunk["content"])
                            embeddings.append(embedding)

                        # Store batch in single transaction
                        for j, chunk in enumerate(batch):
                            embedding = embeddings[j]
                            embedding_list = embedding.tolist()

                            # Prepare metadata
                            metadata = chunk.get("metadata", {})
                            document_id = metadata.get("document_id")
                            fund_id = metadata.get("fund_id")

                            if not document_id or not fund_id:
                                logger.warning(f"Skipping chunk with missing metadata: doc_id={document_id}, fund_id={fund_id}")
                                continue

                            # Insert into database
                            insert_sql = text("""
                                INSERT INTO document_embeddings (document_id, fund_id, content, embedding, metadata)
                                VALUES (:document_id, :fund_id, :content, :embedding, :metadata)
                            """)

                            metadata_json = json.dumps(metadata)

                            db.execute(insert_sql, {
                                "document_id": document_id,
                                "fund_id": fund_id,
                                "content": chunk["content"],
                                "embedding": f"[{','.join(map(str, embedding_list))}]",
                                "metadata": metadata_json
                            })

                        # Commit batch
                        await db.commit()
                        total_stored += len(batch)
                        logger.info(f"Stored batch {batch_num}: {len(batch)} chunks")

                    except Exception as batch_error:
                        logger.error(f"Error storing batch {batch_num}: {batch_error}")
                        await db.rollback()
                        # Continue with next batch instead of failing completely

                logger.info(f"Successfully stored {total_stored}/{len(chunks)} chunks in vector database")
                return total_stored > 0

            except Exception as e:
                logger.error(f"Error storing chunks in vector database: {e}")
                await db.rollback()
                return False
    
    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using cosine similarity.

        TODO: Implement this method
        - Generate query embedding
        - Use pgvector's <=> operator for cosine distance
        - Apply metadata filters if provided
        - Return top k results

        Args:
            query: Search query text
            k: Number of results to return (default: 5)
            filter_metadata: Optional metadata filters (e.g., {"fund_id": 1})
            similarity_threshold: Minimum similarity score (0.0-1.0)

        Returns:
            List of similar documents with similarity scores

        Raises:
            ValueError: If query is empty or k is invalid
            SQLAlchemyError: If database operation fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if k <= 0:
            raise ValueError("k must be positive")

        async with self._get_db_session() as db:
            try:
                # Generate query embedding
                query_embedding = await self._get_embedding(query)
                embedding_list = query_embedding.tolist()

                # Build WHERE clause for metadata filters
                where_conditions = []
                params = {"k": k}

                if filter_metadata:
                    for key, value in filter_metadata.items():
                        if key in ["document_id", "fund_id"]:
                            where_conditions.append(f"{key} = :{key}")
                            params[key] = value

                where_clause = " AND ".join(where_conditions) if where_conditions else ""

                # Build threshold condition
                threshold_condition = ""
                if similarity_threshold is not None and similarity_threshold > 0:
                    # Use WHERE if no other conditions, otherwise AND
                    condition_prefix = "WHERE" if not where_clause else "AND"
                    threshold_condition = f" {condition_prefix} 1 - (embedding <=> (:embedding)::vector) >= :threshold"
                    params["threshold"] = similarity_threshold

                # Search using pgvector cosine similarity
                # <=> operator returns cosine distance, so we order by it ascending
                # Use proper vector syntax for pgvector
                search_sql = text(f"""
                    SELECT
                        id,
                        document_id,
                        fund_id,
                        content,
                        metadata,
                        1 - (embedding <=> (:embedding)::vector) as similarity_score
                    FROM document_embeddings
                    {f"WHERE {where_clause}" if where_clause else ""}{threshold_condition}
                    ORDER BY embedding <=> (:embedding)::vector
                    LIMIT :k
                """)

                result = db.execute(search_sql, {
                    "embedding": f"[{','.join(map(str, embedding_list))}]",
                    **params
                })

                # Format results
                results = []
                for row in result:
                    score = float(row[5])
                    # Validate and clamp similarity score
                    if not (0.0 <= score <= 1.0) or np.isnan(score) or np.isinf(score):
                        score = 0.0

                    # Parse metadata JSON
                    metadata = row[4]
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            metadata = {}

                    results.append({
                        "id": row[0],
                        "document_id": row[1],
                        "fund_id": row[2],
                        "content": row[3],
                        "metadata": metadata,
                        "score": score
                    })

                logger.debug(f"Similarity search returned {len(results)} results for query: {query[:50]}...")
                return results

            except SQLAlchemyError as e:
                logger.error(f"Database error in similarity search: {e}")
                raise
            except Exception as e:
                logger.error(f"Error in similarity search: {e}")
                return []
    def similarity_search_sync(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Synchronous version of similarity_search for Celery.

        Args:
            query: Search query text
            k: Number of results to return (default: 5)
            filter_metadata: Optional metadata filters
            similarity_threshold: Minimum similarity score (0.0-1.0)

        Returns:
            List of similar documents with similarity scores
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if k <= 0:
            raise ValueError("k must be positive")

        db = SessionLocal()
        try:
            # Generate query embedding synchronously
            query_embedding = self._get_embedding_sync(query)
            embedding_list = query_embedding.tolist()

            # Build WHERE clause for metadata filters
            where_conditions = []
            params = {"k": k}

            if filter_metadata:
                for key, value in filter_metadata.items():
                    if key in ["document_id", "fund_id"]:
                        where_conditions.append(f"{key} = :{key}")
                        params[key] = value

            where_clause = " AND ".join(where_conditions) if where_conditions else ""

            # Build threshold condition
            threshold_condition = ""
            if similarity_threshold is not None and similarity_threshold > 0:
                # Use WHERE if no other conditions, otherwise AND
                condition_prefix = "WHERE" if not where_clause else "AND"
                threshold_condition = f" {condition_prefix} 1 - (embedding <=> (:embedding)::vector) >= :threshold"
                params["threshold"] = similarity_threshold

            # Search using pgvector cosine similarity
            search_sql = text(f"""
                SELECT
                    id,
                    document_id,
                    fund_id,
                    content,
                    metadata,
                    1 - (embedding <=> (:embedding)::vector) as similarity_score
                FROM document_embeddings
                {f"WHERE {where_clause}" if where_clause else ""}
                {threshold_condition}
                ORDER BY embedding <=> (:embedding)::vector
                LIMIT :k
            """)

            result = db.execute(search_sql, {
                "embedding": f"[{','.join(map(str, embedding_list))}]",
                **params
            })

            # Format results
            results = []
            for row in result:
                score = float(row[5])
                # Validate and clamp similarity score
                if not (0.0 <= score <= 1.0) or np.isnan(score) or np.isinf(score):
                    score = 0.0

                # Parse metadata JSON
                metadata = row[4]
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}

                results.append({
                    "id": row[0],
                    "document_id": row[1],
                    "fund_id": row[2],
                    "content": row[3],
                    "metadata": metadata,
                    "score": score
                })

            logger.debug(f"Similarity search returned {len(results)} results for query: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Error in synchronous similarity search: {e}")
            return []
        finally:
            db.close()
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using configured provider.

        Args:
            text: Input text to embed

        Returns:
            Numpy array of embedding vector

        Raises:
            ValueError: If text is empty
            RuntimeError: If embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            if self.embedding_provider == "sentence-transformers":
                if self._embedding_model is None:
                    logger.error("Sentence transformers model not initialized - reinitializing...")
                    # Try to reinitialize the model
                    if self._init_sentence_transformers():
                        embedding = self._embedding_model.encode(text, convert_to_numpy=True)
                        return np.array(embedding, dtype=np.float32)
                    else:
                        logger.error("Failed to reinitialize sentence transformers model")
                        return np.zeros(self.embedding_dimensions, dtype=np.float32)
                embedding = self._embedding_model.encode(text, convert_to_numpy=True)
                return np.array(embedding, dtype=np.float32)

            elif self.embedding_provider == "gemini":
                return await self._get_gemini_embedding(text)

            elif self.embedding_provider == "openai":
                return await self._get_openai_embedding(text)

            else:
                logger.warning("No embedding provider configured, returning zero vector")
                return np.zeros(self.embedding_dimensions, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback to prevent system failure
            return np.zeros(self.embedding_dimensions, dtype=np.float32)

    def _get_embedding_sync(self, text: str) -> np.ndarray:
        """
        Synchronous version of _get_embedding for Celery context.

        Args:
            text: Input text to embed

        Returns:
            Numpy array of embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            if self.embedding_provider == "sentence-transformers":
                if self._embedding_model is None:
                    logger.error("Sentence transformers model not initialized - reinitializing...")
                    # Try to reinitialize the model
                    if self._init_sentence_transformers():
                        embedding = self._embedding_model.encode(text, convert_to_numpy=True)
                        return np.array(embedding, dtype=np.float32)
                    else:
                        logger.error("Failed to reinitialize sentence transformers model")
                        return np.zeros(self.embedding_dimensions, dtype=np.float32)
                embedding = self._embedding_model.encode(text, convert_to_numpy=True)
                return np.array(embedding, dtype=np.float32)

            elif self.embedding_provider == "gemini":
                # For sync context, we'll need to handle this differently
                logger.warning("Gemini embeddings not available in sync context")
                return np.zeros(self.embedding_dimensions, dtype=np.float32)

            elif self.embedding_provider == "openai":
                # For sync context, we'll need to handle this differently
                logger.warning("OpenAI embeddings not available in sync context")
                return np.zeros(self.embedding_dimensions, dtype=np.float32)

            else:
                logger.warning("No embedding provider configured, returning zero vector")
                return np.zeros(self.embedding_dimensions, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error generating embedding synchronously: {e}")
            # Return zero vector as fallback to prevent system failure
            return np.zeros(self.embedding_dimensions, dtype=np.float32)

    async def _get_gemini_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using Google Gemini API."""
        try:
            import google.genai as genai
            from google.genai import types

            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=[text],
                config=types.EmbedContentConfig(output_dimensionality=768)
            )
            return np.array(result.embeddings[0].values, dtype=np.float32)

        except ImportError:
            # Fallback to old API
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return np.array(result['embedding'], dtype=np.float32)

    async def _get_openai_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        response = self._openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    async def clear(self, fund_id: Optional[int] = None) -> bool:
        """
        Clear embeddings from the vector store.

        Args:
            fund_id: Optional fund ID to clear embeddings for specific fund.
                    If None, clears all embeddings.

        Returns:
            True if successful, False otherwise

        Raises:
            SQLAlchemyError: If database operation fails
        """
        async with self._get_db_session() as db:
            try:
                if fund_id is not None:
                    delete_sql = text("DELETE FROM document_embeddings WHERE fund_id = :fund_id")
                    result = db.execute(delete_sql, {"fund_id": fund_id})
                    deleted_count = result.rowcount
                    logger.info(f"Cleared {deleted_count} embeddings for fund {fund_id}")
                else:
                    delete_sql = text("DELETE FROM document_embeddings")
                    result = db.execute(delete_sql)
                    deleted_count = result.rowcount
                    logger.warning(f"Cleared all {deleted_count} embeddings from vector store")

                await db.commit()
                return True

            except SQLAlchemyError as e:
                await db.rollback()
                logger.error(f"Database error clearing vector store: {e}")
                raise
            except Exception as e:
                await db.rollback()
                logger.error(f"Error clearing vector store: {e}")
                return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Clear the vector store
        
        TODO: Implement this method
        - Delete all embeddings (or filter by fund_id)

        Get statistics about the vector store.

        Returns:
            Dictionary with store statistics
        """
        try:
            # Get total count
            count_sql = text("SELECT COUNT(*) FROM document_embeddings")
            result = self.db.execute(count_sql)
            total_count = result.scalar()

            # Get count by fund
            fund_sql = text("SELECT fund_id, COUNT(*) FROM document_embeddings GROUP BY fund_id")
            result = self.db.execute(fund_sql)
            fund_counts = {row[0]: row[1] for row in result}

            return {
                "total_embeddings": total_count,
                "funds_count": len(fund_counts),
                "embeddings_by_fund": fund_counts,
                "embedding_provider": self.embedding_provider,
                "embedding_dimensions": self.embedding_dimensions
            }

        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {
                "total_embeddings": 0,
                "funds_count": 0,
                "embeddings_by_fund": {},
                "embedding_provider": self.embedding_provider,
                "embedding_dimensions": self.embedding_dimensions,
                "error": str(e)
            }
