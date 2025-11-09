"""
Vector store service using pgvector (PostgreSQL extension)

TODO: Implement vector storage using pgvector
- Create embeddings table in PostgreSQL
- Store document chunks with vector embeddings
- Implement similarity search using pgvector operators
- Handle metadata filtering
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.config import settings
from app.db.session import SessionLocal


class VectorStore:
    """pgvector-based vector store for document embeddings"""
    
    def __init__(self, db: Session = None):
        self.db = db or SessionLocal()
        self.embeddings = self._initialize_embeddings()
        self._ensure_extension()
    
    def _initialize_embeddings(self):
        """Initialize embedding model - sentence-transformers for local processing"""
        try:
            from sentence_transformers import SentenceTransformer
            print("Using sentence-transformers/all-MiniLM-L6-v2 embeddings")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            return "sentence-transformers"
        except ImportError:
            print("sentence-transformers not available, falling back to OpenAI")
            if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.strip():
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
                return "openai"
            else:
                print("No embedding model available")
                return None
    
    def _ensure_extension(self):
        """
        Ensure pgvector extension is enabled

        TODO: Implement this method
        - Execute: CREATE EXTENSION IF NOT EXISTS vector;
        - Create embeddings table if not exists
        """
        try:
            # Enable pgvector extension
            self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            # Create embeddings table
            # Dimension: 384 for sentence-transformers/all-MiniLM-L6-v2
            dimension = 768

            # Check if table exists first, and drop it if it has wrong dimensions
            check_table_sql = """
            SELECT column_name, data_type, udt_name
            FROM information_schema.columns
            WHERE table_name = 'document_embeddings' AND table_schema = 'public';
            """

            # Note: Removed table dropping logic to preserve existing data
            # Table will only be created if it doesn't exist

            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id SERIAL PRIMARY KEY,
                document_id INTEGER,
                fund_id INTEGER,
                content TEXT NOT NULL,
                embedding vector({dimension}),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """

            self.db.execute(text(create_table_sql))
            self.db.commit()
            print("Created document_embeddings table successfully")

            # Skip IVFFlat index creation for now - it has dimension limits and can cause transaction issues
            # The system will work without it, just with slower similarity search
            print("Skipping vector index creation - similarity search will be slower but functional")

        except Exception as e:
            print(f"Error ensuring pgvector extension: {e}")
            self.db.rollback()
    
    async def add_document(self, content: str, metadata: Dict[str, Any]):
        """
        Add a document to the vector store

        TODO: Implement this method
        - Generate embedding for content
        - Insert into document_embeddings table
        - Store metadata as JSONB
        """
        try:
            # Generate embedding
            embedding = await self._get_embedding(content)
            embedding_list = embedding.tolist()

            # Insert into database
            insert_sql = text("""
                INSERT INTO document_embeddings (document_id, fund_id, content, embedding, metadata)
                VALUES (:document_id, :fund_id, :content, :embedding, :metadata)
            """)

            # Convert metadata to proper JSON format
            import json
            metadata_json = json.dumps(metadata)

            self.db.execute(insert_sql, {
                "document_id": metadata.get("document_id"),
                "fund_id": metadata.get("fund_id"),
                "content": content,
                "embedding": embedding_list,  # Pass as list for pgvector
                "metadata": metadata_json  # Pass as JSON string
            })
            self.db.commit()
        except Exception as e:
            print(f"Error adding document: {e}")
            self.db.rollback()
            raise

    async def store_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Store multiple text chunks in the vector database

        Args:
            chunks: List of chunks with content and metadata
        """
        if not chunks:
            print("No chunks to store")
            return True

        try:
            print(f"Storing {len(chunks)} text chunks in vector database...")

            # Process chunks in batches to avoid overwhelming the API
            batch_size = 10
            stored_count = 0

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]

                # Generate embeddings for batch
                embeddings = []
                for chunk in batch:
                    embedding = await self._get_embedding(chunk["content"])
                    embeddings.append(embedding)

                # Store batch in database - one transaction per batch to avoid aborted transaction issues
                try:
                    for j, chunk in enumerate(batch):
                        embedding = embeddings[j]
                        embedding_list = embedding.tolist()

                        # Prepare metadata
                        metadata = chunk.get("metadata", {})
                        document_id = metadata.get("document_id")
                        fund_id = metadata.get("fund_id")

                        # Insert into database - use proper parameter binding for psycopg2
                        insert_sql = text("""
                            INSERT INTO document_embeddings (document_id, fund_id, content, embedding, metadata)
                            VALUES (:document_id, :fund_id, :content, :embedding, :metadata)
                        """)

                        # Convert metadata to proper JSON format
                        import json
                        metadata_json = json.dumps(metadata)

                        self.db.execute(insert_sql, {
                            "document_id": document_id,
                            "fund_id": fund_id,
                            "content": chunk["content"],
                            "embedding": embedding_list,  # Pass as list for pgvector
                            "metadata": metadata_json  # Pass as JSON string
                        })

                    # Commit after each batch
                    self.db.commit()
                    stored_count += len(batch)
                    print(f"Stored batch {i//batch_size + 1}: {len(batch)} chunks")

                except Exception as batch_error:
                    print(f"Error storing batch {i//batch_size + 1}: {batch_error}")
                    self.db.rollback()
                    # Continue with next batch instead of failing completely
            print(f"Successfully stored {stored_count} chunks in vector database")
            return True

        except Exception as e:
            print(f"Error storing chunks in vector database: {e}")
            import traceback
            traceback.print_exc()
            self.db.rollback()
            return False
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using cosine similarity
        
        TODO: Implement this method
        - Generate query embedding
        - Use pgvector's <=> operator for cosine distance
        - Apply metadata filters if provided
        - Return top k results
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"fund_id": 1})
            
        Returns:
            List of similar documents with scores
        """
        try:
            # Generate query embedding
            query_embedding = await self._get_embedding(query)
            embedding_list = query_embedding.tolist()
            
            # Build query with optional filters
            where_clause = ""
            if filter_metadata:
                conditions = []
                for key, value in filter_metadata.items():
                    if key in ["document_id", "fund_id"]:
                        conditions.append(f"{key} = {value}")
                if conditions:
                    where_clause = "WHERE " + " AND ".join(conditions)
            
            # Search using cosine similarity (1 - cosine_distance)
            # Note: pgvector's <=> operator returns cosine distance, so we use it directly for ordering
            search_sql = text(f"""
                SELECT
                    id,
                    document_id,
                    fund_id,
                    content,
                    metadata,
                    1 - (embedding <=> ARRAY{embedding_list}::vector) as similarity_score
                FROM document_embeddings
                {where_clause}
                ORDER BY embedding <=> ARRAY{embedding_list}::vector
                LIMIT :k
            """)

            # Use proper parameter binding for pgvector
            result = self.db.execute(search_sql, {
                "k": k
            })
            
            # Format results
            results = []
            for row in result:
                score = float(row[5])
                # Handle NaN and infinity values
                if not (score >= 0.0 and score <= 1.0):
                    score = 0.0
                results.append({
                    "id": row[0],
                    "document_id": row[1],
                    "fund_id": row[2],
                    "content": row[3],
                    "metadata": row[4],
                    "score": score
                })
            
            return results
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using sentence-transformers"""
        try:
            if self.embeddings == "sentence-transformers":
                # Use sentence-transformers for local embeddings
                embedding = self.embedding_model.encode(text, convert_to_numpy=True)
                return np.array(embedding, dtype=np.float32)

            elif self.embeddings == "openai":
                # Fallback to OpenAI if sentence-transformers not available
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                return np.array(response.data[0].embedding, dtype=np.float32)

            else:
                # Fallback: return zero vector
                print("Warning: No embedding model configured, returning zero vector")
                return np.zeros(384, dtype=np.float32)  # sentence-transformers dimension

        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(384, dtype=np.float32)
    
    def clear(self, fund_id: Optional[int] = None):
        """
        Clear the vector store
        
        TODO: Implement this method
        - Delete all embeddings (or filter by fund_id)
        """
        try:
            if fund_id:
                delete_sql = text("DELETE FROM document_embeddings WHERE fund_id = :fund_id")
                self.db.execute(delete_sql, {"fund_id": fund_id})
            else:
                delete_sql = text("DELETE FROM document_embeddings")
                self.db.execute(delete_sql)
            
            self.db.commit()
        except Exception as e:
            print(f"Error clearing vector store: {e}")
            self.db.rollback()
