"""
Database initialization
"""
from app.db.base import Base
from app.db.session import engine
from sqlalchemy import text
# Import models to ensure they are registered with SQLAlchemy
from app.models.fund import Fund  # noqa: F401
from app.models.transaction import CapitalCall, Distribution, Adjustment  # noqa: F401
from app.models.document import Document  # noqa: F401


def init_db():
    """Initialize database tables"""
    # Enable pgvector extension
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Create document embeddings table with proper vector type
    with engine.connect() as conn:
        # Check if table exists and create if not
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'document_embeddings'
                AND table_schema = 'public'
            )
        """))
        table_exists = result.fetchone()[0]

        if not table_exists:
            conn.execute(text("""
                CREATE TABLE document_embeddings (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER,
                    fund_id INTEGER,
                    content TEXT NOT NULL,
                    embedding vector(384),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
            print("Document embeddings table created successfully!")

    print("Database tables created successfully!")


if __name__ == "__main__":
    init_db()
