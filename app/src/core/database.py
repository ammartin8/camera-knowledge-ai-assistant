"""Database schema and connection utilities using psycopg.

Provides PostgreSQL setup with pgvector extension for storing:
- Query logs with metrics (conversations table)
- User feedback (thumbs up/down)
- Vector embeddings for retrieval
"""

import os
import psycopg
from psycopg.rows import DictRow


def get_connection():
    """Create a database connection using environment variables."""
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "camera_knowledge_db"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    )


def create_schema():
    """Create the database schema with all required tables."""
    
    schema_sql = """
    -- Enable pgvector extension if not already enabled
    DO IF NOT EXISTS (SELECT FROM pg_extension WHERE extname = 'pgvector')
        CREATE EXTENSION pgvector;

    -- Queries table: stores all user queries and metadata
    CREATE TABLE IF NOT EXISTS queries (
        id SERIAL PRIMARY KEY,
        question TEXT NOT NULL,
        answer TEXT,
        course TEXT DEFAULT 'camera_knowledge',
        model TEXT DEFAULT 'llama3.2',
        instructions TEXT,
        prompt TEXT,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        total_tokens INTEGER,
        response_time_ms FLOAT,
        cost FLOAT,
        timestamp TIMESTAMPTZ DEFAULT NOW()
    );

    -- Feedback table: stores user feedback for each query (score: -1 down, 0 neutral, +1 up)
    CREATE TABLE IF NOT EXISTS feedback (
        id SERIAL PRIMARY KEY,
        query_id INTEGER REFERENCES queries(id) ON DELETE CASCADE,
        source TEXT DEFAULT 'user',
        relevance TEXT,
        explanation TEXT,
        score INTEGER CHECK (score IN (-1, 0, 1)),  -- -1 down, 0 neutral, +1 up
        timestamp TIMESTAMPTZ DEFAULT NOW()
    );

    -- Embeddings table: stores vector embeddings for retrieval with pgvector
    CREATE TABLE IF NOT EXISTS embeddings (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        chunk_text TEXT NOT NULL,
        embedding VECTOR(1536),  -- OpenAI-style embedding dimension
        source_document VARCHAR(255),
        UNIQUE (chunk_text)
    );

    -- Query logs for monitoring: detailed query performance
    CREATE TABLE IF NOT EXISTS query_logs (
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
        question TEXT NOT NULL,
        retrieved_chunk_ids JSONB,
        retrieval_score FLOAT,
        response_tokens INTEGER,
        prompt_tokens INTEGER,
        total_time_ms INTEGER,
        timestamp TIMESTAMPTZ DEFAULT NOW()
    );

    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings USING ivfflat (embedding vector_cosine_ops);
    CREATE INDEX IF NOT EXISTS idx_query_logs_timestamp ON query_logs(timestamp DESC);

    -- Views for common queries
    CREATE OR REPLACE VIEW feedback_summary AS
    SELECT 
        q.id as query_id,
        q.question,
        COUNT(f.id) as total_feedback,
        SUM(CASE WHEN f.score = 1 THEN 1 ELSE 0 END) as up_count,
        SUM(CASE WHEN f.score = -1 THEN 1 ELSE 0 END) as down_count,
        ROUND(
            (SUM(CASE WHEN f.score = 1 THEN 1 ELSE 0 END)::FLOAT / 
             NULLIF(COUNT(f.id), 0)) * 100, 2
        ) as satisfaction_rate
    FROM queries q
    LEFT JOIN feedback f ON q.id = f.query_id
    GROUP BY q.id, q.question;

    CREATE OR REPLACE VIEW daily_stats AS
    SELECT 
        DATE(timestamp AT TIME ZONE 'UTC') as date,
        COUNT(*) as total_queries,
        AVG(completion_tokens::INTEGER) as avg_completion_tokens,
        AVG(response_time_ms::INTEGER) as avg_response_time_ms,
        SUM(CASE WHEN retrieval_method = 'vector' THEN 1 ELSE 0 END) * 1.0 / 
            NULLIF(COUNT(*), 0) as vector_search_ratio
    FROM queries
    GROUP BY DATE(timestamp AT TIME ZONE 'UTC');
    """

    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(schema_sql)
    conn.commit()
    print("Database schema created successfully!")


if __name__ == "__main__":
    create_schema()
