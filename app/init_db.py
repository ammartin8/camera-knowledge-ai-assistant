"""Database initialization script using psycopg."""

import os
from datetime import datetime, timezone
import psycopg


def get_db_connection():
    """Create database connection with environment variables."""
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "camera_knowledge_db"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    )


def init_db():
    """Initialize the database schema."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Enable pgvector extension FIRST before creating tables that use it
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
            """)

            # Create queries table
            cur.execute("""
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
                )
            """)

            # Create feedback table (score: -1 down, 0 neutral, +1 up)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    query_id INTEGER REFERENCES queries(id) ON DELETE CASCADE,
                    score INTEGER CHECK (score IN (-1, 0, 1)),
                    timestamp TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create embeddings table for vector storage
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    query_id INTEGER REFERENCES queries(id) ON DELETE CASCADE,
                    question TEXT,
                    answer TEXT,
                    course TEXT,
                    embedding vector(384),  -- Default dimension for minsearch
                    timestamp TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create views for dashboard queries
            cur.execute("""
                CREATE OR REPLACE VIEW feedback_summary AS
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) FILTER (WHERE score = 1) as upvotes,
                    COUNT(*) FILTER (WHERE score = -1) as downvotes,
                    ROUND(AVG(score)::numeric, 2) as avg_score
                FROM feedback
                GROUP BY DATE(timestamp)
                ORDER BY date DESC;
            """)

            cur.execute("""
                CREATE OR REPLACE VIEW daily_stats AS
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as total_queries,
                    AVG(response_time_ms) as avg_response_time,
                    SUM(prompt_tokens) as total_prompt_tokens,
                    SUM(completion_tokens) as total_completion_tokens,
                    SUM(total_tokens) as total_tokens
                FROM queries
                GROUP BY DATE(timestamp)
                ORDER BY date DESC;
            """)
            
            conn.commit()
            print("✅ Database schema created successfully!")
            
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    print("Initializing database schema...")
    init_db()
