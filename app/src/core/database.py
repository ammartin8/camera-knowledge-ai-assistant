"""Database utilities for camera knowledge RAG application.

Provides PostgreSQL operations including:
- Schema setup with pgvector extension
- Query logging and metrics tracking
- User feedback storage
- Conversation retrieval and statistics
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import psycopg
from psycopg.rows import DictRow


@dataclass
class Stats:
    """Statistics for monitoring dashboard."""
    total: int = 0
    avg_response_time: float = 0.0
    total_cost: float = 0.0
    avg_tokens: float = 0.0


def get_connection():
    """Create database connection using environment variables."""
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
        score INTEGER CHECK (score IN (-1, 0, 1)),
        timestamp TIMESTAMPTZ DEFAULT NOW()
    );

    -- Embeddings table: stores vector embeddings for retrieval with pgvector
    CREATE TABLE IF NOT EXISTS embeddings (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        chunk_text TEXT NOT NULL,
        embedding VECTOR(1536),
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


def save_feedback(
    query_id: int,
    source: str = "user",
    relevance: str = None,
    explanation: str = None,
    score: int = 0
):
    """Save feedback for a conversation.
    
    Args:
        query_id: ID of the query to provide feedback on
        source: Source of feedback (e.g., "user", "evaluator")
        relevance: Optional relevance notes
        explanation: Optional explanation for the feedback
        score: Feedback score (-1 = down, 0 = neutral, +1 = up)
    """
    timestamp = datetime.now(timezone.utc)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (
                    query_id, source, relevance,
                    explanation, score, timestamp
                ) VALUES (
                    %s, %s, %s, %s, %s, %s
                )
                """,
                (query_id, source, relevance,
                 explanation, score, timestamp),
            )
        conn.commit()
    except Exception as e:
        print(f"Error saving feedback: {e}")
        conn.rollback()
    finally:
        conn.close()


def get_feedback(query_id: int) -> List[Dict[str, Any]]:
    """Get all feedback for a specific query.
    
    Args:
        query_id: ID of the query
        
    Returns:
        List of feedback records
    """
    conn = get_connection()
    try:
        with conn.cursor(row_factory=DictRow) as cur:
            cur.execute("""
                SELECT id, score, source, relevance, explanation, timestamp
                FROM feedback
                WHERE query_id = %s
                ORDER BY timestamp DESC
            """, (query_id,))
            rows = cur.fetchall()
    finally:
        conn.close()

    return [dict(row) for row in rows]


def get_conversations(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent conversations ordered by timestamp.
    
    Args:
        limit: Maximum number of records to return
        
    Returns:
        List of dictionaries with conversation data
    """
    conn = get_connection()
    try:
        with conn.cursor(row_factory=DictRow) as cur:
            cur.execute("""
                SELECT id, question, answer, course, model,
                       instructions, prompt,
                       prompt_tokens, completion_tokens, total_tokens,
                       response_time_ms, cost, timestamp
                FROM queries
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
    finally:
        conn.close()

    return [dict(row) for row in rows]


def get_stats() -> Optional[Stats]:
    """Get aggregate statistics for monitoring dashboard.
    
    Returns:
        Stats object with totals and averages, or None if no data
    """
    conn = get_connection()
    try:
        with conn.cursor(row_factory=DictRow) as cur:
            cur.execute("""
                SELECT
                    COUNT(*),
                    AVG(response_time_ms),
                    SUM(cost),
                    AVG(total_tokens)
                FROM queries
            """)
            row = cur.fetchone()
    finally:
        conn.close()

    if not row or row[0] == 0:
        return None
    
    return Stats(
        total=row[0],
        avg_response_time=row[1] or 0.0,
        total_cost=row[2] or 0.0,
        avg_tokens=row[3] or 0.0,
    )


def get_feedback_summary(limit: int = 10) -> List[Dict[str, Any]]:
    """Get feedback summary for recent conversations.
    
    Args:
        limit: Maximum number of records to return
        
    Returns:
        List of dictionaries with feedback data
    """
    conn = get_connection()
    try:
        with conn.cursor(row_factory=DictRow) as cur:
            cur.execute("""
                SELECT q.id, q.question, fs.up_count, fs.down_count, fs.satisfaction_rate
                FROM queries q
                JOIN feedback_summary fs ON q.id = fs.query_id
                ORDER BY q.timestamp DESC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
    finally:
        conn.close()

    return [dict(row) for row in rows]


def calculate_cost(model: str, usage: Dict[str, int]) -> float:
    """Calculate LLM API cost based on model and usage.
    
    Args:
        model: Model name (e.g., "gpt-4", "llama3.2")
        usage: Dictionary with prompt_tokens and completion_tokens
        
    Returns:
        Cost in dollars
    """
    cost = 0.0
    
    if "gpt-4" in model or "gpt-5" in model:
        cost = (usage.get("prompt_tokens", 0) * 0.03 + 
                usage.get("completion_tokens", 0) * 0.06) / 1_000_000
    elif "llama" in model.lower():
        cost = 0.0
    else:
        cost = 0.0
    
    return cost


def track_llm_call(
    model: str,
    prompt: str,
    instructions: str,
    answer: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    response_time: float = 0.0,
    cost: float = 0.0
) -> Dict[str, Any]:
    """Track an LLM call with metrics.
    
    Args:
        model: Model used for the call
        prompt: The prompt sent to the LLM
        instructions: System instructions used
        answer: The LLM's response
        prompt_tokens: Number of tokens in prompt
        completion_tokens: Number of tokens in response
        response_time: Time taken for the call in seconds
        cost: Cost incurred for this call
        
    Returns:
        Dictionary with tracking data including calculated total tokens and cost
    """
    total_tokens = prompt_tokens + completion_tokens
    timestamp = datetime.now(timezone.utc)
    
    return {
        "model": model,
        "prompt": prompt[:100] if prompt else "",  # Truncate for logging
        "instructions": instructions[:100] if instructions else "",
        "answer": answer[:100] if answer else "",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "response_time": response_time,
        "cost": cost,
        "timestamp": timestamp.isoformat(),
    }
