"""Database query helper functions using psycopg.

Provides functions to retrieve conversations and statistics from PostgreSQL.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import psycopg


@dataclass
class Stats:
    """Statistics for monitoring dashboard."""
    total: int
    avg_response_time: float
    total_cost: float
    avg_tokens: float


@dataclass
class LLMCallRecord:
    """Record of an LLM call with metrics."""
    model: str
    prompt: str
    instructions: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float
    timestamp: datetime


def get_connection():
    """Create database connection with environment variables."""
    import os
    
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "camera_knowledge_db"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    )


def get_conversations(limit: int = 10) -> List[LLMCallRecord]:
    """Get recent conversations ordered by timestamp.
    
    Args:
        limit: Maximum number of records to return
        
    Returns:
        List of LLMCallRecord objects
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

    return [row_to_record(row) for row in rows]


def row_to_record(row):
    """Convert database row to LLMCallRecord."""
    return LLMCallRecord(
        model=row.get("model", ""),
        prompt=row.get("prompt", ""),
        instructions=row.get("instructions", ""),
        answer=row.get("answer"),
        prompt_tokens=row.get("prompt_tokens") or 0,
        completion_tokens=row.get("completion_tokens") or 0,
        total_tokens=row.get("total_tokens") or 0,
        response_time=row.get("response_time_ms") or 0.0,
        cost=row.get("cost") or 0.0,
        timestamp=row["timestamp"],
    )


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


def get_feedback_summary(limit: int = 10) -> List[dict]:
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


if __name__ == "__main__":
    print("Testing database queries...")
    records = get_conversations(limit=5)
    for record in records:
        print(record)
    
    stats = get_stats()
    if stats:
        print(f"\nStats: {stats}")
