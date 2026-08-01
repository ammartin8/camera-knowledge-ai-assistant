"""Database feedback functions using psycopg.

Provides functions to save user feedback (thumbs up/down) for conversations.
Similar to reference db_feedback.py in LLM Zoomcamp.
Score: -1 = down, 0 = neutral, +1 = up
"""

from datetime import datetime, timezone
import psycopg


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


def get_feedback(query_id: int) -> list:
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


if __name__ == "__main__":
    print("Testing feedback functions...")
    
    # Test adding feedback
    save_feedback(1, score=1)  # Thumbs up
    save_feedback(2, score=-1)  # Thumbs down
    
    print("Feedback saved successfully!")
