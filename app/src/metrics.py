"""Metrics capture infrastructure for LLM call tracking.

Provides an LLMCallRecord dataclass that encapsulates all LLM call metrics
(model, prompt, instructions, answer, tokens, response time, cost, timestamp).

This provides a single source of truth for tracking every LLM interaction.

Usage example:
    from app.src.metrics import LLMCallRecord, save_record_to_db

    record = LLMCallRecord(
        model="llama3.2",
        prompt="What is the camera sensor size?",
        instructions="You are a helpful assistant about cameras.",
        answer="The camera sensor size is 1/1.7 inches.",
        prompt_tokens=45,
        completion_tokens=32,
        total_tokens=77,
        response_time=0.5,
    )

    save_record_to_db(record)
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import getLogger

# DISABLED: import psycopg

logger = getLogger(__name__)


@dataclass
class LLMCallRecord:
    """LLM call record with complete metrics for tracking and analysis."""
    model: str
    prompt: str  # User's question
    instructions: str  # System/developer prompts
    answer: str  # Response text
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float  # Time in seconds
    cost: float  # Cost in dollars
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate token counts and numeric fields."""
        if self.prompt_tokens < 0:
            raise ValueError("prompt_tokens cannot be negative")
        if self.completion_tokens < 0:
            raise ValueError("completion_tokens cannot be negative")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError(
                f"total_tokens ({self.total_tokens}) must equal "
                f"prompt_tokens ({self.prompt_tokens}) + completion_tokens ({self.completion_tokens})"
            )
        if self.response_time < 0:
            raise ValueError("response_time cannot be negative")
        if self.cost < 0:
            raise ValueError("cost cannot be negative")


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate LLM API cost based on token usage and pricing rates.

    Args:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the response

    Returns:
        Cost in dollars using environment variables for pricing rates:
        - LLM_INPUT_RATE (default: "0.000015")
        - LLM_OUTPUT_RATE (default: "0.000050")

    Note: Zero rates are allowed (e.g., for free models like Llama).

    Formula:
        cost = (prompt_tokens * input_rate) + (completion_tokens * output_rate)

    Example:
        >>> calculate_cost(100, 200)
        0.0075  # (100 * 0.000015) + (200 * 0.000050)
    """
    input_rate = float(os.getenv("LLM_INPUT_RATE", "0.000015"))
    output_rate = float(os.getenv("LLM_OUTPUT_RATE", "0.000050"))

    # Validate that rates are non-negative (zero allowed for free models)
    if input_rate < 0:
        raise ValueError("input_rate cannot be negative")
    if output_rate < 0:
        raise ValueError("output_rate cannot be negative")

    cost = (prompt_tokens * input_rate) + (completion_tokens * output_rate)
    return round(cost, 6)


def save_record_to_db(record: LLMCallRecord) -> None:
    """Save an LLM call record to the PostgreSQL queries table.

    Args:
        record: LLMCallRecord object with all required fields

    Raises:
        psycopg.Error: If database connection or query fails
    """
    conn = None
    try:
        from app.src.core.database import get_connection

        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO queries (
                    question, answer, course, model, instructions, prompt,
                    prompt_tokens, completion_tokens, total_tokens,
                    response_time_ms, cost, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    record.prompt,  # question
                    record.answer,
                    "camera_knowledge",  # course (hardcoded for now)
                    record.model,
                    record.instructions,
                    record.prompt,  # prompt (same as question for MVP)
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.response_time * 1000,  # Convert seconds to milliseconds
                    record.cost,
                    record.timestamp,
                ),
            )
        conn.commit()
        logger.info(
            "LLMCallRecord saved successfully",
            extra={
                "model": record.model,
                "total_tokens": record.total_tokens,
                "response_time_ms": record.response_time * 1000,
                "cost": record.cost,
            },
        )
    except psycopg.Error as e:
        logger.error(
            "Error saving LLM call record to database",
            extra={"error": str(e), "model": getattr(record, "model", "unknown")},
        )
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()
