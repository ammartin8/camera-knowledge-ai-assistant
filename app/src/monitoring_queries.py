"""Monitoring query utilities for dashboard metrics.

Provides database queries for monitoring dashboard including:
- Query volume over time
- Response time by hour  
- Token usage breakdown (prompt vs completion)
- User feedback rate
- Retrieval success rate
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Dict, Any, Optional
import psycopg


def get_connection():
    """Create database connection using environment variables."""
    import os
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "camera_knowledge_db"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    )


def _execute_query(query: str, params: tuple = ()) -> Optional[List[Dict[str, Any]]]:
    """Execute a query and return results as list of dicts.
    
    Uses psycopg 3.x compatible approach - no row_factory to avoid _make_row errors.
    Manually converts tuples to dicts using column names from cursor description.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            
            # Get column names from cursor description
            columns = [column[0] for column in cur.description]
            
            # Fetch all rows and convert to dicts manually
            rows = cur.fetchall()
            if not rows:
                return []
                
            result = []
            for row in rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    row_dict[col] = row[i]
                result.append(row_dict)
            
            return result
    except Exception as e:
        print(f"Error executing query: {e}")
        return []
    finally:
        conn.close()


def get_query_volume_by_date(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit_days: int = 7
) -> Dict[str, Any]:
    """Get query volume grouped by date.
    
    Args:
        start_date: Start of date range (defaults to now - limit_days)
        end_date: End of date range (defaults to now)
        limit_days: Default number of days if no dates provided
    
    Returns:
        Dictionary with 'date' list and 'query_count' list (may be empty)
    """
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    
    if start_date is None:
        start_date = end_date - timedelta(days=limit_days)
    
    results = _execute_query("""
        SELECT 
            DATE(timestamp AT TIME ZONE 'UTC') as date,
            COUNT(*) as query_count
        FROM queries
        WHERE timestamp AT TIME ZONE 'UTC' >= %s
          AND timestamp AT TIME ZONE 'UTC' <= %s
        GROUP BY DATE(timestamp AT TIME ZONE 'UTC')
        ORDER BY date
    """, (start_date.isoformat(), end_date.isoformat()))
    
    # Always return dict with both keys, even if empty
    # Convert datetime objects to strings for Plotly compatibility
    dates = []
    counts = []
    if results:
        for r in results:
            date_val = r.get("date")
            # If DATE() returned NULL, use CURRENT_DATE as fallback
            if isinstance(date_val, datetime):
                dates.append(date_val.strftime("%Y-%m-%d"))
            elif date_val is not None:
                dates.append(str(date_val) if date_val else "NULL")
            else:
                # DATE() returned NULL - use CURRENT_DATE as fallback
                dates.append(datetime.now().strftime("%Y-%m-%d"))
            counts.append(r.get("query_count", 0))
    
    return {"date": dates, "query_count": counts}


def get_response_time_by_hour(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit_days: int = 7
) -> Dict[str, Any]:
    """Get average response time by hour.
    
    Args:
        start_date: Start of date range (defaults to now - limit_days)
        end_date: End of date range (defaults to now)
        limit_days: Default number of days if no dates provided
        
    Returns:
        Dictionary with 'hour' list and 'avg_response_time_ms' list (may be empty)
    """
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    
    if start_date is None:
        start_date = end_date - timedelta(days=limit_days)
    
    results = _execute_query("""
        SELECT 
            EXTRACT(HOUR FROM timestamp AT TIME ZONE 'UTC') as hour,
            AVG(response_time_ms) as avg_response_time_ms
        FROM queries
        WHERE timestamp AT TIME ZONE 'UTC' >= %s
          AND timestamp AT TIME ZONE 'UTC' <= %s
          AND response_time_ms IS NOT NULL
        GROUP BY EXTRACT(HOUR FROM timestamp AT TIME ZONE 'UTC')
        ORDER BY hour
    """, (start_date.isoformat(), end_date.isoformat()))
    
    # Always return dict with both keys, even if empty
    # Convert datetime objects to strings for Plotly compatibility
    hours = []
    avg_times = []
    if results:
        for r in results:
            hour_val = r.get("hour")
            # psycopg 3.x returns EXTRACT(HOUR FROM ...) as Decimal, convert properly
            if isinstance(hour_val, datetime):
                hours.append(int(hour_val))
            elif isinstance(hour_val, Decimal):
                # psycopg 3.x specific: EXTRACT returns Decimal
                hours.append(int(float(hour_val)))
            elif hour_val is not None:
                hours.append(int(hour_val) if isinstance(hour_val, (int, float)) else 0)
            else:
                # EXTRACT returned NULL - use 0 as fallback
                hours.append(0)
            avg_times.append(r.get("avg_response_time_ms", 0))
    
    return {"hour": hours, "avg_response_time_ms": avg_times}


def get_token_usage_breakdown(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit_days: int = 7
) -> Dict[str, int]:
    """Get token usage breakdown (prompt vs completion).
    
    Args:
        start_date: Start of date range (defaults to now - limit_days)
        end_date: End of date range (defaults to now)
        limit_days: Default number of days if no dates provided
        
    Returns:
        Dictionary with 'prompt_tokens', 'completion_tokens', 'total_tokens'
    """
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    
    if start_date is None:
        start_date = end_date - timedelta(days=limit_days)
    
    results = _execute_query("""
        SELECT 
            SUM(prompt_tokens) as prompt_total,
            SUM(completion_tokens) as completion_total
        FROM queries
        WHERE timestamp AT TIME ZONE 'UTC' >= %s
          AND timestamp AT TIME ZONE 'UTC' <= %s
    """, (start_date.isoformat(), end_date.isoformat()))
    
    if not results:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    
    row = results[0]
    prompt_total = int(row["prompt_total"]) or 0
    completion_total = int(row["completion_total"]) or 0
    
    return ["Prompt Tokens", prompt_total, "Completion Tokens", completion_total]


def get_feedback_rate(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit_days: int = 7
) -> Dict[str, Any]:
    """Get user feedback rate (thumbs up/down).
    
    Args:
        start_date: Start of date range (defaults to now - limit_days)
        end_date: End of date range (defaults to now)
        limit_days: Default number of days if no dates provided
        
    Returns:
        Dictionary with up_count, down_count, neutral_count, satisfaction_rate
    """
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    
    if start_date is None:
        start_date = end_date - timedelta(days=limit_days)
    
    results = _execute_query("""
        SELECT 
            COUNT(*) as total_feedback,
            SUM(CASE WHEN score = 1 THEN 1 ELSE 0 END) as up_count,
            SUM(CASE WHEN score = -1 THEN 1 ELSE 0 END) as down_count,
            SUM(CASE WHEN score = 0 THEN 1 ELSE 0 END) as neutral_count,
            ROUND((SUM(CASE WHEN score = 1 THEN 1 ELSE 0 END)::NUMERIC * 100.0 / 
                 NULLIF(COUNT(*), 0)), 2)::NUMERIC(5,2) as satisfaction_rate
        FROM feedback f
        JOIN queries q ON f.query_id = q.id
        WHERE q.timestamp AT TIME ZONE 'UTC' >= %s
          AND q.timestamp AT TIME ZONE 'UTC' <= %s
    """, (start_date.isoformat(), end_date.isoformat()))
    
    if not results or results[0].get("total_feedback", 0) == 0:
        return {
            "total_feedback": 0,
            "up_count": 0,
            "down_count": 0,
            "neutral_count": 0,
            "satisfaction_rate": 0.0,
        }
    
    row = results[0]
    return {
        "total_feedback": int(row.get("total_feedback", 0)),
        "up_count": int(row.get("up_count", 0)),
        "down_count": int(row.get("down_count", 0)),
        "neutral_count": int(row.get("neutral_count", 0)),
        "satisfaction_rate": float(row.get("satisfaction_rate", 0.0)),
    }


def get_retrieval_success_rate(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit_days: int = 7
) -> Dict[str, float]:
    """Get retrieval success rate from queries table.
    
    NOTE: Uses queries.response_time_ms > 0 as a proxy for successful retrievals
    since query_logs table is not yet implemented in the current schema.
    
    Args:
        start_date: Start of date range (defaults to now - limit_days)
        end_date: End of date range (defaults to now)
        limit_days: Default number of days if no dates provided
        
    Returns:
        Dictionary with total_queries, successful_retrievals, success_rate
    """
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    
    if start_date is None:
        start_date = end_date - timedelta(days=limit_days)
    
    results = _execute_query("""
        SELECT 
            COUNT(*) as total_queries,
            SUM(CASE WHEN response_time_ms > 0 THEN 1 ELSE 0 END) as successful_retrievals
        FROM queries
        WHERE timestamp AT TIME ZONE 'UTC' >= %s
          AND timestamp AT TIME ZONE 'UTC' <= %s
    """, (start_date.isoformat(), end_date.isoformat()))
    
    if not results:
        return {
            "total_queries": 0,
            "successful_retrievals": 0,
            "success_rate": 0.0,
        }
    
    row = results[0]
    total = int(row.get("total_queries", 0))
    successful = int(row.get("successful_retrievals", 0))
    
    return {
        "total_queries": total,
        "successful_retrievals": successful,
        "success_rate": float(successful / total * 100) if total > 0 else 0.0,
    }


def get_aggregate_stats(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit_days: int = 7
) -> Dict[str, Any]:
    """Get aggregate statistics for dashboard summary.
    
    Args:
        start_date: Start of date range (defaults to now - limit_days)
        end_date: End of date range (defaults to now)
        limit_days: Default number of days if no dates provided
        
    Returns:
        Dictionary with total_queries, avg_response_time, total_cost, avg_tokens
    """
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    
    if start_date is None:
        start_date = end_date - timedelta(days=limit_days)
    
    results = _execute_query("""
        SELECT
            COUNT(*) as total_queries,
            AVG(response_time_ms) as avg_response_time_ms,
            SUM(cost) as total_cost,
            SUM(total_tokens) as total_tokens
        FROM queries
        WHERE timestamp AT TIME ZONE 'UTC' >= %s
          AND timestamp AT TIME ZONE 'UTC' <= %s
    """, (start_date.isoformat(), end_date.isoformat()))
    
    if not results or results[0].get("total_queries", 0) == 0:
        return {
            "total_queries": 0,
            "avg_response_time_ms": 0.0,
            "total_cost": 0.0,
            "total_tokens": 0,
        }
    
    row = results[0]
    return {
        "total_queries": int(row.get("total_queries", 0)),
        "avg_response_time_ms": float(row.get("avg_response_time_ms") or 0.0),
        "total_cost": float(row.get("total_cost") or 0.0),
        "total_tokens": int(row.get("total_tokens") or 0),
    }


def get_recent_conversations(limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent conversations for display.
    
    Args:
        limit: Maximum number of records to return
        
    Returns:
        List of dictionaries with conversation data
    """
    results = _execute_query("""
        SELECT id, question, answer, course, model,
               instructions, prompt,
               prompt_tokens, completion_tokens, total_tokens,
               response_time_ms, cost, timestamp
        FROM queries
        ORDER BY timestamp DESC
        LIMIT %s
    """, (limit,))
    
    return results if results else []
