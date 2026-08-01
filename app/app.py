"""Camera Knowledge AI - Main Streamlit Application.

This application provides both chat interface and monitoring dashboard
for the Camera Knowledge AI Assistant RAG system.
"""

import os
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
from dataclasses import asdict
from psycopg.rows import DictRow
from dotenv import load_dotenv

from src.core.db_query import get_conversations, get_stats
from src.core.db_feedback import save_feedback

# Load environment variables
load_dotenv()

# Configuration from environment
LLM_URL = os.getenv("LLM_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("API_KEY", "ollama")
MODEL = os.getenv("MODEL", "llama3.2")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "camera_knowledge_db")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "show_monitoring" not in st.session_state:
    st.session_state.show_monitoring = False


def get_feedback_rate():
    """Calculate feedback rate from database."""
    try:
        stats = get_stats()
        if stats.total == 0:
            return 0.0
        # Get up/down counts from feedback_summary view
        conn = psycopg.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        )
        with conn.cursor(row_factory=DictRow) as cur:
            cur.execute("SELECT up_count, down_count FROM feedback_summary LIMIT 1")
            row = cur.fetchone()
            conn.close()
            if row and row['up_count'] > 0 and row['down_count'] > 0:
                return (row['up_count'] / (row['up_count'] + row['down_count'])) * 100
        return 0.0
    except Exception:
        return 0.0


# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Go to",
    ["Chat", "Monitoring"],
)

# Chat Page
if page == "Chat":
    st.title("📷 Camera Knowledge AI")
    st.markdown("""
    Ask questions about the Canon EOS R6 Mark II camera!
    
    **Example questions:**
    - What is ISO and how does it affect image quality?
    - How do I switch aperture mode from Full to Aperture Priority?
    - Explain HDR Mode and when to use it
    - How do I check remaining battery level on the LCD screen?
    """)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                with st.expander("📚 See source citations"):
                    for citation in message["citations"]:
                        st.info(f"**{citation}**")

    # Chat input
    if prompt := st.chat_input("Type your question here..."):
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response (placeholder - will be implemented in later tickets)
        with st.chat_message("assistant"):
            st.markdown("⏳ Generating response...")
            
            # TODO: Implement actual RAG pipeline
            response = "This is a placeholder response. The RAG pipeline will be implemented in later tickets."
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "citations": []
            })

# Monitoring Page
elif page == "Monitoring":
    st.title("📊 Monitoring Dashboard")
    
    # Set up date range filter
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        stats = get_stats()
        col1.metric("Total conversations", stats.total or 0)
        col2.metric("Avg response time", f"{stats.avg_response_time:.2f}s" if stats.avg_response_time else "N/A")
        col3.metric("Total cost", f"${stats.total_cost:.4f}" if stats.total_cost else "$0.0000")
        col4.metric("Avg tokens", f"{stats.avg_tokens or 0:,.0f}")
    except Exception as e:
        st.error(f"Could not load stats: {e}")
    
    # Feedback metrics
    try:
        conn = psycopg.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        )
        with conn.cursor(row_factory=DictRow) as cur:
            cur.execute("SELECT up_count, down_count FROM feedback_summary LIMIT 1")
            row = cur.fetchone()
            conn.close()
            
            if row and row['up_count'] > 0:
                st.metric("Feedback Rate", f"{(row['up_count'] / (row['up_count'] + row['down_count'])) * 100:.1f}%")
    except Exception as e:
        pass
    
    # Charts
    records = get_conversations(limit=100)
    
    if records:
        df = pd.DataFrame([asdict(r) for r in records])
        
        st.subheader("Cost over time")
        st.line_chart(df, x="timestamp", y="cost")

        st.subheader("Response time over time")
        st.line_chart(df, x="timestamp", y="response_time")

        st.subheader("Recent conversations")
        records = get_conversations(limit=20)

        for record in records:
            with st.expander(f"**{record.prompt[:100]}...**"):
                st.write(f"{record.answer or 'No answer yet'[:200]}...")
                st.write(f"Time: {record.response_time:.2f}s | Cost: ${record.cost:.4f}")
                st.divider()
    else:
        st.info("No conversations yet. Ask a question in the chat!")

# Footer
st.markdown("---")
st.caption("Camera Knowledge AI Assistant")
