"""Camera Knowledge AI - Main Streamlit Application."""

import os
import sys
import time
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import asdict
from dotenv import load_dotenv

# Core imports
from src.core.database import (
    get_stats,
    save_feedback,
)
from src.monitoring_queries import (
    get_query_volume_by_date,
    get_response_time_by_hour,
    get_token_usage_breakdown,
    get_feedback_rate,
    get_retrieval_success_rate,
    get_aggregate_stats,
    get_recent_conversations,
)
from src.core.rag_pipeline import RAGPipeline, RAGPipelineConfig
from src.vector_store import MinsearchVectorStore

# Load environment variables
load_dotenv()

# Configuration from environment
LLM_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("LLM_API_KEY", "ollama")
MODEL = os.getenv("LLM_MODEL", "llama3.2")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "camera_knowledge_db")

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", ["Chat", "Monitoring"])

# Chat page
if page == "Chat":
    st.title("Camera Knowledge Assistant")
    
    st.markdown("""
    Ask questions about the Canon EOS R6 Mark II camera!
    
    **Example questions:**
    - What is ISO and how does it affect image quality?
    - How do I switch aperture mode from Full to Aperture Priority?
    - Explain HDR Mode and when to use it
    - How do I check remaining battery level on the LCD screen?
    """)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your camera..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ""
                metrics_info = {}
                start_time = time.time()
                
                # Initialize vector store and RAG pipeline
                persist_path = os.path.join(os.path.dirname(__file__), "data", "vector_store_index.pkl")
                pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "c012.pdf")
                
                # Ensure proper Python path for loading vector store
                app_root = os.path.dirname(os.path.dirname(__file__))
                sys.path.insert(0, app_root)
                sys.path.insert(0, os.path.dirname(__file__))
                
                try:
                    # Initialize or load vector store
                    if "vector_store" not in st.session_state:
                        st.session_state.vector_store = MinsearchVectorStore(persist_path=persist_path)
                    
                    # Create RAG pipeline
                    config = RAGPipelineConfig(num_chunks=5, model=MODEL)
                    st.session_state.rag_pipeline = RAGPipeline(
                        vector_store=st.session_state.vector_store,
                        config=config
                    )
                    
                    # Get response with metrics
                    try:
                        result = st.session_state.rag_pipeline.run(prompt)
                    except Exception as inner_e:
                        raise inner_e  # Re-raise to be caught by outer except
                    
                    # Debug: Print result info
                    print(f"\n=== RESULT INFO ===")
                    print(f"Result type: {type(result)}")
                    print(f"Result is None: {result is None}")
                    print(f"Result is empty: {not result}")
                    if isinstance(result, dict):
                        print(f"Result keys: {list(result.keys())}")
                        print(f"Has 'answer': {'answer' in result}")
                        if 'answer' in result:
                            print(f"Answer preview: {str(result['answer'])[:100]}")
                    elif hasattr(result, '__len__'):
                        print(f"Result length: {len(result)}")
                    
                    if not result:
                        response = "No answer found"
                        duration_ms = (time.time() - start_time) * 1000
                        duration_sec = duration_ms / 1000
                        metrics_info["response_time"] = f"{duration_sec:.2f}s ({duration_ms:.1f}ms)"
                        metrics_info["chunks_retrieved"] = 0
                        metrics_info["prompt_tokens"] = "N/A"
                        metrics_info["completion_tokens"] = "N/A"
                        metrics_info["total_tokens"] = "N/A"
                    else:
                        # Safely extract response and metrics from result
                        if isinstance(result, dict):
                            response = result.get("answer", "No answer found")
                            chunks_used = result.get("chunks_used", 0)
                            prompt_tokens_val = result.get("prompt_tokens") or "N/A"
                            completion_tokens_val = result.get("completion_tokens") or "N/A"
                            total_tokens_val = result.get("total_tokens") or "N/A"
                        elif hasattr(result, 'get'):
                            response = result.get("answer", str(result))
                            chunks_used = result.get("chunks_used", 0)
                            prompt_tokens_val = result.get("prompt_tokens") or "N/A"
                            completion_tokens_val = result.get("completion_tokens") or "N/A"
                            total_tokens_val = result.get("total_tokens") or "N/A"
                        else:
                            response = str(result) if result else "No answer found"
                            chunks_used = 0
                            prompt_tokens_val = "N/A"
                            completion_tokens_val = "N/A"
                            total_tokens_val = "N/A"
                        
                        # Extract metrics from result
                        duration_ms = (time.time() - start_time) * 1000
                        duration_sec = duration_ms / 1000
                        metrics_info["response_time"] = f"{duration_sec:.2f}s ({duration_ms:.1f}ms)"
                        
                        metrics_info["chunks_retrieved"] = chunks_used
                        metrics_info["prompt_tokens"] = prompt_tokens_val
                        metrics_info["completion_tokens"] = completion_tokens_val
                        metrics_info["total_tokens"] = total_tokens_val
                    
                except Exception as e:
                    response = f"Error: {str(e)}"
                    metrics_info["error"] = str(e)
                
                # Add assistant response to chat
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                    
                    # Display metrics below response
                    if not metrics_info.get("error"):
                        st.caption(
                            f"⏱️ **Response Time:** {metrics_info['response_time']} | "
                            f"📊 **Tokens:** Prompt: {metrics_info['prompt_tokens']}, Completion: {metrics_info['completion_tokens']}, Total: {metrics_info['total_tokens']} | "
                            f"🔍 **Chunks Retrieved:** {metrics_info['chunks_retrieved']}"
                        )

# Monitoring page  
elif page == "Monitoring":
    st.title("Monitoring Dashboard")
    
    # Initialize database connection for monitoring
    try:
        from src.core.database import get_connection
        conn = get_connection()
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        conn = None
    
    if conn:
        # Aggregate stats (KPIs) - moved to top
        try:
            stats = get_aggregate_stats()
            if stats:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Queries", stats.get("total_queries", 0))
                with col2:
                    avg_time_sec = stats.get('avg_response_time_ms', 0) / 1000
                    st.metric("Avg Response Time", f"{avg_time_sec:.2f}s")
                with col3:
                    st.metric("Total Tokens", stats.get("total_tokens", 0))
        except Exception as e:
            st.error(f"Failed to load aggregate stats: {e}")
        
        # Query volume by date
        try:
            result = get_query_volume_by_date()
            if result.get('date') and result.get('query_count'):
                df = pd.DataFrame(result)
                fig = px.line(df, x="date", y="query_count", title="Query Volume Over Time")
                st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.error(f"Failed to load query volume: {e}")
        
        # Response time by hour
        try:
            result = get_response_time_by_hour()
            if result.get('hour') and result.get('avg_response_time_ms'):
                df = pd.DataFrame(result)
                fig = px.bar(df, x="hour", y="avg_response_time_ms", title="Response Time by Hour")
                st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.error(f"Failed to load response times: {e}")
        
        # Token usage breakdown (pie chart)
        try:
            result = get_token_usage_breakdown()
            if isinstance(result, list) and len(result) >= 4:
                labels = [result[0], result[2]]
                values = [result[1], result[3]]
                fig = px.pie(values=values, names=labels, title="Token Usage Breakdown")
                st.plotly_chart(fig, width="stretch")
            elif isinstance(result, dict) and 'category' in result:
                df = pd.DataFrame(result)
                fig = px.pie(df, names="category", values="token_count")
                st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.error(f"Failed to load token usage: {e}")
        except Exception as e:
            st.error(f"Failed to load stats: {e}")
        
        # Close connection
        conn.close()

