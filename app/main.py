"""Camera Knowledge AI - Main Streamlit Application.

This application provides both chat interface and monitoring dashboard
for the Camera Knowledge AI Assistant RAG system.
"""

import os
import time
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
from dataclasses import asdict
from dotenv import load_dotenv

from src.core.database import (
    get_conversations,
    get_stats,
    save_feedback,
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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "show_monitoring" not in st.session_state:
    st.session_state.show_monitoring = False
if "rag_pipeline" not in st.session_state:
    # Initialize RAG pipeline with vector store
    try:
        config = RAGPipelineConfig(num_chunks=5, model=os.getenv("MODEL", "llama3.2"))
        # Use existing vector store or create new one
        if "vector_store" in st.session_state:
            vector_store = st.session_state.vector_store
        else:
            # Use persistence so data survives between app restarts
            persist_path = os.path.join(os.path.dirname(__file__), "data", "vector_store_index.pkl")
            
            # Only auto-load PDF if persistence file doesn't exist
            try:
                from load_data import load_pdf_to_vectorstore
                app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                pdf_path = os.path.join(app_root, "data", "c012.pdf")
                
                if not os.path.exists(persist_path) and os.path.exists(pdf_path):
                    st.info(f"📄 Loading PDF: {os.path.basename(pdf_path)}...")
                    vector_store = MinsearchVectorStore(persist_path=persist_path)
                    chunks = load_pdf_to_vectorstore(pdf_path, vector_store)
                    st.success(f"✅ Loaded {chunks} chunks from PDF")
                else:
                    # Persistence exists - initialize and verify vector store
                    vector_store = MinsearchVectorStore(persist_path=persist_path)
                    
                    # Verify data was loaded
                    doc_count = len(vector_store._documents) if hasattr(vector_store, '_documents') else 0
                    if doc_count > 0:
                        st.info(f"📄 Data already loaded ({doc_count} chunks from persistence)")
                    else:
                        st.warning("⚠️ Persistence file exists but no documents loaded - reloading...")
                        # Reload PDF
                        from load_data import load_pdf_to_vectorstore
                        chunks = load_pdf_to_vectorstore(pdf_path, vector_store)
                        st.success(f"✅ Reloaded {chunks} chunks from PDF")
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
                st.code(str(e))
                # Still initialize empty vector store on error
                vector_store = MinsearchVectorStore(persist_path=persist_path)
            
            # Store it for later use
            st.session_state.vector_store = vector_store
        
        st.session_state.rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            config=config
        )
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {e}")
        st.session_state.rag_pipeline = None








# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Go to",
    ["Chat", "Monitoring"],
)

# Chat metrics sidebar (only show on chat page)
if page == "Chat":
    st.sidebar.header("📊 Query Metrics")
    # These will be updated after each query
    if "metrics" in st.session_state:
        m = st.session_state.metrics
        rt = f"{m['response_time']:.2f}s" if m and hasattr(m, 'response_time') else "N/A"
        tt = f"{m.get('total_tokens', 0):,.0f}" if m else 0
        nc = m.get("num_chunks_used") if m and hasattr(m, 'num_chunks_used') else 0
        st.metric("Response Time", rt)
        st.metric("Tokens Used", tt)
        st.metric("Chunks Retrieved", nc)

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

        # Generate response using RAG pipeline
        with st.chat_message("assistant"):
            st.markdown("⏳ Generating response...")
            
            # Initialize metrics tracking
            st.session_state.metrics = {
                "response_time": 0.0,
                "total_tokens": 0,
                "num_chunks_used": 0
            }
            
            start_time = time.time()
            cost = 0.0
            prompt_tokens = 0
            completion_tokens = 0
            query_id = None
            
            try:
                # Run RAG pipeline
                if st.session_state.rag_pipeline:
                    result = st.session_state.rag_pipeline.run(prompt)
                    
                    # Extract metrics from result
                    metadata = result.get("metadata", {})
                    prompt_tokens = metadata.get("prompt_tokens", 0)
                    completion_tokens = metadata.get("completion_tokens", 0)
                    query_id = metadata.get("query_id")
                else:
                    st.error("RAG pipeline not initialized. Please restart the app.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I'm sorry, but I'm currently unavailable. Please try again later.",
                        "citations": []
                    })
                
                # Calculate response time and cost
                end_time = time.time()
                st.session_state.metrics["response_time"] = end_time - start_time
                st.session_state.metrics["total_tokens"] = prompt_tokens + completion_tokens
                st.session_state.metrics["num_chunks_used"] = result.get("chunks_used", 0)
                
                # Calculate cost (simplified - would use real pricing in production)
                input_rate = float(os.getenv("LLM_INPUT_RATE", "0.000015"))
                output_rate = float(os.getenv("LLM_OUTPUT_RATE", "0.000050"))
                cost = (prompt_tokens * input_rate) + (completion_tokens * output_rate)
                
                # Save conversation to database using database.py helper
                try:
                    from src.core.database import get_connection
                    with get_connection().cursor() as cur:
                        cur.execute("""
                            INSERT INTO queries (
                                question, answer, model,
                                prompt_tokens, completion_tokens, total_tokens,
                                response_time_ms, cost, timestamp
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        """, (
                            prompt,
                            result.get("answer", ""),
                            MODEL,
                            prompt_tokens,
                            completion_tokens,
                            st.session_state.metrics["total_tokens"],
                            st.session_state.metrics["response_time"] * 1000,
                            cost
                        ))
                        query_id = getattr(cur, 'lastrowid', None)
                except Exception as e:
                    st.error(f"Failed to save conversation: {e}")
                
                # Save feedback if query_id exists
                if query_id is not None:
                    try:
                        from src.core.database import get_connection
                        with get_connection().cursor() as cur:
                            cur.execute(
                                "INSERT INTO feedback (query_id, source, score, timestamp) VALUES (%s, %s, 0, NOW())",
                                (query_id, "chat_init")
                            )
                    except Exception as e:
                        st.error(f"Failed to initialize feedback: {e}")
                
                # Update metrics in session state
                st.session_state.metrics["total_tokens"] = prompt_tokens + completion_tokens
                st.session_state.metrics["num_chunks_used"] = result.get("chunks_used", 0)
                
                # Display formatted answer with proper markdown
                answer_text = result.get("answer", "I couldn't find an answer in the documentation.")
                
                with st.chat_message("assistant"):
                    # Render markdown-formatted response
                    st.markdown(answer_text)
                    
                    # Display metrics below response (if available)
                    total_tokens = prompt_tokens + completion_tokens
                    if total_tokens > 0:
                        st.caption(f"⏱️ Response time: {end_time - start_time:.2f}s | 💬 Tokens: {total_tokens:,} ({prompt_tokens} in / {completion_tokens} out) | 🔗 Chunks: {result.get('chunks_used', 0)}")
                    else:
                        st.caption(f"⏱️ Response time: {end_time - start_time:.2f}s (no token data available)")
                    
                # Add assistant message to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.get("answer", "I couldn't find an answer."),
                    "citations": result.get("citations", []) if isinstance(result.get("citations"), list) else []
                })
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"I encountered an error: {str(e)}",
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
        if stats:
            col1.metric("Total conversations", stats.total or 0)
            col2.metric("Avg response time", f"{stats.avg_response_time:.2f}s" if stats.avg_response_time else "N/A")
            col3.metric("Total cost", f"${stats.total_cost:.4f}" if stats.total_cost else "$0.0000")
            col4.metric("Avg tokens", f"{stats.avg_tokens or 0:,.0f}")
        else:
            st.info("No conversations yet - ask a question in the chat!")
    except Exception as e:
        st.error(f"Could not load stats: {e}")
    
    # Feedback metrics
    try:
        from src.core.database import get_feedback_summary
        feedback = get_feedback_summary(limit=5)
        if feedback:
            st.info(f"📊 {len(feedback)} conversations with feedback")
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
            with st.expander(f"{record.prompt[:80]}..."):
                st.write(record.answer or "No answer yet")
                st.caption(f"⏱️ {record.response_time:.2f}s | 💰 ${record.cost:.4f}")
    else:
        st.info("No conversations yet. Ask a question in the chat!")





# Data Loading Button (Development only)
st.sidebar.markdown("---")
st.sidebar.subheader("🗄️  Data Management")

if st.sidebar.button("📂 Load Sample PDF Data", use_container_width=True):
    persist_path = os.path.join(os.path.dirname(__file__), "data", "vector_store_index.pkl")
    try:
        from load_data import load_pdf_to_vectorstore
        pdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "c012.pdf")
        
        if os.path.exists(pdf_path):
            chunks = load_pdf_to_vectorstore(pdf_path, st.session_state.vector_store)
            st.success(f"✅ Loaded {chunks} chunks from PDF")
        else:
            st.error("❌ PDF file not found!")
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")

# Display vector store status
if "vector_store" in st.session_state:
    vs = st.session_state.vector_store
    try:
        doc_count = len(vs._index) if hasattr(vs, '_index') else "Unknown"
        st.sidebar.info(f"📊 Vector Store Documents: {doc_count}")
    except Exception:
        st.sidebar.info("📊 Vector Store: Initialized")

# =============================================================================
