"""Camera Knowledge AI - Main Streamlit Application.

This application provides both chat interface and monitoring dashboard
for the Camera Knowledge AI Assistant RAG system.
"""

import os
import time
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import asdict
from dotenv import load_dotenv

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
                                course, instructions, prompt,
                                prompt_tokens, completion_tokens, total_tokens,
                                response_time_ms, cost, timestamp
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        """, (
                            prompt,
                            result.get("answer", ""),
                            MODEL,
                            'camera_knowledge',  # course - default value
                            None,  # instructions - may be empty
                            '',  # prompt - user's original question
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
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    end_date = datetime.now()
    max_days = st.sidebar.slider("Date Range (days)", min_value=1, max_value=30, value=7)
    start_date = end_date - timedelta(days=max_days)
    
    query_type = st.sidebar.selectbox(
        "Query Type Filter",
        ["All", "Camera Setup", "Photography Techniques", "Maintenance", "Settings"],
        index=0
    )
    
    # Aggregate stats
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        stats = get_aggregate_stats(start_date=start_date, end_date=end_date, limit_days=max_days)
        if stats and stats["total_queries"] > 0:
            col1.metric("Total conversations", stats["total_queries"])
            col2.metric("Avg response time", f"{stats['avg_response_time_ms']:.2f}s")
            col3.metric("Total cost", f"${stats['total_cost']:.4f}")
            col4.metric("Avg tokens", f"{stats['avg_total_tokens']:,.0f}")
        else:
            st.info("No conversations yet - ask a question in the chat!")
    except Exception as e:
        st.error(f"Could not load stats: {e}")
        stats = {}
    
    # Feedback metrics
    try:
        from src.core.database import get_feedback_summary
        feedback = get_feedback_summary(limit=5)
        if feedback:
            st.info(f"📊 {len(feedback)} conversations with feedback")
    except Exception as e:
        pass
    
    # 1. Query volume over time (line chart)
    st.subheader("📈 Query Volume Over Time")
    try:
        date_list = []
        count_list = []
        
        results = get_query_volume_by_date(
            start_date=start_date,
            end_date=end_date,
            limit_days=max_days
        )
        
        # Results is a dict with 'date' and 'query_count' keys, each containing lists
        if isinstance(results, dict) and "date" in results:
            date_list = results.get("date", [])
            count_list = results.get("query_count", [])
        else:
            st.error(f"Could not load query volume: Invalid response format")
        
        if len(date_list) > 0:
            # Create DataFrame with date as index for Plotly line chart
            df = pd.DataFrame({"date": date_list, "query_count": count_list})
            st.line_chart(df.set_index("date"))
        else:
            st.info("No query volume data available for selected date range.")
    except Exception as e:
        st.error(f"Could not load query volume: {e}")
    
    # 2. Average response time by hour (bar chart)
    st.subheader("⏱️ Average Response Time by Hour")
    try:
        hour_list = []
        rt_list = []
        
        results = get_response_time_by_hour(
            start_date=start_date,
            end_date=end_date,
            limit_days=max_days
        )
        
        # Results is a dict with 'hour' and 'avg_response_time_ms' keys, each containing lists
        if isinstance(results, dict) and "hour" in results:
            hour_list = results.get("hour", [])
            rt_list = results.get("avg_response_time_ms", [])
        else:
            st.error(f"Could not load response time data: Invalid response format")
        
        if len(hour_list) > 0:
            # Create DataFrame with hour as index for Plotly bar chart
            df = pd.DataFrame({"hour": hour_list, "avg_response_time_ms": rt_list})
            st.bar_chart(df.set_index("hour"))
        else:
            st.info("No response time data available for selected date range.")
    except Exception as e:
        st.error(f"Could not load response time data: {e}")
    
    # 3. Token usage breakdown (pie chart)
    st.subheader("🔁 Token Usage Breakdown")
    try:
        tokens = get_token_usage_breakdown(
            start_date=start_date,
            end_date=end_date,
            limit_days=max_days
        )
        if tokens["total_tokens"] > 0:
            st.plotly_chart(
                px.pie(
                    values=[tokens['prompt_tokens'], tokens['completion_tokens']],
                    names=['Prompt', 'Completion'],
                    title='Token Distribution'
                ),
                width='stretch'
            )
            
            # Display token counts
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prompt Tokens", f"{tokens['prompt_tokens']:,.0f}")
            with col2:
                st.metric("Completion Tokens", f"{tokens['completion_tokens']:,.0f}")
            with col3:
                st.metric("Total Tokens", f"{tokens['total_tokens']:,.0f}")
        else:
            st.info("No token data available for selected date range.")
    except Exception as e:
        st.error(f"Could not load token usage: {e}")
    
    # 4. User feedback rate (gauge/bars)
    st.subheader("👍 User Feedback Rate")
    try:
        from src.core.database import get_feedback_summary
        feedback_data = get_feedback_rate(
            start_date=start_date,
            end_date=end_date,
            limit_days=max_days
        )
        if feedback_data["total_feedback"] > 0:
            # Create bar chart for feedback
            feedback_df = pd.DataFrame([
                {"label": "Thumbs Up", "value": feedback_data['up_count']},
                {"label": "Thumbs Down", "value": feedback_data['down_count']},
                {"label": "Neutral", "value": feedback_data['neutral_count']}
            ])
            
            fig = go.Figure(data=[go.Bar(
                x=feedback_df["label"],
                y=feedback_df["value"]
            )])
            fig.update_layout(title="Feedback Distribution")
            st.plotly_chart(fig, width='stretch')
            
            # Display satisfaction rate
            sat_rate = feedback_data.get('satisfaction_rate', 0)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Satisfaction Rate", f"{sat_rate:.1f}%")
            with col2:
                st.metric("Total Feedback", feedback_data['total_feedback'])
        else:
            st.info("No feedback data available for selected date range.")
    except Exception as e:
        st.error(f"Could not load feedback data: {e}")
    
    # 5. Retrieval success rate (metric card)
    st.subheader("🔍 Retrieval Success Rate")
    try:
        retrieval = get_retrieval_success_rate(
            start_date=start_date,
            end_date=end_date,
            limit_days=max_days
        )
        if retrieval["total_queries"] > 0:
            success_rate = retrieval.get('success_rate', 0)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Success Rate", f"{success_rate:.1f}%")
            with col2:
                st.metric("Total Retrieved", retrieval['successful_retrievals'])
        else:
            st.info("No retrieval data available for selected date range.")
    except Exception as e:
        st.error(f"Could not load retrieval data: {e}")
    
    # Recent conversations
    st.subheader("📝 Recent Conversations")
    try:
        records = get_recent_conversations(limit=20)
        if records and len(records) > 0:
            # records are already dicts, no need for asdict()
            df = pd.DataFrame(records)
            
            # Display as expandable cards
            for _, record in df.iterrows():
                with st.expander(f"{record['question'][:100]}..."):
                    st.markdown(record.get('answer', 'No answer yet') or "")
                    if record.get('response_time_ms'):
                        st.caption(
                            f"⏱️ {record['response_time_ms']/1000:.2f}s | "
                            f"💰 ${record.get('cost', 0):.4f} | "
                            f"🔗 Tokens: {record.get('total_tokens', 0):,}"
                        )
        else:
            st.info("No conversations available.")
    except Exception as e:
        st.error(f"Could not load conversations: {e}")





# Data Loading Button (Development only)
st.sidebar.markdown("---")
st.sidebar.subheader("🗄️  Data Management")

if st.sidebar.button("📂 Load Sample PDF Data", width='stretch'):
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
