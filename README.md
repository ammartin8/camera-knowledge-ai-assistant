# Camera Knowledge AI Assistant

A lightweight RAG (Retrieval-Augmented Generation) application that answers camera-related questions using the Canon EOS R6 Mark II user manual as its knowledge base. This project follows LLM Zoomcamp's course structure and provides a beginner-friendly implementation of production-ready RAG patterns.

## 🎯 Problem Statement

Camera settings and documentation are notoriously complex. The Canon EOS R6 Mark II manual spans 1,086 pages of technical information, making it difficult for users to:
- Find specific information about ISO settings, aperture modes, or HDR configuration
- Understand how camera settings interact with each other
- Get quick answers while shooting without hours of manual searching
- Navigate technical jargon in official documentation

## 💡 Solution

This application transforms the 1,086-page Canon EOS R6 Mark II manual into an interactive knowledge base. Users can ask questions like:
- "What is ISO and how does it affect image quality?"
- "How do I switch aperture mode from Full to Aperture Priority?"
- "Explain HDR Mode and when to use it"
- "How do I check remaining battery level on the LCD screen?"

The assistant provides clear, grounded answers with citations from the official documentation.

## ✨ Features

### Chat Interface
- Natural language Q&A about camera settings
- Step-by-step menu navigation instructions
- Explanations in accessible language
- Citation dropdown showing source documentation links

### Monitoring Dashboard (Streamlit)
- Query volume over time
- Average response time by hour
- Token usage breakdown (prompt vs completion)
- User feedback rate (thumbs up/down)
- Retrieval success rate

### Evaluation Framework
- Hit Rate and MRR metrics for retrieval quality
- LLM-as-a-Judge for answer evaluation
- Chunking strategy comparison

## 🛠️ Technology Stack

- **Backend**: Python 3.12 + LangChain
- **LLM**: Ollama (OpenAI-compatible API)
- **Vector Store**: SQLiteSearch (MVP), PGVector (future)
- **Frontend**: Streamlit (chat + monitoring in one app)
- **Database**: PostgreSQL with pgvector extension
- **Deployment**: Docker Compose

## 📦 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Ollama running locally (or your preferred LLM endpoint)

### One-Command Setup

```bash
# Copy environment template and configure
cp .env.example .env
nano .env  # Edit with your settings

# Start all services
docker-compose up -d

# Access the application
# Chat: http://localhost:8501
# Monitoring: http://localhost:8502 (same app, use sidebar)
```

### Environment Variables

Edit `.env` to configure:

```bash
# LLM Configuration (Ollama default)
LLM_API_KEY=ollama
MODEL="llama3.2"
EMBEDDING_USE_MOCK=0
```

See `.env.example` for all available options.

## 🧪 Example Queries

Try these sample questions:
- "What is ISO and how does it affect image quality?"
- "How do I switch aperture mode from Full to Aperture Priority?"
- "Explain HDR Mode and when to use it"
- "How do I check remaining battery level on the LCD screen?"

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│   User Query    │────▶│  RAG Pipeline│────▶│  PostgreSQL  │
└─────────────────┘     └──────────────┘     └─────────────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │  Vector Search  │
                          │ (SQLite/PGVector)│
                          └─────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────┐
                    │   LLM Response + Citations│
                    └──────────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────┐
                    │  Streamlit Interface     │
                    │  (Chat + Monitoring)     │
                    └──────────────────────────┘
```

## 📁 Project Structure

```
camera-knowledge-ai-assistant/
├── app/
│   ├── src/
│   │   ├── core/              # RAG pipeline, database, LLM client
│   │   ├── embedding_generator.py    # Text-to-embeddings conversion
│   │   ├── llm_client.py     # LLM API wrapper (OpenAI/Ollama)
│   │   ├── vector_store.py   # Vector search abstraction
│   │   └── sqlite_vector_store.py  # SQLite-based vector store
│   ├── main.py               # Streamlit entry point
│   ├── load_data.py          # PDF ingestion script
│   └── Dockerfile
├── .env.example               # Environment template
├── docker-compose.yml         # Service definitions
├── pyproject.toml             # Python dependencies
└── README.md                  # This file
```

## 📈 Monitoring

The Streamlit dashboard displays:
1. Query volume over time
2. Average response time by hour
3. Token usage breakdown (prompt vs completion)
4. User feedback rate (thumbs up/down)
5. Retrieval success rate

Access at `http://localhost:8502` and navigate to the monitoring tab.

## 🚀 Future Enhancements

- [ ] PGVector for production vector storage
- [ ] Hybrid search (keyword + vector)
- [ ] Document re-ranking with Reciprocal Rank Fusion
- [ ] Query rewriting for better retrieval
- [ ] Grafana dashboard (optional upgrade)
- [ ] Multi-document support (other camera manuals)

## 🤝 Contributing

See `contributing.md` for guidelines on adding new camera models or documentation.

## 📄 License

This project is for educational purposes as part of LLM Zoomcamp Final Project.

## 🙏 Acknowledgments

- DataTalksClub for the LLM Zoomcamp course
- LangChain team for the RAG framework
