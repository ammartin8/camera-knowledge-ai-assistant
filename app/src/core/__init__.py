"""Core module utilities and database helpers."""

from .database import (
    get_connection,
    create_schema,
    save_feedback,
    get_feedback,
    get_conversations,
    get_stats,
    get_feedback_summary,
    calculate_cost,
    Stats,
    track_llm_call,
)
from .rag_pipeline import RAGPipeline, RAGPipelineConfig

# Import new vector search modules
try:
    from ..embedding_generator import EmbeddingGenerator, NomicEmbeddingGenerator, MxbaiEmbeddingGenerator
    from ..vector_search_index import PersistentVectorStoreIndex, InMemoryVectorStoreIndex
except ImportError:
    print("Warning: Vector search modules not available")

__all__ = [
    "get_connection",
    "create_schema",
    "save_feedback",
    "get_feedback",
    "get_conversations",
    "get_stats",
    "get_feedback_summary",
    "calculate_cost",
    "Stats",
    "track_llm_call",
    # RAG Pipeline
    "RAGPipeline",
    "RAGPipelineConfig",
    # Vector search exports
    "EmbeddingGenerator",
    "NomicEmbeddingGenerator",
    "MxbaiEmbeddingGenerator",
    "PersistentVectorStoreIndex",
    "InMemoryVectorStoreIndex",
]

