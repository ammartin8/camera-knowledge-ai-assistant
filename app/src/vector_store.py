"""Vector store abstractions and implementations for RAG retrieval.

This module provides the VectorStoreInterface abstraction allowing easy swapping
between Minsearch (MVP - keyword search) and PGVector (production - vector search).

Minsearch is chosen for MVP because:
- Built-in TF-IDF keyword search with stemming
- Simple in-memory storage
- No external dependencies beyond minsearch library
- Follows LLM Zoomcamp patterns more closely
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import pickle


@dataclass
class Document:
    """Represents a document chunk with metadata.

    Attributes:
        id: Unique identifier for the chunk
        text: The main text content of the chunk
        metadata: Dictionary containing chunk context (source, page, section, etc.)
    """
    id: str
    text: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class VectorStoreInterface(ABC):
    """Abstract interface for vector stores.

    Provides a unified API for storing and retrieving documents by similarity.
    Allows swapping between Minsearch (MVP) and PGVector (production).
    """

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> int:
        """Add documents to the index.

        Args:
            documents: List of Document objects with text and metadata

        Returns:
            Number of documents added
        """
        pass

    @abstractmethod
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents by query.

        Args:
            query: Search query string
            num_results: Number of results to return

        Returns:
            List of dicts with document content and metadata
        """
        raise NotImplementedError("Subclasses must implement search()")

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete
        """
        raise NotImplementedError("Subclasses must implement delete()")


class MinsearchVectorStore(VectorStoreInterface):
    """Minsearch-based keyword search store for MVP implementation.

    Uses minsearch library for TF-IDF keyword search with stemming and 
    metadata field indexing. Provides hybrid retrieval combining:
    - Keyword matching (TF-IDF) for exact term retrieval
    - Metadata filtering by chapter, section, source, etc.
    """

    def __init__(self, index_fields: Optional[List[str]] = None,
                 persist_path: str = None):
        """Initialize Minsearch keyword search store."""
        from minsearch import Index

        # Initialize minsearch Index with text and keyword fields
        self.index = Index(
            text_fields=["text"],
            keyword_fields=index_fields or [],
        )

        # Store documents separately for metadata access
        self._documents: Dict[str, Document] = {}

        # Optional persistence using pickle (only stores documents, not index)
        if persist_path:
            self.persist_path = persist_path
            self.load_documents_from_disk()
            
            # Rebuild index with ALL loaded documents (both from disk and any in memory)
            self._rebuild_index()
    
    def load_documents_from_disk(self):
        """Load persisted documents from disk (index is rebuilt at init time)."""
        import os
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)
            
            # Load documents into memory
            saved_docs = data.get('documents', {})
            self._documents.update(saved_docs)
            
        except Exception as e:
            print(f"Warning: Failed to load persisted data: {e}")
    
    def _rebuild_index(self):
        """Rebuild search index with ALL documents (called once at init)."""
        
        # Convert all documents to list of dicts for minsearch Index.fit()
        docs_list = [
            {
                'text': doc.text,
                **{k: v for k, v in doc.metadata.items()}
            }
            for doc in self._documents.values()
        ]
        
        if docs_list:
            # Build the index with ALL documents
            self.index.fit(docs_list)
    
    def save_to_disk(self):
        """Save index and documents to disk."""
        if not hasattr(self, 'persist_path'):
            return
        try:
            index_data = []
            for doc in self._documents.values():
                index_data.append({
                    'text': doc.text,
                    **{k: v for k, v in doc.metadata.items() if k in self.index.keyword_fields}
                })
            data = {
                'index_data': index_data,
                'documents': {doc.id: doc for doc in self._documents.values()}
            }
            with open(self.persist_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception:
            pass

    def add_documents(self, documents: List[Document]) -> int:
        """Add documents to the vector store.
        
        Note: Documents are stored in memory but NOT added to search index immediately.
        The index is rebuilt at app startup with all loaded documents.
        For new documents to be searchable, restart the application.

        Args:
            documents: List of Document objects with text and metadata

        Returns:
            Number of documents added
        """
        # Store document references in memory
        for doc in documents:
            self._documents[doc.id] = doc
        
        count = len(documents)
        
        # Save to disk if persistence is enabled
        if hasattr(self, 'persist_path'):
            self.save_to_disk()
        
        return count

    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents using TF-IDF keyword matching."""
        
        # Perform search with boost for text field
        results = self.index.search(
            query,
            boost_dict={"text": 3.0},
            num_results=num_results,
        )

        # Convert minsearch results to our document format
        
        doc_results = []
        for i, result in enumerate(results):
            if isinstance(result, dict):
                # Extract text and metadata from result
                text = result.get('text', '')
                metadata = {k: v for k, v in result.items() if k != 'text'}
                
                doc_results.append({
                    'text': text,
                    'similarity': 0.0,
                    'metadata': metadata,
                })
            else:
                continue

        return doc_results

    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete
        """
        for id_ in ids:
            self._documents.pop(id_, None)



__all__ = [
    "VectorStoreInterface",
    "Document",
    "MinsearchVectorStore",  # MVP - keyword search (TF-IDF)
]
