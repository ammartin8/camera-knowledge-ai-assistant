"""Vector store abstractions and implementations for RAG retrieval.

This module provides the VectorStoreInterface abstraction allowing easy swapping
between SQLiteSearch (MVP) and PGVector (production).
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from dataclasses import dataclass


@dataclass
class Document:
    """Represents a document chunk with metadata."""
    id: str
    text: str
    metadata: dict


class VectorStoreInterface(ABC):
    """Abstract interface for vector stores.

    Provides a unified API for storing and retrieving documents by similarity.
    Allows swapping between SQLiteSearch (MVP) and PGVector (production).
    """

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store with embeddings.

        Args:
            documents: List of Document objects with text and metadata
        """
        pass

    @abstractmethod
    def search(self, query: str, num_results: int = 5) -> List[Document]:
        """Search for similar documents by query.

        Args:
            query: Search query string
            num_results: Number of results to return

        Returns:
            List of most similar Document objects
        """
        raise NotImplementedError("Subclasses must implement search()")

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete
        """
        raise NotImplementedError("Subclasses must implement delete()")


# Placeholder implementation for testing without external dependencies
class MockVectorStore(VectorStoreInterface):
    """Mock vector store for development/testing."""

    def __init__(self):
        self._documents: Dict[str, Document] = {}  # id -> Document

    def add_documents(self, documents: List[Document]) -> None:
        for doc in documents:
            self._documents[doc.id] = doc

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        # Return all documents as fallback (simple keyword match in text)
        return [self._documents[id_] for id_ in list(self._documents.keys())[:num_results]]

    def delete(self, ids: List[str]) -> None:
        for id_ in ids:
            self._documents.pop(id_, None)


# Placeholder implementation for testing without external dependencies
class MockEmbeddingGenerator:
    """Mock embedding generator for development/testing."""

    @staticmethod
    def generate_embedding(text: str, dimension: int = 1536) -> List[float]:
        """Generate a mock embedding (returns zeros)."""
        return [0.0] * dimension


class MockVectorStoreWithEmbeddings(VectorStoreInterface):
    """Mock vector store with basic embedding simulation."""

    def __init__(self, dimension: int = 1536):
        self._documents: Dict[str, Document] = {}
        self._dimension = dimension
        self._embeddings: Dict[str, List[float]] = {}

    def add_documents(self, documents: List[Document]) -> None:
        for doc in documents:
            self._documents[doc.id] = doc
            # Mock embedding (zeros)
            self._embeddings[doc.id] = [0.0] * self._dimension

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        return [self._documents[id_] for id_ in list(self._documents.keys())[:num_results]]

    def delete(self, ids: List[str]) -> None:
        for id_ in ids:
            self._documents.pop(id_, None)
            self._embeddings.pop(id_, None)


__all__ = [
    "VectorStoreInterface",
    "Document",
    "MockVectorStore",
    "MockEmbeddingGenerator",
    "MockVectorStoreWithEmbeddings",
]
