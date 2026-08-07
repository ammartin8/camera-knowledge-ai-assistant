"""SQLite-based vector store implementation using LangChain's SQLiteSearch.

This provides a lightweight, file-based vector store that doesn't require
PostgreSQL setup - perfect for MVP development and testing.
"""

from typing import List, Optional, TYPE_CHECKING
from dataclasses import dataclass
import os
import tempfile
import sqlite3

if TYPE_CHECKING:
    from langchain_core.documents import Document as LCDocument
    class SQLiteSearch:  # type: ignore
        @staticmethod
        def from_texts(texts, db_path, embedding_function=None):
            pass
else:
    try:
        from langchain_community.vectorstores import SQLiteSearch
        from langchain_core.documents import Document as LCDocument
    except ImportError:  # type: ignore
        print("Warning: langchain-community not installed. Using mock implementation.")
        class SQLiteSearch:  # type: ignore
            def __init__(self, *args, **kwargs):
                pass
            def add_documents(self, documents):  # type: ignore
                pass
            @staticmethod
            def similarity_search(vectorstore, query, k=5):
                return []


@dataclass
class Document:
    """Represents a document chunk with metadata."""
    id: str
    text: str
    metadata: dict


class SQLiteVectorStore:
    """SQLite-based vector store using LangChain's SQLiteSearch.

    Provides keyword-based search in a file-based database.
    Can be swapped for PGVector implementation later without changing the interface.

    Note: This is a simple keyword search implementation, not true vector search.
    For semantic search, use embeddings with pgvector or another vector database.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or tempfile.mktemp(suffix=".db")
        # Ensure we're using an absolute path
        if not os.path.isabs(self.db_path):
            self.db_path = os.path.join(os.getcwd(), self.db_path)
        
        self.vectorstore: Optional["SQLiteSearch"] = None

    def _initialize(self) -> None:
        """Initialize the SQLiteSearch vector store."""
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        try:
            # Initialize with empty documents (will be populated when add_documents is called)
            from langchain_community.vectorstores import SQLiteSearch as LangChainSQLiteSearch  # type: ignore
            self.vectorstore = LangChainSQLiteSearch.from_texts(  # type: ignore
                [],
                self.db_path,
                embedding_function=None  # type: ignore
            )
            print(f"Initialized SQLite vector store at: {self.db_path}")
        except Exception as e:
            print(f"Failed to initialize SQLite vector store: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of Document objects with text and metadata
        """
        if not self.vectorstore:
            self._initialize()
        
        # Convert our Document format to LangChain's LCDocument format
        lc_documents = [  # type: ignore
            LCDocument(
                page_content=doc.text,
                metadata={
                    "id": doc.id,
                    **doc.metadata
                }
            ) for doc in documents
        ]
        
        if self.vectorstore:
            try:
                self.vectorstore.add_documents(lc_documents)  # type: ignore
                print(f"Added {len(documents)} documents to vector store")
            except Exception as e:
                print(f"Failed to add documents: {e}")
                raise

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        """Search for similar documents by keyword.

        Args:
            query: Search query string
            num_results: Number of results to return

        Returns:
            List of most similar Document objects
        """
        if not self.vectorstore:
            self._initialize()
        
        try:
            # Perform search and convert results back to our Document format
            from langchain_community.vectorstores import SQLiteSearch as LangChainSQLiteSearch
            results = LangChainSQLiteSearch.similarity_search(  # type: ignore
                self.vectorstore, query, k=num_results
            )
            
            converted_results = [
                Document(
                    id=doc.metadata.get("id", ""),
                    text=doc.page_content,
                    metadata=doc.metadata
                ) for doc in results
            ]
            
            print(f"Found {len(converted_results)} results for query: {query}")
            return converted_results
        except Exception as e:
            print(f"Search failed: {e}")
            return []

    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID.

        Note: SQLiteSearch doesn't provide a direct delete method.
        This is a placeholder for future implementation.

        Args:
            ids: List of document IDs to delete
        """
        print(f"Delete not fully implemented for SQLiteSearch. Would delete: {ids}")
        # In production, you'd need to either:
        # 1. Delete the entire database and reinitialize
        # 2. Use a different vector store that supports deletion

    def get_db_path(self) -> str:
        """Get the path to the SQLite database file."""
        return self.db_path


class MockSQLiteVectorStore:
    """Mock implementation for development without LangChain dependencies."""

    def __init__(self):
        self._documents = {}  # id -> (text, metadata)

    def add_documents(self, documents) -> None:
        """Add documents to the mock store."""
        for doc in documents:
            self._documents[doc.id] = (doc.text, doc.metadata)

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        """Search using simple keyword matching."""
        # Simple case-insensitive search
        query_lower = query.lower()
        
        matches = []
        for doc_id, (text, metadata) in self._documents.items():
            if query_lower in text.lower():
                matches.append(Document(
                    id=doc_id,
                    text=text,
                    metadata=metadata
                ))
        
        # Return top results
        return matches[:num_results]

    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        for doc_id in ids:
            self._documents.pop(doc_id, None)


__all__ = [
    "Document",
    "SQLiteVectorStore",
    "MockSQLiteVectorStore",
]
