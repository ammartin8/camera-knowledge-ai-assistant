"""Persistent vector search index using SQLite with numpy embeddings."""

import json
import os
import math
import sqlite3
import tempfile
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Document:
    """Represents a document chunk with metadata."""
    id: str
    text: str
    embedding: List[float]
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PersistentVectorStoreIndex:
    """Persistent vector store using SQLite for storage."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize persistent vector store."""
        self.db_path = db_path or tempfile.mktemp(suffix=".db")
        if not os.path.isabs(self.db_path):
            self.db_path = os.path.join(os.getcwd(), self.db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY, text TEXT NOT NULL, 
            embedding REAL[] NOT NULL, metadata TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_text ON embeddings(text)")
        conn.commit()
        conn.close()
        print(f"Initialized vector store at: {self.db_path}")

    def add_documents(self, documents: List[Document]) -> int:
        """Add documents with their embeddings to the store."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        added_count = 0
        for doc in documents:
            try:
                embedding_str = ' '.join(str(x) for x in doc.embedding)
                cursor.execute(
                    "INSERT OR REPLACE INTO embeddings (id, text, embedding, metadata) VALUES (?, ?, ?, ?)",
                    (doc.id, doc.text, embedding_str, self._serialize_metadata(doc.metadata)))
                added_count += 1
            except Exception as e:
                print(f"Error adding document {doc.id}: {e}")
        conn.commit()
        conn.close()
        print(f"Added {added_count} documents to vector store")
        return added_count

    def search(self, query_embedding: List[float], num_results: int = 5) -> List[Document]:
        """Search for similar documents using cosine similarity."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, text, embedding, metadata FROM embeddings')
        rows = cursor.fetchall()
        results = []
        for row in rows:
            doc_id, text, embedding_str, metadata_str = row
            embedding_vec = [float(x) for x in embedding_str.split()]
            similarity = self._cosine_similarity(query_embedding, embedding_vec)
            results.append({
                'id': doc_id, 'text': text, 'embedding': embedding_vec,
                'similarity': similarity, 'metadata': self._deserialize_metadata(metadata_str)})
        results.sort(key=lambda x: x['similarity'], reverse=True)
        conn.close()
        print(f"Found {len(results)} results for query")
        return [Document(id=r['id'], text=r['text'], embedding=r['embedding'], metadata=r['metadata']) for r in results[:num_results]]

    def search_by_keyword(self, query: str, num_results: int = 5) -> List[Document]:
        """Search using keyword matching."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, text, embedding, metadata FROM embeddings WHERE LOWER(text) LIKE LOWER(?)", (f'%{query}%',))
        rows = cursor.fetchall()
        results = []
        for row in rows:
            doc_id, text, embedding_str, metadata_str = row
            embedding_vec = [float(x) for x in embedding_str.split()]
            results.append({
                'id': doc_id, 'text': text, 'embedding': embedding_vec,
                'similarity': 1.0, 'metadata': self._deserialize_metadata(metadata_str)})
        conn.close()
        return [Document(id=r['id'], text=r['text'], embedding=r['embedding'], metadata=r['metadata']) for r in results[:num_results]]

    def delete(self, ids: List[str]) -> int:
        """Delete documents by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        placeholders = ','.join('?' * len(ids))
        cursor.execute(f'DELETE FROM embeddings WHERE id IN ({placeholders})', ids)
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        print(f"Deleted {deleted_count} documents")
        return deleted_count

    def get_all_documents(self) -> List[Document]:
        """Get all documents from the store."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, text, embedding, metadata FROM embeddings')
        rows = cursor.fetchall()
        documents = []
        for row in rows:
            doc_id, text, embedding_str, metadata_str = row
            embedding_vec = [float(x) for x in embedding_str.split()]
            documents.append(Document(id=doc_id, text=text, embedding=embedding_vec, metadata=self._deserialize_metadata(metadata_str)))
        conn.close()
        return documents

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def _serialize_metadata(self, metadata: dict) -> str:
        """Serialize metadata to JSON string."""
        return json.dumps(metadata or {}, ensure_ascii=False)

    def _deserialize_metadata(self, metadata_str: str) -> dict:
        """Deserialize metadata from JSON string."""
        if not metadata_str:
            return {}
        try:
            if metadata_str.startswith('['):
                return json.loads(metadata_str)
            return json.loads(metadata_str)
        except (json.JSONDecodeError, TypeError):
            return {}

    def get_db_path(self) -> str:
        """Get the path to the database file."""
        return self.db_path


class InMemoryVectorStoreIndex:
    """In-memory vector store for development and testing."""

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def __init__(self):
        self._documents = {}

    def add_documents(self, documents: List[Document]) -> int:
        """Add documents to the store."""
        count = 0
        for doc in documents:
            self._documents[doc.id] = doc
            count += 1
        return count

    def search(self, query_embedding: List[float], num_results: int = 5) -> List[Document]:
        """Search using cosine similarity."""
        results = []
        for doc in self._documents.values():
            similarity = self._cosine_similarity(query_embedding, doc.embedding)
            results.append((doc, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in results[:num_results]]

    def delete(self, ids: List[str]) -> int:
        """Delete documents by ID."""
        count = 0
        for doc_id in ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
                count += 1
        return count

    def get_all_documents(self) -> List[Document]:
        """Get all documents."""
        return list(self._documents.values())


__all__ = ["PersistentVectorStoreIndex", "InMemoryVectorStoreIndex", "Document"]
