"""PDF ingestion pipeline abstractions and implementations for RAG.

This module provides the IngestionPipelineInterface abstraction allowing easy swapping
between different PDF processing strategies (fixed-size, semantic, hybrid).
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
import re


@dataclass
class Chunk:
    """Represents a text chunk extracted from a document."""
    id: str
    text: str
    source: str  # Document filename
    page: int
    start_offset: int
    end_offset: int
    metadata: dict


class IngestionPipelineInterface(ABC):
    """Abstract interface for document ingestion pipelines.

    Provides a unified API for extracting, chunking, and preparing documents
    for the RAG system. Allows swapping between different chunking strategies.
    """

    @abstractmethod
    def extract_text(self, filepath: str) -> str:
        """Extract raw text from a document file.

        Args:
            filepath: Path to the PDF file

        Returns:
            Raw extracted text string
        """
        pass

    @abstractmethod
    def chunk_documents(self, text: str, source: str, page: int = 1) -> List[Chunk]:
        """Split raw text into manageable chunks.

        Args:
            text: Raw extracted text
            source: Document filename/source identifier
            page: Page number (for multi-page documents)

        Returns:
            List of Chunk objects
        """
        pass

    @abstractmethod
    def build_index(self, filepath: str) -> List[Chunk]:
        """Complete ingestion: extract and chunk a document.

        Args:
            filepath: Path to the PDF file

        Returns:
            List of processed Chunk objects ready for embedding
        """
        pass


class FixedSizeChunker:
    """Simple fixed-size text chunking strategy.

    Splits text into chunks of approximately specified size,
    with optional overlap between chunks for context preservation.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> List[str]:
        """Split text into chunks of specified size."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.overlap
        return chunks


class SemanticChunker:
    """Semantic chunking by section headers.

    Splits text at natural boundaries like section headers,
    preserving document structure and improving retrieval relevance.
    """

    def __init__(self):
        # Common section header patterns in camera manuals
        self.header_patterns = [
            r'^\s*(Chapter|Section|Part)\s*[\d\.]+?\s*:',
            r'^\s*[A-Z][a-z]+\s+Settings',
            r'^\s*[A-Z][a-z]+\s+Mode',
            r'^\s*[A-Z][a-z]+\s+Function',
            r'^\s*How\s+to\s+',
        ]

    def split(self, text: str) -> List[str]:
        """Split text at section headers."""
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_header: Optional[str] = None

        for line in text.split('\n'):
            is_header = any(re.match(pattern, line.strip(), re.IGNORECASE)
                          for pattern in self.header_patterns)
            
            if is_header:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [line]
                else:
                    current_chunk = [line]
                    current_header = line
            else:
                current_chunk.append(line)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks


class BasicPDFIngestor(IngestionPipelineInterface):
    """Basic PDF ingestion using pdfplumber.

    Extracts text from PDF files and applies fixed-size chunking.
    Simple, beginner-friendly implementation following LLM Zoomcamp patterns.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
        import pdfplumber

    def extract_text(self, filepath: str) -> str:
        """Extract raw text from PDF using pdfplumber."""
        import pdfplumber
        
        with pdfplumber.open(filepath) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append(f"[Page {i+1}]\n{text}")
            
            return '\n\n'.join(pages)

    def chunk_documents(self, text: str, source: str, page: int = 1) -> List[Chunk]:
        """Split extracted text into fixed-size chunks."""
        import hashlib
        
        chunks_list = self.chunker.split(text)
        
        # Generate unique IDs for each chunk
        chunks = []
        for i, chunk_text in enumerate(chunks_list):
            chunk_id = f"{source}_{page}_{i}"
            hash_id = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
            
            chunk = Chunk(
                id=f"{hash_id}_{chunk_id}",
                text=chunk_text.strip(),
                source=source,
                page=page,
                start_offset=0,  # Would calculate actual offset in production
                end_offset=len(chunk_text),
                metadata={
                    "original_length": len(text),
                    "chunk_size": self.chunker.chunk_size,
                    "overlap": self.chunker.overlap,
                }
            )
            chunks.append(chunk)
        
        return chunks

    def build_index(self, filepath: str) -> List[Chunk]:
        """Complete ingestion pipeline for a PDF file."""
        text = self.extract_text(filepath)
        return self.chunk_documents(text, source=filepath.split('/')[-1])


class SemanticPDFIngestor(IngestionPipelineInterface):
    """Semantic PDF ingestion using pdfplumber.

    Extracts text from PDF files and applies semantic chunking
    based on section headers for better retrieval relevance.
    """

    def __init__(self):
        self.chunker = SemanticChunker()
        import pdfplumber

    def extract_text(self, filepath: str) -> str:
        """Extract raw text from PDF using pdfplumber."""
        import pdfplumber
        
        with pdfplumber.open(filepath) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append(f"[Page {i+1}]\n{text}")
            
            return '\n\n'.join(pages)

    def chunk_documents(self, text: str, source: str, page: int = 1) -> List[Chunk]:
        """Split extracted text at section headers."""
        import hashlib
        
        chunks_list = self.chunker.split(text)
        
        # Generate unique IDs for each chunk
        chunks = []
        for i, chunk_text in enumerate(chunks_list):
            chunk_id = f"{source}_{page}_{i}"
            hash_id = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
            
            chunk = Chunk(
                id=f"{hash_id}_{chunk_id}",
                text=chunk_text.strip(),
                source=source,
                page=page,
                start_offset=0,  # Would calculate actual offset in production
                end_offset=len(chunk_text),
                metadata={
                    "chunking_strategy": "semantic",
                    "original_length": len(text),
                }
            )
            chunks.append(chunk)
        
        return chunks

    def build_index(self, filepath: str) -> List[Chunk]:
        """Complete ingestion pipeline for a PDF file."""
        text = self.extract_text(filepath)
        return self.chunk_documents(text, source=filepath.split('/')[-1])


__all__ = [
    "IngestionPipelineInterface",
    "Chunk",
    "FixedSizeChunker",
    "SemanticChunker",
    "BasicPDFIngestor",
    "SemanticPDFIngestor",
]
