"""PDF ingestion pipeline abstractions and implementations for RAG.

This module provides the IngestionPipelineInterface abstraction allowing easy swapping
between different PDF processing strategies (fixed-size, semantic, hybrid).
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import re
import hashlib


@dataclass
class ChunkMetadata:
    """Enhanced metadata for document chunks with hierarchical structure."""
    source: str  # PDF filename or document identifier
    page: int    # Page number for citation
    section_path: List[str] = field(default_factory=list)  # Hierarchy (e.g., ["Chapter 3", "Menu Operations"])
    section_title: Optional[str] = None  # Current section name
    section_level: int = 1  # Depth in hierarchy (1-5)
    start_page: Optional[int] = None  # Where section begins
    end_page: Optional[int] = None  # Where section ends
    keywords: List[str] = field(default_factory=list)  # TF-IDF boosting keywords
    has_table: bool = False  # Contains tables
    has_diagram: bool = False  # Contains diagrams
    chunking_strategy: str = "hierarchical"  # How this chunk was created
    
    def copy(self) -> 'ChunkMetadata':
        """Create a shallow copy of the metadata."""
        return ChunkMetadata(
            source=self.source,
            page=self.page,
            section_path=list(self.section_path),
            section_title=self.section_title,
            section_level=self.section_level,
            start_page=self.start_page,
            end_page=self.end_page,
            keywords=list(self.keywords),
            has_table=self.has_table,
            has_diagram=self.has_diagram,
        )


@dataclass
class Chunk:
    """Represents a text chunk extracted from a document.
    
    Enhanced with hierarchical metadata for better RAG retrieval.
    Supports both direct attribute access and dict-style access.
    """
    id: str
    text: str
    source: str  # Document filename
    page: int
    start_offset: int
    end_offset: int
    metadata: ChunkMetadata  # Enhanced metadata with hierarchy support


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

    def generate_id(self, source: str, page: int, index: int, text: str) -> str:
        """Generate unique chunk ID with hash."""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{content_hash}_{source}_p{page}_idx{index}"





class BasicPDFIngestor(IngestionPipelineInterface):
    """Basic PDF ingestion using pdfplumber with hierarchical metadata.

    Extracts text from PDF files and applies fixed-size chunking with
    enhanced metadata for better RAG retrieval. Supports section-level
    tracking for improved search relevance.
    
    Example:
        >>> ingestor = BasicPDFIngestor(chunk_size=500, overlap=100)
        >>> chunks = ingestor.build_index("Canon_EOS_R6_Manual.pdf")
        >>> # Each chunk has hierarchical metadata for filtering
        >>> [c.metadata.section_path for c in chunks]
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
        import pdfplumber
        
        # Section header patterns for hierarchy detection
        self.header_patterns = [
            r'^\s*(Chapter|Section|Part)\s*[\d\.]+?\s*:',
            r'^\s*[A-Z][a-z]+\s+Settings',
            r'^\s*[A-Z][a-z]+\s+Mode',
            r'^\s*[A-Z][a-z]+\s+Function',
            r'^\s*How\s+to\s+',
            r'^\s*[A-Z]{2}:',  # "P:", "Tv:", "Av:"
            r'^\s*[A-Z]\+:',   # "A+:", "HDR:"
        ]

    def extract_text(self, filepath: str) -> str:
        """Extract raw text from PDF using pdfplumber.
        
        Args:
            filepath: Path to the PDF file
            
        Returns:
            Raw extracted text with page markers for structure tracking
        """
        import pdfplumber
        
        with pdfplumber.open(filepath) as pdf:
            pages = []
            total_pages = len(pdf.pages)
            
            print(f"📄 Extracting from {filepath} ({total_pages} pages)...")
            
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():  # Only add non-empty pages
                    pages.append(f"[Page {i+1}]\n{text}")
                print(f"  ✓ Page {i+1}/{total_pages}: {len(text)} chars")
            
            return '\n\n'.join(pages)

    def _is_section_header(self, line: str) -> bool:
        """Detect if a line is a section header."""
        stripped = line.strip()
        
        # Skip empty lines or very short lines
        if not stripped or len(stripped) < 5:
            return False
        
        # Check against known patterns
        for pattern in self.header_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                return True
        
        # Look for "Chapter X" or "Part X" patterns
        lower_stripped = stripped.lower()
        section_patterns = ['chapter ', 'part ', 'section ']
        has_section_pattern = any(pattern in lower_stripped for pattern in section_patterns)
        if has_section_pattern and not stripped.endswith(':'):
            return True
        
        return False

    def _extract_title(self, line: str) -> str:
        """Extract section title from header line."""
        stripped = line.strip()
        # Remove leading numbers and colons
        clean = re.sub(r'^[\d\.]+[:\s]*', '', stripped)
        return clean.rstrip(':').strip()

    def _find_section_end(self, lines: List[str], start_idx: int) -> int:
        """Find where a section ends (next header or end of text)."""
        for i in range(start_idx + 1, len(lines)):
            if self._is_section_header(lines[i]):
                return i
        return len(lines)

    def extract_hierarchy(self, text: str) -> List[Dict[str, Any]]:
        """Extract section hierarchy from PDF text."""
        sections = []
        lines = text.split('\n')
        current_path: List[str] = []
        
        for i, line in enumerate(lines):
            if self._is_section_header(line):
                title = self._extract_title(line)
                level = len(current_path) + 1
                sections.append({
                    "title": title,
                    "start_line": i,
                    "end_line": self._find_section_end(lines, i),
                    "level": level,
                    "path": current_path + [title]
                })
                current_path = sections[-1]["path"]  # type: ignore
        
        return sections

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text.
        
        Removes numbers and special characters, filters short words.
        """
        clean = re.sub(r'[\d\W]', ' ', text.lower())
        words = clean.split()
        return [w for w in words if len(w) > 3]

    def chunk_documents(self, text: str, source: str, page: int = 1) -> List[Chunk]:
        """Split extracted text into fixed-size chunks with hierarchical metadata."""
        print(f"🔨 Chunking document into {len(text)} characters...")
        
        # Try hierarchical extraction first
        sections = self.extract_hierarchy(text)
        
        if sections:
            print(f"  📊 Found {len(sections)} sections in hierarchy")
            return self._create_hierarchical_chunks(text, source, page, sections)
        else:
            print(f"  ⚠️  No sections detected, using fixed-size chunking")
            return self._create_fixed_chunks(text, source, page)
    
    def _create_hierarchical_chunks(self, text: str, source: str, page: int,
                                    sections: List[Dict[str, Any]]) -> List[Chunk]:
        """Create chunks based on detected section hierarchy."""
        chunks = []
        lines = text.split('\n')
        
        for section in sections:
            # Extract section content
            section_text = '\n'.join(lines[section['start_line']:section['end_line']])
            
            # Parse page number from section text (format: [Page N])
            chunk_page = self._parse_page_from_text(section_text) or page
            
            # Generate chunk ID
            hash_id = hashlib.md5(section_text.encode()).hexdigest()[:8]
            
            # Extract keywords from title and text
            keywords = self._extract_keywords(section['title'])
            if len(keywords) < 3:
                keywords.extend(self._extract_keywords(section_text))
            
            chunk = Chunk(
                id=f"{hash_id}_{source}_sec_{section['level']}_{section['title'].lower().replace(' ', '_')}",
                text=section_text.strip(),
                source=source,
                page=chunk_page,
                start_offset=0,
                end_offset=len(section_text),
                metadata=ChunkMetadata(
                    source=source,
                    page=chunk_page,
                    section_path=section['path'],
                    section_title=section['title'],
                    section_level=section['level'],
                    keywords=keywords[:10],  # Limit to top 10 keywords
                    chunking_strategy="hierarchical",
                )
            )
            chunks.append(chunk)
        
        print(f"  ✅ Created {len(chunks)} hierarchical chunks")
        return chunks
    
    def _parse_page_from_text(self, text: str) -> Optional[int]:
        """Parse page number from text (format: [Page N]).
        
        Args:
            text: Text to parse for page markers
            
        Returns:
            Page number if found, None otherwise
        """
        import re
        # Match patterns like [Page 1], [Page 5], etc.
        match = re.search(r'\[Page\s*(\d+)\]', text)
        if match:
            return int(match.group(1))
        return None
    
    def _create_fixed_chunks(self, text: str, source: str, page: int) -> List[Chunk]:
        """Create fixed-size chunks as fallback."""
        print(f"  🔨 Fixed chunk size: {self.chunker.chunk_size}, overlap: {self.chunker.overlap}")
        
        chunks_list = self.chunker.split(text)
        chunks = []
        
        for i, chunk_text in enumerate(chunks_list):
            # Parse page number from chunk text
            chunk_page = self._parse_page_from_text(chunk_text) or page
            hash_id = self.chunker.generate_id(source, chunk_page, index=i, text=chunk_text)
            keywords = self._extract_keywords(chunk_text[:200])  # First 200 chars
            
            chunk = Chunk(
                id=f"{hash_id}_{source}_p{chunk_page}_idx{i}",
                text=chunk_text.strip(),
                source=source,
                page=chunk_page,
                start_offset=0,
                end_offset=len(chunk_text),
                metadata=ChunkMetadata(
                    source=source,
                    page=chunk_page,
                    section_title=None,
                    section_level=1,
                    keywords=keywords[:5],  # Limit to top 5 keywords
                    chunking_strategy="fixed_size",
                )
            )
            chunks.append(chunk)
        
        print(f"  ✅ Created {len(chunks)} fixed-size chunks")
        return chunks

    def build_index(self, filepath: str) -> List[Chunk]:
        """Complete ingestion pipeline for a PDF file.
        
        Args:
            filepath: Path to the PDF file
            
        Returns:
            List of processed Chunk objects ready for embedding with enhanced metadata
        """
        text = self.extract_text(filepath)
        chunks = self.chunk_documents(text, source=filepath.split('/')[-1])
        print(f"✅ Ingestion complete: {len(chunks)} chunks from {filepath}")
        return chunks


__all__ = [
    "IngestionPipelineInterface",
    "Chunk",
    "FixedSizeChunker",
    "BasicPDFIngestor",
]
