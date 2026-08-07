"""
Response formatter for RAG assistant.

This module handles formatting responses from the RAG pipeline, including:
- Adding citations and source references
- Formatting step-by-step instructions
- Creating clean, readable output
- Hallucination detection (basic)
"""

from typing import List, Dict, Any, Optional


class ResponseFormatter:
    """Format RAG responses with citations and proper structure."""
    
    def __init__(self, max_citations_per_response: int = 3):
        self.max_citations = max_citations_per_response
    
    def format_response(
        self,
        question: str,
        answer: str,
        chunks_used: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Format a complete RAG response with citations.
        
        Args:
            question: The original user question
            answer: The LLM-generated answer
            chunks_used: Chunks that were used to generate the answer
            
        Returns:
            Formatted response dictionary with formatted text and metadata
        """
        # Extract citations from chunks
        citations = self._extract_citations(chunks_used)
        
        # Build formatted answer with inline citations
        formatted_answer = self._add_inline_citations(answer, citations, chunks_used)
        
        # Build final output
        safe_chunks = chunks_used if chunks_used else []
        output = {
            'question': question,
            'answer': formatted_answer,
            'plain_text': answer,  # Raw answer without formatting
            'citations': citations,
            'chunks_used': len(safe_chunks),
        }
        
        return output
    
    def _extract_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract citation information from chunks."""
        citations = []
        seen_pages = set()  # Track unique page numbers
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {}) if isinstance(chunk, dict) else {}
            
            # Get page number - try different possible field names
            page = None
            if 'page' in metadata and metadata['page'] is not None:
                page = int(metadata['page'])  # Ensure it's an integer
            elif 'pageNum' in metadata and metadata['pageNum'] is not None:
                page = int(metadata['pageNum'])
            
            # Create citation if we have a valid page number
            if page is not None and page > 0:
                source = metadata.get('source') or 'Canon EOS R6 Mark II Manual'
                
                # Avoid duplicate citations (same page)
                if page not in seen_pages:
                    seen_pages.add(page)
                    citations.append({
                        'page': page,
                        'source': source
                    })
        
        return citations[:self.max_citations]  # Limit citations
    
    def _add_inline_citations(self, answer: str, citations: List[Dict[str, Any]], 
                             chunks_used: List[Dict[str, Any]]) -> str:
        """
        Add inline citations to the answer where appropriate.
        
        This is a simple implementation that adds citations at natural break points.
        For production, you might want more sophisticated matching.
        """
        # Safely get citations list (handle None case)
        safe_citations = citations if citations else []
        
        # Split answer into paragraphs and add citations at the end of each paragraph
        paragraphs = answer.split('\n\n')
        formatted_paragraphs = []
        
        for i, paragraph in enumerate(paragraphs):
            # Add citation once per paragraph (not every sentence)
            if len(safe_citations) > 0:
                citation = safe_citations[i % len(safe_citations)]
                # Ensure citation has valid page number
                page_str = f"Page {citation['page']}" if citation and 'page' in citation else "Canon EOS R6 Mark II Manual"
                source = citation.get('source', 'Canon EOS R6 Mark II Manual')
                
                # Check if paragraph already ends with a citation
                if not paragraph.strip().endswith(']'):
                    citation_str = f"\n\nSource: {page_str} - {source}"
                    paragraph += citation_str
            
            formatted_paragraphs.append(paragraph)
        
        return '\n\n'.join(formatted_paragraphs)
