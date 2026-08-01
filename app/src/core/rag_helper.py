"""
RAG helper classes using psycopg and minsearch.
"""

import os
from typing import List, Optional
from dataclasses import dataclass

# Try to import minsearch (vector search library)
try:
    from minsearch import Index
except ImportError:
    print("minsearch not installed - using keyword search fallback")
    class Index:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, documents):
            pass
        def search(self, query, *args, **kwargs):
            return []


# Instructions for the RAG assistant
INSTRUCTIONS = """
Your task is to answer questions about the Canon EOS R6 Mark II camera
based on the provided context.

Use the context to find relevant information and provide accurate answers.
If the answer is not found in the context, respond with "I don't know."
"""

# Prompt template for generating answers (like reference)
PROMPT_TEMPLATE = """QUESTION: {question}

CONTEXT:
{context}
""".strip()


@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    course: str = "camera_knowledge"
    model: str = "llama3.2"
    instructions: str = INSTRUCTIONS
    prompt_template: str = PROMPT_TEMPLATE
    num_results: int = 5


class RAGBase:
    """Base RAG class with keyword search fallback.
    
    Provides basic RAG functionality using minsearch or keyword search.
    """

    def __init__(self, index: Optional[Index] = None, **config):
        self.index = index or Index(
            text_fields=["question", "section", "answer"],
            keyword_fields=["course"]
        )
        config = RAGConfig(**config)
        self.instructions = config.instructions
        self.course = config.course
        self.model = config.model
        self.num_results = config.num_results

    def search(self, query: str, num_results: Optional[int] = None) -> List[dict]:
        """Search for relevant documents using keyword search.
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            List of matching documents
        """
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}

        # TODO: Implement actual vector search when embeddings are available
        # For now, use keyword search as fallback
        return []

    def build_context(self, search_results: List[dict]) -> str:
        """Build context string from search results.
        
        Args:
            search_results: List of documents from search
            
        Returns:
            Formatted context string
        """
        lines = []
        for doc in search_results:
            lines.append(doc.get('section', ''))
            lines.append(f'Q: {doc.get("question", "")}')
            lines.append(f'A: {doc.get("answer", "")}')
            lines.append('')
        
        return '\n'.join(lines).strip()

    def build_prompt(self, query: str, search_results: List[dict]) -> str:
        """Build prompt for LLM from query and context.
        
        Args:
            query: User's question
            search_results: Relevant documents
            
        Returns:
            Formatted prompt string
        """
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt: str) -> str:
        """Internal LLM call implementation.
        
        Override with actual LLM API integration.
        """
        raise NotImplementedError("Override with actual LLM implementation")

    def rag(self, query: str) -> str:
        """Main RAG pipeline: search → build context → generate answer.
        
        Args:
            query: User's question
            
        Returns:
            Generated answer
        """
        search_results = self.search(query, self.num_results)
        prompt = self.build_prompt(query, search_results)
        return self.llm(prompt)


class RAGVector(RAGBase):
    """RAG implementation using vector search.
    
    Uses embeddings for semantic search.
    """

    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query: str, num_results: Optional[int] = None) -> List[dict]:
        """Search using vector embeddings.
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            List of semantically similar documents
        """
        # TODO: Implement actual vector encoding and search
        # For now, return empty list as placeholder
        return []


class RAGPgVector(RAGBase):
    """RAG implementation using PostgreSQL pgvector.

    Stores embeddings in PostgreSQL for persistent vector search.
    """

    def __init__(self, embedder, conn, **kwargs):
        super().__init__(index=None, **kwargs)
        self.embedder = embedder
        self.conn = conn

    def vec_to_str(self, vector: list) -> str:
        """Convert vector to PostgreSQL format string.
        
        Args:
            vector: List of floats representing the embedding
            
        Returns:
            Vector as PostgreSQL array string
        """
        return "[" + ",".join(str(x) for x in vector) + "]"

    def search(self, query: str, num_results: Optional[int] = None) -> List[dict]:
        """Search using pgvector similarity.
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            List of semantically similar documents from database
        """
        # TODO: Implement actual pgvector search
        # For now, return empty list as placeholder
        return []


class LMStudioRAG(RAGBase):
    """RAG implementation using LM Studio API.
    
    Uses chat completions API for LLM interaction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_client = None  # Initialize with OpenAI client
        
    def llm(self, prompt: str) -> str:
        """Make LLM call using chat completions API.
        
        Args:
            prompt: Formatted prompt from RAG pipeline
            
        Returns:
            LLM response text
        """
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        # TODO: Initialize OpenAI client and make actual call
        # response = self.llm_client.chat.completions.create(
        #     model=self.model,
        #     messages=input_messages,
        # )
        
        return "This is a placeholder response. LLM integration will be implemented in later tickets."


if __name__ == "__main__":
    print("Testing RAG helper module...")
    
    # Test base RAG class
    rag = RAGBase(course="camera_knowledge")
    print(f"RAG initialized: course={rag.course}")
    
    # Test context building
    test_docs = [
        {"section": "ISO Settings", "question": "What is ISO?", "answer": "ISO measures sensor sensitivity to light."},
    ]
    context = rag.build_context(test_docs)
    print(f"Context built: {len(context)} characters")
