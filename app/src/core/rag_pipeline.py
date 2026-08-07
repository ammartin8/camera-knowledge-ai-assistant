"""RAG Pipeline orchestrates retrieval, context building, and response generation."""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RAGPipelineConfig:
    """Configuration for the RAG pipeline."""
    num_chunks: int = 5
    model: str = "llama3.2"
    similarity_threshold: float = 0.7  # Used by embedding-based stores (ignored by Minsearch)
    use_detailed_prompt: bool = True


class RAGPipeline:
    """RAG pipeline that orchestrates retrieval and response generation."""

    def __init__(
        self, vector_store=None, llm_client=None, config: Optional[RAGPipelineConfig] = None
    ):
        """
        Initialize the RAG pipeline.

        Args:
            vector_store: Vector store instance for retrieval (MinsearchVectorStore or similar)
            llm_client: LLM client instance for generation (OpenAILLMClient or similar)
            config: Pipeline configuration (uses defaults if not provided)
        """
        self.config = config or RAGPipelineConfig()

        # Initialize components with fallbacks
        self.vector_store = vector_store

        # Load LLM configuration from environment variables or .env file
        import dotenv
        
        # Load .env if it exists
        dotenv.load_dotenv()
        
        # Use consistent defaults with app.py (port 11434 for Ollama)
        llm_api_key = os.getenv("LLM_API_KEY", "ollama")
        llm_base_url = os.getenv("LLM_BASE_URL", os.getenv("LLM_URL", "http://localhost:11434/v1"))
        llm_model = os.getenv("LLM_MODEL", self.config.model)

        try:
            from ..llm_client import OpenAILLMClient

            self.llm_client = OpenAILLMClient(
                api_key=llm_api_key, base_url=llm_base_url, model=llm_model
            )
            logger.info(f"Using LLM client with model: {llm_model}")
        except ImportError:
            logger.warning("OpenAI client not available, using mock for testing")
            from ..llm_client import MockLLMClient

            self.llm_client = MockLLMClient(model=self.config.model)

        # Initialize other components
        from .response_formatter import ResponseFormatter

        self.response_formatter = ResponseFormatter()

    def run(self, question: str) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline for a user question.

        Args:
            question: The user's question

        Returns:
            Dictionary containing answer, plain_answer, chunks_used, metadata, hallucination_check
        """
        logger.info(f"Processing question: {question}")

        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self._retrieve(question)
        
        # Handle None or empty case
        if not retrieved_chunks:
            return self._handle_no_results(question)

        # Step 2: Build prompt with context
        prompt = self._build_prompt(question, retrieved_chunks)

        # Step 3: Generate response
        llm_response, usage_info = self._generate_response(prompt)

        # Handle error responses
        if isinstance(llm_response, str) and llm_response.startswith("[ERROR]"):
            return {
                "answer": f"[ERROR] {llm_response}",
                "plain_answer": llm_response,
                "chunks_used": 0,
                "citations": [],  # No citations on error
                "metadata": {"error": True},
                "hallucination_check": None,
            }

        # Step 4: Format response with citations
        # Ensure llm_response is a string (not LLMCallResult object)
        if isinstance(llm_response, dict) and 'response' in llm_response:
            answer_text = llm_response['response']
        elif hasattr(llm_response, 'response'):
            answer_text = llm_response.response
        else:
            answer_text = str(llm_response)
        
        formatted = self.response_formatter.format_response(
            question=question, answer=answer_text, chunks_used=retrieved_chunks
        )

        if "metadata" not in formatted:
            formatted["metadata"] = {}
        
        # Add usage info from LLM call
        if usage_info:
            formatted["metadata"]["prompt_tokens"] = usage_info.get("prompt_tokens", 0)
            formatted["metadata"]["completion_tokens"] = usage_info.get("completion_tokens", 0)
            formatted["metadata"]["total_tokens"] = usage_info.get("total_tokens", 0)
            formatted["metadata"]["cost"] = usage_info.get("cost", 0.0)
        
        # Handle None case
        safe_chunks = retrieved_chunks if retrieved_chunks else []
        formatted["metadata"]["num_chunks_used"] = len(safe_chunks)

        return formatted

    def _retrieve(self, question: str) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks from the vector store."""
        logger.info(f"🔍 Retrieving chunks for question: {question}")

        if not self.vector_store:
            logger.warning("❌ No vector store configured - cannot retrieve chunks")
            return []

        # For MinsearchVectorStore (keyword-based), use direct text search
        if self.vector_store.__class__.__name__ == "MinsearchVectorStore":
            logger.info(f"   Using Minsearch keyword search")

            # Use keyword-based search - trust minsearch ranking
            all_results = self.vector_store.search(question, num_results=self.config.num_chunks * 3)
            
                
            # Handle None or empty case from minsearch
            if not all_results:
                logger.warning("   ⚠️ Minsearch returned no results")
                return []

            logger.info(f"   📊 Found {len(all_results)} raw results from minsearch")

            # Trust minsearch's TF-IDF ranking and take top N directly
            # No need for additional filtering - minsearch already provides relevant results
            results = all_results[:self.config.num_chunks]

            logger.info(f"   ✅ Final retrieval: {len(results)} chunks found")
            
            # Convert minsearch results to chunk format with metadata
            chunks = []
            for result in results:
                text = result.get('text', '') if isinstance(result, dict) else str(result)
                metadata = result.get('metadata', {}) if isinstance(result, dict) else {}
                
                chunks.append({
                    "text": text,
                    "_match_count": 0,
                    "metadata": {
                        "source": metadata.get("source", ""),
                        "page": metadata.get("page", 0),
                        "section_path": str(metadata.get("section_path", [])),
                    },
                })

            
            logger.info(f"   ✅ Returning {len(chunks)} chunks")
            return chunks
        else:
            # Fallback: use keyword search directly (MinsearchVectorStore default behavior)
            results = self.vector_store.search(question, num_results=self.config.num_chunks)
            
            # Handle None case
            if results is None or len(results) == 0:
                return []

            # Convert results to dictionary format with metadata
            chunks = []
            for result in results:
                # Handle both Document objects and dict formats
                if hasattr(result, "metadata"):
                    chunk_dict = {
                        "text": result.text,
                        "_match_count": result.get("_match_count", 0) if isinstance(result, dict) else getattr(result, "_match_count", 0),
                        "metadata": {
                            "source": result.metadata.get("source", ""),
                            "page": result.metadata.get("page", 0),
                            "section_path": str(result.metadata.get("section_path", [])),
                        },
                    }
                else:
                    chunk_dict = {
                        "text": result.get("text", ""),
                        "_match_count": result.get("_match_count", 0) if isinstance(result, dict) else getattr(result, "_match_count", 0),
                        "metadata": result.get("metadata", {}),
                    }
                chunks.append(chunk_dict)

            return chunks

    def _retrieve_with_limit(self, question: str, limit: int) -> List[Dict[str, Any]]:
        """Retrieve chunks with explicit limit (used by search method)."""
        logger.info(f"🔍 Retrieving up to {limit} chunks for: {question}")

        if not self.vector_store:
            logger.warning("❌ No vector store configured")
            return []

        # For MinsearchVectorStore, use keyword-based search
        if self.vector_store.__class__.__name__ == "MinsearchVectorStore":
            logger.info(f"   Using Minsearch keyword search")

            # Get top N results directly - trust minsearch ranking
            results = self.vector_store.search(question, num_results=limit)
            
            # Handle None case
            if results is None:
                logger.warning("   ⚠️ Minsearch returned None results")
                return []
            
            logger.info(f"   ✅ Got {len(results)} results")
        else:
            # Fallback: use keyword search directly (MinsearchVectorStore default behavior)
            results = self.vector_store.search(question, num_results=limit)

            if not results:
                return []

            # Convert to dict format
            chunks = []
            for result in results:
                if hasattr(result, "metadata"):
                    chunk_dict = {
                        "text": result.text,
                        "_match_count": result.get("_match_count", 0) if isinstance(result, dict) else getattr(result, "_match_count", 0),
                        "metadata": {
                            "source": result.metadata.get("source", ""),
                            "page": result.metadata.get("page", 0),
                            "section_path": str(result.metadata.get("section_path", [])),
                        },
                    }
                else:
                    chunk_dict = {
                        "text": result.get("text", ""),
                        "_match_count": result.get("_match_count", 0) if isinstance(result, dict) else getattr(result, "_match_count", 0),
                        "metadata": result.get("metadata", {}),
                    }
                chunks.append(chunk_dict)

            return chunks

    def _build_prompt(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """Build the prompt with context from retrieved chunks."""
        # Format chunks for the prompt template
        formatted_chunks = []
        total_chars = 0
        max_length = 5000
        
        for chunk in chunks:
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            
            # Truncate very long chunks
            if text and len(text) > 2000:
                text = text[:2000] + "..."

            # Build citation info
            citations = []
            if metadata.get("page"):
                citations.append(f"Page {metadata['page']}")
            if metadata.get("section_title"):
                citations.append(metadata['section_title'])
            if metadata.get("source"):
                citations.append(metadata['source'])
            
            citation_str = ", ".join(citations) if citations else "No citation info"
            
            # Add chunk with citation marker
            chunk_entry = f"""
--- Result ---
Source: {citation_str}

{text}

--- End Result ---
""".strip()
            
            formatted_chunks.append(chunk_entry)
            total_chars += len(chunk_entry) + 100  # Add buffer
        
        # Truncate if needed
        context = formatted_chunks[0] if formatted_chunks else "No context available."
        remaining = sum(len(c) for c in formatted_chunks[1:])
        if remaining > max_length - len(context):
            # Trim subsequent chunks
            for chunk_entry in formatted_chunks[1:]:
                if total_chars + len(chunk_entry) <= max_length:
                    context += "\n\n" + chunk_entry
                else:
                    # Truncate this chunk
                    remaining_chars = max_length - len(context)
                    context += "\n\n" + chunk_entry[:remaining_chars]
                    break
        
        # Use PROMPT_TEMPLATE_DETAILED for the prompt
        prompt = f"""QUESTION: {question}

CONTEXT:
{context}

Please provide a comprehensive answer to the question using ONLY the information from the context above.

Your response should:
1. Directly address the question with clear, accurate information
2. Explain technical concepts in accessible language when necessary
3. Include step-by-step instructions if the question asks 'how to' do something
4. Reference specific sections of the manual where applicable (e.g., 'According to the ISO Settings section...')
5. Be honest about limitations - if information is not in the context, say so

IMPORTANT: Do not make up information or add details that are not in the provided context.
"""
        return prompt

    def _generate_response(self, prompt: str) -> tuple:
        """Generate response using LLM.
        
        Returns:
            Tuple of (response_text, usage_info) where usage_info contains
            token counts and cost if available, or None for mock client
        """
        from ..llm_client import LLMCallResult

        try:
            # Get the response - handle both string and LLMCallResult objects
            raw_output = self.llm_client.generate_response(prompt)

            usage_info = None
            if isinstance(raw_output, LLMCallResult):
                llm_output = raw_output.response  # Use the response directly (already a string)
                usage_info = {
                    "prompt_tokens": int(raw_output.usage.prompt_tokens),
                    "completion_tokens": int(raw_output.usage.completion_tokens),
                    "total_tokens": int(raw_output.usage.total_tokens),
                    "cost": float(raw_output.cost),
                }
            else:
                llm_output = str(raw_output)

            return llm_output, usage_info

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "[ERROR] LLM generation failed: " + str(e), None

    def _handle_no_results(self, question: str) -> Dict[str, Any]:
        """Handle case when no chunks were retrieved."""
        return {
            "answer": f"[LIMITATION] I don't have information about '{question}' in my current knowledge base.",
            "plain_answer": f"I don't have information about '{question}'. Based on the provided context from the official Canon EOS R6 Mark II manual, there is no relevant information regarding this specific topic. Therefore, I cannot provide details without making up information outside of the supplied documentation.",
            "chunks_used": [],
            "citations": [],  # No citations when no chunks retrieved
            "metadata": {"error": True, "reason": "no_chunks_retrieved"},
            "hallucination_check": None,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "num_chunks_configured": self.config.num_chunks,
            "model": self.config.model,
            "similarity_threshold": self.config.similarity_threshold,
            "vector_store_type": type(self.vector_store).__name__ if self.vector_store else "None",
        }

    def search(self, question: str, num_chunks: Optional[int] = None) -> List[Dict[str, Any]]:
        """Public method to retrieve chunks with optional limit."""
        # Use passed num_chunks or fall back to config default
        effective_num_chunks = num_chunks if num_chunks is not None else self.config.num_chunks

        # Call _retrieve but override the limit
        return self._retrieve_with_limit(question, effective_num_chunks)
