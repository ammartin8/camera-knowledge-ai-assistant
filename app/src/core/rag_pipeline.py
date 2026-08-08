"""RAG Pipeline orchestrates retrieval, context building, and response generation."""

import os
import time
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.metrics import LLMCallRecord, save_record_to_db, calculate_cost
from src.llm_client import LLM_SYSTEM_PROMPT


@dataclass
class RAGPipelineConfig:
    """Configuration for the RAG pipeline."""
    num_chunks: int = 5
    model: str = "llama3.2"
    similarity_threshold: float = 0.7
    use_detailed_prompt: bool = True


class RAGPipeline:
    """RAG pipeline that orchestrates retrieval and response generation."""

    def __init__(
        self,
        vector_store=None,
        llm_client=None,
        config: Optional[RAGPipelineConfig] = None,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            vector_store: Vector store instance for retrieval (MinsearchVectorStore or similar)
            llm_client: LLM client instance for generation (OpenAILLMClient or similar)
            config: Pipeline configuration (uses defaults if not provided)
        """
        self.config = config or RAGPipelineConfig()
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
            print(f"Using LLM client with model: {llm_model}")
        except ImportError:
            print("OpenAI client not available, using mock for testing")
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
        print(f"Processing question: {question}")

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
                "citations": [],
                "metadata": {"error": True},
                "hallucination_check": None,
            }

        # Step 4: Format response with citations
        if isinstance(llm_response, dict) and 'response' in llm_response:
            answer_text = llm_response['response']
        elif hasattr(llm_response, 'response'):
            answer_text = llm_response.response
        else:
            answer_text = str(llm_response)
        
        formatted = self.response_formatter.format_response(
            question=question, answer=answer_text, chunks_used=retrieved_chunks
        )

        # Add usage info from LLM call BEFORE formatting (so it's preserved)
        if usage_info is not None:
            formatted["prompt_tokens"] = usage_info.get("prompt_tokens", 0) if isinstance(usage_info, dict) else getattr(usage_info, 'prompt_tokens', 0)
            formatted["completion_tokens"] = usage_info.get("completion_tokens", 0) if isinstance(usage_info, dict) else getattr(usage_info, 'completion_tokens', 0)
            formatted["total_tokens"] = usage_info.get("total_tokens", 0) if isinstance(usage_info, dict) else getattr(usage_info, 'total_tokens', 0)
            formatted["cost"] = usage_info.get("cost", 0.0) if isinstance(usage_info, dict) else getattr(usage_info, 'cost', 0.0)
        
        # Handle None case
        safe_chunks = retrieved_chunks if retrieved_chunks else []
        formatted["chunks_used"] = len(safe_chunks)
        
        # Add citations to metadata for display
        formatted["citations"] = formatted.get("citations", [])
        
        return formatted

    def _retrieve(self, question: str) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks from the vector store."""
        print(f"\n🔍 RETRIEVE: Question = '{question}'")

        if not self.vector_store:
            print("❌ No vector store configured!")
            return []

        # For MinsearchVectorStore (keyword-based), use direct text search
        if self.vector_store.__class__.__name__ == "MinsearchVectorStore":
            print(f"   Using Minsearch keyword search")
            
            # Debug: Check vector store state
            num_docs = len(self.vector_store._documents) if hasattr(self.vector_store, '_documents') else 0
            print(f"   Vector store has {num_docs} documents in memory")

            all_results = self.vector_store.search(question, num_results=self.config.num_chunks * 3)
            
            print(f"   Search returned {len(all_results)} raw results from minsearch")

            if not all_results:
                print("   ⚠️ Minsearch returned no results")
                return []

            # Trust minsearch's TF-IDF ranking and take top N directly
            results = all_results[:self.config.num_chunks]

            print(f"   ✅ Final retrieval: {len(results)} chunks found")
            
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
                        "chunk_id": str(metadata.get("chunk_id", "")),
                    },
                })

            print(f"   📄 Converted to {len(chunks)} formatted chunks")
            return chunks

        # Fallback for other vector store types
        print(f"Unknown vector store type: {type(self.vector_store).__name__}")
        return []

    def _build_prompt(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """Build prompt with context from retrieved chunks."""
        print("Building prompt with context")

        # Build context string from chunks
        context_parts = []
        for chunk in chunks:
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            
            source = metadata.get("source", "unknown")
            chunk_id = metadata.get("chunk_id", "0")
            
            context_parts.append(f"[Source: {source}, ID: {chunk_id}]\n{text}")

        context = "\n\n".join(context_parts)

        # Choose prompt template based on config
        if self.config.use_detailed_prompt:
            return self._build_detailed_prompt(question, context)
        else:
            return self._build_simple_prompt(question, context)

    def _build_detailed_prompt(self, question: str, context: str) -> str:
        """Build detailed prompt with system instructions."""
        system_instruction = """You are an AI assistant specialized in camera equipment knowledge. 
Your role is to provide accurate, helpful information about cameras, lenses, accessories, and photography techniques.

When answering questions:
1. Be specific and precise - give exact details when possible
2. Cite sources from the provided context when relevant
3. If you don't know something, admit it rather than making things up
4. Keep answers concise but complete"""

        return f"""You are an AI assistant specialized in camera equipment knowledge.

User Question: {question}

Context (from retrieved documents):
{context}

Instructions:
- Be specific and precise - give exact details when possible
- Cite sources from the provided context when relevant  
- If you don't know something, admit it rather than making things up
- Keep answers concise but complete

Provide your answer below:"""

    def _build_simple_prompt(self, question: str, context: str) -> str:
        """Build simple prompt with minimal instructions."""
        return f"""User Question: {question}

Context:
{context}

Answer:"""

    def _generate_response(self, prompt: str) -> tuple:
        """Generate response using LLM."""
        print("Generating response")
        start_time = time.time()

        try:
            result = self.llm_client.chat(prompt)
            
            duration = time.time() - start_time
            
            # Extract response and usage info
            if isinstance(result, dict):
                response_text = result.get('response', str(result))
                usage_info = result.get('usage', {})
            elif hasattr(result, 'response'):
                response_text = result.response
                # Handle both LLMCallResult (has .usage attribute) and other objects
                if hasattr(result, 'usage'):
                    usage_data = getattr(result, 'usage')
                    # Convert TokenUsage dataclass to dict if needed
                    if hasattr(usage_data, 'prompt_tokens'):
                        usage_info = {
                            'prompt_tokens': usage_data.prompt_tokens,
                            'completion_tokens': usage_data.completion_tokens,
                            'total_tokens': usage_data.total_tokens,
                        }
                    else:
                        usage_info = {}
                else:
                    usage_info = {}
            else:
                response_text = str(result)
                usage_info = {}

            print(f"✅ Response generated in {duration:.2f}s")
            
            return response_text, usage_info

        except Exception as e:
            print(f"❌ Generation failed: {str(e)}")
            raise

    def _handle_no_results(self, question: str) -> Dict[str, Any]:
        """Handle case when no chunks were retrieved."""
        print("No relevant chunks found - providing general answer")

        if self.config.use_detailed_prompt:
            prompt = f"""You are an AI assistant specialized in camera equipment knowledge.

User Question: {question}

I don't have specific documents to reference, but I can help with general camera knowledge. Please provide your best answer based on your training."""
        else:
            prompt = f"""User Question: {question}

Answer:"""

        try:
            llm_response, _ = self._generate_response(prompt)
            
            return {
                "answer": llm_response,
                "plain_answer": llm_response,
                "chunks_used": 0,
                "citations": [],
                "metadata": {"no_chunks_found": True},
                "hallucination_check": None,
            }
        except Exception as e:
            print(f"Failed to generate fallback response: {str(e)}")
            return {
                "answer": "I'm sorry, I couldn't find relevant information for your question.",
                "plain_answer": "I'm sorry, I couldn't find relevant information for your question.",
                "chunks_used": 0,
                "citations": [],
                "metadata": {"error": True},
                "hallucination_check": None,
            }

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Perform a vector store search."""
        if not self.vector_store:
            print("No vector store configured")
            return []
        
        return self.vector_store.search(query, num_results=5)

    def get_stats(self):
        """Get pipeline statistics (placeholder for future implementation)."""
        return {
            "config": asdict(self.config) if hasattr(self.config, '__dict__') else str(self.config),
            "vector_store_type": type(self.vector_store).__name__ if self.vector_store else None,
            "llm_client_type": type(self.llm_client).__name__ if self.llm_client else None,
        }


def asdict(obj):
    """Convert object to dictionary."""
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return str(obj)


class RAGWithMetrics(RAGPipeline):
    """RAG pipeline with metrics collection and database recording."""

    def __init__(
        self,
        vector_store=None,
        llm_client=None,
        config: Optional[RAGPipelineConfig] = None,
        db_connection=None,
    ):
        """
        Initialize RAG pipeline with metrics tracking.

        Args:
            vector_store: Vector store instance for retrieval
            llm_client: LLM client instance for generation
            config: Pipeline configuration
            db_connection: Database connection for metrics storage
        """
        super().__init__(vector_store=vector_store, llm_client=llm_client, config=config)
        self.db_connection = db_connection

    def run(self, question: str) -> Dict[str, Any]:
        """Run pipeline and record metrics to database."""
        start_time = time.time()

        try:
            # Run the pipeline
            result = super().run(question)
            
            duration = time.time() - start_time
            
            # Record metrics if database connection available
            if self.db_connection:
                self._record_metrics(question, result, duration, start_time)
            
            return result

        except Exception as e:
            print(f"Pipeline execution failed: {str(e)}")
            raise

    def _record_metrics(self, question: str, result: Dict[str, Any], 
                       duration: float, start_time: float):
        """Record metrics to database."""
        try:
            # Calculate tokens from metadata if available
            prompt_tokens = result.get("metadata", {}).get("prompt_tokens", 0)
            completion_tokens = result.get("metadata", {}).get("completion_tokens", 0)
            cost = result.get("metadata", {}).get("cost", 0.0)

            # Create metrics record
            record = LLMCallRecord(
                query=question,
                response=result.get("answer", ""),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost=cost,
                duration_ms=duration * 1000,
                timestamp=start_time,
            )

            # Save to database
            with _metrics_lock:
                save_record_to_db(record, self.db_connection)
            
            print(f"✅ Metrics recorded: {prompt_tokens} prompt, {completion_tokens} completion tokens")

        except Exception as e:
            print(f"Failed to record metrics: {str(e)}")
            # Don't fail the pipeline if metrics recording fails
