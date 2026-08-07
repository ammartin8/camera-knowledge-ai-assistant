"""LLM client abstractions and implementations for RAG.

This module provides the LLMClientInterface abstraction allowing easy swapping
between different LLM providers (local models, Ollama, OpenAI, LM Studio).

All implementations use the OpenAI-compatible chat completions API pattern.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict
from dataclasses import dataclass
import time

LLM_SYSTEM_PROMPT = """You are an expert camera documentation assistant. Help users with photography equipment, settings, and techniques.

PURPOSE: Provide accurate, actionable information about cameras, lenses, exposure, and photographic best practices.

GUIDELINES:
- Be factual, cite verified specs when possible
- Explain technical terms briefly (aperture, shutter speed, ISO)
- Give practical examples with specific settings
- Use clear bullet points for structure
- Maintain professional but encouraging tone

RESPONSE FORMAT:
- Keep answers concise (3-5 key points)
- Use bullets for lists/comparisons
- Include specific examples when discussing settings
- End with brief summary or next steps if helpful

TOPICS:
- Camera systems (DSLR, mirrorless, film)
- Lenses (prime, zoom, telephoto, wide-angle, macro)
- Exposure triangle (aperture, shutter speed, ISO)
- Focus modes, image stabilization
- RAW vs JPEG, color science, white balance

AVOID:
- Generic advice without context
- Assuming specific models unless specified
- Presenting opinions as facts

EXAMPLES:
User: "What aperture for portraits?"
Assistant: "Use f/1.8-f/2.8 for background blur. Wider apertures (lower numbers) with fast lenses."

User: "Why is my JPEG noisy?"
Assistant: "High ISO above 3200 introduces noise. Lower ISO when possible, use noise reduction in post."

GOAL: Help photographers make informed equipment and technique decisions."""


@dataclass
class TokenUsage:
    """Token usage metrics for an LLM call."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class LLMCallResult:
    """Result of an LLM call with metadata."""
    response: str
    usage: TokenUsage
    response_time_ms: float
    cost: float = 0.0


class LLMClientInterface(ABC):
    """Abstract interface for LLM clients.

    Provides a unified API for generating responses and tracking metrics.
    Allows swapping between different LLM providers.
    """

    @abstractmethod
    def generate_response(self, prompt: str) -> LLMCallResult:
        """Generate a response to a prompt.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            LLMCallResult with response text and usage metrics
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        pass

    @abstractmethod
    def calculate_cost(self, usage: TokenUsage) -> float:
        """Calculate cost based on token usage.

        Args:
            usage: Token usage from an LLM call

        Returns:
            Cost in dollars
        """
        pass


class MockLLMClient(LLMClientInterface):
    """Mock LLM client for development/testing."""

    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def generate_response(self, prompt: str) -> LLMCallResult:
        """Generate a mock response."""
        response_time = 0.1  # Mock latency
        usage = TokenUsage(
            prompt_tokens=len(prompt) // 4,  # Rough estimate
            completion_tokens=200,
            total_tokens=len(prompt) // 4 + 200
        )
        
        return LLMCallResult(
            response=f"[MOCK RESPONSE] This is a placeholder response for: {prompt[:50]}...",
            usage=usage,
            response_time_ms=response_time * 1000,
            cost=0.0
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens using rough estimation (4 chars per token)."""
        return len(text) // 4

    def calculate_cost(self, usage: TokenUsage) -> float:
        """Calculate cost - returns 0 for local models."""
        # Local models don't charge
        return 0.0


class OpenAILLMClient(LLMClientInterface):
    """OpenAI-compatible LLM client.

    Works with any OpenAI-compatible API endpoint including:
    - Ollama (http://localhost:11434/v1)
    - LM Studio
    - Real OpenAI API
    
    Loads configuration from environment variables by default:
    - OPENAI_API_KEY: API key for authentication
    - OPENAI_BASE_URL: Base URL for the API endpoint
    - LLM_MODEL: Model name to use
    
    Can override with constructor parameters if needed.
    
    Example usage:
        # Using environment variables (recommended)
        client = OpenAILLMClient()  # Uses .env settings
        
        # Local Ollama (override constructor params)
        client = OpenAILLMClient(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            model="llama3.2"
        )
        
        # LM Studio
        client = OpenAILLMClient(
            api_key="your-api-key",
            base_url="http://localhost:1234/v1",
            model="mistral"
        )
    """

    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None, 
                 model: Optional[str] = None):
        # Load from environment variables if not provided
        import os
        self.api_key = api_key or os.getenv("LLM_API_KEY", "ollama")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        self.model = model or os.getenv("LLM_MODEL", "llama3.2")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            raise ImportError("openai library required for OpenAILLMClient. Install with: uv add openai")

    def generate_response(self, prompt: str) -> LLMCallResult:
        """Generate a response using the OpenAI-compatible API.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            LLMCallResult with response text and usage metrics
        """
        start_time = time.time()
        
        try:
            # Send to OpenAI-compatible endpoint using chat completions API
            response = self.client.chat.completions.create(  
                model=self.model or "gpt-4o",  # Fallback to default if None
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for factual accuracy
                timeout=120
            )
            
            llm_response: str = response.choices[0].message.content  
            
            # Get actual token usage from API (handle None safely)
            usage_data = response.usage if response.usage else TokenUsage(0, 0, 0)  
            usage = TokenUsage(
                prompt_tokens=usage_data.prompt_tokens or 0,
                completion_tokens=usage_data.completion_tokens or 0,
                total_tokens=usage_data.total_tokens or 0
            )
            
            response_time = (time.time() - start_time) * 1000
            
            # Calculate cost using official pricing
            cost = self.calculate_cost(usage)
            
            return LLMCallResult(
                response=llm_response,
                usage=usage,
                response_time_ms=response_time,
                cost=cost
            )
            
        except Exception as e:
            error_msg = f"LLM call failed: {str(e)}"
            print(error_msg)
            
            return LLMCallResult(
                response=f"[ERROR] {error_msg}",
                usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                response_time_ms=0,
                cost=0.0
            )

    def count_tokens(self, text: str) -> int:
        """Count tokens using OpenAI's tokenizer for accuracy.
        
        Falls back to rough estimation if tiktoken is not available.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        try:
            from tiktoken import get_encoding  
            encoding = get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except (ImportError, AttributeError):
            # Fallback to rough estimation
            return len(text) // 4

    def calculate_cost(self, usage: TokenUsage) -> float:
        """Calculate cost based on token usage.
        
        Uses environment variables for custom rates if set:
        - LLM_INPUT_RATE: Cost per input token (default: $0.000015)
        - LLM_OUTPUT_RATE: Cost per output token (default: $0.00003)
        
        Local models (Ollama, LM Studio) cost nothing.
        
        Args:
            usage: Token usage from an LLM call
            
        Returns:
            Cost in dollars
        """
        import os
        
        # Check for custom rates in environment
        input_rate_str = os.getenv("LLM_INPUT_RATE")
        output_rate_str = os.getenv("LLM_OUTPUT_RATE")
        
        if input_rate_str and output_rate_str:
            try:
                input_rate = float(input_rate_str)
                output_rate = float(output_rate_str)
                return (usage.prompt_tokens * input_rate + 
                        usage.completion_tokens * output_rate)
            except ValueError:
                pass
        
        # Default rates for OpenAI-compatible models
        default_input_rate = 0.000015
        default_output_rate = 0.00003
        return (usage.prompt_tokens * default_input_rate + 
                usage.completion_tokens * default_output_rate)


__all__ = [
    "LLMClientInterface",
    "TokenUsage",
    "LLMCallResult",
    "MockLLMClient",
    "OpenAILLMClient",
]
