"""LLM client abstractions and implementations for RAG.

This module provides the LLMClientInterface abstraction allowing easy swapping
between different LLM providers (local models, Ollama, OpenAI, LM Studio).

All implementations use the OpenAI-compatible chat completions API pattern.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict
from dataclasses import dataclass
import time


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
        # Example pricing (adjust based on actual API costs)
        if "gpt" in self.model.lower():
            return (usage.prompt_tokens * 0.000015 + 
                   usage.completion_tokens * 0.00003) / 1000
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
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "ollama")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
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
            response = self.client.chat.completions.create(  # type: ignore
                model=self.model or "gpt-4o",  # Fallback to default if None
                messages=[
                    {"role": "system", "content": "You are a helpful camera documentation assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for factual accuracy
                timeout=120
            )
            
            llm_response: str = response.choices[0].message.content  # type: ignore
            
            # Get actual token usage from API (handle None safely)
            usage_data = response.usage if response.usage else TokenUsage(0, 0, 0)  # type: ignore
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
            from tiktoken import get_encoding  # type: ignore
            encoding = get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except (ImportError, AttributeError):
            # Fallback to rough estimation
            return len(text) // 4

    def calculate_cost(self, usage: TokenUsage) -> float:
        """Calculate cost based on pricing configuration.
        
        Supports multiple pricing strategies:
        
        1. **Custom rates via environment** (recommended for flexibility):
           OPENAI_INPUT_RATE=0.000005    # $ per token
           OPENAI_OUTPUT_RATE=0.000015   # $ per token
           
        2. **Model-specific pricing map** (JSON in env):
           OPENAI_PRICING_MAP='{"gpt-4o":{"input":0.000005,"output":0.000015}, "groq-llama3":{"input":0,"output":0}}'
           
        3. **Default fallback**: Uses model name matching or generic rates
        
        Args:
            usage: Token usage from an LLM call
            
        Returns:
            Cost in dollars
        """
        # Try custom pricing from environment
        custom_rates = self._get_custom_pricing()
        if custom_rates:
            return self._calculate_with_custom_rates(usage, custom_rates)
        
        # Fallback to model name matching
        return self._calculate_by_model_name(usage)
    
    def _get_custom_pricing(self) -> Optional[Dict[str, float]]:
        """Get custom pricing configuration from environment variables."""
        import os
        
        # Method 1: Simple rate override (applies to all models)
        input_rate = os.getenv("OPENAI_INPUT_RATE")
        output_rate = os.getenv("OPENAI_OUTPUT_RATE")
        
        if input_rate and output_rate:
            try:
                return {
                    "input": float(input_rate),
                    "output": float(output_rate)
                }
            except ValueError:
                pass
        
        # Method 2: Model-specific pricing map (JSON string)
        pricing_map_str = os.getenv("OPENAI_PRICING_MAP")
        if pricing_map_str:
            try:
                import json
                return json.loads(pricing_map_str)
            except (json.JSONDecodeError, ValueError):
                pass
        
        return None
    
    def _calculate_with_custom_rates(self, usage: TokenUsage, rates: Dict[str, float]) -> float:
        """Calculate cost using custom pricing rates."""
        input_rate = rates.get("input", 0.0)
        output_rate = rates.get("output", 0.0)
        
        return (usage.prompt_tokens * input_rate + 
                usage.completion_tokens * output_rate)
    
    def _calculate_by_model_name(self, usage: TokenUsage) -> float:
        """Calculate cost based on model name matching."""
        model_lower = (self.model or "").lower()
        
        # Check for exact matches first
        pricing_map = {
            "gpt-4o": {"input": 0.000005, "output": 0.000015},
            "gpt-4": {"input": 0.00003, "output": 0.00006},
            "gpt-3.5-turbo": {"input": 0.0000015, "output": 0.000002},
        }
        
        # Check if model matches any known model
        for known_model, rates in pricing_map.items():
            if known_model in model_lower:
                return (usage.prompt_tokens * rates["input"] + 
                        usage.completion_tokens * rates["output"])
        
        # Local models - typically free
        if "llama" in model_lower or "mistral" in model_lower or "qwen" in model_lower:
            return 0.0
        
        # Generic fallback rate (adjust based on your actual API costs)
        # This is a reasonable default for unknown OpenAI-compatible models
        return (usage.prompt_tokens * 0.000015 + 
                usage.completion_tokens * 0.00003)


__all__ = [
    "LLMClientInterface",
    "TokenUsage",
    "LLMCallResult",
    "MockLLMClient",
    "OpenAILLMClient",
]
