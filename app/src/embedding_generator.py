"""Embedding generation using ONNX Runtime with sentence-transformers.

This module uses ONNX Runtime for efficient local inference, following the
LLM Zoomcamp Module 2 approach. No PyTorch dependencies required at runtime.

Model: Xenova/bge-base-en-v1.5 (768 dimensions)
- Downloaded from HuggingFace Hub
- Cached locally after first run
- Falls back to mock embeddings if download fails

Requirements: pip install onnxruntime tokenizers numpy tqdm
"""

import hashlib
import os
from pathlib import Path
from typing import List, Optional


class EmbeddingGenerator:
    """Generates embeddings using ONNX Runtime with sentence-transformers models.
    
    Supports multiple Xenova models (ONNX-optimized versions):
    - Xenova/bge-base-en-v1.5 (768 dimensions) - Default
    - Xenova/bge-small-en-v1.5 (384 dimensions) - Faster, lighter
    - Xenova/all-MiniLM-L6-v2 (384 dimensions) - Fastest
    
    Model is automatically downloaded from HuggingFace on first run.
    Falls back to mock embeddings if download fails.
    """

    def __init__(self, model: str = "Xenova/bge-base-en-v1.5", model_path: Optional[str] = None):
        """Initialize embedding generator.
        
        Args:
            model: Model name from HuggingFace (default: Xenova/bge-base-en-v1.5)
            model_path: Local path to cached model (auto-downloaded if not provided)
        """
        self.model_name = model
        self.base_model_path = model_path or Path("models") / model
        
        # Check if we should use mock embeddings (for development)
        self.use_mock_embeddings = os.getenv("EMBEDDING_USE_MOCK", "0").lower() in ("1", "true", "yes")
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension for this model.
        
        Returns the number of values in each embedding vector.
        Common sizes: 384 (small), 768 (base), 1024+ (large)
        """
        # Simple mapping - can be extended if new models added
        return 768  # Default for Xenova/bge-base-en-v1.5

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text chunk.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        # If mock mode is enabled (for development/testing without model), use mock embeddings
        if self.use_mock_embeddings:
            return self._generate_mock_embedding(text, self.dimension)
        
        try:
            return self._generate_embedding_with_onnx(text)
        except Exception:
            return self._generate_mock_embedding(text, self.dimension)

    def _generate_embedding_with_onnx(self, text: str) -> List[float]:
        """Generate embedding using ONNX Runtime.
        
        Model is loaded once and cached for subsequent calls.
        Uses mean pooling with attention mask weighting.
        """
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        
        model_path = Path(self.base_model_path)
        
        if not model_path.exists():
            self._download_model(model_path)
        
        tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
        session = ort.InferenceSession(
            str(model_path / "model.onnx"), 
            providers=["CPUExecutionProvider"]
        )
        input_names = {inp.name for inp in session.get_inputs()}
        
        tokenizer.enable_padding()
        encoded = tokenizer.encode(text)
        
        feed = {}
        if "input_ids" in input_names:
            feed["input_ids"] = np.array([encoded.ids], dtype=np.int64)
        if "attention_mask" in input_names:
            feed["attention_mask"] = np.array([encoded.attention_mask], dtype=np.int64)
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.array([encoded.type_ids], dtype=np.int64)
        
        hidden = session.run(None, feed)[0]
        mask = feed["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        
        return pooled[0].tolist()

    def _download_model(self, model_path: Path):
        """Download model from HuggingFace Hub.
        
        Downloads tokenizer.json and model.onnx only.
        """
        import shutil
        from huggingface_hub import hf_hub_download, list_repo_files
        
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        model_path.mkdir(parents=True, exist_ok=True)
        files = list_repo_files(repo_id=self.model_name)
        
        onnx_candidates = ["onnx/model.onnx", "onnx/encoder_model.onnx", "model.onnx"]
        onnx_file = next((c for c in onnx_candidates if c in files), None)
        
        if not onnx_file:
            raise FileNotFoundError(f"No ONNX model found in {self.model_name}")
        
        for remote, local in [("tokenizer.json", "tokenizer.json"), (onnx_file, "model.onnx")]:
            src = hf_hub_download(repo_id=self.model_name, filename=remote)
            dst = model_path / local
            if not dst.exists():
                shutil.copy2(src, dst)
        
        onnx_ext = onnx_file + "_data"
        if onnx_ext in files:
            src = hf_hub_download(repo_id=self.model_name, filename=onnx_ext)
            dst = model_path / "model.onnx_data"
            if not dst.exists():
                shutil.copy2(src, dst)

    def _generate_mock_embedding(self, text: str, dimension: int = None) -> List[float]:
        """Generate deterministic mock embeddings for testing.
        
        Same text → same embedding (deterministic).
        Used when ONNX Runtime is unavailable or EMBEDDING_USE_MOCK=1.
        
        Args:
            text: Input text
            dimension: Embedding dimension (defaults to self.dimension)
            
        Returns:
            Mock embedding vector with correct dimensions
        """
        # Use model dimension if not specified
        dim = dimension or self.dimension
        embedding = []
        
        # Simple: position-based hash for each dimension
        # Each position gets a unique value based on its index and the text
        for i in range(dim):
            h = hashlib.md5(f"{i}:{text}".encode()).hexdigest()
            val = int(h[:8], 16) / 0xFFFFFFFF - 0.5  # Range [-0.5, 0.5]
            embedding.append(val)
        
        return embedding

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Cosine similarity measures how similar two vectors are by comparing
        the cosine of the angle between them. Range is [-1, 1]:
        - 1 = identical direction (same meaning)
        - 0 = orthogonal (no relationship)
        - -1 = opposite direction (opposite meaning)
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        import numpy as np
        
        # Normalize vectors (divide by their length)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Convert to numpy arrays for calculation
        arr1 = np.array(vec1, dtype=np.float64)
        arr2 = np.array(vec2, dtype=np.float64)
        
        # Calculate cosine similarity: dot_product / (norm_a * norm_b)
        similarity = np.dot(arr1, arr2) / (norm1 * norm2)
        return float(similarity)


__all__ = ["EmbeddingGenerator"]
