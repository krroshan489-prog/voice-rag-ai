import time
import re
import zlib
import logging
import numpy as np
from typing import List
import functools

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Embedding engine with SentenceTransformers support and high-performance deterministic n-gram fallback."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info("[EmbeddingEngine] Successfully loaded SentenceTransformer model: %s", self.model_name)
        except Exception as e:
            logger.warning(
                "[EmbeddingEngine] SentenceTransformer model unavailable (%s). "
                "Using high-performance deterministic subword n-gram fallback.", e
            )
            self.model = None

    @functools.lru_cache(maxsize=1024)
    def embed_query(self, text: str) -> np.ndarray:
        """Embeds a single query string with caching."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of texts into normalized 384-dim embedding vectors."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                return embeddings.astype(np.float32)
            except Exception as e:
                logger.warning("[EmbeddingEngine] SentenceTransformer encode failed (%s). Falling back to deterministic n-gram embed.", e)

        # High-performance deterministic n-gram & word hashing embedding (384 dims)
        return self._fallback_embed(texts)

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """
        Deterministic 384-dimensional subword n-gram + word hashing vectorizer.
        100% reproducible across process restarts, platforms, and server deployments.
        """
        embeddings = []
        vocab_dim = 384
        for text in texts:
            vec = np.zeros(vocab_dim, dtype=np.float32)
            words = re.findall(r'\b[A-Za-z0-9_]{2,}\b', text.lower())
            for w in words:
                # Word-level feature
                idx = zlib.adler32(w.encode('utf-8')) % vocab_dim
                vec[idx] += 2.0
                # Prefix stem feature (first 5 chars)
                stem = w[:5]
                s_idx = zlib.adler32(f"stem:{stem}".encode('utf-8')) % vocab_dim
                vec[s_idx] += 1.5
                # Subword 3-gram features for fuzzy & partial matching
                for i in range(len(w) - 2):
                    gram = w[i:i+3]
                    g_idx = zlib.adler32(f"gram:{gram}".encode('utf-8')) % vocab_dim
                    vec[g_idx] += 0.5

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)


embedding_engine = EmbeddingEngine()
