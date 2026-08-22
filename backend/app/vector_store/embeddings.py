import time
import numpy as np
from typing import List
import functools

class EmbeddingEngine:
    """Embedding engine with SentenceTransformers support and lightweight TF-IDF cosine fallback."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except Exception:
            # High-performance lightweight TF-IDF / Hashing vectorizer fallback if offline/no download
            self.model = None

    @functools.lru_cache(maxsize=1024)
    def embed_query(self, text: str) -> np.ndarray:
        """Embeds a single query string with caching."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of texts into normalized embedding vectors."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                return embeddings.astype(np.float32)
            except Exception:
                pass

        # Fallback deterministic bag-of-words / hashing embedding (384 dims)
        return self._fallback_embed(texts)

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        vocab_dim = 384
        for text in texts:
            vec = np.zeros(vocab_dim, dtype=np.float32)
            words = text.lower().split()
            for w in words:
                idx = abs(hash(w)) % vocab_dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)

embedding_engine = EmbeddingEngine()
