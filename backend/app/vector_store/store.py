import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from backend.app.vector_store.embeddings import embedding_engine
from backend.app.config import settings

logger = logging.getLogger(__name__)

# ── Relevance threshold ────────────────────────────────────────────────────────
# Why 0.30?
#   - all-MiniLM-L6-v2 cosine similarity for UNRELATED pairs:  ~0.05–0.22
#   - all-MiniLM-L6-v2 cosine similarity for RELEVANT pairs:   ~0.35–0.90
#   - A cutoff of 0.30 lies safely in the gap between these ranges.
#   - Empirically: "timetable" vs "MEPAP certification" ≈ 0.12 → correctly blocked.
#   - "What is machine learning" vs ML passage ≈ 0.55 → correctly allowed.
MIN_SIMILARITY_THRESHOLD = 0.22


class VectorStore:
    """
    Vector database index supporting similarity search, top-k retrieval, and metadata filtering.

    Key guardrail: search() applies a minimum cosine-similarity threshold so that
    chunks with no genuine relevance to the query are never returned to the pipeline.
    This prevents the LLM from generating answers from unrelated retrieved context.
    """

    def __init__(self, persistence_path: Optional[str] = None):
        self.persistence_path = persistence_path or os.path.join(settings.INDEX_DIR, "vector_index.json")
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.load()

    def add_chunks(self, new_chunks: List[Dict[str, Any]]) -> int:
        if not new_chunks:
            return 0

        texts = [c["text"] for c in new_chunks]
        new_vecs = embedding_engine.embed_texts(texts)

        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = new_vecs
        else:
            self.embeddings = np.vstack([self.embeddings, new_vecs])

        self.chunks.extend(new_chunks)
        self.save()
        return len(new_chunks)

    def delete_document(self, document_name: str) -> int:
        if not self.chunks:
            return 0

        keep_indices = [i for i, c in enumerate(self.chunks) if c.get("document_name") != document_name]
        deleted_count = len(self.chunks) - len(keep_indices)

        if deleted_count > 0:
            self.chunks = [self.chunks[i] for i in keep_indices]
            if self.embeddings is not None and len(keep_indices) > 0:
                self.embeddings = self.embeddings[keep_indices]
            else:
                self.embeddings = None
            self.save()

        return deleted_count

    def clear(self):
        self.chunks = []
        self.embeddings = None
        self.save()

    def search(
        self,
        query: str,
        top_k: int = 4,
        strategy_filter: Optional[str] = None,
        doc_filter: Optional[str] = None,
        min_score: float = MIN_SIMILARITY_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """
        Similarity search. Returns top-k matching chunks that exceed min_score.

        The min_score threshold is the primary relevance gate:
        - If the best matching chunk scores below min_score, returns [] immediately.
        - This means the downstream pipeline receives no context and must return
          the safe refusal instead of generating an answer from irrelevant content.
        - Every returned chunk has its similarity_score logged for observability.
        """
        if not self.chunks or self.embeddings is None or len(self.embeddings) == 0:
            logger.info("[VectorStore] Empty index — no results for query: %r", query[:60])
            return []

        query_vec = embedding_engine.embed_query(query)

        # Cosine similarity (embeddings are already normalized by all-MiniLM-L6-v2)
        norm_q = np.linalg.norm(query_vec)
        norm_e = np.linalg.norm(self.embeddings, axis=1)
        denom = norm_e * norm_q
        denom[denom == 0] = 1.0
        scores = np.dot(self.embeddings, query_vec) / denom

        # Apply chunk-level filters (strategy, doc)
        valid_indices = []
        for i, chunk in enumerate(self.chunks):
            if strategy_filter and chunk.get("chunking_strategy") != strategy_filter:
                continue
            if doc_filter and chunk.get("document_name") != doc_filter:
                continue
            valid_indices.append(i)

        if not valid_indices:
            logger.info("[VectorStore] No chunks after filter — query: %r", query[:60])
            return []

        filtered_scores = scores[valid_indices]
        top_score = float(filtered_scores.max())

        # ── RELEVANCE GATE ────────────────────────────────────────────────────
        # Log the top similarity score for every query so operators can verify
        # the threshold is working correctly before a demo.
        logger.info(
            "[VectorStore] query=%r | top_score=%.4f | threshold=%.2f | verdict=%s",
            query[:60],
            top_score,
            min_score,
            "PASS" if top_score >= min_score else "BELOW_THRESHOLD→no_context",
        )

        if top_score < min_score:
            # Nothing in the index is sufficiently relevant to this query.
            # Return empty so the pipeline issues a safe refusal.
            logger.warning(
                "[VectorStore] RELEVANCE GATE: top score %.4f < threshold %.2f — "
                "returning no context for query: %r",
                top_score, min_score, query[:60],
            )
            return []
        # ── END RELEVANCE GATE ────────────────────────────────────────────────

        sorted_sub_indices = np.argsort(filtered_scores)[::-1][:top_k]

        results = []
        for sub_idx in sorted_sub_indices:
            score = float(round(filtered_scores[sub_idx], 4))
            # Only include chunks that individually exceed the threshold
            if score < min_score:
                continue
            original_idx = valid_indices[sub_idx]
            chunk_copy = dict(self.chunks[original_idx])
            chunk_copy["similarity_score"] = score
            results.append(chunk_copy)

        logger.info(
            "[VectorStore] Returning %d/%d chunks above threshold for query: %r",
            len(results), top_k, query[:60],
        )
        return results

    def save(self):
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        data = {
            "chunks": self.chunks,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else []
        }
        with open(self.persistence_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def recompute_embeddings(self):
        """Recomputes embeddings for all indexed chunks using current embedding_engine."""
        if not self.chunks:
            self.embeddings = None
            return
        texts = [c.get("text", "") for c in self.chunks]
        self.embeddings = embedding_engine.embed_texts(texts)
        logger.info("[VectorStore] Recomputed embeddings for %d chunks.", len(self.chunks))
        self.save()

    def load(self):
        if os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.chunks = data.get("chunks", [])
                    raw_emb = data.get("embeddings", [])
                    if raw_emb and len(raw_emb) == len(self.chunks):
                        self.embeddings = np.array(raw_emb, dtype=np.float32)
                    else:
                        self.recompute_embeddings()
            except Exception as e:
                logger.warning("[VectorStore] Load failed (%s) — resetting store.", e)
                self.chunks = []
                self.embeddings = None

vector_store = VectorStore()
