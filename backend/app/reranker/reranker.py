import re
from typing import List, Dict, Any

class TwoStageReranker:
    """Reranks retrieved candidate chunks to prioritize highest relevance before LLM consumption."""

    @staticmethod
    def _stem_word(word: str) -> str:
        word = word.lower()
        if len(word) > 4 and word.endswith('s'):
            return word[:-1]
        if len(word) > 5 and word.endswith('ies'):
            return word[:-3] + 'y'
        if len(word) > 5 and word.endswith('ing'):
            return word[:-3]
        return word

    @staticmethod
    def rerank(query: str, retrieved_chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Reranks chunks by combining similarity vector score with query-term position & density weighting.
        """
        if not retrieved_chunks:
            return []

        raw_terms = set(re.findall(r'\b[A-Za-z0-9_]{2,}\b', query.lower()))
        stemmed_terms = {TwoStageReranker._stem_word(t) for t in raw_terms}
        
        reranked = []
        for chunk in retrieved_chunks:
            text = chunk.get("text", "").lower()
            sim_score = chunk.get("similarity_score", 0.0)

            # Keyword match ratio with stemming support
            chunk_words = {TwoStageReranker._stem_word(w) for w in re.findall(r'\b[A-Za-z0-9_]{2,}\b', text)}
            matched_terms = [t for t in stemmed_terms if t in chunk_words or any(cw.startswith(t) for cw in chunk_words)]
            keyword_ratio = len(matched_terms) / max(1, len(stemmed_terms))

            # Phrase match bonus
            exact_phrase_bonus = 0.25 if any(t in text for t in raw_terms) else 0.0

            # Final Rerank Score
            rerank_score = float(round((0.5 * sim_score) + (0.35 * keyword_ratio) + exact_phrase_bonus, 4))

            chunk_copy = dict(chunk)
            chunk_copy["rerank_score"] = rerank_score
            chunk_copy["matched_keywords"] = matched_terms
            reranked.append(chunk_copy)

        # Sort descending by rerank_score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

reranker = TwoStageReranker()
