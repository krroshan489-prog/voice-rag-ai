"""
Guardrail & Hallucination Verification — 3-pass pipeline:
  Pass 0 (NEW): Query↔Context RELEVANCE check — rejects if retrieved chunks are
                semantically unrelated to the query, even if the answer is
                "faithful" to those irrelevant chunks.
  Pass 1:       Faithfulness check — answer must be grounded in the context.
  Pass 2:       Strict local re-extraction fallback.

This correctly implements Hacker House Goa 2026 Task 2 Requirement 6:
"handling for off-topic queries … Show that your system knows when NOT to answer."
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class GuardrailVerifier:
    """Guardrail & Hallucination verification pipeline enforcing safety, groundedness, and security."""

    INJECTION_PATTERNS = [
        r'ignore\s+previous\s+instructions',
        r'system\s+prompt',
        r'you\s+are\s+now\s+a',
        r'override\s+rules',
        r'reveal\s+secret',
        r'drop\s+(table|database)',
        r'bypass\s+guardrails',
        r'forget\s+(your|all)\s+(previous|prior)',
        r'act\s+as\s+(if|though)',
        r'pretend\s+you',
        r'jailbreak',
    ]

    # Content-safety blocklist — runs at verify_request() before any retrieval.
    # Matches queries that contain unsafe/inappropriate content regardless of
    # whether a corpus record technically covers the topic.
    # Patterns are word-boundary anchored to avoid false positives on substrings
    # (e.g. "sex" matches alone but not inside "sexual harassment policy").
    SAFETY_BLOCKLIST = [
        # Sexual / explicit content
        r'\bsex\b',
        r'\bporn(ography)?\b',
        r'\bnude\b',
        r'\bnaked\b',
        r'\bexplicit\s+(content|material|image|video)\b',
        r'\berotic\b',
        r'\bhentai\b',
        r'\bxxxb\b',
        # Graphic violence / self-harm
        r'\bhow\s+to\s+(kill|murder|stab|shoot)\b',
        r'\bsuicide\s+(method|instruction|how)\b',
        r'\bself\s*harm\b',
        r'\bcut\s+myself\b',
        # Hate speech / slurs
        r'\bn[i!1][g][g][e3]r\b',
        r'\bf[a@][g][g][o0]t\b',
        r'\bkike\b',
        r'\bwhite\s+power\b',
        r'\brace\s+war\b',
    ]

    SAFETY_FALLBACK = "This assistant can't help with that topic."

    UNANSWERABLE_FALLBACK = (
        "I couldn't find enough information in the provided knowledge base to answer that."
    )

    # ── Thresholds ────────────────────────────────────────────────────────────
    # RELEVANCE_THRESHOLD (Pass 0):
    #   Minimum fraction of meaningful query terms that must appear in context.
    RELEVANCE_THRESHOLD = 0.25

    # Minimum number of distinct meaningful query terms that must overlap with context.
    MIN_ABSOLUTE_MATCHES = 2

    # Short-query threshold
    SHORT_QUERY_THRESHOLD = 4

    # SIMILARITY_SCORE_THRESHOLD (Pass 0 hard gate):
    SIMILARITY_SCORE_THRESHOLD = 0.22

    PASS1_THRESHOLD = 0.25   # minimum answer-term support ratio for Pass 1
    PASS2_THRESHOLD = 0.15   # more lenient threshold for Pass 2 strict regen

    # Confidence gate threshold
    CONFIDENCE_THRESHOLD = 0.20

    # Unified stop-word set (excludes generic English function words and question words)
    STOP = {
        "what", "when", "where", "which", "that", "this", "with", "from",
        "have", "been", "will", "were", "they", "their", "then", "than",
        "into", "your", "also", "some", "more", "does", "give", "make",
        "tell", "show", "just", "very", "would", "could", "should", "about",
        "after", "before", "each", "other", "such", "only", "same", "than",
        "there", "them", "these", "those", "is", "it", "in", "to", "of",
        "or", "on", "at", "by", "an", "am", "do", "if", "my", "no", "so",
        "we", "be", "as", "he", "me", "us", "are", "was", "for", "how", "why", "who"
    }

    # ── Input validation ──────────────────────────────────────────────────────

    @staticmethod
    def verify_request(query: str) -> Dict[str, Any]:
        """
        Pre-retrieval validation: checks query length, malformed input, and prompt injection.
        Returns {"is_valid": bool, "reason": str, "sanitized_query": str}
        """
        if not query or not query.strip():
            return {"is_valid": False, "reason": "Empty or whitespace-only query provided.", "sanitized_query": ""}

        cleaned = query.strip()
        if len(cleaned) < 2:
            return {"is_valid": False, "reason": "Query is too short to process.", "sanitized_query": cleaned}

        for pattern in GuardrailVerifier.INJECTION_PATTERNS:
            if re.search(pattern, cleaned, re.IGNORECASE):
                return {
                    "is_valid": False,
                    "reason": "Prompt injection pattern detected.",
                    "sanitized_query": cleaned,
                    "guardrail_status": "REJECTED_INPUT_VALIDATION",
                }

        # Content-safety check — runs after injection detection, before retrieval.
        for pattern in GuardrailVerifier.SAFETY_BLOCKLIST:
            if re.search(pattern, cleaned, re.IGNORECASE):
                logger.warning(
                    "[Guardrail] REJECTED_UNSAFE_CONTENT: matched pattern %r for query: %r",
                    pattern, cleaned[:60],
                )
                return {
                    "is_valid": False,
                    "reason": GuardrailVerifier.SAFETY_FALLBACK,
                    "sanitized_query": cleaned,
                    "guardrail_status": "REJECTED_UNSAFE_CONTENT",
                }

        return {"is_valid": True, "reason": "Passed pre-execution security check.", "sanitized_query": cleaned}

    # ── Pass 0: Relevance check (query ↔ context) ─────────────────────────────

    @staticmethod
    def _compute_query_context_relevance(
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> tuple[float, int]:
        """
        Computes how relevant the retrieved chunks are TO THE QUERY —
        term-overlap ratio on meaningful query terms (including short acronyms like AI/ML/RAG).

        Returns a tuple of (relevance_ratio, abs_matches).
        """
        query_words = [
            w for w in re.findall(r'\b[A-Za-z0-9_]{2,}\b', query.lower())
            if w not in GuardrailVerifier.STOP
        ]
        if not query_words:
            # Query consists entirely of stop/generic words — be lenient
            return 1.0, 0

        # Build stemmed prefix set from meaningful query terms (min length 2)
        query_stems = {w[:5] for w in query_words}

        # Build context vocabulary from all retrieved chunks
        context_text = " ".join(c.get("text", "") for c in retrieved_chunks).lower()
        context_words = re.findall(r'\b[A-Za-z0-9_]{2,}\b', context_text)
        context_stems = {w[:5] for w in context_words if w not in GuardrailVerifier.STOP}

        matched = query_stems & context_stems
        relevance = len(matched) / len(query_stems)

        logger.info(
            "[Guardrail Pass0] query_terms=%s | matched=%s | relevance=%.3f | abs_matches=%d | threshold=%.2f | min_abs=%d | verdict=%s",
            sorted(query_stems)[:8],
            sorted(matched)[:8],
            relevance,
            len(matched),
            GuardrailVerifier.RELEVANCE_THRESHOLD,
            GuardrailVerifier.MIN_ABSOLUTE_MATCHES,
            "RELEVANT" if (
                relevance >= GuardrailVerifier.RELEVANCE_THRESHOLD and
                (len(query_stems) < 2 or len(matched) >= GuardrailVerifier.MIN_ABSOLUTE_MATCHES)
            ) else "OFF-TOPIC->refusal",
        )
        return relevance, len(matched)


    # ── Pass 1: Faithfulness check (answer ↔ context) ─────────────────────────

    @staticmethod
    def _compute_groundedness(answer: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
        """
        Groundedness ratio: fraction of significant answer terms present in context.
        """
        context_corpus = " ".join([c.get("text", "") for c in retrieved_chunks]).lower()
        answer_terms = set(re.findall(r'\b\w{4,}\b', answer.lower()))
        if not answer_terms:
            return 0.0
        supported = [t for t in answer_terms if t in context_corpus]
        return len(supported) / len(answer_terms)

    # ── Main verification entry point ──────────────────────────────────────────

    @staticmethod
    def verify_groundedness(
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        llm_response: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        3-pass post-generation verification:

        Pass 0 — RELEVANCE: Are the retrieved chunks related to the user's query?
                  Rejects even before looking at the answer if the context is off-topic.
                  This is the fix for the "MEPAP / timetable" false-positive bug.

        Pass 1 — FAITHFULNESS: Is the generated answer supported by the context?

        Pass 2 — STRICT REGEN: Local re-extraction attempt before final refusal.
        """
        answer = llm_response.get("answer", "")
        can_answer = llm_response.get("can_answer", True)
        confidence = llm_response.get("confidence", 0.0)
        sources = llm_response.get("sources", [])

        # ── Guard: empty retrieval or explicit unanswerable ───────────────────
        if not retrieved_chunks or not can_answer or confidence < 0.2:
            logger.info("[Guardrail] REJECTED_UNANSWERABLE — no valid retrieval or low confidence.")
            return {
                "answer": GuardrailVerifier.UNANSWERABLE_FALLBACK,
                "confidence": 0.0,
                "sources": [],
                "can_answer": False,
                "guardrail_status": "REJECTED_UNANSWERABLE",
                "guardrail_reason": "Knowledge base contains insufficient evidence.",
                "groundedness_pass": 0,
            }

        # ── Guard: similarity score hard cutoff ───────────────────────────────
        # Belt-and-suspenders: in case the vector store threshold was bypassed.
        top_sim = max(
            (c.get("similarity_score", 0.0) for c in retrieved_chunks),
            default=0.0,
        )
        logger.info(
            "[Guardrail] top_similarity=%.4f | sim_threshold=%.2f",
            top_sim, GuardrailVerifier.SIMILARITY_SCORE_THRESHOLD,
        )
        if top_sim < GuardrailVerifier.SIMILARITY_SCORE_THRESHOLD:
            logger.warning(
                "[Guardrail] REJECTED_LOW_SIMILARITY: top_sim=%.4f < %.2f — off-topic query.",
                top_sim, GuardrailVerifier.SIMILARITY_SCORE_THRESHOLD,
            )
            return {
                "answer": GuardrailVerifier.UNANSWERABLE_FALLBACK,
                "confidence": 0.0,
                "sources": [],
                "can_answer": False,
                "guardrail_status": "REJECTED_LOW_SIMILARITY",
                "guardrail_reason": (
                    f"Retrieved context similarity ({top_sim:.3f}) is below the relevance "
                    f"threshold ({GuardrailVerifier.SIMILARITY_SCORE_THRESHOLD:.2f}). "
                    f"The query appears to be outside the knowledge base scope."
                ),
                "groundedness_pass": 0,
                "top_similarity_score": top_sim,
            }

        # ── PASS 0: Query ↔ Context relevance ────────────────────────────────
        relevance_score, abs_matches = GuardrailVerifier._compute_query_context_relevance(
            query, retrieved_chunks
        )
        n_meaningful = len({
            w[:5] for w in re.findall(r'\b[A-Za-z0-9_]{2,}\b', query.lower())
            if w not in GuardrailVerifier.STOP
        })
        if n_meaningful <= 1:
            strict_gate_fails = False
        elif n_meaningful <= GuardrailVerifier.SHORT_QUERY_THRESHOLD:
            strict_gate_fails = (abs_matches < 1)
        else:
            strict_gate_fails = (abs_matches < GuardrailVerifier.MIN_ABSOLUTE_MATCHES)

        is_relevant = (
            relevance_score >= GuardrailVerifier.RELEVANCE_THRESHOLD
            or top_sim >= 0.45
        )
        if not is_relevant or (strict_gate_fails and top_sim < 0.45):
            logger.warning(
                "[Guardrail] REJECTED_OFF_TOPIC: relevance=%.3f abs_matches=%d n_terms=%d | query=%r",
                relevance_score, abs_matches, n_meaningful, query[:60],
            )
            return {
                "answer": GuardrailVerifier.UNANSWERABLE_FALLBACK,
                "confidence": 0.0,
                "sources": [],
                "can_answer": False,
                "guardrail_status": "REJECTED_OFF_TOPIC",
                "guardrail_reason": (
                    f"Retrieved context is not relevant to the query "
                    f"(relevance score: {relevance_score:.2f}, abs_matches: {abs_matches}, "
                    f"threshold: {GuardrailVerifier.RELEVANCE_THRESHOLD:.2f}). "
                    f"This query is outside the MSMARCO-XI knowledge base scope."
                ),
                "groundedness_pass": 0,
                "query_relevance_score": relevance_score,
                "top_similarity_score": top_sim,
            }

        # ── PASS 1: Faithfulness (answer ↔ context) ───────────────────────────
        answer_terms = set(re.findall(r'\b\w{4,}\b', answer.lower()))
        if not answer_terms:
            return {
                "answer": GuardrailVerifier.UNANSWERABLE_FALLBACK,
                "confidence": 0.0,
                "sources": [],
                "can_answer": False,
                "guardrail_status": "REJECTED_MALFORMED",
                "guardrail_reason": "Answer contained no evaluable semantic content.",
                "groundedness_pass": 1,
            }

        support_ratio = GuardrailVerifier._compute_groundedness(answer, retrieved_chunks)
        logger.info(
            "[Guardrail Pass1] support_ratio=%.3f | threshold=%.2f | query=%r",
            support_ratio, GuardrailVerifier.PASS1_THRESHOLD, query[:60],
        )

        if support_ratio >= GuardrailVerifier.PASS1_THRESHOLD:
            # Bug 1 fix: weakest-link rule — ALL gates must pass.
            # A low confidence score means the LLM itself is uncertain about
            # the match, even if term-overlap scores look acceptable.
            if confidence < GuardrailVerifier.CONFIDENCE_THRESHOLD:
                logger.warning(
                    "[Guardrail] REJECTED_LOW_CONFIDENCE: confidence=%.2f < %.2f "
                    "(weakest-link rule) | query=%r",
                    confidence, GuardrailVerifier.CONFIDENCE_THRESHOLD, query[:60],
                )
                return {
                    "answer": GuardrailVerifier.UNANSWERABLE_FALLBACK,
                    "confidence": 0.0,
                    "sources": [],
                    "can_answer": False,
                    "guardrail_status": "REJECTED_LOW_CONFIDENCE",
                    "guardrail_reason": (
                        f"LLM confidence ({confidence:.0%}) is below threshold "
                        f"({GuardrailVerifier.CONFIDENCE_THRESHOLD:.0%}). "
                        f"Relevance: {relevance_score:.2f} | "
                        f"Faithfulness: {support_ratio:.2f} | "
                        f"Similarity: {top_sim:.3f}"
                    ),
                    "groundedness_pass": 1,
                    "query_relevance_score": relevance_score,
                    "top_similarity_score": top_sim,
                }
            return {
                "answer": answer,
                "confidence": confidence,
                "sources": sources,
                "can_answer": True,
                "guardrail_status": "PASSED",
                "guardrail_reason": (
                    f"Relevance: {relevance_score:.2f} | "
                    f"Faithfulness: {support_ratio:.2f} | "
                    f"Similarity: {top_sim:.3f} | "
                    f"Confidence: {confidence:.0%}"
                ),
                "groundedness_pass": 1,
                "query_relevance_score": relevance_score,
                "top_similarity_score": top_sim,
            }

        # ── PASS 2: Strict local re-extraction ───────────────────────────────
        pass2_answer = GuardrailVerifier._strict_local_reextract(query, retrieved_chunks)
        if pass2_answer:
            pass2_ratio = GuardrailVerifier._compute_groundedness(pass2_answer, retrieved_chunks)
            logger.info(
                "[Guardrail Pass2] pass2_ratio=%.3f | threshold=%.2f", pass2_ratio, GuardrailVerifier.PASS2_THRESHOLD
            )
            if pass2_ratio >= GuardrailVerifier.PASS2_THRESHOLD:
                pass2_sources = [c.get("source_location", "unknown") for c in retrieved_chunks][:3]
                return {
                    "answer": pass2_answer,
                    "confidence": min(0.75, pass2_ratio * 1.5),
                    "sources": list(dict.fromkeys(pass2_sources)),
                    "can_answer": True,
                    "guardrail_status": "PASSED_STRICT_REGEN",
                    "guardrail_reason": (
                        f"Pass 1 faithfulness failed (ratio={support_ratio:.2f}), "
                        f"Pass 2 strict re-extraction succeeded (ratio={pass2_ratio:.2f}). "
                        f"Relevance: {relevance_score:.2f} | Similarity: {top_sim:.3f}"
                    ),
                    "groundedness_pass": 2,
                    "query_relevance_score": relevance_score,
                    "top_similarity_score": top_sim,
                }

        # ── Both passes failed — safe refusal ────────────────────────────────
        logger.warning(
            "[Guardrail] REJECTED_HALLUCINATION_RISK: Pass1=%.2f, Pass2 failed | query=%r",
            support_ratio, query[:60],
        )
        return {
            "answer": GuardrailVerifier.UNANSWERABLE_FALLBACK,
            "confidence": 0.0,
            "sources": [],
            "can_answer": False,
            "guardrail_status": "REJECTED_HALLUCINATION_RISK",
            "guardrail_reason": (
                f"Answer unsupported by context after 3-pass verification "
                f"(Pass0 relevance={relevance_score:.2f}, "
                f"Pass1 faithfulness={support_ratio:.2f}, Pass2 extraction failed)."
            ),
            "groundedness_pass": 2,
            "query_relevance_score": relevance_score,
            "top_similarity_score": top_sim,
        }

    # ── Pass 2 helper: strict re-extraction ───────────────────────────────────

    @staticmethod
    def _strict_local_reextract(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Strict local re-extraction: finds sentences in retrieved context that best
        overlap with query terms. Prioritizes MSMARCO-XI is_selected=1 passages.
        Only used as Pass 2 fallback after faithfulness check fails.
        """
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        stemmed_q = {w[:4] for w in query_words if len(w) >= 4}

        sorted_chunks = sorted(
            retrieved_chunks,
            key=lambda c: (
                -(c.get("is_selected", 0)),
                -(c.get("rerank_score") or c.get("similarity_score", 0.0))
            )
        )

        best_lines = []
        for chunk in sorted_chunks[:3]:
            text = chunk.get("text", "")
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for sent in sentences:
                sent_words = {w[:4] for w in re.findall(r'\b\w{3,}\b', sent.lower())}
                overlap = len(stemmed_q & sent_words)
                if overlap > 0:
                    best_lines.append((overlap, sent.strip()))

        if not best_lines:
            return ""

        best_lines.sort(key=lambda x: -x[0])
        return ". ".join(line for _, line in best_lines[:3]).strip()


guardrail_verifier = GuardrailVerifier()
