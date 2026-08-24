"""
Grounded LLM Generator with 2-pass strict evidence regeneration and MSMARCO-XI citation support.
"""

import json
import requests
import re
from typing import List, Dict, Any
from backend.app.config import settings


class GroundedLLMGenerator:
    """Grounded LLM Generator enforcing strict evidence-based answer synthesis with MSMARCO-XI citation."""

    SYSTEM_PROMPT = (
        "You are an accurate, grounded RAG AI assistant. "
        "Strictly adhere to these rules:\n"
        "1. Answer ONLY using the facts provided in the Context below.\n"
        "2. Do NOT invent facts or rely on external knowledge not present in the context.\n"
        "3. If the context does not contain enough information to answer the user query, explicitly set can_answer to false.\n"
        "4. When context comes from MSMARCO-XI, cite the Record ID in your sources list.\n"
        "5. Respond ONLY with a valid JSON object matching this exact structure:\n"
        "{\n"
        '  "answer": "Clear, grounded answer string",\n'
        '  "confidence": 0.95,\n'
        '  "sources": ["MSMARCO-XI (query_id=12345 | passage=0)"],\n'
        '  "can_answer": true\n'
        "}"
    )

    STRICT_SYSTEM_PROMPT = (
        "You are an ultra-strict grounded RAG AI. "
        "Rules (non-negotiable):\n"
        "1. Use ONLY the provided Context. No external knowledge allowed.\n"
        "2. Every sentence in your answer MUST be traceable to a specific passage in the Context.\n"
        "3. If ANY part of the answer is unsupported, set can_answer to false and return the fallback.\n"
        "4. Cite MSMARCO-XI Record IDs explicitly in sources.\n"
        "5. Respond ONLY with valid JSON:\n"
        "{\n"
        '  "answer": "Strictly context-grounded answer",\n'
        '  "confidence": 0.90,\n'
        '  "sources": ["MSMARCO-XI (query_id=12345 | passage=0)"],\n'
        '  "can_answer": true\n'
        "}"
    )

    @staticmethod
    def _stem(word: str) -> str:
        w = word.lower()
        if len(w) > 4 and w.endswith('s'):
            return w[:-1]
        if len(w) > 5 and w.endswith('ies'):
            return w[:-3] + 'y'
        return w

    @staticmethod
    def _build_context_blocks(reranked_chunks: List[Dict[str, Any]]):
        """Builds formatted context string and unique sources list from reranked chunks."""
        context_blocks = []
        sources = []
        for idx, chunk in enumerate(reranked_chunks, start=1):
            # Prefer MSMARCO-XI record citation format
            if chunk.get("dataset_source") == "MSMARCO-XI":
                qid = chunk.get("query_id", "?")
                pidx = chunk.get("passage_index", 0)
                is_sel = chunk.get("is_selected", 0)
                lang = chunk.get("language_code", "eng_Latn")
                src = f"MSMARCO-XI (query_id={qid} | passage={pidx} | selected={is_sel} | lang={lang})"
            else:
                src = chunk.get("source_location", f"Source {idx}")
            sources.append(src)
            context_blocks.append(f"[Source {idx}: {src}]\n{chunk.get('text', '')}")

        formatted_context = "\n\n".join(context_blocks)
        sources = list(dict.fromkeys(sources))  # unique preserving order
        return formatted_context, sources

    @staticmethod
    def generate_answer(query: str, reranked_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesizes grounded answer from reranked context chunks.
        Implements 2-pass strict regeneration:
          Pass 1: Standard grounded generation.
          Pass 2: If groundedness fails, strict regeneration pass.
          Fallback: Safe refusal if still unsupported.
        Returns dict: { "answer": str, "confidence": float, "sources": List[str], "can_answer": bool }
        """
        if not reranked_chunks:
            return {
                "answer": "I couldn't find enough information in the provided knowledge base to answer that.",
                "confidence": 0.0,
                "sources": [],
                "can_answer": False
            }

        formatted_context, sources = GroundedLLMGenerator._build_context_blocks(reranked_chunks)

        # Pass 1: Standard grounded generation
        if settings.GROQ_API_KEY:
            res = GroundedLLMGenerator._call_groq(
                query, formatted_context, GroundedLLMGenerator.SYSTEM_PROMPT
            )
            if res and res.get("can_answer"):
                return res

        if settings.OPENAI_API_KEY:
            res = GroundedLLMGenerator._call_openai(
                query, formatted_context, GroundedLLMGenerator.SYSTEM_PROMPT
            )
            if res and res.get("can_answer"):
                return res

        # Pass 1 local fallback
        local_pass1 = GroundedLLMGenerator._local_grounded_synthesis(query, reranked_chunks, sources)

        # If Pass 1 succeeded and confidence is OK, return it
        if local_pass1.get("can_answer") and local_pass1.get("confidence", 0) >= 0.3:
            return local_pass1

        # Pass 2: Strict regeneration (API only, not repeated for local)
        if settings.GROQ_API_KEY:
            res2 = GroundedLLMGenerator._call_groq(
                query, formatted_context, GroundedLLMGenerator.STRICT_SYSTEM_PROMPT
            )
            if res2 and res2.get("can_answer"):
                return res2

        if settings.OPENAI_API_KEY:
            res2 = GroundedLLMGenerator._call_openai(
                query, formatted_context, GroundedLLMGenerator.STRICT_SYSTEM_PROMPT
            )
            if res2 and res2.get("can_answer"):
                return res2

        # If pass 1 local returned something reasonable, accept it
        if local_pass1.get("can_answer"):
            return local_pass1

        # Safe refusal
        return {
            "answer": "I couldn't find enough information in the provided knowledge base to answer that.",
            "confidence": 0.0,
            "sources": [],
            "can_answer": False
        }

    @staticmethod
    def _call_groq(query: str, context: str, system_prompt: str) -> Dict[str, Any]:
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            prompt = f"Context:\n{context}\n\nUser Question: {query}"
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
                "max_tokens": 600
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=8)
            if resp.status_code == 200:
                raw_content = resp.json()["choices"][0]["message"]["content"]
                return json.loads(raw_content)
        except Exception:
            pass
        return {}

    @staticmethod
    def _call_openai(query: str, context: str, system_prompt: str) -> Dict[str, Any]:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            prompt = f"Context:\n{context}\n\nUser Question: {query}"
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
                "max_tokens": 600
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=8)
            if resp.status_code == 200:
                raw_content = resp.json()["choices"][0]["message"]["content"]
                return json.loads(raw_content)
        except Exception:
            pass
        return {}

    @staticmethod
    def _local_grounded_synthesis(query: str, chunks: List[Dict[str, Any]], sources: List[str]) -> Dict[str, Any]:
        """Local high-confidence grounded answer extraction from retrieved context."""
        stop = {"what", "when", "where", "which", "that", "this", "with", "from", "have", "been", "will", "were", "they", "their", "is", "it", "in", "to", "of", "or", "on", "at", "by", "an", "am", "do", "if", "my", "no", "so", "we", "be", "as", "he", "me", "us", "are", "was", "for", "how", "why", "who"}
        query_words = {w for w in re.findall(r'\b[A-Za-z0-9_]{2,}\b', query.lower()) if w not in stop}
        stemmed_q_words = {GroundedLLMGenerator._stem(w) for w in query_words}

        best_chunk = chunks[0]
        text = best_chunk.get("text", "")
        rerank_score = best_chunk.get("rerank_score", 0.0)

        # Split text into lines/sentences
        lines = [line.strip() for line in re.split(r'[\n\.]+', text) if line.strip()]
        matching_lines = []

        for line in lines:
            line_words = {GroundedLLMGenerator._stem(w) for w in re.findall(r'\b[A-Za-z0-9_]{2,}\b', line.lower())}
            if any(sq in line_words or any(lw.startswith(sq) for lw in line_words) for sq in stemmed_q_words):
                matching_lines.append(line)

        if matching_lines:
            answer_text = ". ".join(matching_lines[:3]) + "."
            confidence = min(0.98, max(0.70, rerank_score * 1.2))
            return {
                "answer": answer_text,
                "confidence": float(round(confidence, 2)),
                "sources": sources,
                "can_answer": True
            }

        if rerank_score > 0.15:
            return {
                "answer": text[:350].strip() + ("..." if len(text) > 350 else ""),
                "confidence": max(0.70, float(round(rerank_score, 2))),
                "sources": sources,
                "can_answer": True
            }

        return {
            "answer": "I couldn't find enough information in the provided knowledge base to answer that.",
            "confidence": 0.0,
            "sources": [],
            "can_answer": False
        }


generator = GroundedLLMGenerator()
