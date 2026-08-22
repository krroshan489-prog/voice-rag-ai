"""
MSMARCO-XI Dataset Ingestion Engine
------------------------------------
Loads ai4bharat/MSMARCO-XI passages into the vector store with persistent indexing.

Discovered schema (from ms_marco_translations.py and hinval.parquet):
  - query_id        : int32
  - Eng_Query       : str  (original English question)
  - Eng_Answer      : str  (ground-truth English answer)
  - query           : str  (translated query, primary language)
  - Answer          : str  (translated answer, primary language)
  - source_lang     : str  (e.g. "eng_Latn")
  - target_lang     : str  (e.g. "hin_Deva")
  - passages        : dict { is_selected: list[int], English_passages: list[str],
                             Translated_passages: list[str] }
  - query_type      : str
  - meta            : dict (model metadata)
"""

import os
import sys
import json
import re
import time
import uuid
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Persistent marker ──────────────────────────────────────────────────────────
MSMARCO_INDEX_MARKER = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "msmarco_indexed.json"
)
MSMARCO_INDEX_MARKER = os.path.normpath(MSMARCO_INDEX_MARKER)

DATASET_NAME = "ai4bharat/MSMARCO-XI"
SOURCE_TAG   = "MSMARCO-XI"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters."""
    if not text:
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fixed_chunks(text: str, chunk_size: int = 400, overlap: int = 40) -> List[str]:
    """Fixed-size character windowing with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 20]


def _recursive_chunks(text: str, chunk_size: int = 400, overlap: int = 40) -> List[str]:
    """Recursive sentence-boundary splitting."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= chunk_size:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            # carry overlap
            words = current.split()
            carry = " ".join(words[-max(1, overlap // 8):]) if words else ""
            current = (carry + " " + sent).strip() if carry else sent.strip()
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) > 20]


def _structure_chunks(text: str) -> List[str]:
    """Structure-aware splitting on paragraph/list boundaries."""
    parts = re.split(r"\n{2,}|\t|(?<=\.)\s{2,}", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def _chunk_text(text: str, strategy: str) -> List[str]:
    if strategy == "fixed":
        return _fixed_chunks(text)
    elif strategy == "structure_aware":
        return _structure_chunks(text)
    else:
        return _recursive_chunks(text)  # default / recursive / semantic


def _make_chunk(
    text: str,
    idx: int,
    query_id: int,
    eng_query: str,
    is_selected: int,
    passage_index: int,
    language: str,
    target_lang: str,
    strategy: str,
) -> Dict[str, Any]:
    """Build a vector-store compatible chunk dict with full MSMARCO-XI metadata."""
    return {
        "chunk_id": str(uuid.uuid4()),
        "text": text,
        # Source identification
        "document_name": f"MSMARCO-XI (query_id={query_id})",
        "source_location": f"MSMARCO-XI | query_id={query_id} | passage={passage_index}",
        "page": 1,
        "section": f"passage_{passage_index}",
        # MSMARCO-XI specific metadata (preserved for frontend display)
        "dataset_name": DATASET_NAME,
        "dataset_source": SOURCE_TAG,
        "query_id": query_id,
        "eng_query": eng_query,
        "passage_index": passage_index,
        "is_selected": is_selected,
        "language_code": language,
        "target_lang": target_lang,
        # Chunking info
        "chunking_strategy": strategy,
        "chunk_index": idx,
    }


# ── Main Ingestor ──────────────────────────────────────────────────────────────

class MSMARCOIngestor:
    """
    Loads ai4bharat/MSMARCO-XI validation set and indexes English passages into
    the shared VectorStore. Uses persistent marker to avoid re-indexing on restart.
    """

    def __init__(self, vector_store, strategy: str = "recursive"):
        self.vector_store = vector_store
        self.strategy = strategy

    def already_indexed(self) -> bool:
        return os.path.exists(MSMARCO_INDEX_MARKER)

    def _write_marker(self, stats: Dict[str, Any]):
        os.makedirs(os.path.dirname(MSMARCO_INDEX_MARKER), exist_ok=True)
        with open(MSMARCO_INDEX_MARKER, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    def read_marker(self) -> Dict[str, Any]:
        if not self.already_indexed():
            return {}
        with open(MSMARCO_INDEX_MARKER, "r", encoding="utf-8") as f:
            return json.load(f)

    def ingest(
        self,
        max_records: int = 5000,
        strategy: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest MSMARCO-XI English passages into the vector store.

        Args:
            max_records : Max dataset records to process (each record has multiple passages).
            strategy    : Chunking strategy override. Uses self.strategy if None.
            force       : Force re-ingest even if marker exists.

        Returns:
            Stats dict with records_processed, chunks_added, elapsed_sec.
        """
        effective_strategy = strategy or self.strategy

        if self.already_indexed() and not force:
            marker = self.read_marker()
            logger.info(
                "MSMARCO-XI already indexed (%s chunks). Skipping ingestion.",
                marker.get("chunks_added", "?"),
            )
            return {"status": "already_indexed", **marker}

        t0 = time.time()
        chunks_to_add: List[Dict[str, Any]] = []
        records_processed = 0
        passages_processed = 0

        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets package not installed. Run: pip install datasets")
            return {"status": "error", "reason": "datasets package missing"}

        logger.info("Loading ai4bharat/MSMARCO-XI (streaming, max_records=%d)…", max_records)
        try:
            ds = load_dataset(
                DATASET_NAME,
                split="validation",
                streaming=True,
                trust_remote_code=False,
            )
        except Exception as e:
            logger.error("Failed to load MSMARCO-XI dataset: %s", e)
            return {"status": "error", "reason": str(e)}

        for record in ds:
            if records_processed >= max_records:
                break

            try:
                query_id   = int(record.get("query_id", 0))
                eng_query  = _clean_text(record.get("Eng_Query") or "")
                target_lang = str(record.get("target_lang") or "")
                language   = str(record.get("source_lang") or "eng_Latn")

                passages_dict = record.get("passages") or {}
                eng_passages  = passages_dict.get("English_passages") or []
                is_selected_list = passages_dict.get("is_selected") or []

                for p_idx, passage_text in enumerate(eng_passages):
                    clean_passage = _clean_text(str(passage_text))
                    if not clean_passage or len(clean_passage) < 20:
                        continue

                    is_sel = int(is_selected_list[p_idx]) if p_idx < len(is_selected_list) else 0

                    text_chunks = _chunk_text(clean_passage, effective_strategy)
                    for c_idx, chunk_text in enumerate(text_chunks):
                        chunks_to_add.append(
                            _make_chunk(
                                text=chunk_text,
                                idx=c_idx,
                                query_id=query_id,
                                eng_query=eng_query,
                                is_selected=is_sel,
                                passage_index=p_idx,
                                language=language,
                                target_lang=target_lang,
                                strategy=effective_strategy,
                            )
                        )
                    passages_processed += 1

                records_processed += 1

                # Batch-write every 500 records to avoid huge memory spikes
                if len(chunks_to_add) >= 2000:
                    self.vector_store.add_chunks(chunks_to_add)
                    logger.info(
                        "  → Indexed batch: %d chunks (records so far: %d)",
                        len(chunks_to_add), records_processed,
                    )
                    chunks_to_add = []

            except Exception as e:
                logger.warning("Skipping record %d: %s", records_processed, e)
                continue

        # Final flush
        if chunks_to_add:
            self.vector_store.add_chunks(chunks_to_add)

        elapsed = round(time.time() - t0, 2)
        total_chunks = len(self.vector_store.chunks)

        stats = {
            "status": "success",
            "dataset": DATASET_NAME,
            "strategy": effective_strategy,
            "records_processed": records_processed,
            "passages_processed": passages_processed,
            "chunks_added": total_chunks,
            "elapsed_sec": elapsed,
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        self._write_marker(stats)
        logger.info(
            "MSMARCO-XI ingestion complete: %d records, %d chunks in %.1fs",
            records_processed, total_chunks, elapsed,
        )
        return stats
