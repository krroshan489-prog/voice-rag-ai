import os
import time
import shutil
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.app.config import settings
from backend.app.ingestion.parser import DocumentParser
from backend.app.ingestion.chunker import DocumentChunker, ChunkingStrategy
from backend.app.vector_store.store import vector_store
from backend.app.vector_store.embeddings import embedding_engine
from backend.app.reranker.reranker import reranker
from backend.app.llm.generator import generator
from backend.app.guardrails.verifier import guardrail_verifier
from backend.app.observability.metrics import metrics_tracker
from backend.app.stt.stt_service import stt_service
from backend.app.tts.tts_service import tts_service
from backend.app.ingestion.msmarco_ingestor import MSMARCOIngestor, MSMARCO_INDEX_MARKER
from backend.app.utils.translation import translate_query_to_english

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DIST_DIR = os.path.join(settings.BASE_DIR, "dist")

app = FastAPI(
    title="Voice-Enabled RAG System – MSMARCO-XI",
    description="Hacker House Goa 2026 Task #2 Production RAG Engine with MSMARCO-XI corpus",
    version="2.0.0"
)

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory document registry
DOCUMENTS_REGISTRY: Dict[str, Dict[str, Any]] = {}

# Shared MSMARCO ingestor instance
msmarco_ingestor = MSMARCOIngestor(vector_store=vector_store, strategy="recursive")


def initialize_demo_documents():
    """Populates initial high-quality demo document if workspace has no index."""
    if len(vector_store.chunks) == 0:
        demo_doc_name = "Hacker_House_Goa_2026_Voice_RAG_Spec.md"
        demo_text = """# Hacker House Goa 2026 - Voice RAG Architecture Specification

## Executive Overview
The Voice-Enabled RAG system is a real-time, low-latency AI knowledge assistant developed for Hacker House Goa 2026 Task #2.
It combines low-latency Speech-to-Text (STT), high-dimensional vector retrieval, two-stage cross-encoder reranking, strict grounded LLM generation, and automated hallucination guardrails.
The knowledge corpus is powered by ai4bharat/MSMARCO-XI, a multilingual QA dataset with English passages and translated queries across many languages.

## Key Features & Core Components
1. **Multi-Strategy Chunking Pipeline**: Supports Fixed-Size windowing, Recursive/Semantic sentence boundary splitting, and Structure-Aware document sectioning (headers, lists, tables).
2. **Vector Retrieval & Reranking**: Uses normalized 384-dimensional dense embeddings with cosine similarity search and a two-stage reranker that weights term density, exact phrase matching, and vector scores.
3. **Grounded Generation & Guardrails**: Enforces zero-hallucination rules. The system checks answer groundedness against retrieved context chunks and rejects unsupported claims with an explicit fallback message: "I couldn't find enough information in the provided knowledge base to answer that."
4. **Latency Optimization**: Optimized for real-time responsiveness tracking P50, P70, and P100 latencies across STT, embedding, search, rerank, and LLM stages.
5. **MSMARCO-XI Integration**: The corpus uses the ai4bharat/MSMARCO-XI dataset with English passages, query_id metadata, is_selected flags, and multilingual language codes.

## MSMARCO-XI Dataset Schema
The MSMARCO-XI dataset contains: query_id (int32), Eng_Query (original English question), Eng_Answer (ground truth answer), query (translated query), Answer (translated answer), source_lang, target_lang, and passages dict containing is_selected list, English_passages list, and Translated_passages list.

## Performance Benchmarks
- Vector Search Latency: < 15ms
- Grounded Hallucination Guardrail Check: < 5ms
- Multi-strategy indexing speed: > 500 pages/sec
"""
        demo_path = os.path.join(settings.DOCUMENTS_DIR, demo_doc_name)
        with open(demo_path, "w", encoding="utf-8") as f:
            f.write(demo_text)

        parsed_pages = DocumentParser.parse_file(demo_path, demo_doc_name)
        chunks = DocumentChunker.chunk_document(
            parsed_pages,
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=400,
            chunk_overlap=40
        )
        vector_store.add_chunks(chunks)
        DOCUMENTS_REGISTRY[demo_doc_name] = {
            "filename": demo_doc_name,
            "pages": len(parsed_pages),
            "chunks_count": len(chunks),
            "strategy": ChunkingStrategy.RECURSIVE,
            "file_size": len(demo_text.encode('utf-8'))
        }


initialize_demo_documents()

# ---------------------------------------------------------------------------
# MSMARCO-XI Background ingestion: load persistent index at startup if exists
# ---------------------------------------------------------------------------
def _try_restore_msmarco_index():
    """If MSMARCO has been indexed before, log stats; otherwise require manual ingestion."""
    if msmarco_ingestor.already_indexed():
        marker = msmarco_ingestor.read_marker()
        logger.info(
            "MSMARCO-XI persistent index detected: %d chunks, indexed at %s.",
            marker.get("chunks_added", 0),
            marker.get("indexed_at", "?")
        )
    else:
        logger.info("MSMARCO-XI index not found. Manual ingestion required.")


_try_restore_msmarco_index()


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    chunking_strategy: Optional[str] = "recursive"
    top_k: Optional[int] = 4
    stt_latency_ms: Optional[float] = 0.0

class MSMARCOIngestRequest(BaseModel):
    max_records: Optional[int] = 200
    strategy: Optional[str] = "recursive"
    force: Optional[bool] = False


# ---------------------------------------------------------------------------
# Core RAG Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return FileResponse(os.path.join(DIST_DIR, "index.html")) 

@app.post("/api/query")
async def execute_rag_pipeline(request: QueryRequest):
    """
    Executes complete end-to-end RAG pipeline:
    Query validation -> Embedding -> Vector Search -> Reranking -> LLM Generation -> Guardrail Verification
    Supports MSMARCO-XI sourced context with full record ID citations.
    """
    pipeline_start = time.time()

    # Debug: log the exact raw bytes of the incoming query so voice/typed paths
    # can be compared directly in the server log.
    logger.info("[Pipeline] RAW query repr: %r  len=%d", request.query, len(request.query))

    # 1. Pre-validation Guardrails
    val = guardrail_verifier.verify_request(request.query)
    if not val["is_valid"]:
        rejection_status = val.get("guardrail_status", "REJECTED_INPUT_VALIDATION")
        # Safety blocks return the neutral SAFETY_FALLBACK as the answer;
        # injection/other blocks return a generic invalid-query message.
        from backend.app.guardrails.verifier import GuardrailVerifier as _GV
        is_safety = rejection_status == "REJECTED_UNSAFE_CONTENT"
        answer_text = val["reason"] if is_safety else f"Invalid query: {val['reason']}"

        metrics_tracker.record_query(
            query=request.query,
            stt_latency_ms=request.stt_latency_ms or 0.0,
            embedding_latency_ms=0.0,
            retrieval_latency_ms=0.0,
            reranking_latency_ms=0.0,
            llm_latency_ms=0.0,
            total_latency_ms=(time.time() - pipeline_start) * 1000,
            chunks_count=0,
            top_similarity_score=0.0,
            guardrail_status=rejection_status,
            can_answer=False,
            is_success=False
        )
        return {
            "answer": answer_text,
            "confidence": 0.0,
            "sources": [],
            "can_answer": False,
            "latency": {
                "stt_ms": request.stt_latency_ms or 0.0,
                "embedding_ms": 0.0,
                "retrieval_ms": 0.0,
                "reranking_ms": 0.0,
                "llm_ms": 0.0,
                "total_ms": round((time.time() - pipeline_start) * 1000, 2)
            },
            "debug": {
                "query": request.query,
                "guardrail_status": rejection_status,
                "reason": val["reason"]
            }
        }

    # 2. Embedding & Retrieval Stage
    # Bug 2 fix: translate non-English queries to English before retrieval
    # (MSMARCO-XI index stores English passage embeddings only)
    retrieval_query = translate_query_to_english(val["sanitized_query"])
    was_translated = retrieval_query != val["sanitized_query"]
    if was_translated:
        logger.info("[Pipeline] Translated query for retrieval: %r → %r",
                    val["sanitized_query"][:50], retrieval_query[:50])

    emb_start = time.time()
    query_vector = embedding_engine.embed_query(retrieval_query)
    emb_latency = (time.time() - emb_start) * 1000

    ret_start = time.time()
    retrieved_chunks = vector_store.search(
        query=retrieval_query,
        top_k=request.top_k or 4,
        strategy_filter=request.chunking_strategy if request.chunking_strategy != "all" else None
    )
    ret_latency = (time.time() - ret_start) * 1000

    # 3. Reranking Stage (use translated query for scoring)
    rerank_start = time.time()
    reranked_chunks = reranker.rerank(
        query=retrieval_query,
        retrieved_chunks=retrieved_chunks,
        top_k=3
    )
    rerank_latency = (time.time() - rerank_start) * 1000

    # 4. Grounded Generation Stage with MSMARCO-XI citations
    # Use translated query for generation so the LLM works with English context
    llm_start = time.time()
    llm_draft = generator.generate_answer(retrieval_query, reranked_chunks)
    llm_latency = (time.time() - llm_start) * 1000

    # 5. Three-pass Guardrail Verification Stage
    # Pass relevance check with the translated query (what retrieval used)
    verified_response = guardrail_verifier.verify_groundedness(
        query=retrieval_query,
        retrieved_chunks=reranked_chunks,
        llm_response=llm_draft
    )

    total_latency = (time.time() - pipeline_start) * 1000
    top_score = reranked_chunks[0]["similarity_score"] if reranked_chunks else 0.0

    # Record metrics
    metrics_tracker.record_query(
        query=val["sanitized_query"],
        stt_latency_ms=request.stt_latency_ms or 0.0,
        embedding_latency_ms=emb_latency,
        retrieval_latency_ms=ret_latency,
        reranking_latency_ms=rerank_latency,
        llm_latency_ms=llm_latency,
        total_latency_ms=total_latency,
        chunks_count=len(retrieved_chunks),
        top_similarity_score=top_score,
        guardrail_status=verified_response["guardrail_status"],
        can_answer=verified_response["can_answer"],
        is_success=True
    )

    # 6. Text-To-Speech Metadata
    tts_meta = tts_service.synthesize_speech(verified_response["answer"])

    # Enrich response with MSMARCO-XI source metadata
    msmarco_sources = _extract_msmarco_metadata(reranked_chunks)

    return {
        "answer": verified_response["answer"],
        "confidence": verified_response["confidence"],
        "sources": verified_response["sources"],
        "can_answer": verified_response["can_answer"],
        "msmarco_sources": msmarco_sources,
        "knowledge_source": "MSMARCO-XI" if msmarco_sources else "document_corpus",
        "latency": {
            "stt_ms": round(request.stt_latency_ms or 0.0, 2),
            "embedding_ms": round(emb_latency, 2),
            "retrieval_ms": round(ret_latency, 2),
            "reranking_ms": round(rerank_latency, 2),
            "llm_ms": round(llm_latency, 2),
            "total_ms": round(total_latency, 2)
        },
        "debug": {
            "sanitized_query": val["sanitized_query"],
            "translated_query": retrieval_query if was_translated else None,
            "was_translated": was_translated,
            "retrieved_chunks": retrieved_chunks,
            "reranked_chunks": reranked_chunks,
            "llm_draft": llm_draft,
            "guardrail_status": verified_response["guardrail_status"],
            "guardrail_reason": verified_response["guardrail_reason"],
            "groundedness_pass": verified_response.get("groundedness_pass", 1),
            "query_relevance_score": verified_response.get("query_relevance_score"),
            "top_similarity_score": verified_response.get("top_similarity_score"),
        },
        "tts": tts_meta
    }


def _extract_msmarco_metadata(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract MSMARCO-XI specific metadata from retrieved chunks for frontend display."""
    msmarco_meta = []
    for chunk in chunks:
        if chunk.get("dataset_source") == "MSMARCO-XI":
            msmarco_meta.append({
                "query_id": chunk.get("query_id"),
                "eng_query": chunk.get("eng_query", ""),
                "passage_index": chunk.get("passage_index", 0),
                "is_selected": chunk.get("is_selected", 0),
                "language_code": chunk.get("language_code", "eng_Latn"),
                "target_lang": chunk.get("target_lang", ""),
                "source_location": chunk.get("source_location", ""),
                "similarity_score": chunk.get("similarity_score", 0.0),
                "rerank_score": chunk.get("rerank_score", 0.0),
                "chunking_strategy": chunk.get("chunking_strategy", "recursive"),
            })
    return msmarco_meta


# ---------------------------------------------------------------------------
# MSMARCO-XI Specific Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/msmarco/ingest")
async def ingest_msmarco(request: MSMARCOIngestRequest, background_tasks: BackgroundTasks):
    """
    Triggers ingestion of ai4bharat/MSMARCO-XI dataset into the vector store.
    Supports persistent indexing — subsequent calls skip re-ingestion unless force=True.
    """
    if msmarco_ingestor.already_indexed() and not request.force:
        marker = msmarco_ingestor.read_marker()
        return {
            "status": "already_indexed",
            "message": "MSMARCO-XI already indexed. Pass force=true to re-ingest.",
            **marker
        }

    # Run ingestion synchronously for small batches, async for large
    if (request.max_records or 200) <= 500:
        stats = msmarco_ingestor.ingest(
            max_records=request.max_records or 200,
            strategy=request.strategy or "recursive",
            force=request.force or False,
        )
        return stats
    else:
        # For large ingestion, run in background
        def _ingest_bg():
            msmarco_ingestor.ingest(
                max_records=request.max_records or 200,
                strategy=request.strategy or "recursive",
                force=request.force or False,
            )
        background_tasks.add_task(_ingest_bg)
        return {
            "status": "ingestion_started",
            "message": f"Background ingestion started for {request.max_records} records.",
            "strategy": request.strategy
        }

@app.get("/api/msmarco/stats")
def get_msmarco_stats():
    """Returns MSMARCO-XI ingestion statistics and index health."""
    marker = msmarco_ingestor.read_marker()
    msmarco_chunks = [c for c in vector_store.chunks if c.get("dataset_source") == "MSMARCO-XI"]
    
    # Count by strategy
    strategy_counts = {}
    for c in msmarco_chunks:
        strat = c.get("chunking_strategy", "unknown")
        strategy_counts[strat] = strategy_counts.get(strat, 0) + 1

    # Selected passage stats
    selected_count = sum(1 for c in msmarco_chunks if c.get("is_selected", 0) == 1)

    return {
        "indexed": msmarco_ingestor.already_indexed(),
        "total_chunks_in_store": len(vector_store.chunks),
        "msmarco_chunks": len(msmarco_chunks),
        "selected_passage_chunks": selected_count,
        "strategy_breakdown": strategy_counts,
        "marker": marker,
    }

@app.post("/api/msmarco/clear")
def clear_msmarco_index():
    """Clears the MSMARCO-XI persistent index marker to allow re-ingestion."""
    if os.path.exists(MSMARCO_INDEX_MARKER):
        os.remove(MSMARCO_INDEX_MARKER)
    return {"status": "cleared", "message": "MSMARCO-XI index marker removed. Re-ingest to rebuild."}


# ---------------------------------------------------------------------------
# Document Management Endpoints (unchanged)
# ---------------------------------------------------------------------------

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunking_strategy: str = Form("recursive"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50)
):
    """Uploads document file, runs parsing, multi-strategy chunking, embedding, and vector indexing."""
    filename = file.filename or "uploaded_doc.txt"
    file_path = os.path.join(settings.DOCUMENTS_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    parsed_pages = DocumentParser.parse_file(file_path, filename)
    if not parsed_pages:
        raise HTTPException(status_code=400, detail="Could not extract text from document.")

    # Remove existing chunks for this document if replacing
    vector_store.delete_document(filename)

    chunks = DocumentChunker.chunk_document(
        parsed_pages,
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    added_count = vector_store.add_chunks(chunks)

    DOCUMENTS_REGISTRY[filename] = {
        "filename": filename,
        "pages": len(parsed_pages),
        "chunks_count": added_count,
        "strategy": chunking_strategy,
        "file_size": os.path.getsize(file_path)
    }

    return {
        "status": "success",
        "filename": filename,
        "pages": len(parsed_pages),
        "chunks_added": added_count,
        "chunking_strategy": chunking_strategy
    }

@app.get("/api/documents")
def list_documents():
    """Lists all uploaded knowledge base documents and stats."""
    docs_list = list(DOCUMENTS_REGISTRY.values())
    return {
        "total_documents": len(docs_list),
        "total_chunks": len(vector_store.chunks),
        "documents": docs_list
    }

@app.delete("/api/documents/{doc_name}")
def delete_document(doc_name: str):
    """Deletes document from registry and vector store."""
    deleted_chunks = vector_store.delete_document(doc_name)
    if doc_name in DOCUMENTS_REGISTRY:
        del DOCUMENTS_REGISTRY[doc_name]

    file_path = os.path.join(settings.DOCUMENTS_DIR, doc_name)
    if os.path.exists(file_path):
        os.remove(file_path)

    return {
        "status": "deleted",
        "doc_name": doc_name,
        "chunks_removed": deleted_chunks
    }

@app.post("/api/documents/reindex")
def reindex_all(strategy: str = Form("recursive"), chunk_size: int = Form(500)):
    """Re-indexes all existing documents using the selected chunking strategy."""
    vector_store.clear()
    total_reindexed = 0

    for doc_name, info in DOCUMENTS_REGISTRY.items():
        file_path = os.path.join(settings.DOCUMENTS_DIR, doc_name)
        if os.path.exists(file_path):
            parsed = DocumentParser.parse_file(file_path, doc_name)
            chunks = DocumentChunker.chunk_document(
                parsed, strategy=strategy, chunk_size=chunk_size, chunk_overlap=50
            )
            count = vector_store.add_chunks(chunks)
            info["chunks_count"] = count
            info["strategy"] = strategy
            total_reindexed += count

    return {
        "status": "reindexed",
        "new_strategy": strategy,
        "total_chunks": total_reindexed
    }

@app.get("/api/observability/metrics")
def get_metrics():
    """Returns analytics, query count, success rate, and P50/P70/P100 latency percentiles."""
    return metrics_tracker.get_dashboard_metrics()


# ── Serve Built React Frontend (Unified Single-Deployment Mode) ──────────────────


if os.path.exists(DIST_DIR):
    assets_dir = os.path.join(DIST_DIR, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="API route not found")
        file_path = os.path.join(DIST_DIR, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(DIST_DIR, "index.html"))

