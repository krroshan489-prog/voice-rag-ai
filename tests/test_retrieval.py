import pytest
from backend.app.vector_store.store import vector_store
from backend.app.reranker.reranker import reranker

def test_vector_search_and_rerank():
    test_chunks = [
        {
            "chunk_id": "c1",
            "text": "The Voice RAG system uses low-latency Speech-to-Text and high dimensional vector search.",
            "document_name": "test_doc.md",
            "page": 1,
            "section": "Overview",
            "source_location": "test_doc.md (Page 1)",
            "chunking_strategy": "recursive"
        },
        {
            "chunk_id": "c2",
            "text": "Weather in Goa is sunny during November.",
            "document_name": "weather.txt",
            "page": 1,
            "section": "General",
            "source_location": "weather.txt",
            "chunking_strategy": "fixed"
        }
    ]
    
    vector_store.clear()
    vector_store.add_chunks(test_chunks)
    
    # Search
    results = vector_store.search("Voice RAG Speech-to-Text", top_k=2)
    assert len(results) > 0
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["similarity_score"] > 0.0
    
    # Rerank
    reranked = reranker.rerank("Voice RAG Speech-to-Text", results, top_k=1)
    assert len(reranked) == 1
    assert reranked[0]["chunk_id"] == "c1"
    assert "rerank_score" in reranked[0]
