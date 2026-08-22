import pytest
from backend.app.ingestion.chunker import DocumentChunker, ChunkingStrategy

def test_fixed_size_chunking():
    text = "The quick brown fox jumps over the lazy dog. " * 20
    parsed = [{"text": text, "doc_name": "test.txt", "page": 1, "section": "Body"}]
    chunks = DocumentChunker.chunk_document(parsed, strategy=ChunkingStrategy.FIXED, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert chunks[0]["chunking_strategy"] == "fixed"
    assert "test.txt" in chunks[0]["source_location"]

def test_recursive_semantic_chunking():
    text = "Header 1\n\nParagraph one content goes here.\n\nParagraph two content goes here."
    parsed = [{"text": text, "doc_name": "spec.md", "page": 1, "section": "Intro"}]
    chunks = DocumentChunker.chunk_document(parsed, strategy=ChunkingStrategy.RECURSIVE, chunk_size=100, chunk_overlap=10)
    assert len(chunks) >= 1
    assert chunks[0]["chunking_strategy"] == "recursive"

def test_structure_aware_chunking():
    text = "# Section 1\nContent for section 1\n\n# Section 2\nContent for section 2"
    parsed = [{"text": text, "doc_name": "doc.md", "page": 1, "section": "Main"}]
    chunks = DocumentChunker.chunk_document(parsed, strategy=ChunkingStrategy.STRUCTURE_AWARE, chunk_size=200, chunk_overlap=10)
    assert len(chunks) >= 2
    assert chunks[0]["section"] == "Section 1"
    assert chunks[1]["section"] == "Section 2"
