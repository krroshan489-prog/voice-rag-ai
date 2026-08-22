import uuid
import re
from typing import List, Dict, Any, Optional

class ChunkingStrategy:
    FIXED = "fixed"
    RECURSIVE = "recursive"
    STRUCTURE_AWARE = "structure_aware"

class DocumentChunker:
    """Implements Fixed, Recursive, and Structure-aware chunking strategies with rich metadata."""

    @staticmethod
    def chunk_document(
        parsed_pages: List[Dict[str, Any]],
        strategy: str = ChunkingStrategy.RECURSIVE,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Chunks parsed document sections using selected strategy.
        Returns list of chunk objects with rich metadata.
        """
        strategy = strategy.lower()
        chunks = []

        for page_data in parsed_pages:
            text = page_data.get("text", "")
            doc_name = page_data.get("doc_name", "unknown")
            page_num = page_data.get("page", 1)
            section = page_data.get("section", "General")
            source_loc = page_data.get("source_location", f"{doc_name}")

            if not text.strip():
                continue

            if strategy == ChunkingStrategy.FIXED:
                page_chunks = DocumentChunker._fixed_size_chunking(text, chunk_size, chunk_overlap)
            elif strategy == ChunkingStrategy.STRUCTURE_AWARE:
                page_chunks = DocumentChunker._structure_aware_chunking(text, chunk_size)
            else:  # RECURSIVE / SEMANTIC
                page_chunks = DocumentChunker._recursive_semantic_chunking(text, chunk_size, chunk_overlap)

            for idx, item in enumerate(page_chunks):
                chunk_id = f"{doc_name}_p{page_num}_c{len(chunks)+1}_{uuid.uuid4().hex[:6]}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": item["text"],
                    "document_name": doc_name,
                    "page": page_num,
                    "section": item.get("section", section),
                    "chunking_strategy": strategy,
                    "source_location": f"{doc_name} • Page {page_num} ({item.get('section', section)})",
                    "start_char": item.get("start_char", 0),
                    "end_char": item.get("end_char", len(item["text"]))
                })

        return chunks

    @staticmethod
    def _fixed_size_chunking(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_str = text[start:end].strip()
            if chunk_str:
                chunks.append({
                    "text": chunk_str,
                    "start_char": start,
                    "end_char": end,
                    "section": "Fixed Window"
                })
            if end >= text_length:
                break
            start += max(1, chunk_size - chunk_overlap)

        return chunks

    @staticmethod
    def _recursive_semantic_chunking(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
        """Splits recursively on headings, double newlines, single newlines, sentences."""
        separators = ["\n\n", "\n", ". ", "; ", " "]
        
        def split_text(sub_text: str, sep_idx: int) -> List[str]:
            if len(sub_text) <= chunk_size or sep_idx >= len(separators):
                return [sub_text] if sub_text.strip() else []

            sep = separators[sep_idx]
            parts = sub_text.split(sep)
            result = []
            current_chunk = ""

            for part in parts:
                candidate = (current_chunk + sep + part) if current_chunk else part
                if len(candidate) <= chunk_size:
                    current_chunk = candidate
                else:
                    if current_chunk:
                        result.append(current_chunk)
                    if len(part) > chunk_size:
                        result.extend(split_text(part, sep_idx + 1))
                        current_chunk = ""
                    else:
                        current_chunk = part

            if current_chunk:
                result.append(current_chunk)

            return result

        raw_chunks = split_text(text, 0)
        chunks = []
        char_cursor = 0
        
        for c in raw_chunks:
            c_clean = c.strip()
            if c_clean:
                chunks.append({
                    "text": c_clean,
                    "start_char": char_cursor,
                    "end_char": char_cursor + len(c_clean),
                    "section": "Semantic Segment"
                })
            char_cursor += len(c)

        return chunks

    @staticmethod
    def _structure_aware_chunking(text: str, max_chunk_size: int) -> List[Dict[str, Any]]:
        """Preserves structural elements (headers, lists, tables, code blocks)."""
        lines = text.split("\n")
        chunks = []
        current_section = "Main Content"
        current_buffer = []
        current_size = 0

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Detect structural header or bullet
            is_header = line_str.startswith("#") or re.match(r'^(SECTION|CHAPTER|PART|\d+\.\d+)\b', line_str, re.IGNORECASE)
            
            if is_header:
                if current_buffer:
                    chunk_text = "\n".join(current_buffer).strip()
                    if chunk_text:
                        chunks.append({
                            "text": chunk_text,
                            "section": current_section,
                            "start_char": 0,
                            "end_char": len(chunk_text)
                        })
                    current_buffer = []
                    current_size = 0
                current_section = line_str.replace("#", "").strip()
                current_buffer.append(line_str)
                current_size += len(line_str)
            else:
                if current_size + len(line_str) > max_chunk_size and current_buffer:
                    chunk_text = "\n".join(current_buffer).strip()
                    chunks.append({
                        "text": chunk_text,
                        "section": current_section,
                        "start_char": 0,
                        "end_char": len(chunk_text)
                    })
                    current_buffer = [line_str]
                    current_size = len(line_str)
                else:
                    current_buffer.append(line_str)
                    current_size += len(line_str) + 1

        if current_buffer:
            chunk_text = "\n".join(current_buffer).strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "section": current_section,
                    "start_char": 0,
                    "end_char": len(chunk_text)
                })

        return chunks
